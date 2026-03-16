import copy
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from common import evaluate


def train_fixmatch(
    model,
    labeled_loader,
    unlabeled_loader,
    val_loader,
    test_loader,
    device,
    lr: float = 2e-5,
    epochs: int = 50,
    threshold: float = 0.95,
):
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        0,
        epochs * len(labeled_loader),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    def logits_of(enc):
        enc = {k: v.to(device) for k, v in enc.items() if k != "labels"}
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            return model(**enc).logits

    def iterate_minibatches():
        u_iter = iter(unlabeled_loader)
        for l_batch in labeled_loader:
            try:
                u_batch = next(u_iter)
            except StopIteration:
                u_iter = iter(unlabeled_loader)
                u_batch = next(u_iter)
            yield l_batch, u_batch

    best_val = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()

        for l_batch, (enc_w, enc_s) in iterate_minibatches():
            y_l = l_batch["labels"].to(device)

            logits_l = logits_of(l_batch)
            loss_sup = F.cross_entropy(logits_l, y_l)

            with torch.no_grad():
                probs_w = torch.softmax(logits_of(enc_w), dim=1)
                conf, yhat = probs_w.max(dim=1)
                mask = (conf >= threshold).float()

            logits_s = logits_of(enc_s)
            if mask.sum() == 0:
                loss_unsup = torch.tensor(0.0, device=device)
            else:
                per = F.cross_entropy(logits_s, yhat.to(device), reduction="none")
                loss_unsup = (per * mask).sum() / mask.sum().clamp(min=1.0)

            loss = loss_sup + loss_unsup

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

        val_acc = evaluate(model, val_loader, device, use_amp=True)
        print(f"[Ep {ep:02d}] val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = evaluate(model, test_loader, device, use_amp=True)
    return {
        "best_val_acc": best_val,
        "test_acc": test_acc,
    }
