import copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from common import evaluate


def train_flexmatch(
    model,
    labeled_loader,
    unlabeled_loader,
    val_loader,
    test_loader,
    device,
    num_classes: int,
    base_lr: float = 2e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 50,
    patience: int = 8,
    tau: float = 0.95,
    lambda_u: float = 1.0,
    flex_alpha: float = 0.9,
    tau_min_scale: float = 0.5,
    tau_max_scale: float = 1.0,
    class_weights=None,
):
    model.to(device)

    head_params, base_params = [], []
    for n, p in model.named_parameters():
        if "classifier" in n or "score.weight" in n or "score.bias" in n:
            head_params.append(p)
        else:
            base_params.append(p)

    opt = torch.optim.AdamW([
        {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ])

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = max(1, len(labeled_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    eps = 1e-6
    select_rate_ema = torch.zeros(num_classes, dtype=torch.float32, device=device)

    def supervised_ce(logits, targets):
        if class_weights is not None:
            return F.cross_entropy(logits, targets, weight=class_weights.to(dtype=logits.dtype))
        return F.cross_entropy(logits, targets)

    def logits_of(enc_batch):
        enc = {k: v.to(device) for k, v in enc_batch.items() if k != "labels"}
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(**enc)
            return out.logits

    def iterate_minibatches():
        u_iter = iter(unlabeled_loader)
        for l_batch in labeled_loader:
            try:
                u_batch = next(u_iter)
            except StopIteration:
                u_iter = iter(unlabeled_loader)
                u_batch = next(u_iter)
            yield l_batch, u_batch

    @torch.no_grad()
    def flexmatch_pseudolabel_and_mask(logits_w, return_debug=False):
        probs = torch.softmax(logits_w, dim=1)
        conf, yhat = probs.max(dim=1)

        r = select_rate_ema.clamp(min=0.0)
        r_norm = r / (r.max() + eps)
        tau_c = tau * (tau_min_scale + (tau_max_scale - tau_min_scale) * r_norm)
        thr = tau_c[yhat]
        mask = (conf >= thr).float()

        if yhat.numel() > 0:
            for c in range(num_classes):
                in_c = (yhat == c)
                if in_c.any():
                    pick_rate_c = (mask[in_c] > 0).float().mean()
                    select_rate_ema[c] = flex_alpha * select_rate_ema[c] + (1 - flex_alpha) * pick_rate_c

        if return_debug:
            return yhat, mask, thr, conf, tau_c.detach()
        return yhat, mask

    best_val = -1.0
    bad_epochs = 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        sup_loss_meter = 0.0
        unsup_loss_meter = 0.0
        steps = 0

        acc_accept = 0.0
        acc_conf = 0.0
        acc_thr = 0.0
        acc_steps = 0
        last_tau_c = None

        for l_batch, (enc_w, enc_s) in iterate_minibatches():
            y_l = l_batch["labels"].to(device)

            logits_l = logits_of(l_batch)
            loss_sup = supervised_ce(logits_l, y_l)

            with torch.no_grad():
                logits_w = logits_of(enc_w)
                yhat, mask, thr, conf, tau_c = flexmatch_pseudolabel_and_mask(
                    logits_w, return_debug=True
                )

                acc_accept += float(mask.mean().item())
                if mask.sum() > 0:
                    acc_conf += float(conf[mask > 0].mean().item())
                    acc_thr += float(thr[mask > 0].mean().item())
                acc_steps += 1
                last_tau_c = tau_c

            logits_s = logits_of(enc_s)
            if mask.numel() == 0 or mask.sum() == 0:
                loss_unsup = torch.tensor(0.0, device=device)
            else:
                per = F.cross_entropy(logits_s, yhat.to(device), reduction='none')
                loss_unsup = (per * mask.to(device)).sum() / mask.sum().clamp(min=1.0)

            loss = loss_sup + lambda_u * loss_unsup

            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            scheduler.step()

            sup_loss_meter += loss_sup.item()
            unsup_loss_meter += loss_unsup.item()
            steps += 1

        val_acc = evaluate(model, val_loader, device, use_amp=use_amp)

        mean_accept = acc_accept / max(1, acc_steps)
        mean_sel_conf = acc_conf / max(1, acc_steps)
        mean_sel_thr = acc_thr / max(1, acc_steps)
        tau_print = np.round(last_tau_c.detach().cpu().numpy(), 3) if last_tau_c is not None else None

        print(
            f"[Ep {ep:02d}] sup={sup_loss_meter/max(1,steps):.4f}  "
            f"unsup={unsup_loss_meter/max(1,steps):.4f}  val_acc={val_acc:.4f}  "
            f"| r_ema={select_rate_ema.detach().cpu().numpy()}  "
            f"| acc={mean_accept:.3f}  sel_conf={mean_sel_conf:.3f}  "
            f"sel_thr={mean_sel_thr:.3f}  tau_c={tau_print}"
        )

        if val_acc > best_val:
            best_val = val_acc
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[EarlyStop] patience {patience} reached at epoch {ep}. best val_acc={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    test_acc = evaluate(model, test_loader, device, use_amp=use_amp)
    return {
        "best_val_acc": best_val,
        "test_acc": test_acc,
    }
