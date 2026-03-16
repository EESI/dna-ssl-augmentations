import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from augmentations import AUGMENT_FUNCS
from common import (
    set_seed,
    load_csv,
    build_dense_label_map,
    make_ssl_split,
    infer_label_to_drugclass,
    load_tokenizer_and_model,
    truncate_dataframes,
    LabeledSetHF,
    UnlabeledSetHF,
    EvalSetHF,
    build_tokenize_sequences,
    build_collate_fns,
)
from flexmatch_core import train_flexmatch


def main():
    base = "data/amr"
    csv_paths = {
        "train": os.path.join(base, "train_6classes.csv"),
        "val": os.path.join(base, "val_6classes.csv"),
        "test": os.path.join(base, "test_6classes.csv"),
    }

    train_df = load_csv(csv_paths["train"])
    val_df = load_csv(csv_paths["val"])
    test_df = load_csv(csv_paths["test"])

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    label2drugclass, conflicts = infer_label_to_drugclass(all_df)

    label_map, dense2orig = build_dense_label_map(train_df, val_df, test_df)
    num_classes = len(label_map)

    weak_aug = AUGMENT_FUNCS["nn"][0]
    strong_aug = AUGMENT_FUNCS["mutation"][1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "PoetschLab/GROVER"

    acc_list = []

    for seed in [42, 123, 2023, 2025, 777]:
        print(f"\n===== AMR FlexMatch | seed={seed} =====")
        set_seed(seed)

        l_df, u_df = make_ssl_split(train_df, k_per_class=50)
        tokenizer, model = load_tokenizer_and_model(model_name, num_classes)
        max_len = min(1000, tokenizer.model_max_length or 1000)
        truncate_dataframes([l_df, u_df, val_df, test_df], max_len)

        tokenize_sequences = build_tokenize_sequences(tokenizer, max_len)
        collate_l, collate_u, collate_e = build_collate_fns(tokenize_sequences)

        l_loader = DataLoader(
            LabeledSetHF(l_df, label_map, weak_aug),
            batch_size=16,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_l,
        )
        u_loader = DataLoader(
            UnlabeledSetHF(u_df, weak_aug, strong_aug),
            batch_size=16 * 7,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_u,
        )
        val_loader = DataLoader(
            EvalSetHF(val_df, label_map),
            batch_size=64,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_e,
        )
        test_loader = DataLoader(
            EvalSetHF(test_df, label_map),
            batch_size=64,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_e,
        )

        counts = l_df["label"].value_counts()
        weights = np.zeros(num_classes, dtype=np.float32)
        for lab, cnt in counts.items():
            weights[label_map[lab]] = 1.0 / max(1, cnt)
        weights = weights * (num_classes / (weights.sum() + 1e-12))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        out = train_flexmatch(
            model=model,
            labeled_loader=l_loader,
            unlabeled_loader=u_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_classes=num_classes,
            base_lr=2e-5,
            head_lr=1e-3,
            weight_decay=0.01,
            epochs=50,
            patience=8,
            tau=0.95,
            lambda_u=1.0,
            flex_alpha=0.9,
            tau_min_scale=0.5,
            tau_max_scale=1.0,
            class_weights=class_weights,
        )

        print(f"[BEST] Test Acc: {out['test_acc']:.4f}  (best val_acc={out['best_val_acc']:.4f})")
        acc_list.append(out["test_acc"])

    print("\nacc_list:", acc_list)
    print("Average:", float(np.mean(acc_list)))
    print("Var:", float(np.var(acc_list)))
    print("Max:", float(np.max(acc_list)))
    print("Min:", float(np.min(acc_list)))


if __name__ == "__main__":
    main()
