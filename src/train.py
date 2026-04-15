# src/train.py

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from augmentation import AUGMENT_FUNCS
from common import (
    set_seed,
    build_dense_label_map,
    load_tokenizer_and_model,
    truncate_dataframes,
    LabeledSetHF,
    UnlabeledSetHF,
    EvalSetHF,
    build_tokenize_sequences,
    build_collate_fns,
)
from fixmatch_core import train_fixmatch
from flexmatch_core import train_flexmatch


def load_csv_flexible(
    path: str,
    sequence_col: str = "sequence",
    label_col: str = "label",
    require_label: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    # allow "DNA Sequence" fallback like your current common.load_csv
    if sequence_col not in df.columns and "DNA Sequence" in df.columns:
        df = df.rename(columns={"DNA Sequence": sequence_col})

    if sequence_col not in df.columns:
        raise ValueError(f"'{sequence_col}' column missing in {path}")

    out = pd.DataFrame()
    out["sequence"] = df[sequence_col].astype(str).str.upper()

    if require_label:
        if label_col not in df.columns:
            raise ValueError(f"'{label_col}' column missing in {path}")
        out["label"] = df[label_col].astype(str)

    return out


def build_class_weights(labeled_df: pd.DataFrame, label_map: dict, device: torch.device):
    counts = labeled_df["label"].value_counts()
    num_classes = len(label_map)

    weights = np.zeros(num_classes, dtype=np.float32)
    for lab, cnt in counts.items():
        weights[label_map[str(lab)]] = 1.0 / max(1, cnt)

    weights = weights * (num_classes / (weights.sum() + 1e-12))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_artifacts(
    output_dir: str,
    model,
    tokenizer,
    label_map: dict,
    args_dict: dict,
    metrics: dict,
):
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, "model_state.pt")
    torch.save(model.state_dict(), ckpt_path)

    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    inv_label_map = {v: k for k, v in label_map.items()}
    with open(os.path.join(output_dir, "inv_label_map.json"), "w") as f:
        json.dump(inv_label_map, f, indent=2)

    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # optional but convenient for HF users
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"[WARN] tokenizer.save_pretrained failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FixMatch/FlexMatch on labeled and unlabeled genomic sequence CSVs."
    )

    # data
    parser.add_argument("--labeled_csv", type=str, required=True)
    parser.add_argument("--unlabeled_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=None)

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--label_col", type=str, default="label")

    # method / model
    parser.add_argument("--method", type=str, choices=["fixmatch", "flexmatch"], required=True)
    parser.add_argument("--model_name", type=str, default="PoetschLab/GROVER")
    parser.add_argument("--max_len", type=int, default=None)

    # augmentation
    parser.add_argument("--weak_aug", type=str, choices=list(AUGMENT_FUNCS.keys()), required=True)
    parser.add_argument("--strong_aug", type=str, choices=list(AUGMENT_FUNCS.keys()), required=True)

    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled_batch_size", type=int, default=16)
    parser.add_argument("--unlabeled_batch_size", type=int, default=112)
    parser.add_argument("--eval_batch_size", type=int, default=64)

    # fixmatch params
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--threshold", type=float, default=0.95)

    # flexmatch params
    parser.add_argument("--base_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--flex_alpha", type=float, default=0.9)
    parser.add_argument("--tau_min_scale", type=float, default=0.5)
    parser.add_argument("--tau_max_scale", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # load user-provided datasets
    labeled_df = load_csv_flexible(
        args.labeled_csv,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        require_label=True,
    )
    unlabeled_df = load_csv_flexible(
        args.unlabeled_csv,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        require_label=False,
    )
    val_df = load_csv_flexible(
        args.val_csv,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        require_label=True,
    )

    test_df = None
    if args.test_csv is not None:
        test_df = load_csv_flexible(
            args.test_csv,
            sequence_col=args.sequence_col,
            label_col=args.label_col,
            require_label=True,
        )

    # build label map from labeled/val/(optional test)
    dfs_for_label_map = [labeled_df, val_df]
    if test_df is not None:
        dfs_for_label_map.append(test_df)

    label_map, inv_label_map = build_dense_label_map(*dfs_for_label_map)
    num_classes = len(label_map)

    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] label_map={label_map}")

    weak_aug = AUGMENT_FUNCS[args.weak_aug][0]
    strong_aug = AUGMENT_FUNCS[args.strong_aug][1]

    tokenizer, model = load_tokenizer_and_model(args.model_name, num_classes)

    if args.max_len is None:
        model_max = getattr(tokenizer, "model_max_length", 1000)
        if model_max is None or model_max > 1000:
            max_len = 1000
        else:
            max_len = model_max
    else:
        max_len = args.max_len

    truncate_targets = [labeled_df, unlabeled_df, val_df]
    if test_df is not None:
        truncate_targets.append(test_df)
    truncate_dataframes(truncate_targets, max_len)

    tokenize_sequences = build_tokenize_sequences(tokenizer, max_len)
    collate_l, collate_u, collate_e = build_collate_fns(tokenize_sequences)

    labeled_loader = DataLoader(
        LabeledSetHF(labeled_df, label_map, weak_aug),
        batch_size=args.labeled_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_l,
    )

    unlabeled_loader = DataLoader(
        UnlabeledSetHF(unlabeled_df, weak_aug, strong_aug),
        batch_size=args.unlabeled_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_u,
    )

    val_loader = DataLoader(
        EvalSetHF(val_df, label_map),
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_e,
    )

    # current core functions require test_loader.
    # if user does not provide test_csv, we reuse val_loader only to satisfy the signature.
    # later you may want to patch core to make test optional.
    if test_df is not None:
        test_loader = DataLoader(
            EvalSetHF(test_df, label_map),
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_e,
        )
        has_real_test = True
    else:
        test_loader = val_loader
        has_real_test = False
        print("[WARN] No --test_csv provided. test_acc in output will mirror validation-based evaluation for compatibility.")

    if args.method == "fixmatch":
        metrics = train_fixmatch(
            model=model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            lr=args.lr,
            epochs=args.epochs,
            threshold=args.threshold,
        )
    else:
        class_weights = build_class_weights(labeled_df, label_map, device)

        metrics = train_flexmatch(
            model=model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_classes=num_classes,
            base_lr=args.base_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            tau=args.tau,
            lambda_u=args.lambda_u,
            flex_alpha=args.flex_alpha,
            tau_min_scale=args.tau_min_scale,
            tau_max_scale=args.tau_max_scale,
            class_weights=class_weights,
        )

    if not has_real_test:
        metrics["test_acc"] = None

    args_dict = vars(args).copy()
    args_dict["resolved_max_len"] = max_len
    args_dict["num_classes"] = num_classes
    args_dict["device"] = str(device)

    save_artifacts(
        output_dir=str(output_dir),
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        args_dict=args_dict,
        metrics=metrics,
    )

    print("\n[INFO] Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"[INFO] Saved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
