# src/inference.py

import os
import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (
    build_tokenize_sequences,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_input_csv(path: str, sequence_col: str = "sequence") -> pd.DataFrame:
    df = pd.read_csv(path)

    if sequence_col not in df.columns and "DNA Sequence" in df.columns:
        df = df.rename(columns={"DNA Sequence": sequence_col})

    if sequence_col not in df.columns:
        raise ValueError(f"'{sequence_col}' column missing in {path}")

    out = df.copy()
    out[sequence_col] = out[sequence_col].astype(str).str.upper()
    return out


class InferenceSet(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, sequence_col: str = "sequence"):
        self.seqs = df[sequence_col].tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def build_inference_collate(tokenize_sequences):
    def collate_fn(batch):
        return tokenize_sequences(list(batch))
    return collate_fn


@torch.no_grad()
def predict(model, loader, device, use_amp: bool = True):
    model.eval()

    all_pred_ids = []
    all_probs = []

    for batch in loader:
        enc = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=use_amp and torch.cuda.is_available()):
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)

        pred_ids = probs.argmax(dim=1)

        all_pred_ids.extend(pred_ids.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    return all_pred_ids, all_probs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained genomic SSL classifier."
    )

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory produced by train.py containing model_state.pt, label_map.json, train_config.json, and tokenizer files.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--sequence_col", type=str, default=None,
                        help="Sequence column in input CSV. If omitted, tries train_config value, else defaults to 'sequence'.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=None,
                        help="Optional override. If omitted, uses train_config resolved_max_len if available.")
    parser.add_argument("--device", type=str, default=None,
                        help="Optional override: cpu or cuda")

    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    model_state_path = model_dir / "model_state.pt"
    label_map_path = model_dir / "label_map.json"
    config_path = model_dir / "train_config.json"

    if not model_state_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_state_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing label map: {label_map_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    inv_label_map = {int(v): k for k, v in label_map.items()}

    with open(config_path, "r") as f:
        train_config = json.load(f)

    model_name = train_config["model_name"]
    num_classes = train_config["num_classes"]

    if args.sequence_col is not None:
        sequence_col = args.sequence_col
    else:
        sequence_col = train_config.get("sequence_col", "sequence")

    if args.max_len is not None:
        max_len = args.max_len
    else:
        max_len = train_config.get("resolved_max_len", 1000)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] device={device}")
    print(f"[INFO] model_name={model_name}")
    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] max_len={max_len}")

    # tokenizer: prefer local saved tokenizer from model_dir, fallback to model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        print(f"[INFO] Loaded tokenizer from {model_dir}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print(f"[WARN] Could not load tokenizer from {model_dir}; fell back to {model_name}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
    )

    state_dict = torch.load(model_state_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    df = load_input_csv(args.input_csv, sequence_col=sequence_col)

    dataset = InferenceSet(df, sequence_col=sequence_col)
    tokenize_sequences = build_tokenize_sequences(tokenizer, max_len)
    collate_fn = build_inference_collate(tokenize_sequences)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    pred_ids, probs = predict(model, loader, device=device, use_amp=True)

    out_df = df.copy()
    out_df["pred_id"] = pred_ids
    out_df["pred_label"] = [inv_label_map[i] for i in pred_ids]
    out_df["pred_confidence"] = [max(p) for p in probs]

    # per-class probability columns
    for class_id in sorted(inv_label_map.keys()):
        class_name = inv_label_map[class_id]
        safe_name = str(class_name).replace(" ", "_")
        out_df[f"prob_{safe_name}"] = [p[class_id] for p in probs]

    os.makedirs(Path(args.output_csv).parent or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"[INFO] Saved predictions to: {args.output_csv}")
    print(out_df.head())


if __name__ == "__main__":
    main()
