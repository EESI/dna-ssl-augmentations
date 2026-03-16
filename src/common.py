import random
from collections import Counter
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.tokenization_utils_base import PaddingStrategy


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sequence" not in df.columns and "DNA Sequence" in df.columns:
        df = df.rename(columns={"DNA Sequence": "sequence"})
    assert "sequence" in df.columns, f"'sequence' column missing in {path}"
    assert "label" in df.columns, f"'label' column missing in {path}"

    df["label"] = df["label"].astype(str)
    df["sequence"] = df["sequence"].astype(str).str.upper()
    return df


def build_dense_label_map(*dfs, label_col: str = "label") -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq, seen = [], set()
    for d in dfs:
        for v in d[label_col].unique().tolist():
            if v not in seen:
                uniq.append(v)
                seen.add(v)

    uniq_sorted = sorted(uniq, key=lambda x: int(x) if x.isdigit() else x)
    dense = {lab: i for i, lab in enumerate(uniq_sorted)}
    inv = {i: lab for lab, i in dense.items()}
    return dense, inv


def make_ssl_split(df: pd.DataFrame, k_per_class: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labeled_idx, unlabeled_idx = [], []
    for lab, sub in df.groupby("label"):
        idx = sub.index.tolist()
        random.shuffle(idx)
        k = min(k_per_class, len(idx))
        labeled_idx += idx[:k]
        unlabeled_idx += idx[k:]
    labeled_df = df.loc[labeled_idx].reset_index(drop=True)
    unlabeled_df = df.loc[unlabeled_idx].reset_index(drop=True)
    return labeled_df, unlabeled_df


def infer_label_to_drugclass(
    df: pd.DataFrame,
    label_col: str = "label",
    drug_col: str = "Drug Class"
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    if drug_col not in df.columns:
        return {}, {}

    pairs = df[[label_col, drug_col]].dropna().astype({label_col: str, drug_col: str})
    label2drug = {}
    conflicts = {}

    for lab, sub in pairs.groupby(label_col):
        counts = Counter(sub[drug_col])
        top = counts.most_common(1)[0][0]
        label2drug[lab] = top
        if len(counts) > 1:
            conflicts[lab] = dict(counts)

    return label2drug, conflicts


def truncate_dataframes(dfs: List[pd.DataFrame], max_len: int):
    for d in dfs:
        d["sequence"] = d["sequence"].map(lambda s: s[:max_len])


def load_tokenizer_and_model(model_name: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return tokenizer, model


class LabeledSetHF(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: Dict[str, int], weak_aug):
        self.seqs = df["sequence"].tolist()
        self.ys = [label_map[str(y)] for y in df["label"].tolist()]
        self.weak_aug = weak_aug

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.weak_aug(self.seqs[i]), self.ys[i]


class UnlabeledSetHF(Dataset):
    def __init__(self, df: pd.DataFrame, weak_aug, strong_aug):
        self.seqs = df["sequence"].tolist()
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        return self.weak_aug(s), self.strong_aug(s)


class EvalSetHF(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: Dict[str, int]):
        self.seqs = df["sequence"].tolist()
        self.ys = [label_map[str(y)] for y in df["label"].tolist()]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.ys[i]


def build_tokenize_sequences(tokenizer, max_len: int):
    def tokenize_sequences(seqs: List[str]):
        return tokenizer(
            seqs,
            padding=PaddingStrategy.LONGEST,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
            return_tensors="pt",
        )
    return tokenize_sequences


def build_collate_fns(tokenize_sequences):
    def collate_labeled(batch):
        seqs, ys = zip(*batch)
        enc = tokenize_sequences(list(seqs))
        enc["labels"] = torch.tensor(ys, dtype=torch.long)
        return enc

    def collate_unlabeled(batch):
        xw, xs = zip(*batch)
        enc_w = tokenize_sequences(list(xw))
        enc_s = tokenize_sequences(list(xs))
        return enc_w, enc_s

    def collate_eval(batch):
        seqs, ys = zip(*batch)
        enc = tokenize_sequences(list(seqs))
        enc["labels"] = torch.tensor(ys, dtype=torch.long)
        return enc

    return collate_labeled, collate_unlabeled, collate_eval


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool = True):
    model.eval()
    correct = 0
    n = 0

    for batch in loader:
        y = batch["labels"].to(device)
        enc = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        with torch.amp.autocast("cuda", enabled=use_amp and torch.cuda.is_available()):
            logits = model(**enc).logits

        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += y.size(0)

    return correct / max(1, n)
