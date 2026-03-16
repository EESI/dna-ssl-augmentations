import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from augmentations import build_augment_pairs
from common import (
    set_seed,
    load_csv,
    build_dense_label_map,
    make_ssl_split,
    load_tokenizer_and_model,
    truncate_dataframes,
    LabeledSetHF,
    UnlabeledSetHF,
    EvalSetHF,
    build_tokenize_sequences,
    build_collate_fns,
)
from fixmatch_core import train_fixmatch


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

    label_map, _ = build_dense_label_map(train_df, val_df, test_df)
    num_classes = len(label_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "PoetschLab/GROVER"
    results = []

    for combo in build_augment_pairs():
        weak_name = combo["weak_name"]
        strong_name = combo["strong_name"]
        weak_aug = combo["weak_fn"]
        strong_aug = combo["strong_fn"]

        print("\n" + "=" * 60)
        print(f"Running AMR FixMatch: weak={weak_name}, strong={strong_name}")
        print("=" * 60)

        for seed in [777]:
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
                collate_fn=collate_l,
            )
            u_loader = DataLoader(
                UnlabeledSetHF(u_df, weak_aug, strong_aug),
                batch_size=16 * 7,
                shuffle=True,
                collate_fn=collate_u,
            )
            val_loader = DataLoader(
                EvalSetHF(val_df, label_map),
                batch_size=64,
                shuffle=False,
                collate_fn=collate_e,
            )
            test_loader = DataLoader(
                EvalSetHF(test_df, label_map),
                batch_size=64,
                shuffle=False,
                collate_fn=collate_e,
            )

            out = train_fixmatch(
                model=model,
                labeled_loader=l_loader,
                unlabeled_loader=u_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                lr=2e-5,
                epochs=50,
                threshold=0.95,
            )

            print(f"[{weak_name}-{strong_name} | seed={seed}] Test Acc={out['test_acc']:.4f}")

            results.append({
                "dataset": "amr",
                "method": "fixmatch",
                "weak": weak_name,
                "strong": strong_name,
                "seed": seed,
                "best_val_acc": out["best_val_acc"],
                "test_acc": out["test_acc"],
            })

    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(results).to_csv("outputs/amr_fixmatch_results.csv", index=False)


if __name__ == "__main__":
    main()
