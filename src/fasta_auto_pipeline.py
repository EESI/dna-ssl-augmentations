#!/usr/bin/env python3

"""
Auto FASTA → CSV → (optional) train / inference pipeline

Supports:
- Labeled FASTA:  >seq001 0
- Unlabeled FASTA: >seq001

Automatically detects label presence unless overridden.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from fasta_to_csv import read_fasta, write_csv


def detect_labeled(fasta_path: Path) -> bool:
    """Check if FASTA headers contain labels (last token)."""
    for header, _ in read_fasta(fasta_path):
        if len(header.split()) >= 2:
            return True
        return False
    return False


def fasta_to_csv_auto(
    fasta_path: Path,
    output_csv: Path,
    force_unlabeled: bool = False,
) -> None:
    is_labeled = False if force_unlabeled else detect_labeled(fasta_path)

    write_csv(
        records=read_fasta(fasta_path),
        output_csv=output_csv,
        sequence_col="sequence",
        label_col="label",
        unlabeled=not is_labeled,
        label_from_header=False,
        label_key=None,
        header_split_delim=None,
        label_index=None,
        keep_case=False,
        allow_non_acgtn=False,
        deduplicate=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["convert", "train", "inference"], required=True)

    parser.add_argument("--input_fasta")
    parser.add_argument("--output_csv")

    parser.add_argument("--labeled_fasta")
    parser.add_argument("--unlabeled_fasta")
    parser.add_argument("--val_fasta")
    parser.add_argument("--test_fasta")

    parser.add_argument("--model_dir")
    parser.add_argument("--output_dir")

    parser.add_argument("--force-unlabeled", action="store_true")

    args = parser.parse_args()

    # -------- convert only --------
    if args.mode == "convert":
        fasta_to_csv_auto(
            Path(args.input_fasta),
            Path(args.output_csv),
            force_unlabeled=args.force_unlabeled,
        )
        print(f"Converted → {args.output_csv}")
        return

    # -------- train --------
    if args.mode == "train":
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            labeled_csv = tmp / "labeled.csv"
            unlabeled_csv = tmp / "unlabeled.csv"
            val_csv = tmp / "val.csv"
            test_csv = tmp / "test.csv"

            fasta_to_csv_auto(Path(args.labeled_fasta), labeled_csv)
            fasta_to_csv_auto(Path(args.unlabeled_fasta), unlabeled_csv, force_unlabeled=True)
            fasta_to_csv_auto(Path(args.val_fasta), val_csv)
            fasta_to_csv_auto(Path(args.test_fasta), test_csv)

            cmd = [
                "python", "src/train.py",
                "--labeled_csv", str(labeled_csv),
                "--unlabeled_csv", str(unlabeled_csv),
                "--val_csv", str(val_csv),
                "--test_csv", str(test_csv),
                "--output_dir", args.output_dir,
            ]

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

        return

    # -------- inference --------
    if args.mode == "inference":
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            input_csv = tmp / "input.csv"

            fasta_to_csv_auto(
                Path(args.input_fasta),
                input_csv,
                force_unlabeled=True,
            )

            cmd = [
                "python", "src/inference.py",
                "--model_dir", args.model_dir,
                "--input_csv", str(input_csv),
                "--output_csv", args.output_csv,
            ]

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

        return


if __name__ == "__main__":
    main()
