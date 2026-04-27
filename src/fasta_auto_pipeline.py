#!/usr/bin/env python3
"""
Auto FASTA → CSV → train / inference pipeline

Supported FASTA formats:

1) Labeled FASTA
   >seq001 0
   ACGTACGT

   The last whitespace-separated token in the header is used as the label.
   Labels must be integers.

2) Unlabeled FASTA
   >seq001
   ACGTACGT
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Generator, Optional, Tuple


Record = Tuple[str, str]


def read_fasta(path: Path) -> Generator[Record, None, None]:
    header: Optional[str] = None
    seq_chunks: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)

                header = line[1:].strip()
                seq_chunks = []
            else:
                if header is None:
                    raise ValueError(
                        f"Invalid FASTA: sequence found before first header at line {line_num}."
                    )
                seq_chunks.append(line)

    if header is not None:
        yield header, "".join(seq_chunks)


def normalize_sequence(seq: str) -> str:
    seq = re.sub(r"\s+", "", seq)
    return seq.upper()


def validate_sequence(seq: str) -> None:
    if not re.fullmatch(r"[ACGTUNRYKMSWBDHVX\-\.]+", seq, flags=re.IGNORECASE):
        raise ValueError(
            "Sequence contains unsupported characters."
        )


def detect_labeled(fasta_path: Path) -> bool:
    for header, _ in read_fasta(fasta_path):
        return len(header.split()) >= 2
    return False


def extract_integer_label(header: str) -> str:
    fields = header.split()

    if len(fields) < 2:
        raise ValueError(
            "Expected labeled FASTA header format like '>seq001 0', "
            f"but got: '>{header}'"
        )

    label = fields[-1]

    if not label.isdigit():
        raise ValueError(
            "FASTA label must be an integer value such as 0, 1, 2, ... "
            f"but got label='{label}' in header: '>{header}'"
        )

    return label


def fasta_to_csv_auto(
    fasta_path: Path,
    output_csv: Path,
    force_unlabeled: bool = False,
) -> Path:
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    is_labeled = False if force_unlabeled else detect_labeled(fasta_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sequence", "label"] if is_labeled else ["sequence"]
    count = 0

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for header, seq in read_fasta(fasta_path):
            seq = normalize_sequence(seq)

            if not seq:
                continue

            validate_sequence(seq)

            row = {"sequence": seq}

            if is_labeled:
                row["label"] = extract_integer_label(header)

            writer.writerow(row)
            count += 1

    print(f"Converted {fasta_path} → {output_csv} ({count} records)")
    return output_csv


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=["fixmatch", "flexmatch"])
    parser.add_argument("--weak_aug", default="nn")
    parser.add_argument("--strong_aug", default="mutation")

    parser.add_argument("--epochs")
    parser.add_argument("--batch_size")
    parser.add_argument("--lr")
    parser.add_argument("--seed")

    parser.add_argument(
        "--extra_train_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed directly to src/train.py",
    )


def append_optional_arg(cmd: list[str], name: str, value: Optional[str]) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto FASTA to CSV pipeline for training and inference."
    )

    parser.add_argument(
        "--mode",
        choices=["convert", "train", "inference"],
        required=True,
    )

    parser.add_argument("--input_fasta")
    parser.add_argument("--output_csv")

    parser.add_argument("--labeled_fasta")
    parser.add_argument("--unlabeled_fasta")
    parser.add_argument("--val_fasta")
    parser.add_argument("--test_fasta")

    parser.add_argument("--model_dir")
    parser.add_argument("--output_dir")

    parser.add_argument(
        "--force-unlabeled",
        action="store_true",
        help="Treat input FASTA as unlabeled even if headers contain spaces.",
    )

    add_train_args(parser)

    args = parser.parse_args()

    if args.mode == "convert":
        if not args.input_fasta or not args.output_csv:
            raise ValueError("--input_fasta and --output_csv are required for convert mode.")

        fasta_to_csv_auto(
            fasta_path=Path(args.input_fasta),
            output_csv=Path(args.output_csv),
            force_unlabeled=args.force_unlabeled,
        )
        return

    if args.mode == "train":
        required = {
            "--labeled_fasta": args.labeled_fasta,
            "--unlabeled_fasta": args.unlabeled_fasta,
            "--val_fasta": args.val_fasta,
            "--test_fasta": args.test_fasta,
            "--method": args.method,
            "--output_dir": args.output_dir,
        }

        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing required arguments for train mode: {', '.join(missing)}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            labeled_csv = fasta_to_csv_auto(
                Path(args.labeled_fasta),
                tmp / "labeled.csv",
            )
            unlabeled_csv = fasta_to_csv_auto(
                Path(args.unlabeled_fasta),
                tmp / "unlabeled.csv",
                force_unlabeled=True,
            )
            val_csv = fasta_to_csv_auto(
                Path(args.val_fasta),
                tmp / "val.csv",
            )
            test_csv = fasta_to_csv_auto(
                Path(args.test_fasta),
                tmp / "test.csv",
            )

            cmd = [
                "python",
                "src/train.py",
                "--labeled_csv",
                str(labeled_csv),
                "--unlabeled_csv",
                str(unlabeled_csv),
                "--val_csv",
                str(val_csv),
                "--test_csv",
                str(test_csv),
                "--method",
                args.method,
                "--weak_aug",
                args.weak_aug,
                "--strong_aug",
                args.strong_aug,
                "--output_dir",
                args.output_dir,
            ]

            append_optional_arg(cmd, "--epochs", args.epochs)
            append_optional_arg(cmd, "--batch_size", args.batch_size)
            append_optional_arg(cmd, "--lr", args.lr)
            append_optional_arg(cmd, "--seed", args.seed)

            if args.extra_train_args:
                cmd.extend(args.extra_train_args)

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

        return

    if args.mode == "inference":
        if not args.input_fasta or not args.model_dir or not args.output_csv:
            raise ValueError(
                "--input_fasta, --model_dir, and --output_csv are required for inference mode."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = fasta_to_csv_auto(
                fasta_path=Path(args.input_fasta),
                output_csv=Path(tmpdir) / "input.csv",
                force_unlabeled=True,
            )

            cmd = [
                "python",
                "src/inference.py",
                "--model_dir",
                args.model_dir,
                "--input_csv",
                str(input_csv),
                "--output_csv",
                args.output_csv,
            ]

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

        return


if __name__ == "__main__":
    main()
