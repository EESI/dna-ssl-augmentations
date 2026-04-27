#!/usr/bin/env python3
"""
Convert FASTA to CSV using this format:

>sequence_id label
ACGTACGTACGT

The last whitespace-separated token in the FASTA header is used as the label.
Example:
    >seq001 0
    ACGTACGT
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Generator, Optional, Tuple


Record = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FASTA to CSV where the label is the last token in the header."
    )
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--keep-case", action="store_true")
    parser.add_argument("--allow-non-acgtn", action="store_true")
    parser.add_argument("--deduplicate", action="store_true")
    return parser.parse_args()


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


def normalize_sequence(seq: str, keep_case: bool) -> str:
    seq = re.sub(r"\s+", "", seq)
    return seq if keep_case else seq.upper()


def validate_sequence(seq: str) -> None:
    if not re.fullmatch(r"[ACGTUNRYKMSWBDHVX\-\.]+", seq, flags=re.IGNORECASE):
        raise ValueError(
            "Sequence contains unsupported characters. "
            "Use --allow-non-acgtn to skip validation."
        )


def extract_label_from_header(header: str) -> str:
    fields = header.split()

    if len(fields) < 2:
        raise ValueError(
            "Invalid labeled FASTA header. Expected format like:\n"
            ">seq001 0\n"
            f"But got:\n>{header}"
        )

    return fields[-1]


def convert_fasta_to_csv(
    input_fasta: Path,
    output_csv: Path,
    sequence_col: str = "sequence",
    label_col: str = "label",
    keep_case: bool = False,
    allow_non_acgtn: bool = False,
    deduplicate: bool = False,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seen_sequences: set[str] = set()

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[sequence_col, label_col])
        writer.writeheader()

        for header, seq in read_fasta(input_fasta):
            seq = normalize_sequence(seq, keep_case=keep_case)

            if not seq:
                continue

            if not allow_non_acgtn:
                validate_sequence(seq)

            if deduplicate:
                if seq in seen_sequences:
                    continue
                seen_sequences.add(seq)

            label = extract_label_from_header(header)

            writer.writerow(
                {
                    sequence_col: seq,
                    label_col: label,
                }
            )
            count += 1

    return count


def main() -> None:
    args = parse_args()

    input_fasta = Path(args.input_fasta)
    output_csv = Path(args.output_csv)

    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    n = convert_fasta_to_csv(
        input_fasta=input_fasta,
        output_csv=output_csv,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        keep_case=args.keep_case,
        allow_non_acgtn=args.allow_non_acgtn,
        deduplicate=args.deduplicate,
    )

    print(f"Wrote {n} records to {output_csv}")


if __name__ == "__main__":
    main()
