#!/usr/bin/env python3
"""
Convert FASTA files to CSV for dna-ssl-augmentations.

Supported outputs:
- Unlabeled CSV: sequence
- Labeled CSV: sequence,label

Label extraction modes:
1) --unlabeled
   Ignore FASTA headers and export only sequences.

2) --label-from-header
   Use the full FASTA header (without '>') as the label.

3) --header-split-delim + --label-index
   Split the header string and take one field as label.
   Example header:
       >seq001|classA
   Example args:
       --header-split-delim '|' --label-index 1

4) --label-key
   Parse key=value pairs in the FASTA header.
   Example header:
       >seq001 sample=x label=classA source=lab
   Example args:
       --label-key label

By default, sequences are uppercased and internal whitespace is removed.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple


Record = Tuple[str, str]  # (header, sequence)


FASTA_SUFFIXES = {".fasta", ".fa", ".fna", ".ffn", ".faa", ".frn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FASTA to CSV for dna-ssl-augmentations."
    )
    parser.add_argument("--input_fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV file")

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--unlabeled",
        action="store_true",
        help="Create an unlabeled CSV with only the 'sequence' column",
    )
    mode.add_argument(
        "--label-from-header",
        action="store_true",
        help="Use the full FASTA header as the label",
    )
    mode.add_argument(
        "--label-key",
        type=str,
        default=None,
        help="Extract label from key=value in the FASTA header, e.g. --label-key label",
    )

    parser.add_argument(
        "--header-split-delim",
        type=str,
        default=None,
        help="Delimiter used to split the FASTA header, e.g. '|' or '\\t'",
    )
    parser.add_argument(
        "--label-index",
        type=int,
        default=None,
        help="Index of label field after splitting the FASTA header",
    )
    parser.add_argument(
        "--sequence-col",
        default="sequence",
        help="Output sequence column name (default: sequence)",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Output label column name (default: label)",
    )
    parser.add_argument(
        "--keep-case",
        action="store_true",
        help="Preserve sequence case instead of converting to uppercase",
    )
    parser.add_argument(
        "--allow-non-acgtn",
        action="store_true",
        help="Do not validate sequence characters",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Drop duplicate sequences (first occurrence kept)",
    )

    args = parser.parse_args()

    if args.header_split_delim is not None and args.label_index is None:
        parser.error("--header-split-delim requires --label-index")
    if args.label_index is not None and args.header_split_delim is None:
        parser.error("--label-index requires --header-split-delim")

    if (
        not args.unlabeled
        and not args.label_from_header
        and args.label_key is None
        and args.header_split_delim is None
    ):
        parser.error(
            "Choose one labeling mode: --unlabeled, --label-from-header, --label-key, "
            "or --header-split-delim + --label-index"
        )

    return args


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
                        f"Invalid FASTA: sequence data found before first header at line {line_num}."
                    )
                seq_chunks.append(line)

    if header is not None:
        yield header, "".join(seq_chunks)


def normalize_sequence(seq: str, keep_case: bool) -> str:
    seq = re.sub(r"\s+", "", seq)
    return seq if keep_case else seq.upper()


def validate_sequence(seq: str) -> None:
    # Permissive genomic alphabet: canonical bases + common ambiguity codes + gap.
    if not re.fullmatch(r"[ACGTUNRYKMSWBDHVX\-\.]+", seq, flags=re.IGNORECASE):
        raise ValueError(
            "Sequence contains unsupported characters. "
            "Use --allow-non-acgtn to skip validation."
        )


def extract_label(
    header: str,
    unlabeled: bool,
    label_from_header: bool,
    label_key: Optional[str],
    header_split_delim: Optional[str],
    label_index: Optional[int],
) -> Optional[str]:
    if unlabeled:
        return None

    if label_from_header:
        return header

    if label_key is not None:
        pattern = rf"(?:^|\s){re.escape(label_key)}=([^\s]+)"
        match = re.search(pattern, header)
        if not match:
            raise ValueError(
                f"Could not find key '{label_key}' in FASTA header: {header}"
            )
        return match.group(1)

    if header_split_delim is not None and label_index is not None:
        fields = header.split(header_split_delim)
        try:
            return fields[label_index].strip()
        except IndexError as exc:
            raise ValueError(
                f"Header split produced {len(fields)} fields, but label index {label_index} "
                f"was requested. Header: {header}"
            ) from exc

    raise ValueError("No label extraction mode selected.")


def write_csv(
    records: Iterable[Record],
    output_csv: Path,
    sequence_col: str,
    label_col: str,
    unlabeled: bool,
    label_from_header: bool,
    label_key: Optional[str],
    header_split_delim: Optional[str],
    label_index: Optional[int],
    keep_case: bool,
    allow_non_acgtn: bool,
    deduplicate: bool,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_sequences: set[str] = set()

    fieldnames = [sequence_col] if unlabeled else [sequence_col, label_col]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for header, seq in records:
            seq = normalize_sequence(seq, keep_case=keep_case)
            if not seq:
                continue
            if not allow_non_acgtn:
                validate_sequence(seq)
            if deduplicate:
                if seq in seen_sequences:
                    continue
                seen_sequences.add(seq)

            label = extract_label(
                header=header,
                unlabeled=unlabeled,
                label_from_header=label_from_header,
                label_key=label_key,
                header_split_delim=header_split_delim,
                label_index=label_index,
            )

            row = {sequence_col: seq}
            if not unlabeled:
                row[label_col] = label
            writer.writerow(row)
            count += 1

    return count


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_fasta)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_path}")
    if input_path.suffix.lower() not in FASTA_SUFFIXES:
        print(
            f"[warning] Input file suffix '{input_path.suffix}' is not a common FASTA extension. "
            "Continuing anyway."
        )

    n = write_csv(
        records=read_fasta(input_path),
        output_csv=output_path,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        unlabeled=args.unlabeled,
        label_from_header=args.label_from_header,
        label_key=args.label_key,
        header_split_delim=args.header_split_delim,
        label_index=args.label_index,
        keep_case=args.keep_case,
        allow_non_acgtn=args.allow_non_acgtn,
        deduplicate=args.deduplicate,
    )

    print(f"Wrote {n} records to {output_path}")


if __name__ == "__main__":
    main()
