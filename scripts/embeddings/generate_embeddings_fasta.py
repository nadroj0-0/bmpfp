#!/usr/bin/env python3
"""Write the ordered prepared-protein union as a FASTA for ProtT5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ASPECTS = ("BPO", "CCO", "MFO")
SPLITS = ("train", "valid", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--line-width", type=int, default=60)
    args = parser.parse_args()

    sequences: dict[str, str] = {}
    for aspect in ASPECTS:
        for split in SPLITS:
            path = args.data_dir / f"{aspect}_{split}_sequences.json"
            for protein_id, sequence in json.loads(path.read_text()).items():
                sequences.setdefault(str(protein_id), str(sequence))

    output = args.out or args.data_dir / "proteins.fasta"
    with output.open("w", encoding="utf-8") as handle:
        for protein_id, sequence in sequences.items():
            handle.write(f">{protein_id}\n")
            if args.line_width:
                for start in range(0, len(sequence), args.line_width):
                    handle.write(sequence[start : start + args.line_width] + "\n")
            else:
                handle.write(sequence + "\n")

    print(f"Wrote {len(sequences)} proteins to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
