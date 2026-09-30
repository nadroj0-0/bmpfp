#!/usr/bin/env python3
"""Prepare benchmark CSVs for the upstream PFP data layout."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


CSV_NAMES = tuple(
    f"{aspect}-{split}.csv"
    for aspect in ("bp", "cc", "mf")
    for split in ("training", "validation", "test")
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def copy_with_pfp_headers(source: Path, destination: Path) -> None:
    """Copy the nine benchmark CSVs, adapting only ``protein`` to ``proteins``."""

    destination.mkdir(parents=True, exist_ok=True)
    for name in CSV_NAMES:
        source_path = source / name
        destination_path = destination / name
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            first_line = handle.readline()
            remainder = handle.read()
        if first_line.startswith("protein,"):
            first_line = "proteins," + first_line[len("protein,") :]
        with destination_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(first_line)
            handle.write(remainder)


def load_upstream_preparer(path: Path):
    specification = importlib.util.spec_from_file_location("pfp_prepare_data", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.CAFA3DatasetPreparer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pfp-root", type=Path, required=True)
    args = parser.parse_args()

    upstream_path = args.pfp_root / "scripts" / "prepare_cafa3_data.py"
    preparer_type = load_upstream_preparer(upstream_path)

    with tempfile.TemporaryDirectory(prefix="pfp-benchmark-headers-") as temporary:
        adapted = Path(temporary) / "benchmark"
        copy_with_pfp_headers(args.benchmark_dir, adapted)
        preparer = preparer_type(
            cafa3_dir=str(adapted),
            output_dir=str(args.data_dir),
            small_subset=False,
        )
        # Split membership is defined by the benchmark builders.
        preparer._check_data_leakage = lambda *unused: None
        preparer.prepare_dataset()

    subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "scripts"
                / "embeddings"
                / "generate_embeddings_fasta.py"
            ),
            "--data-dir",
            str(args.data_dir),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
