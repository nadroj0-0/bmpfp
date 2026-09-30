"""Command-line entry point for descriptive benchmark analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyse
from .config import load_config
from .reports import write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Characterise PFP benchmark labels, root-only sources, taxa, "
            "optional protein categories, and modality coverage."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    bundle = analyse(config)
    write_reports(args.output_dir, bundle, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
