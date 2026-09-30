"""Plain and gzip-compressed text input helpers."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator, TextIO


def open_text(path: str | Path) -> TextIO:
    source = Path(path)
    if source.suffix == ".gz":
        return gzip.open(source, "rt", encoding="utf-8", errors="strict", newline="")
    return source.open("r", encoding="utf-8", errors="strict", newline="")


def iter_lines(path: str | Path) -> Iterator[str]:
    with open_text(path) as handle:
        yield from handle
