from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator, TextIO


def open_text(path: str | Path) -> TextIO:
    """Open plain text or gzip-compressed text by suffix."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def read_lines(path: str | Path) -> Iterator[str]:
    with open_text(path) as handle:
        for line in handle:
            yield line
