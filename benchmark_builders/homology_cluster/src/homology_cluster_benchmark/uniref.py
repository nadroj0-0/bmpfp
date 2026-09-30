from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
import time
from typing import Iterator

from .inputs import open_text
from .uniref_scaffold import uniref_scaffold


LOGGER = logging.getLogger(__name__)


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    parts: list[str] = []
    with open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line[1:]
                parts = []
            else:
                if header is not None:
                    parts.append(line)
        if header is not None:
            yield header, "".join(parts)


def fasta_identifier(header: str) -> str:
    return header.split(None, 1)[0]


class UniRefIndex:
    """Disk-backed UniRef scaffold identifier index."""

    def __init__(self, database: Path, uniref_level: int = 90):
        self.database = database
        self.scaffold = uniref_scaffold(uniref_level)

    @classmethod
    def build(
        cls, fasta: Path, database: Path, uniref_level: int = 90
    ) -> "UniRefIndex":
        scaffold = uniref_scaffold(uniref_level)
        started = time.monotonic()
        LOGGER.info(
            "%s index build started: %s -> %s",
            scaffold.display_name, fasta, database,
        )
        database.parent.mkdir(parents=True, exist_ok=True)
        database.unlink(missing_ok=True)
        connection = sqlite3.connect(database)
        try:
            # Disable journalling for this rebuildable scratch index.
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE uniref90 ("
                "uniref90_id TEXT PRIMARY KEY, sequence_length INTEGER NOT NULL)"
            )
            batch: list[tuple[str, int]] = []
            processed = 0
            for processed, (header, sequence) in enumerate(iter_fasta(fasta), start=1):
                identifier = fasta_identifier(header)
                batch.append((identifier, len(sequence)))
                if len(batch) >= 10000:
                    cls._insert_batch(connection, batch)
                    batch.clear()
                if processed % 1_000_000 == 0:
                    LOGGER.info(
                        "UniRef90 index progress: sequences=%d elapsed_seconds=%.1f",
                        processed, time.monotonic() - started,
                    )
            if batch:
                cls._insert_batch(connection, batch)
            count = connection.execute("SELECT COUNT(*) FROM uniref90").fetchone()[0]
            connection.commit()
        finally:
            connection.close()
        LOGGER.info(
            "%s index completed: sequences=%d elapsed_seconds=%.1f",
            scaffold.display_name, count, time.monotonic() - started,
        )
        return cls(database, uniref_level)

    @staticmethod
    def _insert_batch(connection: sqlite3.Connection, rows: list[tuple[str, int]]) -> None:
        connection.executemany("INSERT INTO uniref90 VALUES (?, ?)", rows)

    def count(self) -> int:
        with sqlite3.connect(self.database) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM uniref90").fetchone()[0])

    def contains(self, uniref90_id: str) -> bool:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT 1 FROM uniref90 WHERE uniref90_id=?", (uniref90_id,)
            ).fetchone()
        return row is not None

    def present_ids(self, uniref90_ids: set[str]) -> set[str]:
        """Return present IDs using one disk-backed join instead of one connection per ID."""
        if not uniref90_ids:
            return set()
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TEMP TABLE requested (uniref90_id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO requested VALUES (?)",
                ((identifier,) for identifier in sorted(uniref90_ids)),
            )
            return {
                str(row[0])
                for row in connection.execute(
                    "SELECT r.uniref90_id FROM requested r "
                    "JOIN uniref90 u ON u.uniref90_id=r.uniref90_id"
                )
            }

    def iter_records(self) -> Iterator[tuple[str, int]]:
        with sqlite3.connect(self.database) as connection:
            for row in connection.execute(
                "SELECT uniref90_id, sequence_length "
                "FROM uniref90 ORDER BY uniref90_id"
            ):
                yield str(row[0]), int(row[1])
