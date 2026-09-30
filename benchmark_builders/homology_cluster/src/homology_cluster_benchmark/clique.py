from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
from pathlib import Path
import sqlite3
import time

from .inputs import open_text
from .mmseqs import ClusterIndex


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CliqueStats:
    raw_rows: int
    duplicate_rows: int
    members: int
    clusters: int
    uniparc_members: int

    def as_dict(self) -> dict[str, int]:
        return {
            "raw_rows": self.raw_rows,
            "duplicate_rows": self.duplicate_rows,
            "members": self.members,
            "clusters": self.clusters,
            "uniparc_members": self.uniparc_members,
        }


def build_clique_index(
    memberships_csv: Path, database: Path
) -> tuple[ClusterIndex, CliqueStats]:
    """Load an externally supplied unordered clique-family CSV."""
    source = memberships_csv.expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    database.unlink(missing_ok=True)
    started = time.monotonic()
    connection = sqlite3.connect(database)
    raw_rows = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE assignments ("
            "member_id TEXT PRIMARY KEY, cluster_id TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE representatives ("
            "cluster_id TEXT PRIMARY KEY, representative_id TEXT NOT NULL)"
        )
        assignment_sql = (
            "INSERT OR IGNORE INTO assignments(member_id, cluster_id) VALUES (?, ?)"
        )
        representative_sql = (
            "INSERT OR IGNORE INTO representatives(cluster_id, representative_id) VALUES (?, ?)"
        )
        assignment_batch: list[tuple[str, str]] = []
        representative_batch: list[tuple[str, str]] = []
        with open_text(source) as handle:
            for columns in csv.reader(handle):
                if not columns or all(not value.strip() for value in columns):
                    continue
                cluster_id, representative_id, member_id = (
                    value.strip() for value in columns
                )
                raw_rows += 1
                assignment_batch.append((member_id, cluster_id))
                representative_batch.append((cluster_id, representative_id))
                if len(assignment_batch) >= 10_000:
                    connection.executemany(assignment_sql, assignment_batch)
                    connection.executemany(representative_sql, representative_batch)
                    assignment_batch.clear()
                    representative_batch.clear()
                if raw_rows % 1_000_000 == 0:
                    LOGGER.info(
                        "Clique index progress: rows=%d elapsed_seconds=%.1f",
                        raw_rows,
                        time.monotonic() - started,
                    )
        if assignment_batch:
            connection.executemany(assignment_sql, assignment_batch)
            connection.executemany(representative_sql, representative_batch)
        connection.execute(
            "CREATE INDEX assignments_cluster_member_idx "
            "ON assignments(cluster_id, member_id)"
        )
        members = int(connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0])
        clusters = int(
            connection.execute("SELECT COUNT(*) FROM representatives").fetchone()[0]
        )
        uniparc_members = int(
            connection.execute(
                "SELECT COUNT(*) FROM assignments WHERE member_id LIKE 'UPI%'"
            ).fetchone()[0]
        )
        connection.commit()
    finally:
        connection.close()
    stats = CliqueStats(
        raw_rows=raw_rows,
        duplicate_rows=raw_rows - members,
        members=members,
        clusters=clusters,
        uniparc_members=uniparc_members,
    )
    LOGGER.info(
        "Clique index completed: members=%d clusters=%d duplicates=%d elapsed_seconds=%.1f",
        members,
        clusters,
        stats.duplicate_rows,
        time.monotonic() - started,
    )
    return ClusterIndex(database), stats
