"""Disk-backed access to explicit representative/member assignments."""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
import time
from typing import Iterable, Iterator

from .inputs import open_text
from .uniref import UniRefIndex


LOGGER = logging.getLogger(__name__)


class ClusterIndex:
    """Disk-backed representative/member assignment table."""

    def __init__(self, database: Path):
        self.database = database

    @classmethod
    def build(
        cls,
        cluster_tsv: Path,
        uniref: UniRefIndex,
        database: Path,
        *,
        has_header: bool = False,
        member_field: str = "uniref90_id",
    ) -> "ClusterIndex":
        del uniref, member_field
        started = time.monotonic()
        database.parent.mkdir(parents=True, exist_ok=True)
        database.unlink(missing_ok=True)
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE assignments ("
                "member_id TEXT PRIMARY KEY, cluster_id TEXT NOT NULL)"
            )
            batch: list[tuple[str, str]] = []
            first_content = True
            with open_text(cluster_tsv) as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    columns = raw_line.rstrip("\n\r").split("\t")
                    if has_header and first_content:
                        first_content = False
                        continue
                    first_content = False
                    representative, member = columns
                    batch.append((member, representative))
                    if len(batch) >= 10000:
                        cls._insert(connection, batch)
                        batch.clear()
            if batch:
                cls._insert(connection, batch)
            connection.commit()
            connection.execute(
                "CREATE INDEX assignments_cluster_member_idx "
                "ON assignments(cluster_id, member_id)"
            )
        finally:
            connection.close()
        LOGGER.info(
            "MMseqs assignment index built in %.1f seconds",
            time.monotonic() - started,
        )
        return cls(database)

    @staticmethod
    def _insert(connection: sqlite3.Connection, rows: list[tuple[str, str]]) -> None:
        connection.executemany("INSERT INTO assignments VALUES (?, ?)", rows)

    def clusters_for(self, member_ids: set[str]) -> dict[str, str]:
        if not member_ids:
            return {}
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TEMP TABLE requested (member_id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO requested VALUES (?)",
                ((member_id,) for member_id in sorted(member_ids)),
            )
            return {
                str(member): str(cluster)
                for member, cluster in connection.execute(
                    "SELECT r.member_id, a.cluster_id FROM requested r "
                    "JOIN assignments a ON a.member_id=r.member_id"
                )
            }

    def sizes_for(self, cluster_ids: set[str]) -> dict[str, int]:
        if not cluster_ids:
            return {}
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TEMP TABLE retained (cluster_id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO retained VALUES (?)",
                ((value,) for value in sorted(cluster_ids)),
            )
            return {
                str(cluster): int(count)
                for cluster, count in connection.execute(
                    "SELECT a.cluster_id, COUNT(*) FROM assignments a "
                    "JOIN retained r ON r.cluster_id=a.cluster_id "
                    "GROUP BY a.cluster_id ORDER BY a.cluster_id"
                )
            }

    def iter_assignments_for_clusters(
        self, cluster_ids: Iterable[str]
    ) -> Iterator[tuple[str, str]]:
        ordered = sorted(cluster_ids)
        if not ordered:
            return
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TEMP TABLE retained (cluster_id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO retained VALUES (?)", ((value,) for value in ordered)
            )
            for cluster, member in connection.execute(
                "SELECT a.cluster_id, a.member_id FROM retained r "
                "JOIN assignments a ON a.cluster_id=r.cluster_id "
                "ORDER BY a.cluster_id, a.member_id"
            ):
                yield str(cluster), str(member)

    def iter_assignments_with_metadata_for_clusters(
        self, uniref: UniRefIndex, cluster_ids: Iterable[str]
    ) -> Iterator[tuple[str, str, int]]:
        ordered = sorted(cluster_ids)
        if not ordered:
            return
        with sqlite3.connect(self.database) as connection:
            connection.execute("ATTACH DATABASE ? AS u", (str(uniref.database),))
            connection.execute("CREATE TEMP TABLE retained (cluster_id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO retained VALUES (?)", ((value,) for value in ordered)
            )
            for cluster, member, length in connection.execute(
                "SELECT a.cluster_id, a.member_id, u.uniref90.sequence_length "
                "FROM retained r "
                "JOIN assignments a ON a.cluster_id=r.cluster_id "
                "JOIN u.uniref90 ON a.member_id=u.uniref90.uniref90_id "
                "ORDER BY a.cluster_id, a.member_id"
            ):
                yield str(cluster), str(member), int(length)
