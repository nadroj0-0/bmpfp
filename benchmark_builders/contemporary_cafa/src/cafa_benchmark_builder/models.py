from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    taxon_id: str | None = None
    reviewed: bool | None = None
    entry_name: str | None = None
    accessions: tuple[str, ...] = ()


@dataclass
class ProteinCatalog:
    records: dict[str, ProteinRecord] = field(default_factory=dict)
    alias_to_primary: dict[str, str] = field(default_factory=dict)
    ambiguous_aliases: set[str] = field(default_factory=set)

    @property
    def sequences(self) -> dict[str, str]:
        return {protein_id: record.sequence for protein_id, record in self.records.items()}

    @property
    def taxa(self) -> dict[str, str | None]:
        return {protein_id: record.taxon_id for protein_id, record in self.records.items()}


@dataclass(frozen=True)
class IdentityMatch:
    t0_id: str
    t1_id: str | None
    status: str
    reason: str
    sequence_changed: bool = False


@dataclass
class AnnotationLoadResult:
    annotations: dict[str, set[str]]
    counters: Counter = field(default_factory=Counter)
    evidence_counts: Counter = field(default_factory=Counter)
    taxon_counts: Counter = field(default_factory=Counter)
    unmapped_terms: Counter = field(default_factory=Counter)
    out_of_benchmark_terms: Counter = field(default_factory=Counter)
    date_filter_counts: dict[tuple[str, str], Counter] = field(default_factory=dict)
