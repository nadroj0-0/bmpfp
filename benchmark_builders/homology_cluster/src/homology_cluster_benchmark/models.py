"""Scientific records used by homology benchmark construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    taxon_id: str = ""
    aliases: tuple[str, ...] = ()


@dataclass
class ProteinCatalog:
    records: dict[str, ProteinRecord] = field(default_factory=dict)
    alias_to_primary: dict[str, str] = field(default_factory=dict)
    ambiguous_aliases: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class AnnotationRecord:
    raw_accession: str
    protein_id: str
    raw_go_id: str
    go_id: str
    namespace: str
    aspect: str
    evidence: str
    qualifier: str
    taxon_id: str
    accession_action: str = "exact"


@dataclass
class GoaLoadResult:
    records: list[AnnotationRecord]
    annotations: dict[str, set[str]]
    qualifying_accessions: set[str] = field(default_factory=set)
    counters: Counter = field(default_factory=Counter)
    evidence_counts: Counter = field(default_factory=Counter)
    taxonomy_counts: Counter = field(default_factory=Counter)
    direct_go_counts: Counter = field(default_factory=Counter)
    candidate_accessions: set[str] = field(default_factory=set)
    record_spool: Path | None = None
    accession_map: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class MappingDecision:
    raw_accession: str
    protein_id: str
    accession_action: str
    uniref90_id: str = ""
    status: str = "unmapped"
    detail: str = ""
    exists_in_fasta: bool | None = None
    canonical_sequence_available: bool = False
    mmseqs_cluster_id: str = ""
    split: str = ""


@dataclass(frozen=True)
class ClusterInfo:
    cluster_id: str
    member_count: int
    labelled_protein_count: int = 0


@dataclass(frozen=True)
class SplitAssignment:
    cluster_id: str
    split: str
    member_count: int
    labelled_protein_count: int
    stage: str


@dataclass
class LabelBuildResult:
    frames: dict[str, Any]
    unrestricted_annotations: dict[str, tuple[str, ...]]
    restricted_annotations: dict[str, tuple[str, ...]]
    term_universe: tuple[str, ...]
    removed_term_counts: Counter
    no_evaluable_term: Counter
    annotation_exclusion_counts: Counter = field(default_factory=Counter)
    row_attrition_counts: Counter = field(default_factory=Counter)
    protein_attrition_counts: Counter = field(default_factory=Counter)
    intended_annotation_rows: int = 0
    intended_accessions: int = 0
    cluster_assignments: dict[str, SplitAssignment] = field(default_factory=dict)
