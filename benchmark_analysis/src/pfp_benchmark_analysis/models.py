"""Internal records for descriptive benchmark analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Observation:
    dataset_id: str
    aspect: str
    split: str
    protein_id: str
    sequence_length: int
    label_count: int
    non_root_label_count: int
    root_positive: bool
    root_only: bool
    all_zero: bool
    positive_terms: Tuple[str, ...]


@dataclass(frozen=True)
class SourceRecord:
    protein_id: str
    sequence: str
    annotations: Tuple[str, ...]


@dataclass(frozen=True)
class Taxon:
    taxon_id: str
    taxon_name: str
    source: str
    source_name: str
    source_priority: int
    accession_role: str = "direct"
    record_primary_accession: str = ""
    sequence_matches_benchmark: Optional[bool] = None
    resolution_basis: str = "direct identifier mapping"


@dataclass(frozen=True)
class TaxonomyConflict:
    protein_id: str
    selected: Optional[Taxon]
    alternative: Taxon
    resolution: str
    status: str = "resolved"


@dataclass
class DatasetResult:
    dataset_id: str
    config: object
    observations: Tuple[Observation, ...]
    file_profiles: Tuple[dict, ...]
    term_headers: Mapping[str, Tuple[str, ...]]
    sequences: Mapping[str, str]
    source_by_split: Mapping[str, Mapping[str, SourceRecord]]
    taxonomy: Mapping[str, Taxon]
    taxonomy_conflicts: Tuple[TaxonomyConflict, ...]
    modality_states: Mapping[Tuple[str, str], Mapping[str, bool]]
    modalities: Tuple[str, ...]
    category_maps: Mapping[str, Mapping[str, Tuple[Tuple[str, str], ...]]]
    diagnostics: Dict[str, object]


@dataclass(frozen=True)
class AnalysisBundle:
    summary: dict
    tables: Mapping[str, Tuple[dict, ...]]
