from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Evidence codes in the final CAFA3 release (benchmark20171115/00README.txt).
CAFA3_FINAL_EXP_CODES = frozenset({
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC",
})

# Evidence codes in CAFA_benchmark/create_benchmark.py; the final release adds TAS and IC.
CAFA3_PUBLIC_PYTHON_EXP_CODES = frozenset({
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
})

# Evidence codes used by the contemporary and homology benchmarks.
BENCHMARK_EVIDENCE_CODES = frozenset({
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP",
    "HGI", "HEP", "IGC",
})

EVIDENCE_POLICIES = {
    "cafa3-final": CAFA3_FINAL_EXP_CODES,
    "cafa3-public-python": CAFA3_PUBLIC_PYTHON_EXP_CODES,
    "benchmark-evidence": BENCHMARK_EVIDENCE_CODES,
}


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    training_taxon_policy: str
    target_taxon_policy: str
    evidence_policy: str
    training_reviewed_only: bool
    target_reviewed_only: bool
    t0_cutoff: str
    t1_cutoff: str
    test_eligibility_policy: str
    exclude_t1_backfill: bool = True
    require_t0_presence: bool = True
    sequence_change_policy: str = "exclude"
    protein_binding_policy: str = "drop-mf-protein-binding-only"


# Profiles define scientific settings; input paths come from CLI arguments.
BENCHMARK_PROFILES = {
    "cafa3-reconstructed": BenchmarkProfile(
        name="cafa3-reconstructed",
        training_taxon_policy="all",
        target_taxon_policy="cafa3-targets",
        evidence_policy="cafa3-final",
        training_reviewed_only=True,
        target_reviewed_only=False,
        t0_cutoff="20170213",
        t1_cutoff="20171115",
        test_eligibility_policy="ontology-no-knowledge",
    ),
    "contemporary-cafa3-style": BenchmarkProfile(
        name="contemporary-cafa3-style",
        training_taxon_policy="all",
        target_taxon_policy="cafa3-targets",
        evidence_policy="cafa3-final",
        training_reviewed_only=True,
        target_reviewed_only=False,
        t0_cutoff="20250308",
        t1_cutoff="20260617",
        test_eligibility_policy="ontology-no-knowledge",
    ),
    "global-nk": BenchmarkProfile(
        name="global-nk",
        training_taxon_policy="cafa3-targets",
        target_taxon_policy="cafa3-targets",
        evidence_policy="benchmark-evidence",
        training_reviewed_only=False,
        target_reviewed_only=False,
        t0_cutoff="20250308",
        t1_cutoff="20260617",
        test_eligibility_policy="global-no-knowledge",
    ),
    "nk-lk": BenchmarkProfile(
        name="nk-lk",
        training_taxon_policy="cafa3-targets",
        target_taxon_policy="cafa3-targets",
        evidence_policy="benchmark-evidence",
        training_reviewed_only=False,
        target_reviewed_only=False,
        t0_cutoff="20250308",
        t1_cutoff="20260617",
        test_eligibility_policy="ontology-no-knowledge",
    ),
}

ASPECT_TO_PREFIX = {"P": "bp", "C": "cc", "F": "mf"}

PREFIX_TO_NAMESPACE = {
    "bp": "biological_process",
    "cc": "cellular_component",
    "mf": "molecular_function",
}

NAMESPACE_TO_PREFIX = {v: k for k, v in PREFIX_TO_NAMESPACE.items()}


@dataclass(frozen=True)
class BuildConfig:
    uniprot_t0: tuple[Path, ...]
    uniprot_t1: tuple[Path, ...]
    goa_t0: Path | None
    goa_t1: Path | None
    go_obo: Path
    output_dir: Path
    go_obo_t0: Path | None = None
    go_obo_t1: Path | None = None
    target_universe_policy: str = "reconstructed-all-qualifying"
    official_target_fastas: tuple[Path, ...] = ()
    official_target_mapping_dir: Path | None = None
    target_uniprot_t0: tuple[Path, ...] = ()
    target_uniprot_t1: tuple[Path, ...] = ()
    training_annotations_file: Path | None = None
    test_annotations_file: Path | None = None
    profile_name: str = "contemporary-cafa3-style"
    training_taxa: frozenset[str] = field(default_factory=frozenset)
    target_taxa: frozenset[str] = field(default_factory=frozenset)
    evidence_codes: frozenset[str] = CAFA3_FINAL_EXP_CODES
    t0_cutoff: str | None = "20250308"
    t1_cutoff: str | None = "20260617"
    t1_endpoint_policy: str = "assigned-date-proxy"
    test_eligibility_policy: str = "ontology-no-knowledge"
    exclude_t1_backfill: bool = True
    require_t0_presence: bool = True
    sequence_change_policy: str = "exclude"
    protein_binding_policy: str = "drop-mf-protein-binding-only"
    min_count: int = 50
    split: float = 0.9
    seed: int = 0
    reviewed_only: bool = True
    target_reviewed_only: bool = False
    include_rels: bool = True
    relationship_types: frozenset[str] | None = None
    write_intermediates: bool = True
    allow_frozen_source_fallback: bool = True

    @property
    def ontology_t0(self) -> Path:
        return self.go_obo_t0 or self.go_obo

    @property
    def ontology_t1(self) -> Path:
        return self.go_obo_t1 or self.go_obo

    @property
    def reports(self) -> Path:
        return self.output_dir / "reports"


def normalise_taxa(values: Iterable[str]) -> frozenset[str]:
    taxa = set()
    for value in values:
        value = value.strip()
        if not value:
            continue
        if value.startswith("taxon:"):
            value = value.split(":", 1)[1]
        taxa.add(value)
    return frozenset(taxa)


def normalise_gaf_date(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace("-", "").strip()
