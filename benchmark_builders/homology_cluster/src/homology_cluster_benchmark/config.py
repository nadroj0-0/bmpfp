"""Scientific constants shared by the homology benchmark builder."""

from __future__ import annotations


SUPPORTED_IDENTITIES = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)

BENCHMARK_EVIDENCE_CODES = frozenset({
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP",
    "HGI", "HEP", "IGC",
})

ROOT_TERMS = frozenset({"GO:0008150", "GO:0005575", "GO:0003674"})
ASPECT_TO_NAMESPACE = {
    "P": "biological_process",
    "C": "cellular_component",
    "F": "molecular_function",
}
PREFIX_TO_NAMESPACE = {
    "bp": "biological_process",
    "cc": "cellular_component",
    "mf": "molecular_function",
}
SPLITS = ("training", "validation", "test")


def parse_identity(value: str | int | float) -> float:
    """Return an MMseqs identity fraction from a percentage or fraction."""

    numeric = float(value)
    return numeric / 100.0 if numeric > 1 else numeric
