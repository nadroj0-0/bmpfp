from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .models import AnnotationLoadResult
from .ontology import Ontology


def load_released_training_annotations(
    path: Path,
    alias_to_primary: dict[str, str],
    benchmark_ontology: Ontology,
) -> AnnotationLoadResult:
    """Load CAFA accession, GO and aspect direct labels."""
    annotations: dict[str, set[str]] = defaultdict(set)
    counters = Counter()
    unmapped = Counter()
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            counters["processed"] += 1
            protein_id = alias_to_primary.get(fields[0])
            if protein_id is None:
                counters["skipped_outside_sequences"] += 1
                continue
            term = benchmark_ontology.resolve_term(fields[1])
            if term is None:
                counters["unmapped_source_go"] += 1
                unmapped[fields[1]] += 1
                continue
            annotations[protein_id].add(term)
            counters["kept_rows"] += 1
    return AnnotationLoadResult(
        annotations={protein_id: set(terms) for protein_id, terms in annotations.items()},
        counters=counters,
        unmapped_terms=unmapped,
    )
