from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import pandas as pd

from .config import SPLITS
from .goa import iter_annotation_records
from .models import (
    GoaLoadResult,
    LabelBuildResult,
    MappingDecision,
    ProteinCatalog,
    SplitAssignment,
)
from .ontology import Ontology


def _protein_clusters(decisions: list[MappingDecision]) -> dict[str, str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for decision in decisions:
        if decision.status == "mapped" and decision.mmseqs_cluster_id:
            grouped[decision.protein_id].add(decision.mmseqs_cluster_id)
    return {protein: sorted(values)[0] for protein, values in grouped.items()}


def build_labels(
    ontology: Ontology,
    goa: GoaLoadResult,
    catalog: ProteinCatalog,
    mappings: list[MappingDecision],
    cluster_splits: dict[str, SplitAssignment],
    min_count: int,
    finalize_development: Callable[[], dict[str, SplitAssignment]] | None = None,
) -> LabelBuildResult:
    protein_clusters = _protein_clusters(mappings)
    decisions_by_raw = {decision.raw_accession: decision for decision in mappings}
    accepted_direct: dict[str, set[str]] = defaultdict(set)
    accepted_raw_accessions: set[str] = set()
    ontology_accepted_raw_accessions: set[str] = set()
    annotation_exclusions = Counter()
    row_attrition = Counter()
    intended_annotation_rows = 0
    for record in iter_annotation_records(goa):
        intended_annotation_rows += 1
        ontology_accepted_raw_accessions.add(record.raw_accession)
        decision = decisions_by_raw.get(record.raw_accession)
        if decision is None:
            disposition = "missing_mapping_decision"
        elif decision.status != "mapped":
            disposition = f"mapping_status:{decision.status}"
        elif decision.mmseqs_cluster_id not in cluster_splits:
            disposition = "cluster_not_retained_or_split"
        elif decision.protein_id != record.protein_id:
            disposition = "canonical_protein_mismatch"
        else:
            disposition = "eligible_annotation_row"
            accepted_direct[record.protein_id].add(record.go_id)
            accepted_raw_accessions.add(record.raw_accession)
        row_attrition[disposition] += 1
        if disposition != "eligible_annotation_row":
            annotation_exclusions[disposition] += 1
    rows: dict[str, list[dict[str, object]]] = {
        split: [] for split in (*SPLITS, "development")
    }
    unrestricted: dict[str, tuple[str, ...]] = {}
    no_evaluable = Counter()

    terminal_failure_by_protein: dict[str, str] = {}
    for protein_id in sorted(accepted_direct):
        cluster_id = protein_clusters.get(protein_id)
        if not cluster_id:
            no_evaluable["incomplete_mapping_chain"] += 1
            terminal_failure_by_protein[protein_id] = "incomplete_mapping_chain"
            continue
        assignment = cluster_splits.get(cluster_id)
        if assignment is None:
            no_evaluable["cluster_not_retained_or_split"] += 1
            terminal_failure_by_protein[protein_id] = "cluster_not_retained_or_split"
            continue
        protein = catalog.records.get(protein_id)
        if protein is None:
            no_evaluable["missing_canonical_sequence"] += 1
            terminal_failure_by_protein[protein_id] = "missing_canonical_sequence"
            continue
        propagated: set[str] = set()
        for direct_term in sorted(accepted_direct[protein_id]):
            propagated.update(ontology.ancestors(direct_term, exclude_roots=False))
        if not propagated:
            no_evaluable["unresolvable_after_propagation"] += 1
            terminal_failure_by_protein[protein_id] = "unresolvable_after_propagation"
            continue
        annotation_tuple = tuple(sorted(propagated))
        unrestricted[protein_id] = annotation_tuple
        rows[assignment.split].append({
            "proteins": protein_id,
            "sequences": protein.sequence,
            "annotations": annotation_tuple,
            "cluster_id": cluster_id,
        })

    # Use development, or training plus validation, to define the term universe.
    development_counts = Counter()
    development_source_splits = (
        ("development",)
        if any(item.split == "development" for item in cluster_splits.values())
        else ("training", "validation")
    )
    for split in development_source_splits:
        for row in rows[split]:
            development_counts.update(set(row["annotations"]))
    term_universe = tuple(sorted(
        term for term, count in development_counts.items() if count >= min_count
    ))
    universe = set(term_universe)

    if development_source_splits == ("development",):
        final_assignments = finalize_development()
        for row in rows["development"]:
            final_split = final_assignments[str(row["cluster_id"])].split
            rows[final_split].append(row)
        rows["development"].clear()
        cluster_splits = final_assignments

    frames: dict[str, pd.DataFrame] = {}
    restricted: dict[str, tuple[str, ...]] = {}
    removed = Counter()
    for split in SPLITS:
        restricted_rows = []
        for row in sorted(rows[split], key=lambda item: str(item["proteins"])):
            annotation_tuple = row["annotations"]
            original = set(annotation_tuple)
            kept = tuple(sorted(original & universe))
            removed[split] += len(original - universe)
            if not kept:
                no_evaluable[f"{split}_no_development_universe_term"] += 1
            restricted[str(row["proteins"])] = kept
            restricted_rows.append({
                "proteins": row["proteins"],
                "sequences": row["sequences"],
                # Pickles keep complete propagated labels; CSV export applies the term universe.
                "annotations": annotation_tuple,
                "cluster_id": row["cluster_id"],
            })
        frames[split] = pd.DataFrame(
            restricted_rows,
            columns=["proteins", "sequences", "annotations", "cluster_id"],
        )
        rows[split].clear()

    # Count one terminal outcome per raw GOA accession, preserving secondary-accession mappings.
    candidates = set(goa.candidate_accessions) or set(goa.qualifying_accessions)
    candidates.update(accepted_raw_accessions)
    protein_attrition = Counter()
    for raw_accession in sorted(candidates):
        if raw_accession not in ontology_accepted_raw_accessions:
            outcome = "ontology_resolution_or_namespace_rejection"
        else:
            decision = decisions_by_raw.get(raw_accession)
            if decision is None:
                outcome = "missing_mapping_decision"
            elif decision.status != "mapped":
                outcome = f"mapping_status:{decision.status}"
            elif decision.mmseqs_cluster_id not in cluster_splits:
                outcome = "cluster_not_retained_or_split"
            elif decision.protein_id in terminal_failure_by_protein:
                outcome = terminal_failure_by_protein[decision.protein_id]
            elif decision.protein_id not in restricted:
                outcome = "missing_label_row"
            elif restricted[decision.protein_id]:
                outcome = "evaluable_pfp"
            else:
                outcome = "no_development_universe_term"
        protein_attrition[outcome] += 1

    # Clear transient maps after terminal outcomes are assigned.
    accepted_direct.clear()
    decisions_by_raw.clear()
    protein_clusters.clear()

    return LabelBuildResult(
        frames=frames,
        unrestricted_annotations=unrestricted,
        restricted_annotations=restricted,
        term_universe=term_universe,
        removed_term_counts=removed,
        no_evaluable_term=no_evaluable,
        annotation_exclusion_counts=annotation_exclusions,
        row_attrition_counts=row_attrition,
        protein_attrition_counts=protein_attrition,
        intended_annotation_rows=intended_annotation_rows,
        intended_accessions=len(candidates),
        cluster_assignments=cluster_splits,
    )
