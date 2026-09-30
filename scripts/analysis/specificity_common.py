#!/usr/bin/env python3
"""Shared value loading and deterministic binning for GO specificity measures."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from label_space_common import (
    ASPECT_TO_NAMESPACE,
    ASPECT_TO_ROOT,
)


XU_RELATIONSHIPS = ("is_a", "part_of")


@dataclass(frozen=True)
class SpecificityMeasure:
    name: str
    values: np.ndarray
    higher_is_more_specific: bool
    source: dict[str, Any]
    zero_bin_label: str | None = None


@dataclass(frozen=True)
class XuOntology:
    namespaces: Mapping[str, str]
    parents: Mapping[str, tuple[str, ...]]
    alt_ids: Mapping[str, str]
    relationship_counts: Mapping[str, int]
    excluded_cross_namespace_relationship_counts: Mapping[str, int]
    source: Mapping[str, Any]


def read_xu_ontology(
    path: Path, relationships: Sequence[str] = XU_RELATIONSHIPS
) -> XuOntology:
    """Read the active GO DAG required by Xu et al.'s topology-only measure."""
    relationship_set = tuple(dict.fromkeys(str(value) for value in relationships))
    unknown = sorted(set(relationship_set) - set(XU_RELATIONSHIPS))
    if unknown:
        raise ValueError(
            "Paper-faithful Xu totipotency only supports is_a and part_of; "
            f"found {unknown}"
        )

    path = path.expanduser().resolve()
    stanzas: list[dict[str, Any]] = []
    stanza: dict[str, Any] = {}
    def publish() -> None:
        if stanza.get("type") == "Term" and stanza.get("id"):
            stanzas.append(dict(stanza))

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line == "[Term]":
                publish()
                stanza = {
                    "type": "Term",
                    "is_a": [],
                    "relationships": [],
                    "alt_ids": [],
                }
            elif line.startswith("["):
                publish()
                stanza = {}
            elif stanza.get("type") == "Term":
                if line.startswith("id: "):
                    stanza["id"] = line[4:].strip()
                elif line.startswith("namespace: "):
                    stanza["namespace"] = line[11:].strip()
                elif line.startswith("is_obsolete: "):
                    stanza["is_obsolete"] = line[13:].strip() == "true"
                elif line.startswith("alt_id: "):
                    stanza["alt_ids"].append(line[8:].strip())
                elif line.startswith("is_a: "):
                    stanza["is_a"].append(line[6:].split()[0])
                elif line.startswith("relationship: "):
                    fields = line.split()
                    stanza["relationships"].append((fields[1], fields[2]))
        publish()
    active = {
        str(value["id"]): value
        for value in stanzas
        if not value.get("is_obsolete") and value.get("namespace")
    }
    namespaces = {term_id: str(value["namespace"]) for term_id, value in active.items()}
    alt_ids: dict[str, str] = {}
    for term_id, value in active.items():
        for alt_id in value["alt_ids"]:
            alt_ids[alt_id] = term_id

    counts: Counter[str] = Counter()
    excluded_cross_namespace_counts: Counter[str] = Counter()
    parents: dict[str, tuple[str, ...]] = {}
    for term_id, value in active.items():
        selected: list[tuple[str, str]] = []
        if "is_a" in relationship_set:
            selected.extend(("is_a", parent) for parent in value["is_a"])
        selected.extend(
            (relation, parent)
            for relation, parent in value["relationships"]
            if relation in relationship_set
        )
        normalized: set[str] = set()
        for relation, raw_parent in selected:
            parent = alt_ids.get(raw_parent, raw_parent)
            if parent not in active:
                continue
            if namespaces[parent] != namespaces[term_id]:
                if relation == "part_of":
                    excluded_cross_namespace_counts[relation] += 1
                    continue
                continue
            normalized.add(parent)
            counts[relation] += 1
        parents[term_id] = tuple(sorted(normalized))

    return XuOntology(
        namespaces=namespaces,
        parents=parents,
        alt_ids=alt_ids,
        relationship_counts=dict(sorted(counts.items())),
        excluded_cross_namespace_relationship_counts=dict(
            sorted(excluded_cross_namespace_counts.items())
        ),
        source={
            "relationships": list(relationship_set),
            "relationship_counts": dict(sorted(counts.items())),
            "excluded_cross_namespace_relationship_counts": dict(
                sorted(excluded_cross_namespace_counts.items())
            ),
            "active_terms": len(active),
            "alt_ids": len(alt_ids),
        },
    )


def compute_xu_totipotency(
    ontology: XuOntology,
    go_terms: Sequence[str],
    aspect: str,
) -> tuple[SpecificityMeasure, SpecificityMeasure, list[dict[str, Any]]]:
    """Compute Xu T and the explicitly exploratory -log2(T) transform."""
    root = ASPECT_TO_ROOT[aspect]
    namespace = ASPECT_TO_NAMESPACE[aspect]
    children: dict[str, set[str]] = {
        term_id: set()
        for term_id, value in ontology.namespaces.items()
        if value == namespace
    }
    for child, parents in ontology.parents.items():
        if child not in children:
            continue
        for parent in parents:
            if parent in children:
                children[parent].add(child)

    memo: dict[str, frozenset[str]] = {}

    def descendants(term_id: str) -> frozenset[str]:
        if term_id in memo:
            return memo[term_id]
        values = {term_id}
        for child in sorted(children.get(term_id, ())):
            values.update(descendants(child))
        memo[term_id] = frozenset(values)
        return memo[term_id]

    root_descendants = descendants(root)
    root_count = len(root_descendants)

    raw_values: list[float] = []
    neglog_values: list[float] = []
    rows: list[dict[str, Any]] = []
    for supplied_term in go_terms:
        canonical = ontology.alt_ids.get(supplied_term, supplied_term)
        descendant_count = len(descendants(canonical))
        value = descendant_count / root_count
        neglog = -math.log2(value)
        raw_values.append(value)
        neglog_values.append(neglog)
        rows.append(
            {
                "go_id": supplied_term,
                "canonical_go_id": canonical,
                "mapping_status": (
                    "alt_id" if canonical != supplied_term else "active_primary"
                ),
                "descendant_count": descendant_count,
                "aspect_root_descendant_count": root_count,
                "xu_totipotency_T": value,
                "xu_neglog_totipotency": neglog,
                "root": canonical == root,
            }
        )

    raw = SpecificityMeasure(
        name="xu_totipotency_raw",
        values=np.asarray(raw_values, dtype=np.float64),
        higher_is_more_specific=False,
        zero_bin_label=None,
        source=dict(ontology.source),
    )
    neglog = SpecificityMeasure(
        name="xu_neglog_totipotency",
        values=np.asarray(neglog_values, dtype=np.float64),
        higher_is_more_specific=True,
        zero_bin_label="zero_xu_neglog",
        source={
            **dict(ontology.source),
            "transform": "-log2(xu_totipotency_T)",
            "transform_status": "exploratory_not_proposed_by_xu",
        },
    )
    return raw, neglog, rows


def read_nonnegative_term_values(
    path: Path,
    go_terms: Sequence[str],
    *,
    measure_name: str,
    higher_is_more_specific: bool,
    zero_bin_label: str | None,
) -> SpecificityMeasure:
    path = path.expanduser().resolve()
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            term = fields[0]
            values[term] = float(fields[1])
    ordered = np.asarray([values[term] for term in go_terms], dtype=np.float64)
    measure = SpecificityMeasure(
        name=measure_name,
        values=ordered,
        higher_is_more_specific=higher_is_more_specific,
        zero_bin_label=zero_bin_label,
        source={
            "prediction_terms": len(go_terms),
            "source_terms": len(values),
            "extra_source_terms": len(set(values) - set(go_terms)),
            "zero_terms": int(np.count_nonzero(ordered == 0)),
            "positive_terms": int(np.count_nonzero(ordered > 0)),
        },
    )
    return measure


def assign_specificity_bins(
    go_terms: Sequence[str],
    measure: SpecificityMeasure,
    bin_count: int,
    *,
    bin_prefix: str,
    excluded_indices: Sequence[int] = (),
    excluded_label: str = "excluded",
) -> tuple[list[dict[str, Any]], list[str]]:
    excluded_mask = np.zeros(len(go_terms), dtype=bool)
    for index in excluded_indices:
        excluded_mask[index] = True
    zero_mask = (
        (measure.values == 0) & ~excluded_mask
        if measure.zero_bin_label
        else np.zeros(len(go_terms), dtype=bool)
    )
    eligible = ~zero_mask & ~excluded_mask
    ranking = measure.values if measure.higher_is_more_specific else -measure.values
    ranked_values = ranking[eligible]
    inner_edges = (
        np.quantile(
            ranked_values,
            np.linspace(0, 1, bin_count + 1)[1:-1],
            method="linear",
        )
        if ranked_values.size and bin_count > 1
        else np.asarray([], dtype=np.float64)
    )
    assignments: list[str] = []
    for index, rank_value in enumerate(ranking):
        if excluded_mask[index]:
            assignments.append(excluded_label)
        elif zero_mask[index]:
            assignments.append(str(measure.zero_bin_label))
        else:
            bin_index = int(np.searchsorted(inner_edges, rank_value, side="left")) + 1
            assignments.append(f"{bin_prefix}{bin_index}")

    labels = (
        [measure.zero_bin_label] if measure.zero_bin_label is not None else []
    ) + [f"{bin_prefix}{index}" for index in range(1, bin_count + 1)]
    bins: list[dict[str, Any]] = []
    for label in labels:
        indices = [index for index, value in enumerate(assignments) if value == label]
        selected = measure.values[indices]
        bins.append(
            {
                "label": label,
                "term_indices": indices,
                "term_count": len(indices),
                "value_min": float(selected.min()) if selected.size else None,
                "value_max": float(selected.max()) if selected.size else None,
                "value_mean": float(selected.mean()) if selected.size else None,
                "higher_is_more_specific": measure.higher_is_more_specific,
            }
        )
    return bins, assignments
