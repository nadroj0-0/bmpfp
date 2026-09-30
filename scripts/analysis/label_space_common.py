#!/usr/bin/env python3
"""Shared helpers for PFP label-space analyses."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ASPECTS = ("BPO", "CCO", "MFO")
ASPECT_TO_PREFIX = {"BPO": "bp", "CCO": "cc", "MFO": "mf"}
ASPECT_TO_NAMESPACE = {
    "BPO": "biological_process",
    "CCO": "cellular_component",
    "MFO": "molecular_function",
}
ASPECT_TO_ROOT = {
    "BPO": "GO:0008150",
    "CCO": "GO:0005575",
    "MFO": "GO:0003674",
}
CSV_SPLITS = ("training", "validation", "test")
def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def required_csv_names() -> list[str]:
    return [
        f"{ASPECT_TO_PREFIX[aspect]}-{split}.csv"
        for aspect in ASPECTS
        for split in CSV_SPLITS
    ]


def quantile(values: Sequence[int | float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: Sequence[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": sum(float(value) for value in values) / len(values),
        "median": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "p99": quantile(values, 0.99),
        "maximum": float(max(values)),
    }


@dataclass(frozen=True)
class OboTerm:
    namespace: str
    parents: tuple[str, ...]


@dataclass
class OboGraph:
    terms: dict[str, OboTerm]

    def depths(self, aspect: str) -> tuple[dict[str, int], dict[str, int]]:
        namespace = ASPECT_TO_NAMESPACE[aspect]
        root = ASPECT_TO_ROOT[aspect]
        shortest: dict[str, int] = {root: 0}
        longest: dict[str, int] = {root: 0}
        def visit(term_id: str) -> tuple[int, int] | None:
            if term_id in shortest:
                return shortest[term_id], longest[term_id]
            term = self.terms.get(term_id)
            if term is None or term.namespace != namespace:
                return None
            candidates = []
            for parent in term.parents:
                result = visit(parent)
                if result is not None:
                    candidates.append(result)
            if not candidates:
                return None
            shortest[term_id] = 1 + min(value[0] for value in candidates)
            longest[term_id] = 1 + max(value[1] for value in candidates)
            return shortest[term_id], longest[term_id]

        for term_id in self.terms:
            visit(term_id)
        return shortest, longest

    def ancestor_closure(self, aspect: str) -> dict[str, frozenset[str]]:
        namespace = ASPECT_TO_NAMESPACE[aspect]
        root = ASPECT_TO_ROOT[aspect]
        memo: dict[str, frozenset[str]] = {root: frozenset({root})}
        def visit(term_id: str) -> frozenset[str]:
            if term_id in memo:
                return memo[term_id]
            term = self.terms.get(term_id)
            if term is None or term.namespace != namespace:
                return frozenset()
            values = {term_id}
            for parent in term.parents:
                parent_term = self.terms.get(parent)
                if parent_term is not None and parent_term.namespace == namespace:
                    values.update(visit(parent))
            memo[term_id] = frozenset(values)
            return memo[term_id]

        for term_id, term in self.terms.items():
            if term.namespace == namespace:
                visit(term_id)
        return memo


def read_obo(path: Path) -> OboGraph:
    terms: dict[str, OboTerm] = {}
    stanza: dict[str, Any] = {}

    def publish() -> None:
        term_id = stanza.get("id")
        namespace = stanza.get("namespace")
        if (
            stanza.get("type") == "Term"
            and term_id
            and namespace
            and stanza.get("is_obsolete") != "true"
        ):
            terms[str(term_id)] = OboTerm(
                namespace=str(namespace),
                parents=tuple(sorted(set(stanza.get("parents", [])))),
            )

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                publish()
                stanza = {"type": "Term", "parents": []}
            elif line.startswith("["):
                publish()
                stanza = {}
            elif stanza.get("type") == "Term":
                if line.startswith("id: "):
                    stanza["id"] = line[4:].strip()
                elif line.startswith("namespace: "):
                    stanza["namespace"] = line[11:].strip()
                elif line.startswith("is_obsolete: "):
                    stanza["is_obsolete"] = line[13:].strip()
                elif line.startswith("is_a: "):
                    stanza["parents"].append(line[6:].split()[0])
                elif line.startswith("relationship: part_of "):
                    stanza["parents"].append(line[22:].split()[0])
        publish()
    return OboGraph(terms)


def read_ia_file(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            fields = line.split("\t")
            values[fields[0]] = float(fields[1])
    return values


def analyse_csv(
    path: Path,
    aspect: str,
    split: str,
    graph: OboGraph,
    ia_values: Mapping[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    root = ASPECT_TO_ROOT[aspect]
    shortest, longest = graph.depths(aspect)
    term_support: Counter[str] = Counter()
    root_only_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    label_counts: list[int] = []
    non_root_counts: list[int] = []
    row_max_shortest_depths: list[int] = []
    row_max_longest_depths: list[int] = []
    row_max_ia: list[float] = []
    root_recall_values: list[float] = []
    depth_histogram: Counter[str] = Counter()
    ordered_records: list[tuple[str, str, tuple[int, ...]]] = []
    protein_ids: set[str] = set()
    all_zero_rows = 0
    root_positive_rows = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        terms = header[2:]

        for row in reader:
            protein_id, sequence = row[:2]
            protein_ids.add(protein_id)
            labels = row[2:]
            positive_indices = [index for index, value in enumerate(labels) if value == "1"]
            ordered_records.append((protein_id, sequence, tuple(positive_indices)))
            positive_terms = [terms[index] for index in positive_indices]
            for term in positive_terms:
                term_support[term] += 1
            label_count = len(positive_terms)
            label_counts.append(label_count)
            if label_count == 0:
                all_zero_rows += 1
                non_root_counts.append(0)
                root_recall_values.append(0.0)
                depth_histogram["all-zero"] += 1
                continue
            root_present = root in positive_terms
            if root_present:
                root_positive_rows += 1
            root_recall_values.append((1.0 / label_count) if root_present else 0.0)
            non_root = [term for term in positive_terms if term != root]
            non_root_counts.append(len(non_root))
            if not non_root:
                depth_histogram["root-only"] += 1
                root_only_rows.append(
                    {
                        "aspect": aspect,
                        "split": split,
                        "protein_id": protein_id,
                    }
                )
                continue
            shortest_depths = [shortest[term] for term in non_root if term in shortest]
            longest_depths = [longest[term] for term in non_root if term in longest]
            row_max_shortest = max(shortest_depths)
            row_max_longest = max(longest_depths)
            row_max_shortest_depths.append(row_max_shortest)
            row_max_longest_depths.append(row_max_longest)
            depth_histogram[str(row_max_shortest)] += 1
            if ia_values:
                present_ia = [ia_values[term] for term in non_root if term in ia_values]
                if present_ia:
                    row_max_ia.append(max(present_ia))

    row_count = len(label_counts)
    macro_precision = root_positive_rows / row_count
    macro_recall = sum(root_recall_values) / row_count
    baseline_f = (
        2.0 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if macro_precision + macro_recall
        else 0.0
    )
    ranked_support = sorted(term_support.values(), reverse=True)
    positive_total = sum(ranked_support)
    for term in terms:
        term_rows.append(
            {
                "aspect": aspect,
                "split": split,
                "term": term,
                "is_root": term == root,
                "support": term_support[term],
                "shortest_depth": shortest.get(term),
                "longest_depth": longest.get(term),
                "ia": ia_values.get(term),
            }
        )
    report = {
        "file": path.name,
        "rows": row_count,
        "terms": len(terms),
        "ordered_terms": terms,
        "_ordered_records": ordered_records,
        "positive_labels": positive_total,
        "all_zero_rows": all_zero_rows,
        "root": root,
        "root_positive_rows": root_positive_rows,
        "root_only_rows": len(root_only_rows),
        "root_only_fraction": len(root_only_rows) / row_count,
        "rows_with_non_root_labels": row_count - len(root_only_rows) - all_zero_rows,
        "labels_per_protein": describe(label_counts),
        "non_root_labels_per_protein": describe(non_root_counts),
        "max_shortest_depth_per_non_root_row": describe(row_max_shortest_depths),
        "max_longest_depth_per_non_root_row": describe(row_max_longest_depths),
        "max_ia_per_non_root_row": describe(row_max_ia) if ia_values else None,
        "ia_terms_missing": (
            sorted(term for term in terms if term not in ia_values)
            if ia_values
            else None
        ),
        "terms_with_support": sum(value > 0 for value in term_support.values()),
        "positive_mass_top10_terms": (
            sum(ranked_support[:10]) / positive_total if positive_total else 0.0
        ),
        "positive_mass_top100_terms": (
            sum(ranked_support[:100]) / positive_total if positive_total else 0.0
        ),
        "root_only_diagnostic_baseline": {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f": baseline_f,
            "note": "Arithmetic diagnostic on the retained label matrix; not a cafaeval result.",
        },
    }
    return report, root_only_rows, term_rows, depth_histogram
