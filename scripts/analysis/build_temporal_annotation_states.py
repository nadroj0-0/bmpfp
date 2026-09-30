#!/usr/bin/env python3
"""Build temporal cohort and transition tables from explicit input files."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from label_space_common import (
    write_json,
    write_text,
)
from temporal_annotation_common import (
    ASPECTS,
    build_temporal_state_rows,
    build_term_transition_rows,
    read_annotation_rows,
    read_exposure_rows,
    read_protein_scope,
)


def _cohort_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "protein_id",
        "aspect",
        "t0_state_available",
        "t1_state_available",
        "global_knowledge_state",
        "aspect_knowledge_state",
        "cross_ontology_known",
        "same_aspect_partial",
        "root_only_t0",
        "direct_t0_terms",
        "direct_t1_terms",
        "closure_t0_terms",
        "closure_t1_terms",
        "retained_terms",
        "gained_terms",
        "lost_terms",
        "direct_t0_count",
        "direct_t1_count",
        "closure_t0_count",
        "closure_t1_count",
        "retained_count",
        "gained_count",
        "lost_count",
        "global_direct_t0_count",
        "global_direct_t1_count",
        "global_t0_empty",
        "aspect_t0_empty",
        "aspect_has_gain",
        "aspect_has_loss",
        "train_id_member",
        "valid_id_member",
        "train_sequence_member",
        "valid_sequence_member",
        "train_homology_cluster_member",
        "modality_availability",
        "feature_temporal_policy",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        value = dict(row)
        for field in (
            "direct_t0_terms",
            "direct_t1_terms",
            "closure_t0_terms",
            "closure_t1_terms",
            "retained_terms",
            "gained_terms",
            "lost_terms",
        ):
            value[field] = "|".join(value[field])
        for field in (
            "t0_state_available",
            "t1_state_available",
            "cross_ontology_known",
            "same_aspect_partial",
            "root_only_t0",
            "global_t0_empty",
            "aspect_t0_empty",
            "aspect_has_gain",
            "aspect_has_loss",
        ):
            value[field] = int(bool(value[field]))
        for field in (
            "train_id_member",
            "valid_id_member",
            "train_sequence_member",
            "valid_sequence_member",
            "train_homology_cluster_member",
        ):
            value[field] = (
                "unknown" if value[field] is None else int(bool(value[field]))
            )
        writer.writerow(value)
    return stream.getvalue()


def _transition_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "protein_id",
        "aspect",
        "go_id",
        "direct_t0",
        "direct_t1",
        "closure_t0",
        "closure_t1",
        "transition",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        value = dict(row)
        for field in ("direct_t0", "direct_t1", "closure_t0", "closure_t1"):
            value[field] = int(bool(value[field]))
        writer.writerow(value)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0-direct-annotations", type=Path, required=True)
    parser.add_argument("--t1-direct-annotations", type=Path, required=True)
    parser.add_argument("--t0-closure-annotations", type=Path, required=True)
    parser.add_argument("--t1-closure-annotations", type=Path, required=True)
    parser.add_argument("--t0-protein-presence", type=Path, required=True)
    parser.add_argument("--t1-protein-presence", type=Path, required=True)
    parser.add_argument(
        "--exposure-table",
        type=Path,
        help=(
            "Required when any analysed protein has qualifying t0 knowledge; "
            "records train/validation ID, sequence, cluster and feature exposure."
        ),
    )
    parser.add_argument("--protein-scope", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--t0-snapshot", required=True)
    parser.add_argument("--t1-snapshot", required=True)
    parser.add_argument("--benchmark-id", required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    protein_ids, scope_metadata = read_protein_scope(args.protein_scope.resolve())
    scope = set(protein_ids)
    t0_presence_ids, t0_presence_metadata = read_protein_scope(
        args.t0_protein_presence.resolve()
    )
    t1_presence_ids, t1_presence_metadata = read_protein_scope(
        args.t1_protein_presence.resolve()
    )
    t0_direct, t0_direct_metadata = read_annotation_rows(
        args.t0_direct_annotations.resolve(), scope
    )
    t1_direct, t1_direct_metadata = read_annotation_rows(
        args.t1_direct_annotations.resolve(), scope
    )
    t0_closure, t0_closure_metadata = read_annotation_rows(
        args.t0_closure_annotations.resolve(), scope
    )
    t1_closure, t1_closure_metadata = read_annotation_rows(
        args.t1_closure_annotations.resolve(), scope
    )
    rows = build_temporal_state_rows(
        protein_ids,
        t0_direct,
        t1_direct,
        t0_closure,
        t1_closure,
        set(t0_presence_ids),
        set(t1_presence_ids),
    )
    exposure_metadata: dict[str, Any] | None = None
    exposure_rows: dict[str, dict[str, Any]] = {}
    if args.exposure_table is not None:
        exposure_rows, exposure_metadata = read_exposure_rows(
            args.exposure_table.resolve(), scope
        )
    for row in rows:
        exposure = exposure_rows.get(str(row["protein_id"]))
        for field in (
            "train_id_member",
            "valid_id_member",
            "train_sequence_member",
            "valid_sequence_member",
            "train_homology_cluster_member",
        ):
            row[field] = exposure[field] if exposure is not None else None
        row["modality_availability"] = (
            exposure["modality_availability"]
            if exposure is not None
            else "not_required_global_no_qualifying"
        )
        row["feature_temporal_policy"] = (
            exposure["feature_temporal_policy"]
            if exposure is not None
            else "not_required_global_no_qualifying"
        )
    transitions = build_term_transition_rows(rows)
    protein_states: dict[str, str] = {}
    for row in rows:
        protein_states[str(row["protein_id"])] = str(row["global_knowledge_state"])

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
            "analysis_kind": "temporal_annotation_states",
            "benchmark_id": args.benchmark_id,
            "t0_snapshot": args.t0_snapshot,
            "t1_snapshot": args.t1_snapshot,
            "protein_count": len(protein_ids),
            "aspects": list(ASPECTS),
            "state_rows": len(rows),
            "term_transition_rows": len(transitions),
            "cohort_counts": {
                aspect: {
                    state: sum(
                        row["aspect_knowledge_state"] == state
                        for row in rows
                        if row["aspect"] == aspect
                    )
                    for state in (
                        "no_qualifying",
                        "cross_ontology_known",
                        "same_aspect_partial",
                        "root_only",
                        "unknown",
                    )
                }
                for aspect in ASPECTS
            },
            "gainer_cohort_counts": {
                aspect: {
                    state: sum(
                        row["aspect_knowledge_state"] == state
                        and row["aspect_has_gain"]
                        for row in rows
                        if row["aspect"] == aspect
                    )
                    for state in (
                        "no_qualifying",
                        "cross_ontology_known",
                        "same_aspect_partial",
                        "root_only",
                        "unknown",
                    )
                }
                for aspect in ASPECTS
            },
            "global_knowledge_counts": {
                state: sum(value == state for value in protein_states.values())
                for state in (
                    "no_qualifying",
                    "known_qualifying",
                    "root_only",
                    "unknown",
                )
            },
            "transition_counts": {
                transition: sum(row["transition"] == transition for row in transitions)
                for transition in ("retained_known", "gained", "lost")
            },
        }
    write_text(output_dir / "protein_cohorts.tsv", _cohort_tsv(rows))
    write_text(
        output_dir / "protein_term_states.tsv", _transition_tsv(transitions)
    )
    write_json(output_dir / "temporal_annotation_states.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
