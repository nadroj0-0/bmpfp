#!/usr/bin/env python3
"""Reconstruct temporal knowledge cohorts from explicit benchmark inputs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUILDER_SRC = REPOSITORY_ROOT / "benchmark_builders" / "contemporary_cafa" / "src"
if str(BUILDER_SRC) not in sys.path:
    sys.path.insert(0, str(BUILDER_SRC))

from cafa_benchmark_builder.goa import load_normalized_annotation_map  # noqa: E402
from cafa_benchmark_builder.ontology import Ontology  # noqa: E402
from cafa_benchmark_builder.parsers import load_protein_catalog  # noqa: E402
from cafa_benchmark_builder.snapshot import (  # noqa: E402
    _build_identity_crosswalk,
    _drop_protein_binding_only,
)
from label_space_common import ASPECT_TO_NAMESPACE  # noqa: E402


ASPECTS = ("BPO", "CCO", "MFO")
PREFIX_TO_ASPECT = {"bp": "BPO", "cc": "CCO", "mf": "MFO"}
NAMESPACE_TO_ASPECT = {
    namespace: aspect for aspect, namespace in ASPECT_TO_NAMESPACE.items()
}
EXPOSURE_FIELDS = (
    "protein_id",
    "train_id_member",
    "valid_id_member",
    "train_sequence_member",
    "valid_sequence_member",
    "train_homology_cluster_member",
    "modality_availability",
    "feature_temporal_policy",
)


def _split_annotations(
    annotations: Mapping[str, set[str]], ontology: Ontology
) -> dict[str, dict[str, frozenset[str]]]:
    output: dict[str, dict[str, frozenset[str]]] = {}
    for protein_id, terms in annotations.items():
        by_aspect = {aspect: set() for aspect in ASPECTS}
        for term in terms:
            namespace = ontology.get_namespace(term)
            aspect = NAMESPACE_TO_ASPECT.get(namespace)
            by_aspect[aspect].add(term)
        output[protein_id] = {
            aspect: frozenset(by_aspect[aspect]) for aspect in ASPECTS
        }
    return output


def _propagate_annotations(
    direct: Mapping[str, Mapping[str, frozenset[str]]], ontology: Ontology
) -> dict[str, dict[str, frozenset[str]]]:
    ancestor_cache: dict[str, frozenset[str]] = {}
    output: dict[str, dict[str, frozenset[str]]] = {}
    for protein_id, values in direct.items():
        propagated: dict[str, frozenset[str]] = {}
        for aspect in ASPECTS:
            terms: set[str] = set()
            for term in values.get(aspect, frozenset()):
                if term not in ancestor_cache:
                    ancestor_cache[term] = frozenset(ontology.get_ancestors(term))
                terms.update(ancestor_cache[term])
            propagated[aspect] = frozenset(terms)
        output[protein_id] = propagated
    return output


def _write_scope(path: Path, protein_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["protein_id"])
        writer.writerows((protein_id,) for protein_id in protein_ids)


def _write_annotations(
    path: Path,
    protein_ids: Sequence[str],
    annotations: Mapping[str, Mapping[str, frozenset[str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["protein_id", "aspect", "go_term"])
        for protein_id in protein_ids:
            for aspect in ASPECTS:
                for term in sorted(annotations.get(protein_id, {}).get(aspect, ())):
                    writer.writerow([protein_id, aspect, term])


def _load_exposure_sets(
    benchmark_dir: Path,
) -> tuple[set[str], set[str], set[str], set[str]]:
    train_ids: set[str] = set()
    valid_ids: set[str] = set()
    train_sequences: set[str] = set()
    valid_sequences: set[str] = set()
    for prefix in PREFIX_TO_ASPECT:
        for split, ids, sequences in (
            ("training", train_ids, train_sequences),
            ("validation", valid_ids, valid_sequences),
        ):
            path = benchmark_dir / f"{prefix}-{split}.csv"
            proteins = _read_benchmark_membership(path)
            ids.update(proteins)
            sequences.update(proteins.values())
    return train_ids, valid_ids, train_sequences, valid_sequences


def _read_benchmark_membership(path: Path) -> dict[str, str]:
    proteins: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            protein_id, sequence = row[:2]
            proteins[protein_id] = sequence
    return proteins


def _write_exposure(
    path: Path,
    protein_ids: Sequence[str],
    sequences: Mapping[str, str],
    train_ids: set[str],
    valid_ids: set[str],
    train_sequences: set[str],
    valid_sequences: set[str],
    feature_policy: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=EXPOSURE_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for protein_id in protein_ids:
            sequence = sequences[protein_id]
            writer.writerow(
                {
                    "protein_id": protein_id,
                    "train_id_member": int(protein_id in train_ids),
                    "valid_id_member": int(protein_id in valid_ids),
                    "train_sequence_member": int(sequence in train_sequences),
                    "valid_sequence_member": int(sequence in valid_sequences),
                    "train_homology_cluster_member": "unknown",
                    "modality_availability": "not_assessed_in_this_analysis",
                    "feature_temporal_policy": feature_policy,
                }
            )


def _relationship_policy(path: Path) -> tuple[str, ...]:
    relationships = {"is_a"}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith("relationship: "):
                fields = raw.split()
                if len(fields) >= 2:
                    relationships.add(fields[1])
    return tuple(sorted(relationships))


def _cohort_tsv(summary: Mapping[str, Any]) -> str:
    stream = io.StringIO(newline="")
    fields = ("aspect", "knowledge_state", "all_t1_annotated", "with_new_terms")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for aspect in ASPECTS:
        for state, count in summary["cohort_counts"][aspect].items():
            writer.writerow(
                {
                    "aspect": aspect,
                    "knowledge_state": state,
                    "all_t1_annotated": count,
                    "with_new_terms": summary["gainer_cohort_counts"][aspect][state],
                }
            )
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--t0-sprot", type=Path, required=True)
    parser.add_argument("--t0-trembl", type=Path, required=True)
    parser.add_argument("--t1-sprot", type=Path, required=True)
    parser.add_argument("--t1-trembl", type=Path, required=True)
    parser.add_argument("--goa-t0", type=Path, required=True)
    parser.add_argument("--goa-t1", type=Path, required=True)
    parser.add_argument("--benchmark-obo", type=Path, required=True)
    parser.add_argument("--t0-source-obo", type=Path, required=True)
    parser.add_argument("--t1-source-obo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-taxon",
        action="append",
        required=True,
        help="NCBI taxon included in the benchmark; repeat as needed.",
    )
    parser.add_argument(
        "--evidence-code",
        action="append",
        required=True,
        help="Evidence code used for the cohort reconstruction; repeat as needed.",
    )
    parser.add_argument(
        "--sequence-change-policy",
        choices=("exclude", "use-t0", "error"),
        default="exclude",
    )
    parser.add_argument(
        "--protein-binding-policy",
        choices=("keep", "drop-mf-protein-binding-only"),
        default="drop-mf-protein-binding-only",
    )
    parser.add_argument(
        "--allow-frozen-source-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--benchmark-id",
        default="contemporary-global-nk",
    )
    parser.add_argument("--t0-snapshot-label", default="UniProt-2025_01/GOA-225")
    parser.add_argument("--t1-snapshot-label", default="UniProt-2026_02/GOA-234")
    parser.add_argument(
        "--feature-temporal-policy",
        default="text-cutoff-2025-03-08__ppi-paper-faithful",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    benchmark_dir = args.benchmark_dir.resolve()

    sources = {
        "t0_sprot": args.t0_sprot.resolve(),
        "t0_trembl": args.t0_trembl.resolve(),
        "t1_sprot": args.t1_sprot.resolve(),
        "t1_trembl": args.t1_trembl.resolve(),
        "goa_t0": args.goa_t0.resolve(),
        "goa_t1": args.goa_t1.resolve(),
        "benchmark_obo": args.benchmark_obo.resolve(),
        "t0_source_obo": args.t0_source_obo.resolve(),
        "t1_source_obo": args.t1_source_obo.resolve(),
    }
    relationships = _relationship_policy(sources["benchmark_obo"])

    benchmark_go = Ontology(sources["benchmark_obo"], with_rels=True)
    t0_go = Ontology(sources["t0_source_obo"], with_rels=True)
    t1_go = Ontology(sources["t1_source_obo"], with_rels=True)
    taxa = frozenset(str(value).strip() for value in args.target_taxon if str(value).strip())
    evidence_codes = frozenset(
        str(value).strip() for value in args.evidence_code if str(value).strip()
    )

    t0_catalog = load_protein_catalog(
        (sources["t0_sprot"], sources["t0_trembl"]), taxa, False
    )
    t1_catalog = load_protein_catalog(
        (sources["t1_sprot"], sources["t1_trembl"]), taxa, False
    )
    matches, t1_to_t0 = _build_identity_crosswalk(
        t0_catalog, t1_catalog, args.sequence_change_policy
    )
    match_counts = Counter((match.status, match.reason) for match in matches)

    t0_result = load_normalized_annotation_map(
        sources["goa_t0"],
        alias_to_primary=t0_catalog.alias_to_primary,
        source_ontology=t0_go,
        benchmark_ontology=benchmark_go,
        allow_frozen_source_fallback=args.allow_frozen_source_fallback,
        evidence_codes=evidence_codes,
    )
    t1_result = load_normalized_annotation_map(
        sources["goa_t1"],
        alias_to_primary=t1_catalog.alias_to_primary,
        source_ontology=t1_go,
        benchmark_ontology=benchmark_go,
        allow_frozen_source_fallback=args.allow_frozen_source_fallback,
        evidence_codes=evidence_codes,
        target_taxa=taxa,
    )
    translated_t1 = {
        t1_to_t0[t1_id]: _drop_protein_binding_only(
            set(terms), benchmark_go, args.protein_binding_policy
        )
        for t1_id, terms in t1_result.annotations.items()
        if t1_id in t1_to_t0
    }
    scope = tuple(sorted(protein_id for protein_id, terms in translated_t1.items() if terms))
    t0_terms = {
        protein_id: set(t0_result.annotations.get(protein_id, set()))
        for protein_id in scope
        if t0_result.annotations.get(protein_id)
    }
    t1_terms = {protein_id: set(translated_t1[protein_id]) for protein_id in scope}
    t0_direct = _split_annotations(t0_terms, benchmark_go)
    t1_direct = _split_annotations(t1_terms, benchmark_go)
    t0_closure = _propagate_annotations(t0_direct, benchmark_go)
    t1_closure = _propagate_annotations(t1_direct, benchmark_go)

    train_ids, valid_ids, train_sequences, valid_sequences = (
        _load_exposure_sets(benchmark_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cohort-inputs-") as temporary:
        input_dir = Path(temporary)
        _write_scope(input_dir / "protein_scope.tsv", scope)
        _write_scope(input_dir / "t0_presence.tsv", scope)
        _write_scope(input_dir / "t1_presence.tsv", scope)
        _write_annotations(input_dir / "t0_direct.tsv", scope, t0_direct)
        _write_annotations(input_dir / "t1_direct.tsv", scope, t1_direct)
        _write_annotations(input_dir / "t0_closure.tsv", scope, t0_closure)
        _write_annotations(input_dir / "t1_closure.tsv", scope, t1_closure)
        _write_exposure(
            input_dir / "exposure.tsv",
            scope,
            t0_catalog.sequences,
            train_ids,
            valid_ids,
            train_sequences,
            valid_sequences,
            args.feature_temporal_policy,
        )
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "analysis" / "build_temporal_annotation_states.py"),
            "--t0-direct-annotations", str(input_dir / "t0_direct.tsv"),
            "--t1-direct-annotations", str(input_dir / "t1_direct.tsv"),
            "--t0-closure-annotations", str(input_dir / "t0_closure.tsv"),
            "--t1-closure-annotations", str(input_dir / "t1_closure.tsv"),
            "--t0-protein-presence", str(input_dir / "t0_presence.tsv"),
            "--t1-protein-presence", str(input_dir / "t1_presence.tsv"),
            "--exposure-table", str(input_dir / "exposure.tsv"),
            "--protein-scope", str(input_dir / "protein_scope.tsv"),
            "--output-dir", str(output_dir),
            "--t0-snapshot", args.t0_snapshot_label,
            "--t1-snapshot", args.t1_snapshot_label,
            "--benchmark-id", args.benchmark_id,
        ]
        subprocess.run(command, check=True)

        summary_path = output_dir / "temporal_annotation_states.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        preparation = {
            "analysis_kind": "contemporary_knowledge_cohort_preparation",
            "scope_policy": "matched unchanged t0/t1 proteins with at least one policy-eligible direct t1 term",
            "source_policy": {
                "evidence_codes": sorted(evidence_codes),
                "target_taxa": sorted(taxa),
                "sequence_change_policy": args.sequence_change_policy,
                "protein_binding_policy": args.protein_binding_policy,
                "allow_frozen_source_fallback": args.allow_frozen_source_fallback,
                "graph_relationships": list(relationships),
                "graph_semantics": "all relationship lines in the supplied benchmark ontology",
            },
            "catalog_counts": {
                "t0_proteins": len(t0_catalog.records),
                "t1_proteins": len(t1_catalog.records),
                "matched_unchanged": len(t1_to_t0),
                "cohort_scope": len(scope),
            },
            "identity_match_counts": {
                f"{status}:{reason}": count
                for (status, reason), count in sorted(match_counts.items())
            },
            "annotation_load_counts": {
                "t0": dict(sorted(t0_result.counters.items())),
                "t1": dict(sorted(t1_result.counters.items())),
            },
        }
        summary["preparation"] = preparation
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "contemporary_knowledge_cohort_preparation.json").write_text(
            json.dumps(preparation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "contemporary_knowledge_cohorts.tsv").write_text(
            _cohort_tsv(summary), encoding="utf-8"
        )

    print(f"Wrote contemporary knowledge-cohort analysis: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
