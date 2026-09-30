from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .builder import propagate_annotations
from .config import BENCHMARK_EVIDENCE_CODES, PREFIX_TO_NAMESPACE
from .goa import load_normalized_annotation_map
from .ontology import Ontology
from .parsers import iter_uniprot


SOURCE_PICKLES = {
    "training": "train_data_train.pkl",
    "validation": "train_data_valid.pkl",
    "test": "test_data.pkl",
}
ASPECTS = ("bp", "cc", "mf")


def _load_source(
    source_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    frames = []
    split_members: dict[str, set[str]] = {}
    for split, filename in SOURCE_PICKLES.items():
        path = source_dir / filename
        frame = pd.read_pickle(path)
        frame = frame.loc[:, ["proteins", "sequences", "annotations"]].copy()
        frame["annotations"] = frame["annotations"].map(lambda value: tuple(sorted(set(value))))
        frames.append(frame)
        members = set()
        for prefix in ASPECTS:
            members.update(
                pd.read_csv(source_dir / f"{prefix}-{split}.csv", usecols=["proteins"])
                .proteins.astype(str)
            )
        split_members[split] = members

    combined = pd.concat(frames, ignore_index=True)
    visible = set().union(*split_members.values())
    combined = combined[combined.proteins.astype(str).isin(visible)].copy()

    terms = pd.read_pickle(source_dir / "terms.pkl").terms.astype(str).tolist()
    counts = {split: len(members) for split, members in split_members.items()}
    return (
        combined.sort_values("proteins", kind="stable").reset_index(drop=True),
        counts,
        terms,
    )


def _relabel_at_snapshot(
    source: pd.DataFrame,
    uniprot_paths: tuple[Path, ...],
    goa_path: Path,
    source_go: Ontology,
    benchmark_go: Ontology,
    evidence_codes: frozenset[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    source_ids = set(source.proteins.astype(str))
    source_sequences = dict(zip(source.proteins.astype(str), source.sequences.astype(str)))
    records = {}
    source_candidates: dict[str, set[str]] = defaultdict(set)
    alias_candidates: dict[str, set[str]] = defaultdict(set)

    for path in uniprot_paths:
        for record in iter_uniprot(path):
            identifiers = set(record.accessions) | {record.protein_id}
            matched = identifiers & source_ids
            if not matched:
                continue
            records.setdefault(record.protein_id, record)
            for source_id in matched:
                source_candidates[source_id].add(record.protein_id)
            for alias in identifiers:
                alias_candidates[alias].add(record.protein_id)

    missing_sequences = sorted(source_ids - set(source_candidates))

    # Keep the paired source population fixed when an accession has left UniProt.
    source_to_primary = {source_id: source_id for source_id in missing_sequences}
    sequence_disambiguated = 0
    for source_id, candidates in source_candidates.items():
        if len(candidates) == 1:
            source_to_primary[source_id] = next(iter(candidates))
            continue
        exact = {
            candidate
            for candidate in candidates
            if records[candidate].sequence == source_sequences[source_id]
        }
        source_to_primary[source_id] = sorted(exact or candidates)[0]
        sequence_disambiguated += 1

    primary_source_counts = Counter(source_to_primary.values())
    alias_to_primary = {
        alias: next(iter(primaries))
        for alias, primaries in alias_candidates.items()
        if len(primaries) == 1
    }
    alias_to_primary.update({source_id: source_id for source_id in missing_sequences})
    annotations = load_normalized_annotation_map(
        goa_path,
        alias_to_primary=alias_to_primary,
        source_ontology=source_go,
        benchmark_ontology=benchmark_go,
        evidence_codes=evidence_codes,
    )

    rows = []
    missing_labels = []
    for source_id in sorted(source_ids):
        primary = source_to_primary[source_id]
        direct = annotations.annotations.get(primary)
        propagated = propagate_annotations(benchmark_go, direct or set())
        if not propagated:
            missing_labels.append(source_id)
            continue
        rows.append({
            "proteins": source_id,
            "sequences": source_sequences[source_id],
            "annotations": tuple(sorted(propagated)),
        })
    return pd.DataFrame(rows), {
        "source_population_proteins": len(source_to_primary),
        "mapped_source_proteins": len(source_to_primary) - len(missing_sequences),
        "sequence_disambiguated_source_proteins": sequence_disambiguated,
        "source_sequence_fallback_proteins": len(missing_sequences),
        "merged_t1_primary_groups": sum(
            count > 1 for count in primary_source_counts.values()
        ),
        "source_proteins_in_merged_t1_primary_groups": sum(
            count for count in primary_source_counts.values() if count > 1
        ),
        "qualifying_t1_proteins": len(rows),
        "nonqualifying_t1_proteins_excluded": len(missing_labels),
        "nonqualifying_t1_protein_sample": missing_labels[:10],
        "goa_counters": dict(sorted(annotations.counters.items())),
        "goa_evidence_counts": dict(sorted(annotations.evidence_counts.items())),
    }


def _take_groups(groups: list[list[int]], target: int) -> tuple[list[list[int]], list[list[int]]]:
    selected = []
    remaining = []
    selected_rows = 0
    for group in groups:
        if selected_rows < target and len(group) <= target - selected_rows:
            selected.append(group)
            selected_rows += len(group)
        else:
            remaining.append(group)
    return selected, remaining


def _random_split(
    source: pd.DataFrame,
    source_counts: dict[str, int],
    seed: int,
) -> dict[str, pd.DataFrame]:
    sequence_groups: dict[str, list[int]] = defaultdict(list)
    for index, sequence in enumerate(source.sequences):
        sequence_groups[str(sequence)].append(index)
    groups = list(sequence_groups.values())
    np.random.RandomState(seed).shuffle(groups)
    test_groups, groups = _take_groups(groups, source_counts["test"])
    validation_groups, training_groups = _take_groups(groups, source_counts["validation"])
    grouped = {
        "training": training_groups,
        "validation": validation_groups,
        "test": test_groups,
    }
    frames = {}
    for split, split_groups in grouped.items():
        indices = [index for group in split_groups for index in group]
        frames[split] = source.iloc[indices].sort_values("proteins", kind="stable").reset_index(drop=True)
    return frames


def _write_outputs(
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    terms: list[str],
    go: Ontology,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    development = pd.concat([frames["training"], frames["validation"]], ignore_index=True)
    development.sort_values("proteins", kind="stable").to_pickle(output_dir / "train_data.pkl")
    frames["training"].to_pickle(output_dir / "train_data_train.pkl")
    frames["validation"].to_pickle(output_dir / "train_data_valid.pkl")
    frames["test"].to_pickle(output_dir / "test_data.pkl")
    pd.DataFrame({"terms": terms}).to_pickle(output_dir / "terms.pkl")

    stats = {}
    visible_by_split: dict[str, set[str]] = defaultdict(set)
    for prefix, namespace in PREFIX_TO_NAMESPACE.items():
        aspect_terms = [term for term in terms if go.get_namespace(term) == namespace]
        for split, frame in frames.items():
            path = output_dir / f"{prefix}-{split}.csv"
            rows = 0
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["proteins", "sequences", *aspect_terms])
                for row in frame.itertuples(index=False):
                    labels = set(row.annotations)
                    if not labels.intersection(aspect_terms):
                        continue
                    writer.writerow([
                        row.proteins,
                        row.sequences,
                        *(int(term in labels) for term in aspect_terms),
                    ])
                    rows += 1
                    visible_by_split[split].add(str(row.proteins))
            stats[path.name] = {"rows": rows, "terms": len(aspect_terms)}
    return stats, {split: len(visible_by_split[split]) for split in frames}


def build_random_baseline(
    *,
    mode: str,
    source_dir: Path,
    go_obo: Path,
    output_dir: Path,
    seed: int = 0,
    evidence_codes: frozenset[str] = BENCHMARK_EVIDENCE_CODES,
    uniprot_paths: tuple[Path, ...] = (),
    goa_path: Path | None = None,
    source_go_obo: Path | None = None,
) -> Path:
    source, source_counts, terms = _load_source(source_dir)
    go = Ontology(go_obo, with_rels=True)
    snapshot_stats = None
    if mode == "t1-snapshot":
        source_go = Ontology(source_go_obo or go_obo, with_rels=True)
        source, snapshot_stats = _relabel_at_snapshot(
            source, uniprot_paths, goa_path, source_go, go, evidence_codes
        )

    requested_counts = dict(source_counts)
    if mode == "t1-snapshot":
        requested_counts["training"] = (
            len(source) - source_counts["validation"] - source_counts["test"]
        )
    frames = _random_split(source, requested_counts, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_stats, _ = _write_outputs(output_dir, frames, terms, go)
    actual_counts = {split: len(frame) for split, frame in frames.items()}
    summary = {
        "mode": mode,
        "seed": seed,
        "evidence_codes": sorted(evidence_codes),
        "relationship_policy": "all-relationships",
        "source_split_counts": source_counts,
        "output_split_counts": actual_counts,
        "snapshot_filtering_counts": snapshot_stats,
        "csv_outputs": csv_stats,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a random split paired control from an explicit PFP benchmark directory."
    )
    parser.add_argument("--mode", choices=("prepared", "t1-snapshot"), required=True)
    parser.add_argument("--source-benchmark-dir", type=Path, required=True)
    parser.add_argument("--go-obo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evidence-code", action="append", default=[])
    parser.add_argument("--uniprot", type=Path, action="append", default=[])
    parser.add_argument("--goa", type=Path)
    parser.add_argument("--source-go-obo", type=Path)
    args = parser.parse_args()
    build_random_baseline(
        mode=args.mode,
        source_dir=args.source_benchmark_dir,
        go_obo=args.go_obo,
        output_dir=args.output_dir,
        seed=args.seed,
        evidence_codes=(
            frozenset(args.evidence_code) if args.evidence_code else BENCHMARK_EVIDENCE_CODES
        ),
        uniprot_paths=tuple(args.uniprot),
        goa_path=args.goa,
        source_go_obo=args.source_go_obo,
    )


if __name__ == "__main__":
    main()
