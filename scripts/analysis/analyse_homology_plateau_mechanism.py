#!/usr/bin/env python3
"""Analyse homology-cluster sparsity and Random-H supervised exposure."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import gzip
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from label_space_common import (
    OboGraph,
    describe,
    read_obo,
)
from pfp_sensitivity_common import (
    load_aspect_bundle,
    load_evaluation_summary,
)


ASPECTS = ("BPO", "CCO", "MFO")
PREFIX = {"BPO": "bp", "CCO": "cc", "MFO": "mf"}
SPLITS = ("training", "validation", "test")


def read_ids(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            values.add(row[0])
    return values


def memberships(root: Path) -> dict[str, dict[str, set[str]]]:
    return {
        aspect: {split: read_ids(root / f"{PREFIX[aspect]}-{split}.csv") for split in SPLITS}
        for aspect in ASPECTS
    }


def tsv(rows: Iterable[Mapping], fields: Sequence[str]) -> str:
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def size_bin(value: int, exposure: bool = False) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 5:
        return "3-5"
    if exposure:
        return "6+"
    if value <= 10:
        return "6-10"
    return ">10"


def read_clusters(path: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        member_field = next(
            name for name in reader.fieldnames
            if name == "cluster_member_count" or name.endswith("_member_count")
        )
        for row in reader:
            cluster = row["mmseqs_cluster_id"]
            split = row["split"]
            full = int(row[member_field])
            supervised = int(row["qualifying_uniprot_count"])
            result[cluster] = {"split": split, "full": full, "supervised": supervised}
    return result


def read_assignments(
    path: Path, clusters: Mapping[str, Mapping]
) -> tuple[dict[str, tuple[str, str]], Counter[str], Counter[str]]:
    result: dict[str, tuple[str, str]] = {}
    canonical_counts: Counter[str] = Counter()
    canonical_seen: dict[str, tuple[str, str, str]] = {}
    qualifying_members: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        member_field = next(
            name for name in reader.fieldnames
            if name == "cluster_member_id"
            or (name.startswith("uniref") and name.endswith("_id"))
        )
        for row in reader:
            cluster, split = row["mmseqs_cluster_id"], row["split"]
            if not cluster and not split:
                continue
            canonical = row["uniprot_accession"]
            member = row[member_field] or canonical
            current_details = (cluster, split, member)
            previous_details = canonical_seen.get(canonical)
            if previous_details is None:
                canonical_seen[canonical] = current_details
                canonical_counts[cluster] += 1
                qualifying_members[cluster].add(member)
            for field in ("raw_uniprot_accession", "uniprot_accession"):
                protein = row[field]
                if not protein:
                    continue
                current = (cluster, split)
                result[protein] = current
    return result, canonical_counts, Counter(
        {cluster: len(values) for cluster, values in qualifying_members.items()}
    )


def read_retained_members(
    path: Path,
) -> tuple[dict[str, tuple[str, str, int]], Counter[str]]:
    members: dict[str, tuple[str, str, int]] = {}
    counts: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        member_field = next(
            name for name in reader.fieldnames
            if name == "cluster_member_id"
            or (name.startswith("uniref") and name.endswith("_id"))
        )
        for row in reader:
            member = row[member_field]
            cluster = row["mmseqs_cluster_id"]
            split = row["split"]
            flag = row["connected_to_qualifying_uniprot"]
            members[member] = (cluster, split, int(flag))
            counts[cluster] += 1
    return members, counts


def metric_components(truth: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    predicted = scores >= threshold
    positive = truth.astype(bool, copy=False)
    tp = np.logical_and(predicted, positive).sum(axis=1).astype(float)
    predicted_count = predicted.sum(axis=1).astype(float)
    true_count = positive.sum(axis=1).astype(float)
    precision = np.divide(tp, predicted_count, out=np.zeros_like(tp), where=predicted_count > 0)
    recall = tp / true_count
    protein_f = np.divide(2 * tp, predicted_count + true_count, out=np.zeros_like(tp), where=(predicted_count + true_count) > 0)
    return {"precision": precision, "recall": recall, "covered": predicted_count > 0, "protein_f": protein_f}


def aggregate_metric(components: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
    covered = components["covered"][indices]
    precision = float(np.mean(components["precision"][indices][covered])) if covered.any() else 0.0
    recall = float(np.mean(components["recall"][indices])) if len(indices) else 0.0
    f_value = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f": f_value,
        "coverage": float(np.mean(covered)) if len(indices) else 0.0,
        "mean_protein_f": float(np.mean(components["protein_f"][indices])) if len(indices) else 0.0,
    }


def propagation_plan(
    go_terms: Sequence[str], graph: OboGraph, aspect: str
) -> list[tuple[int, tuple[int, ...]]]:
    term_index = {term: index for index, term in enumerate(go_terms)}
    closure = graph.ancestor_closure(aspect)
    plan: list[tuple[int, tuple[int, ...]]] = []
    for child, child_index in term_index.items():
        ancestors = closure[child]
        parents = tuple(
            sorted(
                term_index[ancestor]
                for ancestor in ancestors
                if ancestor in term_index and term_index[ancestor] != child_index
            )
        )
        if parents:
            plan.append((child_index, parents))
    return plan


def chunked_metric_components(
    truth: np.ndarray,
    scores: np.ndarray,
    row_indices: np.ndarray,
    thresholds: Sequence[float],
    plan: Sequence[tuple[int, Sequence[int]]],
    chunk_size: int = 128,
) -> dict[float, dict[str, np.ndarray]]:
    """Reproduce writer rounding and GO max propagation with bounded memory."""
    unique_thresholds = tuple(dict.fromkeys(float(value) for value in thresholds))
    result = {
        threshold: {
            "precision": np.zeros(len(row_indices), dtype=np.float64),
            "recall": np.zeros(len(row_indices), dtype=np.float64),
            "covered": np.zeros(len(row_indices), dtype=bool),
            "protein_f": np.zeros(len(row_indices), dtype=np.float64),
        }
        for threshold in unique_thresholds
    }
    for start in range(0, len(row_indices), chunk_size):
        stop = min(start + chunk_size, len(row_indices))
        selected = row_indices[start:stop]
        original = np.asarray(scores[selected], dtype=np.float32)
        np.around(original, decimals=6, out=original)
        propagated = original.copy()
        for child_index, parent_indices in plan:
            child_scores = original[:, child_index]
            for parent_index in parent_indices:
                np.maximum(
                    propagated[:, parent_index],
                    child_scores,
                    out=propagated[:, parent_index],
                )
        positive = np.asarray(truth[selected], dtype=bool)
        true_count = positive.sum(axis=1).astype(float)
        for threshold in unique_thresholds:
            predicted = propagated >= threshold
            tp = np.logical_and(predicted, positive).sum(axis=1).astype(float)
            predicted_count = predicted.sum(axis=1).astype(float)
            denominator = predicted_count + true_count
            destination = result[threshold]
            destination["precision"][start:stop] = np.divide(
                tp, predicted_count, out=np.zeros_like(tp), where=predicted_count > 0
            )
            destination["recall"][start:stop] = tp / true_count
            destination["covered"][start:stop] = predicted_count > 0
            destination["protein_f"][start:stop] = np.divide(
                2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0
            )
    return result


def bootstrap_delta(
    random_components: Mapping[str, np.ndarray],
    mmseqs2_components: Mapping[str, np.ndarray],
    clusters: Sequence[str],
    indices: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for index in indices.tolist():
        by_cluster[clusters[index]].append(index)
    groups = [np.asarray(values, dtype=int) for _, values in sorted(by_cluster.items())]
    if not groups:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = rng.integers(0, len(groups), size=len(groups))
        sampled = np.concatenate([groups[value] for value in selected])
        deltas[replicate] = (
            aggregate_metric(random_components, sampled)["f"]
            - aggregate_metric(mmseqs2_components, sampled)["f"]
        )
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def analyse_clusters(
    threshold_dirs: Mapping[float, Path],
    mmseqs2_root: Path,
    output: Path,
) -> tuple[dict[str, tuple[str, str]], list[dict], dict]:
    cluster_rows_path = output / "cluster_supervision_counts.tsv.gz"
    cluster_fields = [
        "identity_percent", "cluster_id", "split", "full_members", "supervised_proteins",
        "qualifying_cluster_members", "unlabelled_cluster_members",
        "labelled_cluster_fraction", "BPO_proteins", "CCO_proteins", "MFO_proteins",
    ]
    bin_rows: list[dict] = []
    protein_bin_rows: list[dict] = []
    aspect_bin_rows: list[dict] = []
    quantile_rows: list[dict] = []
    density_rows: list[dict] = []
    threshold_summary: list[dict] = []
    mmseqs2_assignments: dict[str, tuple[str, str]] = {}
    threshold_signatures: dict[
        float,
        tuple[
            dict[str, dict[str, int | str]],
            dict[str, tuple[str, str]],
            dict[str, tuple[str, str, int]],
        ],
    ] = {}
    with gzip.open(cluster_rows_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cluster_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        previous_summary: dict | None = None
        for identity, root in threshold_dirs.items():
            paths = {
                "clusters": root / "cluster_split_assignments.tsv",
                "assignments": root / "protein_cluster_assignments.tsv",
                "members": root / "retained_cluster_members.tsv.gz",
                "summary": root / "benchmark_summary.json",
            }
            clusters = read_clusters(paths["clusters"])
            assignments, canonical_counts, qualifying_members = read_assignments(
                paths["assignments"], clusters
            )
            if root.resolve() == mmseqs2_root.resolve():
                mmseqs2_assignments = assignments
            retained_members, observed_members = read_retained_members(paths["members"])
            threshold_signatures[identity] = (
                clusters,
                assignments,
                retained_members,
            )
            aspect_members = memberships(root)
            aspect_counts: dict[str, Counter[str]] = {}
            for aspect in ASPECTS:
                proteins = set().union(*aspect_members[aspect].values())
                aspect_counts[aspect] = Counter(assignments[protein][0] for protein in proteins)
            for cluster, values in sorted(clusters.items()):
                full, supervised = int(values["full"]), int(values["supervised"])
                writer.writerow({
                    "identity_percent": identity, "cluster_id": cluster, "split": values["split"],
                    "full_members": full, "supervised_proteins": supervised,
                    "qualifying_cluster_members": qualifying_members[cluster],
                    "unlabelled_cluster_members": full - qualifying_members[cluster],
                    "labelled_cluster_fraction": qualifying_members[cluster] / full,
                    **{f"{aspect}_proteins": aspect_counts[aspect][cluster] for aspect in ASPECTS},
                })
            for split in (*SPLITS, "all"):
                selected = [values for values in clusters.values() if split == "all" or values["split"] == split]
                supervised_values = [int(values["supervised"]) for values in selected]
                full_values = [int(values["full"]) for values in selected]
                fractions = [
                    qualifying_members[cluster] / int(values["full"])
                    for cluster, values in clusters.items()
                    if split == "all" or values["split"] == split
                ]
                bins = Counter(size_bin(value) for value in supervised_values)
                proteins_by_bin = Counter()
                for value in supervised_values:
                    proteins_by_bin[size_bin(value)] += value
                for bucket in ("1", "2", "3-5", "6-10", ">10"):
                    bin_rows.append({
                        "identity_percent": identity, "split": split, "size_bin": bucket,
                        "clusters": bins[bucket], "fraction": bins[bucket] / len(selected) if selected else 0.0,
                    })
                    total_supervised = sum(supervised_values)
                    protein_bin_rows.append({
                        "identity_percent": identity, "split": split, "size_bin": bucket,
                        "proteins": proteins_by_bin[bucket],
                        "fraction": proteins_by_bin[bucket] / total_supervised if total_supervised else 0.0,
                    })
                for measure, values in (("supervised", supervised_values), ("full", full_values)):
                    description = describe(values)
                    quantile_rows.append({"identity_percent": identity, "split": split, "measure": measure, **description})
                density = describe(fractions)
                density_rows.append({"identity_percent": identity, "split": split, **density})
            for aspect in ASPECTS:
                for split in (*SPLITS, "all"):
                    values = [
                        aspect_counts[aspect][cluster] for cluster, metadata in clusters.items()
                        if aspect_counts[aspect][cluster] > 0 and (split == "all" or metadata["split"] == split)
                    ]
                    bins = Counter(size_bin(value) for value in values)
                    for bucket in ("1", "2", "3-5", "6-10", ">10"):
                        aspect_bin_rows.append({
                            "identity_percent": identity, "aspect": aspect, "split": split,
                            "size_bin": bucket, "clusters": bins[bucket],
                            "fraction": bins[bucket] / len(values) if values else 0.0,
                        })
            all_supervised = [int(values["supervised"]) for values in clusters.values()]
            current = {
                "identity_percent": identity, "clusters": len(clusters),
                "supervised_proteins": sum(all_supervised),
                "singleton_clusters": sum(value == 1 for value in all_supervised),
                "singleton_fraction": sum(value == 1 for value in all_supervised) / len(all_supervised),
                "mean_supervised_size": float(np.mean(all_supervised)),
                "median_supervised_size": float(np.median(all_supervised)),
            }
            if previous_summary is not None:
                current["delta_singleton_fraction_from_previous"] = current["singleton_fraction"] - previous_summary["singleton_fraction"]
                current["delta_median_from_previous"] = current["median_supervised_size"] - previous_summary["median_supervised_size"]
                current["delta_supervised_proteins_from_previous"] = current["supervised_proteins"] - previous_summary["supervised_proteins"]
            else:
                current["delta_singleton_fraction_from_previous"] = ""
                current["delta_median_from_previous"] = ""
                current["delta_supervised_proteins_from_previous"] = ""
            threshold_summary.append(current)
            previous_summary = current
    specs = [
        ("cluster_size_bins.tsv", bin_rows, ["identity_percent", "split", "size_bin", "clusters", "fraction"]),
        ("protein_weighted_cluster_bins.tsv", protein_bin_rows, ["identity_percent", "split", "size_bin", "proteins", "fraction"]),
        ("aspect_cluster_size_bins.tsv", aspect_bin_rows, ["identity_percent", "aspect", "split", "size_bin", "clusters", "fraction"]),
        ("cluster_size_quantiles.tsv", quantile_rows, ["identity_percent", "split", "measure", "count", "mean", "median", "p90", "p99", "maximum"]),
        ("retained_member_density.tsv", density_rows, ["identity_percent", "split", "count", "mean", "median", "p90", "p99", "maximum"]),
        ("threshold_plateau_summary.tsv", threshold_summary, [
            "identity_percent", "clusters", "supervised_proteins", "singleton_clusters", "singleton_fraction",
            "mean_supervised_size", "median_supervised_size", "delta_singleton_fraction_from_previous",
            "delta_median_from_previous", "delta_supervised_proteins_from_previous",
        ]),
    ]
    for name, rows, fields in specs:
        (output / name).write_text(tsv(rows, fields), encoding="utf-8")
    if not mmseqs2_assignments:
        mmseqs2_clusters = read_clusters(mmseqs2_root / "cluster_split_assignments.tsv")
        mmseqs2_assignments, _, _ = read_assignments(
            mmseqs2_root / "protein_cluster_assignments.tsv", mmseqs2_clusters
        )
    ten = next((value for value in threshold_dirs if math.isclose(value, 10.0)), None)
    five = next((value for value in threshold_dirs if math.isclose(value, 5.0)), None)
    duplicate = (
        None
        if ten is None or five is None
        else threshold_signatures[ten] == threshold_signatures[five]
    )
    report = {"thresholds": threshold_summary, "identity_10_and_5_duplicate": duplicate}
    (output / "cluster_sparsity_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "cluster_sparsity_analysis.md").write_text((
        "# Homology cluster sparsity\n\n"
        + "\n".join(
            f"- {row['identity_percent']}%: {row['clusters']:,} clusters, {row['supervised_proteins']:,} supervised proteins, "
            f"{row['singleton_fraction']:.2%} singleton clusters, median supervised size {row['median_supervised_size']:.1f}."
            for row in threshold_summary
        )
        + (
            "\n\nThe supplied 10% and 5% assignment outputs were "
            + ("identical" if duplicate else "not identical")
            + ".\n"
            if duplicate is not None
            else "\n\nA 10%/5% duplicate comparison was not requested.\n"
        )
    ), encoding="utf-8")
    return mmseqs2_assignments, threshold_summary, {"report": report}


def analyse_exposure(
    mmseqs2_root: Path,
    random_root: Path,
    assignments: Mapping[str, tuple[str, str]],
    mmseqs2_prediction_root: Path,
    random_prediction_root: Path,
    go_graph: OboGraph,
    output: Path,
    bootstrap_replicates: int,
) -> dict:
    random_members = memberships(random_root)
    mmseqs2_members = memberships(mmseqs2_root)
    random_summary, random_artifact_root = load_evaluation_summary(random_prediction_root)
    mmseqs2_summary, mmseqs2_artifact_root = load_evaluation_summary(mmseqs2_prediction_root)
    exposure_rows: list[dict] = []
    overlap_rows: list[dict] = []
    count_rows: list[dict] = []
    metric_rows: list[dict] = []
    delta_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    robustness_rows: list[dict] = []
    for aspect in ASPECTS:
        cluster_counts = {
            split: Counter(assignments[protein][0] for protein in random_members[aspect][split])
            for split in SPLITS
        }
        random_test = random_members[aspect]["test"]
        mmseqs2_test = mmseqs2_members[aspect]["test"]
        shared = random_test & mmseqs2_test
        overlap_rows.append({
            "aspect": aspect, "random_test": len(random_test), "mmseqs2_test": len(mmseqs2_test),
            "shared_test": len(shared), "random_only": len(random_test - mmseqs2_test),
            "mmseqs2_only": len(mmseqs2_test - random_test),
            "jaccard": len(shared) / len(random_test | mmseqs2_test),
        })
        exposure_by_protein: dict[str, tuple[str, int, int, int, int, str, str]] = {}
        for protein in sorted(random_test):
            cluster = assignments[protein][0]
            train_count = cluster_counts["training"][cluster]
            valid_count = cluster_counts["validation"][cluster]
            test_count = cluster_counts["test"][cluster]
            total = train_count + valid_count + test_count
            row = (cluster, train_count, valid_count, test_count, total, size_bin(train_count, True), size_bin(total))
            exposure_by_protein[protein] = row
            exposure_rows.append({
                "aspect": aspect, "protein_id": protein, "cluster_id": cluster,
                "n_random_train": train_count, "n_random_valid": valid_count,
                "n_random_test": test_count, "n_total_aspect": total,
                "exposure_bucket": row[5], "total_cluster_bucket": row[6],
                "in_mmseqs2_test": int(protein in mmseqs2_test),
            })
        for dimension, position, order in (
            ("training_exposure", 5, ("0", "1", "2", "3-5", "6+")),
            ("total_cluster_size", 6, ("1", "2", "3-5", "6-10", ">10")),
        ):
            observed = Counter(exposure_by_protein[p][position] for p in shared)
            all_observed = Counter(exposure_by_protein[p][position] for p in random_test)
            for bucket in order:
                count_rows.append({
                    "aspect": aspect, "dimension": dimension, "bucket": bucket,
                    "random_test_proteins": all_observed[bucket], "shared_test_proteins": observed[bucket],
                    "headline_eligible": int(observed[bucket] >= 100),
                })
        shared_ids = sorted(shared)
        random_threshold = float(random_summary["aspects"][aspect]["cafa_threshold"])
        mmseqs2_threshold = float(mmseqs2_summary["aspects"][aspect]["cafa_threshold"])
        thresholds = (random_threshold, mmseqs2_threshold)

        random_bundle = load_aspect_bundle(random_summary, random_artifact_root, aspect)
        random_ids = random_bundle["protein_ids"]
        random_index = {protein: index for index, protein in enumerate(random_ids)}
        ri = np.asarray([random_index[protein] for protein in shared_ids], dtype=int)
        go_terms = tuple(random_bundle["go_terms"])
        plan = propagation_plan(go_terms, go_graph, aspect)
        random_components = chunked_metric_components(
            random_bundle["truth"], random_bundle["scores"], ri, thresholds, plan
        )
        del random_bundle
        gc.collect()

        mmseqs2_bundle = load_aspect_bundle(mmseqs2_summary, mmseqs2_artifact_root, aspect)
        mmseqs2_ids = mmseqs2_bundle["protein_ids"]
        mmseqs2_index = {protein: index for index, protein in enumerate(mmseqs2_ids)}
        mmseqs2_indices = np.asarray(
            [mmseqs2_index[protein] for protein in shared_ids], dtype=int
        )
        mmseqs2_components = chunked_metric_components(
            mmseqs2_bundle["truth"],
            mmseqs2_bundle["scores"],
            mmseqs2_indices,
            thresholds,
            plan,
        )
        del mmseqs2_bundle
        gc.collect()
        cluster_labels = [exposure_by_protein[protein][0] for protein in shared_ids]
        dimensions = {
            "training_exposure": (5, ("0", "1", "2", "3-5", "6+")),
            "total_cluster_size": (6, ("1", "2", "3-5", "6-10", ">10")),
        }
        primary_random = random_components[random_threshold]
        primary_mmseqs2 = mmseqs2_components[mmseqs2_threshold]
        for dimension, (position, order) in dimensions.items():
            for bucket in order:
                indices = np.asarray([index for index, protein in enumerate(shared_ids) if exposure_by_protein[protein][position] == bucket], dtype=int)
                if not len(indices):
                    continue
                metrics = {}
                for label, components, threshold in (
                    ("Random-H", primary_random, random_threshold),
                    ("MMseqs2-30", primary_mmseqs2, mmseqs2_threshold),
                ):
                    value = aggregate_metric(components, indices)
                    metrics[label] = value
                    metric_rows.append({
                        "aspect": aspect, "dimension": dimension, "bucket": bucket,
                        "model": label, "proteins": len(indices), "threshold": threshold,
                        "headline_eligible": int(len(indices) >= 100), **value,
                    })
                delta = metrics["Random-H"]["f"] - metrics["MMseqs2-30"]["f"]
                delta_rows.append({
                    "aspect": aspect, "dimension": dimension, "bucket": bucket,
                    "proteins": len(indices), "random_f": metrics["Random-H"]["f"],
                    "mmseqs2_f": metrics["MMseqs2-30"]["f"], "delta_f": delta,
                    "mean_protein_f_delta": metrics["Random-H"]["mean_protein_f"] - metrics["MMseqs2-30"]["mean_protein_f"],
                    "headline_eligible": int(len(indices) >= 100),
                })
                low, high = bootstrap_delta(primary_random, primary_mmseqs2, cluster_labels, indices, bootstrap_replicates, 42)
                bootstrap_rows.append({
                    "aspect": aspect, "dimension": dimension, "bucket": bucket,
                    "proteins": len(indices), "clusters": len({cluster_labels[index] for index in indices}),
                    "replicates": bootstrap_replicates, "seed": 42,
                    "delta_f": delta, "ci_2_5": low, "ci_97_5": high,
                })
                for threshold_source, threshold in (
                    ("random", random_threshold),
                    ("mmseqs2", mmseqs2_threshold),
                ):
                    for label, components in (
                        ("Random-H", random_components[threshold]),
                        ("MMseqs2-30", mmseqs2_components[threshold]),
                    ):
                        value = aggregate_metric(components, indices)
                        robustness_rows.append({
                            "aspect": aspect, "dimension": dimension, "bucket": bucket,
                            "model": label, "threshold_source": threshold_source,
                            "threshold": threshold, "proteins": len(indices), **value,
                        })
        del random_components, mmseqs2_components, primary_random, primary_mmseqs2
        gc.collect()
    with gzip.open(output / "random_h_exposure_assignments.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        handle.write(tsv(exposure_rows, [
            "aspect", "protein_id", "cluster_id", "n_random_train", "n_random_valid",
            "n_random_test", "n_total_aspect", "exposure_bucket", "total_cluster_bucket", "in_mmseqs2_test",
        ]))
    specs = [
        ("test_overlap_summary.tsv", overlap_rows, ["aspect", "random_test", "mmseqs2_test", "shared_test", "random_only", "mmseqs2_only", "jaccard"]),
        ("exposure_counts.tsv", count_rows, ["aspect", "dimension", "bucket", "random_test_proteins", "shared_test_proteins", "headline_eligible"]),
        ("exposure_cohort_metrics.tsv", metric_rows, ["aspect", "dimension", "bucket", "model", "proteins", "threshold", "headline_eligible", "precision", "recall", "f", "coverage", "mean_protein_f"]),
        ("paired_model_deltas.tsv", delta_rows, ["aspect", "dimension", "bucket", "proteins", "random_f", "mmseqs2_f", "delta_f", "mean_protein_f_delta", "headline_eligible"]),
        ("bootstrap_intervals.tsv", bootstrap_rows, ["aspect", "dimension", "bucket", "proteins", "clusters", "replicates", "seed", "delta_f", "ci_2_5", "ci_97_5"]),
        ("threshold_robustness.tsv", robustness_rows, ["aspect", "dimension", "bucket", "model", "threshold_source", "threshold", "proteins", "precision", "recall", "f", "coverage", "mean_protein_f"]),
    ]
    for name, rows, fields in specs:
        (output / name).write_text(tsv(rows, fields), encoding="utf-8")
    report = {
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": 42,
        "minimum_headline_cohort": 100, "test_overlap": overlap_rows,
        "interpretation_boundary": "Exposure is observational within the supplied test evaluations; it does not establish causal homology effects or training-seed uncertainty.",
    }
    (output / "random_h_exposure_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "random_h_exposure_analysis.md").write_text((
        "# Random-H supervised-exposure analysis\n\n"
        + "\n".join(
            f"- {row['aspect']}: {row['shared_test']:,} shared tests; test-set Jaccard {row['jaccard']:.3f}."
            for row in overlap_rows
        )
        + "\n\nCohort comparisons use the shared test proteins and each supplied evaluation's selected threshold. "
        "The cluster bootstrap conditions on the supplied splits and predictions and does not measure training-seed uncertainty.\n"
    ), encoding="utf-8")
    return {"report": report}


def plot_outputs(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "threshold_plateau_summary.tsv").open() as handle:
        threshold_rows = list(csv.DictReader(handle, delimiter="\t"))
    identities = [float(row["identity_percent"]) for row in threshold_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(
        identities,
        [float(row["singleton_fraction"]) for row in threshold_rows],
        marker="o",
        label="All",
    )
    with (output / "aspect_cluster_size_bins.tsv").open() as handle:
        aspect_rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row["split"] == "all" and row["size_bin"] == "1"
        ]
    for aspect in ASPECTS:
        lookup = {
            float(row["identity_percent"]): float(row["fraction"])
            for row in aspect_rows if row["aspect"] == aspect
        }
        ax.plot(identities, [lookup[value] for value in identities], marker="o", label=aspect)
    ax.invert_xaxis()
    ax.set_xlabel("MMseqs2 identity threshold (%)")
    ax.set_ylabel("Fraction of supervised clusters that are singletons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "singleton_fraction_by_threshold_and_aspect.png", dpi=180)
    fig.savefig(output / "singleton_fraction_by_threshold_and_aspect.pdf")
    plt.close(fig)

    with (output / "cluster_size_bins.tsv").open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["split"] == "all"
            and not math.isclose(float(row["identity_percent"]), 5.0)
        ]
    identities = sorted({float(row["identity_percent"]) for row in rows}, reverse=True)
    bins = ("1", "2", "3-5", "6-10", ">10")
    lookup = {(float(row["identity_percent"]), row["size_bin"]): float(row["fraction"]) for row in rows}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottoms = np.zeros(len(identities))
    for bucket in bins:
        values = np.asarray([lookup.get((identity, bucket), 0.0) for identity in identities])
        ax.bar(range(len(identities)), values, bottom=bottoms, label=bucket)
        bottoms += values
    ax.set_xticks(range(len(identities)), [str(value) for value in identities])
    ax.set_xlabel("Identity threshold (%)")
    ax.set_ylabel("Fraction of supervised clusters")
    ax.legend(title="Supervised proteins")
    fig.tight_layout()
    fig.savefig(output / "supervised_cluster_size_distribution.png", dpi=180)
    fig.savefig(output / "supervised_cluster_size_distribution.pdf")
    plt.close(fig)

    with (output / "cluster_size_quantiles.tsv").open() as handle:
        quantile_rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row["split"] == "all"
        ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True)
    for ax, statistic, label in (
        (axes[0], "mean", "Mean cluster size"),
        (axes[1], "p90", "90th-percentile cluster size"),
    ):
        for measure, display in (("full", "All retained cluster members"), ("supervised", "Supervised proteins")):
            lookup = {
                float(row["identity_percent"]): float(row[statistic])
                for row in quantile_rows if row["measure"] == measure
            }
            ax.plot(identities, [lookup[value] for value in identities], marker="o", label=display)
        ax.invert_xaxis()
        ax.set_xlabel("Identity threshold (%)")
        ax.set_ylabel(label)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "full_vs_supervised_cluster_size.png", dpi=180)
    fig.savefig(output / "full_vs_supervised_cluster_size.pdf")
    plt.close(fig)

    with (output / "paired_model_deltas.tsv").open() as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["dimension"] == "training_exposure" and row["headline_eligible"] == "1"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bucket_order = ("0", "1", "2", "3-5", "6+")
    x = np.arange(len(bucket_order))
    width = 0.25
    for index, aspect in enumerate(ASPECTS):
        lookup = {row["bucket"]: float(row["delta_f"]) for row in rows if row["aspect"] == aspect}
        ax.bar(x + (index - 1) * width, [lookup.get(bucket, np.nan) for bucket in bucket_order], width, label=aspect)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, bucket_order)
    ax.set_xlabel("Random-H training proteins in the same supplied MMseqs2-30 cluster")
    ax.set_ylabel("Random-H minus MMseqs2-30 fixed-threshold F")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "random_h_advantage_by_training_exposure.png", dpi=180)
    fig.savefig(output / "random_h_advantage_by_training_exposure.pdf")
    plt.close(fig)

    with (output / "paired_model_deltas.tsv").open() as handle:
        cluster_delta_rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row["dimension"] == "total_cluster_size" and row["headline_eligible"] == "1"
        ]
    bucket_order = ("1", "2", "3-5", "6-10", ">10")
    x = np.arange(len(bucket_order))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for index, aspect in enumerate(ASPECTS):
        lookup = {
            row["bucket"]: float(row["delta_f"])
            for row in cluster_delta_rows if row["aspect"] == aspect
        }
        ax.bar(
            x + (index - 1) * width,
            [lookup.get(bucket, np.nan) for bucket in bucket_order],
            width,
            label=aspect,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, bucket_order)
    ax.set_xlabel("Supervised proteins in the supplied MMseqs2-30 cluster")
    ax.set_ylabel("Random-H minus MMseqs2-30 fixed-threshold F")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "random_h_advantage_by_cluster_size.png", dpi=180)
    fig.savefig(output / "random_h_advantage_by_cluster_size.pdf")
    plt.close(fig)

    with (output / "exposure_counts.tsv").open() as handle:
        exposure_rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row["dimension"] == "training_exposure"
        ]
    bucket_order = ("0", "1", "2", "3-5", "6+")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottoms = np.zeros(len(ASPECTS))
    for bucket in bucket_order:
        values = []
        for aspect in ASPECTS:
            aspect_rows = [row for row in exposure_rows if row["aspect"] == aspect]
            total = sum(int(row["random_test_proteins"]) for row in aspect_rows)
            selected = next(row for row in aspect_rows if row["bucket"] == bucket)
            values.append(int(selected["random_test_proteins"]) / total if total else 0.0)
        values_array = np.asarray(values)
        ax.bar(ASPECTS, values_array, bottom=bottoms, label=bucket)
        bottoms += values_array
    ax.set_ylabel("Fraction of Random-H test proteins")
    ax.legend(title="Same-cluster training proteins")
    fig.tight_layout()
    fig.savefig(output / "exposure_population_distribution.png", dpi=180)
    fig.savefig(output / "exposure_population_distribution.pdf")
    plt.close(fig)


def parse_threshold(value: str) -> tuple[float, Path]:
    identity, path = value.split("=", 1)
    return float(identity), Path(path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-dir", action="append", type=parse_threshold, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--mmseqs2-root", type=Path, required=True)
    parser.add_argument("--random-evaluation-dir", type=Path, required=True)
    parser.add_argument("--mmseqs2-evaluation-dir", type=Path, required=True)
    parser.add_argument("--go-obo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    threshold_dirs = dict(args.threshold_dir)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    go_graph = read_obo(args.go_obo)
    assignments, threshold_rows, cluster_report = analyse_clusters(
        threshold_dirs, args.mmseqs2_root, output
    )
    exposure_report = analyse_exposure(
        args.mmseqs2_root, args.random_root, assignments,
        args.mmseqs2_evaluation_dir, args.random_evaluation_dir, go_graph, output,
        args.bootstrap_replicates,
    )
    if not args.skip_plots:
        plot_outputs(output)
    combined = {
        "thresholds": threshold_rows,
        "identity_10_and_5_duplicate": cluster_report["report"]["identity_10_and_5_duplicate"],
        "test_overlap": exposure_report["report"]["test_overlap"],
        "figures_generated": not args.skip_plots,
        "interpretation_boundary": "Results describe the supplied operational clusters and evaluation arrays; they are not proof of evolutionary ancestry or causality.",
    }
    (output / "homology_plateau_mechanism.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "homology_plateau_mechanism.md").write_text((
        "# Homology plateau mechanism\n\n"
        "This report combines the supervised cluster-size sweep with the paired Random-H exposure analysis.\n\n"
        + (
            f"The supplied 10% and 5% cluster outputs were {'identical' if combined['identity_10_and_5_duplicate'] else 'not identical'}.\n\n"
            if combined["identity_10_and_5_duplicate"] is not None
            else "No 10%/5% duplicate comparison was requested.\n\n"
        )
        + "Interpretation is limited to the supplied cluster partitions and evaluation arrays.\n"
    ), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
