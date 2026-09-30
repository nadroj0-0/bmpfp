#!/usr/bin/env python3
"""Analyse PFP predictions across information-accretion and Xu specificity bins."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from label_space_common import ASPECTS
from pfp_sensitivity_common import (
    load_aspect_bundle,
    load_evaluation_summary,
)
from specificity_common import (
    SpecificityMeasure,
    assign_specificity_bins,
    compute_xu_totipotency,
    read_nonnegative_term_values,
    read_xu_ontology,
)


def selected_aspects(values: list[str], available: list[str]) -> list[str]:
    return values or available


def read_information_accretion(
    path: Path, go_terms: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    measure = read_nonnegative_term_values(
        path,
        go_terms,
        measure_name="information_accretion",
        higher_is_more_specific=True,
        zero_bin_label="zero_ia",
    )
    source = measure.source
    return measure.values, {
        "prediction_terms": source["prediction_terms"],
        "ia_terms": source["source_terms"],
        "extra_ia_terms": source["extra_source_terms"],
        "zero_ia_terms": source["zero_terms"],
        "positive_ia_terms": source["positive_terms"],
    }


def scientific_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Retain the interpretable metadata produced while parsing an input."""
    return {key: value for key, value in source.items() if key != "path"}


def assign_information_bins(
    go_terms: Sequence[str],
    ia_values: np.ndarray,
    positive_bin_count: int,
    root_index: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    measure = SpecificityMeasure(
        name="information_accretion",
        values=ia_values,
        higher_is_more_specific=True,
        zero_bin_label="zero_ia",
        source={},
    )
    generic_bins, assignments = assign_specificity_bins(
        go_terms,
        measure,
        positive_bin_count,
        bin_prefix="positive_q",
        excluded_indices=(() if root_index is None else (root_index,)),
        excluded_label="root_excluded",
    )
    bins: list[dict[str, Any]] = []
    for value in generic_bins:
        bins.append(
            {
                "label": value["label"],
                "term_indices": value["term_indices"],
                "term_count": value["term_count"],
                "ia_min": value["value_min"],
                "ia_max": value["value_max"],
                "ia_mean": value["value_mean"],
                "weighted_metrics_available": bool(
                    value["term_indices"] and ia_values[value["term_indices"]].max() > 0
                ),
            }
        )
    return bins, assignments


def assign_measure_bins(
    go_terms: Sequence[str],
    measure: SpecificityMeasure,
    positive_bin_count: int,
    root_index: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    bins, assignments = assign_specificity_bins(
        go_terms,
        measure,
        positive_bin_count,
        bin_prefix="specificity_q",
        excluded_indices=(root_index,),
        excluded_label="root_excluded",
    )
    return bins, assignments


def _mean_ratio(
    numerator: np.ndarray, denominator: np.ndarray, selected: np.ndarray
) -> float:
    if not selected.any():
        return 0.0
    return float(np.mean(numerator[selected] / denominator[selected]))


def threshold_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    true = truth.astype(bool, copy=False)
    predicted = scores >= threshold
    true_positive = np.logical_and(predicted, true)
    false_positive = np.logical_and(predicted, ~true)
    false_negative = np.logical_and(~predicted, true)

    tp_count = true_positive.sum(axis=1, dtype=np.int64)
    predicted_count = predicted.sum(axis=1, dtype=np.int64)
    true_count = true.sum(axis=1, dtype=np.int64)
    union_count = np.logical_or(predicted, true).sum(axis=1, dtype=np.int64)
    covered = predicted_count > 0
    target_positive = true_count > 0
    union_positive = union_count > 0
    precision = _mean_ratio(tp_count, predicted_count, covered)
    recall = _mean_ratio(tp_count, true_count, target_positive)
    f_score = (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    jaccard = _mean_ratio(tp_count, union_count, union_positive)
    tp = int(tp_count.sum())
    fp = int(false_positive.sum())
    fn = int(false_negative.sum())
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    result: dict[str, float | int | None] = {
        "threshold": float(threshold),
        "population_proteins": int(truth.shape[0]),
        "target_positive_proteins": int(target_positive.sum()),
        "predicted_coverage": float(covered.mean()) if len(covered) else 0.0,
        "precision": precision,
        "recall": recall,
        "f": f_score,
        "jaccard_set_agreement": jaccard,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f": micro_f,
    }

    if weights.size == 0 or not np.any(weights > 0):
        result.update(
            {
                "weighted_precision": None,
                "weighted_recall": None,
                "weighted_f": None,
                "weighted_jaccard_set_agreement": None,
                "micro_weighted_precision": None,
                "micro_weighted_recall": None,
                "micro_weighted_f": None,
            }
        )
        return result

    tp_weight = true_positive @ weights
    fp_weight = false_positive @ weights
    fn_weight = false_negative @ weights
    predicted_weight = tp_weight + fp_weight
    true_weight = tp_weight + fn_weight
    union_weight = tp_weight + fp_weight + fn_weight
    weighted_covered = predicted_weight > 0
    weighted_target = true_weight > 0
    weighted_union = union_weight > 0
    weighted_precision = _mean_ratio(tp_weight, predicted_weight, weighted_covered)
    weighted_recall = _mean_ratio(tp_weight, true_weight, weighted_target)
    weighted_f = (
        2.0
        * weighted_precision
        * weighted_recall
        / (weighted_precision + weighted_recall)
        if weighted_precision + weighted_recall
        else 0.0
    )
    weighted_jaccard = _mean_ratio(tp_weight, union_weight, weighted_union)
    total_tp = float(tp_weight.sum())
    total_fp = float(fp_weight.sum())
    total_fn = float(fn_weight.sum())
    micro_weighted_precision = (
        total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    )
    micro_weighted_recall = (
        total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    )
    micro_weighted_f = (
        2.0
        * micro_weighted_precision
        * micro_weighted_recall
        / (micro_weighted_precision + micro_weighted_recall)
        if micro_weighted_precision + micro_weighted_recall
        else 0.0
    )
    result.update(
        {
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f": weighted_f,
            "weighted_jaccard_set_agreement": weighted_jaccard,
            "micro_weighted_precision": micro_weighted_precision,
            "micro_weighted_recall": micro_weighted_recall,
            "micro_weighted_f": micro_weighted_f,
        }
    )
    return result


def bootstrap_fixed_threshold(
    truth: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    if replicates < 1:
        return []
    n = truth.shape[0]
    if n < 1:
        return []

    true = truth.astype(bool, copy=False)
    predicted = scores >= threshold
    true_positive = np.logical_and(predicted, true)
    tp = true_positive.sum(axis=1, dtype=np.float64)
    predicted_count = predicted.sum(axis=1, dtype=np.float64)
    true_count = true.sum(axis=1, dtype=np.float64)
    union_count = np.logical_or(predicted, true).sum(axis=1, dtype=np.float64)

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.zeros_like(numerator, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return result

    components: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "precision": (ratio(tp, predicted_count), predicted_count > 0),
        "recall": (ratio(tp, true_count), true_count > 0),
        "jaccard_set_agreement": (ratio(tp, union_count), union_count > 0),
    }
    if np.any(weights > 0):
        tp_weight = true_positive @ weights
        predicted_weight = predicted @ weights
        true_weight = true @ weights
        union_weight = np.logical_or(predicted, true) @ weights
        components.update(
            {
                "weighted_precision": (
                    ratio(tp_weight, predicted_weight),
                    predicted_weight > 0,
                ),
                "weighted_recall": (
                    ratio(tp_weight, true_weight),
                    true_weight > 0,
                ),
                "weighted_jaccard_set_agreement": (
                    ratio(tp_weight, union_weight),
                    union_weight > 0,
                ),
            }
        )

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "f": [],
        "jaccard_set_agreement": [],
    }
    if "weighted_precision" in components:
        samples["weighted_f"] = []
        samples["weighted_jaccard_set_agreement"] = []

    probabilities = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(replicates):
        counts = rng.multinomial(n, probabilities).astype(np.float64, copy=False)

        def average(name: str) -> float:
            values, eligible = components[name]
            denominator = float(np.dot(counts, eligible))
            return (
                float(np.dot(counts, values) / denominator)
                if denominator
                else float("nan")
            )

        precision = average("precision")
        recall = average("recall")
        samples["f"].append(
            2.0 * precision * recall / (precision + recall)
            if math.isfinite(precision + recall) and precision + recall
            else 0.0
        )
        samples["jaccard_set_agreement"].append(average("jaccard_set_agreement"))
        if "weighted_precision" in components:
            weighted_precision = average("weighted_precision")
            weighted_recall = average("weighted_recall")
            samples["weighted_f"].append(
                2.0
                * weighted_precision
                * weighted_recall
                / (weighted_precision + weighted_recall)
                if math.isfinite(weighted_precision + weighted_recall)
                and weighted_precision + weighted_recall
                else 0.0
            )
            samples["weighted_jaccard_set_agreement"].append(
                average("weighted_jaccard_set_agreement")
            )

    point = threshold_metrics(truth, scores, weights, threshold)
    rows: list[dict[str, Any]] = []
    for metric_name, values in samples.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        rows.append(
            {
                "metric_name": metric_name,
                "estimate": point[metric_name],
                "ci_low": (float(np.quantile(finite, 0.025)) if finite.size else None),
                "ci_high": (float(np.quantile(finite, 0.975)) if finite.size else None),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
                "resampling_unit": "protein",
                "method": "percentile",
            }
        )
    return rows


def evaluate_bin(
    truth: np.ndarray,
    scores: np.ndarray,
    metric_weights: np.ndarray,
    term_indices: Sequence[int],
    fixed_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, float | int | None]]]:
    indices = np.asarray(term_indices, dtype=np.int64)
    if indices.size == 0:
        return {"status": "not_evaluable_no_terms"}, []
    population = truth.any(axis=1)
    if not population.any():
        return {"status": "not_evaluable_no_targets"}, []
    selected_truth = truth[population][:, indices]
    selected_scores = scores[population][:, indices]
    selected_weights = metric_weights[indices]
    if not selected_truth.any():
        return {"status": "not_evaluable_no_positive_targets"}, []
    thresholds = sorted(
        set(np.linspace(0.01, 0.99, 100).tolist() + [float(fixed_threshold)])
    )
    rows = [
        threshold_metrics(selected_truth, selected_scores, selected_weights, threshold)
        for threshold in thresholds
    ]
    weighted = bool(np.any(selected_weights > 0))
    best_unweighted_f = max(
        rows, key=lambda row: (float(row["f"] or 0.0), -float(row["threshold"]))
    )
    best_unweighted_jaccard = max(
        rows,
        key=lambda row: (
            float(row["jaccard_set_agreement"] or 0.0),
            -float(row["threshold"]),
        ),
    )
    best_weighted_f = (
        max(
            rows,
            key=lambda row: (
                float(row["weighted_f"] or 0.0),
                -float(row["threshold"]),
            ),
        )
        if weighted
        else None
    )
    best_weighted_jaccard = (
        max(
            rows,
            key=lambda row: (
                float(row["weighted_jaccard_set_agreement"] or 0.0),
                -float(row["threshold"]),
            ),
        )
        if weighted
        else None
    )
    fixed = min(rows, key=lambda row: abs(float(row["threshold"]) - fixed_threshold))
    return {
        "status": "complete",
        "weighted_metrics_available": weighted,
        "term_count": int(indices.size),
        "population_proteins": int(population.sum()),
        "target_positive_proteins": int(selected_truth.any(axis=1).sum()),
        "best_unweighted_f": best_unweighted_f,
        "best_unweighted_jaccard": best_unweighted_jaccard,
        "best_weighted_f": best_weighted_f,
        "best_weighted_jaccard": best_weighted_jaccard,
        "fixed_at_reference_threshold": fixed,
    }, rows


def _tsv(rows: list[dict[str, Any]], fields: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# PFP Specificity Analysis: {report['benchmark_id']} ({report['mode']})",
        "",
        "This flat post-evaluation analysis does not replace whole-truth CAFA evaluation.",
        "",
        "| Aspect | Specificity measure | Bin | Terms | Positive targets | Value range | Best unweighted F | Fixed-threshold unweighted F | Fixed-threshold IA-weighted F |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for aspect in report["selected_aspects"]:
        for measure_name, measure in report["aspects"][aspect]["measures"].items():
            for label, value in measure["bins"].items():
                if value["status"] != "complete":
                    lines.append(
                        f"| {aspect} | {measure_name} | {label} | "
                        f"{value.get('term_count', 0):,} | 0 | n/a | n/a | n/a | n/a |"
                    )
                    continue
                bounds = value["specificity"]
                value_range = f"{bounds['minimum']:.4f}-{bounds['maximum']:.4f}"
                best = value["metrics"]["best_unweighted_f"]
                fixed = value["metrics"]["fixed_at_reference_threshold"]
                weighted = fixed["weighted_f"]
                lines.append(
                    f"| {aspect} | {measure_name} | {label} | "
                    f"{value['term_count']:,} | "
                    f"{value['metrics']['target_positive_proteins']:,} | "
                    f"{value_range} | {best['f']:.4f} | {fixed['f']:.4f} | "
                    f"{weighted:.4f} |"
                    if weighted is not None
                    else (
                        f"| {aspect} | {measure_name} | {label} | "
                        f"{value['term_count']:,} | "
                        f"{value['metrics']['target_positive_proteins']:,} | "
                        f"{value_range} | {best['f']:.4f} | "
                        f"{fixed['f']:.4f} | n/a |"
                    )
                )
    lines.extend(
        [
            "",
            "- IA is the canonical CAFA weighting. Xu totipotency is reported only as a separate topology-derived stratifier.",
            "- Raw Xu T is lower for more specific terms; the optional -log2(T) panel is an exploratory display transform, not IA and not a formula proposed by Xu et al.",
            "- Roots are retained in the term table but excluded from flat bin metrics.",
            "- Bins are tie-preserving specificity quantiles. Zero IA and zero transformed Xu values are reported separately.",
            "- Precision includes false predictions on every whole-test evaluable protein, including proteins with no true term in that bin.",
            "- Recall is averaged over proteins with at least one true term in the bin.",
            "- Jaccard set agreement is named explicitly and is not ordinary true-negative accuracy.",
            "- Bin-specific optimum thresholds are descriptive. The fixed result uses one overall test-oracle threshold unless a separate validation operating point is supplied.",
            "",
        ]
    )
    return "\n".join(lines)


def selected_specificity_measures(values: Sequence[str]) -> list[str]:
    requested = list(values) or ["ia"]
    expanded: list[str] = []
    for value in requested:
        additions = (
            ["ia", "xu_totipotency_raw"]
            if value in {"all_separate", "both"}
            else [value]
        )
        for addition in additions:
            if addition not in expanded:
                expanded.append(addition)
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aspect", action="append", choices=ASPECTS, default=[])
    parser.add_argument("--positive-bins", type=int, default=4)
    parser.add_argument(
        "--specificity-measure",
        action="append",
        choices=(
            "ia",
            "xu_totipotency_raw",
            "xu_neglog_totipotency",
            "all_separate",
            "both",
        ),
        default=[],
    )
    parser.add_argument(
        "--obo",
        type=Path,
        help="Frozen GO OBO required for Xu totipotency panels.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Protein bootstrap repetitions at the fixed overall threshold.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    evaluation_path = args.evaluation_dir.resolve()
    output_dir = args.output_dir.resolve()
    evaluation, evaluation_root = load_evaluation_summary(evaluation_path)
    aspects = selected_aspects(args.aspect, list(evaluation["selected_aspects"]))
    measures = selected_specificity_measures(args.specificity_measure)
    requires_xu = any(value.startswith("xu_") for value in measures)
    xu_ontology = (
        read_xu_ontology(args.obo.resolve()) if requires_xu else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "analysis_id": f"{args.benchmark_id}__{evaluation['mode']}__specificity_flat",
        "analysis_kind": "pfp_information_content_analysis",
        "scientific_label": "flat_specificity_bin_analysis",
        "canonicality_label": "separate_from_whole_truth_cafa_evaluation",
        "analysis_policy": (
            "post-evaluation flat term-bin analysis; IA is the CAFA weighting "
            "and Xu values are separate topology-only stratifiers"
        ),
        "benchmark_id": args.benchmark_id,
        "mode": evaluation["mode"],
        "selected_aspects": aspects,
        "specificity_measures": measures,
        "positive_bin_count": args.positive_bins,
        "binning_policy": "tie-preserving equal-term quantiles; roots excluded",
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "unit": "protein",
            "method": "percentile",
        },
        "metric_definitions": {
            "primary": "unweighted protein-centric flat F and Jaccard set agreement",
            "weighted_f": (
                "harmonic mean of protein-centric IA-weighted precision and recall"
            ),
            "weighted_jaccard_set_agreement": (
                "mean per-protein IA-weighted intersection divided by union"
            ),
            "fixed_threshold_type": "descriptive_test_oracle_fixed",
            "fixed_threshold": "each model/aspect whole-test Fmax threshold",
            "optimized_threshold": (
                "descriptive subgroup oracle over 0.01-0.99 plus the fixed threshold"
            ),
            "binwise_smin": "not_computed_nonclosed_flat_term_subset",
        },
        "xu_ontology": (
            scientific_source(xu_ontology.source) if xu_ontology else None
        ),
        "aspects": {},
    }
    term_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for aspect in aspects:
        bundle = load_aspect_bundle(evaluation, evaluation_root, aspect)
        ia_values, ia_metadata = read_information_accretion(
            bundle["ia_path"], bundle["go_terms"]
        )
        ia_measure = SpecificityMeasure(
            name="ia",
            values=ia_values,
            higher_is_more_specific=True,
            zero_bin_label="zero_ia",
            source=scientific_source(ia_metadata),
        )
        measure_map: dict[str, SpecificityMeasure] = {"ia": ia_measure}
        xu_rows_by_term: dict[str, dict[str, Any]] = {}
        if xu_ontology is not None:
            raw_xu, neglog_xu, xu_rows = compute_xu_totipotency(
                xu_ontology, bundle["go_terms"], aspect
            )
            measure_map.update(
                {
                    "xu_totipotency_raw": raw_xu,
                    "xu_neglog_totipotency": neglog_xu,
                }
            )
            xu_rows_by_term = {row["go_id"]: row for row in xu_rows}

        whole_test_threshold = float(
            bundle["specification"]["canonical_cafa_metrics"]["threshold"]
        )
        aspect_report: dict[str, Any] = {
            "fixed_reference_threshold": whole_test_threshold,
            "fixed_threshold_type": "descriptive_test_oracle_fixed",
            "information_accretion": scientific_source(ia_metadata),
            "measures": {},
        }
        positive_counts = bundle["truth"].sum(axis=0, dtype=np.int64)
        assignments_by_measure: dict[str, list[str]] = {}
        for measure_name in measures:
            measure = measure_map[measure_name]
            bins, assignments = assign_measure_bins(
                bundle["go_terms"],
                measure,
                args.positive_bins,
                bundle["root_index"],
            )
            assignments_by_measure[measure_name] = assignments
            measure_report: dict[str, Any] = {
                "higher_is_more_specific": measure.higher_is_more_specific,
                "source": scientific_source(measure.source),
                "bins": {},
            }
            for bin_number, specification in enumerate(bins):
                label = specification["label"]
                metrics, rows = evaluate_bin(
                    bundle["truth"],
                    bundle["scores"],
                    ia_values,
                    specification["term_indices"],
                    whole_test_threshold,
                )
                if metrics["status"] == "complete":
                    selected_values = measure.values[specification["term_indices"]]
                    value = {
                        "status": "complete",
                        "term_count": specification["term_count"],
                        "specificity": {
                            "minimum": float(selected_values.min()),
                            "maximum": float(selected_values.max()),
                            "mean": float(selected_values.mean()),
                            "higher_is_more_specific": (
                                measure.higher_is_more_specific
                            ),
                        },
                        "metrics": metrics,
                    }
                    best = metrics["best_unweighted_f"]
                    best_weighted = metrics["best_weighted_f"]
                    fixed = metrics["fixed_at_reference_threshold"]
                    summary_rows.append(
                        {
                            "benchmark_id": args.benchmark_id,
                            "mode": evaluation["mode"],
                            "aspect": aspect,
                            "specificity_measure": measure_name,
                            "specificity_bin": label,
                            "term_count": specification["term_count"],
                            "target_positive_proteins": metrics[
                                "target_positive_proteins"
                            ],
                            "value_min": float(selected_values.min()),
                            "value_max": float(selected_values.max()),
                            "best_unweighted_f_threshold": best["threshold"],
                            "best_unweighted_f": best["f"],
                            "best_weighted_f_threshold": (
                                best_weighted["threshold"]
                                if best_weighted is not None
                                else None
                            ),
                            "best_weighted_f": (
                                best_weighted["weighted_f"]
                                if best_weighted is not None
                                else None
                            ),
                            "fixed_threshold_type": (
                                "descriptive_test_oracle_fixed"
                            ),
                            "fixed_threshold": fixed["threshold"],
                            "fixed_unweighted_f": fixed["f"],
                            "fixed_weighted_f": fixed["weighted_f"],
                            "fixed_unweighted_jaccard": fixed[
                                "jaccard_set_agreement"
                            ],
                            "fixed_weighted_jaccard": fixed[
                                "weighted_jaccard_set_agreement"
                            ],
                        }
                    )
                    for row in rows:
                        threshold_rows.append(
                            {
                                "benchmark_id": args.benchmark_id,
                                "mode": evaluation["mode"],
                                "aspect": aspect,
                                "specificity_measure": measure_name,
                                "specificity_bin": label,
                                "threshold_type": "subgroup_oracle_grid",
                                **row,
                            }
                        )
                    population = bundle["truth"].any(axis=1)
                    indices = np.asarray(
                        specification["term_indices"], dtype=np.int64
                    )
                    interval_rows = bootstrap_fixed_threshold(
                        bundle["truth"][population][:, indices],
                        bundle["scores"][population][:, indices],
                        ia_values[indices],
                        whole_test_threshold,
                        args.bootstrap_replicates,
                        args.bootstrap_seed
                        + (1000 * aspects.index(aspect))
                        + (100 * measures.index(measure_name))
                        + bin_number,
                    )
                    for interval in interval_rows:
                        bootstrap_rows.append(
                            {
                                "benchmark_id": args.benchmark_id,
                                "mode": evaluation["mode"],
                                "aspect": aspect,
                                "specificity_measure": measure_name,
                                "specificity_bin": label,
                                "threshold_type": ("descriptive_test_oracle_fixed"),
                                "threshold": whole_test_threshold,
                                **interval,
                            }
                        )
                else:
                    value = {
                        "status": metrics["status"],
                        "term_count": specification["term_count"],
                    }
                measure_report["bins"][label] = value
            aspect_report["measures"][measure_name] = measure_report

        for index, (term, ia_value, count) in enumerate(
            zip(
                bundle["go_terms"],
                ia_values.tolist(),
                positive_counts.tolist(),
            )
        ):
            xu = xu_rows_by_term.get(term, {})
            term_rows.append(
                {
                    "benchmark_id": args.benchmark_id,
                    "mode": evaluation["mode"],
                    "aspect": aspect,
                    "go_id": term,
                    "active": 1,
                    "root": int(index == bundle["root_index"]),
                    "ia": ia_value,
                    "ia_bin": assignments_by_measure.get(
                        "ia", [""] * len(bundle["go_terms"])
                    )[index],
                    "xu_totipotency_T": xu.get("xu_totipotency_T"),
                    "xu_neglog_totipotency": xu.get("xu_neglog_totipotency"),
                    "xu_bin": assignments_by_measure.get(
                        "xu_totipotency_raw",
                        assignments_by_measure.get(
                            "xu_neglog_totipotency",
                            [""] * len(bundle["go_terms"]),
                        ),
                    )[index],
                    "lower_is_more_specific": (1 if xu_rows_by_term else None),
                    "descendant_count": xu.get("descendant_count"),
                    "aspect_root_descendant_count": xu.get(
                        "aspect_root_descendant_count"
                    ),
                    "relationship_policy": (
                        "is_a+part_of" if xu_ontology else None
                    ),
                    "mapping_status": xu.get("mapping_status", "not_computed"),
                    "test_positive_proteins": int(count),
                    "training_positive_proteins": None,
                }
            )
        report["aspects"][aspect] = aspect_report

    (output_dir / "specificity_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "specificity_analysis.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    (output_dir / "term_specificity.tsv").write_text(
        _tsv(
            term_rows,
            (
                "benchmark_id",
                "mode",
                "aspect",
                "go_id",
                "active",
                "root",
                "ia",
                "ia_bin",
                "xu_totipotency_T",
                "xu_neglog_totipotency",
                "xu_bin",
                "lower_is_more_specific",
                "descendant_count",
                "aspect_root_descendant_count",
                "relationship_policy",
                "mapping_status",
                "test_positive_proteins",
                "training_positive_proteins",
            ),
        ),
        encoding="utf-8",
    )
    summary_fields = (
        "benchmark_id",
        "mode",
        "aspect",
        "specificity_measure",
        "specificity_bin",
        "term_count",
        "target_positive_proteins",
        "value_min",
        "value_max",
        "best_unweighted_f_threshold",
        "best_unweighted_f",
        "best_weighted_f_threshold",
        "best_weighted_f",
        "fixed_threshold_type",
        "fixed_threshold",
        "fixed_weighted_f",
        "fixed_unweighted_f",
        "fixed_weighted_jaccard",
        "fixed_unweighted_jaccard",
    )
    (output_dir / "specificity_bins.tsv").write_text(
        _tsv(summary_rows, summary_fields),
        encoding="utf-8",
    )
    threshold_fields = (
        "benchmark_id",
        "mode",
        "aspect",
        "specificity_measure",
        "specificity_bin",
        "threshold_type",
        "threshold",
        "population_proteins",
        "target_positive_proteins",
        "predicted_coverage",
        "precision",
        "recall",
        "f",
        "jaccard_set_agreement",
        "micro_precision",
        "micro_recall",
        "micro_f",
        "weighted_precision",
        "weighted_recall",
        "weighted_f",
        "weighted_jaccard_set_agreement",
        "micro_weighted_precision",
        "micro_weighted_recall",
        "micro_weighted_f",
    )
    (output_dir / "specificity_thresholds.tsv").write_text(
        _tsv(threshold_rows, threshold_fields),
        encoding="utf-8",
    )
    bootstrap_fields = (
        "benchmark_id",
        "mode",
        "aspect",
        "specificity_measure",
        "specificity_bin",
        "threshold_type",
        "threshold",
        "metric_name",
        "estimate",
        "ci_low",
        "ci_high",
        "bootstrap_replicates",
        "bootstrap_seed",
        "resampling_unit",
        "method",
    )
    (output_dir / "bootstrap_intervals.tsv").write_text(
        _tsv(bootstrap_rows, bootstrap_fields),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
