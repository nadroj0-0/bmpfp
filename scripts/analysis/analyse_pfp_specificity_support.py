#!/usr/bin/env python3
"""Test whether PFP's specificity decline remains at comparable training support."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as ssp  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pfp_sensitivity_common import (  # noqa: E402
    load_aspect_bundle,
    load_evaluation_summary,
)


ASPECTS = ("BPO", "CCO", "MFO")
Q1 = "specificity_q1"
Q4 = "specificity_q4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        required=True,
        type=Path,
        help="Evaluation directory or evaluation_summary.json.",
    )
    parser.add_argument("--prepared-data-dir", required=True, type=Path)
    parser.add_argument("--specificity-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-bins", type=int, default=4)
    parser.add_argument("--minimum-cell-terms", type=int, default=10)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def tsv_text(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue()


def write_gzip_tsv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_specificity_rows(path: Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    rows = read_tsv(path)
    by_aspect: dict[str, list[dict[str, Any]]] = {aspect: [] for aspect in ASPECTS}
    for raw in rows:
        aspect = raw["aspect"]
        if aspect not in by_aspect:
            continue
        row = dict(raw)
        row.update(
            {
                "root": int(raw["root"]),
                "xu_totipotency_T": float(raw["xu_totipotency_T"]),
                "lower_is_more_specific": int(raw["lower_is_more_specific"]),
                "test_positive_proteins": int(raw["test_positive_proteins"]),
            }
        )
        by_aspect[aspect].append(row)
    benchmark_ids = {raw["benchmark_id"] for raw in rows}
    return by_aspect, next(iter(benchmark_ids))


def candidate_support_bins(
    support: np.ndarray, candidate_bins: int
) -> list[tuple[int, int]]:
    if support.size == 0:
        return []
    low = int(support.min())
    high = int(support.max())
    if low == high or candidate_bins == 1:
        return [(low, high)]
    quantiles = np.linspace(0.0, 1.0, candidate_bins + 1)[1:-1]
    cuts = sorted(
        {
            int(value)
            for value in np.quantile(support, quantiles, method="higher").tolist()
            if low <= int(value) < high
        }
    )
    bins: list[tuple[int, int]] = []
    lower = low
    for cut in cuts:
        bins.append((lower, cut))
        lower = cut + 1
    bins.append((lower, high))
    return [value for value in bins if value[0] <= value[1]]


def bin_counts(
    rows: Sequence[Mapping[str, Any]], lower: int, upper: int
) -> tuple[int, int]:
    selected = [
        row
        for row in rows
        if lower <= int(row["training_positive_proteins"]) <= upper
    ]
    return (
        sum(row["xu_bin"] == Q1 for row in selected),
        sum(row["xu_bin"] == Q4 for row in selected),
    )


def build_support_overlap(
    rows: Sequence[Mapping[str, Any]], candidate_bins: int, minimum_cell_terms: int
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if not row["root"]
        and row["test_positive_proteins"] >= 1
        and row["xu_bin"] in {Q1, Q4}
    ]
    q1 = [int(row["training_positive_proteins"]) for row in eligible if row["xu_bin"] == Q1]
    q4 = [int(row["training_positive_proteins"]) for row in eligible if row["xu_bin"] == Q4]
    base: dict[str, Any] = {
        "status": "insufficient_overlap",
        "selection": "non-root Xu Q1/Q4 terms with N_test_positive >= 1",
        "candidate_bin_count": candidate_bins,
        "minimum_terms_per_quartile": minimum_cell_terms,
        "candidate_method": (
            "pooled Q1+Q4 support quantiles within common inclusive support range; "
            "ties retained; sparse adjacent bins merged from low to high"
        ),
        "q1_term_count": len(q1),
        "q4_term_count": len(q4),
        "bins": [],
    }
    if not q1 or not q4:
        base["reason"] = "Q1 or Q4 has no eligible terms"
        return base
    common_low = max(min(q1), min(q4))
    common_high = min(max(q1), max(q4))
    base["common_support_min"] = common_low
    base["common_support_max"] = common_high
    if common_low > common_high:
        base["reason"] = "Q1 and Q4 training-support ranges do not overlap"
        return base
    common = [
        row
        for row in eligible
        if common_low <= int(row["training_positive_proteins"]) <= common_high
    ]
    total_q1, total_q4 = bin_counts(common, common_low, common_high)
    base["common_q1_term_count"] = total_q1
    base["common_q4_term_count"] = total_q4
    if total_q1 < minimum_cell_terms or total_q4 < minimum_cell_terms:
        base["reason"] = "Common support contains too few Q1 or Q4 terms"
        return base
    supports = np.asarray(
        [int(row["training_positive_proteins"]) for row in common], dtype=np.int64
    )
    candidates = candidate_support_bins(supports, candidate_bins)
    merged: list[tuple[int, int]] = []
    pending_low: int | None = None
    pending_high: int | None = None
    for lower, upper in candidates:
        pending_low = lower if pending_low is None else pending_low
        pending_high = upper
        count_q1, count_q4 = bin_counts(common, pending_low, pending_high)
        if count_q1 >= minimum_cell_terms and count_q4 >= minimum_cell_terms:
            merged.append((pending_low, pending_high))
            pending_low = None
            pending_high = None
    if pending_low is not None and pending_high is not None:
        if merged:
            previous_low, _previous_high = merged.pop()
            merged.append((previous_low, pending_high))
        else:
            merged.append((pending_low, pending_high))
    final_bins: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(merged, start=1):
        count_q1, count_q4 = bin_counts(common, lower, upper)
        if count_q1 < minimum_cell_terms or count_q4 < minimum_cell_terms:
            base["reason"] = "Deterministic merging could not form an adequate bin"
            return base
        final_bins.append(
            {
                "label": f"support_bin_{index}",
                "lower_inclusive": lower,
                "upper_inclusive": upper,
                "q1_term_count": count_q1,
                "q4_term_count": count_q4,
            }
        )
    base.update({"status": "complete", "bins": final_bins})
    return base


def term_metrics(truth: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    true = truth.astype(bool, copy=False)
    predicted = scores >= threshold
    tp = np.logical_and(predicted, true).sum(axis=0, dtype=np.int64)
    fp = np.logical_and(predicted, ~true).sum(axis=0, dtype=np.int64)
    fn = np.logical_and(~predicted, true).sum(axis=0, dtype=np.int64)
    precision = np.divide(
        tp, tp + fp, out=np.zeros(tp.shape, dtype=np.float64), where=(tp + fp) > 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.zeros(tp.shape, dtype=np.float64), where=(tp + fn) > 0
    )
    denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros(tp.shape, dtype=np.float64),
        where=denominator > 0,
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else None


def quantiles(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    q1, median, q3 = np.quantile(np.asarray(values), [0.25, 0.5, 0.75]).tolist()
    return float(q1), float(median), float(q3)


def summarize_q1_q4(
    rows: Sequence[Mapping[str, Any]],
    aspect: str,
    minimum_test_positives: int,
    label: str,
    lower: int | None,
    upper: int | None,
    minimum_cell_terms: int,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if not row["root"]
        and row["test_positive_proteins"] >= minimum_test_positives
        and row["xu_bin"] in {Q1, Q4}
        and (lower is None or int(row["training_positive_proteins"]) >= lower)
        and (upper is None or int(row["training_positive_proteins"]) <= upper)
    ]
    broad = [float(row["term_f1"]) for row in selected if row["xu_bin"] == Q1]
    specific = [float(row["term_f1"]) for row in selected if row["xu_bin"] == Q4]
    q1_low, q1_median, q1_high = quantiles(broad)
    q4_low, q4_median, q4_high = quantiles(specific)
    return {
        "aspect": aspect,
        "minimum_test_positives": minimum_test_positives,
        "support_stratum": label,
        "support_lower_inclusive": lower,
        "support_upper_inclusive": upper,
        "q1_broad_term_count": len(broad),
        "q1_broad_f1_q25": q1_low,
        "q1_broad_f1_median": q1_median,
        "q1_broad_f1_q75": q1_high,
        "q4_specific_term_count": len(specific),
        "q4_specific_f1_q25": q4_low,
        "q4_specific_f1_median": q4_median,
        "q4_specific_f1_q75": q4_high,
        "q4_minus_q1_median_f1": (
            q4_median - q1_median
            if q1_median is not None and q4_median is not None
            else None
        ),
        "meets_original_minimum_cell_rule": (
            len(broad) >= minimum_cell_terms and len(specific) >= minimum_cell_terms
        ),
    }


def plot_support(rows: Sequence[Mapping[str, Any]], path_base: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    colors = {Q1: "#2166ac", Q4: "#b2182b"}
    for axis, aspect in zip(axes, ASPECTS):
        selected = [
            row
            for row in rows
            if row["aspect"] == aspect and not row["root"] and row["test_positive_proteins"] >= 1
        ]
        other = [row for row in selected if row["xu_bin"] not in colors]
        axis.scatter(
            [row["log10_training_positive_plus_one"] for row in other],
            [row["term_f1"] for row in other],
            s=9,
            alpha=0.18,
            color="#777777",
            linewidths=0,
            label="Xu Q2/Q3",
        )
        for group, label in ((Q1, "Broad Q1"), (Q4, "Specific Q4")):
            values = [row for row in selected if row["xu_bin"] == group]
            axis.scatter(
                [row["log10_training_positive_plus_one"] for row in values],
                [row["term_f1"] for row in values],
                s=11,
                alpha=0.4,
                color=colors[group],
                linewidths=0,
                label=label,
            )
        rho = safe_spearman(
            [row["training_positive_proteins"] for row in selected],
            [row["term_f1"] for row in selected],
        )
        axis.set_title(f"{aspect}  support-F1 rho={rho:.3f}" if rho is not None else aspect)
        axis.set_xlabel("log10(training positives + 1)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Term-level F1")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(path_base.with_suffix(".png"), dpi=180)
    figure.savefig(path_base.with_suffix(".pdf"))
    plt.close(figure)


def plot_strata(rows: Sequence[Mapping[str, Any]], path_base: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for axis, aspect in zip(axes, ASPECTS):
        selected = [
            row
            for row in rows
            if row["aspect"] == aspect
            and row["minimum_test_positives"] == 1
            and str(row["support_stratum"]).startswith("support_bin_")
        ]
        if not selected:
            axis.text(0.5, 0.5, "Insufficient common support", ha="center", va="center")
            axis.set_title(aspect)
            axis.set_axis_off()
            continue
        x = np.arange(len(selected))
        for offset, prefix, label, color in (
            (-0.12, "q1_broad", "Broad Q1", "#2166ac"),
            (0.12, "q4_specific", "Specific Q4", "#b2182b"),
        ):
            medians = np.asarray([row[f"{prefix}_f1_median"] for row in selected])
            lower = medians - np.asarray([row[f"{prefix}_f1_q25"] for row in selected])
            upper = np.asarray([row[f"{prefix}_f1_q75"] for row in selected]) - medians
            axis.errorbar(
                x + offset,
                medians,
                yerr=np.vstack([lower, upper]),
                fmt="o",
                capsize=3,
                color=color,
                label=label,
            )
        axis.set_xticks(x, [str(row["support_stratum"]).replace("support_bin_", "Bin ") for row in selected])
        axis.set_title(aspect)
        axis.set_xlabel("Frozen training-support stratum")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Median term-level F1 (IQR)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(path_base.with_suffix(".png"), dpi=180)
    figure.savefig(path_base.with_suffix(".pdf"))
    plt.close(figure)


def markdown_report(
    report: Mapping[str, Any], correlations: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]
) -> str:
    def number(value: Any, *, signed: bool = False) -> str:
        if value is None:
            return "NA"
        return f"{float(value):+.3f}" if signed else f"{float(value):.3f}"

    lines = [
        f"# {report.get('benchmark_id', 'PFP Benchmark')} Specificity and Training Support",
        "",
        "This analysis asks whether the broad-to-specific decline in PFP term-level F1 remains when Xu Q1 and Q4 terms receive comparable positive training support.",
        "",
        "## Analysis design",
        "",
        "- Existing full-model predictions only; no retraining or inference.",
        "- Xu topology-based specificity is primary. Lower raw Xu T means greater specificity.",
        "- Support strata were fixed without using F1 and reused unchanged for the N_test+ >= 5 check.",
        "- Term-level F1 is descriptive and is not the protein-centric CAFA F-max.",
        "",
        "## Correlations",
        "",
        "| Aspect | N_test+ filter | Terms | Support-F1 rho | Xu T-support rho |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in correlations:
        support_rho = "NA" if row["support_f1_spearman_rho"] is None else f"{row['support_f1_spearman_rho']:.3f}"
        xu_rho = "NA" if row["xu_support_spearman_rho"] is None else f"{row['xu_support_spearman_rho']:.3f}"
        lines.append(
            f"| {row['aspect']} | >= {row['minimum_test_positives']} | {row['term_count']} | {support_rho} | {xu_rho} |"
        )
    lines.extend(
        [
            "",
            "## Support-matched Q1/Q4 comparison",
            "",
            "| Aspect | N_test+ filter | Stratum | Q1 n | Q1 median F1 | Q4 n | Q4 median F1 | Q4-Q1 | Adequate |",
            "|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summaries:
        if not str(row["support_stratum"]).startswith("support_bin_"):
            continue
        q1 = row["q1_broad_f1_median"]
        q4 = row["q4_specific_f1_median"]
        gap = row["q4_minus_q1_median_f1"]
        lines.append(
            f"| {row['aspect']} | >= {row['minimum_test_positives']} | {row['support_stratum']} "
            f"[{row['support_lower_inclusive']}, {row['support_upper_inclusive']}] | "
            f"{row['q1_broad_term_count']} | {number(q1)} | {row['q4_specific_term_count']} | "
            f"{number(q4)} | {number(gap, signed=True)} | {row['meets_original_minimum_cell_rule']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "A remaining gap is a residual descriptive association after approximate control for positive training support. It does not establish that specificity causes poorer performance. GO branch, annotation quality, test prevalence and functional heterogeneity may still differ.",
            "",
        ]
    )
    for aspect in report["aspects"]:
        overlap = report["aspects"][aspect]["support_overlap"]
        if overlap["status"] != "complete":
            lines.append(f"- {aspect}: insufficient overlap ({overlap.get('reason', 'unspecified')}).")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    evaluation_path = args.evaluation.expanduser().resolve()
    evaluation, evaluation_root = load_evaluation_summary(evaluation_path)
    aspects = [aspect for aspect in ASPECTS if aspect in evaluation["aspects"]]

    prepared_dir = args.prepared_data_dir.expanduser().resolve()
    specificity_dir = args.specificity_dir.expanduser().resolve()
    specificity_path = specificity_dir / "term_specificity.tsv"
    specificity_by_aspect, benchmark_id = load_specificity_rows(specificity_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    term_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "analysis_kind": "specificity_support_analysis",
        "benchmark_id": benchmark_id,
        "mode": evaluation["mode"],
        "scientific_question": (
            "Does the Xu broad-to-specific term-level F1 decline remain among GO terms "
            "with comparable positive training support?"
        ),
        "primary_specificity_measure": "xu_totipotency_raw",
        "xu_orientation": "lower raw T is more specific; Q1 broad and Q4 specific",
        "candidate_bin_count": args.candidate_bins,
        "minimum_terms_per_quartile": args.minimum_cell_terms,
        "test_positive_filters": [1, 5],
        "aspects": {},
    }
    support_rows: dict[str, list[dict[str, Any]]] = {}
    for aspect in aspects:
        go_terms = [str(value) for value in read_json(prepared_dir / f"{aspect}_go_terms.json")]
        train_labels = ssp.load_npz(prepared_dir / f"{aspect}_train_labels.npz").tocsr()
        train_counts = np.asarray(train_labels.sum(axis=0)).ravel().astype(np.int64)
        specificity_rows = specificity_by_aspect[aspect]
        joined: list[dict[str, Any]] = []
        for index, row in enumerate(specificity_rows):
            value = {
                **row,
                "training_positive_proteins": int(train_counts[index]),
                "log10_training_positive_plus_one": float(
                    math.log10(int(train_counts[index]) + 1)
                ),
            }
            joined.append(value)
        overlap = build_support_overlap(
            joined, args.candidate_bins, args.minimum_cell_terms
        )
        support_rows[aspect] = joined
        report["aspects"][aspect] = {
            "training_proteins": int(train_labels.shape[0]),
            "prediction_terms": len(go_terms),
            "zero_training_support_terms": int(np.sum(train_counts == 0)),
            "support_overlap": overlap,
        }

    for aspect in aspects:
        bundle = load_aspect_bundle(evaluation, evaluation_root, aspect)
        joined = support_rows[aspect]
        threshold = float(
            bundle["specification"]["canonical_cafa_metrics"]["threshold"]
        )
        metrics = term_metrics(bundle["truth"], bundle["scores"], threshold)
        for index, row in enumerate(joined):
            term_rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "aspect": aspect,
                        "go_id": row["go_id"],
                        "root": row["root"],
                        "training_positive_proteins": row["training_positive_proteins"],
                        "log10_training_positive_plus_one": row[
                            "log10_training_positive_plus_one"
                        ],
                        "test_positive_proteins": row["test_positive_proteins"],
                        "fixed_aspect_threshold": threshold,
                        "true_positives": int(metrics["tp"][index]),
                        "false_positives": int(metrics["fp"][index]),
                        "false_negatives": int(metrics["fn"][index]),
                        "term_precision": float(metrics["precision"][index]),
                        "term_recall": float(metrics["recall"][index]),
                        "term_f1": float(metrics["f1"][index]),
                        "xu_totipotency_T": row["xu_totipotency_T"],
                        "xu_bin": row["xu_bin"],
                        "xu_group": (
                            "broad_q1"
                            if row["xu_bin"] == Q1
                            else "specific_q4"
                            if row["xu_bin"] == Q4
                            else "middle_q2_q3"
                        ),
                    }
            )
        report["aspects"][aspect]["fixed_aspect_threshold"] = threshold

    for aspect in aspects:
        aspect_rows = [row for row in term_rows if row["aspect"] == aspect]
        overlap = report["aspects"][aspect]["support_overlap"]
        for minimum_test in (1, 5):
            eligible = [
                    row
                    for row in aspect_rows
                    if not row["root"] and row["test_positive_proteins"] >= minimum_test
                ]
            correlation_rows.append(
                    {
                        "aspect": aspect,
                        "minimum_test_positives": minimum_test,
                        "term_count": len(eligible),
                        "zero_training_support_terms": sum(
                            row["training_positive_proteins"] == 0 for row in eligible
                        ),
                        "support_f1_spearman_rho": safe_spearman(
                            [row["training_positive_proteins"] for row in eligible],
                            [row["term_f1"] for row in eligible],
                        ),
                        "xu_support_spearman_rho": safe_spearman(
                            [row["xu_totipotency_T"] for row in eligible],
                            [row["training_positive_proteins"] for row in eligible],
                        ),
                    }
            )
            summary_rows.append(
                    summarize_q1_q4(
                        aspect_rows,
                        aspect,
                        minimum_test,
                        "unadjusted",
                        None,
                        None,
                        args.minimum_cell_terms,
                    )
            )
            if overlap["status"] == "complete":
                summary_rows.append(
                        summarize_q1_q4(
                            aspect_rows,
                            aspect,
                            minimum_test,
                            "common_support_all",
                            int(overlap["common_support_min"]),
                            int(overlap["common_support_max"]),
                            args.minimum_cell_terms,
                        )
                )
                for specification in overlap["bins"]:
                    summary_rows.append(
                            summarize_q1_q4(
                                aspect_rows,
                                aspect,
                                minimum_test,
                                str(specification["label"]),
                                int(specification["lower_inclusive"]),
                                int(specification["upper_inclusive"]),
                                args.minimum_cell_terms,
                            )
                    )

    term_fields = (
        "benchmark_id",
        "aspect",
        "go_id",
        "root",
        "training_positive_proteins",
        "log10_training_positive_plus_one",
        "test_positive_proteins",
        "fixed_aspect_threshold",
        "true_positives",
        "false_positives",
        "false_negatives",
        "term_precision",
        "term_recall",
        "term_f1",
        "xu_totipotency_T",
        "xu_bin",
        "xu_group",
    )
    correlation_fields = (
        "aspect",
        "minimum_test_positives",
        "term_count",
        "zero_training_support_terms",
        "support_f1_spearman_rho",
        "xu_support_spearman_rho",
    )
    summary_fields = (
        "aspect",
        "minimum_test_positives",
        "support_stratum",
        "support_lower_inclusive",
        "support_upper_inclusive",
        "q1_broad_term_count",
        "q1_broad_f1_q25",
        "q1_broad_f1_median",
        "q1_broad_f1_q75",
        "q4_specific_term_count",
        "q4_specific_f1_q25",
        "q4_specific_f1_median",
        "q4_specific_f1_q75",
        "q4_minus_q1_median_f1",
        "meets_original_minimum_cell_rule",
    )
    write_gzip_tsv(
        output_dir / "term_level_specificity_support.tsv.gz", term_rows, term_fields
    )
    (output_dir / "correlations.tsv").write_text(
        tsv_text(correlation_rows, correlation_fields), encoding="utf-8"
    )
    (output_dir / "support_stratified_q1_q4.tsv").write_text(
        tsv_text(summary_rows, summary_fields), encoding="utf-8"
    )
    plot_support(term_rows, output_dir / "support_vs_term_f1")
    plot_strata(summary_rows, output_dir / "support_matched_q1_q4")
    report["correlations"] = correlation_rows
    report["support_stratified_q1_q4"] = summary_rows
    (output_dir / "specificity_support_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "specificity_support_report.md").write_text(
        markdown_report(report, correlation_rows, summary_rows), encoding="utf-8"
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
