#!/usr/bin/env python3
"""Analyse NK+LK predictions and compare them with Global-NK results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ASPECT_TO_ONTOLOGY = {"BPO": "bp", "CCO": "cc", "MFO": "mf"}


def _load_helpers() -> tuple[Any, Any, Any, Any]:
    from analyse_pfp_information_content import (  # type: ignore[import-not-found]
        bootstrap_fixed_threshold,
        read_information_accretion,
        threshold_metrics,
    )
    from pfp_sensitivity_common import (  # type: ignore[import-not-found]
        load_aspect_bundle,
        load_evaluation_summary,
    )

    return (
        bootstrap_fixed_threshold,
        read_information_accretion,
        threshold_metrics,
        (load_evaluation_summary, load_aspect_bundle),
    )


def _load_membership(path: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = (row["ontology"], row["protein"])
            result[key] = row["cohort"]
    return result


def _interval_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["metric_name"]): row for row in rows}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _metric(result: dict[str, Any], name: str, context: str) -> float:
    del context
    return float(result[name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        type=Path,
        required=True,
        help="NK+LK evaluation directory or evaluation_summary.json.",
    )
    parser.add_argument("--membership-tsv", type=Path, required=True)
    parser.add_argument("--nk-evaluation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    (
        bootstrap_fixed_threshold,
        read_information_accretion,
        threshold_metrics,
        artifact_helpers,
    ) = _load_helpers()
    load_evaluation_summary, load_aspect_bundle = artifact_helpers
    evaluation_summary, artifact_root = load_evaluation_summary(args.evaluation.resolve())

    membership_path = args.membership_tsv.resolve()
    membership = _load_membership(membership_path)

    nk_summary_path = args.nk_evaluation_summary.resolve()
    nk_summary, _nk_root = load_evaluation_summary(nk_summary_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    report_aspects: dict[str, Any] = {}

    for aspect_number, aspect in enumerate(ASPECT_TO_ONTOLOGY):
        bundle = load_aspect_bundle(evaluation_summary, artifact_root, aspect)
        ontology = ASPECT_TO_ONTOLOGY[aspect]
        cohorts = []
        for protein_id in bundle["protein_ids"]:
            key = (ontology, protein_id)
            cohorts.append(membership[key])

        keep = np.ones(len(bundle["go_terms"]), dtype=bool)
        keep[bundle["root_index"]] = False
        truth = bundle["truth"][:, keep]
        scores = bundle["scores"][:, keep]
        ia, _ia_metadata = read_information_accretion(
            bundle["ia_path"], bundle["go_terms"]
        )
        weights = ia[keep]
        threshold = _metric(
            bundle["specification"]["canonical_cafa_metrics"],
            "threshold",
            f"NK+LK {aspect}",
        )
        has_nonroot = truth.any(axis=1)
        cohort_array = np.asarray(cohorts)

        masks = {
            "all": np.ones(len(cohorts), dtype=bool),
            "no-knowledge": cohort_array == "no-knowledge",
            "limited-knowledge": cohort_array == "limited-knowledge",
        }
        aspect_metrics: dict[str, Any] = {}
        for cohort_number, (cohort, cohort_mask) in enumerate(masks.items()):
            total = int(cohort_mask.sum())
            informative = int(np.logical_and(cohort_mask, has_nonroot).sum())
            root_only = total - informative
            composition = {
                "aspect": aspect,
                "cohort": cohort,
                "total_proteins": total,
                "root_only_proteins": root_only,
                "root_only_percent": 100.0 * root_only / total,
                "nonroot_proteins": informative,
                "nonroot_percent": 100.0 * informative / total,
            }
            composition_rows.append(composition)

            aspect_metrics[cohort] = {}
            for population_number, (population, mask) in enumerate(
                (
                    ("all-test", cohort_mask),
                    ("nonroot-only", np.logical_and(cohort_mask, has_nonroot)),
                )
            ):
                selected_truth = truth[mask]
                selected_scores = scores[mask]
                metrics = threshold_metrics(
                    selected_truth, selected_scores, weights, threshold
                )
                row = {
                    "aspect": aspect,
                    "cohort": cohort,
                    "population": population,
                    "threshold": threshold,
                    **metrics,
                }
                metric_rows.append(row)
                intervals = bootstrap_fixed_threshold(
                    selected_truth,
                    selected_scores,
                    weights,
                    threshold,
                    args.bootstrap_replicates,
                    args.bootstrap_seed
                    + aspect_number * 100
                    + cohort_number * 10
                    + population_number,
                )
                for interval in intervals:
                    bootstrap_rows.append(
                        {
                            "aspect": aspect,
                            "cohort": cohort,
                            "population": population,
                            "threshold": threshold,
                            **interval,
                        }
                    )
                aspect_metrics[cohort][population] = {
                    "metrics": metrics,
                    "bootstrap": _interval_index(intervals),
                }

        current = bundle["specification"]["canonical_cafa_metrics"]
        previous = nk_summary["aspects"][aspect]
        current_fmax = _metric(current, "fmax", f"NK+LK {aspect}")
        current_wfmax = _metric(current, "wfmax", f"NK+LK {aspect}")
        current_smin = _metric(current, "smin", f"NK+LK {aspect}")
        previous_fmax = _metric(previous, "cafa_fmax", f"Global-NK {aspect}")
        previous_wfmax = _metric(previous, "cafa_wfmax", f"Global-NK {aspect}")
        previous_smin = _metric(previous, "cafa_smin", f"Global-NK {aspect}")
        comparison = {
            "aspect": aspect,
            "nk_fmax": previous_fmax,
            "nk_lk_fmax": current_fmax,
            "fmax_delta": current_fmax - previous_fmax,
            "nk_wfmax": previous_wfmax,
            "nk_lk_wfmax": current_wfmax,
            "wfmax_delta": current_wfmax - previous_wfmax,
            "nk_smin": previous_smin,
            "nk_lk_smin": current_smin,
            "smin_delta": current_smin - previous_smin,
            "nk_wsmin": None,
            "nk_lk_wsmin": None,
            "wsmin_delta": None,
        }
        if previous.get("cafa_wsmin") is not None and current.get("wsmin") is not None:
            previous_wsmin = _metric(
                previous, "cafa_wsmin", f"Global-NK {aspect}"
            )
            current_wsmin = _metric(current, "wsmin", f"NK+LK {aspect}")
            comparison.update(
                {
                    "nk_wsmin": previous_wsmin,
                    "nk_lk_wsmin": current_wsmin,
                    "wsmin_delta": current_wsmin - previous_wsmin,
                }
            )
        comparison_rows.append(comparison)
        report_aspects[aspect] = {
            "threshold": threshold,
            "go_terms": len(bundle["go_terms"]),
            "composition": {
                row["cohort"]: row
                for row in composition_rows
                if row["aspect"] == aspect
            },
            "cohort_metrics": aspect_metrics,
            "nk_vs_nk_lk_evaluation": comparison,
        }

    def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = list(rows[0])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    write_tsv(output_dir / "cohort_composition.tsv", composition_rows)
    write_tsv(output_dir / "cohort_fixed_threshold_metrics.tsv", metric_rows)
    write_tsv(output_dir / "cohort_bootstrap_intervals.tsv", bootstrap_rows)
    write_tsv(output_dir / "nk_vs_nk_lk_evaluation_metrics.tsv", comparison_rows)

    report = {
        "analysis_kind": "contemporary_nk_lk_analysis",
        "mode": evaluation_summary["mode"],
        "nk_lk_evaluation_split": evaluation_summary["evaluation_split"],
        "nk_evaluation_split": nk_summary["evaluation_split"],
        "threshold_policy": (
            "one descriptive full-test CAFA-optimal threshold per aspect, held fixed "
            "for all NK/LK and all-test/nonroot-only subgroup comparisons"
        ),
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed_base": args.bootstrap_seed,
            "unit": "protein",
            "interval": "percentile 95%",
        },
        "interpretation_limits": [
            "NK+LK versus NK-only deltas compare the two supplied evaluation results and may combine model and split-population changes.",
            "Cohort metrics are flat fixed-threshold diagnostics, not independent CAFA evaluations.",
            "Limited knowledge means qualifying t0 knowledge in another ontology, not same-aspect partial knowledge.",
            "The thresholds are selected on the full test set and are descriptive, not deployable validation-fixed thresholds.",
        ],
        "aspects": report_aspects,
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Contemporary NK+LK Analysis",
        "",
        "## Supplied Evaluation Comparison",
        "",
        "| Aspect | NK Fmax | NK+LK Fmax | Delta | NK wFmax | NK+LK wFmax | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['aspect']} | {_fmt(row['nk_fmax'])} | {_fmt(row['nk_lk_fmax'])} | "
            f"{float(row['fmax_delta']):+.3f} | {_fmt(row['nk_wfmax'])} | "
            f"{_fmt(row['nk_lk_wfmax'])} | {float(row['wfmax_delta']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "| Aspect | NK unweighted s | NK+LK unweighted s | Delta | NK IA-weighted S_min | NK+LK IA-weighted S_min | Delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        wsmin_delta = (
            "NA"
            if row["wsmin_delta"] is None
            else f"{float(row['wsmin_delta']):+.3f}"
        )
        lines.append(
            f"| {row['aspect']} | {_fmt(row['nk_smin'])} | {_fmt(row['nk_lk_smin'])} | "
            f"{float(row['smin_delta']):+.3f} | {_fmt(row['nk_wsmin'])} | "
            f"{_fmt(row['nk_lk_wsmin'])} | {wsmin_delta} |"
        )
    lines.extend(
        [
            "",
            "Unweighted count-space s (`cafa_smin`) is shown for both evaluations. IA-weighted S_min is shown only when both supplied evaluations provide `cafa_wsmin`.",
            "",
            "These are CAFA metrics from the two supplied evaluation results. If those results came from separately retrained models and different split populations, the deltas combine both changes.",
            "",
            "## Test Composition",
            "",
            "| Aspect | Cohort | Total | Root-only | Root-only % | Non-root | Non-root % |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in composition_rows:
        lines.append(
            f"| {row['aspect']} | {row['cohort']} | {row['total_proteins']:,} | "
            f"{row['root_only_proteins']:,} | {_fmt(row['root_only_percent'], 2)}% | "
            f"{row['nonroot_proteins']:,} | {_fmt(row['nonroot_percent'], 2)}% |"
        )
    lines.extend(
        [
            "",
            "## Cohort Performance at the Shared Aspect Threshold",
            "",
            "The threshold is the single full-test CAFA-optimal threshold for that aspect and is not re-optimised for either cohort.",
            "",
            "| Aspect | Population | Cohort | N | F | 95% CI | wF | 95% CI |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for aspect in ASPECT_TO_ONTOLOGY:
        for population in ("all-test", "nonroot-only"):
            for cohort in ("no-knowledge", "limited-knowledge"):
                value = report_aspects[aspect]["cohort_metrics"][cohort][population]
                metrics = value["metrics"]
                intervals = value["bootstrap"]
                f_ci = intervals["f"]
                wf_ci = intervals["weighted_f"]
                lines.append(
                    f"| {aspect} | {population} | {cohort} | "
                    f"{metrics['population_proteins']:,} | {_fmt(metrics['f'])} | "
                    f"[{_fmt(f_ci['ci_low'])}, {_fmt(f_ci['ci_high'])}] | "
                    f"{_fmt(metrics['weighted_f'])} | "
                    f"[{_fmt(wf_ci['ci_low'])}, {_fmt(wf_ci['ci_high'])}] |"
                )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Limited knowledge means prior qualifying knowledge in another ontology only; it is not same-aspect annotation extension.",
            "- Root-only and non-root-only results answer different questions and must remain visible separately.",
            "- Cohort results use a shared threshold for a fair within-aspect comparison; they are diagnostics rather than new Fmax claims.",
            "- Bootstrap intervals quantify protein-sampling uncertainty only, not training-seed uncertainty.",
            "",
        ]
    )
    (output_dir / "NK+LK Contemporary Benchmark Analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
