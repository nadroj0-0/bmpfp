#!/usr/bin/env python3
"""Run root-only sensitivity analyses on saved PFP predictions."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

from label_space_common import (
    ASPECTS,
    ASPECT_TO_ROOT,
)
from pfp_sensitivity_common import (
    cohort_masks,
    flat_non_root_metrics,
    load_aspect_bundle,
    load_evaluation_summary,
    run_cafa_evaluation,
)


def selected_aspects(values: list[str], available: list[str]) -> list[str]:
    return values or available


def tsv_text(rows: list[dict[str, Any]]) -> str:
    fields = ["benchmark_id", "mode", "aspect", "protein_id", "cohort"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# PFP Root-Only Sensitivity: {report['benchmark_id']} ({report['mode']})",
        "",
        "This sensitivity analysis does not replace or modify the whole-test evaluation.",
        "",
        "| Aspect | Captured rows | CAFA-evaluable | All-zero | Root-only | Eligible after exclusion | Whole-test Fmax | Excluded Fmax | At original threshold | Root baseline Fmax |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for aspect in report["selected_aspects"]:
        value = report["aspects"][aspect]
        excluded = value["root_only_excluded"]
        if excluded["status"] == "complete":
            excluded_f = f"{excluded['cafa']['fmax']:.4f}"
            fixed_f = f"{excluded['cafa']['fixed_at_canonical_threshold']['f']:.4f}"
        else:
            excluded_f = "n/a"
            fixed_f = "n/a"
        lines.append(
            "| {aspect} | {captured:,} | {evaluable:,} | {all_zero:,} | {root_only:,} | {eligible:,} | {canonical:.4f} | "
            "{excluded} | {fixed} | {baseline:.4f} |".format(
                aspect=aspect,
                captured=value["cohorts"]["captured_rows"],
                evaluable=value["cohorts"]["cafaeval_evaluable_targets"],
                all_zero=value["cohorts"]["all_zero"],
                root_only=value["cohorts"]["root_only"],
                eligible=value["cohorts"]["eligible_non_root"],
                canonical=value["canonical_evaluation"]["fmax"],
                excluded=excluded_f,
                fixed=fixed_f,
                baseline=value["root_only_prediction_baseline"]["fmax"],
            )
        )
    lines.extend(
        [
            "",
            "- `canonical_evaluation` records the metrics and operating threshold from the supplied whole-test evaluation.",
            "- `root_only_excluded` removes test proteins with no positive non-root term; it retrains nothing.",
            "- All-zero rows remain reported as benchmark rows but have no truth record and are not cafaeval targets.",
            "- The original-threshold result uses this run's whole-test Fmax threshold, while the excluded Fmax re-optimizes the threshold and is labelled accordingly.",
            "- `flat_non_root_diagnostic` removes the root column without ontology propagation and is not a CAFA metric.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--obo-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aspect", action="append", choices=ASPECTS, default=[])
    args = parser.parse_args()

    evaluation_path = args.evaluation_dir.resolve()
    obo_file = args.obo_file.resolve()
    output_dir = args.output_dir.resolve()
    evaluation, evaluation_root = load_evaluation_summary(evaluation_path)
    aspects = selected_aspects(args.aspect, list(evaluation["selected_aspects"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "analysis_policy": (
            "whole-test results remain primary; root-only exclusion is a sensitivity "
            "analysis with no retraining"
        ),
        "benchmark_id": args.benchmark_id,
        "mode": evaluation["mode"],
        "selected_aspects": aspects,
        "aspects": {},
    }
    cohort_rows: list[dict[str, Any]] = []
    for aspect in aspects:
        bundle = load_aspect_bundle(evaluation, evaluation_root, aspect)
        scores = bundle["scores"]
        truth = bundle["truth"]
        protein_ids = bundle["protein_ids"]
        go_terms = bundle["go_terms"]
        root_index = bundle["root_index"]
        masks = cohort_masks(truth, root_index)
        evaluable = ~masks["all_zero"]
        captured = bundle["specification"]["canonical_cafa_metrics"]
        canonical_threshold = float(captured["threshold"])

        eligible = masks["eligible_non_root"]
        if eligible.any():
            eligible_ids = [
                protein_id
                for protein_id, keep in zip(protein_ids, eligible.tolist())
                if keep
            ]
            excluded_cafa = run_cafa_evaluation(
                obo_file=obo_file,
                ia_file=bundle["ia_path"],
                destination=output_dir / aspect / "root_only_excluded",
                protein_ids=eligible_ids,
                go_terms=go_terms,
                truth=truth[eligible],
                fixed_threshold=canonical_threshold,
                scores=scores[eligible],
            )
            excluded = {"status": "complete", "cafa": excluded_cafa}
        else:
            excluded = {"status": "not_evaluable_no_non_root_targets"}

        root_baseline = run_cafa_evaluation(
            obo_file=obo_file,
            ia_file=bundle["ia_path"],
            destination=output_dir / aspect / "root_only_prediction_baseline",
            protein_ids=[
                protein_id
                for protein_id, keep in zip(protein_ids, evaluable.tolist())
                if keep
            ],
            go_terms=go_terms,
            truth=truth[evaluable],
            fixed_threshold=canonical_threshold,
            scores=None,
            root_index=root_index,
        )
        flat = flat_non_root_metrics(truth, scores, root_index, canonical_threshold)
        for row_index, protein_id in enumerate(protein_ids):
            if masks["root_only"][row_index]:
                cohort = "root-only"
            elif masks["eligible_non_root"][row_index]:
                cohort = "eligible-non-root"
            else:
                cohort = "all-zero"
            cohort_rows.append(
                {
                    "benchmark_id": args.benchmark_id,
                    "mode": evaluation["mode"],
                    "aspect": aspect,
                    "protein_id": protein_id,
                    "cohort": cohort,
                }
            )

        report["aspects"][aspect] = {
            "root": ASPECT_TO_ROOT[aspect],
            "cohorts": {
                "captured_rows": len(protein_ids),
                "cafaeval_evaluable_targets": int(evaluable.sum()),
                "eligible_non_root": int(masks["eligible_non_root"].sum()),
                "eligible_non_root_fraction": float(masks["eligible_non_root"].mean()),
                "root_only": int(masks["root_only"].sum()),
                "root_only_fraction_of_captured": float(masks["root_only"].mean()),
                "root_only_fraction_of_cafaeval_evaluable": (
                    float(masks["root_only"].sum() / evaluable.sum())
                    if evaluable.any()
                    else None
                ),
                "all_zero": int(masks["all_zero"].sum()),
            },
            "canonical_evaluation": captured,
            "root_only_excluded": excluded,
            "root_only_prediction_baseline": root_baseline,
            "flat_non_root_diagnostic": flat,
        }

    (output_dir / "root_exclusion_sensitivity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "root_exclusion_sensitivity.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    (output_dir / "target_cohorts.tsv").write_text(
        tsv_text(cohort_rows), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
