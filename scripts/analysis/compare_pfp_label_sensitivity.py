#!/usr/bin/env python3
"""Compare root-only sensitivity reports across modes and benchmarks."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

from label_space_common import ASPECTS


BASELINE_MODE = "sequence-only"


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any, label: str) -> float:
    del label
    return float(value)


def tsv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(rows[0]),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# PFP Label-Sensitivity Comparison",
        "",
        "Whole-test results remain primary. All exclusions and mode deltas below are sensitivity analyses.",
        "",
        "## Run Profiles",
        "",
        "| Benchmark | Mode | Aspect | Captured | CAFA-evaluable | All-zero | Root-only % of evaluable | Whole-test threshold | Whole-test Fmax | Excluded Fmax | At original threshold | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["run_rows"]:
        excluded_fmax = (
            f"{row['root_excluded_fmax']:.4f}"
            if row["root_excluded_fmax"] is not None
            else "n/a"
        )
        original_threshold_f = (
            f"{row['root_excluded_at_mode_canonical_threshold_f']:.4f}"
            if row["root_excluded_at_mode_canonical_threshold_f"] is not None
            else "n/a"
        )
        root_fraction = row["root_only_fraction_of_cafaeval_evaluable"]
        lines.append(
            "| {benchmark_id} | {mode} | {aspect} | {captured_rows:,} | "
            "{cafaeval_evaluable_targets:,} | {all_zero:,} | {root_fraction} | "
            "{canonical_threshold:.4f} | {canonical_fmax:.4f} | {excluded_fmax} | "
            "{original_threshold_f} | {exclusion_status} |".format(
                **row,
                root_fraction=(
                    f"{root_fraction:.2%}" if root_fraction is not None else "n/a"
                ),
                excluded_fmax=excluded_fmax,
                original_threshold_f=original_threshold_f,
            )
        )
    lines.extend(
        [
            "",
            "## Mode Minus Sequence-Only",
            "",
            "| Benchmark | Mode | Aspect | Mode threshold | Sequence threshold | Whole-test delta | Excluded optimized delta | Delta at each mode's original threshold | Status |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    if report["mode_delta_rows"]:
        for row in report["mode_delta_rows"]:
            excluded_delta = (
                f"{row['root_excluded_fmax_delta']:.4f}"
                if row["root_excluded_fmax_delta"] is not None
                else "n/a"
            )
            original_delta = (
                f"{row['root_excluded_at_mode_canonical_threshold_delta']:.4f}"
                if row["root_excluded_at_mode_canonical_threshold_delta"] is not None
                else "n/a"
            )
            lines.append(
                "| {benchmark_id} | {comparison_mode} | {aspect} | "
                "{mode_canonical_threshold:.4f} | "
                "{baseline_canonical_threshold:.4f} | {canonical_fmax_delta:.4f} | "
                "{excluded_delta} | {original_delta} | {status} |".format(
                    **row, excluded_delta=excluded_delta, original_delta=original_delta
                )
            )
    else:
        lines.append("| No benchmark has a sequence-only baseline and comparison mode yet |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "The original-threshold value avoids retuning after cohort exclusion within each mode. Each comparison mode and sequence-only may have different original thresholds, which are shown explicitly; this is not a shared-threshold comparison.",
            "These results compare model modes only within the same benchmark and aspect; they are not cross-benchmark model rankings.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    reports = [load_report(path.resolve()) for path in args.report]

    run_rows: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for report in reports:
        for aspect in report["selected_aspects"]:
            value = report["aspects"][aspect]
            excluded = value["root_only_excluded"]
            excluded_cafa = excluded.get("cafa")
            cohorts = value["cohorts"]
            row = {
                "benchmark_id": report["benchmark_id"],
                "mode": report["mode"],
                "aspect": aspect,
                "captured_rows": cohorts["captured_rows"],
                "cafaeval_evaluable_targets": cohorts["cafaeval_evaluable_targets"],
                "all_zero": cohorts["all_zero"],
                "eligible_non_root": cohorts["eligible_non_root"],
                "root_only": cohorts["root_only"],
                "root_only_fraction_of_cafaeval_evaluable": cohorts[
                    "root_only_fraction_of_cafaeval_evaluable"
                ],
                "canonical_fmax": finite_number(
                    value["canonical_evaluation"]["fmax"],
                    f"{report['benchmark_id']}/{report['mode']}/{aspect}/fmax",
                ),
                "canonical_threshold": finite_number(
                    value["canonical_evaluation"]["threshold"],
                    f"{report['benchmark_id']}/{report['mode']}/{aspect}/threshold",
                ),
                "exclusion_status": (
                    "complete" if excluded_cafa is not None else "unavailable"
                ),
                "root_excluded_fmax": (
                    finite_number(
                        excluded["cafa"]["fmax"],
                        f"{report['benchmark_id']}/{report['mode']}/{aspect}/excluded-fmax",
                    )
                    if excluded_cafa is not None
                    else None
                ),
                "root_excluded_at_mode_canonical_threshold_f": (
                    finite_number(
                        excluded["cafa"]["fixed_at_canonical_threshold"]["f"],
                        f"{report['benchmark_id']}/{report['mode']}/{aspect}/excluded-fixed-f",
                    )
                    if excluded_cafa is not None
                    else None
                ),
                "root_baseline_fmax": finite_number(
                    value["root_only_prediction_baseline"]["fmax"],
                    f"{report['benchmark_id']}/{report['mode']}/{aspect}/root-baseline-fmax",
                ),
            }
            row["canonical_minus_root_baseline"] = (
                row["canonical_fmax"] - row["root_baseline_fmax"]
            )
            run_rows.append(row)
            indexed[(row["benchmark_id"], row["mode"], aspect)] = row

    delta_rows = []
    for benchmark_id in sorted({row["benchmark_id"] for row in run_rows}):
        for aspect in ASPECTS:
            baseline = indexed.get((benchmark_id, BASELINE_MODE, aspect))
            if baseline is None:
                continue
            comparison_modes = sorted(
                {
                    row["mode"]
                    for row in run_rows
                    if row["benchmark_id"] == benchmark_id
                    and row["mode"] != BASELINE_MODE
                }
            )
            for comparison_mode in comparison_modes:
                candidate = indexed.get((benchmark_id, comparison_mode, aspect))
                if candidate is None:
                    continue
                complete = (
                    candidate["root_excluded_fmax"] is not None
                    and baseline["root_excluded_fmax"] is not None
                )
                comparison_status = (
                    "complete"
                    if complete
                    else (
                        f"candidate={candidate['exclusion_status']};"
                        f"baseline={baseline['exclusion_status']}"
                    )
                )
                delta_rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "comparison_mode": comparison_mode,
                        "baseline_mode": BASELINE_MODE,
                        "aspect": aspect,
                        "canonical_fmax_delta": candidate["canonical_fmax"]
                        - baseline["canonical_fmax"],
                        "mode_canonical_threshold": candidate["canonical_threshold"],
                        "baseline_canonical_threshold": baseline["canonical_threshold"],
                        "root_excluded_fmax_delta": (
                            candidate["root_excluded_fmax"]
                            - baseline["root_excluded_fmax"]
                            if complete
                            else None
                        ),
                        "root_excluded_at_mode_canonical_threshold_delta": (
                            candidate["root_excluded_at_mode_canonical_threshold_f"]
                            - baseline["root_excluded_at_mode_canonical_threshold_f"]
                            if complete
                            else None
                        ),
                        "status": comparison_status,
                    }
                )

    comparison = {
        "run_rows": sorted(
            run_rows, key=lambda row: (row["benchmark_id"], row["mode"], row["aspect"])
        ),
        "mode_delta_rows": delta_rows,
        "interpretation_policy": (
            "compare each modality mode with sequence-only only within the same "
            "benchmark/aspect; preserve whole-test results; mode-specific "
            "original-threshold deltas are no-retuning controls, "
            "not shared-threshold comparisons"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_sensitivity_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "label_sensitivity_comparison.md").write_text(
        markdown_report(comparison), encoding="utf-8"
    )
    (output_dir / "run_profiles.tsv").write_text(
        tsv_text(comparison["run_rows"]), encoding="utf-8"
    )
    (output_dir / "mode_vs_sequence_deltas.tsv").write_text(
        tsv_text(comparison["mode_delta_rows"]), encoding="utf-8"
    )
    (output_dir / "full_vs_sequence_deltas.tsv").write_text(
        tsv_text(
            [
                row
                for row in comparison["mode_delta_rows"]
                if row["comparison_mode"] == "full"
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
