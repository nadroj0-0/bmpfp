#!/usr/bin/env python3
"""Compare PFP specificity result directories."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ASPECTS = ("BPO", "CCO", "MFO")
MEASURES = ("ia", "xu_totipotency_raw")
BOOTSTRAP_METRICS = {
    "f": "fixed_unweighted_f",
    "weighted_f": "fixed_weighted_f",
    "jaccard_set_agreement": "fixed_unweighted_jaccard",
    "weighted_jaccard_set_agreement": "fixed_weighted_jaccard",
}


def finite_float(value: Any, field: str, context: str) -> float:
    del field, context
    return float(value)


def nonnegative_int(value: Any, field: str, context: str) -> int:
    del field, context
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare supplied IA/Xu specificity result directories."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each directory containing specificity result tables.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def parse_sources(values: Iterable[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        label, _separator, raw_path = value.partition("=")
        path = Path(raw_path).expanduser().resolve()
        result.append((label, path))
    return result


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def tsv(rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return output.getvalue()


def load_source(label: str, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bins_path = root / "specificity_bins.tsv"
    bootstrap_path = root / "bootstrap_intervals.tsv"
    bins = read_tsv(bins_path)
    bootstraps = read_tsv(bootstrap_path)
    benchmark_ids = {row["benchmark_id"] for row in bins}
    modes = {row["mode"] for row in bins}

    bootstrap_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in bootstraps:
        key = (
            row["aspect"],
            row["specificity_measure"],
            row["specificity_bin"],
            row["metric_name"],
        )
        bootstrap_by_key[key] = row
    combined: list[dict[str, Any]] = []
    for row in bins:
        context = (
            f"{label}/{row['aspect']}/{row['specificity_measure']}/"
            f"{row['specificity_bin']}"
        )
        key = (row["aspect"], row["specificity_measure"], row["specificity_bin"])
        row = dict(row)
        row["target_positive_proteins"] = nonnegative_int(
            row["target_positive_proteins"], "target_positive_proteins", context
        )
        for field in ("fixed_threshold", "fixed_unweighted_f"):
            row[field] = finite_float(row[field], field, context)
        row["fixed_weighted_f"] = (
            finite_float(row["fixed_weighted_f"], "fixed_weighted_f", context)
            if row["fixed_weighted_f"] not in ("", None)
            else None
        )
        value: dict[str, Any] = {"source_label": label, **row}
        for metric_name, output_name in BOOTSTRAP_METRICS.items():
            bootstrap = bootstrap_by_key.get(
                (
                    row["aspect"],
                    row["specificity_measure"],
                    row["specificity_bin"],
                    metric_name,
                )
            )
            value[f"{output_name}_ci_low"] = (
                finite_float(bootstrap["ci_low"], "ci_low", context)
                if bootstrap else ""
            )
            value[f"{output_name}_ci_high"] = (
                finite_float(bootstrap["ci_high"], "ci_high", context)
                if bootstrap else ""
            )
        combined.append(value)
    source = {
        "label": label,
        "benchmark_id": next(iter(benchmark_ids)),
        "mode": next(iter(modes)),
    }
    return source, combined


def broad_to_specific(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            row["source_label"],
            row["aspect"],
            row["specificity_measure"],
            row["specificity_bin"],
        ): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for label in sorted({row["source_label"] for row in rows}):
        for aspect in ASPECTS:
            for measure in MEASURES:
                q1 = indexed.get((label, aspect, measure, "specificity_q1"))
                q4 = indexed.get((label, aspect, measure, "specificity_q4"))
                if q1 is None or q4 is None:
                    continue
                q1_f = finite_float(q1["fixed_unweighted_f"], "fixed_unweighted_f", f"{label}/{aspect}/{measure}/Q1")
                q4_f = finite_float(q4["fixed_unweighted_f"], "fixed_unweighted_f", f"{label}/{aspect}/{measure}/Q4")
                q1_wf = q1["fixed_weighted_f"]
                q4_wf = q4["fixed_weighted_f"]
                weighted_available = q1_wf is not None and q4_wf is not None
                output.append(
                    {
                        "source_label": label,
                        "benchmark_id": q1["benchmark_id"],
                        "aspect": aspect,
                        "specificity_measure": measure,
                        "fixed_threshold": q1["fixed_threshold"],
                        "q1_fixed_f": q1_f,
                        "q1_fixed_f_ci_low": q1["fixed_unweighted_f_ci_low"],
                        "q1_fixed_f_ci_high": q1["fixed_unweighted_f_ci_high"],
                        "q4_fixed_f": q4_f,
                        "q4_fixed_f_ci_low": q4["fixed_unweighted_f_ci_low"],
                        "q4_fixed_f_ci_high": q4["fixed_unweighted_f_ci_high"],
                        "q4_minus_q1_fixed_f": q4_f - q1_f,
                        "q4_over_q1_fixed_f": q4_f / q1_f if q1_f else "",
                        "q1_fixed_weighted_f": q1_wf if weighted_available else "",
                        "q4_fixed_weighted_f": q4_wf if weighted_available else "",
                        "q4_minus_q1_fixed_weighted_f": (
                            q4_wf - q1_wf if weighted_available else ""
                        ),
                        "q1_positive_proteins": q1["target_positive_proteins"],
                        "q4_positive_proteins": q4["target_positive_proteins"],
                    }
                )
    return output


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Three-Way PFP Specificity Comparison",
        "",
        "All values use the fixed thresholds from the supplied prediction results. ",
        "Q1 is broad and Q4 is specific for ",
        "both IA and raw Xu totipotency.",
        "",
        "| Benchmark | Aspect | Measure | Q1 F | Q4 F | Q4 - Q1 | Q1 positives | Q4 positives |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {source_label} | {aspect} | {specificity_measure} | {q1_fixed_f:.3f} | "
            "{q4_fixed_f:.3f} | {q4_minus_q1_fixed_f:+.3f} | "
            "{q1_positive_proteins} | {q4_positive_proteins} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- These are robustness measurements, not a controlled trend line.",
            "- Each benchmark has its own training-derived IA values, term universe and fixed threshold.",
            "- Quartiles are benchmark-local rank groups; they do not contain identical GO terms.",
            "- CAFA3 also uses an older ontology snapshot. Xu panels are therefore comparable in direction, not term-for-term identity.",
            "- Bootstrap intervals quantify protein-sampling uncertainty, not training-seed uncertainty.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    sources = parse_sources(args.source)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_used: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for label, path in sources:
        source_record, source_rows = load_source(label, path)
        sources_used.append(source_record)
        rows.extend(source_rows)
    summary = broad_to_specific(rows)
    long_fields = tuple(rows[0])
    summary_fields = tuple(summary[0])
    (output_dir / "specificity_comparison_long.tsv").write_text(
        tsv(rows, long_fields), encoding="utf-8"
    )
    (output_dir / "broad_to_specific_summary.tsv").write_text(
        tsv(summary, summary_fields), encoding="utf-8"
    )
    (output_dir / "specificity_three_way_comparison.md").write_text(
        markdown(summary), encoding="utf-8"
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(
            {
                "analysis_kind": "cross_benchmark_specificity_comparison",
                "sources": sources_used,
                "source_count": len(sources_used),
                "comparison_policy": "descriptive benchmark-local quartiles; no controlled causal ordering",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
