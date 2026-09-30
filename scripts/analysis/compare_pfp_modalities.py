#!/usr/bin/env python3
"""Compare PFP modality evaluation summaries against sequence-only."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping

from label_space_common import ASPECTS
from pfp_sensitivity_common import load_evaluation_summary


BASELINE_MODE = "sequence-only"
def load_evaluation(path: Path) -> dict[str, Any]:
    """Load one model-evaluation directory or summary JSON."""

    summary, _ = load_evaluation_summary(path)
    return {"summary": summary}


def tsv_text(rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return output.getvalue()


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# PFP Modality-Contribution Comparison",
        "",
        "Metrics are read from the supplied evaluation results.",
        "",
        "| Mode | Evaluation split | Aspect | Fmax | wFmax | Unweighted count-space s (`cafa_smin`) | IA-weighted semantic S_min |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["run_rows"]:
        wsmin = row.get("cafa_wsmin")
        lines.append(
            f"| {row['mode']} | {row['evaluation_split']} | {row['aspect']} | {row['cafa_fmax']:.6f} | "
            f"{row['cafa_wfmax']:.6f} | {row['cafa_smin']:.6f} | "
            f"{'NA' if wsmin is None else f'{float(wsmin):.6f}'} |"
        )
    lines.extend(
        [
            "",
            "## Improvement Over Sequence-Only",
            "",
            "| Mode | Baseline split | Candidate split | Aspect | Fmax improvement | wFmax improvement | Unweighted count-space s improvement | IA-weighted semantic S_min improvement |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["delta_rows"]:
        wsmin = row.get("cafa_wsmin_improvement")
        lines.append(
            f"| {row['comparison_mode']} | {row['baseline_evaluation_split']} | "
            f"{row['candidate_evaluation_split']} | {row['aspect']} | "
            f"{row['cafa_fmax_improvement']:+.6f} | "
            f"{row['cafa_wfmax_improvement']:+.6f} | "
            f"{row['cafa_smin_improvement']:+.6f} | "
            f"{'NA' if wsmin is None else f'{float(wsmin):+.6f}'} |"
        )
    lines.extend(
        [
            "",
            "`cafa_smin` is the unweighted count-space distance. A positive improvement means the candidate has a lower distance than sequence-only.",
            "`cafa_wsmin`, when present in both supplied results, is the IA-weighted semantic S_min and is also lower-is-better.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        type=Path,
        required=True,
        help="Evaluation directory or evaluation_summary.json; repeat per mode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    evaluations = [load_evaluation(path) for path in args.evaluation]
    modes = [item["summary"]["mode"] for item in evaluations]
    by_mode = {item["summary"]["mode"]: item for item in evaluations}
    baseline_summary = by_mode[BASELINE_MODE]["summary"]
    baseline_aspect_set = set(baseline_summary["selected_aspects"])
    baseline_aspects = [aspect for aspect in ASPECTS if aspect in baseline_aspect_set]

    run_rows: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for item in evaluations:
        summary = item["summary"]
        mode = str(summary["mode"])
        evaluation_split = str(summary["evaluation_split"])
        for aspect in baseline_aspects:
            result = summary["aspects"][aspect]
            row = {
                "mode": mode,
                "evaluation_split": evaluation_split,
                "aspect": aspect,
                "cafa_fmax": float(result["cafa_fmax"]),
                "cafa_wfmax": float(result["cafa_wfmax"]),
                "cafa_smin": float(result["cafa_smin"]),
                "cafa_wsmin": (
                    float(result["cafa_wsmin"])
                    if result.get("cafa_wsmin") is not None
                    else None
                ),
            }
            run_rows.append(row)
            indexed[(mode, aspect)] = row

    delta_rows: list[dict[str, Any]] = []
    comparison_modes = list(dict.fromkeys(modes))
    for mode in comparison_modes:
        if mode == BASELINE_MODE:
            continue
        for aspect in baseline_aspects:
            candidate = indexed[(mode, aspect)]
            baseline = indexed[(BASELINE_MODE, aspect)]
            wsmin_delta = None
            if (
                baseline["cafa_wsmin"] is not None
                and candidate["cafa_wsmin"] is not None
            ):
                wsmin_delta = baseline["cafa_wsmin"] - candidate["cafa_wsmin"]
            delta_rows.append(
                {
                    "comparison_mode": mode,
                    "baseline_mode": BASELINE_MODE,
                    "baseline_evaluation_split": baseline["evaluation_split"],
                    "candidate_evaluation_split": candidate["evaluation_split"],
                    "aspect": aspect,
                    "cafa_fmax_improvement": (
                        candidate["cafa_fmax"] - baseline["cafa_fmax"]
                    ),
                    "cafa_wfmax_improvement": (
                        candidate["cafa_wfmax"] - baseline["cafa_wfmax"]
                    ),
                    "cafa_smin_improvement": (
                        baseline["cafa_smin"] - candidate["cafa_smin"]
                    ),
                    "cafa_wsmin_improvement": wsmin_delta,
                }
            )

    comparison = {
        "baseline_mode": BASELINE_MODE,
        "run_rows": sorted(
            run_rows,
            key=lambda row: (
                comparison_modes.index(row["mode"]),
                ASPECTS.index(row["aspect"]),
            ),
        ),
        "delta_rows": delta_rows,
        "metric_policy": {
            "primary": "cafa_fmax",
            "secondary": ["cafa_wfmax", "cafa_smin", "cafa_wsmin"],
            "cafa_smin": "unweighted count-space distance; lower is better",
            "cafa_wsmin": "IA-weighted semantic S_min; lower is better when present",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "modality_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "modality_comparison.md").write_text(
        markdown_report(comparison), encoding="utf-8"
    )
    (output_dir / "evaluation_metrics.tsv").write_text(
        tsv_text(
            comparison["run_rows"],
            (
                "mode",
                "evaluation_split",
                "aspect",
                "cafa_fmax",
                "cafa_wfmax",
                "cafa_smin",
                "cafa_wsmin",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "mode_vs_sequence_deltas.tsv").write_text(
        tsv_text(
            delta_rows,
            (
                "comparison_mode",
                "baseline_mode",
                "baseline_evaluation_split",
                "candidate_evaluation_split",
                "aspect",
                "cafa_fmax_improvement",
                "cafa_wfmax_improvement",
                "cafa_smin_improvement",
                "cafa_wsmin_improvement",
            ),
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
