#!/usr/bin/env python3
"""Analyse the label space of a benchmark in the PFP nine-CSV layout."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

from label_space_common import (
    ASPECTS,
    ASPECT_TO_PREFIX,
    ASPECT_TO_ROOT,
    CSV_SPLITS,
    write_json,
    write_text,
    analyse_csv,
    read_ia_file,
    read_obo,
)


def tsv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# PFP Label-Space Analysis: {report['benchmark_id']}",
        "",
        "## Test Label Profile",
        "",
        "| Aspect | Proteins | Terms | Root-only | Root-only % | Mean labels | Mean non-root | Root-only baseline F |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for aspect in ASPECTS:
        item = report["files"][f"{aspect}:test"]
        lines.append(
            "| {aspect} | {rows:,} | {terms:,} | {root_only_rows:,} | "
            "{root_only_fraction:.2%} | {labels:.2f} | {non_root:.2f} | {baseline:.3f} |".format(
                aspect=aspect,
                rows=item["rows"],
                terms=item["terms"],
                root_only_rows=item["root_only_rows"],
                root_only_fraction=item["root_only_fraction"],
                labels=item["labels_per_protein"]["mean"],
                non_root=item["non_root_labels_per_protein"]["mean"],
                baseline=item["root_only_diagnostic_baseline"]["macro_f"],
            )
        )
    lines.extend(
        [
            "",
            "The root-only baseline is an arithmetic diagnostic over the retained label matrix, not a cafaeval result.",
            "Root-only prevalence is reported as quality-control evidence and is not a failure criterion.",
            "",
            "## Files",
            "",
            "- `label_space_analysis.json`: complete machine-readable report.",
            "- `root_only_targets.tsv`: exact root-only target membership.",
            "- `term_support.tsv`: support, depth and optional IA for every retained term.",
            "- `label_depth_histogram.tsv`: per-file maximum shortest-depth histogram.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--obo-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ia-file-dir",
        type=Path,
        help="Optional directory containing BPO_ia.txt, CCO_ia.txt and MFO_ia.txt.",
    )
    args = parser.parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    obo_file = args.obo_file.resolve()
    output_dir = args.output_dir.resolve()
    ia_file_dir = args.ia_file_dir.resolve() if args.ia_file_dir else None
    graph = read_obo(obo_file)

    csv_reports: dict[str, dict[str, Any]] = {}
    root_only_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    for aspect in ASPECTS:
        ia_path = ia_file_dir / f"{aspect}_ia.txt" if ia_file_dir else None
        ia_values = read_ia_file(ia_path)
        for split in CSV_SPLITS:
            path = benchmark_dir / f"{ASPECT_TO_PREFIX[aspect]}-{split}.csv"
            item, roots, supports, histogram = analyse_csv(
                path=path,
                aspect=aspect,
                split=split,
                graph=graph,
                ia_values=ia_values,
            )
            csv_reports[f"{aspect}:{split}"] = item
            root_only_rows.extend(roots)
            term_rows.extend(supports)
            for depth, count in sorted(histogram.items()):
                histogram_rows.append(
                    {"aspect": aspect, "split": split, "depth": depth, "proteins": count}
                )

    for item in csv_reports.values():
        item.pop("_ordered_records", None)
    report = {
        "benchmark_id": args.benchmark_id,
        "roots": ASPECT_TO_ROOT,
        "files": csv_reports,
        "root_only_policy": {
            "definition": "A labelled row with the ontology root positive and zero positive non-root terms.",
            "is_failure_criterion": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "label_space_analysis.json", report)
    write_text(output_dir / "label_space_analysis.md", markdown_report(report))
    write_text(
            output_dir / "root_only_targets.tsv",
            tsv_text(
                ["aspect", "split", "protein_id"],
                sorted(root_only_rows, key=lambda row: (row["aspect"], row["split"], row["protein_id"])),
            ),
        )
    write_text(
            output_dir / "term_support.tsv",
            tsv_text(
                [
                    "aspect",
                    "split",
                    "term",
                    "is_root",
                    "support",
                    "shortest_depth",
                    "longest_depth",
                    "ia",
                ],
                term_rows,
            ),
        )
    write_text(
            output_dir / "label_depth_histogram.tsv",
            tsv_text(["aspect", "split", "depth", "proteins"], histogram_rows),
        )

    print(json.dumps({"output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
