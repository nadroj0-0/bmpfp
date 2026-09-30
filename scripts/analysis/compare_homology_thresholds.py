#!/usr/bin/env python3
"""Compare homology benchmark outputs across supplied thresholds."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import sys


ASPECTS = ("bp", "cc", "mf")
SPLITS = ("training", "validation", "test")
CSV_NAMES = tuple(f"{aspect}-{split_name}.csv" for aspect in ASPECTS for split_name in SPLITS)
ASSIGNMENT_NAME = "protein_cluster_assignments.tsv"
CLUSTER_SPLIT_NAME = "cluster_split_assignments.tsv"


@dataclass(frozen=True)
class CsvRow:
    split: str


@dataclass
class Benchmark:
    label: str
    root: Path
    proteins: dict[str, tuple[str, str]]
    csv_rows: dict[str, dict[str, CsvRow]]
    unassigned_proteins: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _log(message: str) -> None:
    print(f"[{_utc_now()}] {message}", file=sys.stderr, flush=True)


def _parse_spec(raw: str) -> tuple[str, Path]:
    label, path = raw.split("=", 1)
    return label, Path(path).expanduser().resolve()


def _read_assignments(path: Path) -> tuple[dict[str, tuple[str, str]], int]:
    proteins: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            proteins[row["uniprot_accession"]] = (
                row["mmseqs_cluster_id"], row["split"]
            )
    return proteins, sum(not split_name for _, split_name in proteins.values())


def _read_csv(path: Path, expected_split: str) -> dict[str, CsvRow]:
    rows: dict[str, CsvRow] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["proteins"]] = CsvRow(split=expected_split)
    return rows


def _load_benchmark(label: str, root: Path) -> Benchmark:
    _log(f"{label}: reading protein-to-cluster assignments")
    proteins, unassigned = _read_assignments(root / ASSIGNMENT_NAME)

    csv_rows: dict[str, dict[str, CsvRow]] = {aspect: {} for aspect in ASPECTS}
    for name in CSV_NAMES:
        aspect, split_with_suffix = name.split("-", 1)
        split_name = split_with_suffix.removesuffix(".csv")
        _log(f"{label}: reading {name}")
        rows = _read_csv(root / name, split_name)
        csv_rows[aspect].update(rows)
    return Benchmark(label, root, proteins, csv_rows, unassigned)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _partition_metrics(
    left: dict[str, tuple[str, str]], right: dict[str, tuple[str, str]]
) -> dict[str, object]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    left_counts: Counter[str] = Counter()
    right_counts: Counter[str] = Counter()
    left_blocks: defaultdict[str, list[str]] = defaultdict(list)
    right_blocks: defaultdict[str, list[str]] = defaultdict(list)
    for protein in left:
        left_cluster = left[protein][0] or f"__unassigned_left__:{protein}"
        right_cluster = right[protein][0] or f"__unassigned_right__:{protein}"
        pair_counts[left_cluster, right_cluster] += 1
        left_counts[left_cluster] += 1
        right_counts[right_cluster] += 1
        left_blocks[left_cluster].append(protein)
        right_blocks[right_cluster].append(protein)

    together_both = sum(count * (count - 1) // 2 for count in pair_counts.values())
    together_left = sum(count * (count - 1) // 2 for count in left_counts.values())
    together_right = sum(count * (count - 1) // 2 for count in right_counts.values())
    n = len(left)
    total_pairs = n * (n - 1) // 2
    expected = together_left * together_right / total_pairs if total_pairs else 0.0
    maximum = (together_left + together_right) / 2
    exact_left = Counter(frozenset(members) for members in left_blocks.values())
    exact_right = Counter(frozenset(members) for members in right_blocks.values())
    exact_blocks = exact_left & exact_right
    left_sizes = {frozenset(members): len(members) for members in left_blocks.values()}
    exact_members = sum(left_sizes[key] * count for key, count in exact_blocks.items())
    return {
        "proteins": n,
        "left_clusters": len(left_counts),
        "right_clusters": len(right_counts),
        "exact_cluster_blocks": sum(exact_blocks.values()),
        "proteins_in_exact_cluster_blocks": exact_members,
        "partitions_exactly_identical": exact_members == n and len(left_counts) == len(right_counts),
        "pairs_together_in_left": together_left,
        "pairs_together_in_right": together_right,
        "pairs_together_in_both": together_both,
        "left_pair_recall": _ratio(together_both, together_left),
        "right_pair_recall": _ratio(together_both, together_right),
        "pair_jaccard": _ratio(together_both, together_left + together_right - together_both),
        "fowlkes_mallows": _ratio(together_both, math.sqrt(together_left * together_right)),
        "adjusted_rand_index": _ratio(together_both - expected, maximum - expected),
    }


def _transition_counts(
    left: dict[str, str], right: dict[str, str]
) -> tuple[dict[str, int], dict[str, object]]:
    matrix: Counter[str] = Counter()
    for protein in set(left).union(right):
        matrix[f"{left.get(protein, 'absent')}->{right.get(protein, 'absent')}"] += 1
    same = sum(count for transition, count in matrix.items() if transition.split("->")[0] == transition.split("->")[1])
    return dict(sorted(matrix.items())), {
        "union_proteins": len(set(left).union(right)),
        "common_proteins": len(set(left).intersection(right)),
        "left_only_proteins": len(set(left) - set(right)),
        "right_only_proteins": len(set(right) - set(left)),
        "same_state_proteins": same,
        "changed_state_proteins": len(set(left).union(right)) - same,
    }


def _compare(left: Benchmark, right: Benchmark) -> dict[str, object]:
    assigned_left = {
        protein: assignment for protein, assignment in left.proteins.items() if assignment[1]
    }
    assigned_right = {
        protein: assignment for protein, assignment in right.proteins.items() if assignment[1]
    }
    global_left = {protein: split_name for protein, (_, split_name) in assigned_left.items()}
    global_right = {protein: split_name for protein, (_, split_name) in assigned_right.items()}
    global_matrix, global_summary = _transition_counts(global_left, global_right)
    aspects: dict[str, object] = {}
    for aspect in ASPECTS:
        left_rows = left.csv_rows[aspect]
        right_rows = right.csv_rows[aspect]
        matrix, summary = _transition_counts(
            {protein: row.split for protein, row in left_rows.items()},
            {protein: row.split for protein, row in right_rows.items()},
        )
        summary["transition_matrix"] = matrix
        aspects[aspect] = summary

    return {
        "left": left.label,
        "right": right.label,
        "global_split": {**global_summary, "transition_matrix": global_matrix},
        "retained_partition": _partition_metrics(assigned_left, assigned_right),
        "aspects": aspects,
    }


def _benchmark_summary(benchmark: Benchmark) -> dict[str, object]:
    split_counts = Counter(split_name or "unassigned" for _, split_name in benchmark.proteins.values())
    return {
        "label": benchmark.label,
        "protein_universe": len(benchmark.proteins),
        "assigned_proteins": len(benchmark.proteins) - benchmark.unassigned_proteins,
        "unassigned_proteins": benchmark.unassigned_proteins,
        "split_counts": dict(sorted(split_counts.items())),
        "aspect_populations": {aspect: len(benchmark.csv_rows[aspect]) for aspect in ASPECTS},
    }


def _pct(value: int, total: int) -> str:
    return f"{100 * value / total:.3f}%" if total else "n/a"


def _metric(value: object) -> str:
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _write_outputs(output: Path, payload: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (output / "split_transitions.tsv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("left\tright\taspect\ttransition\tproteins\n")
        for comparison in payload["adjacent_comparisons"]:
            for transition, count in comparison["global_split"]["transition_matrix"].items():
                handle.write(
                    f"{comparison['left']}\t{comparison['right']}\tall\t{transition}\t{count}\n"
                )
            for aspect, summary in comparison["aspects"].items():
                for transition, count in summary["transition_matrix"].items():
                    handle.write(
                        f"{comparison['left']}\t{comparison['right']}\t{aspect}\t"
                        f"{transition}\t{count}\n"
                    )

    lines = [
        "# Homology threshold progression comparison",
        "",
        "## Scope",
        "",
        "This report compares homology benchmark directories in the supplied order and measures adjacent split and retained-cluster changes.",
        "",
        "## Threshold inputs",
        "",
        "| Threshold | Protein universe | Assigned | Unassigned | BP | CC | MF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in payload["benchmarks"]:
        lines.append(
            f"| {benchmark['label']} | {benchmark['protein_universe']:,} | "
            f"{benchmark['assigned_proteins']:,} | {benchmark['unassigned_proteins']:,} | "
            f"{benchmark['aspect_populations']['bp']:,} | "
            f"{benchmark['aspect_populations']['cc']:,} | "
            f"{benchmark['aspect_populations']['mf']:,} |"
        )
    lines.extend([
        "",
        "## Adjacent comparisons",
        "",
        "| Pair | Global split moves | Retained partition ARI | Exact-cluster protein coverage | BP moves | CC moves | MF moves |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for comparison in payload["adjacent_comparisons"]:
        partition = comparison["retained_partition"]
        proteins = int(partition["proteins"])
        exact_members = int(partition["proteins_in_exact_cluster_blocks"])
        lines.append(
            f"| {comparison['left']} to {comparison['right']} | "
            f"{comparison['global_split']['changed_state_proteins']:,} | "
            f"{_metric(partition['adjusted_rand_index'])} | "
            f"{exact_members:,}/{proteins:,} ({_pct(exact_members, proteins)}) | "
            f"{comparison['aspects']['bp']['changed_state_proteins']:,} | "
            f"{comparison['aspects']['cc']['changed_state_proteins']:,} | "
            f"{comparison['aspects']['mf']['changed_state_proteins']:,} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- A high ARI means the retained annotated proteins are grouped similarly; it does not prove every pair of sequences satisfies the nominal identity threshold.",
        "- Split moves show how much the actual model inputs change between thresholds.",
        "- Root-excluded performance diagnostics are needed alongside this construction comparison before interpreting flat model scores.",
    ])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def run(args: argparse.Namespace) -> dict[str, object]:
    specs = [_parse_spec(raw) for raw in args.benchmark]
    output = args.output_dir.expanduser().resolve()

    summaries: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    previous: Benchmark | None = None
    for label, root in specs:
        current = _load_benchmark(label, root)
        summaries.append(_benchmark_summary(current))
        if previous is not None:
            _log(f"comparing {previous.label} to {current.label}")
            comparisons.append(_compare(previous, current))
        previous = current
    payload: dict[str, object] = {
        "benchmarks": summaries,
        "adjacent_comparisons": comparisons,
    }
    _write_outputs(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Homology benchmark directory; repeat in comparison order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run(args)
    print(json.dumps({
        "benchmarks": len(payload["benchmarks"]),
        "comparisons": len(payload["adjacent_comparisons"]),
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
