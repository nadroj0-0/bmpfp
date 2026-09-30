#!/usr/bin/env python3
"""Compare two explicit cluster/member partitions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import heapq
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Iterator, TextIO


TOP_DIVERGENCES = 25


@dataclass
class PartitionStats:
    clusters: int = 0
    members: int = 0
    singleton_clusters: int = 0
    maximum_cluster_size: int = 0
    same_partition_pairs: int = 0
    divergent_clusters: int = 0
    members_in_divergent_clusters: int = 0
    size_1: int = 0
    size_2: int = 0
    size_3_to_5: int = 0
    size_6_to_10: int = 0
    size_11_to_100: int = 0
    size_101_to_1000: int = 0
    size_over_1000: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _log(message: str) -> None:
    print(f"[{_utc_now()}] {message}", file=sys.stderr, flush=True)


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _sort_command(
    *, sort_binary: str, output: Path, temporary: Path, keys: list[str],
    parallel: int, memory: str
) -> list[str]:
    command = [sort_binary, "-t", "\t", *keys, "-T", str(temporary), "-o", str(output)]
    probe = subprocess.run(
        [sort_binary, "--version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False,
    )
    if probe.returncode == 0 and "GNU" in probe.stdout:
        command[1:1] = [f"--parallel={parallel}", "-S", memory]
    return command


def _sort_stream(
    rows: Iterator[str], *, output: Path, temporary: Path, sort_binary: str,
    keys: list[str], parallel: int, memory: str,
) -> None:
    command = _sort_command(
        sort_binary=sort_binary, output=output, temporary=temporary, keys=keys,
        parallel=parallel, memory=memory,
    )
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, text=True, encoding="utf-8",
        env={"LC_ALL": "C"},
    )
    incoming = process.stdin
    try:
        for row in rows:
            incoming.write(row)
        incoming.close()
        status = process.wait()
    except BaseException:
        incoming.close()
        process.terminate()
        process.wait()
        raise
    if status != 0:
        raise subprocess.CalledProcessError(status, command)


def _sort_file(
    source: Path, *, output: Path, temporary: Path, sort_binary: str,
    keys: list[str], parallel: int, memory: str,
) -> None:
    command = _sort_command(
        sort_binary=sort_binary, output=output, temporary=temporary, keys=keys,
        parallel=parallel, memory=memory,
    )
    with source.open("rb") as incoming:
        subprocess.run(command, stdin=incoming, env={"LC_ALL": "C"}, check=True)


def _assignment_rows(path: Path, progress_label: str) -> Iterator[str]:
    first_content = True
    header: list[str] | None = None
    cluster_index = 0
    member_index = 1
    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number % 5_000_000 == 0:
                _log(f"{progress_label} conversion rows={line_number:,}")
            if not raw_line.strip():
                continue
            columns = raw_line.rstrip("\r\n").split("\t")
            if first_content and "mmseqs_cluster_id" in columns:
                candidates = [
                    index for index, name in enumerate(columns)
                    if name == "cluster_member_id"
                    or (name.endswith("_id") and name != "mmseqs_cluster_id")
                ]
                header = columns
                cluster_index = columns.index("mmseqs_cluster_id")
                member_index = candidates[0]
                first_content = False
                continue
            first_content = False
            if header is not None:
                cluster = columns[cluster_index]
                member = columns[member_index]
            else:
                cluster, member = columns
            yield f"{member}\t{cluster}\n"


def _mapping_rows(path: Path) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            member, cluster = raw_line.rstrip("\r\n").split("\t")
            yield member, cluster


def _merge_member_mappings(
    left: Path, right: Path, *, pairs_by_left: Path, temporary: Path,
    sort_binary: str, parallel: int, memory: str, sample_limit: int = 100,
) -> dict[str, object]:
    left_rows = _mapping_rows(left)
    right_rows = _mapping_rows(right)
    left_row = next(left_rows, None)
    right_row = next(right_rows, None)
    common = 0
    direct_label_matches = 0
    left_only = 0
    right_only = 0
    left_only_sample: list[str] = []
    right_only_sample: list[str] = []

    def pair_rows() -> Iterator[str]:
        nonlocal left_row, right_row, common, direct_label_matches, left_only, right_only
        while left_row is not None or right_row is not None:
            if right_row is None or (left_row is not None and left_row[0] < right_row[0]):
                left_only += 1
                if len(left_only_sample) < sample_limit:
                    left_only_sample.append(left_row[0])
                left_row = next(left_rows, None)
                continue
            if left_row is None or right_row[0] < left_row[0]:
                right_only += 1
                if len(right_only_sample) < sample_limit:
                    right_only_sample.append(right_row[0])
                right_row = next(right_rows, None)
                continue
            left_cluster = left_row[1]
            right_cluster = right_row[1]
            common += 1
            direct_label_matches += int(left_cluster == right_cluster)
            yield f"{left_cluster}\t{right_cluster}\n"
            left_row = next(left_rows, None)
            right_row = next(right_rows, None)

    _sort_stream(
        pair_rows(), output=pairs_by_left, temporary=temporary, sort_binary=sort_binary,
        keys=["-k1,1", "-k2,2"], parallel=parallel, memory=memory,
    )
    return {
        "common_members": common,
        "left_only_members": left_only,
        "right_only_members": right_only,
        "left_only_member_sample": left_only_sample,
        "right_only_member_sample": right_only_sample,
        "members_with_same_raw_representative": direct_label_matches,
    }


def _size_bucket(stats: PartitionStats, size: int) -> None:
    if size == 1:
        stats.size_1 += 1
    elif size == 2:
        stats.size_2 += 1
    elif size <= 5:
        stats.size_3_to_5 += 1
    elif size <= 10:
        stats.size_6_to_10 += 1
    elif size <= 100:
        stats.size_11_to_100 += 1
    elif size <= 1000:
        stats.size_101_to_1000 += 1
    else:
        stats.size_over_1000 += 1


def _analyse_partition(
    pairs: Path, *, primary_index: int, one_to_one_candidates: Path,
    divergence_path: Path,
) -> tuple[PartitionStats, int]:
    stats = PartitionStats()
    intersection_pairs = 0
    current_primary: str | None = None
    current_secondary: str | None = None
    pair_count = 0
    cluster_size = 0
    degree = 0
    sole_secondary = ""
    largest_overlaps: list[tuple[int, str]] = []
    divergent: list[tuple[int, int, str, str]] = []
    secondary_index = 1 - primary_index

    candidate_handle = one_to_one_candidates.open("w", encoding="utf-8", newline="")

    def finish_pair() -> None:
        nonlocal pair_count, intersection_pairs, cluster_size, degree, sole_secondary
        if current_secondary is None:
            return
        intersection_pairs += pair_count * (pair_count - 1) // 2
        cluster_size += pair_count
        degree += 1
        sole_secondary = current_secondary
        item = (pair_count, current_secondary)
        if len(largest_overlaps) < 5:
            heapq.heappush(largest_overlaps, item)
        elif item > largest_overlaps[0]:
            heapq.heapreplace(largest_overlaps, item)

    def finish_cluster() -> None:
        nonlocal cluster_size, degree, largest_overlaps
        if current_primary is None:
            return
        stats.clusters += 1
        stats.members += cluster_size
        stats.singleton_clusters += int(cluster_size == 1)
        stats.maximum_cluster_size = max(stats.maximum_cluster_size, cluster_size)
        stats.same_partition_pairs += cluster_size * (cluster_size - 1) // 2
        _size_bucket(stats, cluster_size)
        if degree == 1:
            if primary_index == 0:
                candidate_handle.write(f"{sole_secondary}\t{current_primary}\t{cluster_size}\n")
            else:
                candidate_handle.write(f"{current_primary}\t{sole_secondary}\t{cluster_size}\n")
        else:
            stats.divergent_clusters += 1
            stats.members_in_divergent_clusters += cluster_size
            overlaps = ",".join(
                f"{identifier}:{count}" for count, identifier in sorted(largest_overlaps, reverse=True)
            )
            item = (cluster_size, degree, current_primary, overlaps)
            if len(divergent) < TOP_DIVERGENCES:
                heapq.heappush(divergent, item)
            elif item > divergent[0]:
                heapq.heapreplace(divergent, item)
        cluster_size = 0
        degree = 0
        largest_overlaps = []

    try:
        with pairs.open("r", encoding="utf-8", newline="") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if line_number % 10_000_000 == 0:
                    _log(
                        f"partition analysis primary_column={primary_index + 1} "
                        f"rows={line_number:,}"
                    )
                columns = raw_line.rstrip("\r\n").split("\t")
                primary = columns[primary_index]
                secondary = columns[secondary_index]
                if current_primary is None:
                    current_primary, current_secondary, pair_count = primary, secondary, 1
                    continue
                if primary != current_primary:
                    finish_pair()
                    finish_cluster()
                    current_primary, current_secondary, pair_count = primary, secondary, 1
                elif secondary != current_secondary:
                    finish_pair()
                    current_secondary, pair_count = secondary, 1
                else:
                    pair_count += 1
            finish_pair()
            finish_cluster()
    finally:
        candidate_handle.close()

    with divergence_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("cluster_id\tmembers\tcounterpart_clusters\tlargest_overlaps\n")
        for size, degree_value, cluster, overlaps in sorted(divergent, reverse=True):
            handle.write(f"{cluster}\t{size}\t{degree_value}\t{overlaps}\n")
    return stats, intersection_pairs


def _count_exact_clusters(left_candidates: Path, right_candidates: Path) -> dict[str, int]:
    left_rows = _three_column_rows(left_candidates)
    right_rows = _three_column_rows(right_candidates)
    left = next(left_rows, None)
    right = next(right_rows, None)
    clusters = 0
    members = 0
    same_representative = 0
    while left is not None and right is not None:
        left_key = left[:2]
        right_key = right[:2]
        if left_key < right_key:
            left = next(left_rows, None)
        elif right_key < left_key:
            right = next(right_rows, None)
        else:
            clusters += 1
            members += left[2]
            same_representative += int(left[0] == left[1])
            left = next(left_rows, None)
            right = next(right_rows, None)
    return {
        "exact_cluster_blocks": clusters,
        "members_in_exact_cluster_blocks": members,
        "exact_cluster_blocks_with_same_representative": same_representative,
    }


def _three_column_rows(path: Path) -> Iterator[tuple[str, str, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            first, second, size = raw_line.rstrip("\r\n").split("\t")
            yield first, second, int(size)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(n: int, overlap_pairs: int, left_pairs: int, right_pairs: int) -> dict[str, float | int | None]:
    total_pairs = n * (n - 1) // 2
    expected = (left_pairs * right_pairs / total_pairs) if total_pairs else 0.0
    maximum = (left_pairs + right_pairs) / 2
    adjusted_rand = _ratio(overlap_pairs - expected, maximum - expected)
    return {
        "common_member_pairs": total_pairs,
        "pairs_together_in_left": left_pairs,
        "pairs_together_in_right": right_pairs,
        "pairs_together_in_both": overlap_pairs,
        "left_pair_recall": _ratio(overlap_pairs, left_pairs),
        "right_pair_recall": _ratio(overlap_pairs, right_pairs),
        "pair_jaccard": _ratio(overlap_pairs, left_pairs + right_pairs - overlap_pairs),
        "fowlkes_mallows": _ratio(overlap_pairs, math.sqrt(left_pairs * right_pairs)),
        "adjusted_rand_index": adjusted_rand,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pct(value: int, total: int) -> str:
    return f"{100 * value / total:.6f}%" if total else "n/a"


def _decimal(value: object) -> str:
    return f"{value:.12f}" if isinstance(value, (int, float)) else "n/a"


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    universe = payload["member_universe"]
    exact = payload["exact_partition_blocks"]
    left = payload["left_partition"]
    right = payload["right_partition"]
    metrics = payload["pairwise_agreement"]
    inputs = payload["inputs"]
    left_input = inputs["left"]
    right_input = inputs["right"]
    left_label = str(left_input["label"])
    right_label = str(right_input["label"])
    common = int(universe["common_members"])
    exact_members = int(exact["members_in_exact_cluster_blocks"])
    ari = metrics["adjusted_rand_index"]
    lines = [
        f"# {payload['title']}",
        "",
        "## Verdict",
        "",
        (
            "- The two files contain the same member universe."
            if not universe["left_only_members"] and not universe["right_only_members"]
            else "- The two files do not contain the same member universe."
        ),
        (
            "- The partitions are exactly identical."
            if payload["partitions_exactly_identical"]
            else "- The partitions are not exactly identical."
        ),
        f"- Exact cluster blocks cover {exact_members:,}/{common:,} common members ({_pct(exact_members, common)}).",
        f"- Adjusted Rand index: {_decimal(ari)}.",
        "",
        "## Counts",
        "",
        f"| Measure | {left_label} | {right_label} |",
        "|---|---:|---:|",
        f"| Members | {left['members']:,} | {right['members']:,} |",
        f"| Clusters | {left['clusters']:,} | {right['clusters']:,} |",
        f"| Singleton clusters | {left['singleton_clusters']:,} | {right['singleton_clusters']:,} |",
        f"| Maximum cluster size | {left['maximum_cluster_size']:,} | {right['maximum_cluster_size']:,} |",
        f"| Divergent clusters | {left['divergent_clusters']:,} | {right['divergent_clusters']:,} |",
        f"| Members in divergent clusters | {left['members_in_divergent_clusters']:,} | {right['members_in_divergent_clusters']:,} |",
        "",
        f"Here, a divergent {left_label} cluster is split across multiple {right_label} clusters; a divergent {right_label} cluster merges members from multiple {left_label} clusters.",
        "",
        "## Pairwise agreement",
        "",
        f"- Pairs together in both: {metrics['pairs_together_in_both']:,}",
        f"- {left_label} pair recall: {_decimal(metrics['left_pair_recall'])}",
        f"- {right_label} pair recall: {_decimal(metrics['right_pair_recall'])}",
        f"- Pair Jaccard: {_decimal(metrics['pair_jaccard'])}",
        f"- Fowlkes-Mallows: {_decimal(metrics['fowlkes_mallows'])}",
        "",
        "## Interpretation boundary",
        "",
        str(payload["interpretation_boundary"]),
        "",
        "See `comparison.json`, `largest_left_splits.tsv`, and `largest_right_merges.tsv` for full machine-readable results and bounded examples.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(args: argparse.Namespace) -> dict[str, object]:
    left = args.left.expanduser().resolve()
    right = args.right.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    scratch = args.scratch_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    sort_binary = args.sort_binary

    left_by_member = scratch / "left.by_member.tsv"
    right_by_member = scratch / "right.by_member.tsv"
    pairs_by_left = scratch / "pairs.by_left.tsv"
    pairs_by_right = scratch / "pairs.by_right.tsv"
    left_candidates = scratch / "left.one_to_one.tsv"
    right_candidates = scratch / "right.one_to_one.tsv"
    left_candidates_sorted = scratch / "left.one_to_one.sorted.tsv"
    right_candidates_sorted = scratch / "right.one_to_one.sorted.tsv"

    _log(f"sorting {args.left_label} assignments by member ID")
    _sort_stream(
        _assignment_rows(left, "left"), output=left_by_member, temporary=scratch,
        sort_binary=sort_binary, keys=["-k1,1"], parallel=args.sort_parallel,
        memory=args.sort_memory,
    )
    _log(f"sorting {args.right_label} assignments by member ID")
    _sort_stream(
        _assignment_rows(right, "right"), output=right_by_member, temporary=scratch,
        sort_binary=sort_binary, keys=["-k1,1"], parallel=args.sort_parallel,
        memory=args.sort_memory,
    )
    _log("joining member universes and sorting cluster intersections")
    universe = _merge_member_mappings(
        left_by_member, right_by_member, pairs_by_left=pairs_by_left, temporary=scratch,
        sort_binary=sort_binary, parallel=args.sort_parallel, memory=args.sort_memory,
    )
    _log(f"measuring {args.left_label} clusters split across {args.right_label} clusters")
    left_stats, overlap_pairs = _analyse_partition(
        pairs_by_left, primary_index=0, one_to_one_candidates=left_candidates,
        divergence_path=output / "largest_left_splits.tsv",
    )
    _log(f"re-sorting intersections by {args.right_label} cluster")
    _sort_file(
        pairs_by_left, output=pairs_by_right, temporary=scratch, sort_binary=sort_binary,
        keys=["-k2,2", "-k1,1"], parallel=args.sort_parallel, memory=args.sort_memory,
    )
    _log(f"measuring {args.right_label} clusters merged from {args.left_label} clusters")
    right_stats, overlap_pairs_right = _analyse_partition(
        pairs_by_right, primary_index=1, one_to_one_candidates=right_candidates,
        divergence_path=output / "largest_right_merges.tsv",
    )
    _log("matching exact cluster blocks independent of representative labels")
    _sort_file(
        left_candidates, output=left_candidates_sorted, temporary=scratch,
        sort_binary=sort_binary, keys=["-k1,1", "-k2,2"],
        parallel=args.sort_parallel, memory=args.sort_memory,
    )
    _sort_file(
        right_candidates, output=right_candidates_sorted, temporary=scratch,
        sort_binary=sort_binary, keys=["-k1,1", "-k2,2"],
        parallel=args.sort_parallel, memory=args.sort_memory,
    )
    exact = _count_exact_clusters(left_candidates_sorted, right_candidates_sorted)
    common = int(universe["common_members"])
    pairwise = _metrics(
        common, overlap_pairs, left_stats.same_partition_pairs, right_stats.same_partition_pairs
    )
    payload: dict[str, object] = {
        "title": args.title,
        "inputs": {
            "left": {"label": args.left_label},
            "right": {"label": args.right_label},
        },
        "member_universe": universe,
        "left_partition": asdict(left_stats),
        "right_partition": asdict(right_stats),
        "exact_partition_blocks": exact,
        "pairwise_agreement": pairwise,
        "partitions_exactly_identical": (
            not universe["left_only_members"]
            and not universe["right_only_members"]
            and exact["members_in_exact_cluster_blocks"] == common
            and left_stats.clusters == right_stats.clusters == exact["exact_cluster_blocks"]
        ),
        "interpretation_boundary": args.interpretation_boundary,
    }
    _write_json(output / "comparison.json", payload)
    _write_summary(output / "summary.md", payload)
    _log("comparison outputs complete")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="left-run")
    parser.add_argument("--right-label", default="right-run")
    parser.add_argument("--title", default="Homology cluster-partition comparison")
    parser.add_argument(
        "--interpretation-boundary",
        default=(
            "This comparison measures partition agreement only. It does not establish "
            "exhaustive pairwise sequence-identity separation or causal attribution."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--sort-binary", default="sort")
    parser.add_argument("--sort-parallel", type=int, default=2)
    parser.add_argument("--sort-memory", default="12G")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = compare(args)
    print(json.dumps({
        "partitions_exactly_identical": payload["partitions_exactly_identical"],
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
