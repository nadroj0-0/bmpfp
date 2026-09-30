#!/usr/bin/env python3
"""Compare consecutive homology partitions at full-member and supervised levels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Iterator, Mapping, Sequence


SPLITS = ("training", "validation", "test")
ASPECTS = ("bp", "cc", "mf")
MEMBER_FILE = "retained_cluster_members.tsv.gz"
PROTEIN_FILE = "protein_cluster_assignments.tsv"
CLUSTER_FILE = "cluster_split_assignments.tsv"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", file=sys.stderr, flush=True)


def parse_spec(raw: str) -> tuple[str, Path]:
    label, raw_path = raw.split("=", 1)
    label = label.strip()
    return label, Path(raw_path).expanduser().resolve()


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def combinations_two(value: int) -> int:
    return value * (value - 1) // 2


def tsv(rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _single_dynamic_field(
    fieldnames: Sequence[str],
    *,
    preferred: str,
    suffix: str,
    excluded: set[str],
    description: str,
) -> str:
    if preferred in fieldnames:
        return preferred
    candidates = [
        field for field in fieldnames
        if field.endswith(suffix) and field not in excluded
    ]
    return candidates[0]


def read_cluster_metadata(path: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        member_count_field = _single_dynamic_field(
            reader.fieldnames,
            preferred="cluster_member_count",
            suffix="_member_count",
            excluded={"qualifying_uniprot_count"},
            description="cluster-member count",
        )
        for row in reader:
            cluster = row["mmseqs_cluster_id"]
            split = row["split"]
            result[cluster] = {
                "split": split,
                "members": int(row[member_count_field]),
                "proteins": int(row["qualifying_uniprot_count"]),
            }
    return result


def normalize_members(
    source: Path,
    destination: Path,
    sort_temporary: Path,
    metadata: Mapping[str, Mapping[str, int | str]],
) -> dict[str, int]:
    command = [
        "sort", "--parallel=1", "-S", "1G", "-t", "\t", "-k1,1",
        "-T", str(sort_temporary), "-o", str(destination),
    ]
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True, encoding="utf-8", env=environment)
    incoming = process.stdin
    cluster_counts: Counter[str] = Counter()
    linked = 0
    background = 0
    try:
        with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            member_field = _single_dynamic_field(
                reader.fieldnames,
                preferred="cluster_member_id",
                suffix="_id",
                excluded={"mmseqs_cluster_id"},
                description="cluster-member identifier",
            )
            for row in reader:
                cluster = row["mmseqs_cluster_id"]
                split = row["split"]
                member = row[member_field]
                flag = row["connected_to_qualifying_uniprot"]
                incoming.write(f"{member}\t{cluster}\t{split}\t{flag}\n")
                cluster_counts[cluster] += 1
                linked += int(flag == "1")
                background += int(flag == "0")
        incoming.close()
        status = process.wait()
    except BaseException:
        incoming.close()
        process.terminate()
        process.wait()
        raise
    if status != 0:
        raise subprocess.CalledProcessError(status, command)
    return {"members": linked + background, "qualifying_linked_members": linked, "background_members": background}


def member_rows(path: Path) -> Iterator[tuple[str, str, str, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            member, cluster, split, flag_raw = raw.rstrip("\r\n").split("\t")
            yield member, cluster, split, int(flag_raw)


def read_supervised(path: Path, metadata: Mapping[str, Mapping[str, int | str]]) -> dict[str, tuple[str, str, str]]:
    result: dict[str, tuple[str, str, str]] = {}
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        member_candidates = [
            field for field in reader.fieldnames
            if field.endswith("_id") and field != "mmseqs_cluster_id"
        ]
        member_field = member_candidates[0] if len(member_candidates) == 1 else None
        for row in reader:
            protein = row["uniprot_accession"]
            cluster = row["mmseqs_cluster_id"]
            split = row["split"]
            if not cluster and not split:
                continue
            assignment = (cluster, split, row.get(member_field, "") if member_field else "")
            result[protein] = assignment
        counts.update(cluster for cluster, _, _ in result.values())
    return result


def read_aspect_splits(root: Path, aspect: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in SPLITS:
        path = root / f"{aspect}-{split}.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                protein = row["proteins"]
                result[protein] = split
    return result


def transition_counts(left: Mapping[str, str], right: Mapping[str, str]) -> Counter[str]:
    transitions: Counter[str] = Counter()
    for protein in set(left).union(right):
        transitions[f"{left.get(protein, 'absent')}->{right.get(protein, 'absent')}"] += 1
    return transitions


def typed_pair_counts(counts: Iterable[tuple[int, int]]) -> dict[str, int]:
    ll = lu = uu = 0
    for linked, background in counts:
        ll += combinations_two(linked)
        lu += linked * background
        uu += combinations_two(background)
    return {"linked_linked": ll, "linked_background": lu, "background_background": uu}


def partition_metrics(left_counts: Iterable[int], right_counts: Iterable[int], cell_counts: Iterable[int], proteins: int) -> dict[str, float | int | None]:
    left_pairs = sum(combinations_two(value) for value in left_counts)
    right_pairs = sum(combinations_two(value) for value in right_counts)
    shared_pairs = sum(combinations_two(value) for value in cell_counts)
    total_pairs = combinations_two(proteins)
    expected = left_pairs * right_pairs / total_pairs if total_pairs else 0.0
    maximum = (left_pairs + right_pairs) / 2
    return {
        "proteins": proteins,
        "left_pairs": left_pairs,
        "right_pairs": right_pairs,
        "pairs_in_both": shared_pairs,
        "pair_jaccard": ratio(shared_pairs, left_pairs + right_pairs - shared_pairs),
        "adjusted_rand_index": ratio(shared_pairs - expected, maximum - expected),
    }


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def lineage_category(old_count: int, new_count: int, old_only: int, new_only: int) -> str:
    if old_count == 0:
        return "appeared"
    if new_count == 0:
        return "disappeared"
    if old_count > 1 and new_count == 1:
        return "merged"
    if old_count == 1 and new_count > 1:
        return "split"
    if old_count > 1 and new_count > 1:
        return "complex"
    if old_only == 0 and new_only == 0:
        return "unchanged"
    if old_only == 0:
        return "expanded"
    if new_only == 0:
        return "contracted"
    return "reorganised_one_to_one"


def compare_pair(
    left_label: str,
    right_label: str,
    left_members_path: Path,
    right_members_path: Path,
    left_metadata: Mapping[str, Mapping[str, int | str]],
    right_metadata: Mapping[str, Mapping[str, int | str]],
    left_supervised: Mapping[str, tuple[str, str, str]],
    right_supervised: Mapping[str, tuple[str, str, str]],
) -> dict[str, object]:
    supervised_cells: Counter[tuple[str, str]] = Counter()
    supervised_left_counts: Counter[str] = Counter()
    supervised_right_counts: Counter[str] = Counter()
    supervised_transitions: Counter[tuple[str, str]] = Counter()
    for protein, (left_cluster, left_split, _) in left_supervised.items():
        right_cluster, right_split, _ = right_supervised[protein]
        supervised_cells[left_cluster, right_cluster] += 1
        supervised_left_counts[left_cluster] += 1
        supervised_right_counts[right_cluster] += 1
        supervised_transitions[left_split, right_split] += 1

    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    left_common: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    right_common: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    left_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    right_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    full_transitions: Counter[tuple[str, str, str]] = Counter()
    flag_changes: Counter[tuple[int, int]] = Counter()
    common = left_only = right_only = 0

    left_iterator = member_rows(left_members_path)
    right_iterator = member_rows(right_members_path)
    left_row = next(left_iterator, None)
    right_row = next(right_iterator, None)
    while left_row is not None or right_row is not None:
        if right_row is None or (left_row is not None and left_row[0] < right_row[0]):
            _, cluster, split, linked = left_row
            left_totals[cluster][linked] += 1
            full_transitions[("linked" if linked else "background", split, "absent")] += 1
            left_only += 1
            left_row = next(left_iterator, None)
            continue
        if left_row is None or right_row[0] < left_row[0]:
            _, cluster, split, linked = right_row
            right_totals[cluster][linked] += 1
            full_transitions[("linked" if linked else "background", "absent", split)] += 1
            right_only += 1
            right_row = next(right_iterator, None)
            continue
        _, left_cluster, left_split, left_linked = left_row
        _, right_cluster, right_split, right_linked = right_row
        flag_changes[left_linked, right_linked] += 1
        linked = int(bool(left_linked or right_linked))
        index = linked
        cells[left_cluster, right_cluster][index] += 1
        left_common[left_cluster][index] += 1
        right_common[right_cluster][index] += 1
        left_totals[left_cluster][index] += 1
        right_totals[right_cluster][index] += 1
        full_transitions[("linked" if linked else "background", left_split, right_split)] += 1
        common += 1
        left_row = next(left_iterator, None)
        right_row = next(right_iterator, None)

    old_pairs = typed_pair_counts((value[1], value[0]) for value in left_common.values())
    new_pairs = typed_pair_counts((value[1], value[0]) for value in right_common.values())
    retained_pairs = typed_pair_counts((value[1], value[0]) for value in cells.values())
    pair_change_rows = []
    for pair_type in ("linked_linked", "linked_background", "background_background"):
        pair_change_rows.append({
            "left": left_label,
            "right": right_label,
            "population": "retained_cluster_members",
            "pair_type": pair_type,
            "old_same_cluster_pairs": old_pairs[pair_type],
            "new_same_cluster_pairs": new_pairs[pair_type],
            "retained_same_cluster_pairs": retained_pairs[pair_type],
            "gained_same_cluster_pairs": new_pairs[pair_type] - retained_pairs[pair_type],
            "lost_same_cluster_pairs": old_pairs[pair_type] - retained_pairs[pair_type],
        })
    supervised_old_pairs = sum(combinations_two(value) for value in supervised_left_counts.values())
    supervised_new_pairs = sum(combinations_two(value) for value in supervised_right_counts.values())
    supervised_retained_pairs = sum(combinations_two(value) for value in supervised_cells.values())
    pair_change_rows.append({
        "left": left_label,
        "right": right_label,
        "population": "canonical_supervised_proteins",
        "pair_type": "labelled_labelled",
        "old_same_cluster_pairs": supervised_old_pairs,
        "new_same_cluster_pairs": supervised_new_pairs,
        "retained_same_cluster_pairs": supervised_retained_pairs,
        "gained_same_cluster_pairs": supervised_new_pairs - supervised_retained_pairs,
        "lost_same_cluster_pairs": supervised_old_pairs - supervised_retained_pairs,
    })

    union_find = UnionFind()
    for cluster in left_metadata:
        union_find.add("L:" + cluster)
    for cluster in right_metadata:
        union_find.add("R:" + cluster)
    for left_cluster, right_cluster in cells:
        union_find.union("L:" + left_cluster, "R:" + right_cluster)
    components: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"old": [], "new": []})
    for node in union_find.parent:
        side, cluster = node.split(":", 1)
        components[union_find.find(node)]["old" if side == "L" else "new"].append(cluster)

    supervised_by_cell = supervised_cells
    lineage_rows: list[dict[str, object]] = []
    category_summary: Counter[str] = Counter()
    category_members: Counter[str] = Counter()
    category_old_supervised: Counter[str] = Counter()
    category_new_supervised: Counter[str] = Counter()
    for component_id, component in enumerate(components.values(), start=1):
        old_clusters = component["old"]
        new_clusters = component["new"]
        old_members = sum(sum(left_totals[cluster]) for cluster in old_clusters)
        new_members = sum(sum(right_totals[cluster]) for cluster in new_clusters)
        overlap_members = sum(sum(left_common[cluster]) for cluster in old_clusters)
        old_only_members = old_members - overlap_members
        new_only_members = new_members - overlap_members
        category = lineage_category(len(old_clusters), len(new_clusters), old_only_members, new_only_members)
        old_supervised = sum(supervised_left_counts[cluster] for cluster in old_clusters)
        new_supervised = sum(supervised_right_counts[cluster] for cluster in new_clusters)
        category_summary[category] += 1
        category_members[category] += overlap_members
        category_old_supervised[category] += old_supervised
        category_new_supervised[category] += new_supervised
        lineage_rows.append({
            "left": left_label,
            "right": right_label,
            "component_id": component_id,
            "category": category,
            "old_clusters": len(old_clusters),
            "new_clusters": len(new_clusters),
            "old_members": old_members,
            "new_members": new_members,
            "common_members": overlap_members,
            "old_only_members": old_only_members,
            "new_only_members": new_only_members,
            "old_supervised_proteins": old_supervised,
            "new_supervised_proteins": new_supervised,
        })

    predecessor_cells: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    supervised_predecessors: dict[tuple[str, str], int] = supervised_by_cell
    for (old_cluster, new_cluster), counts in cells.items():
        predecessor_cells[new_cluster].append((old_cluster, counts))
    destination_rows: list[dict[str, object]] = []
    for new_cluster, metadata in right_metadata.items():
        predecessors = predecessor_cells.get(new_cluster, [])
        ranked = sorted(predecessors, key=lambda item: (sum(item[1]), item[0]), reverse=True)
        primary = ranked[0][0] if ranked else ""
        common_members = sum(sum(counts) for _, counts in ranked)
        other_members = sum(sum(counts) for cluster, counts in ranked if cluster != primary)
        other_linked = sum(counts[1] for cluster, counts in ranked if cluster != primary)
        other_background = sum(counts[0] for cluster, counts in ranked if cluster != primary)
        supervised_common = sum(supervised_predecessors[old, new_cluster] for old, _ in ranked)
        supervised_other = sum(
            supervised_predecessors[old, new_cluster] for old, _ in ranked if old != primary
        )
        destination_rows.append({
            "left": left_label,
            "right": right_label,
            "new_cluster": new_cluster,
            "new_split": metadata["split"],
            "predecessor_clusters": len(predecessors),
            "primary_predecessor": primary,
            "new_members": int(metadata["members"]),
            "common_members": common_members,
            "new_only_members": int(metadata["members"]) - common_members,
            "members_from_other_predecessors": other_members,
            "qualifying_linked_from_other_predecessors": other_linked,
            "background_from_other_predecessors": other_background,
            "supervised_proteins_from_predecessors": supervised_common,
            "supervised_proteins_from_other_predecessors": supervised_other,
        })

    full_metrics = partition_metrics(
        (sum(value) for value in left_common.values()),
        (sum(value) for value in right_common.values()),
        (sum(value) for value in cells.values()),
        common,
    )
    supervised_metrics = partition_metrics(
        supervised_left_counts.values(),
        supervised_right_counts.values(),
        supervised_cells.values(),
        len(left_supervised),
    )
    same_supervised_split = sum(
        count for (old_split, new_split), count in supervised_transitions.items()
        if old_split == new_split
    )
    full_left_partners: dict[str, set[str]] = defaultdict(set)
    full_right_partners: dict[str, set[str]] = defaultdict(set)
    for left_cluster, right_cluster in cells:
        full_left_partners[left_cluster].add(right_cluster)
        full_right_partners[right_cluster].add(left_cluster)
    supervised_left_partners: dict[str, set[str]] = defaultdict(set)
    supervised_right_partners: dict[str, set[str]] = defaultdict(set)
    for left_cluster, right_cluster in supervised_cells:
        supervised_left_partners[left_cluster].add(right_cluster)
        supervised_right_partners[right_cluster].add(left_cluster)
    full_partition_identical = (
        left_only == 0
        and right_only == 0
        and all(len(full_left_partners[cluster]) == 1 for cluster in left_metadata)
        and all(len(full_right_partners[cluster]) == 1 for cluster in right_metadata)
    )
    supervised_partition_identical = (
        all(len(supervised_left_partners[cluster]) == 1 for cluster in supervised_left_counts)
        and all(len(supervised_right_partners[cluster]) == 1 for cluster in supervised_right_counts)
    )
    return {
        "summary": {
            "left": left_label,
            "right": right_label,
            "common_members": common,
            "left_only_members": left_only,
            "right_only_members": right_only,
            "member_label_flag_changes": sum(count for pair, count in flag_changes.items() if pair[0] != pair[1]),
            "full_partition_ari": full_metrics["adjusted_rand_index"],
            "full_pair_jaccard": full_metrics["pair_jaccard"],
            "supervised_partition_ari": supervised_metrics["adjusted_rand_index"],
            "supervised_pair_jaccard": supervised_metrics["pair_jaccard"],
            "supervised_proteins": len(left_supervised),
            "supervised_same_split": same_supervised_split,
            "supervised_changed_split": len(left_supervised) - same_supervised_split,
            "supervised_same_split_fraction": same_supervised_split / len(left_supervised),
            "full_partition_identical": full_partition_identical,
            "supervised_partition_identical": supervised_partition_identical,
            "supervised_split_assignment_identical": same_supervised_split == len(left_supervised),
        },
        "pair_changes": pair_change_rows,
        "lineages": lineage_rows,
        "destinations": destination_rows,
        "category_summary": [
            {
                "left": left_label,
                "right": right_label,
                "category": category,
                "components": count,
                "common_members": category_members[category],
                "old_supervised_proteins": category_old_supervised[category],
                "new_supervised_proteins": category_new_supervised[category],
            }
            for category, count in sorted(category_summary.items())
        ],
        "full_transitions": [
            {"left": left_label, "right": right_label, "member_type": member_type, "old_split": old, "new_split": new, "members": count}
            for (member_type, old, new), count in sorted(full_transitions.items())
        ],
        "supervised_transitions": [
            {"left": left_label, "right": right_label, "old_split": old, "new_split": new, "proteins": count}
            for (old, new), count in sorted(supervised_transitions.items())
        ],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    specs = [parse_spec(raw) for raw in args.threshold_dir]
    output = args.output_dir.expanduser().resolve()
    scratch = args.scratch_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict[str, dict[str, int | str]]] = {}
    supervised: dict[str, dict[str, tuple[str, str, str]]] = {}
    normalized: dict[str, Path] = {}
    threshold_summaries: list[dict[str, object]] = []
    aspect_splits: dict[str, dict[str, dict[str, str]]] = {}

    for position, (label, root) in enumerate(specs):
        log(f"{label}: reading cluster and supervised assignments")
        metadata[label] = read_cluster_metadata(root / CLUSTER_FILE)
        supervised[label] = read_supervised(root / PROTEIN_FILE, metadata[label])
        aspect_splits[label] = {
            aspect: read_aspect_splits(root, aspect) for aspect in ASPECTS
        }
        normalized[label] = scratch / f"members_{position:03d}.tsv"
        log(f"{label}: normalizing and sorting retained members")
        member_summary = normalize_members(
            root / MEMBER_FILE, normalized[label], scratch, metadata[label]
        )
        threshold_summaries.append({
            "threshold": label,
            "clusters": len(metadata[label]),
            "supervised_proteins": len(supervised[label]),
            **member_summary,
        })

    summary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    full_transition_rows: list[dict[str, object]] = []
    supervised_transition_rows: list[dict[str, object]] = []
    aspect_transition_rows: list[dict[str, object]] = []
    test_overlap_rows: list[dict[str, object]] = []
    lineage_fields = [
        "left", "right", "component_id", "category", "old_clusters", "new_clusters",
        "old_members", "new_members", "common_members", "old_only_members",
        "new_only_members", "old_supervised_proteins", "new_supervised_proteins",
    ]
    destination_fields = [
        "left", "right", "new_cluster", "new_split", "predecessor_clusters",
        "primary_predecessor", "new_members", "common_members", "new_only_members",
        "members_from_other_predecessors", "qualifying_linked_from_other_predecessors",
        "background_from_other_predecessors", "supervised_proteins_from_predecessors",
        "supervised_proteins_from_other_predecessors",
    ]
    with gzip.open(output / "cluster_lineages.tsv.gz", "wt", encoding="utf-8", newline="") as lineage_handle, gzip.open(
        output / "destination_cluster_changes.tsv.gz", "wt", encoding="utf-8", newline=""
    ) as destination_handle:
        lineage_writer = csv.DictWriter(lineage_handle, fieldnames=lineage_fields, delimiter="\t", lineterminator="\n")
        destination_writer = csv.DictWriter(destination_handle, fieldnames=destination_fields, delimiter="\t", lineterminator="\n")
        lineage_writer.writeheader()
        destination_writer.writeheader()
        for (left_label, _), (right_label, _) in zip(specs, specs[1:]):
            log(f"comparing {left_label} to {right_label}")
            result = compare_pair(
                left_label,
                right_label,
                normalized[left_label],
                normalized[right_label],
                metadata[left_label],
                metadata[right_label],
                supervised[left_label],
                supervised[right_label],
            )
            summary_rows.append(result["summary"])
            pair_rows.extend(result["pair_changes"])
            category_rows.extend(result["category_summary"])
            full_transition_rows.extend(result["full_transitions"])
            supervised_transition_rows.extend(result["supervised_transitions"])
            lineage_writer.writerows(result["lineages"])
            destination_writer.writerows(result["destinations"])
            for (old_split, new_split), count in sorted(
                Counter({
                    (row["old_split"], row["new_split"]): row["proteins"]
                    for row in result["supervised_transitions"]
                }).items()
            ):
                aspect_transition_rows.append({
                    "left": left_label,
                    "right": right_label,
                    "aspect": "all",
                    "transition": f"{old_split}->{new_split}",
                    "proteins": count,
                })
            for aspect in ASPECTS:
                left_states = aspect_splits[left_label][aspect]
                right_states = aspect_splits[right_label][aspect]
                for transition, count in sorted(transition_counts(left_states, right_states).items()):
                    aspect_transition_rows.append({
                        "left": left_label,
                        "right": right_label,
                        "aspect": aspect,
                        "transition": transition,
                        "proteins": count,
                    })
                left_test = {
                    protein for protein, split in left_states.items() if split == "test"
                }
                right_test = {
                    protein for protein, split in right_states.items() if split == "test"
                }
                shared = left_test & right_test
                union = left_test | right_test
                test_overlap_rows.append({
                    "left": left_label,
                    "right": right_label,
                    "aspect": aspect,
                    "left_test": len(left_test),
                    "right_test": len(right_test),
                    "shared_test": len(shared),
                    "left_only": len(left_test - right_test),
                    "right_only": len(right_test - left_test),
                    "jaccard": ratio(len(shared), len(union)),
                })
    common_test_rows = []
    for aspect in ASPECTS:
        populations = [
            {
                protein for protein, split in aspect_splits[label][aspect].items()
                if split == "test"
            }
            for label, _ in specs
        ]
        intersection = set.intersection(*populations)
        union = set.union(*populations)
        common_test_rows.append({
            "aspect": aspect,
            "thresholds": ",".join(label for label, _ in specs),
            "common_test_proteins": len(intersection),
            "union_test_proteins": len(union),
            "intersection_over_union": ratio(len(intersection), len(union)),
        })

    (output / "threshold_summary.tsv").write_text(tsv(threshold_summaries, [
        "threshold", "clusters", "supervised_proteins", "members",
        "qualifying_linked_members", "background_members",
    ]), encoding="utf-8")
    (output / "consecutive_summary.tsv").write_text(tsv(summary_rows, [
        "left", "right", "common_members", "left_only_members", "right_only_members",
        "member_label_flag_changes", "full_partition_ari", "full_pair_jaccard",
        "supervised_partition_ari", "supervised_pair_jaccard", "supervised_proteins",
        "supervised_same_split", "supervised_changed_split", "supervised_same_split_fraction",
        "full_partition_identical", "supervised_partition_identical",
        "supervised_split_assignment_identical",
    ]), encoding="utf-8")
    (output / "co_cluster_pair_changes.tsv").write_text(tsv(pair_rows, [
        "left", "right", "population", "pair_type", "old_same_cluster_pairs",
        "new_same_cluster_pairs", "retained_same_cluster_pairs",
        "gained_same_cluster_pairs", "lost_same_cluster_pairs",
    ]), encoding="utf-8")
    (output / "cluster_lineage_categories.tsv").write_text(tsv(category_rows, [
        "left", "right", "category", "components", "common_members",
        "old_supervised_proteins", "new_supervised_proteins",
    ]), encoding="utf-8")
    (output / "full_member_split_transitions.tsv").write_text(tsv(full_transition_rows, [
        "left", "right", "member_type", "old_split", "new_split", "members",
    ]), encoding="utf-8")
    (output / "supervised_split_transitions.tsv").write_text(tsv(supervised_transition_rows, [
        "left", "right", "old_split", "new_split", "proteins",
    ]), encoding="utf-8")
    (output / "aspect_split_transitions.tsv").write_text(tsv(aspect_transition_rows, [
        "left", "right", "aspect", "transition", "proteins",
    ]), encoding="utf-8")
    (output / "consecutive_test_overlap.tsv").write_text(tsv(test_overlap_rows, [
        "left", "right", "aspect", "left_test", "right_test", "shared_test",
        "left_only", "right_only", "jaccard",
    ]), encoding="utf-8")
    (output / "common_test_population.tsv").write_text(tsv(common_test_rows, [
        "aspect", "thresholds", "common_test_proteins", "union_test_proteins", "intersection_over_union",
    ]), encoding="utf-8")
    report = {
        "thresholds": [label for label, _ in specs],
        "threshold_summaries": threshold_summaries,
        "consecutive_summaries": summary_rows,
        "common_test_populations": common_test_rows,
        "interpretation_boundary": (
            "Clusters are operational MMseqs2 or supplied clique groups. Qualifying-linked members "
            "are distinct from canonical supervised UniProt proteins, and neither proves evolutionary ancestry."
        ),
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Consecutive homology-partition comparison",
        "",
        "This analysis compares each supplied threshold with the next at both the full retained cluster-member level and the canonical supervised-protein level.",
        "",
        "## Consecutive summary",
        "",
        "| Pair | Common full members | Full partition identical | Full ARI | Supervised partition identical | Supervised ARI | Labelled proteins changing split | Labelled proteins staying |",
        "|---|---:|:---:|---:|:---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['left']} to {row['right']} | {int(row['common_members']):,} | "
            f"{'yes' if row['full_partition_identical'] else 'no'} | "
            f"{float(row['full_partition_ari']):.6f} | "
            f"{'yes' if row['supervised_partition_identical'] else 'no'} | "
            f"{float(row['supervised_partition_ari']):.6f} | "
            f"{int(row['supervised_changed_split']):,} | {100 * float(row['supervised_same_split_fraction']):.2f}% |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- Full members are retained sequence or clique members; they are not all independently labelled UniProt entries.",
        "- `connected_to_qualifying_uniprot` separates qualifying-linked members from background members.",
        "- Canonical supervised UniProt proteins are counted independently from `protein_cluster_assignments.tsv`.",
        "- Aspect and global split transitions are calculated directly from the supplied build directories.",
        "- Similar cluster counts do not imply identical clusters or model-facing splits.",
    ])
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-dir", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps({
        "thresholds": report["thresholds"],
        "comparisons": len(report["consecutive_summaries"]),
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
