from __future__ import annotations

import hashlib
import heapq
from bisect import bisect_left
import random

from .models import ClusterInfo, SplitAssignment


def _bounded_group_size(count: int, fraction: float) -> int:
    return min(count - 1, max(1, int(round(count * fraction))))


def _cluster_count_partition(
    clusters: list[ClusterInfo], fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    identifiers = sorted(cluster.cluster_id for cluster in clusters)
    random.Random(seed).shuffle(identifiers)
    first_count = _bounded_group_size(len(identifiers), fraction)
    return set(identifiers[:first_count]), set(identifiers[first_count:])


def _seed_tie(seed: int, cluster_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{cluster_id}".encode("utf-8")).hexdigest()


def _membership_tie(seed: int, identifiers: set[str]) -> str:
    digest = hashlib.sha256(str(seed).encode("ascii"))
    for identifier in sorted(identifiers):
        digest.update(b"\0")
        digest.update(identifier.encode("utf-8"))
    return digest.hexdigest()


def _candidate_orders(clusters: list[ClusterInfo], seed: int) -> tuple[list[ClusterInfo], ...]:
    descending = sorted(
        clusters,
        key=lambda item: (-item.member_count, _seed_tie(seed, item.cluster_id), item.cluster_id),
    )
    ascending = list(reversed(descending))
    hashed = sorted(
        clusters, key=lambda item: (_seed_tie(seed + 101, item.cluster_id), item.cluster_id)
    )
    interleaved: list[ClusterInfo] = []
    left, right = 0, len(descending) - 1
    while left <= right:
        interleaved.append(descending[left])
        left += 1
        if left <= right:
            interleaved.append(descending[right])
            right -= 1
    return descending, ascending, hashed, interleaved


def _greedy_subset(
    ordered: list[ClusterInfo], target: float, seed: int
) -> tuple[set[str], int]:
    selected: set[str] = set()
    weight = 0
    for cluster in ordered:
        before = abs(target - weight)
        after = abs(target - weight - cluster.member_count)
        if after < before or (
            after == before and int(_seed_tie(seed + 17, cluster.cluster_id), 16) % 2 == 0
        ):
            selected.add(cluster.cluster_id)
            weight += cluster.member_count
    return selected, weight


def _bounded_improve_subset(
    clusters: list[ClusterInfo], selected: set[str], weight: int, target: float, seed: int
) -> tuple[set[str], int]:
    """Deterministic bounded local search; no global subset-sum optimality is claimed."""
    by_id = {item.cluster_id: item for item in clusters}
    all_ids = set(by_id)
    max_passes = 4
    swap_scan_limit = 4096
    pair_pool_limit = 96

    for pass_index in range(max_passes):
        current_score = abs(target - weight)
        best: tuple[float, str, tuple[str, ...], int] | None = None

        # Check every one-cluster toggle for a closer partition.
        for identifier in sorted(all_ids):
            delta = -by_id[identifier].member_count if identifier in selected else by_id[identifier].member_count
            new_size = len(selected) + (-1 if identifier in selected else 1)
            if not 0 < new_size < len(all_ids):
                continue
            new_weight = weight + delta
            score = abs(target - new_weight)
            key = (score, _seed_tie(seed + pass_index, identifier), (identifier,), new_weight)
            if score < current_score and (best is None or key < best):
                best = key

        # Check deterministic one-for-one swaps near the target.
        selected_scan = sorted(
            (by_id[item] for item in selected),
            key=lambda item: (_seed_tie(seed + 211, item.cluster_id), item.cluster_id),
        )[:swap_scan_limit]
        opposite = sorted(
            (by_id[item] for item in all_ids - selected),
            key=lambda item: (item.member_count, item.cluster_id),
        )
        opposite_weights = [item.member_count for item in opposite]
        for removed in selected_scan:
            desired = target - (weight - removed.member_count)
            position = bisect_left(opposite_weights, desired)
            for candidate_index in range(max(0, position - 2), min(len(opposite), position + 3)):
                added = opposite[candidate_index]
                new_weight = weight - removed.member_count + added.member_count
                score = abs(target - new_weight)
                identifiers = tuple(sorted((removed.cluster_id, added.cluster_id)))
                tie = _membership_tie(seed + pass_index + 307, set(identifiers))
                key = (score, tie, identifiers, new_weight)
                if score < current_score and (best is None or key < best):
                    best = key

        # Check two-cluster moves from a fixed-size pool.
        residual = abs(target - weight)
        pool = heapq.nsmallest(
            pair_pool_limit,
            clusters,
            key=lambda item: (
                abs(item.member_count - residual),
                _seed_tie(seed + 401, item.cluster_id),
                item.cluster_id,
            ),
        )
        for left_index, left in enumerate(pool):
            for right in pool[left_index + 1:]:
                changed = (left.cluster_id, right.cluster_id)
                delta = sum(
                    -by_id[item].member_count if item in selected else by_id[item].member_count
                    for item in changed
                )
                new_size = len(selected) + sum(-1 if item in selected else 1 for item in changed)
                if not 0 < new_size < len(all_ids):
                    continue
                new_weight = weight + delta
                score = abs(target - new_weight)
                tie = _membership_tie(seed + pass_index + 503, set(changed))
                key = (score, tie, tuple(sorted(changed)), new_weight)
                if score < current_score and (best is None or key < best):
                    best = key

        if best is None:
            break
        _, _, changed, new_weight = best
        for identifier in changed:
            if identifier in selected:
                selected.remove(identifier)
            else:
                selected.add(identifier)
        weight = new_weight
    return selected, weight


def _sequence_balanced_partition(
    clusters: list[ClusterInfo], fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    total = sum(cluster.member_count for cluster in clusters)
    # Build the smaller side to avoid 80%/90% overshoot from the largest-first heuristic.
    complement = fraction > 0.5
    target_fraction = 1.0 - fraction if complement else fraction
    target = total * target_fraction
    all_ids = {cluster.cluster_id for cluster in clusters}
    candidates: list[tuple[float, str, set[str], int]] = []
    for candidate_index, ordered in enumerate(_candidate_orders(clusters, seed)):
        selected, selected_weight = _greedy_subset(
            ordered, target, seed + candidate_index * 1009
        )
        if not selected:
            smallest = min(clusters, key=lambda item: (item.member_count, item.cluster_id))
            selected = {smallest.cluster_id}
            selected_weight = smallest.member_count
        if selected == all_ids:
            smallest = min(clusters, key=lambda item: (item.member_count, item.cluster_id))
            selected.remove(smallest.cluster_id)
            selected_weight -= smallest.member_count
        selected, selected_weight = _bounded_improve_subset(
            clusters, selected, selected_weight, target, seed + candidate_index * 2003
        )
        candidates.append((
            abs(target - selected_weight),
            _membership_tie(seed, selected),
            selected,
            selected_weight,
        ))
    _, _, smaller, _ = min(candidates, key=lambda item: (item[0], item[1]))
    if complement:
        first, second = all_ids - smaller, smaller
    else:
        first, second = smaller, all_ids - smaller
    return first, second


def _partition(
    clusters: list[ClusterInfo], fraction: float, policy: str, seed: int
) -> tuple[set[str], set[str]]:
    if policy == "cluster-count-random":
        return _cluster_count_partition(clusters, fraction, seed)
    if policy == "sequence-balanced":
        return _sequence_balanced_partition(clusters, fraction, seed)
    raise ValueError(f"Unknown split policy: {policy}")


def assign_development_test(
    retained: dict[str, ClusterInfo],
    policy: str,
    seed: int = 0,
    development_fraction: float = 0.80,
) -> dict[str, SplitAssignment]:
    clusters = [retained[key] for key in sorted(retained)]
    development_ids, test_ids = _partition(clusters, development_fraction, policy, seed)
    if len(development_ids) < 2:
        movable = min(
            (retained[key] for key in test_ids),
            key=lambda item: (item.member_count, item.cluster_id),
        )
        test_ids.remove(movable.cluster_id)
        development_ids.add(movable.cluster_id)
    assignments = {}
    for cluster_id in sorted(retained):
        split = "development" if cluster_id in development_ids else "test"
        info = retained[cluster_id]
        assignments[cluster_id] = SplitAssignment(
            cluster_id=cluster_id,
            split=split,
            member_count=info.member_count,
            labelled_protein_count=info.labelled_protein_count,
            stage="development-vs-test",
        )
    return assignments


def assign_training_validation(
    retained: dict[str, ClusterInfo],
    development_test: dict[str, SplitAssignment],
    policy: str,
    seed: int = 0,
    training_fraction_within_development: float = 0.90,
) -> dict[str, SplitAssignment]:
    development_ids = {
        key for key, assignment in development_test.items()
        if assignment.split == "development"
    }
    test_ids = {
        key for key, assignment in development_test.items() if assignment.split == "test"
    }
    development = [retained[key] for key in sorted(development_ids)]
    training_ids, validation_ids = _partition(
        development, training_fraction_within_development, policy, seed + 1,
    )

    assignments: dict[str, SplitAssignment] = {}
    for cluster_id in sorted(retained):
        if cluster_id in training_ids:
            split, stage = "training", "development-to-training"
        elif cluster_id in validation_ids:
            split, stage = "validation", "development-to-validation"
        elif cluster_id in test_ids:
            split, stage = "test", "development-vs-test"
        else:
            continue
        info = retained[cluster_id]
        assignments[cluster_id] = SplitAssignment(
            cluster_id=cluster_id,
            split=split,
            member_count=info.member_count,
            labelled_protein_count=info.labelled_protein_count,
            stage=stage,
        )
    return assignments


def assign_splits(
    retained: dict[str, ClusterInfo],
    policy: str,
    seed: int = 0,
    development_fraction: float = 0.80,
    training_fraction_within_development: float = 0.90,
) -> dict[str, SplitAssignment]:
    development_test = assign_development_test(
        retained, policy, seed, development_fraction
    )
    return assign_training_validation(
        retained,
        development_test,
        policy,
        seed,
        training_fraction_within_development,
    )
