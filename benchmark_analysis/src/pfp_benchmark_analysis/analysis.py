"""Descriptive joins and summaries across one or more PFP benchmarks."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

from .config import DatasetConfig, RunConfig
from .models import AnalysisBundle, DatasetResult, Observation
from .ontology import (
    ASPECTS,
    ASPECT_TO_FILE,
    ASPECT_TO_NAMESPACE,
    ASPECT_TO_ROOT,
    SPLITS,
    Ontology,
)
from .readers import (
    load_category_source,
    load_modality_availability,
    load_source_annotations,
    load_taxonomy,
    read_benchmark_csv,
)


ALL = "all"


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def describe(values: Iterable[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "p75": _quantile(ordered, 0.75),
        "p90": _quantile(ordered, 0.9),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _flatten(prefix: str, values: Mapping[str, object]) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _scope_observations(
    observations: Sequence[Observation], aspect: str, split: str
) -> Tuple[Observation, ...]:
    return tuple(
        item
        for item in observations
        if (aspect == ALL or item.aspect == aspect)
        and (split == ALL or item.split == split)
    )


def _scope_ids(
    observations: Sequence[Observation], aspect: str, split: str
) -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.protein_id
                for item in _scope_observations(observations, aspect, split)
            }
        )
    )


def _scopes() -> Tuple[Tuple[str, str], ...]:
    result = []
    for aspect in ASPECTS:
        for split in SPLITS:
            result.append((aspect, split))
        result.append((aspect, ALL))
    for split in SPLITS:
        result.append((ALL, split))
    result.append((ALL, ALL))
    return tuple(result)


def _load_dataset(config: DatasetConfig) -> Tuple[DatasetResult, Ontology]:
    ontology = Ontology.load(config.obo_file)
    observations: List[Observation] = []
    file_profiles = []
    term_headers: Dict[str, Tuple[str, ...]] = {}
    sequence_by_protein: Dict[str, str] = {}
    for aspect in ASPECTS:
        for split in SPLITS:
            path = config.benchmark_dir / f"{ASPECT_TO_FILE[aspect]}-{split}.csv"
            rows, profile, terms, row_sequences = read_benchmark_csv(
                path,
                dataset_id=config.id,
                aspect=aspect,
                split=split,
                root=ASPECT_TO_ROOT[aspect],
            )
            term_headers[aspect] = terms
            file_profiles.append(profile)
            observations.extend(rows)
            for item in rows:
                sequence = row_sequences[item.protein_id]
                sequence_by_protein[item.protein_id] = sequence

    source_by_split: Mapping[str, Mapping[str, object]] = {
        split: {} for split in SPLITS
    }
    if config.source_annotations:
        source_by_split, _source_paths = load_source_annotations(config.source_annotations)

    wanted_ids = set(sequence_by_protein)
    taxonomy, _taxonomy_paths, taxonomy_conflicts = load_taxonomy(
        config.taxonomy_sources, sequence_by_protein
    )
    modality_states: Mapping[Tuple[str, str], Mapping[str, bool]] = {}
    modalities: Tuple[str, ...] = tuple()
    if config.modality_availability:
        modality_states, modalities = load_modality_availability(
            config.modality_availability
        )
    category_maps = {
        spec.name: load_category_source(spec) for spec in config.category_sources
    }
    extras = sum(
        protein_id not in wanted_ids for protein_id, _modality in modality_states
    )
    diagnostics = {
        "unique_proteins": len(wanted_ids),
        "observation_rows": len(observations),
        "source_annotations_configured": config.source_annotations is not None,
        "taxonomy_sources_configured": len(config.taxonomy_sources),
        "taxonomy_mapped_unique_proteins": len(taxonomy),
        "taxonomy_conflict_proteins": len(
            {item.protein_id for item in taxonomy_conflicts}
        ),
        "taxonomy_conflict_observations": len(taxonomy_conflicts),
        "taxonomy_unresolved_conflict_proteins": len(
            {
                item.protein_id
                for item in taxonomy_conflicts
                if item.status == "unresolved"
            }
        ),
        "taxonomy_sequence_resolved_proteins": sum(
            item.resolution_basis
            == "unique taxon identified by exact benchmark sequence"
            for item in taxonomy.values()
        ),
        "taxonomy_secondary_alias_proteins": sum(
            item.accession_role == "secondary" for item in taxonomy.values()
        ),
        "modality_availability_configured": config.modality_availability is not None,
        "modality_availability_extra_pairs": extras,
        "category_sources": sorted(category_maps),
    }
    return DatasetResult(
        dataset_id=config.id,
        config=config,
        observations=tuple(observations),
        file_profiles=tuple(file_profiles),
        term_headers=term_headers,
        sequences=sequence_by_protein,
        source_by_split=source_by_split,
        taxonomy=taxonomy,
        taxonomy_conflicts=taxonomy_conflicts,
        modality_states=modality_states,
        modalities=modalities,
        category_maps=category_maps,
        diagnostics=diagnostics,
    ), ontology


def _label_profiles(dataset: DatasetResult) -> Tuple[dict, ...]:
    rows = []
    for aspect in ASPECTS:
        for split in (*SPLITS, ALL):
            observations = _scope_observations(dataset.observations, aspect, split)
            if not observations:
                continue
            root_only = sum(item.root_only for item in observations)
            all_zero = sum(item.all_zero for item in observations)
            labels = describe(item.label_count for item in observations)
            non_root = describe(item.non_root_label_count for item in observations)
            lengths = describe(item.sequence_length for item in observations)
            row = {
                "dataset_id": dataset.dataset_id,
                "aspect": aspect,
                "split": split,
                "proteins": len(observations),
                "terms": len(dataset.term_headers[aspect]),
                "root_positive_proteins": sum(
                    item.root_positive for item in observations
                ),
                "root_only_proteins": root_only,
                "root_only_fraction": root_only / len(observations),
                "all_zero_proteins": all_zero,
                "all_zero_fraction": all_zero / len(observations),
                "proteins_with_non_root_labels": len(observations)
                - root_only
                - all_zero,
                "positive_labels": sum(item.label_count for item in observations),
            }
            row.update(_flatten("labels", labels))
            row.update(_flatten("non_root_labels", non_root))
            row.update(_flatten("sequence_length", lengths))
            rows.append(row)
    return tuple(rows)


def _term_support(dataset: DatasetResult) -> Tuple[dict, ...]:
    rows = []
    for aspect in ASPECTS:
        root = ASPECT_TO_ROOT[aspect]
        for split in (*SPLITS, ALL):
            observations = _scope_observations(dataset.observations, aspect, split)
            counts = Counter(
                term for item in observations for term in item.positive_terms
            )
            for term in dataset.term_headers[aspect]:
                rows.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "aspect": aspect,
                        "split": split,
                        "term": term,
                        "is_root": term == root,
                        "support": counts[term],
                        "fraction": counts[term] / len(observations),
                    }
                )
    return tuple(rows)


def _root_only_source(
    dataset: DatasetResult, ontology: Ontology
) -> Tuple[Tuple[dict, ...], Tuple[dict, ...]]:
    detail = []
    policy = (
        dataset.config.source_annotations.projection_policy
        if dataset.config.source_annotations
        else "unavailable"
    )
    for item in dataset.observations:
        if not item.root_only:
            continue
        source = dataset.source_by_split[item.split].get(item.protein_id)
        source_terms = []
        unresolved_terms = []
        if source is None:
            classification = "source_unresolved"
            reason = "no matching pre-projection source annotation record"
        else:
            for raw_term in source.annotations:
                canonical = ontology.resolve(raw_term)
                if canonical is None:
                    unresolved_terms.append(raw_term)
                elif ontology.namespace(canonical) == ASPECT_TO_NAMESPACE[item.aspect]:
                    source_terms.append(canonical)
            source_set = set(source_terms)
            source_non_root = source_set - {ASPECT_TO_ROOT[item.aspect]}
            if source_non_root:
                classification = "projection_created"
                reason = policy
            elif ASPECT_TO_ROOT[item.aspect] in source_set:
                classification = "source_root_only"
                reason = "source contained the aspect root and no non-root term"
            else:
                classification = "source_no_aspect_annotation"
                reason = "source record contained no resolvable term in this aspect"
        source_non_root_terms = sorted(
            set(source_terms) - {ASPECT_TO_ROOT[item.aspect]}
        )
        detail.append(
            {
                "dataset_id": dataset.dataset_id,
                "aspect": item.aspect,
                "split": item.split,
                "protein_id": item.protein_id,
                "classification": classification,
                "reason": reason,
                "source_aspect_term_count": len(set(source_terms)),
                "source_non_root_term_count": len(source_non_root_terms),
                "source_non_root_terms": ";".join(source_non_root_terms),
                "unresolved_source_term_count": len(set(unresolved_terms)),
                "unresolved_source_terms": ";".join(sorted(set(unresolved_terms))),
            }
        )
    summary = []
    for aspect in ASPECTS:
        for split in (*SPLITS, ALL):
            observations = _scope_observations(dataset.observations, aspect, split)
            relevant = [
                row
                for row in detail
                if row["aspect"] == aspect and (split == ALL or row["split"] == split)
            ]
            counts = Counter(row["classification"] for row in relevant)
            root_only_count = len(relevant)
            total = len(observations)
            for classification in (
                "source_root_only",
                "projection_created",
                "source_no_aspect_annotation",
                "source_unresolved",
            ):
                count = counts[classification]
                summary.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "aspect": aspect,
                        "split": split,
                        "classification": classification,
                        "proteins": count,
                        "root_only_proteins": root_only_count,
                        "total_proteins": total,
                        "fraction_of_root_only": (
                            count / root_only_count if root_only_count else 0.0
                        ),
                        "fraction_of_total": count / total if total else 0.0,
                        "projection_policy": policy,
                    }
                )
    return tuple(detail), tuple(summary)


def _taxonomy_tables(
    dataset: DatasetResult,
) -> Tuple[Tuple[dict, ...], Tuple[dict, ...]]:
    distribution = []
    coverage = []
    for aspect, split in _scopes():
        protein_ids = _scope_ids(dataset.observations, aspect, split)
        counts: Counter[Tuple[str, str]] = Counter()
        mapped = 0
        for protein_id in protein_ids:
            taxon = dataset.taxonomy.get(protein_id)
            if taxon is None:
                counts[("__UNMAPPED__", "Unmapped taxonomy")] += 1
            else:
                mapped += 1
                counts[(taxon.taxon_id, taxon.taxon_name)] += 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for rank, ((taxon_id, taxon_name), count) in enumerate(ordered, start=1):
            distribution.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "aspect": aspect,
                    "split": split,
                    "rank": rank,
                    "taxon_id": taxon_id,
                    "taxon_name": taxon_name,
                    "proteins": count,
                    "scope_proteins": len(protein_ids),
                    "mapped_proteins": mapped,
                    "fraction_of_total": count / len(protein_ids)
                    if protein_ids
                    else 0.0,
                    "fraction_of_mapped": (
                        count / mapped
                        if mapped and taxon_id != "__UNMAPPED__"
                        else None
                    ),
                }
            )
        coverage.append(
            {
                "dataset_id": dataset.dataset_id,
                "aspect": aspect,
                "split": split,
                "proteins": len(protein_ids),
                "mapped_proteins": mapped,
                "unmapped_proteins": len(protein_ids) - mapped,
                "mapping_fraction": mapped / len(protein_ids) if protein_ids else 0.0,
            }
        )
    return tuple(distribution), tuple(coverage)


def _taxonomy_conflict_table(dataset: DatasetResult) -> Tuple[dict, ...]:
    return tuple(
        {
            "dataset_id": dataset.dataset_id,
            "protein_id": item.protein_id,
            "status": item.status,
            "selected_taxon_id": item.selected.taxon_id if item.selected else "",
            "selected_taxon_name": item.selected.taxon_name if item.selected else "",
            "selected_source_name": item.selected.source_name if item.selected else "",
            "selected_source_path": item.selected.source if item.selected else "",
            "selected_source_priority": (
                item.selected.source_priority if item.selected else ""
            ),
            "selected_accession_role": (
                item.selected.accession_role if item.selected else ""
            ),
            "selected_record_primary_accession": (
                item.selected.record_primary_accession if item.selected else ""
            ),
            "selected_sequence_matches_benchmark": (
                item.selected.sequence_matches_benchmark if item.selected else ""
            ),
            "selected_resolution_basis": (
                item.selected.resolution_basis if item.selected else ""
            ),
            "alternative_taxon_id": item.alternative.taxon_id,
            "alternative_taxon_name": item.alternative.taxon_name,
            "alternative_source_name": item.alternative.source_name,
            "alternative_source_path": item.alternative.source,
            "alternative_source_priority": item.alternative.source_priority,
            "alternative_accession_role": item.alternative.accession_role,
            "alternative_record_primary_accession": (
                item.alternative.record_primary_accession
            ),
            "alternative_sequence_matches_benchmark": (
                item.alternative.sequence_matches_benchmark
            ),
            "resolution": item.resolution,
        }
        for item in dataset.taxonomy_conflicts
    )


def _category_tables(dataset: DatasetResult) -> Tuple[dict, ...]:
    rows = []
    for source_name, mapping in dataset.category_maps.items():
        for aspect, split in _scopes():
            protein_ids = _scope_ids(dataset.observations, aspect, split)
            counts: Counter[Tuple[str, str]] = Counter()
            mapped = 0
            for protein_id in protein_ids:
                categories = mapping.get(protein_id, tuple())
                if not categories:
                    counts[("__UNMAPPED__", "Unmapped category")] += 1
                    continue
                mapped += 1
                counts.update(set(categories))
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            for rank, ((category_id, category_name), count) in enumerate(
                ordered, start=1
            ):
                rows.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "category_source": source_name,
                        "aspect": aspect,
                        "split": split,
                        "rank": rank,
                        "category_id": category_id,
                        "category_name": category_name,
                        "proteins": count,
                        "scope_proteins": len(protein_ids),
                        "mapped_proteins": mapped,
                        "fraction_of_total": count / len(protein_ids)
                        if protein_ids
                        else 0.0,
                        "note": "Fractions may sum above one when proteins have multiple categories.",
                    }
                )
    return tuple(rows)


def _modality_tables(
    dataset: DatasetResult,
) -> Tuple[Tuple[dict, ...], Tuple[dict, ...]]:
    coverage = []
    patterns = []
    if not dataset.config.modality_availability:
        return tuple(), tuple()
    state_names = tuple(dataset.config.modality_availability.states)
    for aspect, split in _scopes():
        protein_ids = _scope_ids(dataset.observations, aspect, split)
        for modality in dataset.modalities:
            pairs = {
                protein_id: dataset.modality_states.get((protein_id, modality))
                for protein_id in protein_ids
            }
            recorded = sum(value is not None for value in pairs.values())
            coverage.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "aspect": aspect,
                    "split": split,
                    "modality": modality,
                    "coverage_state": "availability_recorded",
                    "covered_proteins": recorded,
                    "total_proteins": len(protein_ids),
                    "coverage_fraction": recorded / len(protein_ids)
                    if protein_ids
                    else 0.0,
                }
            )
            for state_name in state_names:
                covered = sum(
                    bool(value and value.get(state_name)) for value in pairs.values()
                )
                coverage.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "aspect": aspect,
                        "split": split,
                        "modality": modality,
                        "coverage_state": state_name,
                        "covered_proteins": covered,
                        "total_proteins": len(protein_ids),
                        "coverage_fraction": (
                            covered / len(protein_ids) if protein_ids else 0.0
                        ),
                    }
                )
        for state_name in state_names:
            counts = Counter()
            for protein_id in protein_ids:
                present = tuple(
                    modality
                    for modality in dataset.modalities
                    if dataset.modality_states.get((protein_id, modality), {}).get(
                        state_name, False
                    )
                )
                counts["+".join(present) if present else "none"] += 1
            for pattern, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            ):
                patterns.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "aspect": aspect,
                        "split": split,
                        "coverage_state": state_name,
                        "modality_pattern": pattern,
                        "proteins": count,
                        "total_proteins": len(protein_ids),
                        "fraction": count / len(protein_ids) if protein_ids else 0.0,
                    }
                )
    return tuple(coverage), tuple(patterns)


def _membership_table(dataset: DatasetResult) -> Tuple[dict, ...]:
    memberships: MutableMapping[str, Set[Tuple[str, str]]] = defaultdict(set)
    for item in dataset.observations:
        memberships[item.protein_id].add((item.aspect, item.split))
    resolved_conflicts = {
        item.protein_id
        for item in dataset.taxonomy_conflicts
        if item.status == "resolved"
    }
    unresolved_conflicts = {
        item.protein_id
        for item in dataset.taxonomy_conflicts
        if item.status == "unresolved"
    }
    rows = []
    for protein_id, pairs in sorted(memberships.items()):
        taxon = dataset.taxonomy.get(protein_id)
        rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "protein_id": protein_id,
                "aspects": ";".join(sorted(aspect for aspect, _split in pairs)),
                "splits": ";".join(sorted({split for _aspect, split in pairs})),
                "memberships": ";".join(
                    f"{aspect}:{split}" for aspect, split in sorted(pairs)
                ),
                "taxonomy_mapped": taxon is not None,
                "taxon_id": taxon.taxon_id if taxon else "",
                "taxon_name": taxon.taxon_name if taxon else "",
                "taxonomy_source_name": taxon.source_name if taxon else "",
                "taxonomy_source_path": taxon.source if taxon else "",
                "taxonomy_source_priority": (taxon.source_priority if taxon else ""),
                "taxonomy_accession_role": taxon.accession_role if taxon else "",
                "taxonomy_record_primary_accession": (
                    taxon.record_primary_accession if taxon else ""
                ),
                "taxonomy_sequence_matches_benchmark": (
                    taxon.sequence_matches_benchmark if taxon else ""
                ),
                "taxonomy_resolution_basis": taxon.resolution_basis if taxon else "",
                "taxonomy_conflict_resolved": protein_id in resolved_conflicts,
                "taxonomy_conflict_unresolved": protein_id in unresolved_conflicts,
            }
        )
    return tuple(rows)


def _cross_metrics(
    dataset_tables: Mapping[str, Mapping[str, Tuple[dict, ...]]],
) -> Tuple[dict, ...]:
    rows = []
    for reference_id, comparison_id in combinations(dataset_tables, 2):
        reference = {
            (row["aspect"], row["split"]): row
            for row in dataset_tables[reference_id]["label_profiles"]
        }
        comparison = {
            (row["aspect"], row["split"]): row
            for row in dataset_tables[comparison_id]["label_profiles"]
        }
        for key in sorted(set(reference) & set(comparison)):
            aspect, split = key
            for metric in (
                "proteins",
                "terms",
                "root_only_fraction",
                "all_zero_fraction",
                "labels_mean",
                "non_root_labels_mean",
                "sequence_length_mean",
            ):
                left = reference[key][metric]
                right = comparison[key][metric]
                if left is None or right is None:
                    continue
                rows.append(
                    {
                        "reference_dataset": reference_id,
                        "comparison_dataset": comparison_id,
                        "aspect": aspect,
                        "split": split,
                        "metric": metric,
                        "reference_value": left,
                        "comparison_value": right,
                        "absolute_delta": right - left,
                        "relative_delta": ((right - left) / left if left else None),
                    }
                )
    return tuple(rows)


def _cross_modality(
    dataset_tables: Mapping[str, Mapping[str, Tuple[dict, ...]]],
) -> Tuple[dict, ...]:
    rows = []
    for reference_id, comparison_id in combinations(dataset_tables, 2):
        reference = {
            (row["aspect"], row["split"], row["modality"], row["coverage_state"]): row
            for row in dataset_tables[reference_id]["modality_coverage"]
        }
        comparison = {
            (row["aspect"], row["split"], row["modality"], row["coverage_state"]): row
            for row in dataset_tables[comparison_id]["modality_coverage"]
        }
        for key in sorted(set(reference) & set(comparison)):
            left = reference[key]["coverage_fraction"]
            right = comparison[key]["coverage_fraction"]
            rows.append(
                {
                    "reference_dataset": reference_id,
                    "comparison_dataset": comparison_id,
                    "aspect": key[0],
                    "split": key[1],
                    "modality": key[2],
                    "coverage_state": key[3],
                    "reference_fraction": left,
                    "comparison_fraction": right,
                    "absolute_delta": right - left,
                }
            )
    return tuple(rows)


def _cross_taxonomy(
    dataset_tables: Mapping[str, Mapping[str, Tuple[dict, ...]]],
) -> Tuple[dict, ...]:
    rows = []
    for reference_id, comparison_id in combinations(dataset_tables, 2):
        reference = {
            (row["aspect"], row["split"], row["taxon_id"]): row
            for row in dataset_tables[reference_id]["taxonomy_distribution"]
        }
        comparison = {
            (row["aspect"], row["split"], row["taxon_id"]): row
            for row in dataset_tables[comparison_id]["taxonomy_distribution"]
        }
        for key in sorted(set(reference) | set(comparison)):
            left_row = reference.get(key)
            right_row = comparison.get(key)
            left = left_row["fraction_of_total"] if left_row else 0.0
            right = right_row["fraction_of_total"] if right_row else 0.0
            rows.append(
                {
                    "reference_dataset": reference_id,
                    "comparison_dataset": comparison_id,
                    "aspect": key[0],
                    "split": key[1],
                    "taxon_id": key[2],
                    "taxon_name": (
                        (right_row or left_row)["taxon_name"]
                        if (right_row or left_row)
                        else ""
                    ),
                    "reference_fraction": left,
                    "comparison_fraction": right,
                    "absolute_delta": right - left,
                }
            )
    return tuple(rows)


def _cross_benchmark_root_source(
    dataset_tables: Mapping[str, Mapping[str, Tuple[dict, ...]]],
) -> Tuple[dict, ...]:
    rows = []
    for reference_id, comparison_id in combinations(dataset_tables, 2):
        reference = {
            (row["aspect"], row["split"], row["classification"]): row
            for row in dataset_tables[reference_id]["root_only_summary"]
        }
        comparison = {
            (row["aspect"], row["split"], row["classification"]): row
            for row in dataset_tables[comparison_id]["root_only_summary"]
        }
        for key in sorted(set(reference) & set(comparison)):
            left = reference[key]["fraction_of_total"]
            right = comparison[key]["fraction_of_total"]
            rows.append(
                {
                    "reference_dataset": reference_id,
                    "comparison_dataset": comparison_id,
                    "aspect": key[0],
                    "split": key[1],
                    "classification": key[2],
                    "reference_fraction_of_total": left,
                    "comparison_fraction_of_total": right,
                    "absolute_delta": right - left,
                }
            )
    return tuple(rows)


def analyse(config: RunConfig) -> AnalysisBundle:
    dataset_results: Dict[str, DatasetResult] = {}
    dataset_tables: Dict[str, Dict[str, Tuple[dict, ...]]] = {}
    for dataset_config in config.datasets:
        dataset, ontology = _load_dataset(dataset_config)
        dataset_results[dataset.dataset_id] = dataset
        root_detail, root_summary = _root_only_source(dataset, ontology)
        taxonomy_distribution, taxonomy_coverage = _taxonomy_tables(dataset)
        modality_coverage, modality_patterns = _modality_tables(dataset)
        dataset_tables[dataset.dataset_id] = {
            "file_profiles": dataset.file_profiles,
            "label_profiles": _label_profiles(dataset),
            "term_support": _term_support(dataset),
            "root_only_source": root_detail,
            "root_only_summary": root_summary,
            "taxonomy_distribution": taxonomy_distribution,
            "taxonomy_coverage": taxonomy_coverage,
            "taxonomy_conflicts": _taxonomy_conflict_table(dataset),
            "category_distribution": _category_tables(dataset),
            "modality_coverage": modality_coverage,
            "modality_patterns": modality_patterns,
            "protein_membership": _membership_table(dataset),
        }

    combined: Dict[str, List[dict]] = defaultdict(list)
    for tables in dataset_tables.values():
        for name, rows in tables.items():
            combined[name].extend(rows)
    combined["cross_benchmark_metrics"].extend(_cross_metrics(dataset_tables))
    combined["cross_benchmark_modality"].extend(_cross_modality(dataset_tables))
    combined["cross_benchmark_taxonomy"].extend(_cross_taxonomy(dataset_tables))
    combined["cross_benchmark_root_source"].extend(
        _cross_benchmark_root_source(dataset_tables)
    )
    summary = {
        "run_name": config.run_name,
        "dataset_order": [dataset.id for dataset in config.datasets],
        "datasets": {
            dataset_id: dataset_results[dataset_id].diagnostics
            for dataset_id in dataset_results
        },
        "interpretation_boundaries": [
            "Taxonomy distributions describe organisms, not protein families.",
            "UniProt alias conflicts are resolved only by a unique benchmark-sequence taxon; cross-source changes use explicit source priority. Zero-match, multi-match, and equal-priority conflicts are retained as unresolved and left taxonomically unmapped without aborting the benchmark analysis.",
            "Protein-family claims require an explicit category mapping source.",
            "Projection-created root-only rows are attributed to the configured projection policy only when pre-projection annotations are supplied.",
            "Modality availability reflects the state columns explicitly selected in the supplied long-table input.",
            "Cross-benchmark deltas are descriptive and do not establish a causal explanation for performance differences.",
            "Different label universes, taxa, target difficulty, annotation depth, and modality coverage remain partially confounded.",
        ],
    }
    return AnalysisBundle(
        summary=summary,
        tables={name: tuple(rows) for name, rows in combined.items()},
    )
