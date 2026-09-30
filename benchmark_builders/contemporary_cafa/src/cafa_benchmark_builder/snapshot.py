from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import logging
from pathlib import Path

import pandas as pd

from .builder import (
    build_deepgoplus_dataframe_from_fasta,
    build_training_dataframe,
    load_deepgoplus_annotation_file,
    make_terms_dataframe,
    prepare_pfp_csv_frames,
    propagate_annotations,
    split_train_valid,
    write_pfp_csv_frames,
)
from .config import BuildConfig, PREFIX_TO_NAMESPACE
from .goa import load_normalized_annotation_map
from .models import AnnotationLoadResult, IdentityMatch, ProteinCatalog
from .official_targets import load_official_target_catalog
from .ontology import Ontology
from .parsers import load_protein_catalog
from .training_annotations import load_released_training_annotations


LOGGER = logging.getLogger(__name__)
PROTEIN_BINDING = "GO:0005515"
OFFICIAL_TARGET_FIELDS = [
    "snapshot", "target_id", "taxon_id", "sequence_length", "status", "reason", "mapping_method",
    "uniprot_accession", "uniprot_entry_name", "reviewed", "source_candidate_count",
    "exact_sequence_candidate_count", "present_in_snapshot",
]


def _catalog_from_records(catalog: ProteinCatalog, protein_ids: set[str]) -> ProteinCatalog:
    subset = ProteinCatalog()
    for protein_id in sorted(protein_ids):
        record = catalog.records[protein_id]
        subset.records[protein_id] = record
        for alias in set(record.accessions) | {protein_id}:
            if alias in catalog.ambiguous_aliases:
                subset.ambiguous_aliases.add(alias)
            elif catalog.alias_to_primary.get(alias) == protein_id:
                subset.alias_to_primary[alias] = protein_id
    return subset


def _merge_catalogs(*catalogs: ProteinCatalog) -> ProteinCatalog:
    merged = ProteinCatalog()
    for catalog in catalogs:
        for protein_id, record in catalog.records.items():
            merged.records.setdefault(protein_id, record)
        for alias in catalog.ambiguous_aliases:
            merged.ambiguous_aliases.add(alias)
            merged.alias_to_primary.pop(alias, None)
        for alias, protein_id in catalog.alias_to_primary.items():
            if alias in merged.ambiguous_aliases:
                continue
            previous = merged.alias_to_primary.get(alias)
            if previous is None or previous == protein_id:
                merged.alias_to_primary[alias] = protein_id
            else:
                merged.ambiguous_aliases.add(alias)
                del merged.alias_to_primary[alias]
    return merged


def _target_catalog(
    paths: tuple[Path, ...],
    training_catalog: ProteinCatalog,
    training_taxa: frozenset[str],
    target_taxa: frozenset[str],
    training_reviewed_only: bool,
    target_reviewed_only: bool,
) -> ProteinCatalog:
    matching = {
        protein_id for protein_id, record in training_catalog.records.items()
        if not target_taxa or record.taxon_id in target_taxa
    }
    # Reuse training records only when they cover the complete target population.
    training_covers_targets = (
        (not training_taxa or target_taxa <= training_taxa)
        and (not training_reviewed_only or target_reviewed_only)
    )
    if training_covers_targets:
        return _catalog_from_records(training_catalog, matching)
    return load_protein_catalog(paths, target_taxa, target_reviewed_only)


def _build_identity_crosswalk(
    t0_catalog: ProteinCatalog,
    t1_catalog: ProteinCatalog,
    sequence_change_policy: str,
) -> tuple[list[IdentityMatch], dict[str, str]]:
    preliminary: dict[str, tuple[str | None, str, bool]] = {}
    reverse: dict[str, list[str]] = defaultdict(list)

    for t0_id in sorted(t0_catalog.records):
        record = t0_catalog.records[t0_id]
        candidates = {
            t1_catalog.alias_to_primary[alias]
            for alias in set(record.accessions) | {t0_id}
            if alias in t1_catalog.alias_to_primary
        }
        if not candidates:
            preliminary[t0_id] = (None, "missing_at_t1", False)
        elif len(candidates) > 1:
            preliminary[t0_id] = (None, "ambiguous_t1_identity", False)
        else:
            t1_id = next(iter(candidates))
            changed = record.sequence != t1_catalog.records[t1_id].sequence
            preliminary[t0_id] = (t1_id, "matched", changed)
            reverse[t1_id].append(t0_id)

    matches: list[IdentityMatch] = []
    t1_to_t0: dict[str, str] = {}
    for t0_id in sorted(preliminary):
        t1_id, reason, changed = preliminary[t0_id]
        if t1_id is None:
            matches.append(IdentityMatch(t0_id, None, "excluded", reason, changed))
            continue
        if len(reverse[t1_id]) > 1:
            matches.append(IdentityMatch(t0_id, t1_id, "excluded", "many_t0_ids_map_to_one_t1", changed))
            continue
        if changed and sequence_change_policy == "error":
            raise ValueError(f"Sequence changed between t0 and t1 for {t0_id} -> {t1_id}")
        if changed and sequence_change_policy == "exclude":
            matches.append(IdentityMatch(t0_id, t1_id, "excluded", "sequence_changed", True))
            continue
        matches.append(IdentityMatch(t0_id, t1_id, "matched", "matched", changed))
        t1_to_t0[t1_id] = t0_id

    mapped_t1 = set(t1_to_t0)
    for t1_id in sorted(set(t1_catalog.records) - mapped_t1):
        if not any(match.t1_id == t1_id for match in matches):
            matches.append(IdentityMatch("", t1_id, "excluded", "not_present_at_t0", False))
    return matches, t1_to_t0


def _drop_protein_binding_only(terms: set[str], go: Ontology, policy: str) -> set[str]:
    if policy == "keep":
        return set(terms)
    if policy != "drop-mf-protein-binding-only":
        raise ValueError(f"Unknown protein-binding policy: {policy}")
    mf_terms = {term for term in terms if go.get_namespace(term) == "molecular_function"}
    if mf_terms == {PROTEIN_BINDING}:
        return set(terms) - mf_terms
    return set(terms)


def _test_eligible_terms(
    t0_terms: set[str],
    t1_terms: set[str],
    go: Ontology,
    policy: str,
) -> tuple[set[str], set[str]]:
    """Return newly assigned terms and t0-blocked namespaces for a test target."""
    if policy == "global-no-knowledge":
        blocked = {go.get_namespace(term) for term in t0_terms if go.has_term(term)}
        return (set() if t0_terms else set(t1_terms)), blocked
    if policy != "ontology-no-knowledge":
        raise ValueError(f"Unknown test eligibility policy: {policy}")

    blocked = {go.get_namespace(term) for term in t0_terms if go.has_term(term)}
    eligible = {
        term for term in t1_terms
        if go.has_term(term) and go.get_namespace(term) not in blocked
    }
    return eligible, blocked


def _build_test_dataframe(
    go: Ontology,
    t0_catalog: ProteinCatalog,
    t1_catalog: ProteinCatalog,
    matches: list[IdentityMatch],
    t1_to_t0: dict[str, str],
    t0_annots: dict[str, set[str]],
    t1_annots: dict[str, set[str]],
    protein_binding_policy: str,
    test_eligibility_policy: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    translated_t1 = {
        t1_to_t0[t1_id]: set(terms)
        for t1_id, terms in t1_annots.items()
        if t1_id in t1_to_t0
    }
    rows = []
    flow: list[dict[str, object]] = []

    for match in sorted(matches, key=lambda item: (item.t0_id, item.t1_id or "")):
        t0_record = t0_catalog.records.get(match.t0_id)
        t1_record = t1_catalog.records.get(match.t1_id or "")
        entry = {
            "t0_id": match.t0_id,
            "t1_id": match.t1_id or "",
            "taxon_id": (t0_record or t1_record).taxon_id if (t0_record or t1_record) else "",
            "sequence_changed": int(match.sequence_changed),
            "t0_annotation_count": len(t0_annots.get(match.t0_id, set())),
            "t1_annotation_count": len(translated_t1.get(match.t0_id, set())),
            "blocked_ontologies": "",
            "eligible_ontologies": "",
            "status": match.status,
            "reason": match.reason,
        }
        if match.status != "matched":
            flow.append(entry)
            continue
        t0_terms = t0_annots.get(match.t0_id, set())
        if test_eligibility_policy == "global-no-knowledge" and t0_terms:
            entry.update(status="excluded", reason="qualifying_annotation_at_t0")
            flow.append(entry)
            continue

        direct_terms = translated_t1.get(match.t0_id, set())
        if not direct_terms:
            entry.update(status="excluded", reason="no_new_qualifying_annotation_at_t1")
            flow.append(entry)
            continue
        direct_terms, blocked = _test_eligible_terms(
            t0_terms, direct_terms, go, test_eligibility_policy
        )
        entry["blocked_ontologies"] = ",".join(sorted(blocked))
        if not direct_terms:
            entry.update(
                status="excluded",
                reason="qualifying_annotation_at_t0_in_gained_ontology",
            )
            flow.append(entry)
            continue
        direct_terms = _drop_protein_binding_only(direct_terms, go, protein_binding_policy)
        if not direct_terms:
            entry.update(status="excluded", reason="protein_binding_only")
            flow.append(entry)
            continue
        entry["eligible_ontologies"] = ",".join(sorted({
            go.get_namespace(term) for term in direct_terms if go.has_term(term)
        }))
        propagated = propagate_annotations(go, direct_terms)
        if not propagated:
            entry.update(status="excluded", reason="empty_annotation_after_propagation")
            flow.append(entry)
            continue

        rows.append({
            "proteins": match.t0_id,
            "sequences": t0_record.sequence,
            "annotations": tuple(sorted(propagated)),
        })
        entry.update(status="selected", reason="new_annotation_after_t0")
        flow.append(entry)

    frame = pd.DataFrame(rows, columns=["proteins", "sequences", "annotations"])
    if not frame.empty:
        frame = frame.sort_values("proteins", kind="stable").reset_index(drop=True)
    return frame, flow


def _build_released_test_dataframe(
    go: Ontology,
    fasta_paths: tuple[Path, ...],
    annotations_path: Path,
    target_catalog: ProteinCatalog,
) -> tuple[pd.DataFrame, list[dict[str, object]], AnnotationLoadResult]:
    """Recreate DeepGOPlus test_data.pkl from released CAFA ground truth."""
    direct_annotations = load_deepgoplus_annotation_file(annotations_path)
    frames = []
    for fasta_path in fasta_paths:
        frame, _ = build_deepgoplus_dataframe_from_fasta(
            go=go,
            sequences_file=fasta_path,
            annots=direct_annotations,
            count_terms=False,
        )
        frames.append(frame)
    test_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["proteins", "sequences", "annotations"]
    )
    selected_ids = set(test_df["proteins"])
    flow = []
    for protein_id in test_df["proteins"]:
        record = target_catalog.records[protein_id]
        flow.append({
            "t0_id": protein_id,
            "t1_id": protein_id,
            "taxon_id": record.taxon_id or "",
            "sequence_changed": 0,
            "t0_annotation_count": 0,
            "t1_annotation_count": len(direct_annotations[protein_id]),
            "blocked_ontologies": "",
            "eligible_ontologies": "released-groundtruth",
            "status": "selected",
            "reason": "released_official_groundtruth",
        })

    kept_rows = sum(len(direct_annotations[protein_id]) for protein_id in selected_ids)
    skipped_rows = sum(
        len(terms) for protein_id, terms in direct_annotations.items()
        if protein_id not in selected_ids
    )
    result = AnnotationLoadResult(
        annotations={protein_id: set(direct_annotations[protein_id]) for protein_id in selected_ids},
        counters=Counter({
            "processed": kept_rows + skipped_rows,
            "kept_rows": kept_rows,
            "proteins": len(selected_ids),
            "skipped_outside_sequences": skipped_rows,
            "released_groundtruth": kept_rows,
        }),
    )
    return test_df, flow, result


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_csv_outputs(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Summarise the in-memory CSV tables produced by the calculation."""
    stats: dict[str, dict[str, int]] = {}
    split_data: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    for prefix in sorted(PREFIX_TO_NAMESPACE):
        for split in ("training", "validation", "test"):
            key = f"{prefix}-{split}"
            frame = frames[key]
            proteins = set(frame["proteins"].astype(str))
            sequences = set(frame["sequences"].astype(str))
            stats[key] = {"rows": len(frame), "go_terms": len(frame.columns) - 2}
            split_data[(prefix, split)] = (proteins, sequences)

    global_data: dict[str, tuple[set[str], set[str]]] = {}
    for split in ("training", "validation", "test"):
        ids: set[str] = set()
        sequences: set[str] = set()
        for prefix in PREFIX_TO_NAMESPACE:
            prefix_ids, prefix_sequences = split_data[(prefix, split)]
            ids.update(prefix_ids)
            sequences.update(prefix_sequences)
        global_data[split] = (ids, sequences)

    # Report split overlap in the produced CSV populations.
    overlap_diagnostics: dict[str, int] = {}
    for left, right in (
        ("training", "validation"),
        ("training", "test"),
        ("validation", "test"),
    ):
        key = f"{left}_{right}"
        overlap_diagnostics[f"global_{key}_protein_ids"] = len(
            global_data[left][0] & global_data[right][0]
        )
        overlap_diagnostics[f"global_{key}_exact_sequences"] = len(
            global_data[left][1] & global_data[right][1]
        )
    return stats, overlap_diagnostics


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _write_reports(
    config: BuildConfig,
    go: Ontology,
    written: dict[str, Path],
    flow: list[dict[str, object]],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    terms_df: pd.DataFrame,
    training_catalog: ProteinCatalog,
    target_catalog: ProteinCatalog,
    t0_result: AnnotationLoadResult,
    t1_result: AnnotationLoadResult,
    csv_stats: dict[str, dict[str, int]],
    cross_ontology_overlap_diagnostics: dict[str, int],
    training_annotation_result: AnnotationLoadResult | None = None,
    official_target_rows: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    report_dir = config.reports
    report_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Path] = {}

    if official_target_rows:
        target_path = report_dir / "official_target_mapping.tsv"
        _write_tsv(target_path, OFFICIAL_TARGET_FIELDS, official_target_rows)
        reports["official_target_mapping"] = target_path

    flow_path = report_dir / "protein_flow.tsv"
    _write_tsv(flow_path, [
        "t0_id", "t1_id", "taxon_id", "sequence_changed", "t0_annotation_count",
        "t1_annotation_count", "blocked_ontologies", "eligible_ontologies", "status", "reason",
    ], flow)
    reports["protein_flow_report"] = flow_path

    exclusions = Counter(row["reason"] for row in flow if row["status"] != "selected")
    exclusion_path = report_dir / "exclusion_reasons.tsv"
    _write_tsv(exclusion_path, ["reason", "count"], [
        {"reason": reason, "count": exclusions[reason]} for reason in sorted(exclusions)
    ])
    reports["exclusion_report"] = exclusion_path

    evidence_rows = []
    for timepoint, result in (("t0", t0_result), ("t1", t1_result)):
        for evidence in sorted(result.evidence_counts):
            evidence_rows.append({
                "timepoint": timepoint,
                "evidence_code": evidence,
                "kept_rows": result.evidence_counts[evidence],
            })
    evidence_path = report_dir / "evidence_summary.tsv"
    _write_tsv(evidence_path, ["timepoint", "evidence_code", "kept_rows"], evidence_rows)
    reports["evidence_report"] = evidence_path

    date_filter_path = report_dir / "t1_date_filter_by_protein.tsv"
    date_filter_rows = []
    for (protein_id, aspect), counts in sorted(t1_result.date_filter_counts.items()):
        date_filter_rows.append({
            "protein_id": protein_id,
            "aspect": aspect,
            "passed_date_filter": counts["passed_date_filter"],
            "skipped_backfill": counts["skipped_backfill"],
            "skipped_after_cutoff": counts["skipped_after_cutoff"],
        })
    _write_tsv(date_filter_path, [
        "protein_id", "aspect", "passed_date_filter", "skipped_backfill",
        "skipped_after_cutoff",
    ], date_filter_rows)
    reports["t1_date_filter_report"] = date_filter_path

    taxon_rows = []
    train_ids = set(train_df["proteins"].tolist())
    test_ids = set(test_df["proteins"].tolist())
    for stage, protein_ids, catalog in (
        ("training_before_split", train_ids, training_catalog),
        ("test_before_ontology_export", test_ids, target_catalog),
    ):
        counts = Counter(catalog.records[protein_id].taxon_id or "unknown" for protein_id in protein_ids)
        for taxon_id in sorted(counts):
            taxon_rows.append({"stage": stage, "taxon_id": taxon_id, "proteins": counts[taxon_id]})
    taxon_path = report_dir / "taxon_summary.tsv"
    _write_tsv(taxon_path, ["stage", "taxon_id", "proteins"], taxon_rows)
    reports["taxon_report"] = taxon_path

    selected_flow = [row for row in flow if row["status"] == "selected"]
    selected_flow_by_id = {str(row["t0_id"]): row for row in selected_flow}
    cohort_rows = []
    cohort_counts: Counter[tuple[str, str]] = Counter()
    for prefix, namespace in sorted(PREFIX_TO_NAMESPACE.items()):
        final_test_ids = set(pd.read_csv(written[f"{prefix}-test"])["proteins"].astype(str))
        for protein_id in sorted(final_test_ids):
            row = selected_flow_by_id[protein_id]
            eligible = set(filter(None, str(row["eligible_ontologies"]).split(",")))
            if namespace not in eligible:
                blocked = set()
                cohort = "not-classified"
            else:
                blocked = set(filter(None, str(row["blocked_ontologies"]).split(",")))
                cohort = "limited-knowledge" if blocked else "no-knowledge"
            cohort_counts[(prefix, cohort)] += 1
            cohort_rows.append({
                "ontology": prefix,
                "protein": protein_id,
                "cohort": cohort,
                "t0_known_ontologies": ",".join(sorted(blocked)),
            })

    cohort_path = report_dir / "test_knowledge_cohort_membership.tsv"
    _write_tsv(cohort_path, [
        "ontology", "protein", "cohort", "t0_known_ontologies",
    ], cohort_rows)
    reports["test_knowledge_cohort_membership"] = cohort_path

    cohort_summary_rows = []
    for prefix in sorted(PREFIX_TO_NAMESPACE):
        for cohort in ("no-knowledge", "limited-knowledge", "not-classified"):
            cohort_summary_rows.append({
                "ontology": prefix,
                "cohort": cohort,
                "proteins": cohort_counts[(prefix, cohort)],
            })
    cohort_summary_path = report_dir / "test_knowledge_cohort_summary.tsv"
    _write_tsv(
        cohort_summary_path,
        ["ontology", "cohort", "proteins"],
        cohort_summary_rows,
    )
    reports["test_knowledge_cohort_summary"] = cohort_summary_path

    gain_rows = []
    for prefix, namespace in sorted(PREFIX_TO_NAMESPACE.items()):
        namespace_terms = {
            term for term in terms_df["terms"].tolist()
            if go.has_term(term) and go.get_namespace(term) == namespace
        }
        proteins = sum(
            1 for annotations in test_df["annotations"].tolist()
            if set(annotations) & namespace_terms
        )
        gain_rows.append({
            "ontology": prefix,
            "selected_test_proteins": proteins,
            "training_defined_terms": len(namespace_terms),
        })
    gain_path = report_dir / "annotation_gain_summary.tsv"
    _write_tsv(gain_path, ["ontology", "selected_test_proteins", "training_defined_terms"], gain_rows)
    reports["annotation_gain_report"] = gain_path

    statistics = {
        "profile": config.profile_name,
        "benchmark_configuration": {
            "target_universe_policy": config.target_universe_policy,
            "evidence_codes": sorted(config.evidence_codes),
            "training_taxa": sorted(config.training_taxa),
            "target_taxa": sorted(config.target_taxa),
            "t0_cutoff": config.t0_cutoff,
            "t1_cutoff": config.t1_cutoff,
            "t1_endpoint_policy": config.t1_endpoint_policy,
            "test_eligibility_policy": config.test_eligibility_policy,
            "exclude_t1_backfill": config.exclude_t1_backfill,
            "require_t0_presence": config.require_t0_presence,
            "sequence_change_policy": config.sequence_change_policy,
            "protein_binding_policy": config.protein_binding_policy,
            "min_count": config.min_count,
            "split": config.split,
            "seed": config.seed,
            "training_reviewed_only": config.reviewed_only,
            "target_reviewed_only": config.target_reviewed_only,
            "allow_frozen_source_fallback": config.allow_frozen_source_fallback,
            "include_relationships": config.include_rels,
            "included_relationship_types": (
                sorted(config.relationship_types)
                if config.relationship_types is not None else "all"
            ),
        },
        "training_proteins_before_split": len(train_df),
        "selected_test_proteins_before_ontology_export": len(test_df),
        "selected_target_flow_rows": len(selected_flow),
        "test_knowledge_cohorts": {
            prefix: {
                cohort: cohort_counts[(prefix, cohort)]
                for cohort in (
                    "no-knowledge", "limited-knowledge", "not-classified",
                )
            }
            for prefix in sorted(PREFIX_TO_NAMESPACE)
        },
        "training_defined_terms": len(terms_df),
        "csv_outputs": csv_stats,
        "cross_ontology_split_overlap_counts": (
            cross_ontology_overlap_diagnostics
        ),
        "t0_goa": _counter_dict(t0_result.counters),
        "t1_goa": _counter_dict(t1_result.counters),
        "excluded_target_proteins": _counter_dict(exclusions),
        "t0_unmapped_terms": _counter_dict(t0_result.unmapped_terms),
        "t1_unmapped_terms": _counter_dict(t1_result.unmapped_terms),
        "t0_terms_outside_frozen_ontology": _counter_dict(t0_result.out_of_benchmark_terms),
        "t1_terms_outside_frozen_t0_ontology": _counter_dict(t1_result.out_of_benchmark_terms),
        "training_annotation_source": (
            _counter_dict(training_annotation_result.counters)
            if training_annotation_result else "t0_goa"
        ),
        "test_annotation_source": (
            "released_official_groundtruth" if config.test_annotations_file else "temporal_goa"
        ),
        "official_target_mapping": _counter_dict(Counter(
            row["status"] for row in (official_target_rows or [])
        )),
    }
    statistics_path = report_dir / "benchmark_statistics.json"
    statistics_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n")
    reports["statistics"] = statistics_path
    return reports


def build_snapshot_benchmark(config: BuildConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    LOGGER.info("Loading frozen benchmark ontology from %s", config.go_obo)
    benchmark_go = Ontology(
        config.go_obo,
        with_rels=config.include_rels,
        relationship_types=config.relationship_types,
    )
    t0_go = benchmark_go if config.ontology_t0 == config.go_obo else Ontology(
        config.ontology_t0,
        with_rels=config.include_rels,
        relationship_types=config.relationship_types,
    )
    t1_go = benchmark_go if config.ontology_t1 == config.go_obo else Ontology(
        config.ontology_t1,
        with_rels=config.include_rels,
        relationship_types=config.relationship_types,
    )
    LOGGER.info("Loading t0 training UniProt catalogue")
    training_paths = tuple(
        path for path in config.uniprot_t0
        if not (config.reviewed_only and "trembl" in path.name.lower())
    )
    training_catalog = load_protein_catalog(
        training_paths, config.training_taxa, config.reviewed_only
    )
    LOGGER.info("Loaded %d canonical t0 training proteins", len(training_catalog.records))
    official_target_rows: list[dict[str, object]] = []
    if config.target_universe_policy == "official-cafa3-targets":
        released_groundtruth = config.test_annotations_file is not None
        t0_reference_paths = config.target_uniprot_t0 or config.uniprot_t0
        t1_reference_paths = config.target_uniprot_t1 or config.uniprot_t1
        t0_reference = (
            ProteinCatalog() if released_groundtruth
            else load_protein_catalog(t0_reference_paths, frozenset(), False)
        )
        t1_reference = (
            ProteinCatalog() if released_groundtruth
            else load_protein_catalog(t1_reference_paths, frozenset(), False)
        )
        t0_official = load_official_target_catalog(
            config.official_target_fastas,
            config.official_target_mapping_dir,
            t0_reference,
            config.target_taxa,
            "t0",
        )
        t1_official = load_official_target_catalog(
            config.official_target_fastas,
            config.official_target_mapping_dir,
            t1_reference,
            config.target_taxa,
            "t1",
        )
        t0_target_catalog = t0_official.catalog
        t1_target_catalog = t1_official.catalog
        if not released_groundtruth:
            official_target_rows = t0_official.rows + t1_official.rows
    else:
        reconstructed_t0_paths = config.target_uniprot_t0 or config.uniprot_t0
        reconstructed_t1_paths = config.target_uniprot_t1 or config.uniprot_t1
        t0_target_catalog = _target_catalog(
            reconstructed_t0_paths,
            training_catalog,
            config.training_taxa,
            config.target_taxa,
            config.reviewed_only,
            config.target_reviewed_only,
        )
        t1_target_catalog = load_protein_catalog(
            reconstructed_t1_paths, config.target_taxa, config.target_reviewed_only
        )
    LOGGER.info("Loaded %d canonical t0 target proteins", len(t0_target_catalog.records))
    LOGGER.info("Loaded %d canonical t1 target proteins", len(t1_target_catalog.records))

    annotation_catalog = (
        t0_target_catalog
        if config.training_annotations_file
        else _merge_catalogs(training_catalog, t0_target_catalog)
    )
    if config.goa_t0:
        LOGGER.info("Loading and normalising t0 GOA annotations from %s", config.goa_t0)
        t0_result = load_normalized_annotation_map(
            config.goa_t0,
            alias_to_primary=annotation_catalog.alias_to_primary,
            source_ontology=t0_go,
            benchmark_ontology=benchmark_go,
            allow_frozen_source_fallback=config.allow_frozen_source_fallback,
            evidence_codes=config.evidence_codes,
        )
    else:
        LOGGER.info("Bypassing t0 GOA because released training and test annotations were supplied")
        t0_result = AnnotationLoadResult(annotations={})

    released_test_df: pd.DataFrame | None = None
    released_test_flow: list[dict[str, object]] | None = None
    if config.test_annotations_file:
        LOGGER.info("Loading released CAFA3 test ground truth from %s", config.test_annotations_file)
        released_test_df, released_test_flow, t1_result = _build_released_test_dataframe(
            benchmark_go,
            config.official_target_fastas,
            config.test_annotations_file,
            t0_target_catalog,
        )
    else:
        LOGGER.info("Loading and normalising t1 GOA annotations from %s", config.goa_t1)
        t1_result = load_normalized_annotation_map(
            config.goa_t1,
            alias_to_primary=t1_target_catalog.alias_to_primary,
            source_ontology=t1_go,
            benchmark_ontology=benchmark_go,
            allow_frozen_source_fallback=config.allow_frozen_source_fallback,
            evidence_codes=config.evidence_codes,
            target_taxa=config.target_taxa,
            exclude_on_or_before=config.t0_cutoff if config.exclude_t1_backfill else None,
            include_on_or_before=(
                config.t1_cutoff
                if config.t1_endpoint_policy == "assigned-date-proxy"
                else None
            ),
        )
    training_ids = set(training_catalog.records)
    target_t0_ids = set(t0_target_catalog.records)
    training_annotation_result: AnnotationLoadResult | None = None
    if config.training_annotations_file:
        training_annotation_result = load_released_training_annotations(
            config.training_annotations_file,
            training_catalog.alias_to_primary,
            benchmark_go,
        )
        train_annots = training_annotation_result.annotations
    else:
        train_annots = {
            protein_id: terms for protein_id, terms in t0_result.annotations.items()
            if protein_id in training_ids
        }
    target_t0_annots = {
        protein_id: terms for protein_id, terms in t0_result.annotations.items()
        if protein_id in target_t0_ids
    }

    LOGGER.info("Building deterministic DeepGOPlus-style training dataframe")
    train_all_df, counts = build_training_dataframe(
        benchmark_go, training_catalog.sequences, train_annots
    )
    terms_df = make_terms_dataframe(counts, config.min_count)
    train_df, valid_df = split_train_valid(
        train_all_df, split=config.split, seed=config.seed
    )
    LOGGER.info(
        "Training proteins=%d terms=%d split=%d/%d",
        len(train_all_df), len(terms_df), len(train_df), len(valid_df),
    )

    if released_test_df is not None and released_test_flow is not None:
        test_df, flow = released_test_df, released_test_flow
    else:
        matches, t1_to_t0 = _build_identity_crosswalk(
            t0_target_catalog, t1_target_catalog, config.sequence_change_policy
        )
        test_df, flow = _build_test_dataframe(
            benchmark_go,
            t0_target_catalog,
            t1_target_catalog,
            matches,
            t1_to_t0,
            target_t0_annots,
            t1_result.annotations,
            config.protein_binding_policy,
            config.test_eligibility_policy,
        )

    LOGGER.info("Selected %d temporal test proteins", len(test_df))

    written: dict[str, Path] = {}
    if config.write_intermediates:
        intermediates = {
            "train_data": config.output_dir / "train_data.pkl",
            "train_data_train": config.output_dir / "train_data_train.pkl",
            "train_data_valid": config.output_dir / "train_data_valid.pkl",
            "test_data": config.output_dir / "test_data.pkl",
            "terms": config.output_dir / "terms.pkl",
        }
        train_all_df.to_pickle(intermediates["train_data"])
        train_df.to_pickle(intermediates["train_data_train"])
        valid_df.to_pickle(intermediates["train_data_valid"])
        test_df.to_pickle(intermediates["test_data"])
        terms_df.to_pickle(intermediates["terms"])
        written.update(intermediates)

    csv_frames = prepare_pfp_csv_frames(
        benchmark_go, train_df, valid_df, test_df, terms_df
    )
    written.update(write_pfp_csv_frames(csv_frames, config.output_dir))
    csv_stats, cross_ontology_overlap_diagnostics = _summarize_csv_outputs(
        csv_frames
    )
    reports = _write_reports(
        config,
        benchmark_go,
        written,
        flow,
        train_all_df,
        test_df,
        terms_df,
        training_catalog,
        t0_target_catalog,
        t0_result,
        t1_result,
        csv_stats,
        cross_ontology_overlap_diagnostics,
        training_annotation_result,
        official_target_rows,
    )
    written.update(reports)
    return written
