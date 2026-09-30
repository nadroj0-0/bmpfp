"""Write descriptive benchmark tables and a readable report."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .config import RunConfig
from .models import AnalysisBundle


TABLE_FIELDS = {
    "file_profiles": (
        "dataset_id",
        "aspect",
        "split",
        "file",
        "proteins",
        "terms",
        "positive_labels",
        "root_positive_proteins",
        "root_only_proteins",
        "all_zero_proteins",
        "terms_with_support",
    ),
    "label_profiles": (
        "dataset_id",
        "aspect",
        "split",
        "proteins",
        "terms",
        "root_positive_proteins",
        "root_only_proteins",
        "root_only_fraction",
        "all_zero_proteins",
        "all_zero_fraction",
        "proteins_with_non_root_labels",
        "positive_labels",
        "labels_count",
        "labels_min",
        "labels_p25",
        "labels_median",
        "labels_p75",
        "labels_p90",
        "labels_p95",
        "labels_p99",
        "labels_max",
        "labels_mean",
        "non_root_labels_count",
        "non_root_labels_min",
        "non_root_labels_p25",
        "non_root_labels_median",
        "non_root_labels_p75",
        "non_root_labels_p90",
        "non_root_labels_p95",
        "non_root_labels_p99",
        "non_root_labels_max",
        "non_root_labels_mean",
        "sequence_length_count",
        "sequence_length_min",
        "sequence_length_p25",
        "sequence_length_median",
        "sequence_length_p75",
        "sequence_length_p90",
        "sequence_length_p95",
        "sequence_length_p99",
        "sequence_length_max",
        "sequence_length_mean",
    ),
    "term_support": (
        "dataset_id",
        "aspect",
        "split",
        "term",
        "is_root",
        "support",
        "fraction",
    ),
    "root_only_source": (
        "dataset_id",
        "aspect",
        "split",
        "protein_id",
        "classification",
        "reason",
        "source_aspect_term_count",
        "source_non_root_term_count",
        "source_non_root_terms",
        "unresolved_source_term_count",
        "unresolved_source_terms",
    ),
    "root_only_summary": (
        "dataset_id",
        "aspect",
        "split",
        "classification",
        "proteins",
        "root_only_proteins",
        "total_proteins",
        "fraction_of_root_only",
        "fraction_of_total",
        "projection_policy",
    ),
    "taxonomy_distribution": (
        "dataset_id",
        "aspect",
        "split",
        "rank",
        "taxon_id",
        "taxon_name",
        "proteins",
        "scope_proteins",
        "mapped_proteins",
        "fraction_of_total",
        "fraction_of_mapped",
    ),
    "taxonomy_coverage": (
        "dataset_id",
        "aspect",
        "split",
        "proteins",
        "mapped_proteins",
        "unmapped_proteins",
        "mapping_fraction",
    ),
    "taxonomy_conflicts": (
        "dataset_id",
        "protein_id",
        "status",
        "selected_taxon_id",
        "selected_taxon_name",
        "selected_source_name",
        "selected_source_path",
        "selected_source_priority",
        "selected_accession_role",
        "selected_record_primary_accession",
        "selected_sequence_matches_benchmark",
        "selected_resolution_basis",
        "alternative_taxon_id",
        "alternative_taxon_name",
        "alternative_source_name",
        "alternative_source_path",
        "alternative_source_priority",
        "alternative_accession_role",
        "alternative_record_primary_accession",
        "alternative_sequence_matches_benchmark",
        "resolution",
    ),
    "category_distribution": (
        "dataset_id",
        "category_source",
        "aspect",
        "split",
        "rank",
        "category_id",
        "category_name",
        "proteins",
        "scope_proteins",
        "mapped_proteins",
        "fraction_of_total",
        "note",
    ),
    "modality_coverage": (
        "dataset_id",
        "aspect",
        "split",
        "modality",
        "coverage_state",
        "covered_proteins",
        "total_proteins",
        "coverage_fraction",
    ),
    "modality_patterns": (
        "dataset_id",
        "aspect",
        "split",
        "coverage_state",
        "modality_pattern",
        "proteins",
        "total_proteins",
        "fraction",
    ),
    "protein_membership": (
        "dataset_id",
        "protein_id",
        "aspects",
        "splits",
        "memberships",
        "taxonomy_mapped",
        "taxon_id",
        "taxon_name",
        "taxonomy_source_name",
        "taxonomy_source_path",
        "taxonomy_source_priority",
        "taxonomy_accession_role",
        "taxonomy_record_primary_accession",
        "taxonomy_sequence_matches_benchmark",
        "taxonomy_resolution_basis",
        "taxonomy_conflict_resolved",
        "taxonomy_conflict_unresolved",
    ),
    "cross_benchmark_metrics": (
        "reference_dataset",
        "comparison_dataset",
        "aspect",
        "split",
        "metric",
        "reference_value",
        "comparison_value",
        "absolute_delta",
        "relative_delta",
    ),
    "cross_benchmark_modality": (
        "reference_dataset",
        "comparison_dataset",
        "aspect",
        "split",
        "modality",
        "coverage_state",
        "reference_fraction",
        "comparison_fraction",
        "absolute_delta",
    ),
    "cross_benchmark_taxonomy": (
        "reference_dataset",
        "comparison_dataset",
        "aspect",
        "split",
        "taxon_id",
        "taxon_name",
        "reference_fraction",
        "comparison_fraction",
        "absolute_delta",
    ),
    "cross_benchmark_root_source": (
        "reference_dataset",
        "comparison_dataset",
        "aspect",
        "split",
        "classification",
        "reference_fraction_of_total",
        "comparison_fraction_of_total",
        "absolute_delta",
    ),
}


def _write_tsv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def _markdown(bundle: AnalysisBundle, config: RunConfig) -> str:
    lines = [
        f"# Benchmark analysis: {config.run_name}",
        "",
        "This report is descriptive. It identifies benchmark composition and coverage "
        "differences; it does not by itself prove why model performance changed.",
        "",
        "## Dataset overview",
        "",
        "| Dataset | Unique proteins | CSV memberships | Taxonomy mapped | Taxonomy conflicts | Modality availability |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset in config.datasets:
        diagnostics = bundle.summary["datasets"][dataset.id]
        lines.append(
            f"| {dataset.id} | {diagnostics['unique_proteins']:,} | "
            f"{diagnostics['observation_rows']:,} | "
            f"{diagnostics['taxonomy_mapped_unique_proteins']:,} | "
            f"{diagnostics['taxonomy_conflict_proteins']:,} | "
            f"{'yes' if diagnostics['modality_availability_configured'] else 'no'} |"
        )

    label_rows = bundle.tables["label_profiles"]
    lines.extend(
        [
            "",
            "## Label and root-only profile",
            "",
            "| Dataset | Aspect | Split | Proteins | Terms | Root-only | Mean non-root labels |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in label_rows:
        if row["split"] not in {"test", "all"}:
            continue
        lines.append(
            f"| {row['dataset_id']} | {row['aspect']} | {row['split']} | "
            f"{row['proteins']:,} | {row['terms']:,} | "
            f"{_percent(row['root_only_fraction'])} | "
            f"{row['non_root_labels_mean']:.2f} |"
        )

    root_rows = bundle.tables["root_only_summary"]
    lines.extend(
        [
            "",
            "## Root-only source",
            "",
            "`projection_created` means the source had an informative term in the aspect "
            "but the final CSV retained only the root. It is attributed to the named "
            "projection policy, not guessed when source annotations are unavailable.",
            "",
            "| Dataset | Aspect | Split | Source-root-only | Projection-created | Unresolved |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    grouped = {}
    for row in root_rows:
        if row["split"] not in {"test", "all"}:
            continue
        grouped.setdefault((row["dataset_id"], row["aspect"], row["split"]), {})[
            row["classification"]
        ] = row
    for key, values in grouped.items():
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | "
            f"{_percent(values['source_root_only']['fraction_of_total'])} | "
            f"{_percent(values['projection_created']['fraction_of_total'])} | "
            f"{_percent(values['source_unresolved']['fraction_of_total'])} |"
        )

    modality_rows = bundle.tables["modality_coverage"]
    if modality_rows:
        preferred_state = modality_rows[0]["coverage_state"]
        lines.extend(
            [
                "",
                f"## Modality availability ({preferred_state})",
                "",
                "| Dataset | Aspect | Split | Modality | Coverage |",
                "|---|---|---|---|---:|",
            ]
        )
        for row in modality_rows:
            if row["coverage_state"] != preferred_state or row["split"] not in {
                "test",
                "all",
            }:
                continue
            if row["aspect"] == "all":
                continue
            lines.append(
                f"| {row['dataset_id']} | {row['aspect']} | {row['split']} | "
                f"{row['modality']} | {_percent(row['coverage_fraction'])} |"
            )

    conflict_rows = bundle.tables["taxonomy_conflicts"]
    if conflict_rows:
        lines.extend(
            [
                "",
                "## Taxonomy source conflicts",
                "",
                "UniProt alias conflicts use exact benchmark-sequence evidence; "
                "cross-source disagreements use configured priority. Every resolved "
                "or unresolved case is retained in `taxonomy_conflicts.tsv`. Ambiguous "
                "proteins remain taxonomically unmapped without aborting the analysis.",
                "",
                "| Dataset | Proteins with conflicts | Unresolved proteins | Conflict observations |",
                "|---|---:|---:|---:|",
            ]
        )
        for dataset in config.datasets:
            diagnostics = bundle.summary["datasets"][dataset.id]
            lines.append(
                f"| {dataset.id} | "
                f"{diagnostics['taxonomy_conflict_proteins']:,} | "
                f"{diagnostics['taxonomy_unresolved_conflict_proteins']:,} | "
                f"{diagnostics['taxonomy_conflict_observations']:,} |"
            )

    taxonomy_rows = bundle.tables["taxonomy_distribution"]
    lines.extend(
        [
            "",
            f"## Leading taxa (top {config.top_n})",
            "",
            "These are organism/taxonomy distributions, not protein-family distributions.",
        ]
    )
    for dataset in config.datasets:
        for aspect in ("BPO", "CCO", "MFO"):
            selected = [
                row
                for row in taxonomy_rows
                if row["dataset_id"] == dataset.id
                and row["aspect"] == aspect
                and row["split"] == "all"
                and row["rank"] <= config.top_n
            ]
            lines.extend(
                [
                    "",
                    f"### {dataset.id}: {aspect}",
                    "",
                    "| Rank | Taxon | Name | Proteins | Share |",
                    "|---:|---|---|---:|---:|",
                ]
            )
            for row in selected:
                lines.append(
                    f"| {row['rank']} | {row['taxon_id']} | {row['taxon_name']} | "
                    f"{row['proteins']:,} | {_percent(row['fraction_of_total'])} |"
                )

    lines.extend(["", "## Interpretation boundaries", ""])
    lines.extend(f"- {item}" for item in bundle.summary["interpretation_boundaries"])
    lines.extend(
        [
            "",
            "## Complete machine-readable tables",
            "",
            "The full split-level distributions, all taxa, every modality availability "
            "state, root-only row sources, term support, and cross-benchmark deltas are "
            "stored in the TSV files beside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_payload(output: Path, bundle: AnalysisBundle, config: RunConfig) -> None:
    for table_name, fields in TABLE_FIELDS.items():
        _write_tsv(
            output / f"{table_name}.tsv", fields, bundle.tables.get(table_name, ())
        )
    (output / "benchmark_analysis.md").write_text(
        _markdown(bundle, config), encoding="utf-8"
    )


def write_reports(
    output_dir: Path,
    bundle: AnalysisBundle,
    config: RunConfig,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_payload(output_dir, bundle, config)
