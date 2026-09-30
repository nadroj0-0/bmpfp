from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass, replace
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Iterable, Iterator

from .clique import CliqueStats, build_clique_index
from .clustering import connect_proteins_to_clusters, retained_cluster_info
from .config import BENCHMARK_EVIDENCE_CODES, SPLITS, parse_identity
from .export import export_all
from .goa import iter_annotation_records, load_goa
from .idmapping import load_uniref_mappings
from .labels import build_labels
from .mapping import canonicalize_goa_accessions, load_requested_proteins
from .mmseqs import ClusterIndex
from .models import MappingDecision, SplitAssignment
from .ontology import Ontology
from .splitting import assign_development_test, assign_training_validation
from .uniref import UniRefIndex
from .uniref_scaffold import uniref_scaffold


CSV_FILES = tuple(
    f"{prefix}-{split}.csv"
    for prefix in ("bp", "cc", "mf")
    for split in SPLITS
)
SIDE_TABLE_FILES = (
    "protein_cluster_assignments.tsv",
    "cluster_split_assignments.tsv",
    "retained_cluster_members.tsv.gz",
    "qualifying_annotations.tsv.gz",
)


@dataclass(frozen=True)
class BuildConfig:
    """Scientific inputs and settings for one homology benchmark build."""

    go_obo: Path
    goa: Path
    uniref_fasta: Path | None
    idmapping: Path | None
    output_dir: Path
    uniprot_sprot: Path | None = None
    uniprot_trembl: Path | None = None
    uniref_level: int | None = None
    identity: float = 0.30
    coverage: float = 0.80
    seed: int = 0
    min_count: int = 50
    split_policy: str = "cluster-count-random"
    development_fraction: float = 0.80
    training_fraction_within_development: float = 0.90
    include_relationships: bool = True
    clique_memberships: Path | None = None
    mmseqs_bin: str | None = None
    threads: int | None = None
    sensitivity: float | None = None
    cov_mode: int | None = None
    cluster_mode: int | None = None
    alignment_mode: int | None = None
    createdb_shuffle: int | None = None
    cluster_reassign: int | None = None
    evalue: float | None = None

    @property
    def uniprot_sources(self) -> dict[str, Path]:
        sources = {}
        if self.uniprot_sprot is not None:
            sources["sprot"] = self.uniprot_sprot
        if self.uniprot_trembl is not None:
            sources["trembl"] = self.uniprot_trembl
        return sources

    @property
    def source_scope(self) -> str:
        selected = set(self.uniprot_sources)
        if selected == {"sprot"}:
            return "sprot-only"
        if selected == {"trembl"}:
            return "trembl-only"
        if selected == {"sprot", "trembl"}:
            return "sprot-and-trembl"
        return "none"

    @property
    def effective_sensitivity(self) -> float:
        return (
            uniref_scaffold(self.uniref_level).recommended_sensitivity
            if self.sensitivity is None else self.sensitivity
        )


ClusterRunner = Callable[[BuildConfig, Path], Path]


def _run_mmseqs(config: BuildConfig, work_dir: Path) -> Path:
    """Run the three MMseqs stages needed to produce cluster assignments."""

    work_dir.mkdir(parents=True, exist_ok=True)
    requested_executable = config.mmseqs_bin or "mmseqs"
    candidate = (
        Path(requested_executable).expanduser()
        if "/" in requested_executable
        else None
    )
    executable = (
        str(candidate)
        if candidate is not None
        else (shutil.which(requested_executable) or requested_executable)
    )

    database = work_dir / f"uniref{config.uniref_level}_db"
    cluster_database = work_dir / "clusters_db"
    temporary = work_dir / "mmseqs_tmp"
    assignments = work_dir / f"uniref{config.uniref_level}_clusters.tsv"
    createdb = [
        executable, "createdb", str(config.uniref_fasta), str(database),
        "--dbtype", "1", "--threads", str(config.threads),
    ]
    if config.createdb_shuffle is not None:
        createdb.extend(("--shuffle", str(config.createdb_shuffle)))
    cluster = [
        executable, "cluster", str(database), str(cluster_database), str(temporary),
        "--min-seq-id", f"{parse_identity(config.identity):.2f}",
        "-c", repr(config.coverage),
        "--cov-mode", str(config.cov_mode),
        "--cluster-mode", str(config.cluster_mode),
        "--alignment-mode", str(config.alignment_mode),
        "--seq-id-mode", "0",
        "-s", f"{config.effective_sensitivity:g}",
    ]
    if config.cluster_reassign:
        cluster.extend(("--cluster-reassign", str(config.cluster_reassign)))
    if config.evalue is not None:
        cluster.extend(("-e", f"{config.evalue:g}"))
    cluster.extend(("--threads", str(config.threads)))
    createtsv = [
        executable, "createtsv", str(database), str(database),
        str(cluster_database), str(assignments),
    ]
    for command in (createdb, cluster, createtsv):
        subprocess.run(command, check=True)
    return assignments


@contextmanager
def _output_text(path: Path) -> Iterator[io.TextIOBase]:
    if path.suffix != ".gz":
        with path.open("w", encoding="utf-8", newline="") as handle:
            yield handle
        return
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                yield handle


def _write_tsv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]],
) -> None:
    with _output_text(path) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _decision_indexes(
    decisions: list[MappingDecision],
) -> tuple[dict[str, MappingDecision], dict[str, MappingDecision]]:
    by_raw = {decision.raw_accession: decision for decision in decisions}
    by_protein: dict[str, MappingDecision] = {}
    for decision in decisions:
        current = by_protein.get(decision.protein_id)
        if current is None or (not current.mmseqs_cluster_id and decision.mmseqs_cluster_id):
            by_protein[decision.protein_id] = decision
    return by_raw, by_protein


def _clique_candidates(
    requested_accessions: set[str],
    catalog,
) -> list[MappingDecision]:
    """Describe explicit UniProt members without requiring a UniRef lookup."""

    decisions = []
    for raw_accession in sorted(requested_accessions):
        if raw_accession in catalog.ambiguous_aliases:
            decisions.append(MappingDecision(
                raw_accession=raw_accession,
                protein_id=raw_accession,
                accession_action="ambiguous-secondary",
                status="ambiguous-secondary",
                detail="secondary accession maps to more than one selected-UniProt primary",
                canonical_sequence_available=False,
            ))
            continue
        protein_id = catalog.alias_to_primary.get(raw_accession, raw_accession)
        accession_action = (
            "exact" if protein_id == raw_accession else "secondary-to-primary"
        )
        if protein_id not in catalog.records:
            decisions.append(MappingDecision(
                raw_accession=raw_accession,
                protein_id=protein_id,
                accession_action=accession_action,
                status="not-in-selected-uniprot",
                detail="GOA accession is absent from the explicitly selected UniProt source",
                canonical_sequence_available=False,
            ))
            continue
        decisions.append(MappingDecision(
            raw_accession=raw_accession,
            protein_id=protein_id,
            accession_action=accession_action,
            status="clique-candidate",
            detail="",
            canonical_sequence_available=True,
        ))
    return decisions


def _connect_clique_candidates(
    decisions: list[MappingDecision], cluster_index: ClusterIndex,
) -> list[MappingDecision]:
    """Connect UniProt candidates to the supplied clique assignments."""

    member_ids = {
        decision.protein_id
        for decision in decisions
        if decision.canonical_sequence_available and decision.protein_id
    }
    cluster_by_member = cluster_index.clusters_for(member_ids)
    connected = []
    for decision in decisions:
        cluster_id = (
            cluster_by_member.get(decision.protein_id, "")
            if decision.canonical_sequence_available
            else ""
        )
        if cluster_id:
            connected.append(replace(
                decision,
                mmseqs_cluster_id=cluster_id,
                status="mapped",
                detail="resolved directly from supplied clique membership",
            ))
        elif decision.canonical_sequence_available:
            connected.append(replace(
                decision,
                status="missing-mmseqs-assignment",
                detail=(
                    "annotated UniProt accession has no supplied clique assignment"
                ),
            ))
        else:
            connected.append(decision)
    return connected


def _write_side_tables(
    output_dir: Path,
    config: BuildConfig,
    ontology: Ontology,
    goa,
    uniref: UniRefIndex | None,
    cluster_index: ClusterIndex,
    retained: dict,
    assignments: dict[str, SplitAssignment],
    decisions: list[MappingDecision],
) -> tuple[int, int]:
    del ontology  # The ontology has already resolved every emitted annotation record.
    clique_build = config.clique_memberships is not None
    scaffold = None if clique_build else uniref_scaffold(config.uniref_level)
    exists_field = None if scaffold is None else f"exists_in_{scaffold.slug}_fasta"
    mapping_fields = [
        "raw_uniprot_accession", "uniprot_accession", "accession_action",
        "mapping_status", "mapping_detail",
        "canonical_sequence_available",
        "mmseqs_cluster_id", "split",
    ]
    if scaffold is not None and exists_field is not None:
        mapping_fields.insert(3, scaffold.id_field)
        mapping_fields.insert(6, exists_field)
    _write_tsv(
        output_dir / "protein_cluster_assignments.tsv",
        mapping_fields,
        (
            {
                "raw_uniprot_accession": item.raw_accession,
                "uniprot_accession": item.protein_id,
                "accession_action": item.accession_action,
                "mapping_status": item.status,
                "mapping_detail": item.detail,
                "canonical_sequence_available": int(item.canonical_sequence_available),
                "mmseqs_cluster_id": item.mmseqs_cluster_id,
                "split": item.split,
                **(
                    {
                        scaffold.id_field: item.uniref90_id,
                        exists_field: (
                            "" if item.exists_in_fasta is None
                            else int(item.exists_in_fasta)
                        ),
                    }
                    if scaffold is not None and exists_field is not None
                    else {}
                ),
            }
            for item in sorted(decisions, key=lambda row: (row.protein_id, row.raw_accession))
        ),
    )

    member_field = (
        "cluster_member_id" if clique_build else scaffold.id_field
    )
    member_count_field = (
        "cluster_member_count" if clique_build else f"{scaffold.slug}_member_count"
    )
    _write_tsv(
        output_dir / "cluster_split_assignments.tsv",
        [
            "mmseqs_cluster_id", "split", member_count_field,
            "qualifying_uniprot_count", "assignment_stage",
        ],
        (
            {
                "mmseqs_cluster_id": item.cluster_id,
                "split": item.split,
                member_count_field: item.member_count,
                "qualifying_uniprot_count": item.labelled_protein_count,
                "assignment_stage": item.stage,
            }
            for item in (assignments[key] for key in sorted(assignments))
        ),
    )

    retained_ids = set(retained)
    annotated_members = {
        item.protein_id if clique_build else item.uniref90_id
        for item in decisions
        if item.mmseqs_cluster_id in retained_ids
        and (item.protein_id if clique_build else item.uniref90_id)
    }
    if clique_build:
        retained_rows = (
            (cluster, member, "") for cluster, member in
            cluster_index.iter_assignments_for_clusters(retained_ids)
        )
    else:
        retained_rows = cluster_index.iter_assignments_with_metadata_for_clusters(
            uniref, retained_ids
        )
    retained_member_count = 0
    retained_unannotated_count = 0
    with _output_text(output_dir / "retained_cluster_members.tsv.gz") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mmseqs_cluster_id", "split", member_field, "sequence_length",
                "connected_to_qualifying_uniprot",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for cluster_id, member_id, sequence_length in retained_rows:
            retained_member_count += 1
            connected = member_id in annotated_members
            retained_unannotated_count += int(not connected)
            writer.writerow({
                "mmseqs_cluster_id": cluster_id,
                "split": assignments[cluster_id].split,
                member_field: member_id,
                "sequence_length": sequence_length,
                "connected_to_qualifying_uniprot": int(connected),
            })

    by_raw, by_protein = _decision_indexes(decisions)
    annotation_fields = [
        "raw_uniprot_accession", "uniprot_accession",
        "raw_go_id", "canonical_go_id", "namespace", "aspect", "evidence_code",
        "qualifier", "taxon_id", "accession_action",
        "mmseqs_cluster_id", "split",
    ]
    if scaffold is not None:
        annotation_fields.insert(-2, scaffold.id_field)
    _write_tsv(
        output_dir / "qualifying_annotations.tsv.gz",
        annotation_fields,
        (
            {
                "raw_uniprot_accession": record.raw_accession,
                "uniprot_accession": record.protein_id,
                "raw_go_id": record.raw_go_id,
                "canonical_go_id": record.go_id,
                "namespace": record.namespace,
                "aspect": record.aspect,
                "evidence_code": record.evidence,
                "qualifier": record.qualifier,
                "taxon_id": record.taxon_id,
                "accession_action": record.accession_action,
                "mmseqs_cluster_id": (
                    (by_raw.get(record.raw_accession) or by_protein.get(record.protein_id)).mmseqs_cluster_id
                    if (by_raw.get(record.raw_accession) or by_protein.get(record.protein_id))
                    else ""
                ),
                "split": (
                    (by_raw.get(record.raw_accession) or by_protein.get(record.protein_id)).split
                    if (by_raw.get(record.raw_accession) or by_protein.get(record.protein_id))
                    else ""
                ),
                **(
                    {
                        scaffold.id_field: (
                            (
                                by_raw.get(record.raw_accession)
                                or by_protein.get(record.protein_id)
                            ).uniref90_id
                            if (
                                by_raw.get(record.raw_accession)
                                or by_protein.get(record.protein_id)
                            )
                            else ""
                        )
                    }
                    if scaffold is not None
                    else {}
                ),
            }
            for record in iter_annotation_records(goa)
        ),
    )
    return retained_member_count, retained_unannotated_count


def _csv_statistics(
    ontology: Ontology,
    labels,
) -> dict[str, dict[str, int]]:
    """Summarise the exported benchmark CSVs."""

    statistics: dict[str, dict[str, int]] = {}
    for prefix, namespace in (
        ("bp", "biological_process"),
        ("cc", "cellular_component"),
        ("mf", "molecular_function"),
    ):
        terms = {
            term
            for term in labels.term_universe
            if ontology.namespace(term) == namespace
        }
        for split in SPLITS:
            row_count = sum(
                bool(set(protein_annotations) & terms)
                for protein_annotations in labels.frames[split].annotations
            )
            statistics[f"{prefix}-{split}.csv"] = {
                "rows": row_count,
                "terms": len(terms),
            }
    return statistics


def _scientific_summary(
    config: BuildConfig,
    goa,
    labels,
    assignments: dict[str, SplitAssignment],
    retained: dict,
    retained_members: int,
    retained_unannotated: int,
    csv_stats: dict[str, dict[str, int]],
    clique_stats: CliqueStats | None,
) -> dict[str, object]:
    split_counts = {}
    for split in SPLITS:
        selected = [item for item in assignments.values() if item.split == split]
        split_counts[split] = {
            "clusters": len(selected),
            "cluster_members": sum(item.member_count for item in selected),
            "qualifying_proteins": len(labels.frames[split]),
        }
    summary = {
        "cluster_source": "clique-memberships" if clique_stats else "mmseqs",
        "identity_fraction": parse_identity(config.identity),
        "coverage": config.coverage,
        "seed": config.seed,
        "min_count": config.min_count,
        "split_policy": config.split_policy,
        "development_fraction": config.development_fraction,
        "training_fraction_within_development": config.training_fraction_within_development,
        "evidence_codes": sorted(BENCHMARK_EVIDENCE_CODES),
        "include_go_relationships": config.include_relationships,
        "source_scope": config.source_scope,
        "counts": {
            "qualifying_goa_accessions": len(goa.qualifying_accessions),
            "retained_clusters": len(retained),
            "retained_cluster_members": retained_members,
            "retained_unannotated_cluster_members": retained_unannotated,
            "development_defined_terms": len(labels.term_universe),
        },
        "annotation_row_attrition": {
            "source_rows": labels.intended_annotation_rows,
            "terminal_counts": dict(sorted(labels.row_attrition_counts.items())),
        },
        "protein_attrition": {
            "source_proteins": labels.intended_accessions,
            "terminal_counts": dict(sorted(labels.protein_attrition_counts.items())),
        },
        "split_counts": split_counts,
        "csv_outputs": csv_stats,
        "clique_membership_counts": clique_stats.as_dict() if clique_stats else None,
    }
    if clique_stats is None:
        summary["uniref_level"] = config.uniref_level
        summary["coverage_argument"] = repr(config.coverage)
        summary["mmseqs_settings"] = {
            "sensitivity": config.effective_sensitivity,
            "cov_mode": config.cov_mode,
            "cluster_mode": config.cluster_mode,
            "alignment_mode": config.alignment_mode,
            "createdb_shuffle": config.createdb_shuffle,
            "cluster_reassign": config.cluster_reassign,
            "evalue": config.evalue,
        }
    return summary


def build_benchmark(
    config: BuildConfig,
    *,
    cluster_runner: ClusterRunner | None = None,
) -> Path:
    """Build one homology benchmark from explicit scientific inputs."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{config.output_dir.name}.build-",
        dir=config.output_dir.parent,
    ) as temporary_root:
        root = Path(temporary_root)
        scratch = root / "scratch"
        scratch.mkdir()

        ontology = Ontology(config.go_obo, config.include_relationships)
        goa = load_goa(
            config.goa,
            ontology,
            BENCHMARK_EVIDENCE_CODES,
            spool_dir=scratch / "goa",
        )
        requested_raw = set(goa.qualifying_accessions or goa.annotations)
        catalog = load_requested_proteins(
            config.uniprot_sources,
            requested_raw,
        )
        goa = canonicalize_goa_accessions(goa, catalog)
        clique_stats = None
        if config.clique_memberships is not None:
            uniref = None
            decisions = _clique_candidates(requested_raw, catalog)
            cluster_index, clique_stats = build_clique_index(
                config.clique_memberships,
                scratch / "clusters.sqlite",
            )
            clique_build = True
        else:
            uniref = UniRefIndex.build(
                config.uniref_fasta,
                scratch / f"uniref{config.uniref_level}.sqlite",
                config.uniref_level,
            )
            decisions = load_uniref_mappings(
                config.idmapping,
                requested_raw,
                catalog,
                uniref,
                config.uniref_level,
            )
            assignments_path = (
                cluster_runner(config, scratch / "mmseqs")
                if cluster_runner is not None else _run_mmseqs(config, scratch / "mmseqs")
            )
            cluster_index = ClusterIndex.build(
                assignments_path,
                uniref,
                scratch / "clusters.sqlite",
                member_field=uniref_scaffold(config.uniref_level).id_field,
            )
            clique_build = False

        if clique_build:
            decisions = _connect_clique_candidates(decisions, cluster_index)
        else:
            decisions = connect_proteins_to_clusters(
                decisions,
                cluster_index,
                config.uniref_level,
            )
        retained = retained_cluster_info(decisions, cluster_index)
        development_test = assign_development_test(
            retained,
            config.split_policy,
            config.seed,
            config.development_fraction,
        )
        decisions = [
            replace(item, split=development_test[item.mmseqs_cluster_id].split)
            if item.mmseqs_cluster_id in development_test else item
            for item in decisions
        ]
        labels = build_labels(
            ontology,
            goa,
            catalog,
            decisions,
            development_test,
            config.min_count,
            finalize_development=lambda: assign_training_validation(
                retained,
                development_test,
                config.split_policy,
                config.seed,
                config.training_fraction_within_development,
            ),
        )
        assignments = labels.cluster_assignments
        decisions = [
            replace(item, split=assignments[item.mmseqs_cluster_id].split)
            if item.mmseqs_cluster_id in assignments else item
            for item in decisions
        ]

        export_all(labels, ontology, config.output_dir)
        retained_members, retained_unannotated = _write_side_tables(
            config.output_dir,
            config,
            ontology,
            goa,
            uniref,
            cluster_index,
            retained,
            assignments,
            decisions,
        )
        csv_stats = _csv_statistics(ontology, labels)
        summary = _scientific_summary(
            config,
            goa,
            labels,
            assignments,
            retained,
            retained_members,
            retained_unannotated,
            csv_stats,
            clique_stats,
        )
        (config.output_dir / "benchmark_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return config.output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a PFP benchmark using MMseqs2 clusters or supplied clique memberships."
    )
    parser.add_argument("--go-obo", type=Path, required=True)
    parser.add_argument("--goa", type=Path, required=True)
    parser.add_argument(
        "--uniref-fasta",
        type=Path,
        help="Required for an MMseqs build; not used with --clique-memberships.",
    )
    parser.add_argument("--uniref-level", type=int, choices=(50, 90))
    parser.add_argument(
        "--idmapping",
        type=Path,
        help="Required for an MMseqs build; not used with --clique-memberships.",
    )
    parser.add_argument("--uniprot-sprot", type=Path)
    parser.add_argument("--uniprot-trembl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clique-memberships", type=Path)
    parser.add_argument("--identity", type=float, default=30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument(
        "--split-policy",
        choices=("cluster-count-random", "sequence-balanced"),
        default="cluster-count-random",
    )
    parser.add_argument("--development-fraction", type=float, default=0.80)
    parser.add_argument("--training-fraction", type=float, default=0.90)
    parser.add_argument("--mmseqs-bin")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--sensitivity", type=float)
    parser.add_argument("--cov-mode", type=int)
    parser.add_argument("--cluster-mode", type=int)
    parser.add_argument("--alignment-mode", type=int)
    parser.add_argument("--createdb-shuffle", type=int)
    parser.add_argument("--cluster-reassign", type=int)
    parser.add_argument("--evalue", type=float)
    parser.add_argument("--no-go-relationships", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    clique_build = args.clique_memberships is not None
    config = BuildConfig(
        go_obo=args.go_obo,
        goa=args.goa,
        uniref_fasta=args.uniref_fasta,
        idmapping=args.idmapping,
        output_dir=args.output_dir,
        uniprot_sprot=args.uniprot_sprot,
        uniprot_trembl=args.uniprot_trembl,
        uniref_level=(
            args.uniref_level if clique_build
            else (50 if args.uniref_level is None else args.uniref_level)
        ),
        identity=parse_identity(args.identity),
        coverage=args.coverage,
        seed=args.seed,
        min_count=args.min_count,
        split_policy=args.split_policy,
        development_fraction=args.development_fraction,
        training_fraction_within_development=args.training_fraction,
        include_relationships=not args.no_go_relationships,
        clique_memberships=args.clique_memberships,
        mmseqs_bin=(
            args.mmseqs_bin if clique_build else (args.mmseqs_bin or "mmseqs")
        ),
        threads=(
            args.threads
            if clique_build else (1 if args.threads is None else args.threads)
        ),
        sensitivity=args.sensitivity,
        cov_mode=(
            args.cov_mode
            if clique_build else (0 if args.cov_mode is None else args.cov_mode)
        ),
        cluster_mode=(
            args.cluster_mode
            if clique_build else (0 if args.cluster_mode is None else args.cluster_mode)
        ),
        alignment_mode=(
            args.alignment_mode
            if clique_build
            else (3 if args.alignment_mode is None else args.alignment_mode)
        ),
        createdb_shuffle=args.createdb_shuffle,
        cluster_reassign=(
            args.cluster_reassign
            if clique_build
            else (0 if args.cluster_reassign is None else args.cluster_reassign)
        ),
        evalue=args.evalue,
    )
    output = build_benchmark(config)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
