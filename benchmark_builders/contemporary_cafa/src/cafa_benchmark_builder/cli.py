from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

from .builder import (
    build_benchmark,
    export_from_deepgoplus_pickles,
    generate_deepgoplus_pickles_from_cafa_files,
)
from .config import (
    BENCHMARK_PROFILES,
    BuildConfig,
    EVIDENCE_POLICIES,
    normalise_gaf_date,
    normalise_taxa,
)


def read_taxa_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    values = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(line)
    return values


def read_packaged_cafa3_taxa() -> list[str]:
    path = resources.files("cafa_benchmark_builder").joinpath("resources/cafa3_target_taxa.txt")
    with path.open("r") as handle:
        return [
            line.strip() for line in handle
            if line.strip() and not line.startswith("#")
        ]


def require_args(args: argparse.Namespace, names: list[str]) -> None:
    missing = [name for name in names if getattr(args, name) in (None, [])]
    if missing:
        formatted = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise SystemExit(f"Missing required arguments for --source-mode {args.source_mode}: {formatted}")


def _taxa_for_policy(policy: str, values: list[str], taxa_file: Path | None) -> frozenset[str]:
    collected = []
    if policy == "cafa3-targets":
        collected.extend(read_packaged_cafa3_taxa())
    collected.extend(values)
    collected.extend(read_taxa_file(taxa_file))
    if policy == "custom" and not collected:
        raise SystemExit("A custom taxon policy requires explicit taxon IDs or a taxa file")
    return normalise_taxa(collected)


SOURCE_MODES = ("snapshots", "deepgoplus", "cafa3-files")


def selected_source_mode(argv: list[str]) -> str:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--source-mode", choices=SOURCE_MODES, default="snapshots")
    return selector.parse_known_args(argv)[0].source_mode


def build_parser(source_mode: str = "snapshots") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic CAFA-style temporal benchmark CSVs for PFP."
    )
    parser.add_argument(
        "--source-mode", choices=SOURCE_MODES, default="snapshots",
        help=("snapshots parses UniProt/GOA/GO; deepgoplus exports released pickles; "
              "cafa3-files recreates pickles and exports PFP CSVs from official CAFA files."),
    )
    parser.add_argument("--go-obo", type=Path, required=True,
                        help="Frozen benchmark ontology. For temporal builds this should be the t0 GO snapshot.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-rels",
        action="store_true",
        help="Restrict ontology propagation to is_a edges.",
    )

    if source_mode == "deepgoplus":
        parser.add_argument("--deepgoplus-dir", type=Path, required=True)
        parser.add_argument(
            "--no-intermediates",
            action="store_true",
            help="Do not write intermediate pickle files.",
        )
        return parser

    if source_mode == "cafa3-files":
        parser.add_argument("--train-sequences-file", type=Path, required=True)
        parser.add_argument("--train-annotations-file", type=Path, required=True)
        parser.add_argument("--test-sequences-file", type=Path, required=True)
        parser.add_argument("--test-annotations-file", type=Path, required=True)
        parser.add_argument("--min-count", type=int, default=50)
        parser.add_argument("--split", type=float, default=0.9)
        parser.add_argument("--seed", type=int, default=0)
        return parser

    parser.add_argument("--profile", choices=sorted(BENCHMARK_PROFILES),
                        default="global-nk")
    parser.add_argument("--test-annotations-file", type=Path)
    parser.add_argument("--uniprot-t0", action="append", type=Path, required=True,
                        help="t0 UniProt FASTA/DAT. Repeat to combine Swiss-Prot and TrEMBL inputs.")
    parser.add_argument("--uniprot-t1", action="append", type=Path,
                        help="t1 UniProt FASTA/DAT. Repeat to combine Swiss-Prot and TrEMBL inputs.")
    parser.add_argument("--target-uniprot-t0", action="append", type=Path,
                        help="Optional UniProt inputs used only to map a fixed official t0 target universe.")
    parser.add_argument("--target-uniprot-t1", action="append", type=Path,
                        help="Optional UniProt inputs used only to map a fixed official t1 target universe.")
    parser.add_argument(
        "--target-universe-policy",
        choices=("reconstructed-all-qualifying", "official-cafa3-targets"),
        default="reconstructed-all-qualifying",
    )
    parser.add_argument("--official-target-fasta", action="append", type=Path, default=[])
    parser.add_argument("--official-target-mapping-dir", type=Path)
    parser.add_argument("--training-annotations-file", type=Path,
                        help="Optional released direct-label file used instead of GOA for training labels.")
    parser.add_argument("--goa-t0", type=Path)
    parser.add_argument("--goa-t1", type=Path)
    parser.add_argument("--go-obo-t0", type=Path,
                        help="Ontology matching the t0 GAF. Defaults to --go-obo.")
    parser.add_argument("--go-obo-t1", type=Path,
                        help="Ontology matching the t1 GAF. Defaults to --go-obo.")

    policy_choices = ("all", "cafa3-targets", "custom")
    parser.add_argument("--training-taxon-policy", choices=policy_choices)
    parser.add_argument("--target-taxon-policy", choices=policy_choices)
    parser.add_argument("--training-taxon", action="append", default=[])
    parser.add_argument("--training-taxa-file", type=Path)
    parser.add_argument("--target-taxon", action="append", default=[])
    parser.add_argument("--target-taxa-file", type=Path)

    parser.add_argument("--evidence-policy", choices=sorted(EVIDENCE_POLICIES))
    parser.add_argument("--evidence-code", action="append", default=[])
    parser.add_argument("--t0-cutoff",
                        help="Backfill cutoff in YYYYMMDD or YYYY-MM-DD form.")
    parser.add_argument("--t1-cutoff",
                        help="Latest included t1 annotation date in YYYYMMDD or YYYY-MM-DD form.")
    parser.add_argument(
        "--t1-endpoint-policy",
        choices=("assigned-date-proxy", "snapshot-membership"),
        default="assigned-date-proxy",
        help=("assigned-date-proxy applies --t1-cutoff to GAF column 14; "
              "snapshot-membership uses every qualifying row present in the t1 file."),
    )
    backfill = parser.add_mutually_exclusive_group()
    backfill.add_argument("--exclude-t1-backfill", dest="exclude_t1_backfill",
                          action="store_true",
                          help="Keep t1 annotations assigned after the t0 cutoff.")
    backfill.add_argument("--allow-t1-backfill", dest="exclude_t1_backfill",
                          action="store_false",
                          help="Keep qualifying t1 annotations regardless of the t0 cutoff.")
    parser.set_defaults(exclude_t1_backfill=None)
    parser.add_argument("--sequence-change-policy", choices=("exclude", "use-t0", "error"))
    parser.add_argument("--protein-binding-policy",
                        choices=("keep", "drop-mf-protein-binding-only"))
    parser.add_argument(
        "--test-eligibility-policy",
        choices=("global-no-knowledge", "ontology-no-knowledge"),
        help=("global-no-knowledge excludes every protein with any qualifying t0 annotation; "
              "ontology-no-knowledge reproduces CAFA NK/LK eligibility per ontology."),
    )

    training_review = parser.add_mutually_exclusive_group()
    training_review.add_argument("--training-reviewed-only", dest="training_reviewed_only",
                                 action="store_true",
                                 help="Use reviewed training records only.")
    training_review.add_argument("--include-unreviewed-training", dest="training_reviewed_only",
                                 action="store_false",
                                 help="Use reviewed and unreviewed training records.")
    parser.set_defaults(training_reviewed_only=None)
    target_review = parser.add_mutually_exclusive_group()
    target_review.add_argument(
        "--target-reviewed-only",
        dest="target_reviewed_only",
        action="store_true",
        help="Use reviewed target records only.",
    )
    target_review.add_argument(
        "--include-unreviewed-targets",
        dest="target_reviewed_only",
        action="store_false",
        help="Use reviewed and unreviewed target records.",
    )
    parser.set_defaults(target_reviewed_only=None)

    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--relationship-type",
        action="append",
        default=[],
        help=("Relationship type to include in addition to is_a. Repeat as needed. "
              "The existing default includes every relationship type."),
    )
    parser.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Do not write intermediate pickle files.",
    )
    parser.add_argument(
        "--no-frozen-source-fallback",
        action="store_true",
        help=("Require GOA IDs to resolve in the source ontology before mapping "
              "them into the benchmark ontology."),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> BuildConfig:
    required = []
    if args.test_annotations_file is None:
        required.extend(["uniprot_t1", "goa_t0", "goa_t1"])
    elif args.training_annotations_file is None:
        required.append("goa_t0")
    require_args(args, required)
    profile = BENCHMARK_PROFILES[args.profile]
    training_policy = args.training_taxon_policy or profile.training_taxon_policy
    target_policy = args.target_taxon_policy or profile.target_taxon_policy

    evidence_policy = args.evidence_policy or profile.evidence_policy
    evidence = frozenset(args.evidence_code) if args.evidence_code else EVIDENCE_POLICIES[evidence_policy]
    training_reviewed = (
        profile.training_reviewed_only
        if args.training_reviewed_only is None else args.training_reviewed_only
    )
    target_reviewed = (
        profile.target_reviewed_only
        if args.target_reviewed_only is None else args.target_reviewed_only
    )
    exclude_backfill = (
        profile.exclude_t1_backfill
        if args.exclude_t1_backfill is None else args.exclude_t1_backfill
    )

    return BuildConfig(
        uniprot_t0=tuple(args.uniprot_t0),
        uniprot_t1=tuple(args.uniprot_t1 or ()),
        goa_t0=args.goa_t0,
        goa_t1=args.goa_t1,
        go_obo=args.go_obo,
        go_obo_t0=args.go_obo_t0,
        go_obo_t1=args.go_obo_t1,
        output_dir=args.output_dir,
        target_universe_policy=args.target_universe_policy,
        official_target_fastas=tuple(args.official_target_fasta),
        official_target_mapping_dir=args.official_target_mapping_dir,
        target_uniprot_t0=tuple(args.target_uniprot_t0 or ()),
        target_uniprot_t1=tuple(args.target_uniprot_t1 or ()),
        training_annotations_file=args.training_annotations_file,
        test_annotations_file=args.test_annotations_file,
        profile_name=args.profile,
        training_taxa=_taxa_for_policy(
            training_policy, args.training_taxon, args.training_taxa_file
        ),
        target_taxa=_taxa_for_policy(
            target_policy, args.target_taxon, args.target_taxa_file
        ),
        evidence_codes=evidence,
        t0_cutoff=normalise_gaf_date(args.t0_cutoff or profile.t0_cutoff),
        t1_cutoff=normalise_gaf_date(args.t1_cutoff or profile.t1_cutoff),
        t1_endpoint_policy=args.t1_endpoint_policy,
        test_eligibility_policy=(
            args.test_eligibility_policy or profile.test_eligibility_policy
        ),
        exclude_t1_backfill=exclude_backfill,
        require_t0_presence=profile.require_t0_presence,
        sequence_change_policy=args.sequence_change_policy or profile.sequence_change_policy,
        protein_binding_policy=args.protein_binding_policy or profile.protein_binding_policy,
        min_count=args.min_count,
        split=args.split,
        seed=args.seed,
        reviewed_only=training_reviewed,
        target_reviewed_only=target_reviewed,
        include_rels=not args.no_rels,
        relationship_types=(frozenset(args.relationship_type) or None),
        write_intermediates=not args.no_intermediates,
        allow_frozen_source_fallback=not args.no_frozen_source_fallback,
    )


def main(argv: list[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(selected_source_mode(raw_args))
    args = parser.parse_args(raw_args)
    if args.source_mode == "deepgoplus":
        written = export_from_deepgoplus_pickles(
            deepgoplus_dir=args.deepgoplus_dir,
            go_obo=args.go_obo,
            output_dir=args.output_dir,
            include_rels=not args.no_rels,
            write_intermediates=not args.no_intermediates,
        )
    elif args.source_mode == "cafa3-files":
        written = generate_deepgoplus_pickles_from_cafa_files(
            go_obo=args.go_obo,
            train_sequences_file=args.train_sequences_file,
            train_annotations_file=args.train_annotations_file,
            test_sequences_file=args.test_sequences_file,
            test_annotations_file=args.test_annotations_file,
            output_dir=args.output_dir,
            min_count=args.min_count,
            split=args.split,
            seed=args.seed,
            include_rels=not args.no_rels,
        )
    else:
        written = build_benchmark(config_from_args(args))
    print("Wrote:")
    for key in sorted(written):
        print(f"  {key}: {written[key]}")
