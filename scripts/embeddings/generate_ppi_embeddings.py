#!/usr/bin/env python3
"""Generate STRING PPI embeddings for proteins in prepared names files."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np


STRING_ALIAS_SOURCES = frozenset(
    {
        "UniProt_AC",
        "UniProt_ID",
        "UniProt_DR_FlyBase",
        "Ensembl_gene",
        "Ensembl_flybase_gene_id",
    }
)


def is_cafa3_id(protein_id: str) -> bool:
    return protein_id.startswith("T") and len(protein_id) > 10


def load_upstream_extractor(path: Path):
    specification = importlib.util.spec_from_file_location("pfp_ppi_extractor", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def requested_ids(paths: list[Path]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for value in np.load(path, allow_pickle=True).tolist():
            protein_id = str(value)
            if protein_id not in seen:
                seen.add(protein_id)
                ordered.append(protein_id)
    return ordered


def load_string_aliases(path: Path) -> dict[str, str]:
    """Read the upstream three-column STRING alias table, plain or gzipped."""
    opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
    open_args = (path, "rt") if opener is gzip.open else (path, "r")
    aliases: dict[str, str] = {}
    with opener(*open_args, encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            string_id, alias, source, *_ = line.rstrip("\r\n").split("\t")
            if source in STRING_ALIAS_SOURCES and alias not in aliases:
                aliases[alias] = string_id
    return aliases


def load_cafa3_aliases(path: Path, extractor) -> dict[str, str]:
    """Read JSON or a headerless CAFA3-target-to-alias TSV."""
    if path.suffix.lower() == ".json":
        return extractor.load_cafa3_id_mapping(path)
    with path.open("r", encoding="utf-8") as handle:
        return {
            fields[0]: fields[1]
            for line in handle
            if (fields := line.rstrip("\r\n").split("\t")) and fields[0]
        }


def map_to_string(
    protein_ids: list[str],
    cafa_to_uniprot: dict,
    cafa_to_id: dict,
    alias_to_string: dict,
    uniprot_to_entry: dict,
    entry_to_uniprots: dict,
    is_cafa3_id,
) -> dict[str, set[str]]:
    needed: dict[str, set[str]] = {}

    def record(protein_id: str, string_id: str | None) -> None:
        if string_id:
            needed.setdefault(string_id, set()).add(protein_id)

    def map_uniprot(accession: str) -> str | None:
        string_id = alias_to_string.get(accession)
        if string_id is None:
            string_id = alias_to_string.get(accession.split("-", 1)[0])
        if string_id is None and accession in uniprot_to_entry:
            entry = uniprot_to_entry[accession]
            string_id = alias_to_string.get(entry)
            if string_id is None:
                for alternative in entry_to_uniprots.get(entry, []):
                    if alternative in alias_to_string:
                        string_id = alias_to_string[alternative]
                        break
        return string_id

    for protein_id in protein_ids:
        if not is_cafa3_id(protein_id):
            record(protein_id, map_uniprot(protein_id))
            continue
        string_id = None
        if protein_id in cafa_to_uniprot:
            string_id = map_uniprot(cafa_to_uniprot[protein_id])
        if string_id is None and protein_id in cafa_to_id:
            string_id = alias_to_string.get(cafa_to_id[protein_id])
        record(protein_id, string_id)
    return needed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--names-file", type=Path, action="append", required=True)
    parser.add_argument("--string-h5", type=Path, required=True)
    parser.add_argument("--string-alias", type=Path, required=True)
    parser.add_argument("--cafa-assessment-dir", type=Path)
    parser.add_argument("--cafa3-id-mapping", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    extractor = load_upstream_extractor(args.extractor)
    proteins = requested_ids(args.names_file)
    has_cafa_ids = any(is_cafa3_id(value) for value in proteins)

    cafa_to_uniprot: dict = {}
    uniprot_to_entry: dict = {}
    entry_to_uniprots: dict = {}
    if args.cafa_assessment_dir is not None:
        if has_cafa_ids:
            cafa_to_uniprot = extractor.build_cafa_to_uniprot_mapping(
                args.cafa_assessment_dir
            )
        uniprot_to_entry, entry_to_uniprots = (
            extractor.build_uniprot_alternative_mapping(args.cafa_assessment_dir)
        )
    cafa_to_id = (
        load_cafa3_aliases(args.cafa3_id_mapping, extractor)
        if has_cafa_ids and args.cafa3_id_mapping is not None
        else {}
    )
    alias_to_string = load_string_aliases(args.string_alias)
    needed = map_to_string(
        proteins,
        cafa_to_uniprot,
        cafa_to_id,
        alias_to_string,
        uniprot_to_entry,
        entry_to_uniprots,
        is_cafa3_id,
    )
    extractor.extract_and_save_ppi(needed, args.string_h5, args.output_dir)
    print(f"Generated STRING PPI embeddings in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
