from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from .io_utils import open_text
from .models import ProteinCatalog, ProteinRecord

OX_RE = re.compile(r"\bOX=(\d+)\b")
DAT_TAXON_RE = re.compile(r"NCBI_TaxID=(\d+)")


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Stream FASTA headers without ``>`` and their sequences."""
    header = None
    parts: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line[1:]
                parts = []
            else:
                parts.append(line)
        if header is not None:
            yield header, "".join(parts)


def _protein_id_from_fasta_header(header: str) -> tuple[str, str | None, bool | None]:
    first = header.split()[0]
    fields = first.split("|")
    if len(fields) >= 3 and fields[0] in {"sp", "tr"}:
        return fields[1], fields[2], fields[0] == "sp"
    return first, None, None


def iter_uniprot_fasta(path: str | Path) -> Iterator[ProteinRecord]:
    for header, sequence in iter_fasta(path):
        protein_id, entry_name, reviewed = _protein_id_from_fasta_header(header)
        taxon_match = OX_RE.search(header)
        taxon_id = taxon_match.group(1) if taxon_match else None
        yield ProteinRecord(
            protein_id=protein_id,
            sequence=sequence,
            taxon_id=taxon_id,
            reviewed=reviewed,
            entry_name=entry_name,
            accessions=(protein_id,),
        )


def iter_uniprot_dat(path: str | Path) -> Iterator[ProteinRecord]:
    """Stream accession, sequence, taxon and review state from UniProt DAT records."""
    entry_name = None
    reviewed = None
    accessions: list[str] = []
    taxon_id = None
    in_sequence = False
    sequence_parts: list[str] = []

    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            tag = line[:2]

            if tag == "ID":
                fields = line.split()
                entry_name = fields[1] if len(fields) > 1 else None
                reviewed = "Reviewed;" in line
            elif tag == "AC":
                ac_text = line[5:].strip()
                accessions.extend([x.strip() for x in ac_text.split(";") if x.strip()])
            elif tag == "OX":
                match = DAT_TAXON_RE.search(line)
                if match:
                    taxon_id = match.group(1)
            elif tag == "SQ":
                in_sequence = True
            elif line == "//":
                if accessions and sequence_parts:
                    yield ProteinRecord(
                        protein_id=accessions[0],
                        sequence="".join(sequence_parts),
                        taxon_id=taxon_id,
                        reviewed=reviewed,
                        entry_name=entry_name,
                        accessions=tuple(accessions),
                    )
                entry_name = None
                reviewed = None
                accessions = []
                taxon_id = None
                in_sequence = False
                sequence_parts = []
            elif in_sequence:
                sequence_parts.append("".join(ch for ch in line if ch.isalpha()))


def iter_uniprot(path: str | Path) -> Iterator[ProteinRecord]:
    path = Path(path)
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".dat.gz") or suffixes.endswith(".dat"):
        yield from iter_uniprot_dat(path)
    else:
        yield from iter_uniprot_fasta(path)


def load_protein_catalog(
    paths: tuple[Path, ...],
    target_taxa: frozenset[str] = frozenset(),
    reviewed_only: bool = False,
) -> ProteinCatalog:
    """Load primary UniProt records and unambiguous accession aliases.

    The first DAT accession is primary; later accessions are aliases only.
    """
    catalog = ProteinCatalog()
    for path in paths:
        for record in iter_uniprot(path):
            if target_taxa and record.taxon_id not in target_taxa:
                continue
            if reviewed_only and record.reviewed is not True:
                continue

            primary = record.protein_id
            catalog.records.setdefault(primary, record)

            aliases = set(record.accessions) | {primary}
            for alias in aliases:
                if alias in catalog.ambiguous_aliases:
                    continue
                previous = catalog.alias_to_primary.get(alias)
                if previous is None or previous == primary:
                    catalog.alias_to_primary[alias] = primary
                else:
                    catalog.ambiguous_aliases.add(alias)
                    del catalog.alias_to_primary[alias]
    return catalog
