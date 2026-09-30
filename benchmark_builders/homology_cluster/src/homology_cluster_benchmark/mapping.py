"""UniProt sequence loading and accession canonicalisation."""

from __future__ import annotations

from dataclasses import replace
import re
from pathlib import Path
from typing import Iterator

from .inputs import open_text
from .models import AnnotationRecord, GoaLoadResult, ProteinCatalog, ProteinRecord
from .uniref import fasta_identifier, iter_fasta


OX_RE = re.compile(r"\bOX=(\d+)\b")
DAT_TAXON_RE = re.compile(r"NCBI_TaxID=(\d+)")


def _fasta_accession(header: str) -> str:
    first = fasta_identifier(header)
    fields = first.split("|")
    if len(fields) >= 3 and fields[0] in {"sp", "tr"}:
        return fields[1]
    return first


def _iter_uniprot_fasta(path: Path) -> Iterator[ProteinRecord]:
    for header, sequence in iter_fasta(path):
        protein_id = _fasta_accession(header)
        taxon = OX_RE.search(header)
        yield ProteinRecord(
            protein_id=protein_id,
            sequence=sequence.upper(),
            taxon_id=taxon.group(1) if taxon else "",
            aliases=(protein_id,),
        )


def _iter_uniprot_dat(path: Path) -> Iterator[ProteinRecord]:
    accessions: list[str] = []
    taxon = ""
    in_sequence = False
    sequence_parts: list[str] = []
    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            tag = line[:2]
            if tag == "AC":
                accessions.extend(
                    value.strip() for value in line[5:].split(";") if value.strip()
                )
            elif tag == "OX":
                match = DAT_TAXON_RE.search(line)
                if match:
                    taxon = match.group(1)
            elif tag == "SQ":
                in_sequence = True
            elif line == "//":
                yield ProteinRecord(
                    protein_id=accessions[0],
                    sequence="".join(sequence_parts).upper(),
                    taxon_id=taxon,
                    aliases=tuple(accessions),
                )
                accessions = []
                taxon = ""
                in_sequence = False
                sequence_parts = []
            elif in_sequence:
                sequence_parts.append(
                    "".join(character for character in line if character.isalpha())
                )


def iter_uniprot(path: Path) -> Iterator[ProteinRecord]:
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".dat") or suffixes.endswith(".dat.gz"):
        yield from _iter_uniprot_dat(path)
    else:
        yield from _iter_uniprot_fasta(path)


def load_requested_proteins(
    sources: dict[str, Path], requested_accessions: set[str]
) -> ProteinCatalog:
    """Load selected records and leave ambiguous secondary aliases unmapped."""

    catalog = ProteinCatalog()
    source_order = [name for name in ("sprot", "trembl", "fixture") if name in sources]
    source_order.extend(sorted(set(sources) - set(source_order)))
    for source_population in source_order:
        for record in iter_uniprot(sources[source_population]):
            aliases = set(record.aliases) | {record.protein_id}
            matches = aliases & requested_accessions
            if not matches:
                continue
            catalog.records.setdefault(record.protein_id, record)
            for accession in sorted(matches | {record.protein_id}):
                if accession in catalog.ambiguous_aliases:
                    continue
                previous = catalog.alias_to_primary.get(accession)
                if previous is not None and previous != record.protein_id:
                    catalog.ambiguous_aliases.add(accession)
                    catalog.alias_to_primary.pop(accession, None)
                    continue
                catalog.alias_to_primary[accession] = record.protein_id
    return catalog


def canonicalize_goa_accessions(
    result: GoaLoadResult, catalog: ProteinCatalog
) -> GoaLoadResult:
    records: list[AnnotationRecord] = []
    annotations: dict[str, set[str]] = {}
    canonical_accessions: set[str] = set()
    accession_map: dict[str, tuple[str, str]] = {}
    raw_accessions = result.qualifying_accessions or set(result.annotations)
    for raw in sorted(raw_accessions):
        if raw in catalog.ambiguous_aliases:
            protein_id = raw
            action = "ambiguous-secondary"
        else:
            protein_id = catalog.alias_to_primary.get(raw, raw)
            action = "exact" if protein_id == raw else "secondary-to-primary"
        accession_map[raw] = (protein_id, action)
        canonical_accessions.add(protein_id)
        if result.record_spool is None:
            annotations.setdefault(protein_id, set()).update(result.annotations[raw])

    if result.record_spool is None:
        for record in result.records:
            protein_id, action = accession_map[record.raw_accession]
            records.append(
                replace(record, protein_id=protein_id, accession_action=action)
            )
        result.records = records
    else:
        result.records = []
    result.annotations = annotations
    result.qualifying_accessions = canonical_accessions
    result.accession_map = accession_map
    return result
