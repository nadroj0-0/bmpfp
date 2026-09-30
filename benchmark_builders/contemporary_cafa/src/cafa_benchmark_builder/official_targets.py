from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .models import ProteinCatalog, ProteinRecord
from .parsers import iter_fasta


@dataclass
class OfficialTargetLoadResult:
    catalog: ProteinCatalog
    rows: list[dict[str, object]]


def _mapping_sources(mapping_dir: Path | None) -> dict[str, set[str]]:
    identifiers: dict[str, set[str]] = defaultdict(set)
    if mapping_dir is None:
        return identifiers
    for path in sorted(mapping_dir.glob("*")):
        with path.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                identifiers[fields[0]].update(value for value in fields[1:] if value)
    return identifiers


def _add_alias(catalog: ProteinCatalog, alias: str, target_id: str) -> None:
    if not alias or alias in catalog.ambiguous_aliases:
        return
    previous = catalog.alias_to_primary.get(alias)
    if previous is None or previous == target_id:
        catalog.alias_to_primary[alias] = target_id
    else:
        catalog.alias_to_primary.pop(alias, None)
        catalog.ambiguous_aliases.add(alias)


def _taxon_from_target(target_id: str, target_taxa: frozenset[str]) -> str | None:
    body = target_id[1:] if target_id[:1] in {"T", "M"} else target_id
    matches = [taxon for taxon in target_taxa if body.startswith(taxon)]
    return max(matches, key=len) if matches else None


def load_official_target_catalog(
    fasta_paths: tuple[Path, ...],
    mapping_dir: Path | None,
    reference_catalog: ProteinCatalog,
    target_taxa: frozenset[str],
    snapshot: str,
) -> OfficialTargetLoadResult:
    """Load released CAFA targets and map them to UniProt by identifier and sequence."""
    source_ids = _mapping_sources(mapping_dir)
    entry_index: dict[str, set[str]] = defaultdict(set)
    sequence_index: dict[str, set[str]] = defaultdict(set)
    for protein_id, record in reference_catalog.records.items():
        if record.entry_name:
            entry_index[record.entry_name].add(protein_id)
        sequence_index[record.sequence].add(protein_id)

    catalog = ProteinCatalog()
    rows: list[dict[str, object]] = []
    for fasta_path in fasta_paths:
        for header, sequence in iter_fasta(fasta_path):
            fields = header.split()
            target_id = fields[0]
            identifiers = set(source_ids.get(target_id, set()))
            identifiers.update(fields[1:])

            source_candidates: set[str] = set()
            for identifier in identifiers:
                primary = reference_catalog.alias_to_primary.get(identifier)
                if primary:
                    source_candidates.add(primary)
                source_candidates.update(entry_index.get(identifier, set()))
            sequence_candidates = set(sequence_index.get(sequence, set()))
            intersection = source_candidates & sequence_candidates

            selected: str | None = None
            method = ""
            reason = ""
            if len(intersection) == 1:
                selected = next(iter(intersection))
                method = "source-and-exact-sequence"
            elif len(source_candidates) == 1:
                candidate = next(iter(source_candidates))
                if reference_catalog.records[candidate].sequence == sequence:
                    selected = candidate
                    method = "source-and-exact-sequence"
                else:
                    reason = "source_mapping_sequence_mismatch"
            elif len(sequence_candidates) == 1:
                selected = next(iter(sequence_candidates))
                method = "unique-exact-sequence"
            elif len(intersection) > 1:
                reason = "ambiguous_source_and_sequence_mapping"
            elif len(source_candidates) > 1:
                reason = "ambiguous_source_mapping"
            elif len(sequence_candidates) > 1:
                reason = "ambiguous_exact_sequence_mapping"
            else:
                reason = "no_uniprot_mapping"

            mapped = reference_catalog.records.get(selected) if selected else None
            taxon_id = mapped.taxon_id if mapped else _taxon_from_target(target_id, target_taxa)
            accessions = {target_id, *identifiers}
            if mapped:
                accessions.update(mapped.accessions)
                accessions.add(mapped.protein_id)
            record = ProteinRecord(
                protein_id=target_id,
                sequence=sequence,
                taxon_id=taxon_id,
                reviewed=mapped.reviewed if mapped else None,
                entry_name=mapped.entry_name if mapped else (fields[1] if len(fields) > 1 else None),
                accessions=tuple(sorted(accessions)),
            )
            catalog.records[target_id] = record
            for alias in record.accessions:
                _add_alias(catalog, alias, target_id)

            status = "mapped" if mapped else ("ambiguous" if reason.startswith("ambiguous") else "unmapped")
            rows.append({
                "snapshot": snapshot,
                "target_id": target_id,
                "taxon_id": taxon_id or "",
                "sequence_length": len(sequence),
                "status": status,
                "reason": reason or "mapped",
                "mapping_method": method,
                "uniprot_accession": selected or "",
                "uniprot_entry_name": mapped.entry_name if mapped and mapped.entry_name else "",
                "reviewed": "" if mapped is None or mapped.reviewed is None else int(mapped.reviewed),
                "source_candidate_count": len(source_candidates),
                "exact_sequence_candidate_count": len(sequence_candidates),
                "present_in_snapshot": int(bool(source_candidates or sequence_candidates)),
            })
    return OfficialTargetLoadResult(catalog=catalog, rows=rows)
