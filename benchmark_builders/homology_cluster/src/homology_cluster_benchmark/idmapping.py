from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
import time

from .inputs import open_text
from .models import MappingDecision, ProteinCatalog
from .uniref import UniRefIndex
from .uniref_scaffold import uniref_scaffold


LOGGER = logging.getLogger(__name__)


def _split_mapping_values(value: str) -> set[str]:
    return {item.strip() for item in value.split(";") if item.strip()}


def load_uniref_mappings(
    path: Path,
    requested_accessions: set[str],
    catalog: ProteinCatalog,
    uniref_index: UniRefIndex,
    uniref_level: int = 90,
) -> list[MappingDecision]:
    """Stream the headerless 22-column idmapping_selected table."""
    scaffold = uniref_scaffold(uniref_level)
    started = time.monotonic()
    LOGGER.info(
        "ID mapping scan started: %s requested_accessions=%d", path, len(requested_accessions)
    )
    lookup_accessions = set(requested_accessions)
    lookup_accessions.update(
        catalog.alias_to_primary[accession]
        for accession in requested_accessions
        if accession in catalog.alias_to_primary
    )
    values: dict[str, set[str]] = defaultdict(set)
    seen: set[str] = set()
    with open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number % 1_000_000 == 0:
                LOGGER.info(
                    "ID mapping progress: lines=%d requested_rows_seen=%d elapsed_seconds=%.1f",
                    line_number, len(seen), time.monotonic() - started,
                )
            if not raw_line.strip():
                continue
            columns = raw_line.rstrip("\n\r").split("\t")
            accession = columns[0]
            if accession not in lookup_accessions:
                continue
            seen.add(accession)
            values[accession].update(
                _split_mapping_values(columns[scaffold.idmapping_column_index])
            )

    candidate_sets: dict[str, set[str]] = {}
    all_candidate_ids: set[str] = set()
    for raw_accession in requested_accessions:
        candidates = (
            set(values.get(raw_accession, set()))
            | set(values.get(catalog.alias_to_primary.get(raw_accession, raw_accession), set()))
        )
        candidate_sets[raw_accession] = candidates
        all_candidate_ids.update(candidates)
    del lookup_accessions, values
    present_uniref_ids = uniref_index.present_ids(all_candidate_ids)
    del all_candidate_ids

    decisions: list[MappingDecision] = []
    for raw_accession in sorted(requested_accessions):
        if raw_accession in catalog.ambiguous_aliases:
            protein_id = raw_accession
            accession_action = "ambiguous-secondary"
        else:
            protein_id = catalog.alias_to_primary.get(raw_accession, raw_accession)
            accession_action = "exact" if protein_id == raw_accession else "secondary-to-primary"
        candidates = candidate_sets.pop(raw_accession)
        if raw_accession in catalog.ambiguous_aliases:
            decisions.append(MappingDecision(
                raw_accession=raw_accession,
                protein_id=protein_id,
                accession_action=accession_action,
                status="ambiguous-secondary",
                detail="secondary accession maps to more than one selected-UniProt primary",
                exists_in_fasta=None,
                canonical_sequence_available=False,
            ))
        elif protein_id not in catalog.records:
            present = sorted(candidates & present_uniref_ids)
            decisions.append(MappingDecision(
                raw_accession=raw_accession,
                protein_id=protein_id,
                accession_action=accession_action,
                status="not-in-selected-uniprot",
                detail=(
                    "GOA accession is outside the selected UniProt population; "
                    "idmapping candidates in FASTA="
                    + (";".join(present) or "none")
                ),
                exists_in_fasta=None,
                canonical_sequence_available=False,
            ))
        elif len(candidates) > 1:
            present = sorted(candidates & present_uniref_ids)
            missing = sorted(candidates - present_uniref_ids)
            decisions.append(MappingDecision(
                raw_accession=raw_accession, protein_id=protein_id,
                accession_action=accession_action, status="ambiguous",
                detail=(
                    f"multiple {scaffold.display_name} mappings; present_in_fasta="
                    + (";".join(present) or "none")
                    + "; missing_from_fasta="
                    + (";".join(missing) or "none")
                ),
                exists_in_fasta=None,
                canonical_sequence_available=protein_id in catalog.records,
            ))
        elif not candidates:
            seen_mapping_row = raw_accession in seen or protein_id in seen
            status = "unmapped-blank" if seen_mapping_row else "unmapped-absent"
            reason = (
                f"blank {scaffold.display_name} column" if seen_mapping_row
                else "accession absent from idmapping; obsolete status cannot be inferred without a deleted-accession source"
            )
            decisions.append(MappingDecision(
                raw_accession=raw_accession, protein_id=protein_id,
                accession_action=accession_action, status=status, detail=reason,
                exists_in_fasta=None,
                canonical_sequence_available=protein_id in catalog.records,
            ))
        else:
            selected_uniref_id = next(iter(candidates))
            exists = selected_uniref_id in present_uniref_ids
            decisions.append(MappingDecision(
                raw_accession=raw_accession, protein_id=protein_id,
                accession_action=accession_action, uniref90_id=selected_uniref_id,
                status=(
                    "mapped" if exists
                    else f"missing-from-{scaffold.slug}-fasta"
                ),
                detail=(
                    "" if exists else
                    f"mapped {scaffold.display_name} identifier is absent from frozen FASTA"
                ),
                exists_in_fasta=exists,
                canonical_sequence_available=protein_id in catalog.records,
            ))
    LOGGER.info(
        "ID mapping scan completed: lines=%d decisions=%d requested_rows_seen=%d "
        "elapsed_seconds=%.1f",
        line_number if 'line_number' in locals() else 0,
        len(decisions),
        len(seen),
        time.monotonic() - started,
    )
    return decisions
