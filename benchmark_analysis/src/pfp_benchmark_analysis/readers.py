"""Readers for benchmark CSVs and optional scientific inputs."""

from __future__ import annotations

import csv
import gzip
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    TextIO,
    Tuple,
)

from .config import (
    CategorySourceSpec,
    ModalityAvailabilitySpec,
    SourceAnnotationSpec,
    TaxonomySourceSpec,
)
from .models import Observation, SourceRecord, Taxon, TaxonomyConflict


@contextmanager
def open_text(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield handle


def read_benchmark_csv(
    path: Path,
    *,
    dataset_id: str,
    aspect: str,
    split: str,
    root: str,
) -> Tuple[Tuple[Observation, ...], dict, Tuple[str, ...], Mapping[str, str]]:
    observations = []
    sequences = {}
    term_support: MutableMapping[str, int] = defaultdict(int)
    label_counts = []
    non_root_counts = []
    sequence_lengths = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        terms = tuple(header[2:])
        for row in reader:
            protein_id, sequence = row[:2]
            sequences[protein_id] = sequence
            labels = row[2:]
            positive = tuple(term for term, value in zip(terms, labels) if value == "1")
            positive_set = set(positive)
            for term in positive:
                term_support[term] += 1
            non_root = tuple(term for term in positive if term != root)
            all_zero = not positive
            root_positive = root in positive_set
            root_only = root_positive and not non_root
            observations.append(
                Observation(
                    dataset_id=dataset_id,
                    aspect=aspect,
                    split=split,
                    protein_id=protein_id,
                    sequence_length=len(sequence),
                    label_count=len(positive),
                    non_root_label_count=len(non_root),
                    root_positive=root_positive,
                    root_only=root_only,
                    all_zero=all_zero,
                    positive_terms=positive,
                )
            )
            label_counts.append(len(positive))
            non_root_counts.append(len(non_root))
            sequence_lengths.append(len(sequence))
    profile = {
        "dataset_id": dataset_id,
        "aspect": aspect,
        "split": split,
        "file": path.name,
        "proteins": len(observations),
        "terms": len(terms),
        "positive_labels": sum(label_counts),
        "root_positive_proteins": sum(item.root_positive for item in observations),
        "root_only_proteins": sum(item.root_only for item in observations),
        "all_zero_proteins": sum(item.all_zero for item in observations),
        "terms_with_support": sum(term_support[term] > 0 for term in terms),
    }
    return tuple(observations), profile, terms, sequences


def _normalise_annotations(value: object, separator: str) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return tuple(
            sorted(term.strip() for term in value.split(separator) if term.strip())
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(sorted(set(value)))
    try:
        if value != value:  # pandas NaN
            return tuple()
    except Exception:
        pass
    return tuple(sorted(value))


def load_source_annotations(
    spec: SourceAnnotationSpec,
) -> Tuple[Mapping[str, Mapping[str, SourceRecord]], Tuple[Path, ...]]:
    import pandas as pd

    by_split: Dict[str, Dict[str, SourceRecord]] = {
        "training": {},
        "validation": {},
        "test": {},
    }
    input_paths = []
    if spec.type == "pfp-pickle-directory":
        for split, filename in spec.split_files.items():
            path = spec.path / filename
            input_paths.append(path)
            frame = pd.read_pickle(path)
            for row in frame[
                [spec.protein_id_column, spec.sequence_column, spec.annotations_column]
            ].itertuples(index=False, name=None):
                protein_id, sequence, annotations = row
                by_split[split][protein_id] = SourceRecord(
                    protein_id,
                    sequence,
                    _normalise_annotations(annotations, spec.annotation_separator),
                )
    else:
        path = spec.path
        input_paths.append(path)
        with open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                split = row[spec.split_column]
                protein_id = row[spec.protein_id_column]
                sequence = row[spec.sequence_column]
                by_split[split][protein_id] = SourceRecord(
                    protein_id,
                    sequence,
                    _normalise_annotations(
                        row[spec.annotations_column], spec.annotation_separator
                    ),
                )
    return by_split, tuple(input_paths)


def _record_taxon(
    result: MutableMapping[str, List[Taxon]],
    protein_id: str,
    taxon: Taxon,
) -> None:
    if not protein_id:
        return
    values = result.setdefault(protein_id, [])
    if taxon not in values:
        values.append(taxon)


def _load_taxonomy_tsv(
    spec: TaxonomySourceSpec,
    wanted_ids: Set[str],
    result: MutableMapping[str, List[Taxon]],
) -> None:
    with open_text(spec.path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            taxon_id = row[spec.taxon_id_column].strip()
            if not taxon_id:
                continue
            taxon_name = (
                row[spec.taxon_name_column].strip() if spec.taxon_name_column else ""
            )
            taxon = Taxon(
                taxon_id,
                taxon_name,
                str(spec.path.resolve()),
                spec.name,
                spec.priority,
            )
            for column in spec.id_columns:
                for protein_id in re.split(r"[;,]", row[column]):
                    protein_id = protein_id.strip()
                    if protein_id in wanted_ids:
                        _record_taxon(result, protein_id, taxon)


def _load_uniprot_dat(
    spec: TaxonomySourceSpec,
    wanted_sequences: Mapping[str, str],
    result: MutableMapping[str, List[Taxon]],
) -> None:
    accessions = []
    taxon_id = ""
    organism_parts = []
    sequence_parts = []
    in_sequence = False

    def normalize_sequence(value: str) -> str:
        return re.sub(r"[^A-Z]", "", value.upper())

    def publish() -> None:
        if not taxon_id or not accessions:
            return
        organism = " ".join(organism_parts).strip()
        primary_accession = accessions[0]
        record_sequence = normalize_sequence("".join(sequence_parts))
        for accession in accessions:
            benchmark_sequence = wanted_sequences.get(accession)
            if benchmark_sequence is not None:
                taxon = Taxon(
                    taxon_id,
                    organism,
                    str(spec.path.resolve()),
                    spec.name,
                    spec.priority,
                    accession_role=(
                        "primary" if accession == primary_accession else "secondary"
                    ),
                    record_primary_accession=primary_accession,
                    sequence_matches_benchmark=(
                        record_sequence == normalize_sequence(benchmark_sequence)
                        if record_sequence
                        else None
                    ),
                    resolution_basis="unresolved UniProt accession candidate",
                )
                _record_taxon(result, accession, taxon)

    with open_text(spec.path) as handle:
        for raw in handle:
            if raw.startswith("AC   "):
                accessions.extend(
                    item.strip() for item in raw[5:].split(";") if item.strip()
                )
            elif raw.startswith("OX   "):
                match = re.search(r"NCBI_TaxID=(\d+)", raw)
                if match:
                    taxon_id = match.group(1)
            elif raw.startswith("OS   "):
                organism_parts.append(raw[5:].strip())
            elif raw.startswith("SQ   "):
                in_sequence = True
            elif raw.startswith("//"):
                publish()
                accessions = []
                taxon_id = ""
                organism_parts = []
                sequence_parts = []
                in_sequence = False
            elif in_sequence:
                sequence_parts.append(re.sub(r"[^A-Z]", "", raw))
        publish()


def _candidate_order(item: Taxon) -> tuple:
    role_rank = {"primary": 0, "direct": 1, "secondary": 2}
    return (
        item.sequence_matches_benchmark is not True,
        role_rank.get(item.accession_role, 3),
        not bool(item.taxon_name),
        item.record_primary_accession,
        item.taxon_id,
        item.source,
    )


def _unresolved_conflicts(
    protein_id: str, candidates: Iterable[Taxon], reason: str
) -> Tuple[TaxonomyConflict, ...]:
    return tuple(
        TaxonomyConflict(
            protein_id=protein_id,
            selected=None,
            alternative=item,
            resolution=reason,
            status="unresolved",
        )
        for item in sorted(candidates, key=_candidate_order)
    )


def _select_within_source(
    protein_id: str, source_name: str, candidates: Iterable[Taxon]
) -> Tuple[Optional[Taxon], Tuple[TaxonomyConflict, ...]]:
    options = sorted(candidates, key=_candidate_order)
    taxon_ids = {item.taxon_id for item in options}
    if len(taxon_ids) == 1:
        selected = options[0]
        basis = (
            "exact benchmark-sequence match within taxonomy source"
            if selected.sequence_matches_benchmark is True
            else "unambiguous accession mapping within taxonomy source"
        )
        return replace(selected, resolution_basis=basis), tuple()

    exact = [item for item in options if item.sequence_matches_benchmark is True]
    exact_ids = {item.taxon_id for item in exact}
    if len(exact_ids) == 1:
        selected = replace(
            exact[0],
            resolution_basis="unique taxon identified by exact benchmark sequence",
        )
        conflicts = tuple(
            TaxonomyConflict(
                protein_id=protein_id,
                selected=selected,
                alternative=item,
                resolution=selected.resolution_basis,
            )
            for item in options
            if item.taxon_id != selected.taxon_id
        )
        return selected, conflicts

    if exact:
        reason = (
            f"multiple taxa in taxonomy source {source_name!r} exactly match the "
            "benchmark sequence"
        )
    else:
        reason = (
            f"conflicting aliases in taxonomy source {source_name!r} have no exact "
            "benchmark-sequence match"
        )
    return None, _unresolved_conflicts(protein_id, options, reason)


def _resolve_taxonomy(
    candidates: Mapping[str, Iterable[Taxon]],
) -> Tuple[Mapping[str, Taxon], Tuple[TaxonomyConflict, ...]]:
    resolved: Dict[str, Taxon] = {}
    conflicts = []
    for protein_id in sorted(candidates):
        by_source: MutableMapping[str, List[Taxon]] = {}
        for item in candidates[protein_id]:
            by_source.setdefault(item.source_name, []).append(item)
        source_winners = []
        for source_name, items in sorted(by_source.items()):
            winner, source_conflicts = _select_within_source(
                protein_id, source_name, items
            )
            conflicts.extend(source_conflicts)
            if winner is not None:
                source_winners.append(winner)
        if not source_winners:
            continue
        ranked = sorted(
            source_winners,
            key=lambda item: (
                -item.source_priority,
                *_candidate_order(item),
                item.source_name,
            ),
        )
        highest_priority = ranked[0].source_priority
        highest = [item for item in ranked if item.source_priority == highest_priority]
        highest_ids = sorted({item.taxon_id for item in highest})
        if len(highest_ids) > 1:
            reason = (
                f"equal-priority taxonomy sources disagree at priority "
                f"{highest_priority}"
            )
            conflicts.extend(_unresolved_conflicts(protein_id, highest, reason))
            continue
        selected = highest[0]
        resolved[protein_id] = selected
        for alternative in ranked:
            if alternative.taxon_id == selected.taxon_id:
                continue
            conflicts.append(
                TaxonomyConflict(
                    protein_id=protein_id,
                    selected=selected,
                    alternative=alternative,
                    resolution="selected explicitly higher-priority taxonomy source",
                )
            )
    return resolved, tuple(conflicts)


def load_taxonomy(
    specs: Iterable[TaxonomySourceSpec], wanted_sequences: Mapping[str, str]
) -> Tuple[Mapping[str, Taxon], Tuple[Path, ...], Tuple[TaxonomyConflict, ...]]:
    candidates: Dict[str, List[Taxon]] = {}
    wanted_ids = set(wanted_sequences)
    paths = []
    for spec in specs:
        paths.append(spec.path)
        if spec.type == "tsv":
            _load_taxonomy_tsv(spec, wanted_ids, candidates)
        else:
            _load_uniprot_dat(spec, wanted_sequences, candidates)
    result, conflicts = _resolve_taxonomy(candidates)
    return result, tuple(paths), conflicts


def load_modality_availability(
    spec: ModalityAvailabilitySpec,
) -> Tuple[Mapping[Tuple[str, str], Mapping[str, bool]], Tuple[str, ...]]:
    records: Dict[Tuple[str, str], Mapping[str, bool]] = {}
    modalities = set()
    with open_text(spec.path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            protein_id = row[spec.protein_id_column].strip()
            modality = row[spec.modality_column].strip()
            key = (protein_id, modality)
            records[key] = {
                name: row[state.column].strip().casefold() in state.true_values
                for name, state in spec.states.items()
            }
            modalities.add(modality)
    return records, tuple(sorted(modalities))


def load_category_source(
    spec: CategorySourceSpec,
) -> Mapping[str, Tuple[Tuple[str, str], ...]]:
    result: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    with open_text(spec.path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            protein_id = row[spec.protein_id_column].strip()
            category_id = row[spec.category_id_column].strip()
            category_name = (
                row[spec.category_name_column].strip()
                if spec.category_name_column
                else ""
            )
            result[protein_id].add((category_id, category_name))
    return {protein: tuple(sorted(values)) for protein, values in result.items()}
