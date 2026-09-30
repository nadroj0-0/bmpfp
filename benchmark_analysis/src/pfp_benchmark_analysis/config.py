"""Configuration loading for descriptive benchmark analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

@dataclass(frozen=True)
class SourceAnnotationSpec:
    type: str
    path: Path
    split_files: Mapping[str, str]
    protein_id_column: str
    sequence_column: str
    annotations_column: str
    split_column: str
    annotation_separator: str
    projection_policy: str


@dataclass(frozen=True)
class TaxonomySourceSpec:
    type: str
    path: Path
    name: str
    priority: int
    id_columns: Tuple[str, ...]
    taxon_id_column: str
    taxon_name_column: str


@dataclass(frozen=True)
class ModalityStateSpec:
    column: str
    true_values: Tuple[str, ...]


@dataclass(frozen=True)
class ModalityAvailabilitySpec:
    type: str
    path: Path
    protein_id_column: str
    modality_column: str
    states: Mapping[str, ModalityStateSpec]


@dataclass(frozen=True)
class CategorySourceSpec:
    name: str
    path: Path
    protein_id_column: str
    category_id_column: str
    category_name_column: str


@dataclass(frozen=True)
class DatasetConfig:
    id: str
    benchmark_dir: Path
    obo_file: Path
    source_annotations: Optional[SourceAnnotationSpec]
    taxonomy_sources: Tuple[TaxonomySourceSpec, ...]
    modality_availability: Optional[ModalityAvailabilitySpec]
    category_sources: Tuple[CategorySourceSpec, ...]


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    top_n: int
    datasets: Tuple[DatasetConfig, ...]
    source_path: Path


def _mapping(value: Any) -> Mapping[str, Any]:
    return value


def _text(value: Any) -> str:
    return str(value)


def _path(value: Any, base: Path) -> Path:
    text = _text(value)
    path = Path(text).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _string_tuple(value: Any) -> Tuple[str, ...]:
    return tuple(str(item) for item in value)


def _source_annotations(raw: Any, base: Path) -> SourceAnnotationSpec:
    value = _mapping(raw)
    source_type = _text(value["type"])
    split_files_raw = value.get(
        "split_files",
        {
            "training": "train_data_train.pkl",
            "validation": "train_data_valid.pkl",
            "test": "test_data.pkl",
        },
    )
    split_files = _mapping(split_files_raw)
    if source_type == "pfp-pickle-directory":
        split_files = {split: _text(name) for split, name in split_files.items()}
    return SourceAnnotationSpec(
        type=source_type,
        path=_path(value["path"], base),
        split_files=split_files,
        protein_id_column=_text(value.get("protein_id_column", "proteins")),
        sequence_column=_text(value.get("sequence_column", "sequences")),
        annotations_column=_text(value.get("annotations_column", "annotations")),
        split_column=_text(value.get("split_column", "split")),
        annotation_separator=_text(value.get("annotation_separator", ";")),
        projection_policy=_text(
            value.get("projection_policy", "unspecified-label-universe-projection")
        ),
    )


def _taxonomy_source(raw: Any, base: Path) -> TaxonomySourceSpec:
    value = _mapping(raw)
    source_type = _text(value["type"])
    priority = value.get("priority", 0)
    path = _path(value["path"], base)
    return TaxonomySourceSpec(
        type=source_type,
        path=path,
        name=_text(value.get("name", str(path))),
        priority=priority,
        id_columns=_string_tuple(value.get("id_columns", ["protein_id"])),
        taxon_id_column=_text(value.get("taxon_id_column", "taxon_id")),
        taxon_name_column=_text(value.get("taxon_name_column", "taxon_name")),
    )


def _modality_availability(raw: Any, base: Path) -> ModalityAvailabilitySpec:
    value = _mapping(raw)
    availability_type = _text(value["type"])
    states_raw = _mapping(value["states"])
    states = {}
    for state_name, state_raw in states_raw.items():
        state = _mapping(state_raw)
        states[state_name] = ModalityStateSpec(
            column=_text(state["column"]),
            true_values=tuple(
                item.casefold()
                for item in _string_tuple(state["true_values"])
            ),
        )
    return ModalityAvailabilitySpec(
        type=availability_type,
        path=_path(value["path"], base),
        protein_id_column=_text(value.get("protein_id_column", "protein_id")),
        modality_column=_text(value.get("modality_column", "modality")),
        states=states,
    )


def _category_source(raw: Any, base: Path) -> CategorySourceSpec:
    value = _mapping(raw)
    return CategorySourceSpec(
        name=_text(value["name"]),
        path=_path(value["path"], base),
        protein_id_column=_text(value["protein_id_column"]),
        category_id_column=_text(value["category_id_column"]),
        category_name_column=_text(value.get("category_name_column", "")),
    )


def _dataset(raw: Any, base: Path) -> DatasetConfig:
    value = _mapping(raw)
    taxonomy_raw = value.get("taxonomy_sources", [])
    category_raw = value.get("category_sources", [])
    category_sources = tuple(
        _category_source(item, base) for item in category_raw
    )
    taxonomy_sources = tuple(
        _taxonomy_source(item, base) for item in taxonomy_raw
    )
    return DatasetConfig(
        id=_text(value["id"]),
        benchmark_dir=_path(value["benchmark_dir"], base),
        obo_file=_path(value["obo_file"], base),
        source_annotations=(
            _source_annotations(value["source_annotations"], base)
            if "source_annotations" in value
            else None
        ),
        taxonomy_sources=taxonomy_sources,
        modality_availability=(
            _modality_availability(value["modality_availability"], base)
            if "modality_availability" in value
            else None
        ),
        category_sources=category_sources,
    )


def load_config(path: Path) -> RunConfig:
    source_path = path.expanduser().resolve()
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    value = _mapping(raw)
    datasets_raw = value["datasets"]
    datasets = tuple(
        _dataset(item, source_path.parent) for item in datasets_raw
    )
    top_n = value.get("top_n", 20)
    return RunConfig(
        run_name=_text(value["run_name"]),
        top_n=top_n,
        datasets=datasets,
        source_path=source_path,
    )
