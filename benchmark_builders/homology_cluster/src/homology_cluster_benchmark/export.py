from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .config import PREFIX_TO_NAMESPACE, SPLITS
from .models import LabelBuildResult
from .ontology import Ontology


PICKLE_FILES = (
    "train_data.pkl", "train_data_train.pkl", "train_data_valid.pkl",
    "test_data.pkl", "terms.pkl",
)


def _pickle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ["proteins", "sequences", "annotations"]].reset_index(drop=True)


def export_pickles(labels: LabelBuildResult, output_dir: Path) -> dict[str, Path]:
    train = _pickle_frame(labels.frames["training"])
    validation = _pickle_frame(labels.frames["validation"])
    test = _pickle_frame(labels.frames["test"])
    development = pd.concat([train, validation], ignore_index=True)
    if not development.empty:
        development = development.sort_values("proteins", kind="stable").reset_index(drop=True)
    terms = pd.DataFrame({"terms": list(labels.term_universe)})
    paths = {
        "train_data.pkl": output_dir / "train_data.pkl",
        "train_data_train.pkl": output_dir / "train_data_train.pkl",
        "train_data_valid.pkl": output_dir / "train_data_valid.pkl",
        "test_data.pkl": output_dir / "test_data.pkl",
        "terms.pkl": output_dir / "terms.pkl",
    }
    development.to_pickle(paths["train_data.pkl"])
    train.to_pickle(paths["train_data_train.pkl"])
    validation.to_pickle(paths["train_data_valid.pkl"])
    test.to_pickle(paths["test_data.pkl"])
    terms.to_pickle(paths["terms.pkl"])
    return paths


def export_pfp_csvs(
    labels: LabelBuildResult,
    ontology: Ontology,
    output_dir: Path,
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for prefix, namespace in PREFIX_TO_NAMESPACE.items():
        terms = [
            term for term in labels.term_universe if ontology.namespace(term) == namespace
        ]
        for split in SPLITS:
            frame = labels.frames[split]
            path = output_dir / f"{prefix}-{split}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["proteins", "sequences", *terms])
                for row in frame.sort_values("proteins", kind="stable").itertuples(index=False):
                    annotation_set = set(row.annotations)
                    if not annotation_set.intersection(terms):
                        continue
                    writer.writerow([
                        row.proteins,
                        row.sequences,
                        *(int(term in annotation_set) for term in terms),
                    ])
            written[path.name] = path
    return written


def export_all(
    labels: LabelBuildResult,
    ontology: Ontology,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = export_pickles(labels, output_dir)
    paths.update(export_pfp_csvs(labels, ontology, output_dir))
    return paths
