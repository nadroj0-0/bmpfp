#!/usr/bin/env python3
"""Capture prediction and truth arrays during PFP CAFA evaluation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class EvaluationArrayCapture:
    """Observe the normal PFP writers and retain their numerical arrays."""

    def __init__(self, evaluation_module: Any) -> None:
        self.module = evaluation_module
        self.predictions: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.protein_ids: list[str] | None = None
        self.go_terms: list[str] | None = None
        self.computed_ia_values: dict[str, float] | None = None
        self._originals: dict[str, Any] = {}

    def __enter__(self) -> "EvaluationArrayCapture":
        names = (
            "save_predictions_cafa_format",
            "save_ground_truth_cafa_format",
            "save_ia_file",
        )
        self._originals = {name: getattr(self.module, name) for name in names}

        def save_predictions(
            predictions: np.ndarray,
            protein_ids: Sequence[str],
            go_terms: Sequence[str],
            output_file: str | Path,
        ) -> None:
            self.predictions = np.asarray(predictions)
            self.protein_ids = [str(value) for value in protein_ids]
            self.go_terms = [str(value) for value in go_terms]
            self._originals["save_predictions_cafa_format"](
                predictions, protein_ids, go_terms, output_file
            )

        def save_truth(
            labels: np.ndarray,
            protein_ids: Sequence[str],
            go_terms: Sequence[str],
            output_file: str | Path,
        ) -> None:
            self.labels = np.asarray(labels)
            self.protein_ids = [str(value) for value in protein_ids]
            self.go_terms = [str(value) for value in go_terms]
            self._originals["save_ground_truth_cafa_format"](
                labels, protein_ids, go_terms, output_file
            )

        def save_ia(values: Mapping[str, float], output_file: str | Path) -> None:
            self.computed_ia_values = {
                str(term): float(value) for term, value in values.items()
            }
            self._originals["save_ia_file"](values, output_file)

        self.module.save_predictions_cafa_format = save_predictions
        self.module.save_ground_truth_cafa_format = save_truth
        self.module.save_ia_file = save_ia
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for name, value in self._originals.items():
            setattr(self.module, name, value)

    def persist(self, output_dir: Path, ia_file: Path | None) -> None:
        np.savez_compressed(
            output_dir / "predictions.npz",
            scores=np.asarray(self.predictions),
            truth=np.asarray(self.labels),
            protein_ids=np.asarray(self.protein_ids, dtype=str),
            go_terms=np.asarray(self.go_terms, dtype=str),
        )
        destination = output_dir / "ia.txt"
        if ia_file is not None:
            shutil.copyfile(ia_file, destination)
        elif self.computed_ia_values is not None:
            destination.write_text(
                "".join(
                    f"{term}\t{value:.6f}\n"
                    for term, value in self.computed_ia_values.items()
                ),
                encoding="utf-8",
            )
