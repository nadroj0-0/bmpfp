#!/usr/bin/env python3
"""Shared loading and scoring helpers for PFP sensitivity analyses."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from label_space_common import ASPECT_TO_ROOT


def load_evaluation_summary(
    evaluation_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Load the checkpoint-evaluation output used by the analyses."""
    requested = evaluation_path.expanduser().resolve()
    summary_path = (
        requested / "evaluation_summary.json" if requested.is_dir() else requested
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, summary_path.parent


def load_aspect_bundle(
    evaluation: Mapping[str, Any], root: Path, aspect: str
) -> dict[str, Any]:
    aspect_root = root / aspect
    result = json.loads(
        (aspect_root / "results.json").read_text(encoding="utf-8")
    )
    array_path = aspect_root / "predictions.npz"
    with np.load(array_path, allow_pickle=False) as archive:
        scores = np.asarray(archive["scores"])
        truth = np.asarray(archive["truth"])
        protein_ids = [str(value) for value in archive["protein_ids"].tolist()]
        go_terms = [str(value) for value in archive["go_terms"].tolist()]
    ia_path = aspect_root / "ia.txt"
    required_root = ASPECT_TO_ROOT[aspect]
    canonical_metrics = {
        key.removeprefix("cafa_"): float(value)
        for key, value in result.items()
        if key.startswith("cafa_") and not isinstance(value, bool)
    }
    specification = {
        **dict(result),
        "canonical_cafa_metrics": canonical_metrics,
    }
    return {
        "specification": specification,
        "scores": scores,
        "truth": truth.astype(np.uint8, copy=False),
        "protein_ids": protein_ids,
        "go_terms": go_terms,
        "ia_path": ia_path,
        "root_index": go_terms.index(required_root),
        "evaluation_split": result["evaluation_split"],
    }


def cohort_masks(truth: np.ndarray, root_index: int) -> dict[str, np.ndarray]:
    non_root = np.ones(truth.shape[1], dtype=bool)
    non_root[root_index] = False
    has_non_root = truth[:, non_root].any(axis=1)
    has_any = truth.any(axis=1)
    root_only = (truth[:, root_index] == 1) & ~has_non_root
    return {
        "eligible_non_root": has_non_root,
        "root_only": root_only,
        "all_zero": ~has_any,
        "non_root_columns": non_root,
    }


def _threshold_metrics(truth: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    true = truth.astype(bool, copy=False)
    true_positive = np.logical_and(predicted, true).sum(axis=1)
    predicted_count = predicted.sum(axis=1)
    true_count = true.sum(axis=1)
    covered = predicted_count > 0
    precision = (
        float(np.mean(true_positive[covered] / predicted_count[covered]))
        if covered.any()
        else 0.0
    )
    recall = float(np.mean(true_positive / true_count)) if len(true_count) else 0.0
    macro_f = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    tp = int(true_positive.sum())
    fp = int(np.logical_and(predicted, ~true).sum())
    fn = int(np.logical_and(~predicted, true).sum())
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f": macro_f,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f": micro_f,
        "coverage": float(np.mean(covered)) if len(covered) else 0.0,
    }


def flat_non_root_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    root_index: int,
    fixed_threshold: float,
) -> dict[str, Any]:
    masks = cohort_masks(truth, root_index)
    eligible = masks["eligible_non_root"]
    columns = masks["non_root_columns"]
    cohort_truth = truth[eligible][:, columns]
    cohort_scores = scores[eligible][:, columns]
    if cohort_truth.shape[0] == 0:
        return {"status": "not_evaluable_no_non_root_targets"}
    thresholds = sorted(
        set(np.linspace(0.01, 0.99, 100).tolist() + [float(fixed_threshold)])
    )
    rows = [_threshold_metrics(cohort_truth, cohort_scores, value) for value in thresholds]
    best = max(rows, key=lambda row: (row["macro_f"], -row["threshold"]))
    fixed = min(rows, key=lambda row: abs(row["threshold"] - fixed_threshold))
    return {
        "status": "complete",
        "policy": (
            "protein-centric flat metrics after removing the ontology root; "
            "no GO propagation; diagnostic only, not a CAFA metric"
        ),
        "best": best,
        "fixed_at_canonical_threshold": fixed,
    }


def write_prediction_file(
    path: Path,
    protein_ids: Sequence[str],
    go_terms: Sequence[str],
    scores: np.ndarray | None,
    root_index: int | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if scores is None:
            root = go_terms[root_index]
            for protein_id in protein_ids:
                handle.write(f"{protein_id}\t{root}\t1.000000\n")
            return
        lines: list[str] = []
        for row_index, protein_id in enumerate(protein_ids):
            for term_index in np.flatnonzero(scores[row_index] > 0):
                lines.append(
                    f"{protein_id}\t{go_terms[int(term_index)]}\t"
                    f"{float(scores[row_index, term_index]):.6f}\n"
                )
                if len(lines) >= 16_384:
                    handle.write("".join(lines))
                    lines.clear()
        if lines:
            handle.write("".join(lines))


def write_truth_file(
    path: Path,
    protein_ids: Sequence[str],
    go_terms: Sequence[str],
    truth: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        lines: list[str] = []
        for row_index, protein_id in enumerate(protein_ids):
            for term_index in np.flatnonzero(truth[row_index] == 1):
                lines.append(f"{protein_id}\t{go_terms[int(term_index)]}\n")
                if len(lines) >= 16_384:
                    handle.write("".join(lines))
                    lines.clear()
        if lines:
            handle.write("".join(lines))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_cafa_results(result_dir: Path, fixed_threshold: float) -> dict[str, Any]:
    all_path = result_dir / "evaluation_all.tsv"
    best_s_path = result_dir / "evaluation_best_s.tsv"
    best_w_path = result_dir / "evaluation_best_f_w.tsv"
    rows = _read_tsv(all_path)
    best = max(rows, key=lambda row: float(row["f"]))
    fixed = min(rows, key=lambda row: abs(float(row["tau"]) - fixed_threshold))
    best_s = _read_tsv(best_s_path)[0]
    best_w = _read_tsv(best_w_path)[0]

    def selected(row: Mapping[str, str], fields: Sequence[str]) -> dict[str, float]:
        return {
            field: float(row[field])
            for field in fields
            if field in row and row[field] not in ("", None)
        }

    return {
        "fmax": float(best["f"]),
        "threshold": float(best["tau"]),
        "precision": float(best["pr"]),
        "recall": float(best["rc"]),
        "coverage": float(best["cov"]) if "cov" in best else None,
        "wfmax": float(best_w["f_w"]),
        "wthreshold": float(best_w["tau"]),
        "wprecision": float(best_w["pr_w"]),
        "wrecall": float(best_w["rc_w"]),
        "smin": float(best_s["s"]),
        "fixed_at_canonical_threshold": selected(
            fixed, ("tau", "f", "pr", "rc", "cov", "f_w", "pr_w", "rc_w", "s")
        ),
    }


def run_cafa_evaluation(
    *,
    obo_file: Path,
    ia_file: Path,
    destination: Path,
    protein_ids: Sequence[str],
    go_terms: Sequence[str],
    truth: np.ndarray,
    fixed_threshold: float,
    scores: np.ndarray | None,
    root_index: int | None = None,
) -> dict[str, Any]:
    from cafaeval.evaluation import cafa_eval, write_results

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pfp-label-sensitivity-") as name:
        temporary = Path(name)
        prediction_dir = temporary / "predictions"
        prediction_dir.mkdir()
        write_prediction_file(
            prediction_dir / "model.tsv",
            protein_ids,
            go_terms,
            scores,
            root_index=root_index,
        )
        truth_file = temporary / "truth.tsv"
        write_truth_file(truth_file, protein_ids, go_terms, truth)
        results = cafa_eval(
            str(obo_file),
            str(prediction_dir),
            str(truth_file),
            ia=str(ia_file),
            no_orphans=False,
            norm="cafa",
            prop="max",
        )
        write_results(*results, out_dir=str(destination))
    parsed = parse_cafa_results(destination, fixed_threshold)
    return parsed
