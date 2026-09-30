#!/usr/bin/env python3
"""Deterministic calibration helpers for captured PFP prediction arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from label_space_common import OboGraph


@dataclass(frozen=True)
class CalibrationPolicy:
    score_clip_epsilon: float = 1e-6
    bin_l2: float = 5.0
    term_l2: float = 20.0
    minimum_bin_positives: int = 20
    minimum_bin_negatives: int = 20
    minimum_term_positives: int = 20
    minimum_term_negatives: int = 20
    maximum_iterations: int = 200
    optimizer_tolerance: float = 1e-7
    protein_chunk_size: int = 256

    def as_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in (
                "score_clip_epsilon",
                "bin_l2",
                "term_l2",
                "minimum_bin_positives",
                "minimum_bin_negatives",
                "minimum_term_positives",
                "minimum_term_negatives",
                "maximum_iterations",
                "optimizer_tolerance",
                "protein_chunk_size",
            )
        }


@dataclass(frozen=True)
class FFPredPlattPolicy:
    """Settings for the PFP adaptation of FFPred's per-term Platt scaling."""

    score_clip_epsilon: float = 1e-6
    minimum_term_positives: int = 1
    minimum_term_negatives: int = 1
    maximum_iterations: int = 200
    maximum_restarts: int = 4
    optimizer_tolerance: float = 1e-7
    protein_chunk_size: int = 256

    def as_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in (
                "score_clip_epsilon",
                "minimum_term_positives",
                "minimum_term_negatives",
                "maximum_iterations",
                "maximum_restarts",
                "optimizer_tolerance",
                "protein_chunk_size",
            )
        }


def _logit_scores(scores: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def propagate_scores_max(
    scores: np.ndarray,
    go_terms: Sequence[str],
    graph: OboGraph,
    aspect: str,
) -> np.ndarray:
    """Apply CAFA-style max propagation over the is_a + part_of output graph."""
    term_index = {term: index for index, term in enumerate(go_terms)}
    closure = graph.ancestor_closure(aspect)
    propagated = np.asarray(scores, dtype=np.float64).copy()
    original = np.asarray(scores, dtype=np.float64)
    for child, child_index in term_index.items():
        ancestors = closure.get(child)
        if not ancestors:
            continue
        for ancestor in ancestors:
            parent_index = term_index.get(ancestor)
            if parent_index is not None and parent_index != child_index:
                np.maximum(
                    propagated[:, parent_index],
                    original[:, child_index],
                    out=propagated[:, parent_index],
                )
    return propagated


def fit_monotone_hierarchical_calibrator(
    scores: np.ndarray,
    truth: np.ndarray,
    term_ids: Sequence[str],
    term_bins: Sequence[str],
    policy: CalibrationPolicy,
) -> dict[str, Any]:
    """Fit a positive-slope logistic model with regularized bin/term intercepts."""
    protein_count, term_count = scores.shape
    positives = truth.sum(axis=0, dtype=np.int64)
    negatives = protein_count - positives
    total_positive = int(positives.sum())
    total_events = protein_count * term_count
    total_negative = total_events - total_positive
    if not total_positive or not total_negative:
        result = {
            "family": "uncalibrated_insufficient_support",
            "status": "uncalibrated_insufficient_support",
            "reason": "calibration population does not contain both outcome classes",
            "term_ids": list(term_ids),
            "term_bins": list(term_bins),
            "fallback_by_term": ["uncalibrated"] * term_count,
            "support": {
                "proteins": protein_count,
                "events": total_events,
                "positives": total_positive,
                "negatives": total_negative,
                "term_positives": positives.tolist(),
                "term_negatives": negatives.tolist(),
            },
            "policy": policy.as_dict(),
        }
        return result

    bin_labels = sorted(set(term_bins))
    bin_lookup = {label: index for index, label in enumerate(bin_labels)}
    term_bin_index = np.asarray(
        [bin_lookup[value] for value in term_bins], dtype=np.int64
    )
    bin_positives = np.bincount(
        term_bin_index, weights=positives, minlength=len(bin_labels)
    ).astype(np.int64)
    bin_events = np.bincount(
        term_bin_index,
        weights=np.full(term_count, protein_count, dtype=np.int64),
        minlength=len(bin_labels),
    ).astype(np.int64)
    bin_negatives = bin_events - bin_positives
    supported_bins = np.flatnonzero(
        (bin_positives >= policy.minimum_bin_positives)
        & (bin_negatives >= policy.minimum_bin_negatives)
    )
    supported_terms = np.flatnonzero(
        (positives >= policy.minimum_term_positives)
        & (negatives >= policy.minimum_term_negatives)
    )
    bin_parameter = {int(value): index for index, value in enumerate(supported_bins)}
    term_parameter = {int(value): index for index, value in enumerate(supported_terms)}
    bin_slice = slice(2, 2 + len(supported_bins))
    term_slice = slice(bin_slice.stop, bin_slice.stop + len(supported_terms))
    parameter_count = term_slice.stop

    prevalence = total_positive / total_events
    initial = np.zeros(parameter_count, dtype=np.float64)
    initial[0] = math.log(prevalence / (1.0 - prevalence))
    initial[1] = 0.0

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = float(parameters[0])
        slope = math.exp(float(parameters[1]))
        bin_effect_by_term = np.zeros(term_count, dtype=np.float64)
        for bin_index, parameter_index in bin_parameter.items():
            bin_effect_by_term[term_bin_index == bin_index] = parameters[
                bin_slice.start + parameter_index
            ]
        term_effect_by_term = np.zeros(term_count, dtype=np.float64)
        if supported_terms.size:
            term_effect_by_term[supported_terms] = parameters[term_slice]

        loss = 0.0
        gradient = np.zeros_like(parameters)
        residual_by_term = np.zeros(term_count, dtype=np.float64)
        for start in range(0, protein_count, policy.protein_chunk_size):
            stop = min(protein_count, start + policy.protein_chunk_size)
            x = _logit_scores(scores[start:stop], policy.score_clip_epsilon)
            y = truth[start:stop]
            eta = (
                alpha
                + slope * x
                + bin_effect_by_term[np.newaxis, :]
                + term_effect_by_term[np.newaxis, :]
            )
            loss += float(np.sum(np.logaddexp(0.0, eta) - y * eta))
            residual = expit(eta) - y
            gradient[0] += float(residual.sum())
            gradient[1] += float(np.sum(residual * (slope * x)))
            residual_by_term += residual.sum(axis=0)

        if supported_bins.size:
            aggregated = np.bincount(
                term_bin_index,
                weights=residual_by_term,
                minlength=len(bin_labels),
            )
            gradient[bin_slice] = aggregated[supported_bins]
            bin_parameters = parameters[bin_slice]
            loss += 0.5 * policy.bin_l2 * float(np.dot(bin_parameters, bin_parameters))
            gradient[bin_slice] += policy.bin_l2 * bin_parameters
        if supported_terms.size:
            gradient[term_slice] = residual_by_term[supported_terms]
            term_parameters = parameters[term_slice]
            loss += (
                0.5 * policy.term_l2 * float(np.dot(term_parameters, term_parameters))
            )
            gradient[term_slice] += policy.term_l2 * term_parameters
        return loss, gradient

    bounds = [(-30.0, 30.0), (-8.0, 8.0)] + [(-20.0, 20.0)] * (parameter_count - 2)
    fit = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": policy.maximum_iterations,
            "ftol": policy.optimizer_tolerance,
            "gtol": policy.optimizer_tolerance,
            "maxls": 30,
        },
    )
    parameters = np.asarray(fit.x, dtype=np.float64)
    slope = math.exp(float(parameters[1]))

    bin_effects = {
        bin_labels[bin_index]: float(parameters[bin_slice.start + parameter_index])
        for bin_index, parameter_index in bin_parameter.items()
    }
    term_effects = {
        term_ids[term_index]: float(parameters[term_slice.start + parameter_index])
        for term_index, parameter_index in term_parameter.items()
    }
    fallback_by_term = []
    for term_index, bin_index in enumerate(term_bin_index):
        if term_index in term_parameter:
            fallback_by_term.append("term_shrinkage")
        elif int(bin_index) in bin_parameter:
            fallback_by_term.append("aspect_mode_ia_bin")
        else:
            fallback_by_term.append("aspect_mode_platt")

    result = {
        "family": "monotone_multilevel_logistic",
        "formula": (
            "logit(q)=alpha+positive_slope*logit(score)+"
            "regularized_ia_bin_intercept+regularized_term_intercept"
        ),
        "alpha": float(parameters[0]),
        "positive_slope": slope,
        "bin_effects": bin_effects,
        "term_effects": term_effects,
        "term_ids": list(term_ids),
        "term_bins": list(term_bins),
        "fallback_by_term": fallback_by_term,
        "support": {
            "proteins": protein_count,
            "events": total_events,
            "positives": total_positive,
            "negatives": total_negative,
            "bin_labels": bin_labels,
            "bin_positives": bin_positives.tolist(),
            "bin_negatives": bin_negatives.tolist(),
            "term_positives": positives.tolist(),
            "term_negatives": negatives.tolist(),
            "supported_bins": [bin_labels[index] for index in supported_bins],
            "supported_terms": [term_ids[index] for index in supported_terms],
        },
        "policy": policy.as_dict(),
        "optimizer": {
            "name": "scipy_L-BFGS-B",
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "iterations": int(fit.nit),
            "function_evaluations": int(fit.nfev),
            "objective": float(fit.fun),
        },
    }
    return result


def apply_calibrator(
    scores: np.ndarray,
    term_ids: Sequence[str],
    term_bins: Sequence[str],
    model: Mapping[str, Any],
    *,
    protein_chunk_size: int = 256,
) -> tuple[np.ndarray, list[str]]:
    if model["family"] == "uncalibrated_insufficient_support":
        return (
            np.full(scores.shape, np.nan, dtype=np.float32),
            list(model["fallback_by_term"]),
        )
    epsilon = float(model["policy"]["score_clip_epsilon"])
    alpha = float(model["alpha"])
    slope = float(model["positive_slope"])
    bin_effects = dict(model["bin_effects"])
    term_effects = dict(model["term_effects"])
    intercept_by_term = np.asarray(
        [
            float(bin_effects.get(term_bin, 0.0))
            + float(term_effects.get(term_id, 0.0))
            for term_id, term_bin in zip(term_ids, term_bins)
        ],
        dtype=np.float64,
    )
    calibrated = np.empty(scores.shape, dtype=np.float32)
    for start in range(0, scores.shape[0], protein_chunk_size):
        stop = min(scores.shape[0], start + protein_chunk_size)
        x = _logit_scores(scores[start:stop], epsilon)
        calibrated[start:stop] = expit(
            alpha + slope * x + intercept_by_term[np.newaxis, :]
        ).astype(np.float32)
    return calibrated, list(model["fallback_by_term"])


def fit_ffpred_platt_calibrator(
    scores: np.ndarray,
    truth: np.ndarray,
    term_ids: Sequence[str],
    policy: FFPredPlattPolicy,
) -> dict[str, Any]:
    """Fit independent per-term sigmoid calibrators in the style of FFPred."""
    protein_count, term_count = scores.shape

    positives = truth.sum(axis=0, dtype=np.int64)
    negatives = protein_count - positives
    supported = np.flatnonzero(
        (positives >= policy.minimum_term_positives)
        & (negatives >= policy.minimum_term_negatives)
    )
    status_by_term = np.full(term_count, "uncalibrated_insufficient_support", dtype=object)
    status_by_term[supported] = "term_platt"

    if not supported.size:
        result = {
            "family": "ffpred_style_independent_platt",
            "status": "uncalibrated_insufficient_support",
            "formula": "q=1/(1+exp(-(intercept+positive_slope*logit(score))))",
            "term_ids": list(term_ids),
            "intercepts": {},
            "positive_slopes": {},
            "status_by_term": status_by_term.tolist(),
            "support": {
                "proteins": protein_count,
                "term_positives": positives.tolist(),
                "term_negatives": negatives.tolist(),
                "supported_terms": [],
            },
            "policy": policy.as_dict(),
        }
        return result

    selected_positives = positives[supported].astype(np.float64)
    selected_negatives = negatives[supported].astype(np.float64)
    positive_targets = (selected_positives + 1.0) / (selected_positives + 2.0)
    negative_targets = 1.0 / (selected_negatives + 2.0)
    prevalence = (
        selected_positives + 1.0
    ) / (selected_positives + selected_negatives + 2.0)

    selected_count = int(supported.size)
    objective_scale = float(protein_count * selected_count)
    initial = np.ones(2 * selected_count, dtype=np.float64)
    initial[:selected_count] = np.log(prevalence) - np.log1p(-prevalence)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[:selected_count]
        slope = parameters[selected_count:]
        loss = 0.0
        intercept_gradient = np.zeros(selected_count, dtype=np.float64)
        slope_gradient = np.zeros(selected_count, dtype=np.float64)
        for start in range(0, protein_count, policy.protein_chunk_size):
            stop = min(protein_count, start + policy.protein_chunk_size)
            x = _logit_scores(
                scores[start:stop, supported], policy.score_clip_epsilon
            )
            y = truth[start:stop, supported]
            target = np.where(
                y == 1,
                positive_targets[np.newaxis, :],
                negative_targets[np.newaxis, :],
            )
            eta = intercept[np.newaxis, :] + slope[np.newaxis, :] * x
            loss += float(np.sum(np.logaddexp(0.0, eta) - target * eta))
            residual = expit(eta) - target
            intercept_gradient += residual.sum(axis=0)
            slope_gradient += np.sum(residual * x, axis=0)
        return (
            loss / objective_scale,
            np.concatenate((intercept_gradient, slope_gradient)) / objective_scale,
        )

    bounds = [(-30.0, 30.0)] * selected_count + [
        (math.exp(-8.0), math.exp(8.0))
    ] * selected_count
    attempts = []
    fit = None
    parameters = initial
    for attempt_index in range(policy.maximum_restarts + 1):
        fit = minimize(
            objective,
            parameters,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": policy.maximum_iterations,
                "ftol": policy.optimizer_tolerance,
                "gtol": policy.optimizer_tolerance,
                "maxls": 30,
            },
        )
        attempts.append({
            "attempt": attempt_index + 1,
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "iterations": int(fit.nit),
            "function_evaluations": int(fit.nfev),
            "objective": float(fit.fun),
        })
        if fit.success:
            break
        if (
            int(fit.status) != 1
            or not np.isfinite(fit.fun)
            or not np.isfinite(fit.x).all()
        ):
            break
        parameters = np.asarray(fit.x, dtype=np.float64)
    parameters = np.asarray(fit.x, dtype=np.float64)
    intercept_values = parameters[:selected_count]
    slope_values = parameters[selected_count:]

    result = {
        "family": "ffpred_style_independent_platt",
        "status": "complete",
        "formula": "q=1/(1+exp(-(intercept+positive_slope*logit(score))))",
        "historical_correspondence": (
            "FFPred converted each GO-term SVM decision value with independently "
            "fitted sigmoid A/B parameters; PFP sigmoid outputs are converted back "
            "to logits before the analogous per-term fit"
        ),
        "term_ids": list(term_ids),
        "intercepts": {
            term_ids[index]: float(value)
            for index, value in zip(supported, intercept_values)
        },
        "positive_slopes": {
            term_ids[index]: float(value)
            for index, value in zip(supported, slope_values)
        },
        "status_by_term": status_by_term.tolist(),
        "support": {
            "proteins": protein_count,
            "term_positives": positives.tolist(),
            "term_negatives": negatives.tolist(),
            "supported_terms": [term_ids[index] for index in supported],
            "platt_target_smoothing": "Lin_Lin_Weng_2007",
        },
        "policy": policy.as_dict(),
        "optimizer": {
            "name": "scipy_L-BFGS-B",
            "parameterization": "bounded_positive_slope",
            "objective_scaling": "mean_over_supported_protein_term_events",
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "iterations": int(fit.nit),
            "function_evaluations": int(fit.nfev),
            "objective": float(fit.fun),
            "attempts": attempts,
        },
    }
    return result


def apply_ffpred_platt_calibrator(
    scores: np.ndarray,
    term_ids: Sequence[str],
    model: Mapping[str, Any],
    *,
    protein_chunk_size: int = 256,
) -> tuple[np.ndarray, list[str]]:
    statuses = list(model["status_by_term"])
    calibrated = np.full(scores.shape, np.nan, dtype=np.float32)
    supported = np.asarray(
        [index for index, status in enumerate(statuses) if status == "term_platt"],
        dtype=np.int64,
    )
    if not supported.size:
        return calibrated, statuses

    intercept = np.asarray(
        [float(model["intercepts"][term_ids[index]]) for index in supported],
        dtype=np.float64,
    )
    slope = np.asarray(
        [float(model["positive_slopes"][term_ids[index]]) for index in supported],
        dtype=np.float64,
    )
    epsilon = float(model["policy"]["score_clip_epsilon"])
    for start in range(0, scores.shape[0], protein_chunk_size):
        stop = min(scores.shape[0], start + protein_chunk_size)
        x = _logit_scores(scores[start:stop, supported], epsilon)
        calibrated[start:stop, supported] = expit(
            intercept[np.newaxis, :] + slope[np.newaxis, :] * x
        ).astype(np.float32)
    return calibrated, statuses


def ffpred_term_reliability(
    probabilities: np.ndarray,
    truth: np.ndarray,
    term_ids: Sequence[str],
    statuses: Sequence[str],
) -> list[dict[str, Any]]:
    """Apply FFPred's published H/L classifier reliability conditions."""
    rows = []
    for index, (term_id, status) in enumerate(zip(term_ids, statuses)):
        if status != "term_platt":
            rows.append(
                {
                    "go_term": term_id,
                    "calibration_status": status,
                    "reliability_level": "unavailable",
                    "mcc": None,
                    "sensitivity": None,
                    "specificity": None,
                    "precision": None,
                    "positives": int(truth[:, index].sum()),
                    "negatives": int(truth.shape[0] - truth[:, index].sum()),
                }
            )
            continue
        observed = truth[:, index].astype(bool)
        predicted = probabilities[:, index] >= 0.5
        tp = int(np.sum(predicted & observed))
        tn = int(np.sum(~predicted & ~observed))
        fp = int(np.sum(predicted & ~observed))
        fn = int(np.sum(~predicted & observed))
        sensitivity = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        precision = tp / (tp + fp) if tp + fp else None
        denominator = math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else None
        high = (
            mcc is not None
            and sensitivity is not None
            and specificity is not None
            and precision is not None
            and mcc >= 0.3
            and sensitivity >= 0.3
            and specificity >= 0.7
            and precision >= 0.3
        )
        rows.append(
            {
                "go_term": term_id,
                "calibration_status": status,
                "reliability_level": "high" if high else "low",
                "mcc": mcc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "positives": tp + fn,
                "negatives": tn + fp,
            }
        )
    return rows


def calibration_metrics(
    probabilities: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1.0 - 1e-12)
    target = truth.astype(np.float64, copy=False)
    brier = float(np.mean((clipped - target) ** 2))
    log_loss = float(
        -np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))
    )
    return {
        "brier": brier,
        "binary_log_loss": log_loss,
        "mean_probability": float(clipped.mean()),
        "observed_prevalence": float(target.mean()),
    }


def calibration_intercept_slope(
    probabilities: np.ndarray,
    truth: np.ndarray,
    *,
    protein_chunk_size: int = 256,
) -> dict[str, float | None]:
    positives = int(truth.sum())
    events = truth.size
    if not positives or positives == events:
        return {"calibration_intercept": None, "calibration_slope": None}

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.0
        gradient = np.zeros(2, dtype=np.float64)
        for start in range(0, probabilities.shape[0], protein_chunk_size):
            stop = min(probabilities.shape[0], start + protein_chunk_size)
            x = _logit_scores(probabilities[start:stop], 1e-12)
            y = truth[start:stop]
            eta = parameters[0] + parameters[1] * x
            loss += float(np.sum(np.logaddexp(0.0, eta) - y * eta))
            residual = expit(eta) - y
            gradient[0] += float(residual.sum())
            gradient[1] += float(np.sum(residual * x))
        return loss, gradient

    result = minimize(
        objective,
        np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-30.0, 30.0), (-20.0, 20.0)],
        options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-8},
    )
    if not result.success:
        return {"calibration_intercept": None, "calibration_slope": None}
    return {
        "calibration_intercept": float(result.x[0]),
        "calibration_slope": float(result.x[1]),
    }


def reliability_rows(
    probabilities: np.ndarray,
    truth: np.ndarray,
    bin_count: int,
) -> list[dict[str, Any]]:
    flattened = probabilities.astype(np.float64, copy=False).reshape(-1)
    target = truth.reshape(-1)
    order = np.argsort(flattened, kind="stable")
    groups = np.array_split(order, bin_count)
    rows = []
    for index, group in enumerate(groups, start=1):
        if not len(group):
            rows.append(
                {
                    "reliability_bin": index,
                    "events": 0,
                    "mean_probability": None,
                    "observed_fraction": None,
                    "minimum_probability": None,
                    "maximum_probability": None,
                }
            )
            continue
        values = flattened[group]
        labels = target[group]
        rows.append(
            {
                "reliability_bin": index,
                "events": int(len(group)),
                "mean_probability": float(values.mean()),
                "observed_fraction": float(labels.mean()),
                "minimum_probability": float(values.min()),
                "maximum_probability": float(values.max()),
            }
        )
    return rows


def expected_calibration_error(rows: Sequence[Mapping[str, Any]]) -> float:
    total = sum(int(row["events"]) for row in rows)
    if not total:
        return float("nan")
    return float(
        sum(
            int(row["events"])
            * abs(float(row["mean_probability"]) - float(row["observed_fraction"]))
            for row in rows
            if row["events"]
        )
        / total
    )
