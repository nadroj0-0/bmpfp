#!/usr/bin/env python3
"""Generate confidence views from PFP evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from calibrate_pfp_predictions import (
    _savez,
    _calibration_metrics_with_slope,
    _hierarchy_summary,
    _selected_aspects,
    _tsv,
)
from calibration_common import (
    CalibrationPolicy,
    FFPredPlattPolicy,
    apply_calibrator,
    apply_ffpred_platt_calibrator,
    ffpred_term_reliability,
    fit_ffpred_platt_calibrator,
    fit_monotone_hierarchical_calibrator,
    propagate_scores_max,
)
from label_space_common import (
    write_json,
    write_text,
    read_obo,
)
from pfp_sensitivity_common import (
    load_aspect_bundle,
    load_evaluation_summary,
)
from specificity_common import (
    SpecificityMeasure,
    assign_specificity_bins,
    read_nonnegative_term_values,
)


METHODS = ("raw-sigmoid", "ffpred-style", "hierarchy-calibrated")


def _methods(requested: Sequence[str]) -> list[str]:
    values = list(requested) or list(METHODS)
    if "all" in values:
        values = list(METHODS)
    return list(dict.fromkeys(values))


def _append_metrics(
    metric_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    *,
    benchmark_id: str,
    validation_mode: str,
    test_mode: str,
    aspect: str,
    split: str,
    method: str,
    probabilities: np.ndarray,
    truth: np.ndarray,
    bin_count: int,
) -> dict[str, Any]:
    metrics, reliability = _calibration_metrics_with_slope(
        probabilities, truth, bin_count
    )
    metric_rows.extend(
        {
            "benchmark_id": benchmark_id,
            "validation_mode": validation_mode,
            "test_mode": test_mode,
            "aspect": aspect,
            "split": split,
            "confidence_method": method,
            "metric_name": name,
            "estimate": value,
        }
        for name, value in metrics.items()
    )
    reliability_rows.extend(
        {
            "benchmark_id": benchmark_id,
            "validation_mode": validation_mode,
            "test_mode": test_mode,
            "aspect": aspect,
            "split": split,
            "confidence_method": method,
            **row,
        }
        for row in reliability
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-evaluation-dir", type=Path, required=True)
    parser.add_argument("--test-evaluation-dir", type=Path, required=True)
    parser.add_argument("--obo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark-label",
        help="Optional human-readable label for these explicitly selected evaluations.",
    )
    parser.add_argument("--aspect", action="append", default=[])
    parser.add_argument(
        "--confidence-method",
        action="append",
        choices=(*METHODS, "all"),
        default=[],
        help="Repeat to select methods; default is all three.",
    )
    parser.add_argument("--positive-ia-bins", type=int, default=4)
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument("--score-clip-epsilon", type=float, default=1e-6)
    parser.add_argument("--bin-l2", type=float, default=5.0)
    parser.add_argument("--term-l2", type=float, default=20.0)
    parser.add_argument("--minimum-bin-positives", type=int, default=20)
    parser.add_argument("--minimum-bin-negatives", type=int, default=20)
    parser.add_argument("--minimum-term-positives", type=int, default=20)
    parser.add_argument("--minimum-term-negatives", type=int, default=20)
    parser.add_argument("--ffpred-minimum-term-positives", type=int, default=1)
    parser.add_argument("--ffpred-minimum-term-negatives", type=int, default=1)
    parser.add_argument("--maximum-iterations", type=int, default=200)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-7)
    parser.add_argument("--protein-chunk-size", type=int, default=256)
    parser.add_argument("--hierarchy-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    methods = _methods(args.confidence_method)
    output_dir = args.output_dir.resolve()
    validation_path = args.validation_evaluation_dir.resolve()
    test_path = args.test_evaluation_dir.resolve()
    validation, validation_root = load_evaluation_summary(validation_path)
    test, test_root = load_evaluation_summary(test_path)
    aspects = _selected_aspects(args.aspect, validation, test)

    obo_path = args.obo.resolve()
    graph = read_obo(obo_path)
    benchmark_id = args.benchmark_label or validation_root.name
    validation_mode = str(validation["mode"])
    test_mode = str(test["mode"])
    hierarchy_policy = CalibrationPolicy(
        score_clip_epsilon=args.score_clip_epsilon,
        bin_l2=args.bin_l2,
        term_l2=args.term_l2,
        minimum_bin_positives=args.minimum_bin_positives,
        minimum_bin_negatives=args.minimum_bin_negatives,
        minimum_term_positives=args.minimum_term_positives,
        minimum_term_negatives=args.minimum_term_negatives,
        maximum_iterations=args.maximum_iterations,
        optimizer_tolerance=args.optimizer_tolerance,
        protein_chunk_size=args.protein_chunk_size,
    )
    ffpred_policy = FFPredPlattPolicy(
        score_clip_epsilon=args.score_clip_epsilon,
        minimum_term_positives=args.ffpred_minimum_term_positives,
        minimum_term_negatives=args.ffpred_minimum_term_negatives,
        maximum_iterations=args.maximum_iterations,
        optimizer_tolerance=args.optimizer_tolerance,
        protein_chunk_size=args.protein_chunk_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = output_dir
    metric_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    hierarchy_rows: list[dict[str, Any]] = []
    ffpred_rows: list[dict[str, Any]] = []
    model_aspects: dict[str, Any] = {}
    prediction_outputs: dict[str, Any] = {}

    try:
        for aspect in aspects:
            print(f"==> Loading {aspect} evaluation outputs", flush=True)
            valid_bundle = load_aspect_bundle(validation, validation_root, aspect)
            test_bundle = load_aspect_bundle(test, test_root, aspect)
            ia_measure = read_nonnegative_term_values(
                valid_bundle["ia_path"],
                valid_bundle["go_terms"],
                measure_name="information_accretion",
                higher_is_more_specific=True,
                zero_bin_label="zero_ia",
            )
            _, assignments = assign_specificity_bins(
                valid_bundle["go_terms"],
                SpecificityMeasure(
                    name="information_accretion",
                    values=ia_measure.values,
                    higher_is_more_specific=True,
                    zero_bin_label="zero_ia",
                    source=ia_measure.source,
                ),
                args.positive_ia_bins,
                bin_prefix="positive_ia_q",
                excluded_indices=(valid_bundle["root_index"],),
                excluded_label="root_excluded",
            )
            nonroot = np.asarray(
                [
                    index
                    for index in range(len(valid_bundle["go_terms"]))
                    if index != valid_bundle["root_index"]
                ],
                dtype=np.int64,
            )
            term_ids = [valid_bundle["go_terms"][index] for index in nonroot]
            term_bins = [assignments[index] for index in nonroot]
            valid_raw = np.asarray(valid_bundle["scores"][:, nonroot], dtype=np.float32)
            test_raw = np.asarray(test_bundle["scores"][:, nonroot], dtype=np.float32)
            valid_truth = np.asarray(valid_bundle["truth"][:, nonroot], dtype=np.uint8)
            test_truth = np.asarray(test_bundle["truth"][:, nonroot], dtype=np.uint8)

            valid_postprop = propagate_scores_max(
                valid_bundle["scores"], valid_bundle["go_terms"], graph, aspect
            )[:, nonroot]
            test_postprop = propagate_scores_max(
                test_bundle["scores"], test_bundle["go_terms"], graph, aspect
            )[:, nonroot]

            arrays: dict[str, np.ndarray] = {
                "truth": test_truth,
                "protein_ids": np.asarray(test_bundle["protein_ids"], dtype=str),
                "go_terms": np.asarray(term_ids, dtype=str),
                "information_accretion": np.asarray(
                    ia_measure.values[nonroot], dtype=np.float64
                ),
                "ia_bins": np.asarray(term_bins, dtype=str),
            }
            aspect_models: dict[str, Any] = {}
            aspect_metrics: dict[str, Any] = {}

            if "raw-sigmoid" in methods:
                print(f"==> Recording {aspect} raw sigmoid scores", flush=True)
                arrays["raw_sigmoid_score"] = test_raw
                aspect_models["raw-sigmoid"] = {
                    "family": "raw_model_sigmoid_score",
                    "fit": "none",
                    "interpretation": (
                        "bounded neural-network output; not calibrated, not a "
                        "probability claim, and not a p-value"
                    ),
                }
                for split, values, target in (
                    ("valid", valid_raw, valid_truth),
                    ("test", test_raw, test_truth),
                ):
                    aspect_metrics[f"{split}:raw-sigmoid"] = _append_metrics(
                        metric_rows,
                        reliability_rows,
                        benchmark_id=benchmark_id,
                        validation_mode=validation_mode,
                        test_mode=test_mode,
                        aspect=aspect,
                        split=split,
                        method="raw-sigmoid",
                        probabilities=values,
                        truth=target,
                        bin_count=args.reliability_bins,
                    )
                    hierarchy_rows.append(
                        {
                            "benchmark_id": benchmark_id,
                            "validation_mode": validation_mode,
                            "test_mode": test_mode,
                            "aspect": aspect,
                            "split": split,
                            "confidence_method": "raw-sigmoid",
                            **_hierarchy_summary(
                                values, term_ids, graph, args.hierarchy_tolerance
                            ),
                        }
                    )

            if "ffpred-style" in methods:
                print(f"==> Fitting {aspect} FFPred-style per-term Platt models", flush=True)
                fitted_ffpred = fit_ffpred_platt_calibrator(
                    valid_raw, valid_truth, term_ids, ffpred_policy
                )
                valid_ffpred, valid_status = apply_ffpred_platt_calibrator(
                    valid_raw,
                    term_ids,
                    fitted_ffpred,
                    protein_chunk_size=args.protein_chunk_size,
                )
                test_ffpred, test_status = apply_ffpred_platt_calibrator(
                    test_raw,
                    term_ids,
                    fitted_ffpred,
                    protein_chunk_size=args.protein_chunk_size,
                )
                arrays["ffpred_style_posterior"] = test_ffpred
                arrays["ffpred_style_status"] = np.asarray(test_status, dtype=str)
                aspect_models["ffpred-style"] = fitted_ffpred
                supported = np.asarray(
                    [i for i, value in enumerate(valid_status) if value == "term_platt"],
                    dtype=np.int64,
                )
                for row in ffpred_term_reliability(
                    valid_ffpred, valid_truth, term_ids, valid_status
                ):
                    ffpred_rows.append(
                        {
                            "benchmark_id": benchmark_id,
                            "validation_mode": validation_mode,
                            "test_mode": test_mode,
                            "aspect": aspect,
                            "assessment_split": "validation_post_selection_in_sample",
                            **row,
                        }
                    )
                if supported.size:
                    supported_terms = [term_ids[index] for index in supported]
                    for split, values, target in (
                        ("valid", valid_ffpred[:, supported], valid_truth[:, supported]),
                        ("test", test_ffpred[:, supported], test_truth[:, supported]),
                    ):
                        aspect_metrics[f"{split}:ffpred-style"] = _append_metrics(
                            metric_rows,
                            reliability_rows,
                            benchmark_id=benchmark_id,
                            validation_mode=validation_mode,
                            test_mode=test_mode,
                            aspect=aspect,
                            split=split,
                            method="ffpred-style",
                            probabilities=values,
                            truth=target,
                            bin_count=args.reliability_bins,
                        )
                        hierarchy_rows.append(
                            {
                                "benchmark_id": benchmark_id,
                                "validation_mode": validation_mode,
                                "test_mode": test_mode,
                                "aspect": aspect,
                                "split": split,
                                "confidence_method": "ffpred-style",
                                **_hierarchy_summary(
                                    values,
                                    supported_terms,
                                    graph,
                                    args.hierarchy_tolerance,
                                ),
                            }
                        )

            if "hierarchy-calibrated" in methods:
                print(f"==> Fitting {aspect} hierarchy-calibrated model", flush=True)
                fitted_hierarchy = fit_monotone_hierarchical_calibrator(
                    valid_postprop,
                    valid_truth,
                    term_ids,
                    term_bins,
                    hierarchy_policy,
                )
                valid_pre, valid_fallback = apply_calibrator(
                    valid_postprop,
                    term_ids,
                    term_bins,
                    fitted_hierarchy,
                    protein_chunk_size=args.protein_chunk_size,
                )
                test_pre, test_fallback = apply_calibrator(
                    test_postprop,
                    term_ids,
                    term_bins,
                    fitted_hierarchy,
                    protein_chunk_size=args.protein_chunk_size,
                )
                available = fitted_hierarchy["family"] != "uncalibrated_insufficient_support"
                if available:
                    valid_projected = propagate_scores_max(
                        valid_pre, term_ids, graph, aspect
                    ).astype(np.float32)
                    test_projected = propagate_scores_max(
                        test_pre, term_ids, graph, aspect
                    ).astype(np.float32)
                else:
                    valid_projected = valid_pre
                    test_projected = test_pre
                arrays["hierarchy_calibrated_probability_preprojection"] = test_pre
                arrays["hierarchy_calibrated_probability"] = test_projected
                arrays["hierarchy_calibrated_fallback"] = np.asarray(
                    test_fallback, dtype=str
                )
                aspect_models["hierarchy-calibrated"] = fitted_hierarchy
                if available:
                    for split, label, values, target in (
                        ("valid", "preprojection", valid_pre, valid_truth),
                        ("test", "preprojection", test_pre, test_truth),
                        ("valid", "projected", valid_projected, valid_truth),
                        ("test", "projected", test_projected, test_truth),
                    ):
                        method = f"hierarchy-calibrated-{label}"
                        aspect_metrics[f"{split}:{method}"] = _append_metrics(
                            metric_rows,
                            reliability_rows,
                            benchmark_id=benchmark_id,
                            validation_mode=validation_mode,
                            test_mode=test_mode,
                            aspect=aspect,
                            split=split,
                            method=method,
                            probabilities=values,
                            truth=target,
                            bin_count=args.reliability_bins,
                        )
                        summary = _hierarchy_summary(
                            values, term_ids, graph, args.hierarchy_tolerance
                        )
                        summary["projection"] = (
                            "upward_max_then_checked" if label == "projected" else "checked_without_projection"
                        )
                        hierarchy_rows.append(
                            {
                                "benchmark_id": benchmark_id,
                                "validation_mode": validation_mode,
                                "test_mode": test_mode,
                                "aspect": aspect,
                                "split": split,
                                "confidence_method": method,
                                **summary,
                            }
                        )

            prediction_path = stage / f"{aspect}_confidence_outputs.npz"
            print(f"==> Writing {aspect} confidence arrays", flush=True)
            _savez(prediction_path, **arrays)
            prediction_outputs[aspect] = {
                "path": prediction_path.name,
                "shape": list(test_truth.shape),
                "arrays": sorted(arrays),
            }
            model_aspects[aspect] = {
                "validation_data": {
                    "proteins": len(valid_bundle["protein_ids"]),
                    "go_terms": len(valid_bundle["go_terms"]),
                },
                "test_data": {
                    "proteins": len(test_bundle["protein_ids"]),
                    "go_terms": len(test_bundle["go_terms"]),
                },
                "methods": aspect_models,
                "metrics": aspect_metrics,
            }

        model = {
            "analysis_label": "post_selection_validation_confidence_profiles",
            "selected_methods": methods,
            "target_definition": "benchmark_observed_qualifying_propagated_t1",
            "shared_interpretation": (
                "no output is a p-value; calibrated outputs estimate membership in "
                "the benchmark-observed qualifying propagated t1 label set rather "
                "than unobserved biological truth"
            ),
            "checkpoint_selection_overlap_disclosure": (
                "the validation population previously influenced checkpoint selection "
                "and early stopping; all fitted profiles are exploratory post-selection "
                "calibration rather than independent-holdout calibration"
            ),
            "method_definitions": {
                "raw-sigmoid": (
                    "unmodified PFP sigmoid model output, reported as a score only"
                ),
                "ffpred-style": (
                    "independent per-GO-term Platt scaling of reconstructed PFP logits; "
                    "FFPred H/L term reliability rules are reported separately"
                ),
                "hierarchy-calibrated": (
                    "positive-slope multilevel calibration with IA-bin and supported-term "
                    "intercepts, followed by is_a+part_of upward-max projection"
                ),
            },
            "benchmark_id": benchmark_id,
            "validation_mode": validation_mode,
            "test_mode": test_mode,
            "selected_aspects": aspects,
            "candidate_universe": "all non-root model-output terms",
            "policy": {
                "p_values": "prohibited",
                "positive_ia_bins": args.positive_ia_bins,
                "reliability_bins": args.reliability_bins,
                "ffpred_style": ffpred_policy.as_dict(),
                "hierarchy_calibrated": hierarchy_policy.as_dict(),
                "hierarchy_projection": "is_a+part_of_upward_max",
            },
            "aspects": model_aspects,
            "prediction_outputs": prediction_outputs,
        }
        write_json(stage / "confidence_model.json", model)
        write_text(
            stage / "confidence_metrics.tsv",
            _tsv(
                metric_rows,
                (
                    "benchmark_id",
                    "validation_mode",
                    "test_mode",
                    "aspect",
                    "split",
                    "confidence_method",
                    "metric_name",
                    "estimate",
                ),
            ),
        )
        write_text(
            stage / "confidence_reliability.tsv",
            _tsv(
                reliability_rows,
                (
                    "benchmark_id",
                    "validation_mode",
                    "test_mode",
                    "aspect",
                    "split",
                    "confidence_method",
                    "reliability_bin",
                    "events",
                    "mean_probability",
                    "observed_fraction",
                    "minimum_probability",
                    "maximum_probability",
                ),
            ),
        )
        write_text(
            stage / "ffpred_term_reliability.tsv",
            _tsv(
                ffpred_rows,
                (
                    "benchmark_id",
                    "validation_mode",
                    "test_mode",
                    "aspect",
                    "assessment_split",
                    "go_term",
                    "calibration_status",
                    "reliability_level",
                    "mcc",
                    "sensitivity",
                    "specificity",
                    "precision",
                    "positives",
                    "negatives",
                ),
            ),
        )
        write_text(
            stage / "confidence_hierarchy_summary.tsv",
            _tsv(
                hierarchy_rows,
                (
                    "benchmark_id",
                    "validation_mode",
                    "test_mode",
                    "aspect",
                    "split",
                    "confidence_method",
                    "edges_checked",
                    "protein_edge_events",
                    "violation_events",
                    "affected_edges",
                    "maximum_child_minus_parent",
                    "tolerance",
                    "projection",
                ),
            ),
        )
        analysis = {
            "analysis_kind": "post_selection_confidence_profiles",
            "scientific_label": "post_selection_validation_confidence_profiles",
            "benchmark_id": benchmark_id,
            "validation_mode": validation_mode,
            "test_mode": test_mode,
            "selected_methods": methods,
            "selected_aspects": aspects,
        }
        write_json(stage / "confidence_analysis.json", analysis)
    except BaseException:
        raise

    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
