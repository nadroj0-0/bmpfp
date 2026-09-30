#!/usr/bin/env python3
"""Evaluate PFP checkpoints and save prediction arrays and metrics."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prediction_artifacts import EvaluationArrayCapture


ASPECTS = ("BPO", "CCO", "MFO")
MODALITY_MODES = (
    "full",
    "sequence-only",
    "sequence-text",
    "sequence-structure",
    "sequence-ppi",
)
ACTIVE_MODALITIES = {
    "full": ("sequence", "text", "structure", "ppi"),
    "sequence-only": ("sequence",),
    "sequence-text": ("sequence", "text"),
    "sequence-structure": ("sequence", "structure"),
    "sequence-ppi": ("sequence", "ppi"),
}
MODEL_SETTINGS = {
    "seq_model": "prott5",
    "fusion_type": "gated_bilinear",
    "hidden_dim": 512,
    "dropout": 0.4,
    "modality_dropout": 0.1,
    "use_late_fusion": True,
    "late_output_mode": "hybrid",
}


def strict_cafa_runner(
    obo_file: str,
    pred_file: str,
    truth_file: str,
    output_dir: str,
    ia_file: str | None = None,
) -> Path:
    """Run cafaeval with the configured CAFA evaluation settings."""

    from cafaeval.evaluation import cafa_eval, write_results

    kwargs: Dict[str, Any] = {"norm": "cafa", "prop": "max", "no_orphans": False}
    if ia_file:
        kwargs["ia"] = str(ia_file)
    results = cafa_eval(str(obo_file), str(pred_file), str(truth_file), **kwargs)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_results(*results, out_dir=str(destination))
    return destination


def load_state(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_aspect_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        aspect, raw_path = value.split("=", 1)
        if aspect not in ASPECTS:
            raise ValueError(f"Unknown aspect: {aspect}")
        result[aspect] = Path(raw_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfp-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-embeddings", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path)
    parser.add_argument("--structure-embeddings", type=Path)
    parser.add_argument("--ppi-embeddings", type=Path)
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="ASPECT=PATH"
    )
    parser.add_argument("--obo-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODALITY_MODES, required=True)
    parser.add_argument("--evaluation-split", choices=("test", "valid"), default="test")
    parser.add_argument("--ia-dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoints = parse_aspect_paths(args.checkpoint)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.pfp_root))
    from mmfp.dataset import MultiModalDataset, collate_fn  # noqa: E402
    import mmfp.evaluation as pfp_evaluation  # noqa: E402
    from mmfp.models import create_model  # noqa: E402
    from train import evaluate, set_seed  # noqa: E402

    pfp_evaluation.run_cafa_evaluator = strict_cafa_runner
    evaluate_with_cafa = pfp_evaluation.evaluate_with_cafa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "mode": args.mode,
        "evaluation_split": args.evaluation_split,
        "selected_aspects": list(checkpoints),
        "aspects": {},
    }

    supplied = {
        "sequence": args.sequence_embeddings,
        "text": args.text_embeddings,
        "structure": args.structure_embeddings,
        "ppi": args.ppi_embeddings,
    }
    with tempfile.TemporaryDirectory(prefix="pfp-empty-modality-") as empty:
        active = set(ACTIVE_MODALITIES[args.mode])
        selected = {
            name: supplied[name] if name in active and supplied[name] is not None else Path(empty)
            for name in supplied
        }
        embedding_dirs = {
            "prott5": str(selected["sequence"]),
            "text": str(selected["text"]),
            "struct": str(selected["structure"]),
            "ppi": str(selected["ppi"]),
        }

        for aspect, checkpoint in checkpoints.items():
            output = args.output_dir / aspect
            output.mkdir(parents=True, exist_ok=True)
            go_terms = json.loads(
                (args.data_dir / f"{aspect}_go_terms.json").read_text(encoding="utf-8")
            )
            train_dataset = MultiModalDataset(
                str(args.data_dir),
                embedding_dirs,
                MODEL_SETTINGS["seq_model"],
                aspect,
                "train",
                normalize="standard",
            )
            evaluation_dataset = MultiModalDataset(
                str(args.data_dir),
                embedding_dirs,
                MODEL_SETTINGS["seq_model"],
                aspect,
                args.evaluation_split,
                normalize="standard",
                norm_stats=train_dataset.norm_stats,
            )
            loader = DataLoader(
                evaluation_dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
                pin_memory=(device == "cuda"),
            )
            model = create_model(
                fusion_type=MODEL_SETTINGS["fusion_type"],
                seq_dim=1024,
                text_dim=768,
                struct_dim=512,
                ppi_dim=512,
                hidden_dim=MODEL_SETTINGS["hidden_dim"],
                num_go_terms=len(go_terms),
                dropout=MODEL_SETTINGS["dropout"],
                modality_dropout=MODEL_SETTINGS["modality_dropout"],
                use_late_fusion=MODEL_SETTINGS["use_late_fusion"],
                late_output_mode=MODEL_SETTINGS["late_output_mode"],
            ).to(device)
            model.load_state_dict(load_state(checkpoint, device), strict=True)
            ordinary = evaluate(
                model,
                loader,
                nn.BCEWithLogitsLoss().to(device),
                device,
                compute_weight_stats_detailed=True,
            )

            selected_ia = args.ia_dir / f"{aspect}_ia.txt" if args.ia_dir else None
            capture = EvaluationArrayCapture(pfp_evaluation)
            with capture:
                cafa = evaluate_with_cafa(
                    model=model,
                    loader=loader,
                    device=device,
                    protein_ids=evaluation_dataset.protein_ids.tolist(),
                    go_terms=go_terms,
                    obo_file=str(args.obo_file),
                    output_dir=str(output / "cafa_eval"),
                    model_name="prott5_gated_bilinear",
                    train_labels=train_dataset.labels,
                    ia_file=str(selected_ia) if selected_ia is not None else None,
                )
            capture.persist(output, selected_ia)

            result: Dict[str, Any] = {
                "aspect": aspect,
                "mode": args.mode,
                "evaluation_split": args.evaluation_split,
                "prediction_file": "predictions.npz",
                "num_go_terms": len(go_terms),
                f"{args.evaluation_split}_fmax": float(ordinary["fmax"]),
                f"{args.evaluation_split}_micro_auprc": float(ordinary["micro_auprc"]),
                f"{args.evaluation_split}_macro_auprc": float(ordinary["macro_auprc"]),
            }
            if (output / "ia.txt").exists():
                result["ia_file"] = "ia.txt"
            for key, value in cafa.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    result[f"cafa_{key}"] = float(value)
            (output / "results.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary["aspects"][aspect] = result

    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
