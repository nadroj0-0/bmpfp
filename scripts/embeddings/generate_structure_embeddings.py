#!/usr/bin/env python3
"""Generate ESM-IF1 embeddings from prepared proteins, PDBs, and local weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np


def requested_ids(paths: list[Path]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for value in np.load(path, allow_pickle=True).tolist():
            protein_id = str(value)
            if protein_id not in seen:
                seen.add(protein_id)
                ordered.append(protein_id)
    return ordered


def load_upstream_extractor(path: Path):
    specification = importlib.util.spec_from_file_location("pfp_if1_extractor", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def device_safe_get_encoder_output(model, alphabet, coords):
    util = importlib.import_module("esm.inverse_folding.util")
    device = next(model.parameters()).device
    encoder_coords, confidence, _, _, padding_mask = util.CoordBatchConverter(
        alphabet
    )([(coords, None, None)], device=device)
    encoder_out = model.encoder.forward(
        encoder_coords,
        padding_mask,
        confidence,
        return_all_hiddens=False,
    )
    return encoder_out["encoder_out"][0][1:-1, 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--names-file", type=Path, action="append", required=True)
    parser.add_argument("--pdb-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pooling", choices=("mean", "max", "cls"), default="mean")
    parser.add_argument("--chain")
    args = parser.parse_args()

    proteins = requested_ids(args.names_file)

    esm = importlib.import_module("esm")
    util = importlib.import_module("esm.inverse_folding.util")
    extractor = load_upstream_extractor(args.extractor)
    local_loader = esm.pretrained.load_model_and_alphabet_local
    original_model_loader = esm.pretrained.esm_if1_gvp4_t16_142M_UR50
    original_get_encoder_output = util.get_encoder_output

    with tempfile.TemporaryDirectory(prefix="pfp-if1-selected-") as temporary:
        selected = Path(temporary)
        for protein_id in proteins:
            source = args.pdb_dir / f"{protein_id}.pdb"
            (selected / source.name).symlink_to(source.resolve())

        util.get_encoder_output = device_safe_get_encoder_output
        esm.pretrained.esm_if1_gvp4_t16_142M_UR50 = (
            lambda: local_loader(str(args.weights.resolve()))
        )
        try:
            extractor.extract_embeddings(
                str(selected),
                str(args.output_dir),
                pooling=args.pooling,
                device=args.device,
                chain_id=args.chain,
            )
        finally:
            util.get_encoder_output = original_get_encoder_output
            esm.pretrained.esm_if1_gvp4_t16_142M_UR50 = original_model_loader

    print(f"Generated ESM-IF1 embeddings in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
