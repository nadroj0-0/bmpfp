#!/usr/bin/env python3
"""Generate temporal PubMedBERT CLS embeddings from description TSVs.

When historical text is supplied, proteins in any test split use it with
punctuation normalization; otherwise every prepared protein uses current text
unchanged. Raw hidden states are reduced with ``array[0, 0, :]``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np


ASPECTS = ("BPO", "CCO", "MFO")
SPLITS = ("train", "valid", "test")


def clean_historical_description_artifacts(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"(?:\s*\.\s*){2,}", ". ", text)
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def prepared_populations(data_dir: Path) -> tuple[list[str], set[str]]:
    ordered: list[str] = []
    seen: set[str] = set()
    test_ids: set[str] = set()
    for aspect in ASPECTS:
        for split in SPLITS:
            names = np.load(
                data_dir / f"{aspect}_{split}_names.npy", allow_pickle=True
            ).tolist()
            if split == "test":
                test_ids.update(str(value) for value in names)
            for value in names:
                protein_id = str(value)
                if protein_id not in seen:
                    seen.add(protein_id)
                    ordered.append(protein_id)
    return ordered, test_ids


def load_descriptions(path: Path) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            protein_id, description = line.split("\t", 1)
            descriptions[protein_id] = description
    return descriptions


def select_descriptions(
    ordered_ids: list[str],
    test_ids: set[str],
    current: dict[str, str],
    historical: dict[str, str] | None,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for protein_id in ordered_ids:
        if historical is not None and protein_id in test_ids:
            text = clean_historical_description_artifacts(historical[protein_id])
        else:
            text = current[protein_id]
        rows.append((protein_id, text))
    return rows


def load_upstream_embedder(path: Path):
    specification = importlib.util.spec_from_file_location("pfp_text_embedder", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data-dir", type=Path, required=True)
    parser.add_argument("--current-descriptions", type=Path, required=True)
    parser.add_argument("--historical-descriptions", type=Path)
    parser.add_argument("--embed-script", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ordered_ids, test_ids = prepared_populations(args.prepared_data_dir)
    historical = (
        load_descriptions(args.historical_descriptions)
        if args.historical_descriptions is not None
        else None
    )
    selected = select_descriptions(
        ordered_ids,
        test_ids,
        load_descriptions(args.current_descriptions),
        historical,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if selected:
        with tempfile.TemporaryDirectory(prefix="pfp-temporal-text-") as temporary:
            data_dir = Path(temporary) / "data"
            descriptions = (
                data_dir
                / "embedding_cache"
                / "uniprot_text"
                / "protein_descriptions.tsv"
            )
            descriptions.parent.mkdir(parents=True)
            with descriptions.open("w", encoding="utf-8") as handle:
                for protein_id, description in selected:
                    text = description.replace("\r", " ").replace("\n", " ").replace("\t", " ")
                    handle.write(f"{protein_id}\t{text}\n")

            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            embedder = load_upstream_embedder(args.embed_script)
            embedder.precompute_embeddings(
                data_dir, model_name=str(args.model_path)
            )
            generated = data_dir / "embedding_cache" / "exp_text_embeddings"
            for protein_id, _ in selected:
                hidden_state = np.load(
                    generated / f"{protein_id}.npy", allow_pickle=True
                )
                np.save(
                    args.output_dir / f"{protein_id}.npy",
                    np.asarray(hidden_state[0, 0, :], dtype=np.float32),
                )

    print(f"Generated temporal text embeddings in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
