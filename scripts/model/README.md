# PFP preparation, training and evaluation

Run the stages in order, or begin from compatible prepared data, embeddings or
checkpoints. The commands use the upstream PFP implementation.

## 1. Prepare benchmark data

`BENCHMARK` contains the nine `bp|cc|mf` by `training|validation|test` CSVs.

```bash
python scripts/model/prepare_pfp_benchmark.py \
  --benchmark-dir BENCHMARK --data-dir DATA --pfp-root /external/PFP
```

`DATA` receives `{BPO,CCO,MFO}_{train,valid,test}_names.npy`,
`*_labels.npz`, `*_sequences.json`, each aspect's `*_go_terms.json`, and
`proteins.fasta`. Generate or supply the four [embedding directories](../embeddings/README.md).

## 2. Train

| Setting | Value |
|---|---|
| Modality dimensions | ProtT5 1024; PubMedBERT 768; ESM-IF1 512; PPI 512 |
| Fusion | gated bilinear; hybrid late fusion |
| Hidden size / dropout | 512 / 0.4 |
| Modality dropout / auxiliary-loss weight | 0.1 / 0.8 |
| Default seed | 42 |
| Normalisation | training-split statistics reused for evaluation |

```bash
bash scripts/model/run_pfp_benchmark.sh \
  --prepared-data-dir DATA --sequence-embedding-dir EMBEDDINGS/prott5 \
  --text-embedding-dir EMBEDDINGS/text --structure-embedding-dir EMBEDDINGS/structure \
  --ppi-embedding-dir EMBEDDINGS/ppi --pfp-root /external/PFP \
  --obo-file go-basic.obo --output-dir TRAINING --mode full --seed 42
```

Modes are `full`, `sequence-only`, `sequence-text`, `sequence-structure` and
`sequence-ppi`. Repeat `--aspect BPO|CCO|MFO` to select aspects; all three run
by default. Each aspect writes
`TRAINING/fusion_comparison/prott5/<ASPECT>/gated_bilinear/best_model.pt` and
`results.json`.

## 3. Evaluate checkpoints

```bash
python scripts/model/evaluate_pfp_checkpoints.py \
  --pfp-root /external/PFP --data-dir DATA --sequence-embeddings EMBEDDINGS/prott5 \
  --text-embeddings EMBEDDINGS/text --structure-embeddings EMBEDDINGS/structure \
  --ppi-embeddings EMBEDDINGS/ppi \
  --checkpoint BPO=TRAINING/fusion_comparison/prott5/BPO/gated_bilinear/best_model.pt \
  --obo-file go-basic.obo --output-dir EVALUATION --mode full \
  --evaluation-split test --seed 42
```

Repeat `--checkpoint ASPECT=PATH` for each aspect; checkpoints load strictly. Use
`--evaluation-split valid` for calibration inputs and `--ia-dir DIR` for
precomputed `<ASPECT>_ia.txt`. Cafaeval uses `norm=cafa`, `prop=max` and `no_orphans=false`.
`cafa_smin` is the unweighted count-space distance; `cafa_wsmin`, when present,
is the IA-weighted semantic S-min.

## Outputs

```text
EVALUATION/evaluation_summary.json
EVALUATION/<ASPECT>/results.json
EVALUATION/<ASPECT>/predictions.npz
EVALUATION/<ASPECT>/cafa_eval/
EVALUATION/<ASPECT>/ia.txt        # when supplied or produced
```

`predictions.npz` stores `scores`, `truth`, ordered `protein_ids` and ordered `go_terms`; see the [analysis scripts](../analysis/README.md) for downstream use.
