# Protein Function Prediction Experiments

This repository builds protein-function benchmarks, generates multimodal embeddings, trains or evaluates PFP models, and analyses their results. Each stage takes explicit scientific inputs and writes outputs that can be passed directly to the next stage.

## Pipeline

`raw snapshots -> benchmark CSVs -> prepared PFP data -> embeddings -> training/evaluation -> analysis`

## Components

| Component | Purpose |
|---|---|
| [Contemporary and CAFA3 builder](benchmark_builders/contemporary_cafa/README.md) | Temporal Global-NK, NK+LK, Random-T and CAFA3 benchmarks |
| [Homology and clique builder](benchmark_builders/homology_cluster/README.md) | MMseqs2 thresholds, Random-H and supplied clique assignments |
| [Benchmark analysis](benchmark_analysis/README.md) | Label, taxonomy, modality and cross-benchmark descriptions |
| [Embeddings](scripts/embeddings/README.md) | ProtT5, PubMedBERT, ESM-IF1 and STRING representations |
| [Model](scripts/model/README.md) | PFP preparation, training, checkpoint evaluation and predictions |
| [Result analyses](scripts/analysis/README.md) | Homology, knowledge-state, sensitivity, calibration and specificity analyses |

Required database snapshots, model weights and generated arrays are stored outside Git. The supported environment is Linux with Python, PyTorch and CUDA appropriate to the selected workload; fresh homology clustering also requires MMseqs2.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e benchmark_builders/contemporary_cafa
python -m pip install -e benchmark_builders/homology_cluster
python -m pip install -e benchmark_analysis
```

PFP preparation, training and several embedding commands also require an [upstream PFP](https://github.com/psipred/PFP) checkout. See [data sources](data_sources/README.md) for the other external inputs.

## Main commands

```bash
# Contemporary or CAFA3 benchmark
cafa-benchmark-builder --source-mode snapshots --profile global-nk \
  --go-obo <GO_OBO> --output-dir <BENCHMARK_DIR> <SNAPSHOT_ARGUMENTS>

# Homology or clique benchmark; 30% identity is the MMseqs2-30 condition
homology-cluster-benchmark --go-obo <GO_OBO> --goa <GOA> \
  --identity 30 --coverage 0.8 --output-dir <BENCHMARK_DIR> <SOURCE_ARGUMENTS>

# Descriptive benchmark analysis
pfp-benchmark-analysis --config configs/benchmark_analysis.example.json \
  --output-dir <ANALYSIS_DIR>

# Prepare the nine CSVs for PFP
python scripts/model/prepare_pfp_benchmark.py --benchmark-dir <BENCHMARK_DIR> \
  --data-dir <PREPARED_DIR> --pfp-root <PFP_ROOT>

# Generate one embedding modality
bash scripts/embeddings/generate_embeddings_sequence.sh \
  --extractor <PFP_ROOT>/scripts/extract_prott5_embeddings.py \
  --fasta-file <PREPARED_DIR>/proteins.fasta --model-dir <PROTT5_MODEL> \
  --output-dir <SEQUENCE_EMBEDDINGS>

# Train PFP
bash scripts/model/run_pfp_benchmark.sh --prepared-data-dir <PREPARED_DIR> \
  --sequence-embedding-dir <SEQUENCE_DIR> --text-embedding-dir <TEXT_DIR> \
  --structure-embedding-dir <STRUCTURE_DIR> --ppi-embedding-dir <PPI_DIR> \
  --pfp-root <PFP_ROOT> --obo-file <GO_OBO> --output-dir <RUN_DIR>

# Analyse a benchmark label space
python scripts/analysis/analyse_pfp_label_space.py --benchmark-id <ID> \
  --benchmark-dir <BENCHMARK_DIR> --obo-file <GO_OBO> --output-dir <ANALYSIS_DIR>
```

Checkpoint-only evaluation and result-analysis commands are documented in the [model](scripts/model/README.md) and [analysis](scripts/analysis/README.md) guides.

## Outputs

- nine benchmark CSVs: BP, CC and MF training, validation and test files;
- prepared PFP arrays and `proteins.fasta`;
- per-protein NumPy arrays for generated embeddings;
- trained checkpoints and metric files;
- `evaluation_summary.json` plus per-aspect `results.json`, `predictions.npz`, `cafa_eval/` and optional `ia.txt`;
- benchmark and result-analysis TSV, JSON, Markdown and plot files.

Release references are listed in [current data releases](data_sources/current_releases.md).
