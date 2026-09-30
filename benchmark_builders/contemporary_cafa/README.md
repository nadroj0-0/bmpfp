# Temporal and CAFA3 benchmark builder

This package builds PFP-compatible temporal and CAFA3 benchmarks from explicit
scientific inputs.

## Install

```bash
python -m pip install -e benchmark_builders/contemporary_cafa
```

## Routes

| Route | Source mode and profile | Main input |
|---|---|---|
| Contemporary Global-NK or NK+LK | `snapshots`, `global-nk` or `nk-lk` | UniProt, GOA and GO snapshots |
| Historical CAFA3 snapshots | `snapshots`, `cafa3-reconstructed` | Historical UniProt, GOA and GO snapshots |
| Original PFP CAFA3 | `deepgoplus` | Original PFP CAFA3 pickle directory |
| Official CAFA3 files | `cafa3-files` | Training and test sequences and annotations |

## Build commands

```bash
cafa-benchmark-builder --source-mode snapshots --profile global-nk \
  --uniprot-t0 <T0_UNIPROT> --uniprot-t1 <T1_UNIPROT> \
  --goa-t0 <T0_GAF> --goa-t1 <T1_GAF> \
  --go-obo <BENCHMARK_OBO> --go-obo-t0 <T0_OBO> --go-obo-t1 <T1_OBO> \
  --output-dir <OUTPUT>
```

Use `--profile nk-lk` for NK+LK. Repeat either UniProt option to combine
Swiss-Prot and TrEMBL files.
`global-nk` uses global no-knowledge eligibility; `nk-lk` applies eligibility
within each ontology.

```bash
cafa-benchmark-builder --source-mode snapshots --profile cafa3-reconstructed \
  --uniprot-t0 <T0_UNIPROT> --uniprot-t1 <T1_UNIPROT> \
  --goa-t0 <T0_GAF> --goa-t1 <T1_GAF> --go-obo <T0_OBO> \
  --go-obo-t0 <T0_OBO> --go-obo-t1 <T1_OBO> --output-dir <OUTPUT>

cafa-benchmark-builder --source-mode deepgoplus \
  --deepgoplus-dir <PFP_CAFA3_PICKLES> --go-obo <GO_OBO> --output-dir <OUTPUT>

cafa-benchmark-builder --source-mode cafa3-files --go-obo <GO_OBO> \
  --train-sequences-file <TRAIN_FASTA> --train-annotations-file <TRAIN_LABELS> \
  --test-sequences-file <TEST_FASTA> --test-annotations-file <TEST_LABELS> \
  --min-count 50 --split 0.9 --seed 0 --output-dir <OUTPUT>
```

`global-nk` and `nk-lk` use `EXP`, `IDA`, `IPI`, `IMP`, `IGI`, `IEP`, `HTP`,
`HDA`, `HMP`, `HGI`, `HEP` and `IGC`. `cafa3-reconstructed` and
`contemporary-cafa3-style` use `EXP`, `IDA`, `IPI`, `IMP`, `IGI`, `IEP`, `TAS` and `IC`.
Profiles use seed `0`, a `0.9/0.1` training/validation split and minimum term count `50`;
their temporal cutoffs are defined in the selected profile.

## Random-T

```bash
pfp-random-baseline --mode t1-snapshot --source-benchmark-dir <BENCHMARK> \
  --go-obo <BENCHMARK_OBO> --uniprot <T1_UNIPROT> --goa <T1_GAF> \
  --source-go-obo <T1_OBO> --seed 0 --output-dir <OUTPUT>
```

## Outputs

Each route writes `bp-*`, `cc-*` and `mf-*` training, validation and test CSVs.
Unless `--no-intermediates` is set, snapshot routes write and the DeepGOPlus route
copies `train_data.pkl`, `train_data_train.pkl`, `train_data_valid.pkl`, `test_data.pkl`
and `terms.pkl`. The `cafa3-files` route always writes only these three pickles:
`train_data.pkl`, `test_data.pkl` and `terms.pkl`.
Snapshot routes write scientific counts to `reports/benchmark_statistics.json`.
