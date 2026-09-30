# Homology and clique benchmark builder

`homology-cluster-benchmark` builds PFP benchmarks from explicit ontology,
annotation, sequence and family inputs.

## Fixed settings

| Setting | Value |
|---|---|
| Scaffold | UniRef50 |
| Identity thresholds | `30`, `25`, `20`, `15`, `10`, `5` percent |
| MMseqs2 | sensitivity `4`, coverage `0.8`, modes `0/0/3`, reassignment off; no explicit E-value or `createdb` shuffle |
| Splits | development/test `0.8/0.2`; training/validation `0.9/0.1` |
| Other | seed `0`; minimum development term count `50` |

The 30% identity-threshold condition is `MMseqs2-30`. The annotation set is
`EXP`, `IDA`, `IPI`, `IMP`, `IGI`, `IEP`, `HTP`, `HDA`, `HMP`, `HGI`, `HEP`
and `IGC`, with GO relationship propagation.

## Inputs

- MMseqs2 builds: GO OBO, GOA, UniRef50 FASTA, UniProt ID mapping and MMseqs2.
- Clique builds: GO OBO, GOA, UniProt DAT files and a membership CSV.

Install with `python -m pip install -e benchmark_builders/homology_cluster`.

## MMseqs2 thresholds

```bash
homology-cluster-benchmark --go-obo <GO_OBO> --goa <GOA> \
  --uniref-fasta <UNIREF50_FASTA> --uniref-level 50 --idmapping <IDMAPPING> \
  --uniprot-sprot <SPROT_DAT> --uniprot-trembl <TREMBL_DAT> \
  --identity 30 --coverage 0.8 --sensitivity 4 \
  --cov-mode 0 --cluster-mode 0 --alignment-mode 3 --cluster-reassign 0 \
  --split-policy cluster-count-random --development-fraction 0.8 \
  --training-fraction 0.9 --seed 0 --min-count 50 \
  --mmseqs-bin <MMSEQS> --threads <N> --output-dir <OUTPUT>
```

Run once per identity threshold with a separate output directory.

## Random-H

```bash
pfp-random-baseline --mode prepared \
  --source-benchmark-dir <MMSEQS2_30_OUTPUT> --go-obo <GO_OBO> \
  --seed 0 --output-dir <RANDOM_H_OUTPUT>
```

## Clique 80/30 and 30/30

The membership input is a headerless CSV or CSV.GZ with three columns:
`clique_id,representative_id,member_id`. Members may use UniProt or UniParc
identifiers; exact duplicate rows collapse to one membership.

```bash
homology-cluster-benchmark --go-obo <GO_OBO> --goa <GOA> \
  --uniprot-sprot <SPROT_DAT> --uniprot-trembl <TREMBL_DAT> \
  --clique-memberships <CLIQUE_CSV_GZ> --identity 30 --coverage 0.8 \
  --split-policy cluster-count-random --development-fraction 0.8 \
  --training-fraction 0.9 --seed 0 --min-count 50 --output-dir <OUTPUT>
```

Use coverage `0.8` for clique 80/30 and `0.3` for clique 30/30.

## Outputs

Each build writes nine `bp-*`, `cc-*` and `mf-*` CSVs; five pickles
(`train_data.pkl`, `train_data_train.pkl`, `train_data_valid.pkl`,
`test_data.pkl`, `terms.pkl`); `benchmark_summary.json`; and assignment tables
used by the homology analyses.

See the [analysis commands](../../scripts/analysis/README.md) for threshold,
partition, exposure and clique comparisons.
