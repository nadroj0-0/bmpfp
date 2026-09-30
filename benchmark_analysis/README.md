# Benchmark analysis

This package describes PFP-compatible benchmarks and compares their populations,
labels and available modalities. Its reports are descriptive: differences between
benchmarks do not by themselves explain differences in model performance.

## Installation

```bash
python -m pip install -e benchmark_analysis
```

## Inputs

- a benchmark directory containing the nine PFP CSV files and its GO OBO file;
- optional pre-projection PFP pickles for root-only source classification;
- optional UniProt DAT or tabular taxonomy sources;
- optional protein-category maps;
- optional long-table modality-availability data.

## Commands

```bash
pfp-benchmark-analysis --config configs/benchmark_analysis.example.json \
  --output-dir /path/to/temporal-analysis
pfp-benchmark-analysis --config configs/benchmark_analysis.homology.example.json \
  --output-dir /path/to/homology-analysis
```

Replace the example paths in each JSON configuration before running the command.

## Outputs

| Result | Files |
|---|---|
| Labels and term support | `file_profiles.tsv`, `label_profiles.tsv`, `term_support.tsv` |
| Root-only labels | `root_only_source.tsv`, `root_only_summary.tsv` |
| Taxonomy | `taxonomy_distribution.tsv`, `taxonomy_coverage.tsv`, `taxonomy_conflicts.tsv` |
| Protein categories | `category_distribution.tsv` |
| Modality availability | `modality_coverage.tsv`, `modality_patterns.tsv` |
| Benchmark membership | `protein_membership.tsv` |
| Cross-benchmark comparisons | `cross_benchmark_metrics.tsv`, `cross_benchmark_modality.tsv`, `cross_benchmark_taxonomy.tsv`, `cross_benchmark_root_source.tsv` |

`benchmark_analysis.md` summarises the complete TSV output. Taxonomy distributions
describe organisms; protein-family claims require an explicit category map.
