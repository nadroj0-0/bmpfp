# Result analyses

These scripts analyse benchmark CSVs, homology partitions, PFP evaluation directories
and saved prediction bundles. Each command writes its tables and summary to the
specified output directory.

## Scripts

| Script | Scientific purpose | Primary input | Primary output |
|---|---|---|---|
| `analyse_cafa3_knowledge_states.py` | Classify original PFP CAFA3 test proteins by official knowledge state | PFP CSVs and official CAFA3 archive | `cafa3_knowledge_states.json` |
| `analyse_contemporary_knowledge_cohorts.py` | Reconstruct contemporary temporal knowledge cohorts | Benchmark, UniProt, GOA and OBO snapshots | `contemporary_knowledge_cohorts.tsv` |
| `build_temporal_annotation_states.py` | Build protein cohorts and term transitions | Direct annotations, closures, presence and scope tables | `temporal_annotation_states.json` |
| `analyse_nk_lk.py` | Compare NK+LK cohorts with Global-NK | NK+LK evaluation, cohort membership and Global-NK summary | `analysis_summary.json` |
| `analyse_pfp_label_space.py` | Describe roots, label depths and term support | Nine benchmark CSVs and GO OBO | `label_space_analysis.json` |
| `analyse_pfp_label_sensitivity.py` | Measure sensitivity to root-only targets | Evaluation directory and GO OBO | `root_exclusion_sensitivity.json` |
| `compare_pfp_label_sensitivity.py` | Compare label-sensitivity results | Sensitivity result JSON files | `label_sensitivity_comparison.json` |
| `compare_pfp_modalities.py` | Compare modality evaluation summaries with sequence-only | Evaluation directories | `modality_comparison.json` |
| `analyse_pfp_information_content.py` | Analyse IA and Xu specificity across prediction terms | Evaluation directory and GO OBO | `specificity_analysis.json` |
| `compare_pfp_specificity.py` | Compare completed specificity analyses | Specificity result directories | `comparison.json` |
| `analyse_pfp_specificity_support.py` | Relate specificity effects to training support | Evaluation, prepared data and specificity results | `specificity_support_summary.json` |
| `calibrate_pfp_predictions.py` | Fit calibration on validation predictions and assess test transport | Validation/test evaluations and GO OBO | `calibration_analysis.json` |
| `generate_pfp_confidence.py` | Generate raw, FFPred-style and hierarchy-calibrated confidence views | Validation/test evaluations and GO OBO | `confidence_analysis.json` |
| `compare_homology_thresholds.py` | Compare supplied MMseqs2 threshold benchmarks | Ordered homology benchmark directories | `comparison.json` |
| `compare_consecutive_homology_partitions.py` | Measure changes between consecutive homology partitions | Ordered homology benchmark directories | `comparison.json` |
| `compare_homology_cluster_assignments.py` | Compare two cluster or clique assignments | Two member-assignment files | `comparison.json` |
| `analyse_homology_plateau_mechanism.py` | Analyse cluster sparsity and Random-H supervised exposure | Homology benchmarks and evaluation directories | `homology_plateau_mechanism.json` |
| `compare_go_relationship_policy.py` | Compare CSVs built under two GO relationship policies | Two benchmark directories and GO OBO | `comparison.json` |
| `analyse_taxonomy_and_aspects.py` | Describe taxonomy and GO-aspect membership | Taxonomy dump and benchmark outputs | `taxonomy_and_aspect_analysis.json` |

For `analyse_contemporary_knowledge_cohorts.py`, reproduce the reported Global-NK
and NK+LK cohorts by repeating `--target-taxon` for every taxon in
`benchmark_builders/contemporary_cafa/src/cafa_benchmark_builder/resources/cafa3_target_taxa.txt`
and `--evidence-code` for `EXP`, `IDA`, `IPI`, `IMP`, `IGI`, `IEP`, `HTP`, `HDA`,
`HMP`, `HGI`, `HEP` and `IGC`.

## Examples

Benchmark-level label analysis:

```bash
python scripts/analysis/analyse_pfp_label_space.py --benchmark-id global-nk \
  --benchmark-dir /path/to/benchmark --obo-file /path/to/go.obo \
  --output-dir /path/to/label-space-analysis
```

Prediction-level specificity analysis:

```bash
python scripts/analysis/analyse_pfp_information_content.py \
  --evaluation-dir /path/to/evaluation --benchmark-id global-nk \
  --specificity-measure all_separate --obo /path/to/go.obo \
  --output-dir /path/to/specificity-analysis
```

Homology-threshold comparison:

```bash
python scripts/analysis/compare_homology_thresholds.py \
  --benchmark 30=/path/to/mmseqs2-30 --benchmark 25=/path/to/mmseqs2-25 \
  --output-dir /path/to/threshold-comparison
```

## Interpretation

Plateau and exposure results are observational associations, not causal evidence.
IA and Xu specificity are distinct measures; raw Xu `T` decreases as specificity
increases, while `-log2(T)` is a display transform. Bin-specific thresholds and
quartiles are local to each analysed benchmark and should not be treated as shared
term sets across runs.
