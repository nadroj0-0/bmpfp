# Embedding generation

These scripts generate the four per-protein embedding modalities consumed by
upstream PFP. Preparation produces `proteins.fasta` and the
`{BPO,CCO,MFO}_{train,valid,test}_names.npy` files used to select proteins.

| Modality | Model | Required source | Values per protein |
|---|---|---|---:|
| Sequence | ProtT5 | prepared `proteins.fasta` and local model directory | 1024 |
| Text | PubMedBERT | prepared names, current descriptions and optional historical descriptions | 768 |
| Structure | ESM-IF1 | prepared names, PDB directory and local IF1 weights | 512 |
| PPI | STRING | prepared names, STRING aliases and embedding HDF5 | 512 |

## Generate embeddings

```bash
bash scripts/embeddings/generate_embeddings_sequence.sh \
  --extractor /external/PFP/scripts/extract_prott5_embeddings.py \
  --fasta-file DATA/proteins.fasta --model-dir MODELS/prott5 \
  --output-dir EMBEDDINGS/prott5 --device cuda:0

bash scripts/embeddings/generate_embeddings_text_temporal_cls.sh \
  --prepared-data-dir DATA --current-descriptions CURRENT.tsv \
  --historical-descriptions HISTORICAL.tsv \
  --embed-script /external/PFP/scripts/embed_uniprot_descriptions.py \
  --model-path MODELS/pubmedbert --output-dir EMBEDDINGS/text

bash scripts/embeddings/generate_embeddings_structure.sh \
  --extractor /external/PFP/scripts/extract_esm_if1_embeddings.py \
  --names-file DATA/BPO_train_names.npy --pdb-dir PDBS \
  --weights MODELS/esm_if1.pt --output-dir EMBEDDINGS/structure --device cuda:0

bash scripts/embeddings/generate_embeddings_ppi.sh \
  --extractor /external/PFP/scripts/extract_ppi_embeddings.py \
  --names-file DATA/BPO_train_names.npy --string-h5 STRING/embeddings.h5 \
  --string-alias STRING/protein.aliases.txt.gz --output-dir EMBEDDINGS/ppi
```

Repeat `--names-file` for every required aspect and split in the structure and
PPI commands. Structure inputs are named `PDBS/<protein_id>.pdb`; ESM-IF1
weights must be available locally. STRING aliases may be plain text or gzip.
For CAFA3 target IDs, PPI generation also accepts `--cafa-assessment-dir` or a
`--cafa3-id-mapping` JSON or headerless two-column TSV.

Text inputs are headerless UTF-8 tables:

```text
protein_id<TAB>description
```

Omit `--historical-descriptions` to use current text unchanged for every protein.
When it is supplied, test proteins use historical text and all others use current
text. Use cutoff `2016-02-17` for CAFA3 and `2025-03-08` for contemporary runs;
for each accession, select the highest UniSave `entryVersion` whose
`firstReleaseDate` is on or before the cutoff. This command consumes that
preselected TSV and does not query UniSave. Homology, clique and Random-H use
current-only text. The script writes the PubMedBERT CLS vector `array[0, 0, :]`.

## Output layout

```text
EMBEDDINGS/
  prott5/<protein_id>.npy
  text/<protein_id>.npy
  structure/<protein_id>.npy
  ppi/<protein_id>.npy
```

Pass these directories to the [model commands](../model/README.md).
