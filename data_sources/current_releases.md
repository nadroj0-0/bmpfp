# Contemporary Benchmark Source Releases

The contemporary benchmark uses an earlier boundary for prediction eligibility and later snapshots for evaluation.

| Input family | Release or date | Source | Temporal role |
|---|---|---|---|
| UniProtKB | `2025_01` | [UniProt release archive](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/) | Earlier protein snapshot |
| UniProtKB | `2026_02`, 10 June 2026 | [UniProt current release](https://ftp.uniprot.org/pub/databases/uniprot/current_release/) | Later protein snapshot |
| GOA UniProt | `234`, 17 June 2026 | [GOA UniProt releases](https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/) | Later annotation snapshot |
| Gene Ontology | data version `2026-06-15`; product directory `2026-06-19` | [GO current release](https://current.geneontology.org/) | Later ontology snapshot |
| GOA UniProt | `225`; declared GO version `2025-03-07` | [GOA UniProt releases](https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/) | Earlier annotation snapshot |
| Gene Ontology | `2025-02-06` | [GO release archive](https://release.geneontology.org/) | Prediction-eligibility boundary |
| Gene Ontology | `2025-03-16` | [GO release archive](https://release.geneontology.org/) | Earlier source-ID resolution |

No standalone `2025-03-07` GO product directory was available, so prediction eligibility and source-ID resolution use the two dated ontology files shown above.
