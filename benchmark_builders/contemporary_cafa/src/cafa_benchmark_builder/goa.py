from __future__ import annotations

from contextlib import contextmanager
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import shutil
import subprocess

from .config import CAFA3_FINAL_EXP_CODES
from .io_utils import open_text
from .models import AnnotationLoadResult
from .ontology import Ontology

LOGGER = logging.getLogger(__name__)


def parse_taxon(value: str) -> str:
    first = value.split("|", 1)[0]
    if first.startswith("taxon:"):
        return first.split(":", 1)[1]
    return first


@contextmanager
def open_gaf_text(path: str | Path):
    """Open GAF text, optionally using pigz for faster gzip decompression."""
    path = Path(path)
    use_pigz = (
        path.suffix == ".gz"
        and os.environ.get("CAFA_BUILDER_USE_PIGZ", "1") != "0"
        and shutil.which("pigz") is not None
    )
    if not use_pigz:
        with open_text(path) as handle:
            yield handle
        return

    LOGGER.info("Streaming gzip with pigz: %s", path)
    proc = subprocess.Popen(["pigz", "-dc", str(path)], stdout=subprocess.PIPE, text=True)
    try:
        yield proc.stdout
    finally:
        proc.stdout.close()
        return_code = proc.wait()
        # A consumer may close the pipe before pigz reaches EOF.
        if return_code not in (0, -13, 141):
            raise RuntimeError(f"pigz failed for {path} with exit code {return_code}")


def progress_interval_from_env() -> int:
    return max(0, int(os.environ.get("CAFA_BUILDER_GOA_PROGRESS_INTERVAL", "1000000")))


def load_annotation_map(
    path: str | Path,
    evidence_codes: frozenset[str] = CAFA3_FINAL_EXP_CODES,
    target_taxa: frozenset[str] = frozenset(),
    allowed_proteins: set[str] | frozenset[str] | None = None,
    progress_interval: int | None = None,
) -> dict[str, set[str]]:
    annots: dict[str, set[str]] = defaultdict(set)
    processed = 0
    kept = 0
    skipped_not_allowed = 0
    skipped_evidence = 0
    skipped_not = 0
    skipped_aspect = 0
    skipped_taxon = 0
    skipped_db = 0
    progress_every = progress_interval_from_env() if progress_interval is None else progress_interval

    if allowed_proteins is not None:
        LOGGER.info("Filtering GOA %s to %d loaded UniProt sequence IDs", path, len(allowed_proteins))

    with open_gaf_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            processed += 1

            if cols[0] != "UniProtKB":
                skipped_db += 1
            else:
                protein_id = cols[1]
                if allowed_proteins is not None and protein_id not in allowed_proteins:
                    skipped_not_allowed += 1
                elif cols[6] not in evidence_codes:
                    skipped_evidence += 1
                elif cols[3] and "NOT" in cols[3].split("|"):
                    skipped_not += 1
                elif cols[8] not in {"P", "C", "F"}:
                    skipped_aspect += 1
                elif target_taxa and parse_taxon(cols[12]) not in target_taxa:
                    skipped_taxon += 1
                else:
                    annots[protein_id].add(cols[4])
                    kept += 1

            if progress_every and processed % progress_every == 0:
                LOGGER.info(
                    "GOA progress for %s: processed=%d kept_rows=%d proteins=%d skipped_outside_sequences=%d",
                    path,
                    processed,
                    kept,
                    len(annots),
                    skipped_not_allowed,
                )
    LOGGER.info(
        "Loaded GOA annotations from %s: processed=%d kept_rows=%d proteins=%d "
        "skipped_db=%d skipped_outside_sequences=%d skipped_evidence=%d skipped_not=%d "
        "skipped_aspect=%d skipped_taxon=%d",
        path,
        processed,
        kept,
        len(annots),
        skipped_db,
        skipped_not_allowed,
        skipped_evidence,
        skipped_not,
        skipped_aspect,
        skipped_taxon,
    )
    return dict(annots)


def load_normalized_annotation_map(
    path: str | Path,
    *,
    alias_to_primary: dict[str, str],
    source_ontology: Ontology,
    benchmark_ontology: Ontology,
    allow_frozen_source_fallback: bool = True,
    evidence_codes: frozenset[str] = CAFA3_FINAL_EXP_CODES,
    target_taxa: frozenset[str] = frozenset(),
    exclude_on_or_before: str | None = None,
    include_on_or_before: str | None = None,
    progress_interval: int | None = None,
) -> AnnotationLoadResult:
    """Map filtered GOA rows through UniProt aliases and source-to-benchmark GO IDs."""
    annots: dict[str, set[str]] = defaultdict(set)
    counters = Counter()
    evidence_counts = Counter()
    taxon_counts = Counter()
    unmapped_terms = Counter()
    out_of_benchmark_terms = Counter()
    date_filter_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    progress_every = progress_interval_from_env() if progress_interval is None else progress_interval

    with open_gaf_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            counters["processed"] += 1

            if cols[0] != "UniProtKB":
                counters["skipped_db"] += 1
            else:
                protein_id = alias_to_primary.get(cols[1])
                taxon_id = parse_taxon(cols[12])
                if protein_id is None:
                    counters["skipped_outside_sequences"] += 1
                elif cols[6] not in evidence_codes:
                    counters["skipped_evidence"] += 1
                elif cols[3] and "NOT" in cols[3].split("|"):
                    counters["skipped_not"] += 1
                elif cols[8] not in {"P", "C", "F"}:
                    counters["skipped_aspect"] += 1
                elif target_taxa and taxon_id not in target_taxa:
                    counters["skipped_taxon"] += 1
                elif exclude_on_or_before is not None and cols[13] <= exclude_on_or_before:
                    counters["skipped_backfill"] += 1
                    date_filter_counts[(protein_id, cols[8])]["skipped_backfill"] += 1
                elif include_on_or_before is not None and cols[13] > include_on_or_before:
                    counters["skipped_after_cutoff"] += 1
                    date_filter_counts[(protein_id, cols[8])]["skipped_after_cutoff"] += 1
                else:
                    date_filter_counts[(protein_id, cols[8])]["passed_date_filter"] += 1
                    source_term = source_ontology.resolve_term(cols[4])
                    if source_term is None:
                        frozen_term = benchmark_ontology.resolve_term(cols[4])
                        use_frozen = bool(allow_frozen_source_fallback and frozen_term)
                        if use_frozen:
                            source_term = frozen_term
                            counters["resolved_in_frozen_fallback"] += 1
                        else:
                            counters["unmapped_source_go"] += 1
                            unmapped_terms[cols[4]] += 1
                            source_term = None
                    if source_term is not None:
                        benchmark_term = benchmark_ontology.resolve_term(source_term)
                        if benchmark_term is None:
                            # A replaced source term can still resolve from its original GAF ID.
                            benchmark_term = benchmark_ontology.resolve_term(cols[4])
                        if benchmark_term is None:
                            counters["outside_frozen_ontology"] += 1
                            out_of_benchmark_terms[source_term] += 1
                        else:
                            annots[protein_id].add(benchmark_term)
                            counters["kept_rows"] += 1
                            evidence_counts[cols[6]] += 1
                            taxon_counts[taxon_id] += 1

            if progress_every and counters["processed"] % progress_every == 0:
                LOGGER.info(
                    "GOA progress for %s: processed=%d kept_rows=%d proteins=%d "
                    "skipped_outside_sequences=%d skipped_backfill=%d skipped_after_cutoff=%d",
                    path,
                    counters["processed"],
                    counters["kept_rows"],
                    len(annots),
                    counters["skipped_outside_sequences"],
                    counters["skipped_backfill"],
                    counters["skipped_after_cutoff"],
                )
    LOGGER.info(
        "Loaded normalized GOA annotations from %s: processed=%d kept_rows=%d proteins=%d "
        "skipped_outside_sequences=%d skipped_evidence=%d skipped_not=%d "
        "skipped_backfill=%d skipped_after_cutoff=%d outside_frozen_ontology=%d "
        "unmapped_source_go=%d",
        path,
        counters["processed"],
        counters["kept_rows"],
        len(annots),
        counters["skipped_outside_sequences"],
        counters["skipped_evidence"],
        counters["skipped_not"],
        counters["skipped_backfill"],
        counters["skipped_after_cutoff"],
        counters["outside_frozen_ontology"],
        counters["unmapped_source_go"],
    )
    return AnnotationLoadResult(
        annotations={protein_id: set(terms) for protein_id, terms in annots.items()},
        counters=counters,
        evidence_counts=evidence_counts,
        taxon_counts=taxon_counts,
        unmapped_terms=unmapped_terms,
        out_of_benchmark_terms=out_of_benchmark_terms,
        date_filter_counts={key: Counter(value) for key, value in date_filter_counts.items()},
    )
