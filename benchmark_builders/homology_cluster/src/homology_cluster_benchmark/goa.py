from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import gzip
import json
import logging
from pathlib import Path
import re
import time

from .config import ASPECT_TO_NAMESPACE, BENCHMARK_EVIDENCE_CODES, ROOT_TERMS
from .inputs import open_text
from .models import AnnotationRecord, GoaLoadResult
from .ontology import Ontology


LOGGER = logging.getLogger(__name__)


ISOFORM_ACCESSION_RE = re.compile(r"^.+-\d+$")


def _taxon(value: str) -> str:
    first = value.split("|", 1)[0]
    return first.split(":", 1)[1] if first.startswith("taxon:") else first


def load_goa(
    path: Path,
    ontology: Ontology,
    evidence_codes: frozenset[str] = BENCHMARK_EVIDENCE_CODES,
    spool_dir: Path | None = None,
) -> GoaLoadResult:
    started = time.monotonic()
    LOGGER.info("GOA scan started: %s", path)
    records: list[AnnotationRecord] = []
    annotations: dict[str, set[str]] = {}
    qualifying_accessions: set[str] = set()
    counters = Counter()
    evidence_counts = Counter()
    taxonomy_counts = Counter()
    direct_go_counts = Counter()
    candidate_accessions: set[str] = set()
    record_spool = spool_dir / "qualifying_annotations.raw.jsonl.gz" if spool_dir else None
    if spool_dir:
        spool_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        # Use fast compression for the temporary annotation spool.
        record_handle = stack.enter_context(
            gzip.open(record_spool, "wt", encoding="utf-8", compresslevel=1)
        ) if record_spool else None
        handle = stack.enter_context(open_text(path))
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number % 1_000_000 == 0:
                LOGGER.info(
                    "GOA progress: lines=%d accepted=%d rejected=%d",
                    line_number,
                    counters["kept_rows"],
                    sum(value for key, value in counters.items() if key.startswith("rejected_")),
                )
            if raw_line.startswith("!"):
                continue
            if not raw_line.strip():
                continue
            columns = raw_line.rstrip("\n\r").split("\t")
            counters["processed"] += 1
            accession = columns[1].strip()
            qualifier_tokens = tuple(item for item in columns[3].split("|") if item)
            raw_go_id = columns[4].strip()
            evidence = columns[6].strip()
            aspect = columns[8].strip()
            object_type = columns[11].strip()
            taxon_id = _taxon(columns[12].strip())
            gene_product_form = columns[16].strip()

            reason = ""
            detail = ""
            if columns[0] != "UniProtKB":
                reason = "database"
            elif object_type != "protein":
                reason = "object_type"
            elif not accession:
                reason = "missing_accession"
            elif ISOFORM_ACCESSION_RE.fullmatch(accession):
                reason = "isoform_accession"
                detail = "isoform suffix is preserved and excluded; no canonicalization policy is assumed"
            elif evidence not in evidence_codes:
                reason = "evidence_code"
            elif "NOT" in qualifier_tokens:
                reason = "not_qualifier"
            elif aspect not in ASPECT_TO_NAMESPACE:
                reason = "aspect"
            elif gene_product_form:
                reason = "isoform_specific"
                detail = gene_product_form

            term_description: dict[str, object] = {}
            canonical = None
            if not reason:
                candidate_accessions.add(accession)
                term_description = ontology.describe(raw_go_id)
                canonical = str(term_description["canonical_id"] or "")
                if not canonical:
                    reason = str(term_description["status"])
                    detail = "consider=" + "|".join(term_description["consider"])  # type: ignore[arg-type]
                elif ontology.namespace(canonical) != ASPECT_TO_NAMESPACE[aspect]:
                    reason = "namespace_mismatch"
                    detail = (
                        f"GAF aspect={aspect}; ontology namespace={ontology.namespace(canonical)}"
                    )

            if reason:
                counters[f"rejected_{reason}"] += 1
                continue

            if canonical in ROOT_TERMS:
                counters["accepted_direct_root"] += 1
            record = AnnotationRecord(
                raw_accession=accession,
                protein_id=accession,
                raw_go_id=raw_go_id,
                go_id=canonical,
                namespace=ontology.namespace(canonical),
                aspect=aspect,
                evidence=evidence,
                qualifier="|".join(qualifier_tokens),
                taxon_id=taxon_id,
            )
            if record_handle:
                record_handle.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
            else:
                records.append(record)
            # A spool keeps direct terms on disk while accession membership remains in memory.
            qualifying_accessions.add(accession)
            if record_spool is None:
                annotations.setdefault(accession, set()).add(canonical)
            counters["kept_rows"] += 1
            evidence_counts[evidence] += 1
            taxonomy_counts[taxon_id or "unknown"] += 1
            direct_go_counts[canonical] += 1

    counters["qualifying_proteins"] = len(qualifying_accessions)
    LOGGER.info(
        "GOA scan completed: lines=%d accepted_rows=%d qualifying_accessions=%d "
        "elapsed_seconds=%.1f",
        line_number if 'line_number' in locals() else 0,
        counters["kept_rows"],
        len(qualifying_accessions),
        time.monotonic() - started,
    )
    return GoaLoadResult(
        records=records,
        annotations=annotations,
        qualifying_accessions=qualifying_accessions,
        counters=counters,
        evidence_counts=evidence_counts,
        taxonomy_counts=taxonomy_counts,
        direct_go_counts=direct_go_counts,
        candidate_accessions=candidate_accessions,
        record_spool=record_spool,
    )


def iter_annotation_records(result: GoaLoadResult):
    if result.record_spool is None:
        yield from result.records
        return
    with gzip.open(result.record_spool, "rt", encoding="utf-8") as handle:
        for line in handle:
            values = json.loads(line)
            protein_id, action = result.accession_map.get(
                values["raw_accession"], (values["protein_id"], values["accession_action"])
            )
            values["protein_id"] = protein_id
            values["accession_action"] = action
            yield AnnotationRecord(**values)
