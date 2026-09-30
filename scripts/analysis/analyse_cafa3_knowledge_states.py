#!/usr/bin/env python3
"""Classify the original PFP CAFA3 test rows by official knowledge state."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from label_space_common import (
    ASPECTS,
    ASPECT_TO_PREFIX,
    ASPECT_TO_ROOT,
    CSV_SPLITS,
)


KNOWLEDGE_STATES = (
    "no_knowledge",
    "limited_knowledge",
    "unclassified_by_official_lists",
)
OFFICIAL_LIST_TEMPLATE = "benchmark20171115/lists/{prefix}o_all_{kind}.txt"
TOO_FEW_PATTERN = re.compile(
    r"^benchmark20171115/lists/too_few/(bp|cc|mf)o_[^/]+_(type1|type2)\.txt$"
)


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def _read_split(
    path: Path,
    aspect: str,
    keep_rows: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = ASPECT_TO_ROOT[aspect]
    rows: dict[str, dict[str, Any]] = {}
    counts = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        terms = header[2:]

        for row in reader:
            protein_id = row[0].strip()
            positives = tuple(term for term, value in zip(terms, row[2:]) if value == "1")
            has_root = root in positives
            root_only = positives == (root,)
            has_non_root = any(term != root for term in positives)
            all_zero = not positives
            if has_non_root and not has_root:
                counts["non_root_without_root"] += 1
            if root_only:
                counts["root_only"] += 1
            elif has_non_root:
                counts["non_root_observed_truth"] += 1
            elif all_zero:
                counts["all_zero"] += 1
            counts["rows"] += 1
            if keep_rows:
                rows[protein_id] = {
                    "positive_label_count": len(positives),
                    "root_present": has_root,
                    "root_only": root_only,
                    "non_root_observed_truth": has_non_root,
                    "all_zero": all_zero,
                }

    total = counts["rows"]
    summary = {
        "rows": total,
        "root_only": counts["root_only"],
        "root_only_percent": _percent(counts["root_only"], total),
        "non_root_observed_truth": counts["non_root_observed_truth"],
        "non_root_observed_truth_percent": _percent(
            counts["non_root_observed_truth"], total
        ),
        "all_zero": counts["all_zero"],
        "all_zero_percent": _percent(counts["all_zero"], total),
        "non_root_without_root": counts["non_root_without_root"],
        "go_terms": len(terms),
        "root_term": root,
    }
    return summary, rows


def _match_tar_member(member_name: str, expected: str) -> bool:
    normalized = member_name.lstrip("./")
    return normalized == expected or normalized.endswith(f"/{expected}")


def _read_archive_members(
    archive_path: Path,
) -> dict[str, bytes]:
    expected: set[str] = set()
    for aspect in ASPECTS:
        prefix = ASPECT_TO_PREFIX[aspect]
        for kind in ("type1", "type2", "typex"):
            expected.add(OFFICIAL_LIST_TEMPLATE.format(prefix=prefix, kind=kind))

    payloads: dict[str, bytes] = {}
    archive = tarfile.open(archive_path, mode="r:*")
    with archive:
        for member in archive:
            normalized = member.name.lstrip("./")
            matches = [name for name in expected if _match_tar_member(member.name, name)]
            dynamic_too_few = TOO_FEW_PATTERN.match(normalized)
            if not matches and dynamic_too_few:
                matches = [normalized]
            if not matches:
                continue
            expected_name = matches[0]
            handle = archive.extractfile(member)
            payload = handle.read()
            if not payload:
                if dynamic_too_few:
                    continue
            payloads[expected_name] = payload
    return payloads


def _parse_target_list(payload: bytes) -> set[str]:
    values: set[str] = set()
    text = payload.decode("utf-8-sig")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        protein_id = line.split()[0]
        values.add(protein_id)
    return values


def _tsv(rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue()


def _build_summary(
    test_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    official_lists: Mapping[str, Mapping[str, set[str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    aspect_summaries: dict[str, Any] = {}
    protein_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []

    for aspect in ASPECTS:
        by_type = official_lists[aspect]
        test = test_rows[aspect]
        test_ids = set(test)
        state_counts = {
            state: Counter() for state in KNOWLEDGE_STATES
        }
        for protein_id in sorted(test):
            in_type1 = protein_id in by_type["all_type1"]
            in_type2 = protein_id in by_type["all_type2"]
            if in_type1:
                state = "no_knowledge"
                official_type = "type1"
            elif in_type2:
                state = "limited_knowledge"
                official_type = "type2"
            else:
                state = "unclassified_by_official_lists"
                official_type = ""
            # The organizer's aggregate all-target lists include the targets
            # repeated under too_few, so test the narrower diagnostic first.
            if protein_id in by_type["too_few_type1"] or protein_id in by_type["too_few_type2"]:
                official_list_group = "too_few"
            elif protein_id in by_type["main_type1"] or protein_id in by_type["main_type2"]:
                official_list_group = "main_all"
            else:
                official_list_group = ""
            truth = test[protein_id]
            counter = state_counts[state]
            counter["proteins"] += 1
            counter["root_only"] += int(truth["root_only"])
            counter["non_root_observed_truth"] += int(truth["non_root_observed_truth"])
            counter["all_zero"] += int(truth["all_zero"])
            protein_rows.append(
                {
                    "aspect": aspect,
                    "protein_id": protein_id,
                    "knowledge_state": state,
                    "official_type": official_type,
                    "official_list_group": official_list_group,
                    "in_official_typex": int(protein_id in by_type["main_typex"]),
                    "root_only": int(truth["root_only"]),
                    "non_root_observed_truth": int(truth["non_root_observed_truth"]),
                    "all_zero": int(truth["all_zero"]),
                    "positive_label_count": truth["positive_label_count"],
                }
            )

        total = len(test)
        states = {}
        for state in KNOWLEDGE_STATES:
            values = state_counts[state]
            states[state] = {
                "proteins": values["proteins"],
                "percent_of_test": _percent(values["proteins"], total),
                "root_only": values["root_only"],
                "non_root_observed_truth": values["non_root_observed_truth"],
                "all_zero": values["all_zero"],
            }
        type_union = by_type["main_type1"] | by_type["main_type2"]
        typex = by_type["main_typex"]
        aspect_summaries[aspect] = {
            "pfp_test_rows": total,
            "states": states,
            "classified_rows": total - states["unclassified_by_official_lists"]["proteins"],
            "classified_percent": _percent(
                total - states["unclassified_by_official_lists"]["proteins"], total
            ),
            "official_list_counts": {kind: len(values) for kind, values in by_type.items()},
            "typex_diagnostic": {
                "equals_type1_union_type2": typex == type_union,
                "typex_not_in_type1_union_type2": len(typex - type_union),
                "type1_union_type2_not_in_typex": len(type_union - typex),
            },
        }
        for kind, values in by_type.items():
            alignment_rows.append(
                {
                    "aspect": aspect,
                    "official_list": kind,
                    "official_targets": len(values),
                    "pfp_test_overlap": len(values & test_ids),
                    "official_targets_absent_from_pfp_test": len(values - test_ids),
                    "pfp_test_rows_absent_from_list": len(test_ids - values),
                }
            )
    return aspect_summaries, protein_rows, alignment_rows


def _organize_official_lists(
    archive_payloads: Mapping[str, bytes],
) -> dict[str, dict[str, set[str]]]:
    official_lists: dict[str, dict[str, set[str]]] = {}
    for aspect in ASPECTS:
        prefix = ASPECT_TO_PREFIX[aspect]
        main: dict[str, set[str]] = {}
        for kind in ("type1", "type2", "typex"):
            member = OFFICIAL_LIST_TEMPLATE.format(prefix=prefix, kind=kind)
            main[kind] = _parse_target_list(archive_payloads[member])
        too_few = {"type1": set(), "type2": set()}
        member_prefix = f"benchmark20171115/lists/too_few/{prefix}o_"
        for member, payload in archive_payloads.items():
            if not member.startswith(member_prefix):
                continue
            kind = "type1" if member.endswith("_type1.txt") else "type2"
            too_few[kind].update(_parse_target_list(payload))
        official_lists[aspect] = {
            "main_type1": main["type1"],
            "main_type2": main["type2"],
            "main_typex": main["typex"],
            "too_few_type1": too_few["type1"],
            "too_few_type2": too_few["type2"],
            "all_type1": main["type1"] | too_few["type1"],
            "all_type2": main["type2"] | too_few["type2"],
        }
    return official_lists


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfp-csv-dir", type=Path, required=True)
    parser.add_argument("--official-cafa-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-id", default="cafa3/original-pfp-nine-csvs")
    args = parser.parse_args()

    csv_dir = args.pfp_csv_dir.resolve()
    archive_path = args.official_cafa_archive.resolve()
    output_dir = args.output_dir.resolve()

    csv_paths: dict[str, Path] = {}
    for aspect in ASPECTS:
        prefix = ASPECT_TO_PREFIX[aspect]
        for split in CSV_SPLITS:
            key = f"{aspect}_{split}"
            path = csv_dir / f"{prefix}-{split}.csv"
            csv_paths[key] = path

    split_summaries: dict[str, dict[str, Any]] = {aspect: {} for aspect in ASPECTS}
    test_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for aspect in ASPECTS:
        for split in CSV_SPLITS:
            summary, rows = _read_split(
                csv_paths[f"{aspect}_{split}"], aspect, keep_rows=split == "test"
            )
            split_summaries[aspect][split] = summary
            if split == "test":
                test_rows[aspect] = rows

    archive_payloads = _read_archive_members(archive_path)
    official_lists = _organize_official_lists(archive_payloads)

    aspects, protein_rows, alignment_rows = _build_summary(test_rows, official_lists)
    summary = {
        "analysis_kind": "cafa3_knowledge_states",
        "benchmark_id": args.benchmark_id,
        "scope": "Original PFP CAFA3 test rows; train/validation receive root-state counts only",
        "knowledge_state_mapping": {
            "type1": "no_knowledge",
            "type2": "limited_knowledge",
            "limited_knowledge_definition": (
                "no prior qualifying experimental annotation in the evaluated ontology, "
                "but prior qualifying experimental annotation in at least one other ontology"
            ),
            "important_boundary": (
                "official CAFA3 target lists are organizer classifications; this analysis "
                "does not reconstruct organizer-private endpoint annotations from later snapshots"
            ),
            "too_few_policy": (
                "type1/type2 lists under lists/too_few retain organizer knowledge-state "
                "classification but were excluded from the main all-target evaluation lists"
            ),
        },
        "root_state_definition": {
            "root_only": "the projected CSV truth has exactly the ontology root as positive",
            "non_root_observed_truth": "the projected CSV truth contains at least one non-root positive",
        },
        "aspects": aspects,
        "split_root_state_counts": split_summaries,
    }

    split_rows = []
    state_rows = []
    for aspect in ASPECTS:
        for split in CSV_SPLITS:
            split_rows.append(
                {"aspect": aspect, "split": split, **split_summaries[aspect][split]}
            )
        for state in KNOWLEDGE_STATES:
            state_rows.append(
                {
                    "aspect": aspect,
                    "knowledge_state": state,
                    **aspects[aspect]["states"][state],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cafa3_knowledge_states.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "cafa3_split_root_state_counts.tsv").write_text(
        _tsv(
            split_rows,
            (
                "aspect",
                "split",
                "rows",
                "root_only",
                "root_only_percent",
                "non_root_observed_truth",
                "non_root_observed_truth_percent",
                "all_zero",
                "all_zero_percent",
                "non_root_without_root",
                "go_terms",
                "root_term",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "cafa3_test_knowledge_state_counts.tsv").write_text(
        _tsv(
            state_rows,
            (
                "aspect",
                "knowledge_state",
                "proteins",
                "percent_of_test",
                "root_only",
                "non_root_observed_truth",
                "all_zero",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "cafa3_test_knowledge_states.tsv").write_text(
        _tsv(
            protein_rows,
            (
                "aspect",
                "protein_id",
                "knowledge_state",
                "official_type",
                "official_list_group",
                "in_official_typex",
                "root_only",
                "non_root_observed_truth",
                "all_zero",
                "positive_label_count",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "cafa3_official_list_alignment.tsv").write_text(
        _tsv(
            alignment_rows,
            (
                "aspect",
                "official_list",
                "official_targets",
                "pfp_test_overlap",
                "official_targets_absent_from_pfp_test",
                "pfp_test_rows_absent_from_list",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "METHOD_NOTE.md").write_text(
        "# CAFA3 knowledge-state analysis\n\n"
        "This analysis classifies the exact original PFP CAFA3 test rows with the "
        "organiser-provided `type1` and `type2` lists. CAFA `type1` is no knowledge; "
        "`type2` is limited knowledge in the evaluated aspect. It does not infer "
        "historical state from a later public GOA release.\n\n"
        "Rows in the organiser's `lists/too_few` directory keep their type1/type2 "
        "classification and are marked separately from the main all-target lists.\n\n"
        "Training and validation rows are not assigned CAFA target types because the "
        "official lists describe challenge targets. Their root-only/non-root CSV truth "
        "counts are reported separately.\n",
        encoding="utf-8",
    )

    print(f"Original PFP CAFA3 knowledge-state analysis: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
