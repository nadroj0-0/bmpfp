#!/usr/bin/env python3
"""Describe benchmark taxonomy and exact GO-aspect membership."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
from typing import Iterable, Mapping


ASPECTS = ("BPO", "CCO", "MFO")
PREFIX = {"BPO": "bp", "CCO": "cc", "MFO": "mf"}
SPLITS = ("training", "validation", "test")
EXPECTED_TAXA = {
    "7227", "208963", "237561", "9606", "243232", "273057",
    "160488", "170187", "223283", "224308", "243273", "321314",
    "83333", "85962", "99287", "10090", "10116", "284812", "3702",
    "44689", "559292", "7955", "8355",
}
PATTERN_ORDER = (
    "BPO", "CCO", "MFO", "BPO+CCO", "BPO+MFO", "CCO+MFO", "BPO+CCO+MFO",
)


def read_ids(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            values.add(row[0])
    return values


def read_memberships(root: Path) -> dict[str, dict[str, set[str]]]:
    return {
        split: {
            aspect: read_ids(root / f"{PREFIX[aspect]}-{split}.csv")
            for aspect in ASPECTS
        }
        for split in SPLITS
    }


def add_taxon(mapping: dict[str, str], protein: str, taxon: str, context: str) -> None:
    taxon = taxon.removeprefix("taxon:").strip()
    if not protein or not taxon:
        return
    mapping[protein] = taxon


def temporal_taxa(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=2):
            for field in ("t0_id", "t1_id"):
                add_taxon(result, row[field], row["taxon_id"], f"{path}:{line_number}:{field}")
    return result


def homology_taxa(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=2):
            for field in ("raw_uniprot_accession", "uniprot_accession"):
                add_taxon(result, row[field], row["taxon_id"], f"{path}:{line_number}:{field}")
    return result


def parse_dmp(path: Path) -> Iterable[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            yield [field.strip("\t \r\n") for field in raw.split("|")]


class Taxonomy:
    def __init__(self, directory: Path) -> None:
        self.parents: dict[str, str] = {}
        self.ranks: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self.merged: dict[str, str] = {}
        self.deleted: set[str] = set()
        for row in parse_dmp(directory / "nodes.dmp"):
            self.parents[row[0]] = row[1]
            self.ranks[row[0]] = row[2]
        for row in parse_dmp(directory / "names.dmp"):
            if len(row) >= 4 and row[3] == "scientific name":
                self.names[row[0]] = row[1]
        for row in parse_dmp(directory / "merged.dmp"):
            self.merged[row[0]] = row[1]
        for row in parse_dmp(directory / "delnodes.dmp"):
            self.deleted.add(row[0])

    def resolve(self, source: str) -> dict[str, str]:
        current = source
        action = "exact"
        while current in self.merged:
            current = self.merged[current]
            action = "merged"
        if current in self.deleted:
            return self._unresolved(source, current, "deleted")
        if current not in self.parents:
            return self._unresolved(source, current, "unresolved")
        lineage: list[str] = []
        seen: set[str] = set()
        node = current
        while node not in seen:
            lineage.append(node)
            seen.add(node)
            parent = self.parents.get(node)
            if parent is None or parent == node:
                break
            node = parent
        domain = "Other cellular"
        if "10239" in lineage:
            domain = "Viruses"
        elif "2" in lineage:
            domain = "Bacteria"
        elif "2157" in lineage:
            domain = "Archaea"
        elif "2759" in lineage:
            domain = "Eukaryota"
        return {
            "original_taxid": source,
            "resolved_taxid": current,
            "action": action,
            "scientific_name": self.names.get(current, ""),
            "rank": self.ranks.get(current, ""),
            "domain": domain,
            "lineage_taxids": ";".join(lineage),
            "lineage_names": ";".join(self.names.get(value, value) for value in lineage),
        }

    @staticmethod
    def _unresolved(source: str, current: str, action: str) -> dict[str, str]:
        return {
            "original_taxid": source,
            "resolved_taxid": current,
            "action": action,
            "scientific_name": "",
            "rank": "",
            "domain": "Unresolved",
            "lineage_taxids": "",
            "lineage_names": "",
        }


def pattern_for(protein: str, memberships: Mapping[str, set[str]]) -> str:
    return "+".join(aspect for aspect in ASPECTS if protein in memberships[aspect])


def tsv(rows: list[dict], fields: list[str]) -> str:
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def add_summary_rows(
    benchmark: str,
    scope: str,
    memberships: Mapping[str, Mapping[str, set[str]]],
    taxon_map: Mapping[str, str],
    resolutions: Mapping[str, Mapping[str, str]],
    assignment_rows: list[dict],
    species_rows: list[dict],
    domain_rows: list[dict],
    pattern_rows: list[dict],
    intersection_rows: list[dict],
    aspect_count_rows: list[dict],
    domain_pattern_rows: list[dict],
) -> None:
    for split in (*SPLITS, "union"):
        aspect_sets = {
            aspect: (
                set().union(*(memberships[value][aspect] for value in SPLITS))
                if split == "union" else set(memberships[split][aspect])
            )
            for aspect in ASPECTS
        }
        if scope == "cafa3-23-taxa":
            aspect_sets = {
                aspect: {p for p in values if taxon_map.get(p) in EXPECTED_TAXA}
                for aspect, values in aspect_sets.items()
            }
        union = set().union(*aspect_sets.values())
        sets_with_all = {**aspect_sets, "ALL": union}
        for aspect, proteins in sets_with_all.items():
            species = Counter()
            domains = Counter()
            for protein in proteins:
                source_taxon = taxon_map.get(protein, "")
                resolution = resolutions.get(source_taxon, {
                    "resolved_taxid": "", "scientific_name": "", "domain": "Unresolved"
                })
                species[(resolution["resolved_taxid"], resolution["scientific_name"])] += 1
                domains[resolution["domain"]] += 1
                assignment_rows.append({
                    "benchmark": benchmark, "scope": scope, "split": split,
                    "aspect": aspect, "protein_id": protein, "source_taxid": source_taxon,
                    "resolved_taxid": resolution["resolved_taxid"],
                    "scientific_name": resolution["scientific_name"],
                    "domain": resolution["domain"],
                })
            total = len(proteins)
            for (taxon, name), count in sorted(species.items(), key=lambda item: (-item[1], item[0])):
                species_rows.append({
                    "benchmark": benchmark, "scope": scope, "split": split, "aspect": aspect,
                    "taxon_id": taxon, "scientific_name": name, "proteins": count,
                    "fraction": count / total if total else 0.0,
                })
            for domain, count in sorted(domains.items()):
                domain_rows.append({
                    "benchmark": benchmark, "scope": scope, "split": split, "aspect": aspect,
                    "domain": domain, "proteins": count,
                    "fraction": count / total if total else 0.0,
                })
        patterns = Counter(pattern_for(protein, aspect_sets) for protein in union)
        domains_by_pattern: Counter[tuple[str, str]] = Counter()
        for protein in union:
            pattern = pattern_for(protein, aspect_sets)
            source_taxon = taxon_map.get(protein, "")
            domain = resolutions.get(source_taxon, {"domain": "Unresolved"})["domain"]
            domains_by_pattern[(pattern, domain)] += 1
        for pattern in PATTERN_ORDER:
            count = patterns[pattern]
            pattern_rows.append({
                "benchmark": benchmark, "scope": scope, "split": split,
                "pattern": pattern, "proteins": count,
                "fraction": count / len(union) if union else 0.0,
            })
        for left, right in (("BPO", "CCO"), ("BPO", "MFO"), ("CCO", "MFO")):
            intersection_rows.append({
                "benchmark": benchmark, "scope": scope, "split": split,
                "set": f"{left}&{right}", "proteins": len(aspect_sets[left] & aspect_sets[right]),
            })
        intersection_rows.extend([
            {"benchmark": benchmark, "scope": scope, "split": split, "set": aspect,
             "proteins": len(aspect_sets[aspect])} for aspect in ASPECTS
        ])
        intersection_rows.extend([
            {"benchmark": benchmark, "scope": scope, "split": split, "set": "BPO&CCO&MFO",
             "proteins": len(set.intersection(*aspect_sets.values()))},
            {"benchmark": benchmark, "scope": scope, "split": split, "set": "UNION",
             "proteins": len(union)},
        ])
        aspect_counts = Counter(sum(protein in aspect_sets[aspect] for aspect in ASPECTS) for protein in union)
        for count in (1, 2, 3):
            aspect_count_rows.append({
                "benchmark": benchmark, "scope": scope, "split": split,
                "aspects_per_protein": count, "proteins": aspect_counts[count],
                "fraction": aspect_counts[count] / len(union) if union else 0.0,
            })
        for (pattern, domain), count in sorted(domains_by_pattern.items()):
            domain_pattern_rows.append({
                "benchmark": benchmark, "scope": scope, "split": split,
                "pattern": pattern, "domain": domain, "proteins": count,
            })


def plot_domain(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["split"] == "test" and row["scope"] == "full" and row["aspect"] != "ALL"]
    benchmarks = sorted({row["benchmark"] for row in selected})
    domains = ["Eukaryota", "Bacteria", "Archaea", "Viruses", "Other cellular", "Unresolved"]
    labels = [(benchmark, aspect) for benchmark in benchmarks for aspect in ASPECTS]
    lookup = {(row["benchmark"], row["aspect"], row["domain"]): row["fraction"] for row in selected}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottoms = [0.0] * len(labels)
    for domain in domains:
        values = [lookup.get((benchmark, aspect, domain), 0.0) for benchmark, aspect in labels]
        ax.bar(range(len(labels)), values, bottom=bottoms, label=domain)
        bottoms = [left + right for left, right in zip(bottoms, values)]
    ax.set_xticks(range(len(labels)), [f"{b}\n{a}" for b, a in labels], rotation=35, ha="right")
    ax.set_ylabel("Fraction of test proteins")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "domain_composition_by_aspect.png", dpi=180)
    fig.savefig(output / "domain_composition_by_aspect.pdf")
    plt.close(fig)


def plot_patterns(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["split"] == "test" and row["scope"] == "full"]
    benchmarks = sorted({row["benchmark"] for row in selected})
    lookup = {(row["benchmark"], row["pattern"]): row["fraction"] for row in selected}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = list(range(len(PATTERN_ORDER)))
    width = 0.8 / max(1, len(benchmarks))
    for index, benchmark in enumerate(benchmarks):
        offset = (index - (len(benchmarks) - 1) / 2) * width
        ax.bar([value + offset for value in x], [lookup.get((benchmark, pattern), 0.0) for pattern in PATTERN_ORDER], width, label=benchmark)
    ax.set_xticks(x, PATTERN_ORDER, rotation=35, ha="right")
    ax.set_ylabel("Fraction of test-protein union")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "aspect_membership_patterns.png", dpi=180)
    fig.savefig(output / "aspect_membership_patterns.pdf")
    plt.close(fig)


def plot_top_species(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["benchmark"] == "Global-NK" and row["scope"] == "full" and row["split"] == "test" and row["aspect"] != "ALL"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for axis, aspect in zip(axes, ASPECTS):
        values = sorted((row for row in selected if row["aspect"] == aspect), key=lambda row: -row["proteins"])[:10]
        axis.barh(range(len(values)), [row["proteins"] for row in reversed(values)])
        axis.set_yticks(range(len(values)), [row["scientific_name"] or row["taxon_id"] for row in reversed(values)], fontsize=7)
        axis.set_title(aspect)
        axis.set_xlabel("Test proteins")
    fig.tight_layout()
    fig.savefig(output / "top_taxa_by_aspect.png", dpi=180)
    fig.savefig(output / "top_taxa_by_aspect.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy-dir", type=Path, required=True)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--nklk-root", type=Path, required=True)
    parser.add_argument("--homology-root", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    global_flow = args.global_root / "reports/protein_flow.tsv"
    nklk_flow = args.nklk_root / "reports/protein_flow.tsv"
    homology_annotations = args.homology_root / "qualifying_annotations.tsv.gz"

    taxonomy = Taxonomy(args.taxonomy_dir)
    global_taxa = temporal_taxa(global_flow)
    nklk_taxa = temporal_taxa(nklk_flow)
    homology_taxon_map = homology_taxa(homology_annotations)
    memberships = {
        "Global-NK": read_memberships(args.global_root),
        "NK+LK": read_memberships(args.nklk_root),
        "MMseqs2-30": read_memberships(args.homology_root),
        "Random-H": read_memberships(args.random_root),
    }
    taxon_maps = {
        "Global-NK": global_taxa,
        "NK+LK": nklk_taxa,
        "MMseqs2-30": homology_taxon_map,
        "Random-H": homology_taxon_map,
    }
    used_taxa: set[str] = set()
    for benchmark, values in memberships.items():
        proteins = set().union(*(values[split][aspect] for split in SPLITS for aspect in ASPECTS))
        used_taxa.update(taxon_maps[benchmark][protein] for protein in proteins)
    resolutions = {taxon: taxonomy.resolve(taxon) for taxon in sorted(used_taxa, key=int)}

    assignments: list[dict] = []
    species: list[dict] = []
    domains: list[dict] = []
    patterns: list[dict] = []
    intersections: list[dict] = []
    aspect_counts: list[dict] = []
    domain_patterns: list[dict] = []
    for benchmark in ("Global-NK", "NK+LK", "MMseqs2-30", "Random-H"):
        scopes = ("full", "cafa3-23-taxa") if benchmark in {"MMseqs2-30", "Random-H"} else ("full",)
        for scope in scopes:
            add_summary_rows(
                benchmark, scope, memberships[benchmark], taxon_maps[benchmark], resolutions,
                assignments, species, domains, patterns, intersections, aspect_counts, domain_patterns,
            )

    (output / "taxonomy_id_resolution.tsv").write_text(tsv(list(resolutions.values()), [
        "original_taxid", "resolved_taxid", "action", "scientific_name", "rank", "domain",
        "lineage_taxids", "lineage_names",
    ]), encoding="utf-8")
    with gzip.open(output / "protein_taxonomy_assignments.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        handle.write(tsv(assignments, [
            "benchmark", "scope", "split", "aspect", "protein_id", "source_taxid",
            "resolved_taxid", "scientific_name", "domain",
        ]))
    table_specs = [
        ("species_composition.tsv", species, ["benchmark", "scope", "split", "aspect", "taxon_id", "scientific_name", "proteins", "fraction"]),
        ("domain_composition.tsv", domains, ["benchmark", "scope", "split", "aspect", "domain", "proteins", "fraction"]),
        ("aspect_membership_patterns.tsv", patterns, ["benchmark", "scope", "split", "pattern", "proteins", "fraction"]),
        ("aspect_intersections.tsv", intersections, ["benchmark", "scope", "split", "set", "proteins"]),
        ("aspects_per_protein.tsv", aspect_counts, ["benchmark", "scope", "split", "aspects_per_protein", "proteins", "fraction"]),
        ("domain_by_membership.tsv", domain_patterns, ["benchmark", "scope", "split", "pattern", "domain", "proteins"]),
    ]
    for name, rows, fields in table_specs:
        (output / name).write_text(tsv(rows, fields), encoding="utf-8")
    unresolved = [row for row in resolutions.values() if row["domain"] == "Unresolved" or row["action"] in {"merged", "deleted"}]
    (output / "unresolved_taxa.tsv").write_text(tsv(unresolved, [
        "original_taxid", "resolved_taxid", "action", "scientific_name", "rank", "domain",
        "lineage_taxids", "lineage_names",
    ]), encoding="utf-8")
    if not args.skip_plots:
        plot_domain(domains, output)
        plot_patterns(patterns, output)
        plot_top_species(species, output)
    summary = {
        "benchmarks": sorted(memberships), "taxa_used": len(used_taxa),
        "unresolved_taxa": len([row for row in resolutions.values() if row["domain"] == "Unresolved"]),
        "figures_generated": not args.skip_plots,
        "cafa3_target_taxa_filter": sorted(EXPECTED_TAXA, key=int),
        "interpretation_boundary": "Composition is descriptive and does not establish that taxonomy causes PFP performance differences.",
    }
    (output / "taxonomy_and_aspect_analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "taxonomy_and_aspect_analysis.md").write_text((
        "# Taxonomy and GO-aspect analysis\n\n"
        f"Analysed {len(memberships)} explicitly supplied benchmarks.\n\n"
        f"Resolved {len(used_taxa)} source taxon IDs; {summary['unresolved_taxa']} remained unresolved.\n\n"
        "The MMseqs2-30 and Random-H results include both unrestricted and CAFA3-23-taxon views. "
        "These tables describe benchmark composition and do not establish a causal taxonomic explanation for model performance.\n"
    ), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
