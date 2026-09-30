"""Small GO OBO reader used to classify source annotations by namespace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Mapping, Optional, Set

from .readers import open_text


ASPECTS = ("BPO", "CCO", "MFO")
SPLITS = ("training", "validation", "test")
ASPECT_TO_FILE = {"BPO": "bp", "CCO": "cc", "MFO": "mf"}
ASPECT_TO_NAMESPACE = {
    "BPO": "biological_process",
    "CCO": "cellular_component",
    "MFO": "molecular_function",
}
ASPECT_TO_ROOT = {
    "BPO": "GO:0008150",
    "CCO": "GO:0005575",
    "MFO": "GO:0003674",
}


@dataclass(frozen=True)
class Term:
    id: str
    namespace: str
    parents: FrozenSet[str]


class Ontology:
    def __init__(
        self,
        terms: Mapping[str, Term],
        aliases: Mapping[str, str],
        replacements: Mapping[str, str],
    ) -> None:
        self.terms = dict(terms)
        self.aliases = dict(aliases)
        self.replacements = dict(replacements)

    @classmethod
    def load(cls, path: Path) -> "Ontology":
        objects: Dict[str, dict] = {}
        current: Optional[dict] = None
        with open_text(path) as handle:
            for raw in handle:
                line = raw.strip()
                if line == "[Term]":
                    if current and "id" in current:
                        objects[current["id"]] = current
                    current = {
                        "alt_ids": [],
                        "parents": [],
                        "obsolete": False,
                        "replaced_by": [],
                    }
                    continue
                if line == "[Typedef]":
                    if current and "id" in current:
                        objects[current["id"]] = current
                    current = None
                    continue
                if current is None or ": " not in line:
                    continue
                key, value = line.split(": ", 1)
                if key == "id":
                    current["id"] = value
                elif key == "namespace":
                    current["namespace"] = value
                elif key == "alt_id":
                    current["alt_ids"].append(value)
                elif key == "is_a":
                    current["parents"].append(value.split(" ! ", 1)[0])
                elif key == "is_obsolete" and value == "true":
                    current["obsolete"] = True
                elif key == "replaced_by":
                    current["replaced_by"].append(value)
        if current and "id" in current:
            objects[current["id"]] = current

        aliases: Dict[str, str] = {}
        replacements: Dict[str, str] = {}
        live_ids = {
            term_id for term_id, item in objects.items() if not item["obsolete"]
        }
        for term_id in live_ids:
            aliases[term_id] = term_id
            for alt_id in objects[term_id]["alt_ids"]:
                aliases[alt_id] = term_id
        for term_id, item in objects.items():
            if item["obsolete"] and len(item["replaced_by"]) == 1:
                replacement = aliases.get(
                    item["replaced_by"][0], item["replaced_by"][0]
                )
                if replacement in live_ids:
                    replacements[term_id] = replacement
                    for alt_id in item["alt_ids"]:
                        replacements[alt_id] = replacement

        terms: Dict[str, Term] = {}
        for term_id in live_ids:
            item = objects[term_id]
            namespace = item.get("namespace", "")
            parents = frozenset(
                aliases.get(parent, replacements.get(parent, parent))
                for parent in item["parents"]
                if aliases.get(parent, replacements.get(parent, parent)) in live_ids
            )
            terms[term_id] = Term(term_id, namespace, parents)
        return cls(terms, aliases, replacements)

    def resolve(self, term_id: str) -> Optional[str]:
        return self.aliases.get(term_id) or self.replacements.get(term_id)

    def namespace(self, term_id: str) -> Optional[str]:
        canonical = self.resolve(term_id)
        return self.terms[canonical].namespace if canonical else None

    def ancestors(self, term_id: str) -> FrozenSet[str]:
        canonical = self.resolve(term_id)
        if canonical is None:
            return frozenset()
        seen: Set[str] = set()
        pending = [canonical]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(self.terms[current].parents)
        return frozenset(seen)
