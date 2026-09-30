from __future__ import annotations

from collections import deque
from pathlib import Path

from .io_utils import open_text


BIOLOGICAL_PROCESS = "GO:0008150"
MOLECULAR_FUNCTION = "GO:0003674"
CELLULAR_COMPONENT = "GO:0005575"
FUNC_DICT = {"cc": CELLULAR_COMPONENT, "mf": MOLECULAR_FUNCTION, "bp": BIOLOGICAL_PROCESS}
NAMESPACES = {
    "cc": "cellular_component",
    "mf": "molecular_function",
    "bp": "biological_process",
}


class Ontology(object):
    """DeepGOPlus-compatible OBO graph with explicit GO-ID normalisation."""

    def __init__(
        self,
        filename: str | Path,
        with_rels: bool = False,
        relationship_types: frozenset[str] | None = None,
    ):
        self.filename = Path(filename)
        self.relationship_types = relationship_types
        self.canonical_ids: dict[str, str] = {}
        self.obsolete_replacements: dict[str, str] = {}
        self.primary_terms: set[str] = set()
        self.ont = self.load(filename, with_rels)

    def has_term(self, term_id):
        return self.resolve_term(term_id) is not None

    def resolve_term(self, term_id: str) -> str | None:
        """Return the current primary ID for a primary, alt or replaced term."""
        seen = set()
        current = term_id
        while current not in seen:
            seen.add(current)
            canonical = self.canonical_ids.get(current)
            if canonical is not None:
                return canonical
            current = self.obsolete_replacements.get(current, "")
            if not current:
                return None
        return None

    def get_term(self, term_id):
        canonical = self.resolve_term(term_id)
        return self.ont.get(canonical) if canonical else None

    def load(self, filename, with_rels):
        objects: dict[str, dict] = {}
        obj = None
        with open_text(filename) as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("data-version:"):
                    continue
                if line == "[Term]":
                    if obj is not None and "id" in obj:
                        objects[obj["id"]] = obj
                    obj = {
                        "is_a": [],
                        "alt_ids": [],
                        "is_obsolete": False,
                        "replaced_by": [],
                        "consider": [],
                    }
                    continue
                if line == "[Typedef]":
                    if obj is not None and "id" in obj:
                        objects[obj["id"]] = obj
                    obj = None
                    continue
                if obj is None or ": " not in line:
                    continue
                key, value = line.split(": ", 1)
                if key == "id":
                    obj["id"] = value
                elif key == "alt_id":
                    obj["alt_ids"].append(value)
                elif key == "namespace":
                    obj["namespace"] = value
                elif key == "is_a":
                    obj["is_a"].append(value.split(" ! ")[0])
                elif with_rels and key == "relationship":
                    fields = value.split()
                    if (
                        len(fields) >= 2
                        and (
                            self.relationship_types is None
                            or fields[0] in self.relationship_types
                        )
                    ):
                        obj["is_a"].append(fields[1])
                elif key == "name":
                    obj["name"] = value
                elif key == "is_obsolete" and value == "true":
                    obj["is_obsolete"] = True
                elif key == "replaced_by":
                    obj["replaced_by"].append(value)
                elif key == "consider":
                    obj["consider"].append(value)
            if obj is not None and "id" in obj:
                objects[obj["id"]] = obj

        ont: dict[str, dict] = {}
        for term_id, term in objects.items():
            if term["is_obsolete"]:
                if len(term["replaced_by"]) == 1:
                    replacement = term["replaced_by"][0]
                    self.obsolete_replacements[term_id] = replacement
                    for alt_id in term["alt_ids"]:
                        self.obsolete_replacements[alt_id] = replacement
                continue
            term["children"] = set()
            ont[term_id] = term
            self.primary_terms.add(term_id)
            self.canonical_ids[term_id] = term_id
            for alt_id in term["alt_ids"]:
                self.canonical_ids[alt_id] = term_id

        # Resolve obsolete replacements after loading primary and alternative IDs.
        for obsolete_id, replacement in list(self.obsolete_replacements.items()):
            canonical = self.canonical_ids.get(replacement)
            if canonical is not None:
                self.obsolete_replacements[obsolete_id] = canonical

        for term_id in sorted(self.primary_terms):
            term = ont[term_id]
            canonical_parents = []
            for parent_id in term["is_a"]:
                parent = self.resolve_term(parent_id)
                if parent is not None and parent in ont:
                    canonical_parents.append(parent)
                    ont[parent]["children"].add(term_id)
            term["is_a"] = canonical_parents

        # Expose alternative GO IDs through the DeepGOPlus-compatible lookup.
        for alias, canonical in self.canonical_ids.items():
            if alias != canonical:
                ont[alias] = ont[canonical]
        return ont

    def get_anchestors(self, term_id):
        canonical = self.resolve_term(term_id)
        if canonical is None:
            return set()
        term_set = set()
        queue = deque([canonical])
        while queue:
            current = queue.popleft()
            if current in term_set:
                continue
            term_set.add(current)
            queue.extend(self.ont[current]["is_a"])
        return term_set

    def get_ancestors(self, term_id):
        return self.get_anchestors(term_id)

    def get_parents(self, term_id):
        canonical = self.resolve_term(term_id)
        if canonical is None:
            return set()
        return set(self.ont[canonical]["is_a"])

    def get_namespace_terms(self, namespace):
        return {
            go_id for go_id in self.primary_terms
            if self.ont[go_id].get("namespace") == namespace
        }

    def get_namespace(self, term_id):
        canonical = self.resolve_term(term_id)
        if canonical is None:
            raise KeyError(term_id)
        return self.ont[canonical]["namespace"]

    def get_term_set(self, term_id):
        canonical = self.resolve_term(term_id)
        if canonical is None:
            return set()
        term_set = set()
        queue = deque([canonical])
        while queue:
            current = queue.popleft()
            if current in term_set:
                continue
            term_set.add(current)
            queue.extend(self.ont[current]["children"])
        return term_set
