from __future__ import annotations

from collections import deque
from pathlib import Path

from .config import ROOT_TERMS
from .inputs import open_text


class Ontology:
    """DeepGOPlus-compatible OBO graph with explicit ID diagnostics."""

    def __init__(
        self,
        path: str | Path,
        include_relationships: bool = True,
        relationship_types: frozenset[str] | None = None,
    ):
        self.path = Path(path)
        self.include_relationships = include_relationships
        self.included_relationship_types = relationship_types
        self.term_metadata: dict[str, dict[str, object]] = {}
        self.alias_to_primary: dict[str, str] = {}
        self.live_terms: set[str] = set()
        self.obsolete_replacements: dict[str, str] = {}
        self.parents: dict[str, tuple[str, ...]] = {}
        self.relationship_types: set[str] = set()
        self._ancestor_cache: dict[tuple[str, bool], frozenset[str]] = {}
        self._load()

    def _load(self) -> None:
        objects: dict[str, dict[str, object]] = {}
        current: dict[str, object] | None = None
        with open_text(self.path) as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    if current and "id" in current:
                        objects[str(current["id"])] = current
                    current = {
                        "alt_ids": [], "parents": [], "relationships": [],
                        "is_obsolete": False, "replaced_by": [], "consider": [],
                    }
                    continue
                if line == "[Typedef]":
                    if current and "id" in current:
                        objects[str(current["id"])] = current
                    current = None
                    continue
                if current is None or ": " not in line:
                    continue
                key, value = line.split(": ", 1)
                if key == "id":
                    current["id"] = value
                elif key == "alt_id":
                    current["alt_ids"].append(value)  # type: ignore[union-attr]
                elif key == "namespace":
                    current["namespace"] = value
                elif key == "is_a":
                    current["parents"].append(value.split(" ! ", 1)[0])  # type: ignore[union-attr]
                elif key == "relationship":
                    parts = value.split()
                    if len(parts) >= 2:
                        self.relationship_types.add(parts[0])
                        current["relationships"].append((parts[0], parts[1]))  # type: ignore[union-attr]
                elif key == "is_obsolete" and value == "true":
                    current["is_obsolete"] = True
                elif key == "replaced_by":
                    current["replaced_by"].append(value)  # type: ignore[union-attr]
                elif key == "consider":
                    current["consider"].append(value)  # type: ignore[union-attr]
            if current and "id" in current:
                objects[str(current["id"])] = current

        self.term_metadata = objects
        for term_id, metadata in objects.items():
            self.alias_to_primary[term_id] = term_id
            for alt_id in metadata["alt_ids"]:  # type: ignore[union-attr]
                self.alias_to_primary[str(alt_id)] = term_id
            if not metadata["is_obsolete"]:
                self.live_terms.add(term_id)

        for term_id, metadata in objects.items():
            if metadata["is_obsolete"] and len(metadata["replaced_by"]) == 1:  # type: ignore[arg-type]
                replacement = str(metadata["replaced_by"][0])  # type: ignore[index]
                self.obsolete_replacements[term_id] = replacement
                for alt_id in metadata["alt_ids"]:  # type: ignore[union-attr]
                    self.obsolete_replacements[str(alt_id)] = replacement

        for term_id in sorted(self.live_terms):
            metadata = objects[term_id]
            raw_parents = list(metadata["parents"])  # type: ignore[arg-type]
            if self.include_relationships:
                raw_parents.extend(
                    target
                    for relation, target in metadata["relationships"]  # type: ignore[union-attr]
                    if (
                        self.included_relationship_types is None
                        or relation in self.included_relationship_types
                    )
                )
            resolved = []
            for raw_parent in raw_parents:
                parent = self.resolve(str(raw_parent))
                if parent and parent in self.live_terms:
                    resolved.append(parent)
            self.parents[term_id] = tuple(sorted(set(resolved)))

    def resolve(self, term_id: str) -> str | None:
        seen: set[str] = set()
        current = term_id
        while current and current not in seen:
            seen.add(current)
            primary = self.alias_to_primary.get(current)
            if primary in self.live_terms:
                return primary
            current = self.obsolete_replacements.get(current, "")
        return None

    def describe(self, term_id: str) -> dict[str, object]:
        primary = self.alias_to_primary.get(term_id, term_id)
        metadata = self.term_metadata.get(primary)
        canonical = self.resolve(term_id)
        if metadata is None:
            status = "unknown"
        elif metadata["is_obsolete"] and canonical:
            status = "obsolete_replaced"
        elif metadata["is_obsolete"]:
            status = "obsolete_unresolved"
        elif term_id != primary:
            status = "alt_id"
        else:
            status = "primary"
        return {
            "status": status,
            "primary_id": primary if metadata else "",
            "canonical_id": canonical or "",
            "replaced_by": tuple(metadata["replaced_by"]) if metadata else (),
            "consider": tuple(metadata["consider"]) if metadata else (),
        }

    def namespace(self, term_id: str) -> str:
        canonical = self.resolve(term_id)
        if canonical is None:
            raise KeyError(term_id)
        return str(self.term_metadata[canonical].get("namespace", ""))

    def ancestors(self, term_id: str, exclude_roots: bool = False) -> set[str]:
        canonical = self.resolve(term_id)
        if canonical is None:
            return set()
        cache_key = (canonical, exclude_roots)
        cached = self._ancestor_cache.get(cache_key)
        if cached is not None:
            # Return a copy to protect the cached ontology closure.
            return set(cached)
        result: set[str] = set()
        queue = deque([canonical])
        while queue:
            current = queue.popleft()
            if current in result:
                continue
            result.add(current)
            queue.extend(self.parents.get(current, ()))
        if exclude_roots:
            result -= ROOT_TERMS
        self._ancestor_cache[cache_key] = frozenset(result)
        return set(result)
