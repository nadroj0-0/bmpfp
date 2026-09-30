from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_UNIREF_LEVELS = (90, 50)
DEFAULT_SENSITIVITY_BY_UNIREF_LEVEL = {90: 7.5, 50: 4.0}


@dataclass(frozen=True)
class UniRefScaffold:
    level: int

    @property
    def display_name(self) -> str:
        return f"UniRef{self.level}"

    @property
    def slug(self) -> str:
        return f"uniref{self.level}"

    @property
    def input_role(self) -> str:
        return f"{self.slug}_fasta"

    @property
    def id_prefix(self) -> str:
        return f"{self.display_name}_"

    @property
    def id_field(self) -> str:
        return f"{self.slug}_id"

    @property
    def source_population(self) -> str:
        return f"{self.slug}-clustering-scaffold"

    @property
    def idmapping_column_index(self) -> int:
        # Headerless idmapping_selected.tab: UniRef90 is column 9 and UniRef50 is 10.
        return {90: 8, 50: 9}[self.level]

    @property
    def recommended_sensitivity(self) -> float:
        return DEFAULT_SENSITIVITY_BY_UNIREF_LEVEL[self.level]


def uniref_scaffold(level: int | str) -> UniRefScaffold:
    numeric = int(level)
    if numeric not in SUPPORTED_UNIREF_LEVELS:
        raise ValueError(
            f"Unsupported UniRef scaffold {level!r}; choose 90 or 50"
        )
    return UniRefScaffold(numeric)
