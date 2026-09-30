from __future__ import annotations

from collections import Counter
from dataclasses import replace
from .models import ClusterInfo, MappingDecision
from .mmseqs import ClusterIndex
from .uniref_scaffold import uniref_scaffold


def connect_proteins_to_clusters(
    decisions: list[MappingDecision], cluster_index: ClusterIndex,
    uniref_level: int = 90,
    direct_uniprot_members: bool = False,
) -> list[MappingDecision]:
    scaffold = uniref_scaffold(uniref_level)
    member_ids = {
        decision.protein_id if direct_uniprot_members else decision.uniref90_id
        for decision in decisions
        if (
            decision.canonical_sequence_available
            if direct_uniprot_members else decision.status == "mapped"
        )
        and (decision.protein_id if direct_uniprot_members else decision.uniref90_id)
    }
    cluster_by_member = cluster_index.clusters_for({
        str(member_id) for member_id in member_ids
    })
    connected: list[MappingDecision] = []
    for decision in decisions:
        cluster_id = ""
        status = decision.status
        detail = decision.detail
        member_id = decision.protein_id if direct_uniprot_members else decision.uniref90_id
        eligible = (
            decision.canonical_sequence_available
            if direct_uniprot_members else decision.status == "mapped"
        )
        if eligible and member_id:
            cluster_id = cluster_by_member.get(member_id, "")
            if cluster_id and direct_uniprot_members:
                status = "mapped"
                detail = (
                    "resolved directly from supplied clique membership; "
                    f"prior_uniref_status={decision.status}"
                )
            elif not cluster_id:
                status = "missing-mmseqs-assignment"
                detail = (
                    "annotated UniProt accession has no supplied clique assignment"
                    if direct_uniprot_members
                    else f"mapped {scaffold.display_name} identifier has no MMseqs2 cluster assignment"
                )
        connected.append(replace(
            decision, mmseqs_cluster_id=cluster_id, status=status, detail=detail,
        ))
    return connected


def retained_cluster_info(
    decisions: list[MappingDecision], cluster_index: ClusterIndex
) -> dict[str, ClusterInfo]:
    labelled_pairs = {
        (decision.mmseqs_cluster_id, decision.protein_id) for decision in decisions
        if (
            decision.status == "mapped"
            and decision.mmseqs_cluster_id
            and decision.canonical_sequence_available
        )
    }
    labelled = Counter(cluster_id for cluster_id, _ in labelled_pairs)
    retained_ids = set(labelled)
    sizes = cluster_index.sizes_for(retained_ids)
    retained: dict[str, ClusterInfo] = {}
    for cluster_id in sorted(labelled):
        retained[cluster_id] = ClusterInfo(
            cluster_id=cluster_id,
            member_count=sizes[cluster_id],
            labelled_protein_count=labelled[cluster_id],
        )
    return retained
