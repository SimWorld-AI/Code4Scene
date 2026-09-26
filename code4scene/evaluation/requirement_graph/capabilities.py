"""Explicit verifier capability matrix for RequirementGraph semantic leaves.

Stage0 owns meaning, never executability.  This matrix is the narrow boundary
that records whether the current verifier stack has an implementation for a
normalized semantic type. Unknown rich semantics remain in the graph with an
``unplanned`` binding status; they are not silently waived or misrouted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .bundle import RequirementStatus
from .contracts import (
    EntityGroundingMode,
    EntityNode,
    PredicateNode,
    PredicateType,
    RequirementGraph,
)

CAPABILITY_MATRIX_VERSION = "2.2.0"

# Rich authoring deliberately preserves semantic shapes that do not have a sound
# deterministic implementation.  These requirements are still executable:
# they use a scene-wide RGB portfolio and the structured Stage3 VLM judge.
# Keeping this list here makes the fallback an explicit capability decision,
# rather than silently pretending that (for example) approximate density is a
# deterministic count.
GLOBAL_VLM_PLANNER_ROUTE = "visual_global_vlm"
GLOBAL_VLM_SEMANTICS = frozenset(
    {
        "quantity",
        "set",
        "spatial_relation",
        "distribution",
        "composition",
        "style_bundle",
        "environment",
        "logic",
        "boundary",
    }
)


@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    semantic_type: str
    status: RequirementStatus
    planner_route: str | None
    verifier_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic_type": self.semantic_type,
            "status": self.status.value,
            "planner_route": self.planner_route,
            "verifier_ids": list(self.verifier_ids),
            "reason": self.reason,
        }


PREDICATE_ROUTES: dict[PredicateType, tuple[str, tuple[str, ...]]] = {
    PredicateType.EXISTENCE: (
        "identity_then_visual",
        ("stage1.inventory_identity", "stage2.object_existence"),
    ),
    PredicateType.ATTRIBUTE: ("visual_atomic", ("stage2.attribute",)),
    PredicateType.MATERIAL: ("visual_atomic", ("stage2.material",)),
    PredicateType.SPATIAL_RELATION: (
        "visual_atomic",
        ("stage2.spatial_relation",),
    ),
    PredicateType.COUNT: ("visual_atomic", ("stage2.count",)),
    PredicateType.ATMOSPHERE: ("visual_global", ("stage2.atmosphere",)),
    PredicateType.SCENE_IDENTITY: (
        "visual_global",
        ("stage2.scene_identity",),
    ),
}

DIRECT_ATOMIC_SEMANTICS = frozenset(
    {
        "presence",
        "attribute",
        "material",
        "scene_identity",
    }
)


def _grounding_entities_for_node(
    graph: RequirementGraph,
    node: PredicateNode,
) -> tuple[EntityNode, ...]:
    entity_ids = {
        edge.target_id for edge in graph.arguments_for(node.id)
    }
    scope = node.semantic_parameters.get("scope")
    if isinstance(scope, Mapping) and scope.get("entity_id") is not None:
        entity_ids.add(str(scope["entity_id"]))
    return tuple(
        entity
        for entity_id in sorted(entity_ids)
        if isinstance((entity := graph.node(entity_id)), EntityNode)
    )


def capability_for_node(
    graph: RequirementGraph,
    node: EntityNode | PredicateNode,
) -> CapabilityDecision:
    if isinstance(node, EntityNode):
        return CapabilityDecision(
            semantic_type="entity_identity",
            status=RequirementStatus.SUPPORTED,
            planner_route="identity_then_visual",
            verifier_ids=("stage1.inventory_identity", "stage2.object_existence"),
            reason="entity identity is supported by the existing evidence ladder",
        )
    if graph.schema_version == "1.0":
        route, verifiers = PREDICATE_ROUTES[node.predicate_type]
        return CapabilityDecision(
            semantic_type=node.predicate_type.value,
            status=RequirementStatus.SUPPORTED,
            planner_route=route,
            verifier_ids=verifiers,
            reason="RequirementGraph 1.0 predicate has an existing atomic verifier route",
        )

    semantic_type = str(
        node.semantic_parameters.get("requirement_type")
        or node.predicate_type.value
    )
    if semantic_type in DIRECT_ATOMIC_SEMANTICS:
        if any(
            entity.effective_grounding_mode
            in {
                EntityGroundingMode.DERIVED_REGION,
                EntityGroundingMode.SCENE_GLOBAL,
            }
            for entity in _grounding_entities_for_node(graph, node)
        ):
            return CapabilityDecision(
                semantic_type=semantic_type,
                status=RequirementStatus.SUPPORTED,
                planner_route=GLOBAL_VLM_PLANNER_ROUTE,
                verifier_ids=(
                    "stage2.scene_overview",
                    "stage3.structured_vlm_requirement_judge",
                ),
                reason=(
                    "the authored entity is a derived/global region and must "
                    "not be blocked on a same-named Candidate Actor"
                ),
            )
        predicate = PREDICATE_ROUTES[node.predicate_type]
        return CapabilityDecision(
            semantic_type=semantic_type,
            status=RequirementStatus.SUPPORTED,
            planner_route=predicate[0],
            verifier_ids=predicate[1],
            reason="semantic shape maps losslessly to an existing atomic route",
        )
    if semantic_type in GLOBAL_VLM_SEMANTICS:
        return CapabilityDecision(
            semantic_type=semantic_type,
            status=RequirementStatus.SUPPORTED,
            planner_route=GLOBAL_VLM_PLANNER_ROUTE,
            verifier_ids=(
                "stage2.scene_overview",
                "stage3.structured_vlm_requirement_judge",
            ),
            reason=(
                "the full semantic payload is non-deterministic but visually "
                "judgeable from a validated multi-view scene overview"
            ),
        )
    return CapabilityDecision(
        semantic_type=semantic_type,
        status=RequirementStatus.UNPLANNED,
        planner_route=None,
        verifier_ids=(),
        reason=(
            "Stage0 preserves this requirement, but the current verifier "
            "stack has no declared capability for its full semantic payload"
        ),
    )


def capability_summary(graph: RequirementGraph) -> dict[str, object]:
    decisions = [
        capability_for_node(graph, graph.node(node_id))
        for node_id in graph.effective_weights()
        if isinstance(graph.node(node_id), (EntityNode, PredicateNode))
    ]
    supported = sum(value.status is RequirementStatus.SUPPORTED for value in decisions)
    return {
        "matrix_version": CAPABILITY_MATRIX_VERSION,
        "requirement_count": len(decisions),
        "supported_count": supported,
        "unplanned_count": len(decisions) - supported,
        "by_semantic_type": {
            semantic_type: {
                "supported": sum(
                    value.semantic_type == semantic_type
                    and value.status is RequirementStatus.SUPPORTED
                    for value in decisions
                ),
                "unplanned": sum(
                    value.semantic_type == semantic_type
                    and value.status is RequirementStatus.UNPLANNED
                    for value in decisions
                ),
            }
            for semantic_type in sorted({value.semantic_type for value in decisions})
        },
    }


__all__ = [
    "CAPABILITY_MATRIX_VERSION",
    "CapabilityDecision",
    "GLOBAL_VLM_PLANNER_ROUTE",
    "capability_for_node",
    "capability_summary",
]
