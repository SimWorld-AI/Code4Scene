"""Graph-aware task scheduling and scene-local query construction for Stage 2."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from .capabilities import GLOBAL_VLM_PLANNER_ROUTE, capability_for_node
from .contracts import (
    ClaimVerdict,
    EntityGroundingMode,
    EntityNode,
    Polarity,
    PredicateNode,
    PredicateType,
    RequirementGraph,
    ScopeEdge,
)
from .semantic_retrieval import EntityIdentityQuery
from .stage1 import Stage1Result
from .stage2_contracts import Stage2Task, Stage2TaskArgument, Stage2TaskKind

_PREDICATE_TASK_KINDS: dict[PredicateType, Stage2TaskKind] = {
    PredicateType.EXISTENCE: Stage2TaskKind.OBJECT_EXISTENCE,
    PredicateType.ATTRIBUTE: Stage2TaskKind.ATTRIBUTE,
    PredicateType.MATERIAL: Stage2TaskKind.MATERIAL,
    PredicateType.SPATIAL_RELATION: Stage2TaskKind.SPATIAL_RELATION,
    PredicateType.COUNT: Stage2TaskKind.COUNT,
    PredicateType.ATMOSPHERE: Stage2TaskKind.ATMOSPHERE,
    PredicateType.SCENE_IDENTITY: Stage2TaskKind.SCENE_IDENTITY,
}


def _stage1_existence_is_resolved(
    node: EntityNode | PredicateNode,
    stage1_result: Stage1Result,
) -> bool:
    """Return whether Stage 1 has already made the scored existence decision."""

    if isinstance(node, EntityNode):
        assessment = stage1_result.for_entity(node.id)
    elif node.predicate_type is PredicateType.EXISTENCE:
        assessment = stage1_result.for_predicate(node.id)
    else:
        return False
    if assessment is None or assessment.routed_to_stage2:
        return False
    return assessment.verdict in {ClaimVerdict.MATCH, ClaimVerdict.MISMATCH}


def _direct_arguments(
    graph: RequirementGraph,
    predicate: PredicateNode,
) -> list[Stage2TaskArgument]:
    return [
        Stage2TaskArgument(
            entity_id=edge.target_id,
            role=edge.role,
            ordinal=edge.ordinal,
        )
        for edge in graph.arguments_for(predicate.id)
    ]


def _task_arguments(
    graph: RequirementGraph,
    node: EntityNode | PredicateNode,
) -> tuple[Stage2TaskArgument, ...]:
    if isinstance(node, EntityNode):
        return (Stage2TaskArgument(node.id, "subject", 0),)

    arguments = _direct_arguments(graph, node)
    if node.predicate_type not in {PredicateType.COUNT, PredicateType.LOGIC}:
        return tuple(arguments)

    # COUNT may scope a relation such as "five candles around a statue". LOGIC
    # owns support-only operand predicates. Retain direct dependencies and add
    # novel scoped/operand entities as acquisition anchors so those dependencies
    # can be localized without acquiring their own scoring budget.
    seen_entities = {value.entity_id for value in arguments}
    scope_edges = sorted(
        (
            edge
            for edge in graph.edges
            if isinstance(edge, ScopeEdge) and edge.source_id == node.id
        ),
        key=lambda edge: edge.target_id,
    )
    for scope in scope_edges:
        scoped = graph.node(scope.target_id)
        if not isinstance(scoped, PredicateNode):  # graph invariant
            continue
        for edge in graph.arguments_for(scoped.id):
            if edge.target_id in seen_entities:
                continue
            arguments.append(
                Stage2TaskArgument(
                    entity_id=edge.target_id,
                    role=edge.role,
                    ordinal=edge.ordinal,
                    source_predicate_id=scoped.id,
                )
            )
            seen_entities.add(edge.target_id)
    return tuple(arguments)


def build_stage2_tasks(
    graph: RequirementGraph,
    stage1_result: Stage1Result,
    *,
    skip_node_ids: Collection[str] = (),
) -> tuple[Stage2Task, ...]:
    """Schedule every unresolved scored visual leaf exactly once.

    The scheduler is driven exclusively by ``RequirementGraph.effective_weights``:
    requirement nodes and support-only members therefore cannot acquire an
    independent task or scoring budget.  Stage 1 resolved entity/existence
    leaves are omitted.  Unresolved entities, explicit ``stage2_visual``
    entities, and all scored non-existence predicates become active tasks.
    """

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    if not isinstance(stage1_result, Stage1Result):
        raise TypeError("stage1_result must be a Stage1Result")

    skipped = {str(value) for value in skip_node_ids}
    weights = graph.effective_weights()
    tasks: list[Stage2Task] = []
    for node_id, weight in sorted(
        weights.items(), key=lambda item: (item[0].casefold(), item[0])
    ):
        if node_id in skipped:
            continue
        node = graph.node(node_id)
        if isinstance(node, EntityNode):
            if _stage1_existence_is_resolved(node, stage1_result):
                continue
            kind = Stage2TaskKind.OBJECT_EXISTENCE
            polarity = Polarity.AFFIRMATIVE
        elif isinstance(node, PredicateNode):
            if (
                node.predicate_type is PredicateType.EXISTENCE
                and _stage1_existence_is_resolved(node, stage1_result)
            ):
                continue
            capability = capability_for_node(graph, node)
            kind = (
                Stage2TaskKind.VISUAL_GLOBAL
                if capability.planner_route == GLOBAL_VLM_PLANNER_ROUTE
                else _PREDICATE_TASK_KINDS[node.predicate_type]
            )
            polarity = node.polarity
        else:
            # effective_weights() normally returns only entity/predicate leaves;
            # keep the boundary fail-closed if a future graph schema changes.
            continue
        tasks.append(
            Stage2Task(
                task_id=f"stage2:{node.id}",
                node_id=node.id,
                kind=kind,
                weight=weight,
                polarity=polarity,
                arguments=_task_arguments(graph, node),
            )
        )
    return tuple(tasks)


def build_stage2_queries(
    graph: RequirementGraph,
    tasks: Sequence[Stage2Task],
) -> tuple[EntityIdentityQuery, ...]:
    """Build one de-duplicated query per active Actor-groundable dependency.

    Query terms use the graph canonical name followed by aliases.  This is the
    sole scheduler boundary that consumes aliases: Stage 1 remains canonical
    exact-only, while these queries are locator-only inputs to
    :func:`retrieve_scene_identities`.
    """

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    task_values = tuple(tasks)
    if any(not isinstance(value, Stage2Task) for value in task_values):
        raise TypeError("tasks must contain Stage2Task values")

    dependency_ids: dict[str, None] = {}
    for task in task_values:
        # Reject stale tasks early instead of constructing queries against a
        # graph other than the one from which they were scheduled.
        graph.node(task.node_id)
        for entity_id in task.dependency_entity_ids:
            dependency_ids.setdefault(entity_id, None)

    def grouped_identity_terms(root: EntityNode) -> tuple[str, ...]:
        terms: list[str] = []
        pending = [root.id]
        seen: set[str] = set()
        while pending:
            entity_id = pending.pop(0)
            if entity_id in seen:
                continue
            seen.add(entity_id)
            entity = graph.node(entity_id)
            if not isinstance(entity, EntityNode):
                raise TypeError(
                    f"entity member {entity_id!r} is not an EntityNode"
                )
            terms.extend((entity.name, *entity.aliases))
            pending.extend(entity.member_ids)
        return tuple(terms)

    queries: list[EntityIdentityQuery] = []
    for entity_id in dependency_ids:
        node = graph.node(entity_id)
        if not isinstance(node, EntityNode):  # task contract/graph invariant
            raise TypeError(f"task dependency {entity_id!r} is not an entity")
        if node.effective_grounding_mode not in {
            EntityGroundingMode.ACTOR,
            EntityGroundingMode.ACTOR_COLLECTION,
        }:
            continue
        try:
            query = EntityIdentityQuery(node.id, grouped_identity_terms(node))
        except ValueError:
            # An object name with no normalizable semantic identity cannot use
            # Top-K.  Keeping it out of retrieval deliberately routes the task
            # to deterministic overview/grid fallback instead of fabricating a
            # query or dropping the visual task.
            continue
        queries.append(query)
    return tuple(queries)


__all__ = ["build_stage2_queries", "build_stage2_tasks"]
