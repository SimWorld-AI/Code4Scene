"""Metadata-isolated routing from Stage 1 locations into Stage 2 acquisition.

This module is intentionally a planner/adapter, not a visual evaluator.  It
joins Stage 1's decision matches/name grounding and an explicitly supplied
Stage 2 locator retrieval back to the immutable actor inventory, then emits
only live actor ids and bounds for a camera controller.  Actor labels, Unreal
names, asset paths, retrieval terms, scores, and Stage 1 verdicts are never
copied into a route target.

The separate :func:`visual_claim_payload` helper serializes only the
prompt-grounded RequirementGraph node.  A caller can therefore pass that
payload plus RGB across the VLM boundary without passing this routing plan or
the actor inventory.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .actor_inventory import ActorBounds, ActorDescriptor, ActorInventorySnapshot
from .contracts import (
    EntityEvaluationRoute,
    EntityNode,
    JsonSerializable,
    PredicateNode,
    RequirementGraph,
    RequirementNode,
    SceneBounds,
    ScopeEdge,
)
from .semantic_retrieval import QueryIdentityRetrieval, SceneIdentityRetrievalResult
from .stage1 import Stage1Result


class Stage2RouteSource(str, Enum):
    """Acquisition provenance for one entity route.

    These values describe how the controller obtained locations; they are not
    visual evidence and must not be included in a VLM request.
    """

    # Serialized legacy name retained for artifact compatibility. It now means
    # decision-eligible normalized-exact Stage 1 identity, not a heuristic
    # "strict" string matcher. Non-decisive name grounding stays locator-only.
    STRICT_MATCH = "strict_match"
    LOCATOR_ONLY = "locator_only"
    MIXED = "mixed"
    UNLOCALIZED = "unlocalized"


class Stage2RoutingDiagnosticReason(str, Enum):
    """Why a Stage 1/retrieval actor id was not exposed as a camera target."""

    ACTOR_NOT_IN_INVENTORY = "actor_not_in_inventory"
    ACTOR_NOT_ELIGIBLE = "actor_not_eligible"
    ACTOR_NOT_IN_STAGE1_SNAPSHOT = "actor_not_in_stage1_snapshot"
    ACTOR_OUTSIDE_POPULATION = "actor_outside_population"


@dataclass(frozen=True, slots=True)
class Stage2ActorTarget(JsonSerializable):
    """Minimal geometry-only target safe to hand to an acquisition controller."""

    actor_id: str
    bounds: ActorBounds

    def __post_init__(self) -> None:
        actor_id = str(self.actor_id).strip()
        if not actor_id:
            raise ValueError("actor_id must be non-empty")
        if not isinstance(self.bounds, ActorBounds):
            raise TypeError("bounds must be ActorBounds")
        object.__setattr__(self, "actor_id", actor_id)


@dataclass(frozen=True, slots=True)
class Stage2RoutingDiagnostic(JsonSerializable):
    entity_id: str
    actor_id: str
    reason: Stage2RoutingDiagnosticReason | str

    def __post_init__(self) -> None:
        entity_id = str(self.entity_id).strip()
        actor_id = str(self.actor_id).strip()
        if not entity_id or not actor_id:
            raise ValueError("routing diagnostics require entity_id and actor_id")
        reason = (
            self.reason
            if isinstance(self.reason, Stage2RoutingDiagnosticReason)
            else Stage2RoutingDiagnosticReason(str(self.reason).strip().casefold())
        )
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "actor_id", actor_id)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class Stage2EntityRoute(JsonSerializable):
    """Deterministic actor targets for one graph entity."""

    entity_id: str
    source: Stage2RouteSource | str
    targets: tuple[Stage2ActorTarget, ...] = ()
    exact_actor_ids: tuple[str, ...] = ()
    locator_actor_ids: tuple[str, ...] = ()
    stage2_visual_only: bool = False
    omitted_locator_actor_count: int = 0

    def __post_init__(self) -> None:
        entity_id = str(self.entity_id).strip()
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        source = (
            self.source
            if isinstance(self.source, Stage2RouteSource)
            else Stage2RouteSource(str(self.source).strip().casefold())
        )
        targets = tuple(self.targets)
        if any(not isinstance(value, Stage2ActorTarget) for value in targets):
            raise TypeError("targets must contain Stage2ActorTarget values")
        target_ids = [value.actor_id.casefold() for value in targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("route targets must have unique actor ids")
        exact_actor_ids = _ordered_unique_actor_ids(self.exact_actor_ids)
        locator_actor_ids = _ordered_unique_actor_ids(self.locator_actor_ids)
        # Older callers only encoded provenance in ``source``. Infer the new
        # explicit partitions for single-source routes; MIXED must always be
        # authored explicitly because its boundary cannot be reconstructed.
        if not exact_actor_ids and not locator_actor_ids:
            if source is Stage2RouteSource.STRICT_MATCH:
                exact_actor_ids = tuple(value.actor_id for value in targets)
            elif source is Stage2RouteSource.LOCATOR_ONLY:
                locator_actor_ids = tuple(value.actor_id for value in targets)
        target_id_set = set(target_ids)
        if any(value.casefold() not in target_id_set for value in exact_actor_ids):
            raise ValueError("exact_actor_ids must refer to route targets")
        if any(value.casefold() not in target_id_set for value in locator_actor_ids):
            raise ValueError("locator_actor_ids must refer to route targets")
        if {value.casefold() for value in exact_actor_ids}.intersection(
            value.casefold() for value in locator_actor_ids
        ):
            raise ValueError("exact and locator actor ids must be disjoint")
        if not isinstance(self.stage2_visual_only, bool):
            raise TypeError("stage2_visual_only must be a bool")
        omitted = self.omitted_locator_actor_count
        if isinstance(omitted, bool) or not isinstance(omitted, int) or omitted < 0:
            raise ValueError("omitted_locator_actor_count must be non-negative")
        if source is Stage2RouteSource.UNLOCALIZED and targets:
            raise ValueError("an unlocalized route cannot contain targets")
        if source is not Stage2RouteSource.UNLOCALIZED and not targets:
            raise ValueError("a localized route must contain at least one target")
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "exact_actor_ids", exact_actor_ids)
        object.__setattr__(self, "locator_actor_ids", locator_actor_ids)

    @property
    def actor_ids(self) -> tuple[str, ...]:
        return tuple(value.actor_id for value in self.targets)


@dataclass(frozen=True, slots=True)
class Stage2RoutingPlan(JsonSerializable):
    """All Stage 2 entity locations plus fail-closed join diagnostics."""

    routes: tuple[Stage2EntityRoute, ...]
    diagnostics: tuple[Stage2RoutingDiagnostic, ...] = ()
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        routes = tuple(self.routes)
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(value, Stage2EntityRoute) for value in routes):
            raise TypeError("routes must contain Stage2EntityRoute values")
        if any(not isinstance(value, Stage2RoutingDiagnostic) for value in diagnostics):
            raise TypeError("diagnostics must contain Stage2RoutingDiagnostic values")
        route_ids = [value.entity_id for value in routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("routing plan entity ids must be unique")
        if self.schema_version != "1.0":
            raise ValueError("unsupported Stage 2 routing schema version")
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "diagnostics", diagnostics)

    def for_entity(self, entity_id: str) -> Stage2EntityRoute | None:
        key = str(entity_id).strip()
        return next((value for value in self.routes if value.entity_id == key), None)


def _ordered_unique_actor_ids(values: Iterable[str]) -> tuple[str, ...]:
    selected: dict[str, str] = {}
    for value in values:
        actor_id = str(value).strip()
        if actor_id:
            selected.setdefault(actor_id.casefold(), actor_id)
    return tuple(selected.values())


def _retrieval_actor_ids(
    retrieval: QueryIdentityRetrieval | None,
) -> tuple[str, ...]:
    """Return exact locations first, then locator candidates in rank order."""

    if retrieval is None:
        return ()
    exact_ids = (
        actor_id
        for hit in retrieval.exact_matches
        for actor_id in sorted(
            hit.actor_ids, key=lambda value: (value.casefold(), value)
        )
    )
    ranked_candidates = sorted(
        retrieval.locator_candidates,
        key=lambda value: (
            value.rank,
            value.identity_term.casefold(),
            value.identity_term,
        ),
    )
    locator_ids = (
        actor_id
        for candidate in ranked_candidates
        for actor_id in sorted(
            candidate.actor_ids, key=lambda value: (value.casefold(), value)
        )
    )
    return _ordered_unique_actor_ids((*exact_ids, *locator_ids))


def _resolve_target(
    *,
    entity_id: str,
    actor_id: str,
    actors_by_id: dict[str, ActorDescriptor],
    stage1_eligible_ids: frozenset[str],
    scene_bounds: SceneBounds,
) -> tuple[Stage2ActorTarget | None, Stage2RoutingDiagnostic | None]:
    actor = actors_by_id.get(actor_id.casefold())
    if actor is None:
        return None, Stage2RoutingDiagnostic(
            entity_id,
            actor_id,
            Stage2RoutingDiagnosticReason.ACTOR_NOT_IN_INVENTORY,
        )
    if not actor.eligible_for_stage1(scene_bounds) or actor.bounds is None:
        return None, Stage2RoutingDiagnostic(
            entity_id,
            actor.live_actor_id,
            Stage2RoutingDiagnosticReason.ACTOR_NOT_ELIGIBLE,
        )
    if actor.live_actor_id.casefold() not in stage1_eligible_ids:
        return None, Stage2RoutingDiagnostic(
            entity_id,
            actor.live_actor_id,
            Stage2RoutingDiagnosticReason.ACTOR_NOT_IN_STAGE1_SNAPSHOT,
        )
    return Stage2ActorTarget(actor.live_actor_id, actor.bounds), None


def plan_stage2_entity_routes(
    graph: RequirementGraph,
    stage1_result: Stage1Result,
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    *,
    locator_retrieval: SceneIdentityRetrievalResult | None = None,
    stage2_only_entity_ids: Iterable[str] = (),
    max_locator_actors_per_entity: int = 12,
    eligible_actor_ids_by_entity: Mapping[str, Iterable[str]] | None = None,
) -> Stage2RoutingPlan:
    """Build geometry-only Stage 2 targets for every graph entity.

    Decision-eligible Stage 1 matches are retained first.  Non-decisive Stage
    1 name grounding, retrieval exact hits, and retrieval locator candidates
    then augment them, including for entities explicitly routed to visual
    evaluation.  Supplemental actor targets are capped per entity; Stage 1
    decision matches are never discarded by that locator budget.

    Surface and region entities are retained as ``UNLOCALIZED`` routes when no
    explicit retrieval source exists.  That is a deliberate controller signal
    to use overview/grid acquisition rather than silently dropping a Stage 2
    dependency. Stage 1 performs no candidate retrieval.

    ``locator_retrieval`` must be produced explicitly by Stage 2 when Top-K
    semantic localization is wanted.  Omitting it leaves decision-eligible
    Stage 1 matches, non-decisive Stage 1 name grounding, and declared
    assemblies; there is deliberately no implicit semantic-retrieval fallback.
    """

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    if not isinstance(stage1_result, Stage1Result):
        raise TypeError("stage1_result must be a Stage1Result")
    if not isinstance(inventory, ActorInventorySnapshot):
        raise TypeError("inventory must be an ActorInventorySnapshot")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")
    if locator_retrieval is not None and not isinstance(
        locator_retrieval, SceneIdentityRetrievalResult
    ):
        raise TypeError("locator_retrieval must be a SceneIdentityRetrievalResult")
    if (
        isinstance(max_locator_actors_per_entity, bool)
        or not isinstance(max_locator_actors_per_entity, int)
        or max_locator_actors_per_entity < 1
    ):
        raise ValueError("max_locator_actors_per_entity must be a positive integer")

    explicit_stage2_ids = frozenset(
        value
        for value in (str(item).strip() for item in stage2_only_entity_ids)
        if value
    )
    graph_entity_ids = {node.id for node in graph.nodes if isinstance(node, EntityNode)}
    unknown_explicit_ids = sorted(explicit_stage2_ids - graph_entity_ids)
    if unknown_explicit_ids:
        raise ValueError(
            "stage2_only_entity_ids contains unknown graph entities: "
            f"{unknown_explicit_ids!r}"
        )

    actors_by_id = {actor.live_actor_id.casefold(): actor for actor in inventory.actors}
    stage1_eligible_ids = frozenset(
        value.casefold() for value in stage1_result.eligible_actor_ids
    )
    scoped_actor_ids = {
        str(entity_id): frozenset(str(value).casefold() for value in actor_ids)
        for entity_id, actor_ids in (eligible_actor_ids_by_entity or {}).items()
    }

    routes: list[Stage2EntityRoute] = []
    diagnostics: list[Stage2RoutingDiagnostic] = []
    entities = sorted(
        (node for node in graph.nodes if isinstance(node, EntityNode)),
        key=lambda value: (value.id.casefold(), value.id),
    )
    for entity in entities:
        allowed_ids = scoped_actor_ids.get(entity.id)
        stage2_visual_only = (
            entity.evaluation_route is EntityEvaluationRoute.STAGE2_VISUAL
            or entity.id in explicit_stage2_ids
        )
        assessment = stage1_result.for_entity(entity.id)
        strict_ids: tuple[str, ...] = ()
        if assessment is not None and assessment.stage1_decision_applicable:
            strict_ids = _ordered_unique_actor_ids(
                (
                    *assessment.matched_actor_ids,
                    *(
                        actor_id
                        for match in assessment.assembly_matches
                        for actor_id in match.member_actor_ids
                    ),
                )
            )
        strict_id_keys = {value.casefold() for value in strict_ids}
        name_grounded_ids = (
            _ordered_unique_actor_ids(
                actor_id
                for actor_id in assessment.grounded_actor_ids
                if actor_id.casefold() not in strict_id_keys
            )
            if assessment is not None
            else ()
        )
        if allowed_ids is not None:
            strict_ids = tuple(
                value for value in strict_ids if value.casefold() in allowed_ids
            )
            name_grounded_ids = tuple(
                value
                for value in name_grounded_ids
                if value.casefold() in allowed_ids
            )

        retrieval = (
            locator_retrieval.for_query(entity.id)
            if locator_retrieval is not None
            else None
        )
        strict_id_keys = {value.casefold() for value in strict_ids}
        supplemental_ids = _ordered_unique_actor_ids(
            (
                *name_grounded_ids,
                *(
                    actor_id
                    for actor_id in _retrieval_actor_ids(retrieval)
                    if actor_id.casefold() not in strict_id_keys
                    and (
                        allowed_ids is None
                        or actor_id.casefold() in allowed_ids
                    )
                ),
            )
        )

        strict_targets: list[Stage2ActorTarget] = []
        seen_target_ids: set[str] = set()
        for actor_id in strict_ids:
            target, diagnostic = _resolve_target(
                entity_id=entity.id,
                actor_id=actor_id,
                actors_by_id=actors_by_id,
                stage1_eligible_ids=stage1_eligible_ids,
                scene_bounds=scene_bounds,
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)
            elif (
                target is not None and target.actor_id.casefold() not in seen_target_ids
            ):
                strict_targets.append(target)
                seen_target_ids.add(target.actor_id.casefold())

        valid_supplemental: list[Stage2ActorTarget] = []
        for actor_id in supplemental_ids:
            target, diagnostic = _resolve_target(
                entity_id=entity.id,
                actor_id=actor_id,
                actors_by_id=actors_by_id,
                stage1_eligible_ids=stage1_eligible_ids,
                scene_bounds=scene_bounds,
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)
            elif (
                target is not None and target.actor_id.casefold() not in seen_target_ids
            ):
                valid_supplemental.append(target)
                seen_target_ids.add(target.actor_id.casefold())

        selected_supplemental = valid_supplemental[:max_locator_actors_per_entity]
        targets = (*strict_targets, *selected_supplemental)
        if strict_targets and selected_supplemental:
            source = Stage2RouteSource.MIXED
        elif strict_targets:
            source = Stage2RouteSource.STRICT_MATCH
        elif selected_supplemental:
            source = Stage2RouteSource.LOCATOR_ONLY
        else:
            source = Stage2RouteSource.UNLOCALIZED
        routes.append(
            Stage2EntityRoute(
                entity_id=entity.id,
                source=source,
                targets=targets,
                exact_actor_ids=tuple(value.actor_id for value in strict_targets),
                locator_actor_ids=tuple(
                    value.actor_id for value in selected_supplemental
                ),
                stage2_visual_only=stage2_visual_only,
                omitted_locator_actor_count=max(
                    0,
                    len(valid_supplemental) - max_locator_actors_per_entity,
                ),
            )
        )

    unique_diagnostics: dict[
        tuple[str, str, Stage2RoutingDiagnosticReason], Stage2RoutingDiagnostic
    ] = {}
    for diagnostic in diagnostics:
        key = (
            diagnostic.entity_id,
            diagnostic.actor_id.casefold(),
            diagnostic.reason,
        )
        unique_diagnostics.setdefault(key, diagnostic)
    ordered_diagnostics = tuple(
        sorted(
            unique_diagnostics.values(),
            key=lambda value: (
                value.entity_id.casefold(),
                value.entity_id,
                value.actor_id.casefold(),
                value.actor_id,
                value.reason.value,
            ),
        )
    )
    return Stage2RoutingPlan(tuple(routes), ordered_diagnostics)


def _v2_dsl_payload(
    graph: RequirementGraph,
    predicate: PredicateNode,
) -> dict[str, Any] | None:
    """Project the prompt-grounded v2 IR without controller identifiers."""

    raw = predicate.semantic_parameters
    if str(raw.get("stage0_ir_version") or "") != "2.0":
        return None
    payload: dict[str, Any] = {
        "requirement_type": str(
            raw.get("requirement_type") or predicate.predicate_type.value
        ),
        "relation": raw.get("relation"),
        "quantity": raw.get("quantity"),
        "scope": None,
        "logic": None,
        "qualifiers": [],
    }
    scope = raw.get("scope")
    if isinstance(scope, Mapping):
        scope_entity_id = scope.get("entity_id")
        scope_entity = (
            graph.node(str(scope_entity_id))
            if scope_entity_id is not None
            else None
        )
        payload["scope"] = {
            "quantifier": scope.get("quantifier"),
            "entity_claim_text": (
                scope_entity.text if isinstance(scope_entity, EntityNode) else None
            ),
        }
    logic = raw.get("logic")
    if isinstance(logic, Mapping):
        operand_claims = []
        for node_id in logic.get("requirement_ids") or ():
            try:
                operand = graph.node(str(node_id))
            except KeyError:
                continue
            if isinstance(operand, PredicateNode):
                operand_claims.append(operand.text)
        payload["logic"] = {
            "operator": logic.get("operator"),
            "operand_claims": operand_claims,
        }
    qualifiers = raw.get("qualifiers")
    if isinstance(qualifiers, (list, tuple)):
        payload["qualifiers"] = [
            {
                "kind": value.get("kind"),
                "name": value.get("name"),
                "source_text": value.get("source_text"),
            }
            for value in qualifiers
            if isinstance(value, Mapping)
        ]
    return payload


def _predicate_payload(
    graph: RequirementGraph,
    predicate: PredicateNode,
) -> dict[str, Any]:
    arguments: list[dict[str, Any]] = []
    for edge in graph.arguments_for(predicate.id):
        target = graph.node(edge.target_id)
        if not isinstance(target, EntityNode):  # pragma: no cover - graph invariant
            continue
        arguments.append(
            {
                "role": edge.role,
                "ordinal": edge.ordinal,
                "claim_text": target.text,
                "entity_name": target.name,
            }
        )
    payload: dict[str, Any] = {
        "node_type": "predicate",
        "claim_text": predicate.text,
        "predicate_name": predicate.name,
        "predicate_type": predicate.predicate_type.value,
        "polarity": predicate.polarity.value,
        "arguments": arguments,
    }
    if predicate.constraint is not None:
        payload["constraint"] = {
            "operator": predicate.constraint.operator.value,
            "value": predicate.constraint.value,
            "upper_value": predicate.constraint.upper_value,
        }
    dsl = _v2_dsl_payload(graph, predicate)
    if dsl is not None:
        payload["semantic_dsl"] = dsl
    return payload


def visual_claim_payload(
    graph: RequirementGraph,
    node_id: str,
) -> dict[str, Any]:
    """Serialize prompt-grounded claim semantics without UE/retrieval metadata.

    The payload intentionally has no argument for a routing plan, actor
    inventory, or Stage 1 result.  It is suitable for the semantic VLM boundary
    when paired with RGB frames selected by the controller.
    """

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    node = graph.node(str(node_id).strip())
    if isinstance(node, EntityNode):
        return {
            "node_type": "entity",
            "claim_text": node.text,
            "entity_name": node.name,
        }
    if isinstance(node, PredicateNode):
        payload = _predicate_payload(graph, node)
        scopes = []
        for edge in graph.edges:
            if not isinstance(edge, ScopeEdge) or edge.source_id != node.id:
                continue
            target = graph.node(edge.target_id)
            if isinstance(target, PredicateNode):  # graph validation guarantees it
                scopes.append(_predicate_payload(graph, target))
        if scopes:
            payload["scopes"] = scopes
        return payload
    if isinstance(node, RequirementNode):
        return {"node_type": "requirement", "claim_text": node.text}
    raise TypeError("unsupported RequirementGraph node")  # pragma: no cover


__all__ = [
    "Stage2ActorTarget",
    "Stage2EntityRoute",
    "Stage2RouteSource",
    "Stage2RoutingDiagnostic",
    "Stage2RoutingDiagnosticReason",
    "Stage2RoutingPlan",
    "plan_stage2_entity_routes",
    "visual_claim_payload",
]
