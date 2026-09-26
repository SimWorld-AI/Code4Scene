"""Predicate-aware, geometry-only capture planning for graph Stage 2.

This module consumes only graph semantics and the minimal actor-id/AABB routes
produced by :mod:`stage2_routing`.  Retrieval ranks choose which route targets
are considered first, but neither ranks nor geometry can create a semantic
verdict.  All accepted RGB is placed in the shared :class:`FrameStore`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .actions import generate_scout_poses
from .actor_inventory import ActorBounds
from .asset_candidates import (
    assess_bounds_camera_pose,
    plan_bounds_camera_poses,
    plan_bounds_camera_recovery_poses,
)
from .contracts import (
    CameraPose,
    EntityGroundingMode,
    EntityNode,
    PredicateNode,
    ReferentKind,
    RequirementGraph,
    SceneBounds,
)
from .runtime import FrameProvider, RgbFrameHealthError
from .semantic_retrieval import SceneIdentityRetrievalResult
from .stage2_contracts import (
    CaptureProgram,
    CaptureRequest,
    CaptureShotRole,
    SearchCoverage,
    Stage2Budget,
    Stage2CapturePlan,
    Stage2Task,
    Stage2TaskKind,
)
from .stage2_frames import FrameAdmission, FrameRejectionReason, FrameStore
from .stage2_routing import (
    Stage2ActorTarget,
    Stage2RoutingPlan,
)


class CaptureAttemptStatus(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    CAPTURE_ERROR = "capture_error"
    SKIPPED_RECOVERY = "skipped_recovery"


@dataclass(frozen=True, slots=True)
class CaptureAttempt:
    request_id: str
    task_ids: tuple[str, ...]
    shot_role: CaptureShotRole | str
    status: CaptureAttemptStatus | str
    frame_id: str | None = None
    reason: str | None = None
    recovery_step: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shot_role", CaptureShotRole.coerce(self.shot_role))
        status = (
            self.status
            if isinstance(self.status, CaptureAttemptStatus)
            else CaptureAttemptStatus(str(self.status).strip().casefold())
        )
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_ids": list(self.task_ids),
            "shot_role": self.shot_role.value,
            "status": self.status.value,
            "frame_id": self.frame_id,
            "reason": self.reason,
            "recovery_step": self.recovery_step,
        }


@dataclass(frozen=True, slots=True)
class Stage2CaptureExecution:
    attempts: tuple[CaptureAttempt, ...]
    coverage_by_task: Mapping[str, SearchCoverage]
    runtime_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "coverage_by_task", dict(self.coverage_by_task))

    @property
    def attempted_capture_count(self) -> int:
        return sum(
            value.status is not CaptureAttemptStatus.SKIPPED_RECOVERY
            for value in self.attempts
        )

    @property
    def recovery_attempt_count(self) -> int:
        return sum(
            value.recovery_step > 0
            and value.status is not CaptureAttemptStatus.SKIPPED_RECOVERY
            for value in self.attempts
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": [value.to_dict() for value in self.attempts],
            "coverage_by_task": {
                key: value.to_dict() for key, value in self.coverage_by_task.items()
            },
            "runtime_error": self.runtime_error,
            "attempted_capture_count": self.attempted_capture_count,
            "recovery_attempt_count": self.recovery_attempt_count,
        }


def union_actor_bounds(targets: Iterable[Stage2ActorTarget]) -> ActorBounds:
    """Return the AABB union of one or more geometry-only route targets."""

    values = tuple(targets)
    if not values:
        raise ValueError("at least one Stage2ActorTarget is required")
    if any(not isinstance(value, Stage2ActorTarget) for value in values):
        raise TypeError("targets must contain Stage2ActorTarget values")
    minimum = tuple(
        min(value.bounds.min_cm[axis] for value in values) for axis in range(3)
    )
    maximum = tuple(
        max(value.bounds.max_cm[axis] for value in values) for axis in range(3)
    )
    return ActorBounds.from_min_max(minimum, maximum)


def _same_spatial_identity(left: Stage2ActorTarget, right: Stage2ActorTarget) -> bool:
    return (
        math.dist(left.bounds.center_cm, right.bounds.center_cm) <= 5.0
        and math.dist(left.bounds.extent_cm, right.bounds.extent_cm) <= 5.0
    )


def cluster_stage2_targets(
    targets: Iterable[Stage2ActorTarget],
    *,
    limit: int,
    identity_keys_by_actor: Mapping[str, Sequence[str]] | None = None,
    deduplicate_spatial_identity: bool = True,
) -> tuple[Stage2ActorTarget, ...]:
    """Deduplicate geometry and choose spatially diverse representatives.

    Retrieval order supplies the primary representative.  Subsequent choices
    prefer an unseen retrieved identity document, then maximize distance from
    already selected centers.  Identity keys remain acquisition metadata and
    are never copied to capture requests or judge frames.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    unique: list[Stage2ActorTarget] = []
    seen_ids: set[str] = set()
    for target in targets:
        if not isinstance(target, Stage2ActorTarget):
            raise TypeError("targets must contain Stage2ActorTarget values")
        key = target.actor_id.casefold()
        if key in seen_ids or (
            deduplicate_spatial_identity
            and any(_same_spatial_identity(target, value) for value in unique)
        ):
            continue
        seen_ids.add(key)
        unique.append(target)
    if len(unique) <= limit:
        return tuple(unique)

    identities = identity_keys_by_actor or {}
    selected = [unique.pop(0)]
    seen_identity_keys = set(identities.get(selected[0].actor_id, ()))
    while unique and len(selected) < limit:

        def score(value: Stage2ActorTarget) -> tuple[float, float, str]:
            identity_keys = set(identities.get(value.actor_id, ()))
            unseen_identity = float(bool(identity_keys - seen_identity_keys))
            distance = min(
                math.dist(value.bounds.center_cm, chosen.bounds.center_cm)
                for chosen in selected
            )
            return (unseen_identity, distance, value.actor_id)

        winner = max(unique, key=score)
        unique.remove(winner)
        selected.append(winner)
        seen_identity_keys.update(identities.get(winner.actor_id, ()))
    return tuple(selected)


def _identity_keys_for_entity(
    entity_id: str,
    retrieval: SceneIdentityRetrievalResult | None,
) -> dict[str, tuple[str, ...]]:
    result = retrieval.for_query(entity_id) if retrieval is not None else None
    if result is None:
        return {}
    values: dict[str, list[str]] = {}
    for hit in result.exact_matches:
        for actor_id in hit.actor_ids:
            values.setdefault(actor_id, []).append(hit.identity_term)
    for candidate in result.locator_candidates:
        for actor_id in candidate.actor_ids:
            values.setdefault(actor_id, []).append(candidate.identity_term)
    return {key: tuple(dict.fromkeys(items)) for key, items in values.items()}


def _route_targets(
    routes: Stage2RoutingPlan,
    entity_id: str,
    *,
    locator_only: bool = False,
) -> tuple[Stage2ActorTarget, ...]:
    route = routes.for_entity(entity_id)
    if route is None:
        return ()
    if not locator_only:
        return route.targets
    locator_ids = {value.casefold() for value in route.locator_actor_ids}
    return tuple(
        value for value in route.targets if value.actor_id.casefold() in locator_ids
    )


def _representatives_for_entity(
    graph: RequirementGraph,
    routes: Stage2RoutingPlan,
    entity_id: str,
    *,
    budget: Stage2Budget,
    retrieval: SceneIdentityRetrievalResult | None,
    count_anchors: bool = False,
    locator_only: bool = False,
) -> tuple[Stage2ActorTarget, ...]:
    node = graph.node(entity_id)
    if not isinstance(node, EntityNode):
        return ()
    if count_anchors:
        limit = budget.max_count_anchors
    elif node.referent_kind is ReferentKind.INDIVIDUAL:
        limit = budget.max_individual_representatives
    else:
        limit = budget.max_collection_representatives
    return cluster_stage2_targets(
        _route_targets(routes, entity_id, locator_only=locator_only),
        limit=limit,
        identity_keys_by_actor=_identity_keys_for_entity(entity_id, retrieval),
        # Distinct live Actor ids are the count population even when their
        # bounds overlap (crossed swords are a common example). Spatial
        # deduplication remains useful only for representative, non-count
        # acquisition.
        deduplicate_spatial_identity=not count_anchors,
    )


def _is_around_task(graph: RequirementGraph, task: Stage2Task) -> bool:
    if task.kind is not Stage2TaskKind.SPATIAL_RELATION:
        return False
    node = graph.node(task.node_id)
    return isinstance(node, PredicateNode) and node.name.strip().casefold() == "around"


def preferred_shot_roles_for_task(task: Stage2Task) -> tuple[CaptureShotRole, ...]:
    return {
        Stage2TaskKind.OBJECT_EXISTENCE: (
            CaptureShotRole.CONTEXT,
            CaptureShotRole.CLOSE,
            CaptureShotRole.GRID,
        ),
        Stage2TaskKind.ATTRIBUTE: (
            CaptureShotRole.DETAIL,
            CaptureShotRole.CONTEXT,
            CaptureShotRole.CLOSE,
        ),
        Stage2TaskKind.MATERIAL: (
            CaptureShotRole.CLOSE,
            CaptureShotRole.OBLIQUE,
            CaptureShotRole.DETAIL,
        ),
        Stage2TaskKind.SPATIAL_RELATION: (
            CaptureShotRole.JOINT,
            CaptureShotRole.COLLECTION_WIDE,
            CaptureShotRole.GRID,
        ),
        Stage2TaskKind.COUNT: (
            CaptureShotRole.COLLECTION_WIDE,
            CaptureShotRole.OBLIQUE,
            CaptureShotRole.GRID,
        ),
        Stage2TaskKind.ATMOSPHERE: (CaptureShotRole.OVERVIEW, CaptureShotRole.GRID),
        Stage2TaskKind.SCENE_IDENTITY: (CaptureShotRole.OVERVIEW, CaptureShotRole.GRID),
        Stage2TaskKind.VISUAL_GLOBAL: (
            CaptureShotRole.OVERVIEW,
            CaptureShotRole.GRID,
        ),
    }[task.kind]


def _pairs_without_cartesian_product(
    left: Sequence[Stage2ActorTarget],
    right: Sequence[Stage2ActorTarget],
    *,
    limit: int,
) -> tuple[tuple[Stage2ActorTarget, Stage2ActorTarget], ...]:
    """Greedily select nearby pairs without enumerating an A x B product."""

    if not left or not right:
        return ()
    remaining_left = list(left)
    remaining_right = list(right)
    pairs: list[tuple[Stage2ActorTarget, Stage2ActorTarget]] = []
    # Each iteration performs a linear nearest-neighbour lookup for only the
    # next left representative; it never materializes or schedules A x B.
    while remaining_left and remaining_right and len(pairs) < limit:
        subject = remaining_left.pop(0)
        reference = min(
            remaining_right,
            key=lambda value: math.dist(
                subject.bounds.center_cm, value.bounds.center_cm
            ),
        )
        remaining_right.remove(reference)
        pairs.append((subject, reference))
    return tuple(pairs)


def plan_stage2_capture_groups(
    graph: RequirementGraph,
    tasks: Sequence[Stage2Task],
    routes: Stage2RoutingPlan,
    scene_bounds: SceneBounds,
    *,
    budget: Stage2Budget | None = None,
    locator_retrieval: SceneIdentityRetrievalResult | None = None,
    content_floor_z: float | None = None,
    preloaded_overview_count: int = 0,
    overview_poses: Sequence[CameraPose] | None = None,
) -> Stage2CapturePlan:
    """Create shared predicate-aware capture programs.

    Production derives a finite capture plan from the graph: varied overviews
    for global claims, at least one actor-focused view per localizable entity
    group, collection-wide views for sets/counts, joint views for relations,
    and a requirement-owned grid fallback for unlocalized claims.  Explicitly
    constrained callers can still enable aggregate limits; already-admitted
    overview RGB counts against those limits and is not acquired again.
    """

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    if not isinstance(routes, Stage2RoutingPlan):
        raise TypeError("routes must be a Stage2RoutingPlan")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")
    if content_floor_z is not None and not (
        math.isfinite(float(content_floor_z))
        and scene_bounds.min_z <= float(content_floor_z) <= scene_bounds.max_z
    ):
        raise ValueError("content_floor_z must be finite and inside scene bounds")
    if (
        isinstance(preloaded_overview_count, bool)
        or not isinstance(preloaded_overview_count, int)
        or preloaded_overview_count < 0
    ):
        raise ValueError("preloaded_overview_count must be a non-negative integer")
    supplied_overviews = None if overview_poses is None else tuple(overview_poses)
    if supplied_overviews is not None and any(
        not isinstance(value, CameraPose) for value in supplied_overviews
    ):
        raise TypeError("overview_poses must contain CameraPose values")
    budget = Stage2Budget() if budget is None else budget
    if not isinstance(budget, Stage2Budget):
        raise TypeError("budget must be a Stage2Budget")
    task_values = tuple(tasks)
    if any(not isinstance(value, Stage2Task) for value in task_values):
        raise TypeError("tasks must contain Stage2Task values")
    if not task_values:
        return Stage2CapturePlan((), budget)

    overview_task_ids = tuple(
        value.task_id
        for value in task_values
        if value.allows_overview_fallback
    )

    request_number = 1
    program_number = 1
    overview_count = (
        min(preloaded_overview_count, budget.max_overview_frames)
        if overview_task_ids
        else 0
    )
    targeted_count = 0
    recovery_count = 0
    programs: list[CaptureProgram] = []
    targeted_tasks: set[str] = set()
    overview_program_index: int | None = None

    def new_request(
        task_ids: Sequence[str],
        role: CaptureShotRole,
        pose: Any,
        *,
        actor_ids: Sequence[str] = (),
        target_bounds: ActorBounds | None = None,
        representative_index: int = 0,
        recovery_step: int = 0,
    ) -> CaptureRequest | None:
        nonlocal request_number, overview_count, targeted_count, recovery_count
        # Reject invalid geometry before consuming acquisition budget.  Global
        # views must not sit below the inventory-derived content floor; target
        # views must put the requested AABB in front of the camera with a useful
        # viewport footprint.  Occlusion remains a visual-judge concern.
        if (
            target_bounds is None
            and content_floor_z is not None
            and pose.z < float(content_floor_z) + 20.0
        ):
            return None
        if target_bounds is not None and not assess_bounds_camera_pose(
            target_bounds.center_cm,
            target_bounds.extent_cm,
            pose,
        ).healthy:
            return None
        if recovery_step:
            if (
                budget.global_capture_limits_enabled
                and recovery_count >= budget.max_recovery_frames
            ):
                return None
            recovery_count += 1
        elif role in {CaptureShotRole.OVERVIEW, CaptureShotRole.GRID}:
            if budget.global_capture_limits_enabled and (
                overview_count >= budget.max_overview_frames
                or overview_count + targeted_count >= budget.max_valid_frames
            ):
                return None
            overview_count += 1
        else:
            if budget.global_capture_limits_enabled and (
                targeted_count >= budget.max_targeted_frames
                or overview_count + targeted_count >= budget.max_valid_frames
            ):
                return None
            targeted_count += 1
        value = CaptureRequest(
            request_id=f"s2c_{request_number:06d}",
            task_ids=tuple(task_ids),
            shot_role=role,
            pose=pose,
            actor_ids=tuple(actor_ids),
            representative_index=representative_index,
            recovery_step=recovery_step,
        )
        request_number += 1
        return value

    def add_program(
        task_ids: Sequence[str], requests: Iterable[CaptureRequest | None]
    ) -> None:
        nonlocal program_number
        selected = tuple(value for value in requests if value is not None)
        if not selected:
            return
        programs.append(
            CaptureProgram(
                program_id=f"s2p_{program_number:04d}",
                task_ids=tuple(task_ids),
                requests=selected,
            )
        )
        program_number += 1
        for request in selected:
            if request.recovery_step == 0 and request.shot_role not in {
                CaptureShotRole.OVERVIEW,
                CaptureShotRole.GRID,
            }:
                targeted_tasks.update(request.task_ids)

    def append_to_program(
        program_index: int | None,
        requests: Iterable[CaptureRequest | None],
    ) -> None:
        """Append delayed recovery requests without splitting their program.

        Health failures are counted and consumed within one CaptureProgram, so
        a recovery request must stay beside the primary view it can replace.
        Overview/grid recovery is used only by an explicitly unconstrained
        legacy profile, where there is no scene-wide recovery quota.
        """

        if program_index is None:
            return
        selected = tuple(value for value in requests if value is not None)
        if not selected:
            return
        program = programs[program_index]
        programs[program_index] = CaptureProgram(
            program.program_id,
            program.task_ids,
            (*program.requests, *selected),
        )

    scout_poses = (
        supplied_overviews
        if supplied_overviews
        else generate_scout_poses(scene_bounds, floor_z=content_floor_z)
    )
    remaining_initial_overviews = (
        max(
            0,
            min(
                budget.overview_frames - overview_count,
                budget.max_overview_frames - overview_count,
            ),
        )
        if overview_task_ids
        else 0
    )
    overview_requests = [
        new_request(overview_task_ids, CaptureShotRole.OVERVIEW, pose)
        for pose in scout_poses[:remaining_initial_overviews]
    ]
    overview_program_candidate = len(programs)
    add_program(overview_task_ids, overview_requests)
    if len(programs) > overview_program_candidate:
        overview_program_index = overview_program_candidate

    # First retain the semantic entity grouping used by the RequirementGraph.
    # Different visible facts may deliberately own different entity nodes even
    # when Stage 1 resolves all of them to the same live Candidate Actor.
    entity_groups: dict[str, list[Stage2Task]] = {}
    for task in task_values:
        for argument in task.arguments:
            entity = graph.node(argument.entity_id)
            if (
                not isinstance(entity, EntityNode)
                or entity.effective_grounding_mode
                not in {
                    EntityGroundingMode.ACTOR,
                    EntityGroundingMode.ACTOR_COLLECTION,
                }
            ):
                continue
            route = routes.for_entity(entity.id)
            needs_identity = bool(route and route.locator_actor_ids)
            needs_visual_facet = task.kind in {
                Stage2TaskKind.OBJECT_EXISTENCE,
                Stage2TaskKind.ATTRIBUTE,
                Stage2TaskKind.MATERIAL,
            }
            if not needs_identity and not needs_visual_facet:
                continue
            values = entity_groups.setdefault(entity.id, [])
            if task not in values:
                values.append(task)
    identity_search_by_entity = {
        entity_id: (
            bool(
                (route := routes.for_entity(entity_id))
                and route.locator_actor_ids
            )
            and (
                not route.exact_actor_ids
                or any(
                    task.kind
                    in {
                        Stage2TaskKind.OBJECT_EXISTENCE,
                        Stage2TaskKind.COUNT,
                        Stage2TaskKind.SPATIAL_RELATION,
                    }
                    for task in entity_groups[entity_id]
                )
            )
        )
        for entity_id in entity_groups
    }
    representatives_by_entity = {
        entity_id: _representatives_for_entity(
            graph,
            routes,
            entity_id,
            budget=budget,
            retrieval=locator_retrieval,
            locator_only=identity_search_by_entity[entity_id],
        )
        for entity_id in entity_groups
    }

    # Acquisition ownership is actor-based, not graph-node-based.  Coalesce
    # semantically separate entity groups only when they resolve to the same
    # complete representative Actor set and have the same referent scope.  The
    # resulting requests carry every original task id, so FrameStore can reuse
    # the pixels while Stage 3 still judges each claim independently.
    target_groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for entity_id in entity_groups:
        representatives = representatives_by_entity[entity_id]
        if not representatives:
            continue
        node = graph.node(entity_id)
        referent_scope = (
            "collection"
            if isinstance(node, EntityNode)
            and node.effective_grounding_mode
            is EntityGroundingMode.ACTOR_COLLECTION
            else "individual"
        )
        actor_key = tuple(
            sorted(value.actor_id.casefold() for value in representatives)
        )
        target_groups.setdefault((referent_scope, actor_key), []).append(entity_id)

    localized_target_keys = tuple(target_groups)
    extra_entity_view_count = (
        max(
            0,
            min(
                budget.max_targeted_frames - len(localized_target_keys),
                budget.max_valid_frames
                - overview_count
                - len(localized_target_keys),
            ),
        )
    )
    extra_view_target_keys = frozenset(
        localized_target_keys[:extra_entity_view_count]
    )

    for target_key, entity_ids in target_groups.items():
        entity_id = entity_ids[0]
        # One multi-entity task can appear in several entity groups.  When
        # those entities resolve to the same Actor set, flattening the groups
        # must retain that task once: CaptureRequest deliberately rejects
        # duplicate ownership ids, and one task still needs only one copy of
        # the shared pixels.
        grouped: list[Stage2Task] = []
        for grouped_entity_id in entity_ids:
            for task in entity_groups[grouped_entity_id]:
                if task not in grouped:
                    grouped.append(task)
        task_ids = tuple(value.task_id for value in grouped)
        representatives = representatives_by_entity[entity_id]
        node = graph.node(entity_id)
        is_collection = (
            isinstance(node, EntityNode)
            and node.effective_grounding_mode
            is EntityGroundingMode.ACTOR_COLLECTION
        )
        needs_attribute = any(
            value.kind is Stage2TaskKind.ATTRIBUTE for value in grouped
        )
        needs_material = any(value.kind is Stage2TaskKind.MATERIAL for value in grouped)
        requests: list[CaptureRequest | None] = []
        primary = representatives[0]
        context, close, alternate = plan_bounds_camera_poses(
            primary.bounds.center_cm,
            primary.bounds.extent_cm,
            scene_bounds,
        )
        # A locator-only route is an identity hypothesis, not evidence.  Its
        # first image must therefore be the close actor-focus view: spending a
        # one-view budget on the wider context pose leaves the VLM trying to
        # distinguish the prompt noun from surrounding clutter.  Exact Stage 1
        # routes retain context-first framing for non-identity visual facets.
        locator_identity_confirmation = (
            any(identity_search_by_entity[value] for value in entity_ids)
        )
        primary_role = (
            CaptureShotRole.CLOSE
            if locator_identity_confirmation
            else CaptureShotRole.CONTEXT
        )
        primary_pose = close if locator_identity_confirmation else context
        requests.append(
            new_request(
                task_ids,
                primary_role,
                primary_pose,
                actor_ids=(primary.actor_id,),
                target_bounds=primary.bounds,
            )
        )
        allow_extra_view = (
            not budget.global_capture_limits_enabled
            or target_key in extra_view_target_keys
        )
        has_shape_specific_task = any(
            value.kind in {Stage2TaskKind.COUNT, Stage2TaskKind.SPATIAL_RELATION}
            for value in grouped
        )
        group_view_limit = (
            budget.max_targeted_views_per_entity_group
            if not budget.global_capture_limits_enabled
            else 1
            if has_shape_specific_task
            else 1 + int(allow_extra_view)
        )
        next_collection_rep_index = 1
        if group_view_limit >= 2:
            if is_collection and len(representatives) > 1:
                # A collection is not adequately represented by unrelated
                # member close-ups.  Give it one union-AABB frame so the judge
                # can inspect set extent, layout, and approximate completeness.
                collection_bounds = union_actor_bounds(representatives)
                second_pose = plan_bounds_camera_poses(
                    collection_bounds.center_cm,
                    collection_bounds.extent_cm,
                    scene_bounds,
                )[0]
                requests.append(
                    new_request(
                        task_ids,
                        CaptureShotRole.COLLECTION_WIDE,
                        second_pose,
                        actor_ids=tuple(
                            value.actor_id for value in representatives
                        ),
                        target_bounds=collection_bounds,
                        representative_index=1,
                    )
                )
                next_collection_rep_index = len(representatives)
            else:
                second_target = primary
                if needs_material:
                    second_pose = alternate
                    second_role = CaptureShotRole.OBLIQUE
                elif locator_identity_confirmation:
                    # The locator-only primary already consumed ``close``.
                    # Repeating that exact pose wastes a UE capture and adds no
                    # evidence; use an alternate detail view for attributes or
                    # wider context for existence confirmation.
                    second_pose = alternate if needs_attribute else context
                    second_role = (
                        CaptureShotRole.DETAIL
                        if needs_attribute
                        else CaptureShotRole.CONTEXT
                    )
                else:
                    second_pose = close
                    second_role = (
                        CaptureShotRole.DETAIL
                        if needs_attribute
                        else CaptureShotRole.CLOSE
                    )
                second_index = 0
                requests.append(
                    new_request(
                        task_ids,
                        second_role,
                        second_pose,
                        actor_ids=(second_target.actor_id,),
                        target_bounds=second_target.bounds,
                        representative_index=second_index,
                    )
                )
        # Collections use spatially diverse actors only after every localized
        # entity group has received its primary focus view.
        if is_collection:
            remaining_group_views = max(
                0,
                group_view_limit - sum(value is not None for value in requests),
            )
            for index, target in enumerate(
                representatives[
                    next_collection_rep_index :
                    next_collection_rep_index + remaining_group_views
                ],
                start=next_collection_rep_index,
            ):
                rep_context = plan_bounds_camera_poses(
                    target.bounds.center_cm,
                    target.bounds.extent_cm,
                    scene_bounds,
                )[0]
                requests.append(
                    new_request(
                        task_ids,
                        CaptureShotRole.CONTEXT,
                        rep_context,
                        actor_ids=(target.actor_id,),
                        target_bounds=target.bounds,
                        representative_index=index,
                    )
                )
        # Do not create a recovery-only program after the targeted budget has
        # already rejected every normal request in this group.
        if any(value is not None for value in requests):
            # The material primary already uses ``alternate`` for its oblique
            # view, so retry it from ``close`` instead.  Other facets use
            # context+close primaries and recover from the third, alternate
            # direction.
            recovery_pose = close if needs_material else alternate
            requests.append(
                new_request(
                    task_ids,
                    CaptureShotRole.RECOVERY,
                    recovery_pose,
                    actor_ids=(primary.actor_id,),
                    target_bounds=primary.bounds,
                    recovery_step=1,
                )
            )
            if not is_collection and len(representatives) > 1:
                fallback = representatives[1]
                fallback_pose = plan_bounds_camera_poses(
                    fallback.bounds.center_cm,
                    fallback.bounds.extent_cm,
                    scene_bounds,
                )[0]
                requests.append(
                    new_request(
                        task_ids,
                        CaptureShotRole.RECOVERY,
                        fallback_pose,
                        actor_ids=(fallback.actor_id,),
                        target_bounds=fallback.bounds,
                        representative_index=1,
                        recovery_step=2,
                    )
                )
        add_program(task_ids, requests)

    # Count and around use a collection union, never actor/ISM multiplicity.
    collection_tasks = tuple(
        value
        for value in task_values
        if value.kind is Stage2TaskKind.COUNT or _is_around_task(graph, value)
    )
    for task in collection_tasks:
        anchors: list[Stage2ActorTarget] = []
        for entity_id in task.dependency_entity_ids:
            anchors.extend(
                _representatives_for_entity(
                    graph,
                    routes,
                    entity_id,
                    budget=budget,
                    retrieval=locator_retrieval,
                    count_anchors=True,
                )
            )
        anchors = (
            list(
                cluster_stage2_targets(
                    anchors,
                    limit=budget.max_count_anchors,
                    deduplicate_spatial_identity=False,
                )
            )
            if anchors
            else []
        )
        if not anchors:
            continue
        bounds = union_actor_bounds(anchors)
        # Low collections such as chairs need a modest overhead angle; a
        # centered horizontal camera otherwise looks through table edges or
        # chair undersides.  Elevated shelf/wall targets need the opposite:
        # retain a frontal context candidate so shelf beams do not occlude the
        # contents.  The inventory-derived content floor separates the cases
        # without relying on semantic category names.
        minimum_context_elevation = (
            12.0
            if content_floor_z is not None
            and bounds.min_cm[2] <= float(content_floor_z) + 20.0
            else 0.0
        )
        wide, close, oblique = plan_bounds_camera_poses(
            bounds.center_cm,
            bounds.extent_cm,
            scene_bounds,
            minimum_context_elevation_degrees=minimum_context_elevation,
        )
        actor_ids = tuple(value.actor_id for value in anchors)
        add_program(
            (task.task_id,),
            (
                new_request(
                    (task.task_id,),
                    CaptureShotRole.COLLECTION_WIDE,
                    wide,
                    actor_ids=actor_ids,
                    target_bounds=bounds,
                ),
                new_request(
                    (task.task_id,),
                    CaptureShotRole.OBLIQUE,
                    oblique,
                    actor_ids=actor_ids,
                    target_bounds=bounds,
                ),
                new_request(
                    (task.task_id,),
                    CaptureShotRole.RECOVERY,
                    close,
                    actor_ids=actor_ids,
                    target_bounds=bounds,
                    recovery_step=1,
                )
                if close not in {wide, oblique}
                else None,
            ),
        )

    # Other relations select only one-to-one nearest representative pairs and
    # produce at most max_relation_joint_views total views. Atomic predicates
    # over the same ordered participant pair share those pixels: splitting
    # "winding through" from "divides" must not repeat the camera sweep.
    relation_groups: dict[
        tuple[tuple[str, str], tuple[str, str]], list[Stage2Task]
    ] = {}
    for task in task_values:
        if task.kind is not Stage2TaskKind.SPATIAL_RELATION or _is_around_task(
            graph, task
        ):
            continue
        if len(task.arguments) < 2:
            continue
        pair_key = tuple(
            (argument.role, argument.entity_id)
            for argument in task.arguments[:2]
        )
        relation_groups.setdefault(pair_key, []).append(task)

    for grouped_tasks in relation_groups.values():
        task = grouped_tasks[0]
        task_ids = tuple(value.task_id for value in grouped_tasks)
        left = _representatives_for_entity(
            graph,
            routes,
            task.arguments[0].entity_id,
            budget=budget,
            retrieval=locator_retrieval,
        )
        right = _representatives_for_entity(
            graph,
            routes,
            task.arguments[1].entity_id,
            budget=budget,
            retrieval=locator_retrieval,
        )
        pairs = _pairs_without_cartesian_product(
            left, right, limit=budget.max_relation_joint_views
        )
        requests: list[CaptureRequest | None] = []
        recovery_requests: list[
            tuple[CameraPose, tuple[str, ...], ActorBounds, int]
        ] = []
        remaining_views = budget.max_relation_joint_views
        for pair_index, pair in enumerate(pairs):
            bounds = union_actor_bounds(pair)
            pair_actor_ids = tuple(
                dict.fromkeys(value.actor_id for value in pair)
            )
            context, close, alternate = plan_bounds_camera_poses(
                bounds.center_cm, bounds.extent_cm, scene_bounds
            )
            scheduled_poses: list[CameraPose] = []
            for pose in (context, alternate):
                if remaining_views <= 0:
                    break
                requests.append(
                    new_request(
                        task_ids,
                        CaptureShotRole.JOINT,
                        pose,
                        actor_ids=pair_actor_ids,
                        target_bounds=bounds,
                        representative_index=pair_index,
                    )
                )
                scheduled_poses.append(pose)
                remaining_views -= 1
            if scheduled_poses:
                recovery_pose = (
                    close
                    if close not in scheduled_poses
                    else next(
                        (
                            pose
                            for pose in (alternate, context)
                            if pose not in scheduled_poses
                        ),
                        None,
                    )
                )
                if recovery_pose is not None:
                    recovery_requests.append(
                        (
                            recovery_pose,
                            pair_actor_ids,
                            bounds,
                            pair_index,
                        )
                    )
        for recovery_step, (pose, actor_ids, bounds, pair_index) in enumerate(
            recovery_requests[: budget.max_recovery_frames],
            start=1,
        ):
            requests.append(
                new_request(
                    task_ids,
                    CaptureShotRole.RECOVERY,
                    pose,
                    actor_ids=actor_ids,
                    target_bounds=bounds,
                    representative_index=pair_index,
                    recovery_step=recovery_step,
                )
            )
        add_program(task_ids, requests)

    # Overview association is useful context, but it is not targeted coverage.
    # When the bounded profile has room, give an unlocalized task its own grid
    # search frame.  Never associate one generic grid sweep with every missing
    # object requirement: that made unrelated claims receive an identical VLM
    # input while pretending each target had been searched.
    fallback_tasks = tuple(
        task
        for task in task_values
        if not task.allows_overview_fallback
        and task.task_id not in targeted_tasks
    )
    remaining_overview_slots = (
        min(
            budget.max_overview_frames - overview_count,
            budget.max_valid_frames - overview_count - targeted_count,
        )
        if budget.global_capture_limits_enabled
        else len(fallback_tasks)
    )
    grid_count = min(len(fallback_tasks), max(0, remaining_overview_slots))
    if grid_count:
        grid_poses = tuple(
            scout_poses[(overview_count + index) % len(scout_poses)]
            for index in range(grid_count)
        )
        for task, pose in zip(fallback_tasks[:grid_count], grid_poses, strict=True):
            add_program(
                (task.task_id,),
                (
                    new_request(
                        (task.task_id,),
                        CaptureShotRole.GRID,
                        pose,
                    ),
                ),
            )

    # An explicitly unconstrained legacy profile has no scene-wide recovery
    # quota, while this program still has at most the two unused opposing scout
    # directions and executes them only after a failed primary capture.  The
    # production profile keeps its recovery slot for localized programs.
    overview_primary_poses = tuple(
        value.pose for value in overview_requests if value is not None
    )
    overview_alternates = tuple(
        pose for pose in scout_poses if pose not in overview_primary_poses
    )[: budget.max_recovery_frames]
    if not budget.global_capture_limits_enabled:
        append_to_program(
            overview_program_index,
            (
                new_request(
                    overview_task_ids,
                    CaptureShotRole.RECOVERY,
                    pose,
                    recovery_step=step,
                )
                for step, pose in enumerate(overview_alternates, start=1)
            ),
        )

    return Stage2CapturePlan(tuple(programs), budget)


def _attempt_from_admission(
    request: CaptureRequest,
    admission: FrameAdmission,
) -> CaptureAttempt:
    if admission.accepted and admission.record is not None:
        return CaptureAttempt(
            request.request_id,
            request.task_ids,
            request.shot_role,
            CaptureAttemptStatus.ACCEPTED,
            frame_id=admission.record.frame_id,
            recovery_step=request.recovery_step,
        )
    if admission.rejection_reason is FrameRejectionReason.NEAR_DUPLICATE:
        return CaptureAttempt(
            request.request_id,
            request.task_ids,
            request.shot_role,
            CaptureAttemptStatus.DUPLICATE,
            frame_id=admission.duplicate_of_frame_id,
            reason=admission.rejection_reason.value,
            recovery_step=request.recovery_step,
        )
    reason = (
        admission.rejection_reason.value if admission.rejection_reason else "rejected"
    )
    return CaptureAttempt(
        request.request_id,
        request.task_ids,
        request.shot_role,
        CaptureAttemptStatus.REJECTED,
        reason=reason,
        recovery_step=request.recovery_step,
    )


def _capture_coverage_for_task(task_id: str, frame_store: FrameStore) -> SearchCoverage:
    records = tuple(value for value in frame_store.records if task_id in value.task_ids)
    overview = tuple(
        value
        for value in records
        if value.shot_role in {CaptureShotRole.OVERVIEW, CaptureShotRole.GRID}
    )
    targeted = tuple(
        value
        for value in records
        if value.shot_role not in {CaptureShotRole.OVERVIEW, CaptureShotRole.GRID}
    )
    return SearchCoverage(
        valid_target_views=len(targeted),
        distinct_target_actors=len(
            {actor_id for value in targeted for actor_id in value.actor_ids}
        ),
        complementary_overviews=len(overview),
        has_joint_view=any(
            value.shot_role is CaptureShotRole.JOINT for value in records
        ),
        has_collection_wide_view=any(
            value.shot_role is CaptureShotRole.COLLECTION_WIDE for value in records
        ),
        has_detail_view=any(
            value.shot_role
            in {CaptureShotRole.CLOSE, CaptureShotRole.DETAIL, CaptureShotRole.OBLIQUE}
            for value in records
        ),
        # Top-K, grid, and any finite set of ordinary views are never an
        # exhaustive existence search in schema 1.0.
        exhaustive_existence_search=False,
    )


_RGB_HEALTH_REASONS = frozenset(
    {
        "camera_below_content_floor",
        "camera_pose_unresolved",
        FrameRejectionReason.NEAR_CONSTANT.value,
        FrameRejectionReason.LOW_SPATIAL_DETAIL.value,
        FrameRejectionReason.SEVERELY_UNDEREXPOSED.value,
        FrameRejectionReason.SEVERELY_OVEREXPOSED.value,
    }
)


def _resolve_capture_poses(
    provider: FrameProvider,
    requests: Sequence[CaptureRequest],
) -> tuple[CameraPose | None, ...]:
    """Use an optional live-geometry solver without changing mock providers."""

    values = tuple(requests)
    resolver = getattr(provider, "resolve_camera_poses", None)
    if not callable(resolver):
        return tuple(value.pose for value in values)
    resolved = tuple(
        resolver(
            tuple(value.pose for value in values),
            actor_ids_by_pose=tuple(value.actor_ids for value in values),
        )
    )
    if len(resolved) != len(values) or any(
        value is not None and not isinstance(value, CameraPose)
        for value in resolved
    ):
        raise RuntimeError(
            "camera pose resolver must return one CameraPose or None per request"
        )
    scene_bounds = getattr(getattr(provider, "scene", None), "scene_bounds", None)
    if isinstance(scene_bounds, SceneBounds) and any(value is None for value in resolved):
        inventory = getattr(getattr(provider, "scene", None), "inventory", None)
        actors = tuple(getattr(inventory, "actors", ()) or ())
        bounds_by_actor_id = {
            str(getattr(actor, "live_actor_id", "")).casefold(): getattr(
                actor, "bounds", None
            )
            for actor in actors
            if str(getattr(actor, "live_actor_id", "")).strip()
            and isinstance(getattr(actor, "bounds", None), ActorBounds)
        }
        recovered = list(resolved)
        for index, (request, current) in enumerate(
            zip(values, resolved, strict=True)
        ):
            request_bounds = tuple(
                bounds_by_actor_id[actor_id.casefold()]
                for actor_id in request.actor_ids
                if actor_id.casefold() in bounds_by_actor_id
            )
            if current is not None or not request_bounds:
                continue
            target_bounds = ActorBounds.from_min_max(
                tuple(
                    min(bounds.min_cm[axis] for bounds in request_bounds)
                    for axis in range(3)
                ),
                tuple(
                    max(bounds.max_cm[axis] for bounds in request_bounds)
                    for axis in range(3)
                ),
            )
            candidates = tuple(
                pose
                for pose in plan_bounds_camera_recovery_poses(
                    target_bounds.center_cm,
                    target_bounds.extent_cm,
                    scene_bounds,
                )
                if pose != request.pose
            )
            if not candidates:
                continue
            alternatives = tuple(
                resolver(
                    candidates,
                    actor_ids_by_pose=tuple(
                        request.actor_ids for _ in candidates
                    ),
                )
            )
            if len(alternatives) != len(candidates) or any(
                value is not None and not isinstance(value, CameraPose)
                for value in alternatives
            ):
                raise RuntimeError(
                    "camera pose resolver returned an invalid recovery sequence"
                )
            recovered[index] = next(
                (value for value in alternatives if value is not None),
                None,
            )
        resolved = tuple(recovered)
    return resolved


def _is_rgb_health_failure(attempt: CaptureAttempt) -> bool:
    return (
        attempt.status is CaptureAttemptStatus.REJECTED
        and attempt.reason in _RGB_HEALTH_REASONS
    )


def execute_stage2_capture_plan(
    plan: Stage2CapturePlan,
    provider: FrameProvider,
    frame_store: FrameStore,
    *,
    content_floor_z: float | None = None,
) -> Stage2CaptureExecution:
    """Execute the plan's primary poses as one bounded camera sweep.

    A generic provider/RPC failure stops further session calls and is surfaced
    as ``runtime_error``.  Only deterministic RGB-health failures may consume
    the already-planned recovery sequence, globally capped by the plan budget.
    Duplicate frames are associated by :class:`FrameStore` and need no
    recovery.  The constrained profile applies a global recovery cap; the
    explicitly unconstrained legacy profile executes each finite program's
    own health recovery without a scene-wide quota.  All primary poses are
    sent to the provider in a single
    call so the UE implementation can reuse one CameraActor.  Recovery
    remains conditional and is therefore captured only after a measured
    health failure.  No model is called during this process.
    """

    if not isinstance(plan, Stage2CapturePlan):
        raise TypeError("plan must be a Stage2CapturePlan")
    if not isinstance(frame_store, FrameStore):
        raise TypeError("frame_store must be a FrameStore")
    if content_floor_z is not None and not math.isfinite(float(content_floor_z)):
        raise ValueError("content_floor_z must be finite")
    attempts: list[CaptureAttempt] = []
    runtime_error: str | None = None
    recovery_attempts = 0

    def execute(request: CaptureRequest) -> CaptureAttempt:
        nonlocal runtime_error
        if (
            content_floor_z is not None
            and not request.actor_ids
            and request.pose.z < float(content_floor_z) + 20.0
        ):
            return CaptureAttempt(
                request.request_id,
                request.task_ids,
                request.shot_role,
                CaptureAttemptStatus.REJECTED,
                reason="camera_below_content_floor",
                recovery_step=request.recovery_step,
            )
        try:
            (resolved_pose,) = _resolve_capture_poses(provider, (request,))
            if resolved_pose is None:
                return CaptureAttempt(
                    request.request_id,
                    request.task_ids,
                    request.shot_role,
                    CaptureAttemptStatus.REJECTED,
                    reason="camera_pose_unresolved",
                    recovery_step=request.recovery_step,
                )
            captured = provider.capture(
                (resolved_pose,),
                phase=(
                    "stage2_recovery" if request.recovery_step else "stage2_capture"
                ),
                # Provider ids are discarded by FrameStore, but keep even this
                # controller-facing prefix neutral to prevent accidental leaks.
                frame_id_prefix="frame",
            )
            if len(captured) != 1:
                raise RuntimeError(
                    f"provider returned {len(captured)} frames for one pose"
                )
            admission = frame_store.admit_captured_frame(
                captured[0],
                shot_role=request.shot_role,
                task_ids=request.task_ids,
                actor_ids=request.actor_ids,
            )
            return _attempt_from_admission(request, admission)
        except RgbFrameHealthError as exc:
            return CaptureAttempt(
                request.request_id,
                request.task_ids,
                request.shot_role,
                CaptureAttemptStatus.REJECTED,
                reason=exc.reason,
                recovery_step=request.recovery_step,
            )
        except Exception as exc:  # noqa: BLE001  # provider/RPC failure poisons session
            runtime_error = f"{type(exc).__name__}: {exc}"
            return CaptureAttempt(
                request.request_id,
                request.task_ids,
                request.shot_role,
                CaptureAttemptStatus.CAPTURE_ERROR,
                reason=runtime_error,
                recovery_step=request.recovery_step,
            )

    def execute_primary_batch(
        requests: Sequence[CaptureRequest],
    ) -> tuple[CaptureAttempt, ...]:
        """Capture the plan's normal poses together, preserving order."""

        nonlocal runtime_error
        if not requests:
            return ()
        invalid: dict[str, CaptureAttempt] = {}
        eligible: list[CaptureRequest] = []
        for request in requests:
            if (
                content_floor_z is not None
                and not request.actor_ids
                and request.pose.z < float(content_floor_z) + 20.0
            ):
                invalid[request.request_id] = CaptureAttempt(
                    request.request_id,
                    request.task_ids,
                    request.shot_role,
                    CaptureAttemptStatus.REJECTED,
                    reason="camera_below_content_floor",
                    recovery_step=request.recovery_step,
                )
            else:
                eligible.append(request)
        admitted: dict[str, CaptureAttempt] = {}
        if eligible:
            try:
                resolved = _resolve_capture_poses(provider, eligible)
                capture_requests: list[CaptureRequest] = []
                capture_poses: list[CameraPose] = []
                for request, resolved_pose in zip(
                    eligible, resolved, strict=True
                ):
                    if resolved_pose is None:
                        admitted[request.request_id] = CaptureAttempt(
                            request.request_id,
                            request.task_ids,
                            request.shot_role,
                            CaptureAttemptStatus.REJECTED,
                            reason="camera_pose_unresolved",
                            recovery_step=request.recovery_step,
                        )
                    else:
                        capture_requests.append(request)
                        capture_poses.append(resolved_pose)
                captured = (
                    tuple(
                        provider.capture(
                            tuple(capture_poses),
                            phase="stage2_capture",
                            frame_id_prefix="frame",
                        )
                    )
                    if capture_poses
                    else ()
                )
                if len(captured) != len(capture_requests):
                    raise RuntimeError(
                        "provider returned "
                        f"{len(captured)} frames for "
                        f"{len(capture_requests)} resolved poses"
                    )
                for request, resolved_pose, frame in zip(
                    capture_requests, capture_poses, captured, strict=True
                ):
                    if frame.pose != resolved_pose:
                        raise RuntimeError(
                            "provider changed capture pose order inside a program"
                        )
                    admission = frame_store.admit_captured_frame(
                        frame,
                        shot_role=request.shot_role,
                        task_ids=request.task_ids,
                        actor_ids=request.actor_ids,
                    )
                    admitted[request.request_id] = _attempt_from_admission(
                        request, admission
                    )
            except RgbFrameHealthError as exc:
                for request in eligible:
                    admitted[request.request_id] = CaptureAttempt(
                        request.request_id,
                        request.task_ids,
                        request.shot_role,
                        CaptureAttemptStatus.REJECTED,
                        reason=exc.reason,
                        recovery_step=request.recovery_step,
                    )
            except Exception as exc:  # noqa: BLE001 - provider poisons session
                runtime_error = f"{type(exc).__name__}: {exc}"
                for request in eligible:
                    admitted[request.request_id] = CaptureAttempt(
                        request.request_id,
                        request.task_ids,
                        request.shot_role,
                        CaptureAttemptStatus.CAPTURE_ERROR,
                        reason=runtime_error,
                        recovery_step=request.recovery_step,
                    )
        return tuple(
            invalid.get(request.request_id)
            or admitted[request.request_id]
            for request in requests
        )

    all_primary = tuple(
        request
        for program in plan.programs
        for request in program.requests
        if request.recovery_step == 0
    )
    primary_attempts = {
        value.request_id: value for value in execute_primary_batch(all_primary)
    }

    for program in plan.programs:
        primary = tuple(value for value in program.requests if value.recovery_step == 0)
        recovery = tuple(value for value in program.requests if value.recovery_step > 0)
        pending_health_failures = 0
        for request in primary:
            attempt = primary_attempts[request.request_id]
            attempts.append(attempt)
            if _is_rgb_health_failure(attempt):
                pending_health_failures += 1
        if runtime_error is not None:
            break
        for request in recovery:
            if pending_health_failures == 0 or (
                plan.budget.global_capture_limits_enabled
                and recovery_attempts >= plan.budget.max_recovery_frames
            ):
                attempts.append(
                    CaptureAttempt(
                        request.request_id,
                        request.task_ids,
                        request.shot_role,
                        CaptureAttemptStatus.SKIPPED_RECOVERY,
                        reason=(
                            "primary_capture_sufficient"
                            if pending_health_failures == 0
                            else "recovery_budget_exhausted"
                        ),
                        recovery_step=request.recovery_step,
                    )
                )
                continue
            recovery_attempts += 1
            attempt = execute(request)
            attempts.append(attempt)
            if attempt.status in {
                CaptureAttemptStatus.ACCEPTED,
                CaptureAttemptStatus.DUPLICATE,
            }:
                pending_health_failures -= 1
            elif not _is_rgb_health_failure(attempt):
                # A malformed/store-cap rejection is not permission to keep
                # calling the provider.  Only a fresh RGB-health failure may
                # continue the bounded recovery chain.
                pending_health_failures = 0
            if runtime_error is not None:
                break

    task_ids = tuple(
        dict.fromkeys(
            task_id for program in plan.programs for task_id in program.task_ids
        )
    )
    coverage = {
        task_id: _capture_coverage_for_task(task_id, frame_store)
        for task_id in task_ids
    }
    return Stage2CaptureExecution(tuple(attempts), coverage, runtime_error)


__all__ = [
    "CaptureAttempt",
    "CaptureAttemptStatus",
    "Stage2CaptureExecution",
    "cluster_stage2_targets",
    "execute_stage2_capture_plan",
    "plan_stage2_capture_groups",
    "preferred_shot_roles_for_task",
    "union_actor_bounds",
]
