"""Geometry-routed, then coverage-first exploration for RequirementGraph Stage 3.

The explorer receives scene geometry, already-captured overview RGB, and an
active frame provider.  It deliberately has no prompt/claim argument.  For an
UNKNOWN task it may receive only controller-side actor ids and AABBs so it can
retry a focused view before spending budget on blind global search.  None of
that acquisition metadata crosses the RGB-only judge boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Protocol

from .actions import (
    HeightMode,
    InspectCellAction,
    PitchMode,
    camera_pose_for_cell,
    generate_scout_poses,
    height_for_mode,
)
from .actor_inventory import ActorBounds, ActorInventorySnapshot
from .asset_candidates import (
    assess_bounds_camera_pose,
    plan_bounds_camera_poses,
    plan_bounds_camera_recovery_poses,
)
from .contracts import CameraPose, SceneBounds
from .overview_planning import plan_overview_poses
from .runtime import FrameProvider, RgbFrameHealthError
from .stage2_contracts import CaptureShotRole
from .stage2_capture import union_actor_bounds
from .stage2_frames import FrameRecord, FrameRejectionReason, FrameStore
from .stage2_routing import Stage2ActorTarget
from .stage3_contracts import (
    ExplorationAttemptStatus,
    ExplorationSource,
    PortfolioRequirement,
    Stage3Budget,
    Stage3Coverage,
    Stage3ExplorationAttempt,
    Stage3ExplorationResult,
)

RPC_CIRCUIT_BREAK_THRESHOLD = 2
MIN_GLOBAL_PORTFOLIO_FRAMES = 2
MIN_INTERIOR_PORTFOLIO_FRAMES = 2
MIN_DISTINCT_PORTFOLIO_POSES = 4


class HolisticFrameSelector(Protocol):
    """Optional RGB-only portfolio selector."""

    def select(self, frames: Sequence[Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class _Selection:
    ordered_frame_ids: tuple[str, ...]
    unusable_frame_ids: tuple[str, ...] = ()


def _pose_cell(pose: CameraPose, bounds: SceneBounds) -> tuple[int, int]:
    x_fraction = (pose.x - bounds.min_x) / bounds.width
    y_fraction = (pose.y - bounds.min_y) / bounds.depth
    return (
        max(0, min(4, int(x_fraction * 5))),
        max(0, min(4, int(y_fraction * 5))),
    )


def _angular_distance(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _pose_distance(left: CameraPose, right: CameraPose) -> float:
    return (
        math.dist(left.location_cm, right.location_cm) / 1000.0
        + _angular_distance(left.yaw, right.yaw) / 180.0
        + abs(left.pitch - right.pitch) / 180.0
    )


def _same_pose(left: CameraPose, right: CameraPose) -> bool:
    return (
        math.dist(left.location_cm, right.location_cm) <= 5.0
        and _angular_distance(left.yaw, right.yaw) <= 2.0
        and abs(left.pitch - right.pitch) <= 2.0
    )


def _pose_key(pose: CameraPose) -> tuple[float, ...]:
    return (
        round(pose.x, 1),
        round(pose.y, 1),
        round(pose.z, 1),
        round(pose.pitch, 1),
        round(pose.yaw % 360.0, 1),
    )


def _resolve_actor_camera_poses(
    provider: FrameProvider,
    poses: Sequence[CameraPose],
    actor_ids_by_pose: Sequence[Sequence[str]],
) -> tuple[CameraPose | None, ...]:
    """Apply the optional live-geometry camera solver before RGB capture."""

    requested = tuple(poses)
    actor_groups = tuple(tuple(value) for value in actor_ids_by_pose)
    if len(requested) != len(actor_groups):
        raise ValueError("poses and actor_ids_by_pose must have equal length")
    resolver = getattr(provider, "resolve_camera_poses", None)
    if not callable(resolver):
        return requested
    resolved = tuple(
        resolver(
            requested,
            actor_ids_by_pose=actor_groups,
        )
    )
    if len(resolved) != len(requested) or any(
        value is not None and not isinstance(value, CameraPose)
        for value in resolved
    ):
        raise RuntimeError(
            "camera pose resolver must return one CameraPose or None per request"
        )
    return resolved


def _resolve_actor_camera_pose_with_recovery(
    provider: FrameProvider,
    actor_ids: Sequence[str],
    bounds: ActorBounds,
    scene_bounds: SceneBounds,
    *,
    excluded_poses: Sequence[CameraPose] = (),
) -> tuple[CameraPose | None, int]:
    """Try diverse live-safe poses after the primary AABB pose is rejected."""

    candidates = tuple(
        pose
        for pose in plan_bounds_camera_recovery_poses(
            bounds.center_cm,
            bounds.extent_cm,
            scene_bounds,
        )
        if not any(_same_pose(pose, excluded) for excluded in excluded_poses)
    )
    if not candidates:
        return None, 0
    resolved = _resolve_actor_camera_poses(
        provider,
        candidates,
        tuple(tuple(actor_ids) for _ in candidates),
    )
    return next((value for value in resolved if value is not None), None), len(candidates)


def _is_aerial_pose(pose: CameraPose, bounds: SceneBounds) -> bool:
    # ``height_for_mode`` is deterministic, but a small tolerance admits
    # replayed/canonicalized poses without misclassifying ordinary eye views.
    threshold = (bounds.max_z + height_for_mode(bounds, HeightMode.AERIAL)) / 2.0
    return pose.z >= threshold and pose.pitch <= -20.0


def _is_global(record: FrameRecord, bounds: SceneBounds) -> bool:
    del bounds
    # OVERVIEW is the acquisition role: it frames the scene rather than one
    # actor.  A high ring view may simultaneously satisfy the aerial quota;
    # excluding it from global coverage caused four frozen formal overviews to
    # count as zero global evidence and triggered redundant scout captures.
    return record.shot_role is CaptureShotRole.OVERVIEW


def _is_interior(record: FrameRecord, bounds: SceneBounds) -> bool:
    # Stage 2 also labels its diagonal aerial fallbacks as GRID.  A shot only
    # satisfies Stage 3's interior quota when it is a non-aerial grid view.
    return (
        record.shot_role is CaptureShotRole.GRID
        and not _is_aerial_pose(record.pose, bounds)
    )


def _record_order(records: Sequence[FrameRecord]) -> Mapping[str, int]:
    return {record.frame_id: index for index, record in enumerate(records)}


def _farthest_score(
    record: FrameRecord,
    selected: Sequence[FrameRecord],
    order: Mapping[str, int],
) -> tuple[float, float, float]:
    diversity = (
        min(_pose_distance(record.pose, value.pose) for value in selected)
        if selected
        else 0.0
    )
    return (
        diversity,
        record.quality.standard_deviation,
        float(-order[record.frame_id]),
    )


def _choose_seed_records(
    stage2_frame_store: Any | None,
    bounds: SceneBounds,
    limit: int,
    *,
    eligible_task_ids: Sequence[str] = (),
    eligible_actor_ids: Sequence[str] = (),
    require_global_context: bool = True,
) -> tuple[FrameRecord, ...]:
    if stage2_frame_store is None or limit <= 0:
        return ()
    # Accept either the store itself or the full Stage2Evaluation product.
    store = getattr(stage2_frame_store, "frame_store", stage2_frame_store)
    raw_records = tuple(getattr(store, "records", ()))
    candidates = [value for value in raw_records if isinstance(value, FrameRecord)]
    eligible = frozenset(str(value).strip() for value in eligible_task_ids)
    eligible_actors = frozenset(
        str(value).strip().casefold() for value in eligible_actor_ids
    )
    if not require_global_context:
        candidates = [
            value
            for value in candidates
            if value.shot_role is not CaptureShotRole.OVERVIEW
            and (
                bool(eligible.intersection(value.task_ids))
                or bool(
                    eligible_actors.intersection(
                        actor_id.casefold() for actor_id in value.actor_ids
                    )
                )
            )
        ]
    if not candidates:
        return ()
    order = _record_order(candidates)
    selected: list[FrameRecord] = []
    global_roles = {CaptureShotRole.OVERVIEW, CaptureShotRole.GRID}

    def take_best(predicate: Any) -> None:
        remaining = [
            value
            for value in candidates
            if value not in selected and predicate(value)
        ]
        if remaining and len(selected) < limit:
            selected.append(
                max(
                    remaining,
                    key=lambda value: _farthest_score(value, selected, order),
                )
            )

    # Reuse one already-valid focus frame for as many distinct unresolved
    # tasks/actors as the seed budget permits.  Earlier code capped this at two
    # and then re-photographed actors Stage 2 had just captured.  Keep up to two
    # slots for holistic views when the budget is large enough.
    represented_tasks: set[str] = set()
    represented_actors: set[str] = set()
    reserve_global = (
        min(2, max(0, limit - 1)) if require_global_context else 0
    )
    focus_limit = max(0, limit - reserve_global)
    while len(selected) < focus_limit:
        focus_candidates = [
            value
            for value in candidates
            if value not in selected
            and bool(value.actor_ids)
            and value.shot_role not in global_roles
            and (
                bool(eligible.intersection(value.task_ids))
                or bool(
                    eligible_actors.intersection(
                        actor_id.casefold() for actor_id in value.actor_ids
                    )
                )
            )
        ]
        if not focus_candidates:
            break

        def focus_score(value: FrameRecord) -> tuple[int, float, float, float]:
            task_gain = len(
                eligible.intersection(value.task_ids) - represented_tasks
            )
            actor_values = {
                actor_id.casefold() for actor_id in value.actor_ids
            }
            actor_gain = len(
                eligible_actors.intersection(actor_values) - represented_actors
            )
            return (
                task_gain + actor_gain,
                *_farthest_score(value, selected, order),
            )

        winner = max(focus_candidates, key=focus_score)
        if focus_score(winner)[0] <= 0:
            break
        selected.append(winner)
        represented_tasks.update(eligible.intersection(winner.task_ids))
        represented_actors.update(
            eligible_actors.intersection(
                actor_id.casefold() for actor_id in winner.actor_ids
            )
        )

    if require_global_context:
        for _ in range(2):
            take_best(lambda value: _is_global(value, bounds))
        take_best(lambda value: _is_aerial_pose(value.pose, bounds))
        take_best(lambda value: _is_interior(value, bounds))
    while len(selected) < min(limit, len(candidates)):
        take_best(lambda value: True)
    return tuple(selected)


def _coerce_selector_result(raw: Any, allowed_ids: Sequence[str]) -> _Selection:
    if isinstance(raw, Mapping):
        transport_status = raw.get("transport_status")
        parse_status = raw.get("parse_status")
        error = raw.get("error")
        ordered_raw = raw.get("ordered_frame_ids", raw.get("frame_ids"))
        unusable_raw = raw.get("unusable_frame_ids", ())
    elif hasattr(raw, "ordered_frame_ids"):
        transport_status = getattr(raw, "transport_status", None)
        parse_status = getattr(raw, "parse_status", None)
        error = getattr(raw, "error", None)
        ordered_raw = raw.ordered_frame_ids
        unusable_raw = getattr(raw, "unusable_frame_ids", ())
    else:
        transport_status = None
        parse_status = None
        error = None
        ordered_raw = raw
        unusable_raw = ()
    if transport_status is not None and str(transport_status) != "success":
        raise ValueError(str(error).strip() if error else "selector transport failed")
    if parse_status is not None and str(parse_status) not in {"success", "valid"}:
        raise ValueError(str(error).strip() if error else "selector response was invalid")
    if isinstance(ordered_raw, (str, bytes)) or ordered_raw is None:
        raise ValueError("selector must return an ordered frame-id sequence")
    if isinstance(unusable_raw, (str, bytes)) or unusable_raw is None:
        raise ValueError("unusable_frame_ids must be a frame-id sequence")
    try:
        ordered = tuple(str(value).strip() for value in ordered_raw)
        unusable = tuple(str(value).strip() for value in unusable_raw)
    except TypeError as exc:
        raise ValueError("selector frame ids must be iterable") from exc
    if any(not value for value in (*ordered, *unusable)):
        raise ValueError("selector frame ids must be non-empty")
    if len(ordered) != len(set(ordered)) or len(unusable) != len(set(unusable)):
        raise ValueError("selector frame ids must be unique")
    allowed = set(allowed_ids)
    unknown = (set(ordered) | set(unusable)) - allowed
    if unknown:
        raise ValueError("selector returned unknown frame ids")
    if set(ordered) & set(unusable):
        raise ValueError("ordered and unusable frame ids must not overlap")
    # Omitted usable ids remain eligible after the model's explicit ordering.
    complete_order = (*ordered, *(value for value in allowed_ids if value not in ordered))
    return _Selection(tuple(complete_order), unusable)


def _deterministic_rank(records: Sequence[FrameRecord]) -> tuple[str, ...]:
    """Rank healthy frames without prompt semantics or controller metadata."""

    order = _record_order(records)
    remaining = list(records)
    selected: list[FrameRecord] = []
    while remaining:
        winner = max(
            remaining,
            key=lambda value: _farthest_score(value, selected, order),
        )
        selected.append(winner)
        remaining.remove(winner)
    return tuple(value.frame_id for value in selected)


def _portfolio_from_rank(
    records: Sequence[FrameRecord],
    ranked_ids: Sequence[str],
    unusable_ids: Sequence[str],
    bounds: SceneBounds,
    budget: Stage3Budget,
    *,
    require_global_context: bool,
) -> tuple[str, ...]:
    by_id = {record.frame_id: record for record in records}
    unusable = set(unusable_ids)
    usable_rank = [value for value in ranked_ids if value not in unusable]
    rank = {frame_id: index for index, frame_id in enumerate(usable_rank)}
    selected: list[FrameRecord] = []

    def choose(predicate: Any, *, distinct_cells: bool = False) -> bool:
        selected_cells = {_pose_cell(value.pose, bounds) for value in selected}
        candidates = [
            by_id[frame_id]
            for frame_id in usable_rank
            if frame_id in by_id
            and by_id[frame_id] not in selected
            and predicate(by_id[frame_id])
            and (
                not distinct_cells
                or _pose_cell(by_id[frame_id].pose, bounds) not in selected_cells
            )
        ]
        if not candidates or len(selected) >= budget.max_portfolio_frames:
            return False

        def score(value: FrameRecord) -> tuple[float, float, float]:
            diversity = (
                min(_pose_distance(value.pose, item.pose) for item in selected)
                if selected
                else 0.0
            )
            return (
                diversity,
                float(-rank[value.frame_id]),
                value.quality.standard_deviation,
            )

        selected.append(max(candidates, key=score))
        return True

    if require_global_context:
        # Hard whole-scene roles apply only to scene-identity/atmosphere
        # claims. Targeted semantic claims retain their routed actor evidence
        # without manufacturing an unrelated holistic portfolio.
        choose(lambda value: _is_aerial_pose(value.pose, bounds))
        while (
            sum(_is_global(value, bounds) for value in selected)
            < MIN_GLOBAL_PORTFOLIO_FRAMES
        ):
            if not choose(lambda value: _is_global(value, bounds)):
                break
        while (
            sum(_is_interior(value, bounds) for value in selected)
            < MIN_INTERIOR_PORTFOLIO_FRAMES
        ):
            is_interior = lambda value: _is_interior(value, bounds)
            if not choose(is_interior, distinct_cells=True) and not choose(is_interior):
                break

        # Ensure four genuinely different camera transforms before ordinary fill.
        while (
            len({_pose_key(value.pose) for value in selected})
            < MIN_DISTINCT_PORTFOLIO_POSES
        ):
            selected_keys = {_pose_key(value.pose) for value in selected}
            if not choose(
                lambda value, keys=selected_keys: _pose_key(value.pose) not in keys
            ):
                break

    target_size = min(budget.max_portfolio_frames, len(usable_rank))
    target_size = max(
        min(budget.min_portfolio_frames, len(usable_rank)),
        target_size,
    )
    while len(selected) < target_size:
        if not choose(lambda value: True):
            break
    return tuple(value.frame_id for value in selected)


def _coverage(
    store: FrameStore,
    portfolio_ids: Sequence[str],
    bounds: SceneBounds,
    budget: Stage3Budget,
    *,
    focus_frame_ids: Sequence[str] = (),
    focused_task_ids: Sequence[str] = (),
    require_global_context: bool = True,
) -> Stage3Coverage:
    records = store.records
    selected = tuple(store.get(value) for value in portfolio_ids)
    covered_cells = tuple(sorted({_pose_cell(value.pose, bounds) for value in records}))
    aerial = sum(_is_aerial_pose(value.pose, bounds) for value in selected)
    global_frames = sum(_is_global(value, bounds) for value in selected)
    interior = sum(_is_interior(value, bounds) for value in selected)
    distinct = len({_pose_key(value.pose) for value in selected})
    missing: list[PortfolioRequirement] = []
    if require_global_context:
        if len(selected) < budget.min_portfolio_frames:
            missing.append(PortfolioRequirement.MINIMUM_SIZE)
        if aerial < 1:
            missing.append(PortfolioRequirement.AERIAL)
        if global_frames < MIN_GLOBAL_PORTFOLIO_FRAMES:
            missing.append(PortfolioRequirement.GLOBAL)
        if interior < MIN_INTERIOR_PORTFOLIO_FRAMES or len(
            {
                _pose_cell(value.pose, bounds)
                for value in selected
                if _is_interior(value, bounds)
            }
        ) < min(MIN_INTERIOR_PORTFOLIO_FRAMES, interior):
            missing.append(PortfolioRequirement.INTERIOR)
        if distinct < min(MIN_DISTINCT_PORTFOLIO_POSES, budget.min_portfolio_frames):
            missing.append(PortfolioRequirement.POSE_DIVERSITY)
    return Stage3Coverage(
        covered_cells=covered_cells,
        valid_seed_frames=sum(value.phase == "stage3_seed" for value in records),
        valid_new_frames=sum(value.phase != "stage3_seed" for value in records),
        focus_frames=len(set(focus_frame_ids)),
        focused_task_count=len(set(focused_task_ids)),
        aerial_frames=aerial,
        global_frames=global_frames,
        interior_frames=interior,
        distinct_pose_count=distinct,
        missing_requirements=tuple(missing),
    )


def _restore_selector_coverage(
    store: FrameStore,
    records: Sequence[FrameRecord],
    ranked_ids: Sequence[str],
    unusable_ids: Sequence[str],
    bounds: SceneBounds,
    budget: Stage3Budget,
    *,
    require_global_context: bool,
) -> tuple[tuple[str, ...], Stage3Coverage, tuple[str, ...]]:
    """Minimally override selector exclusions that break hard coverage.

    Every candidate here has already passed the controller-owned FrameStore
    health checks.  The optional RGB-only selector may still call a frame
    unusable, but it cannot veto every accepted frame for a controller-required
    role.  Search is bounded by the ten-frame exploration hard cap and prefers
    the smallest deterministic set of restored frame ids.
    """

    portfolio_ids = _portfolio_from_rank(
        records,
        ranked_ids,
        unusable_ids,
        bounds,
        budget,
        require_global_context=require_global_context,
    )
    coverage = _coverage(
        store,
        portfolio_ids,
        bounds,
        budget,
        require_global_context=require_global_context,
    )
    if coverage.constraints_met or not unusable_ids:
        return portfolio_ids, coverage, ()

    unusable = set(unusable_ids)
    recovery_order = tuple(
        frame_id
        for frame_id in _deterministic_rank(records)
        if frame_id in unusable
    )
    for restored_count in range(1, len(recovery_order) + 1):
        for restored_ids in combinations(recovery_order, restored_count):
            restored = set(restored_ids)
            remaining_unusable = tuple(
                frame_id
                for frame_id in unusable_ids
                if frame_id not in restored
            )
            candidate_ids = _portfolio_from_rank(
                records,
                ranked_ids,
                remaining_unusable,
                bounds,
                budget,
                require_global_context=require_global_context,
            )
            candidate_coverage = _coverage(
                store,
                candidate_ids,
                bounds,
                budget,
                require_global_context=require_global_context,
            )
            if candidate_coverage.constraints_met:
                return candidate_ids, candidate_coverage, restored_ids
    return portfolio_ids, coverage, ()


def _inward_yaw_bin(cell: tuple[int, int], index: int) -> int:
    dx = 2.0 - cell[0]
    dy = 2.0 - cell[1]
    if dx == 0.0 and dy == 0.0:
        return (index * 3) % 12
    yaw = math.degrees(math.atan2(dy, dx)) % 360.0
    return round(yaw / 30.0) % 12


def _interior_pose(
    cell: tuple[int, int],
    index: int,
    bounds: SceneBounds,
    *,
    content_floor_z: float | None = None,
) -> CameraPose:
    height = HeightMode.EYE if index % 2 == 0 else HeightMode.MID
    pitch = PitchMode.LEVEL if height is HeightMode.EYE else PitchMode.DOWN
    return camera_pose_for_cell(
        InspectCellAction(
            cell_x=cell[0],
            cell_y=cell[1],
            height=height,
            yaw_bin=_inward_yaw_bin(cell, index),
            pitch_mode=pitch,
        ),
        bounds,
        floor_z=content_floor_z,
    )


def _recovery_pose(
    pose: CameraPose,
    cell: tuple[int, int] | None,
    recovery_step: int,
    bounds: SceneBounds,
    *,
    content_floor_z: float | None = None,
) -> CameraPose:
    if cell is not None:
        height = (
            HeightMode.MID
            if pose.z
            <= height_for_mode(
                bounds,
                HeightMode.EYE,
                floor_z=content_floor_z,
            )
            + 1.0
            else HeightMode.EYE
        )
        pitch = PitchMode.DOWN if height is HeightMode.MID else PitchMode.LEVEL
        return camera_pose_for_cell(
            InspectCellAction(
                cell_x=cell[0],
                cell_y=cell[1],
                height=height,
                yaw_bin=(_inward_yaw_bin(cell, recovery_step) + 2 * recovery_step) % 12,
                pitch_mode=pitch,
            ),
            bounds,
            floor_z=content_floor_z,
        )
    # Scout recovery keeps the vantage point but uses an oblique direction.
    return CameraPose(
        pose.x,
        pose.y,
        pose.z,
        max(-85.0, min(30.0, pose.pitch - 10.0)),
        (pose.yaw + 60.0 * recovery_step) % 360.0,
        pose.roll,
    )


def _uncovered_cell(
    covered: set[tuple[int, int]],
    attempted: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = [
        (x, y)
        for x in range(5)
        for y in range(5)
        if (x, y) not in attempted
    ]
    if not candidates:
        return None

    def score(cell: tuple[int, int]) -> tuple[float, float, int, int]:
        nearest = (
            min(math.dist(cell, value) for value in covered)
            if covered
            else math.dist(cell, (2, 2)) * -1.0
        )
        # Prefer the central 3x3 on ties, then deterministic lexical order.
        interior = float(0 < cell[0] < 4 and 0 < cell[1] < 4)
        center = -math.dist(cell, (2, 2))
        return (nearest, interior, center, -(cell[0] * 5 + cell[1]))

    return max(candidates, key=score)


class HolisticExplorer:
    """Acquire and select a bounded, prompt-blind holistic RGB portfolio.

    The supplied provider is already inside the graph runner's UE session.
    This class never enters, exits, or closes it.
    """

    def __init__(
        self,
        budget: Stage3Budget | None = None,
        selector: HolisticFrameSelector | None = None,
    ) -> None:
        self.budget = Stage3Budget() if budget is None else budget
        if not isinstance(self.budget, Stage3Budget):
            raise TypeError("budget must be a Stage3Budget")
        if selector is not None and not callable(getattr(selector, "select", None)):
            raise TypeError("selector must provide select(frames)")
        self.selector = selector
        self.last_result: Stage3ExplorationResult | None = None

    def explore(
        self,
        scene_bounds: SceneBounds,
        frame_provider: FrameProvider,
        *,
        stage2_frame_store: Any | None = None,
        focus_targets_by_task: Mapping[
            str, Sequence[Stage2ActorTarget]
        ] | None = None,
        grid_search_task_ids: Sequence[str] = (),
        require_global_context: bool = True,
        content_floor_z: float | None = None,
        adaptive_reframe_reserve: int = 0,
    ) -> Stage3ExplorationResult:
        if not isinstance(scene_bounds, SceneBounds):
            raise TypeError("scene_bounds must be a SceneBounds")
        if not isinstance(require_global_context, bool):
            raise TypeError("require_global_context must be a bool")
        if content_floor_z is not None and not (
            math.isfinite(float(content_floor_z))
            and scene_bounds.min_z <= float(content_floor_z) <= scene_bounds.max_z
        ):
            raise ValueError("content_floor_z must be finite and inside scene bounds")
        if (
            isinstance(adaptive_reframe_reserve, bool)
            or not isinstance(adaptive_reframe_reserve, int)
            or not 0 <= adaptive_reframe_reserve
            <= self.budget.max_valid_exploration_frames
            - self.budget.min_portfolio_frames
        ):
            raise ValueError(
                "adaptive_reframe_reserve must fit inside the total Stage 3 "
                "frame budget after the minimum portfolio"
            )
        focus_targets = dict(focus_targets_by_task or {})
        for task_id, targets in focus_targets.items():
            if not str(task_id).strip() or any(
                not isinstance(value, Stage2ActorTarget) for value in targets
            ):
                raise TypeError(
                    "focus_targets_by_task must map task ids to Stage2ActorTarget sequences"
                )
        search_tasks = tuple(
            dict.fromkeys(str(value).strip() for value in grid_search_task_ids)
        )
        if any(not value for value in search_tasks):
            raise ValueError("grid_search_task_ids entries must be non-empty")
        if any(value not in focus_targets for value in search_tasks):
            raise ValueError(
                "grid_search_task_ids must name tasks in focus_targets_by_task"
            )
        grouped_focus: dict[str, tuple[Stage2ActorTarget, set[str]]] = {}
        for task_id in sorted(focus_targets):
            for target in focus_targets[task_id]:
                key = target.actor_id.casefold()
                if key not in grouped_focus:
                    grouped_focus[key] = (target, set())
                grouped_focus[key][1].add(task_id)
        capture = getattr(frame_provider, "capture", None)
        if not callable(capture):
            raise TypeError("frame_provider must provide capture(poses, ...)")

        budget = self.budget
        provider_scene = getattr(frame_provider, "scene", None)
        provider_inventory = getattr(provider_scene, "inventory", None)
        dense_overview_poses = (
            plan_overview_poses(
                provider_inventory,
                scene_bounds,
                count=6,
            )
            if isinstance(provider_inventory, ActorInventorySnapshot)
            else ()
        )
        store = FrameStore(budget, frame_id_prefix="s3f")
        initial_valid_cap = max(
            budget.min_portfolio_frames,
            budget.max_valid_exploration_frames - adaptive_reframe_reserve,
        )
        task_frame_ids: dict[str, list[str]] = {}
        frame_actor_ids: dict[str, set[str]] = {}
        attempts: list[Stage3ExplorationAttempt] = []
        attempt_number = 1
        new_capture_count = 0
        recovery_count = 0
        consecutive_rpc_errors = 0
        circuit_open = False
        runtime_error: str | None = None

        def associate_route(
            frame_id: str,
            *,
            task_ids: Sequence[str] = (),
            actor_ids: Sequence[str] = (),
        ) -> None:
            """Keep routing metadata outside the publishable RGB FrameStore."""

            for task_id in task_ids:
                routed = task_frame_ids.setdefault(task_id, [])
                if frame_id not in routed:
                    routed.append(frame_id)
            if actor_ids:
                frame_actor_ids.setdefault(frame_id, set()).update(
                    value.casefold() for value in actor_ids
                )

        def record_attempt(
            *,
            source: ExplorationSource,
            pose: CameraPose,
            shot_role: CaptureShotRole,
            status: ExplorationAttemptStatus,
            frame_id: str | None = None,
            reason: str | None = None,
            cell: tuple[int, int] | None = None,
            recovery_step: int = 0,
        ) -> None:
            nonlocal attempt_number
            attempts.append(
                Stage3ExplorationAttempt(
                    attempt_id=f"s3a_{attempt_number:06d}",
                    source=source,
                    pose=pose,
                    shot_role=shot_role,
                    status=status,
                    frame_id=frame_id,
                    reason=reason,
                    cell=cell,
                    recovery_step=recovery_step,
                )
            )
            attempt_number += 1

        # Re-admission copies RGB and pose only.  Stage 2 task/actor association
        # stays in controller-side route maps and never enters the publishable
        # Stage 3 FrameStore or the RGB-only judge payload.
        for seed in _choose_seed_records(
            stage2_frame_store,
            scene_bounds,
            budget.max_seed_frames,
            eligible_task_ids=tuple(focus_targets),
            eligible_actor_ids=tuple(grouped_focus),
            require_global_context=require_global_context,
        ):
            seed_task_values: set[str] = set()
            if seed.shot_role is CaptureShotRole.GRID:
                seed_task_values.update(
                    value for value in seed.task_ids if value in focus_targets
                )
            for actor_id in seed.actor_ids:
                grouped = grouped_focus.get(actor_id.casefold())
                if grouped is not None:
                    seed_task_values.update(grouped[1])
            seed_task_ids = tuple(sorted(seed_task_values))
            admission = store.admit(
                seed.rgb,
                pose=seed.pose,
                phase="stage3_seed",
                shot_role=seed.shot_role,
            )
            if admission.record is not None:
                associate_route(
                    admission.record.frame_id,
                    task_ids=seed_task_ids,
                    actor_ids=seed.actor_ids if seed_task_ids else (),
                )
                record_attempt(
                    source=ExplorationSource.SEED,
                    pose=seed.pose,
                    shot_role=seed.shot_role,
                    status=ExplorationAttemptStatus.ACCEPTED,
                    frame_id=admission.record.frame_id,
                )
            else:
                if admission.duplicate_of_frame_id is not None:
                    associate_route(
                        admission.duplicate_of_frame_id,
                        task_ids=seed_task_ids,
                        actor_ids=seed.actor_ids if seed_task_ids else (),
                    )
                reason = (
                    admission.rejection_reason.value
                    if admission.rejection_reason is not None
                    else "rejected"
                )
                status = (
                    ExplorationAttemptStatus.DUPLICATE
                    if admission.rejection_reason is FrameRejectionReason.NEAR_DUPLICATE
                    else ExplorationAttemptStatus.REJECTED
                )
                record_attempt(
                    source=ExplorationSource.SEED,
                    pose=seed.pose,
                    shot_role=seed.shot_role,
                    status=status,
                    reason=reason,
                )

        def can_capture() -> bool:
            return (
                not circuit_open
                and new_capture_count < budget.max_new_capture_attempts
                and store.valid_frame_count < initial_valid_cap
            )

        def capture_once(
            pose: CameraPose,
            *,
            source: ExplorationSource,
            role: CaptureShotRole,
            cell: tuple[int, int] | None = None,
            recovery_step: int = 0,
            task_ids: Sequence[str] = (),
            actor_ids: Sequence[str] = (),
            target_bounds: ActorBounds | None = None,
        ) -> bool:
            nonlocal new_capture_count, consecutive_rpc_errors, circuit_open, runtime_error
            if not can_capture():
                return False
            if (
                target_bounds is None
                and content_floor_z is not None
                and pose.z < float(content_floor_z) + 20.0
            ):
                record_attempt(
                    source=source,
                    pose=pose,
                    shot_role=role,
                    status=ExplorationAttemptStatus.REJECTED,
                    reason="camera_below_content_floor",
                    cell=cell,
                    recovery_step=recovery_step,
                )
                return False
            if target_bounds is not None and not assess_bounds_camera_pose(
                target_bounds.center_cm,
                target_bounds.extent_cm,
                pose,
            ).healthy:
                record_attempt(
                    source=source,
                    pose=pose,
                    shot_role=role,
                    status=ExplorationAttemptStatus.REJECTED,
                    reason="target_not_usefully_framed",
                    cell=cell,
                    recovery_step=recovery_step,
                )
                return False
            new_capture_count += 1
            phase_suffix = (
                "focus"
                if source is ExplorationSource.FOCUS
                else "interior"
                if role is CaptureShotRole.GRID
                else "scout"
            )
            phase = (
                f"stage3_recovery_{phase_suffix}"
                if source is ExplorationSource.RECOVERY
                else f"stage3_{phase_suffix}"
            )
            try:
                captured_values = tuple(
                    capture(
                        (pose,),
                        phase=phase,
                        frame_id_prefix=f"s3_{source.value}_{new_capture_count:02d}",
                    )
                )
                consecutive_rpc_errors = 0
            except RgbFrameHealthError as exc:
                record_attempt(
                    source=source,
                    pose=pose,
                    shot_role=role,
                    status=ExplorationAttemptStatus.REJECTED,
                    reason=str(getattr(exc, "reason", "rgb_health_error")),
                    cell=cell,
                    recovery_step=recovery_step,
                )
                return False
            except Exception as exc:  # noqa: BLE001 - RPC/provider failure boundary
                consecutive_rpc_errors += 1
                reason = f"{type(exc).__name__}: {exc}"
                if consecutive_rpc_errors >= RPC_CIRCUIT_BREAK_THRESHOLD:
                    circuit_open = True
                    runtime_error = (
                        "capture_rpc_circuit_break: " + reason
                    )
                    reason += "; circuit_break"
                record_attempt(
                    source=source,
                    pose=pose,
                    shot_role=role,
                    status=ExplorationAttemptStatus.CAPTURE_ERROR,
                    reason=reason,
                    cell=cell,
                    recovery_step=recovery_step,
                )
                return False
            if not captured_values:
                record_attempt(
                    source=source,
                    pose=pose,
                    shot_role=role,
                    status=ExplorationAttemptStatus.EMPTY_CAPTURE,
                    reason="provider_returned_no_frame",
                    cell=cell,
                    recovery_step=recovery_step,
                )
                return False

            admission = store.admit_captured_frame(
                captured_values[0],
                phase=phase,
                shot_role=role,
            )
            if admission.record is not None:
                associate_route(
                    admission.record.frame_id,
                    task_ids=task_ids,
                    actor_ids=actor_ids,
                )
                record_attempt(
                    source=source,
                    pose=pose,
                    shot_role=role,
                    status=ExplorationAttemptStatus.ACCEPTED,
                    frame_id=admission.record.frame_id,
                    cell=cell,
                    recovery_step=recovery_step,
                )
                return True
            if admission.duplicate_of_frame_id is not None:
                associate_route(
                    admission.duplicate_of_frame_id,
                    task_ids=task_ids,
                    actor_ids=actor_ids,
                )
            reason = (
                admission.rejection_reason.value
                if admission.rejection_reason is not None
                else "rejected"
            )
            status = (
                ExplorationAttemptStatus.DUPLICATE
                if admission.rejection_reason is FrameRejectionReason.NEAR_DUPLICATE
                else ExplorationAttemptStatus.REJECTED
            )
            record_attempt(
                source=source,
                pose=pose,
                shot_role=role,
                status=status,
                reason=reason,
                cell=cell,
                recovery_step=recovery_step,
            )
            return False

        def capture_planned_batch(specs: Sequence[Mapping[str, Any]]) -> None:
            """Capture independent focus/scout poses in one UE camera sweep."""

            nonlocal new_capture_count, consecutive_rpc_errors, circuit_open, runtime_error
            capacity = min(
                budget.max_new_capture_attempts - new_capture_count,
                initial_valid_cap - store.valid_frame_count,
            )
            requested_specs = tuple(specs[: max(0, capacity)])
            if not requested_specs or circuit_open:
                return
            selected: tuple[Mapping[str, Any], ...] = ()
            resolution_completed = False
            try:
                resolved_poses = _resolve_actor_camera_poses(
                    frame_provider,
                    tuple(value["pose"] for value in requested_specs),
                    tuple(value.get("actor_ids", ()) for value in requested_specs),
                )
                resolution_completed = True
                selected_values: list[dict[str, Any]] = []
                for spec, resolved_pose in zip(
                    requested_specs, resolved_poses, strict=True
                ):
                    if resolved_pose is None:
                        target_bounds = spec.get("target_bounds")
                        if isinstance(target_bounds, ActorBounds):
                            resolved_pose, recovery_candidates = (
                                _resolve_actor_camera_pose_with_recovery(
                                    frame_provider,
                                    spec.get("actor_ids", ()),
                                    target_bounds,
                                    scene_bounds,
                                    excluded_poses=(
                                        spec["pose"],
                                        *tuple(spec.get("excluded_poses", ())),
                                    ),
                                )
                            )
                        else:
                            recovery_candidates = 0
                        if resolved_pose is None:
                            record_attempt(
                                source=spec["source"],
                                pose=spec["pose"],
                                shot_role=spec["role"],
                                status=ExplorationAttemptStatus.REJECTED,
                                reason=(
                                    "camera_pose_unresolved:"
                                    f"{recovery_candidates}_recovery_candidates"
                                ),
                                cell=spec.get("cell"),
                                recovery_step=spec.get("recovery_step", 0),
                            )
                            continue
                        spec = {
                            **spec,
                            "source": ExplorationSource.RECOVERY,
                            "recovery_step": 1,
                        }
                    selected_values.append({**spec, "pose": resolved_pose})
                selected = tuple(selected_values)
                if not selected:
                    return
                new_capture_count += len(selected)
                captured_values = tuple(
                    capture(
                        tuple(value["pose"] for value in selected),
                        phase="stage3_batch",
                        frame_id_prefix="s3_batch",
                    )
                )
                if len(captured_values) != len(selected):
                    raise RuntimeError(
                        "provider returned "
                        f"{len(captured_values)} frames for {len(selected)} Stage 3 poses"
                    )
                if any(
                    frame.pose != spec["pose"]
                    for spec, frame in zip(selected, captured_values, strict=True)
                ):
                    raise RuntimeError("provider changed Stage 3 batch pose order")
                consecutive_rpc_errors = 0
            except RgbFrameHealthError as exc:
                failed_specs = selected if resolution_completed else requested_specs
                for spec in failed_specs:
                    record_attempt(
                        source=spec["source"],
                        pose=spec["pose"],
                        shot_role=spec["role"],
                        status=ExplorationAttemptStatus.REJECTED,
                        reason=str(getattr(exc, "reason", "rgb_health_error")),
                        cell=spec.get("cell"),
                        recovery_step=spec.get("recovery_step", 0),
                    )
                return
            except Exception as exc:  # noqa: BLE001 - provider failure boundary
                consecutive_rpc_errors += 1
                reason = f"{type(exc).__name__}: {exc}"
                if consecutive_rpc_errors >= RPC_CIRCUIT_BREAK_THRESHOLD:
                    circuit_open = True
                    runtime_error = "capture_rpc_circuit_break: " + reason
                    reason += "; circuit_break"
                failed_specs = selected if resolution_completed else requested_specs
                for spec in failed_specs:
                    record_attempt(
                        source=spec["source"],
                        pose=spec["pose"],
                        shot_role=spec["role"],
                        status=ExplorationAttemptStatus.CAPTURE_ERROR,
                        reason=reason,
                        cell=spec.get("cell"),
                        recovery_step=spec.get("recovery_step", 0),
                    )
                return

            for spec, captured in zip(selected, captured_values, strict=True):
                admission = store.admit_captured_frame(
                    captured,
                    phase="stage3_batch",
                    shot_role=spec["role"],
                )
                if admission.record is not None:
                    associate_route(
                        admission.record.frame_id,
                        task_ids=spec.get("task_ids", ()),
                        actor_ids=spec.get("actor_ids", ()),
                    )
                    record_attempt(
                        source=spec["source"],
                        pose=spec["pose"],
                        shot_role=spec["role"],
                        status=ExplorationAttemptStatus.ACCEPTED,
                        frame_id=admission.record.frame_id,
                        cell=spec.get("cell"),
                        recovery_step=spec.get("recovery_step", 0),
                    )
                    continue
                if admission.duplicate_of_frame_id is not None:
                    associate_route(
                        admission.duplicate_of_frame_id,
                        task_ids=spec.get("task_ids", ()),
                        actor_ids=spec.get("actor_ids", ()),
                    )
                reason = (
                    admission.rejection_reason.value
                    if admission.rejection_reason is not None
                    else "rejected"
                )
                record_attempt(
                    source=spec["source"],
                    pose=spec["pose"],
                    shot_role=spec["role"],
                    status=(
                        ExplorationAttemptStatus.DUPLICATE
                        if admission.rejection_reason
                        is FrameRejectionReason.NEAR_DUPLICATE
                        else ExplorationAttemptStatus.REJECTED
                    ),
                    reason=reason,
                    cell=spec.get("cell"),
                    recovery_step=spec.get("recovery_step", 0),
                )

        def capture_with_recovery(
            pose: CameraPose,
            *,
            source: ExplorationSource,
            role: CaptureShotRole,
            cell: tuple[int, int] | None = None,
            task_ids: Sequence[str] = (),
        ) -> None:
            nonlocal recovery_count
            accepted = capture_once(
                pose,
                source=source,
                role=role,
                cell=cell,
                task_ids=task_ids,
            )
            if (
                accepted
                or circuit_open
                or recovery_count >= budget.max_recovery_attempts
                or not can_capture()
            ):
                return
            recovery_count += 1
            capture_once(
                _recovery_pose(
                    pose,
                    cell,
                    recovery_count,
                    scene_bounds,
                    content_floor_z=content_floor_z,
                ),
                source=ExplorationSource.RECOVERY,
                role=role,
                cell=cell,
                recovery_step=recovery_count,
                task_ids=task_ids,
            )

        # Re-localize UNKNOWN tasks before blind exploration.  Reuse a healthy
        # Stage 2 focus frame when available; otherwise spend at most one new
        # AABB-framed capture per actor while reserving two slots for scene-level
        # coverage.  This is controller routing only, not an embodied rollout.
        represented_actor_ids: set[str] = set()
        for record in tuple(store.records):
            record_actor_ids = frame_actor_ids.get(record.frame_id, set())
            for key, (_, task_ids) in grouped_focus.items():
                if key in record_actor_ids:
                    associate_route(
                        record.frame_id,
                        task_ids=tuple(sorted(task_ids)),
                    )
                    represented_actor_ids.add(key)

        reserve = (
            2
            if require_global_context and budget.max_new_capture_attempts >= 2
            else 0
        )
        focus_limit = max(
            0,
            min(
                budget.max_new_capture_attempts - reserve,
                initial_valid_cap - store.valid_frame_count
                - reserve,
            ),
        )
        focus_attempts = 0
        planned_batch: list[dict[str, Any]] = []
        for key in sorted(grouped_focus):
            if (
                key in represented_actor_ids
                or focus_attempts >= focus_limit
                or len(planned_batch) >= budget.max_new_capture_attempts
            ):
                continue
            target, task_ids = grouped_focus[key]
            poses = plan_bounds_camera_poses(
                target.bounds.center_cm,
                target.bounds.extent_cm,
                scene_bounds,
            )
            existing_poses = tuple(value.pose for value in store.records)
            pose = next(
                (
                    value
                    for value in (poses[2], poses[1], poses[0])
                    if not any(_same_pose(value, existing) for existing in existing_poses)
                    and assess_bounds_camera_pose(
                        target.bounds.center_cm,
                        target.bounds.extent_cm,
                        value,
                    ).healthy
                ),
                None,
            )
            if pose is None:
                continue
            focus_attempts += 1
            planned_batch.append(
                {
                    "pose": pose,
                    "source": ExplorationSource.FOCUS,
                    "role": CaptureShotRole.CONTEXT,
                    "task_ids": tuple(sorted(task_ids)),
                    "actor_ids": (target.actor_id,),
                    "target_bounds": target.bounds,
                    "excluded_poses": existing_poses,
                }
            )

        # Give unlocalized object requirements their own bounded search cells
        # before spending remaining capacity on generic scene coverage.  A
        # generic scout frame is useful context, but it is not proof that one
        # particular missing object was searched for.
        batch_limit = min(
            budget.max_new_capture_attempts,
            initial_valid_cap - store.valid_frame_count,
        )
        planned_interior_cells: set[tuple[int, int]] = set()
        interior_index = 0
        grid_search_index = 0
        while (
            grid_search_index < len(search_tasks)
            and len(planned_batch) < batch_limit
        ):
            covered = {
                _pose_cell(value.pose, scene_bounds) for value in store.records
            }
            covered.update(
                _pose_cell(value["pose"], scene_bounds) for value in planned_batch
            )
            cell = _uncovered_cell(covered, planned_interior_cells)
            if cell is None:
                break
            planned_interior_cells.add(cell)
            planned_batch.append(
                {
                    "pose": _interior_pose(
                        cell,
                        interior_index,
                        scene_bounds,
                        content_floor_z=content_floor_z,
                    ),
                    "source": ExplorationSource.INTERIOR,
                    "role": CaptureShotRole.GRID,
                    "cell": cell,
                    "task_ids": (search_tasks[grid_search_index],),
                }
            )
            interior_index += 1
            grid_search_index += 1

        # Add only genuinely missing holistic poses to the same sweep after
        # requirement-owned evidence. The opposing aerials have priority over
        # duplicate edge coverage.
        planned_scouts: list[CameraPose] = []
        if require_global_context:
            scout_poses = dense_overview_poses or generate_scout_poses(
                scene_bounds,
                floor_z=content_floor_z,
            )
            existing = list(store.records)
            missing_edge_scouts = [
                pose
                for pose in scout_poses[:4]
                if not any(_same_pose(pose, value.pose) for value in existing)
            ]
            existing_global = sum(
                _is_global(value, scene_bounds) for value in existing
            )
            edge_needed = max(0, 4 - existing_global)
            aerial_needed = max(
                0,
                1
                - sum(
                    _is_aerial_pose(value.pose, scene_bounds)
                    for value in existing
                ),
            )
            planned_scouts = [
                *tuple(
                    pose
                    for pose in scout_poses[4:]
                    if not any(_same_pose(pose, value.pose) for value in existing)
                )[:aerial_needed],
                *missing_edge_scouts[:edge_needed],
            ]
        for pose in planned_scouts:
            if len(planned_batch) >= min(
                budget.max_new_capture_attempts,
                initial_valid_cap - store.valid_frame_count,
            ):
                break
            planned_batch.append(
                {
                    "pose": pose,
                    "source": ExplorationSource.SCOUT,
                    "role": CaptureShotRole.OVERVIEW,
                }
            )

        # If focus and scout coverage do not consume the batch, add distinct
        # interior grid cells now rather than returning to UE one pose at a
        # time after the sweep.
        while require_global_context and len(planned_batch) < batch_limit:
            covered = {
                _pose_cell(value.pose, scene_bounds) for value in store.records
            }
            covered.update(
                _pose_cell(value["pose"], scene_bounds) for value in planned_batch
            )
            cell = _uncovered_cell(covered, planned_interior_cells)
            if cell is None:
                break
            planned_interior_cells.add(cell)
            spec: dict[str, Any] = {
                    "pose": _interior_pose(
                        cell,
                        interior_index,
                        scene_bounds,
                        content_floor_z=content_floor_z,
                    ),
                    "source": ExplorationSource.INTERIOR,
                    "role": CaptureShotRole.GRID,
                    "cell": cell,
                }
            if grid_search_index < len(search_tasks):
                spec["task_ids"] = (search_tasks[grid_search_index],)
                grid_search_index += 1
            planned_batch.append(spec)
            interior_index += 1
        capture_planned_batch(planned_batch)

        # Spend the remaining bounded budget on uncovered 5x5 cells.  Planned
        # poses never update coverage; only admitted records do.
        attempted_interior_cells: set[tuple[int, int]] = set()
        interior_index = 0
        while can_capture() and (
            require_global_context or grid_search_index < len(search_tasks)
        ):
            covered = {_pose_cell(value.pose, scene_bounds) for value in store.records}
            cell = _uncovered_cell(covered, attempted_interior_cells)
            if cell is None:
                break
            attempted_interior_cells.add(cell)
            pose = _interior_pose(
                cell,
                interior_index,
                scene_bounds,
                content_floor_z=content_floor_z,
            )
            interior_index += 1
            capture_with_recovery(
                pose,
                source=ExplorationSource.INTERIOR,
                role=CaptureShotRole.GRID,
                cell=cell,
                task_ids=(
                    (search_tasks[grid_search_index],)
                    if grid_search_index < len(search_tasks)
                    else ()
                ),
            )
            if grid_search_index < len(search_tasks):
                grid_search_index += 1

        records = store.records
        default_rank = _deterministic_rank(records)
        ranked_ids = default_rank
        unusable_ids: tuple[str, ...] = ()
        selector_used = self.selector is not None
        selector_error: str | None = None
        if self.selector is not None and records:
            # The selector gets exactly opaque ids and owned RGB, never poses,
            # phase, shot roles, task ids, actor ids, or scene semantics.
            from .stage3_judge import Stage3JudgeFrame

            selector_frames = tuple(
                Stage3JudgeFrame(value.frame_id, value.rgb) for value in records
            )
            try:
                selection = _coerce_selector_result(
                    self.selector.select(selector_frames),
                    tuple(value.frame_id for value in records),
                )
                ranked_ids = selection.ordered_frame_ids
                unusable_ids = selection.unusable_frame_ids
            except Exception as exc:  # noqa: BLE001 - optional selector boundary
                selector_error = f"{type(exc).__name__}: {exc}"
                ranked_ids = default_rank
                unusable_ids = ()

        portfolio_ids, coverage, restored_ids = _restore_selector_coverage(
            store,
            records,
            ranked_ids,
            unusable_ids,
            scene_bounds,
            budget,
            require_global_context=require_global_context,
        )
        # Focus routing is controller-only and does not affect the selector's
        # hard holistic constraints.  Add its auditable counts after selection.
        coverage = _coverage(
            store,
            portfolio_ids,
            scene_bounds,
            budget,
            focus_frame_ids=tuple(frame_actor_ids),
            focused_task_ids=tuple(task_frame_ids),
            require_global_context=require_global_context,
        )
        if restored_ids:
            selector_error = (
                "selector_coverage_recovery: restored controller-accepted frame ids "
                + ", ".join(restored_ids)
            )
        result = Stage3ExplorationResult(
            budget=budget,
            attempts=tuple(attempts),
            coverage=coverage,
            portfolio_frame_ids=portfolio_ids,
            frame_store=store,
            task_frame_ids={
                task_id: tuple(frame_ids)
                for task_id, frame_ids in sorted(task_frame_ids.items())
            },
            frame_actor_ids={
                frame_id: tuple(sorted(actor_ids))
                for frame_id, actor_ids in sorted(frame_actor_ids.items())
            },
            selector_used=selector_used,
            selector_error=selector_error,
            runtime_error=runtime_error,
            global_context_required=require_global_context,
        )
        self.last_result = result
        return result


def capture_stage3_actor_reframes(
    exploration: Stage3ExplorationResult,
    scene_bounds: SceneBounds,
    frame_provider: FrameProvider,
    *,
    targets_by_task: Mapping[str, Sequence[Stage2ActorTarget]],
    content_floor_z: float | None = None,
) -> Stage3ExplorationResult:
    """Batch collection recovery and alternate Actor views after visibility failures.

    This controller-only pass runs after the first structured Stage 3 verdict.
    Every multi-Actor collection first gets one union-AABB recovery view so a
    global relation is not reduced to disconnected object close-ups. Every
    requested Actor then keeps its own AABB for the remaining follow-up focus
    frames. The controller chooses healthy poses distinct from existing views
    and captures all requests in one UE camera sweep.
    Requirements sharing an Actor reuse the same frame through explicit task
    associations. Prompt text and semantic identity never cross into the
    capture provider.
    """

    if not isinstance(exploration, Stage3ExplorationResult):
        raise TypeError("exploration must be a Stage3ExplorationResult")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")
    if content_floor_z is not None and not (
        math.isfinite(float(content_floor_z))
        and scene_bounds.min_z <= float(content_floor_z) <= scene_bounds.max_z
    ):
        raise ValueError("content_floor_z must be finite and inside scene bounds")
    requested = {
        str(task_id).strip(): tuple(targets)
        for task_id, targets in targets_by_task.items()
    }
    if any(
        not task_id
        or not targets
        or any(not isinstance(value, Stage2ActorTarget) for value in targets)
        for task_id, targets in requested.items()
    ):
        raise TypeError(
            "targets_by_task must map non-empty task ids to Stage2ActorTarget sequences"
        )

    budget = exploration.budget
    store = exploration.frame_store
    capacity = store.valid_frame_hard_cap - store.valid_frame_count
    if capacity <= 0 or not requested:
        return exploration
    capture = getattr(frame_provider, "capture", None)
    if not callable(capture):
        raise TypeError("frame_provider must provide capture(poses, ...)")

    task_routes = {
        task_id: list(frame_ids)
        for task_id, frame_ids in exploration.task_frame_ids.items()
    }
    actor_routes = {
        frame_id: set(actor_ids)
        for frame_id, actor_ids in exploration.frame_actor_ids.items()
    }
    attempts = list(exploration.attempts)
    attempt_number = (
        max(
            (
                int(value.attempt_id.removeprefix("s3a_"))
                for value in attempts
            ),
            default=0,
        )
        + 1
    )

    def append_attempt(
        spec: Mapping[str, Any],
        status: ExplorationAttemptStatus,
        *,
        frame_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        nonlocal attempt_number
        attempts.append(
            Stage3ExplorationAttempt(
                attempt_id=f"s3a_{attempt_number:06d}",
                source=ExplorationSource.REFRAME,
                pose=spec["pose"],
                shot_role=spec["role"],
                status=status,
                frame_id=frame_id,
                reason=reason,
            )
        )
        attempt_number += 1

    specs: list[dict[str, Any]] = []
    planned_poses: list[CameraPose] = []
    spec_by_actor: dict[str, dict[str, Any]] = {}
    spec_by_collection: dict[tuple[str, ...], dict[str, Any]] = {}
    for task_id in sorted(requested):
        routed_actor_ids = {
            actor_id
            for frame_id in task_routes.get(task_id, ())
            for actor_id in actor_routes.get(frame_id, ())
        }
        ordered_targets = tuple(
            sorted(
                requested[task_id],
                key=lambda value: (
                    value.actor_id.casefold() not in routed_actor_ids,
                    value.actor_id.casefold(),
                ),
            )
        )
        collection_key = tuple(
            sorted(value.actor_id.casefold() for value in ordered_targets)
        )
        if len(ordered_targets) > 1:
            shared_collection = spec_by_collection.get(collection_key)
            if shared_collection is not None:
                shared_collection["task_ids"].append(task_id)
            elif len(specs) < capacity:
                collection_bounds = union_actor_bounds(ordered_targets)
                minimum_context_elevation = (
                    12.0
                    if content_floor_z is not None
                    and collection_bounds.min_cm[2]
                    <= float(content_floor_z) + 20.0
                    else 0.0
                )
                collection_poses = plan_bounds_camera_poses(
                    collection_bounds.center_cm,
                    collection_bounds.extent_cm,
                    scene_bounds,
                    minimum_context_elevation_degrees=minimum_context_elevation,
                )
                existing_collection_poses = tuple(
                    store.get(frame_id).pose
                    for frame_id in task_routes.get(task_id, ())
                )
                viable_collection_poses = tuple(
                    value
                    for value in collection_poses
                    if not any(
                        _same_pose(value, existing)
                        for existing in (
                            *existing_collection_poses,
                            *planned_poses,
                        )
                    )
                    and assess_bounds_camera_pose(
                        collection_bounds.center_cm,
                        collection_bounds.extent_cm,
                        value,
                    ).healthy
                )
                if viable_collection_poses:
                    collection_pose = max(
                        viable_collection_poses,
                        key=lambda value: min(
                            (
                                _pose_distance(value, existing)
                                for existing in existing_collection_poses
                            ),
                            default=float("inf"),
                        ),
                    )
                    planned_poses.append(collection_pose)
                    collection_spec = {
                        "task_ids": [task_id],
                        "targets": ordered_targets,
                        "bounds": collection_bounds,
                        "pose": collection_pose,
                        "role": CaptureShotRole.COLLECTION_WIDE,
                        "excluded_poses": existing_collection_poses,
                    }
                    specs.append(collection_spec)
                    spec_by_collection[collection_key] = collection_spec
        shared_specs: list[dict[str, Any]] = []
        for target in ordered_targets:
            shared = spec_by_actor.get(target.actor_id.casefold())
            if shared is not None and shared not in shared_specs:
                shared_specs.append(shared)
        for shared in shared_specs:
            shared["task_ids"].append(task_id)
        remaining_targets = tuple(
            target
            for target in ordered_targets
            if target.actor_id.casefold() not in spec_by_actor
        )
        if not remaining_targets or len(specs) >= capacity:
            continue
        existing_poses = tuple(
            store.get(frame_id).pose
            for frame_id in task_routes.get(task_id, ())
        )
        for target in remaining_targets:
            if len(specs) >= capacity:
                break
            bounds = target.bounds
            context, close, alternate = plan_bounds_camera_poses(
                bounds.center_cm,
                bounds.extent_cm,
                scene_bounds,
            )
            pose = next(
                (
                    value
                    for value in (alternate, context, close)
                    if not any(
                        _same_pose(value, existing)
                        for existing in (*existing_poses, *planned_poses)
                    )
                    and assess_bounds_camera_pose(
                        bounds.center_cm,
                        bounds.extent_cm,
                        value,
                    ).healthy
                ),
                None,
            )
            if pose is None:
                continue
            planned_poses.append(pose)
            spec = {
                "task_ids": [task_id],
                "targets": (target,),
                "bounds": bounds,
                "pose": pose,
                "role": CaptureShotRole.OBLIQUE,
                "excluded_poses": existing_poses,
            }
            specs.append(spec)
            spec_by_actor[target.actor_id.casefold()] = spec

    if not specs:
        return exploration

    runtime_error = exploration.runtime_error
    captured_values: tuple[Any, ...] = ()
    capture_specs: tuple[Mapping[str, Any], ...] = ()
    try:
        resolved_poses = _resolve_actor_camera_poses(
            frame_provider,
            tuple(value["pose"] for value in specs),
            tuple(
                tuple(target.actor_id for target in value["targets"])
                for value in specs
            ),
        )
        ready_specs: list[dict[str, Any]] = []
        for spec, resolved_pose in zip(specs, resolved_poses, strict=True):
            if resolved_pose is None:
                resolved_pose, recovery_candidates = (
                    _resolve_actor_camera_pose_with_recovery(
                        frame_provider,
                        tuple(
                            target.actor_id for target in spec["targets"]
                        ),
                        spec["bounds"],
                        scene_bounds,
                        excluded_poses=(
                            *tuple(planned_poses),
                            *tuple(spec.get("excluded_poses", ())),
                        ),
                    )
                )
                if resolved_pose is None:
                    append_attempt(
                        spec,
                        ExplorationAttemptStatus.REJECTED,
                        reason=(
                            "camera_pose_unresolved:"
                            f"{recovery_candidates}_recovery_candidates"
                        ),
                    )
                    continue
            ready_specs.append({**spec, "pose": resolved_pose})
        capture_specs = tuple(ready_specs)
        if not capture_specs:
            raise RgbFrameHealthError(
                "camera_pose_unresolved",
                "no adaptive Actor reframe has an unobstructed camera pose",
            )
        captured_values = tuple(
            capture(
                tuple(value["pose"] for value in capture_specs),
                phase="stage3_adaptive_reframe",
                frame_id_prefix="s3_reframe",
            )
        )
        if len(captured_values) != len(capture_specs):
            raise RuntimeError(
                "provider returned "
                f"{len(captured_values)} frames for "
                f"{len(capture_specs)} adaptive reframes"
            )
        if any(
            captured.pose != spec["pose"]
            for spec, captured in zip(capture_specs, captured_values, strict=True)
        ):
            raise RuntimeError("provider changed adaptive reframe pose order")
    except RgbFrameHealthError as exc:
        for spec in capture_specs:
            append_attempt(
                spec,
                ExplorationAttemptStatus.REJECTED,
                reason=str(getattr(exc, "reason", "rgb_health_error")),
            )
    except Exception as exc:  # noqa: BLE001 - provider failure boundary
        runtime_error = f"adaptive_reframe_capture_failed: {type(exc).__name__}: {exc}"
        for spec in capture_specs or tuple(specs):
            append_attempt(
                spec,
                ExplorationAttemptStatus.CAPTURE_ERROR,
                reason=runtime_error,
            )
    else:
        for spec, captured in zip(capture_specs, captured_values, strict=True):
            targets = spec["targets"]
            admission = store.admit_captured_frame(
                captured,
                phase="stage3_adaptive_reframe",
                shot_role=spec["role"],
            )
            frame_id = (
                admission.record.frame_id
                if admission.record is not None
                else admission.duplicate_of_frame_id
            )
            if frame_id is not None:
                for task_id in spec["task_ids"]:
                    task_routes.setdefault(task_id, []).append(frame_id)
                    task_routes[task_id] = list(
                        dict.fromkeys(task_routes[task_id])
                    )
                actor_routes.setdefault(frame_id, set()).update(
                    target.actor_id.casefold() for target in targets
                )
            if admission.record is not None:
                append_attempt(
                    spec,
                    ExplorationAttemptStatus.ACCEPTED,
                    frame_id=admission.record.frame_id,
                )
            else:
                reason = (
                    admission.rejection_reason.value
                    if admission.rejection_reason is not None
                    else "rejected"
                )
                append_attempt(
                    spec,
                    (
                        ExplorationAttemptStatus.DUPLICATE
                        if admission.rejection_reason
                        is FrameRejectionReason.NEAR_DUPLICATE
                        else ExplorationAttemptStatus.REJECTED
                    ),
                    reason=reason,
                )

    coverage = _coverage(
        store,
        exploration.portfolio_frame_ids,
        scene_bounds,
        budget,
        focus_frame_ids=tuple(actor_routes),
        focused_task_ids=tuple(task_routes),
        require_global_context=exploration.global_context_required,
    )
    return Stage3ExplorationResult(
        budget=budget,
        attempts=tuple(attempts),
        coverage=coverage,
        portfolio_frame_ids=exploration.portfolio_frame_ids,
        frame_store=store,
        task_frame_ids={
            task_id: tuple(frame_ids)
            for task_id, frame_ids in sorted(task_routes.items())
        },
        frame_actor_ids={
            frame_id: tuple(sorted(actor_ids))
            for frame_id, actor_ids in sorted(actor_routes.items())
        },
        selector_used=exploration.selector_used,
        selector_error=exploration.selector_error,
        runtime_error=runtime_error,
        global_context_required=exploration.global_context_required,
    )


# Backward-compatible public name used by the merged RequirementGraph pipeline.
Stage3Explorer = HolisticExplorer


def explore_stage3_scene(
    scene_bounds: SceneBounds,
    frame_provider: FrameProvider,
    *,
    stage2_frame_store: Any | None = None,
    focus_targets_by_task: Mapping[
        str, Sequence[Stage2ActorTarget]
    ] | None = None,
    grid_search_task_ids: Sequence[str] = (),
    require_global_context: bool = True,
    content_floor_z: float | None = None,
    adaptive_reframe_reserve: int = 0,
    budget: Stage3Budget | None = None,
    selector: HolisticFrameSelector | None = None,
) -> Stage3ExplorationResult:
    """Functional wrapper around :class:`HolisticExplorer`."""

    return HolisticExplorer(budget=budget, selector=selector).explore(
        scene_bounds,
        frame_provider,
        stage2_frame_store=stage2_frame_store,
        focus_targets_by_task=focus_targets_by_task,
        grid_search_task_ids=grid_search_task_ids,
        require_global_context=require_global_context,
        content_floor_z=content_floor_z,
        adaptive_reframe_reserve=adaptive_reframe_reserve,
    )


__all__ = [
    "MIN_DISTINCT_PORTFOLIO_POSES",
    "MIN_GLOBAL_PORTFOLIO_FRAMES",
    "MIN_INTERIOR_PORTFOLIO_FRAMES",
    "RPC_CIRCUIT_BREAK_THRESHOLD",
    "HolisticExplorer",
    "HolisticFrameSelector",
    "Stage3Explorer",
    "capture_stage3_actor_reframes",
    "explore_stage3_scene",
]
