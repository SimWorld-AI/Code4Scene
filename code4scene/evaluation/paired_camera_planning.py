"""Environment-aware absolute camera planning for GT-repair evidence.

The task chooses only the physical topology (indoor/outdoor). Geometry and
live UE traces choose verifier-planned cameras. Authored cameras are admitted
only when a same-pose render of the current task's Input and GT proves the edit
observable; packaged release QA remains provenance, not an online dependency.
Overlap/enclosure remain hard safety gates, while framing and target-identity
LOS are advisory. Prompt text, Candidate contents and model judgement never
choose a camera. Every resulting portfolio is frozen once and reused unchanged
for Candidate and GT.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import render, scene_graph_capture
from .requirement_graph.actor_inventory import (
    ActorBounds,
    ActorDescriptor,
    ActorInventorySnapshot,
)
from .requirement_graph.asset_candidates import assess_bounds_camera_pose
from .requirement_graph.camera_visibility import (
    camera_pose_key,
    resolve_visibility_camera_poses,
    visibility_pose_candidates,
)
from .requirement_graph.contracts import CameraPose, SceneBounds
from .requirement_graph.evidence_adapter import adapt_scene_snapshot
from .scene_diff import ActorIdentityError, actor_identity, index_actors


_INDOOR_ANCHOR_EXCLUDED_TOKENS = (
    "camera",
    "ceiling",
    "decal",
    "floor",
    "fog",
    "groupactor",
    "landscape",
    "light",
    "postprocess",
    "reflectioncapture",
    "roof",
    "sky",
    "spline",
    "volume",
    "wall",
    "worldsettings",
)
_MAX_INDOOR_ANCHOR_EXTENT_CM = 2_500.0
_MIN_INDOOR_ANCHOR_EXTENT_CM = 2.0
_MIN_INDOOR_ANCHOR_ASPECT_RATIO = 0.08
_INDOOR_ROOM_MIN_SPAN_CM = 240.0
_INDOOR_ROOM_MIN_HEIGHT_CM = 180.0
_INDOOR_CAMERA_HEIGHT_CM = 150.0
_INDOOR_CAMERA_MIN_INSET_CM = 70.0
_INDOOR_CAMERA_MAX_INSET_CM = 140.0
_INDOOR_CAMERA_INSET_FRACTION = 0.14
_INDOOR_CAMERA_FALLBACK_OFFSETS_CM = (
    (0.0, 0.0),
    (45.0, 0.0),
    (0.0, 45.0),
    (45.0, 45.0),
    (90.0, 45.0),
    (45.0, 90.0),
    (90.0, 90.0),
)
_INDOOR_HANGING_CEILING_GAP_CM = 45.0
_INDOOR_HANGING_MIN_HEIGHT_FRACTION = 0.55
_INDOOR_HANGING_CAMERA_MARGIN_CM = 35.0
_INDOOR_HANGING_CAMERA_MIN_FLOOR_CLEARANCE_CM = 60.0
_INDOOR_HANGING_CAMERA_MIN_CEILING_CLEARANCE_CM = 45.0
_INDOOR_HANGING_VIEW_VARIANTS = (
    (0.60, -18.0),
    (0.60, -30.0),
    (0.72, -24.0),
)
_INDOOR_HANGING_AZIMUTH_OFFSETS_DEG = (
    0.0,
    15.0,
    -15.0,
    30.0,
    -30.0,
    45.0,
    -45.0,
    60.0,
    -60.0,
    75.0,
    -75.0,
    90.0,
    -90.0,
    105.0,
    -105.0,
    120.0,
    -120.0,
    135.0,
    -135.0,
    150.0,
    -150.0,
    165.0,
    -165.0,
    180.0,
)
_INDOOR_HANGING_MAX_ADDITIONAL_CANDIDATES = (
    len(_INDOOR_HANGING_VIEW_VARIANTS)
    * len(_INDOOR_HANGING_AZIMUTH_OFFSETS_DEG)
)
_INDOOR_HANGING_REQUIRED_ACTOR_CLEAR_FRACTION = 0.20
_INDOOR_FALLBACK_REQUIRED_ACTOR_CLEAR_FRACTION = 0.20
_INDOOR_LOCAL_FALLBACK_REQUIRED_ACTOR_CLEAR_FRACTION = 0.10
# Legacy case cameras without paired pixel QA need one direct target ray out of
# the fixed fifteen-point probe. QA-attested poses retain this trace as an
# advisory while overlap and enclosure continue to fail closed.
_INDOOR_CASE_CAMERA_REQUIRED_ACTOR_CLEAR_FRACTION = 1.0 / 15.0
_INDOOR_FALLBACK_MAX_CANDIDATES_PER_VIEW = 64
_INDOOR_THIN_TARGET_MAX_HALF_THICKNESS_CM = 5.0
_INDOOR_THIN_TARGET_MIN_ASPECT_RATIO = 4.0
_INDOOR_AUTHORED_CAMERA_NEIGHBORHOOD_OFFSETS_CM = (
    (0.0, 0.0),
    (-20.0, 0.0),
    (20.0, 0.0),
    (0.0, -15.0),
    (0.0, 15.0),
)


@dataclass(frozen=True, slots=True)
class CaseCameraPortfolio:
    """Integrity-checked authored views eligible for exact-pose reuse."""

    views: tuple[render.Viewpoint, ...]
    manifest_sha256: str
    schema_version: str
    authored_view_names: tuple[str, ...]
    paired_render_qa_sha256: str | None = None
    paired_render_qa_by_view: tuple[Mapping[str, Any], ...] = ()
    runtime_neighbor_sources: tuple[str | None, ...] = ()


def case_camera_portfolio(task: Any) -> CaseCameraPortfolio | None:
    """Convert task-bound camera metadata and optional release QA to inputs."""

    manifest = getattr(task, "case_camera_manifest", None)
    if manifest is None:
        return None
    digest = str(getattr(task, "case_camera_manifest_sha256", None) or "")
    paired_qa = getattr(task, "paired_render_qa", None)
    paired_qa_digest = getattr(task, "paired_render_qa_sha256", None)
    views = []
    names = []
    for index, spec in enumerate(manifest["views"]):
        location = tuple(float(value) for value in spec["location_cm"])
        target = tuple(float(value) for value in spec["target_cm"])
        pose = _look_at_room_pose(location, target)
        views.append(
            _view(
                f"case_view_{index + 1}",
                pose,
                float(spec["fov_deg"]),
            )
        )
        names.append(str(spec["name"]))
    return CaseCameraPortfolio(
        views=tuple(views),
        manifest_sha256=digest,
        schema_version=str(manifest["schema_version"]),
        authored_view_names=tuple(names),
        paired_render_qa_sha256=(
            str(paired_qa_digest) if paired_qa is not None else None
        ),
        paired_render_qa_by_view=(
            tuple(dict(value) for value in paired_qa["per_view"])
            if paired_qa is not None
            else ()
        ),
        runtime_neighbor_sources=tuple(None for _value in views),
    )


def expand_case_camera_runtime_neighborhood(
    portfolio: CaseCameraPortfolio,
    target_actors: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> CaseCameraPortfolio:
    """Add an overcomplete, rotation-only neighborhood around authored views.

    The authored camera location is already known to be usable enough to
    produce the release view. Moving that location even a few centimetres can
    put a generated camera inside a shelf, wall, or prop. Small yaw/pitch
    changes preserve the safe location while giving the runtime Input/GT trial
    enough distinct candidates to select the strict four-view portfolio.
    """

    if len(portfolio.views) >= count:
        return portfolio
    if len(portfolio.views) < 2:
        raise ValueError(
            "runtime case-camera neighborhood needs at least two authored views"
        )
    # Resolve the target bounds before constructing the trial portfolio so an
    # empty or malformed target set still fails at the same boundary as the
    # rest of the current planner. The rotation neighborhood itself intentionally
    # stays relative to the authored aim: runtime Input/GT pixels, rather than
    # a possibly drifted actor identity, decide whether the edit remains seen.
    _target_actor_bounds(target_actors)
    views = list(portfolio.views)
    names = list(portfolio.authored_view_names)
    sources = list(
        portfolio.runtime_neighbor_sources
        or (None for _value in portfolio.views)
    )
    used = {camera_pose_key(_pose(value)) for value in views}
    rotation_offsets_deg = (
        (0.0, 2.0),
        (0.0, -2.0),
        (2.0, 0.0),
        (-2.0, 0.0),
        (1.5, 1.5),
        (-1.5, -1.5),
    )
    trial_count = max(count, len(portfolio.views) * 4)
    for offset_index, (pitch_offset, yaw_offset) in enumerate(
        rotation_offsets_deg
    ):
        seed_index = offset_index % len(portfolio.views)
        seed = portfolio.views[seed_index]
        seed_pose = _pose(seed)
        pose = CameraPose(
            seed_pose.x,
            seed_pose.y,
            seed_pose.z,
            seed_pose.pitch + pitch_offset,
            seed_pose.yaw + yaw_offset,
            seed_pose.roll,
        )
        key = camera_pose_key(pose)
        if key in used:
            continue
        used.add(key)
        views.append(
            _view(
                f"case_view_{len(views) + 1}",
                pose,
                seed.fov_deg,
            )
        )
        source_name = portfolio.authored_view_names[seed_index]
        names.append(
            f"{source_name}__runtime-neighbor-{len(views) - len(portfolio.views)}"
        )
        sources.append(source_name)
        if len(views) == trial_count:
            break
    if len(views) < count:
        raise ValueError(
            "runtime case-camera neighborhood produced only "
            f"{len(views)}/{count} required views"
        )
    return CaseCameraPortfolio(
        views=tuple(views),
        manifest_sha256=portfolio.manifest_sha256,
        schema_version=portfolio.schema_version,
        authored_view_names=tuple(names),
        paired_render_qa_sha256=portfolio.paired_render_qa_sha256,
        paired_render_qa_by_view=portfolio.paired_render_qa_by_view,
        runtime_neighbor_sources=tuple(sources),
    )


def case_camera_trial_plan(
    portfolio: CaseCameraPortfolio,
    target_actors: Sequence[Mapping[str, Any]],
) -> scene_graph_capture.CameraPlan:
    """Freeze exact authored poses for Candidate-blind Input/GT trial renders."""

    if not portfolio.views:
        raise ValueError("case camera portfolio has no authored views")
    target_bounds = _target_actor_bounds(target_actors)
    preferred = scene_graph_capture.CameraPlan(
        portfolio.views,
        target_bounds.center_cm,
        {
            "planner_id": scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN,
            "pairing_policy": (
                "current-task-input-gt-authored-camera-observability-trial"
            ),
            "planning_source": "integrity_pinned_case_camera_manifest",
            "case_camera_manifest_schema": portfolio.schema_version,
            "case_camera_manifest_sha256": portfolio.manifest_sha256,
            "case_camera_paired_render_qa_sha256": (
                portfolio.paired_render_qa_sha256
            ),
            "case_camera_release_qa_role": "optional_provenance_only",
            "case_camera_authored_view_names": list(
                portfolio.authored_view_names
            ),
            "case_camera_runtime_neighbor_sources": list(
                portfolio.runtime_neighbor_sources
            ),
        },
    )
    return scene_graph_capture.freeze_visibility_camera_plan(
        preferred,
        portfolio.views,
        planner_id=scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN,
        pairing_policy="current-task-input-gt-authored-camera-observability-trial",
        planning_source="integrity_pinned_case_camera_manifest",
        scene_environment="indoor",
        visibility_audit=tuple(
            {
                "view": view.name,
                "authored_view": authored_name,
                "selection": "exact_pose_runtime_input_gt_trial",
                "safety_acceptance": "deferred-to-live-planner",
            }
            for view, authored_name in zip(
                portfolio.views,
                portfolio.authored_view_names,
                strict=True,
            )
        ),
        visibility_policy="runtime-input-gt-trial-exact-authored-pose",
    )


class _CameraPortfolioUnavailable(ValueError):
    """The requested portfolio was valid, but UE found no usable poses."""


@dataclass(frozen=True, slots=True)
class _IndoorRoomFrame:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    floor_z: float
    ceiling_z: float
    boundary_actor_id: str
    boundary_kind: str

    @property
    def center_cm(self) -> tuple[float, float, float]:
        return (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            self.floor_z + 0.45 * (self.ceiling_z - self.floor_z),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_cm": [self.min_x, self.min_y, self.floor_z],
            "max_cm": [self.max_x, self.max_y, self.ceiling_z],
            "center_cm": list(self.center_cm),
            "boundary_actor_id": self.boundary_actor_id,
            "boundary_kind": self.boundary_kind,
        }


@dataclass(frozen=True, slots=True)
class _ThinTargetSupportFrame:
    actor_id: str
    thin_axis: int
    exposed_sign: int
    support_center_cm: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "thin_axis": self.thin_axis,
            "exposed_sign": self.exposed_sign,
            "support_center_cm": list(self.support_center_cm),
        }


def require_scene_environment(task: Any) -> str:
    """Return the explicit I2S topology or fail before any capture is scored."""

    value = getattr(task, "scene_environment", None)
    value = str(value).strip().casefold() if value is not None else ""
    if value not in {"indoor", "outdoor"}:
        raise ValueError(
            "current image-to-scene capture requires explicit "
            "scene_environment: indoor|outdoor"
        )
    return value


def _pose(view: render.Viewpoint) -> CameraPose:
    return CameraPose(
        float(view.location[0]),
        float(view.location[1]),
        float(view.location[2]),
        float(view.rotation[1]),
        float(view.rotation[2]),
        float(view.rotation[0]),
    )


def _view(name: str, pose: CameraPose, fov_deg: float) -> render.Viewpoint:
    return render.Viewpoint(
        name=name,
        location=[pose.x, pose.y, pose.z],
        rotation=[pose.roll, pose.pitch, pose.yaw],
        fov_deg=float(fov_deg),
    )


def _unchanged_signature(actor: Mapping[str, Any]) -> tuple[Any, ...]:
    """Fields whose equality proves that an overview anchor was not edited."""

    bounds = actor.get("bounds") if isinstance(actor.get("bounds"), Mapping) else {}
    transform = (
        actor.get("transform")
        if isinstance(actor.get("transform"), Mapping)
        else {}
    )
    return (
        str(actor.get("class") or ""),
        str(actor.get("asset_path") or ""),
        tuple(str(value) for value in actor.get("component_asset_paths") or ()),
        tuple(float(value) for value in bounds.get("origin_cm") or ()),
        tuple(float(value) for value in bounds.get("extent_cm") or ()),
        tuple(float(value) for value in transform.get("location_cm") or ()),
        tuple(float(value) for value in transform.get("rotation_deg") or ()),
        tuple(float(value) for value in transform.get("scale") or ()),
    )


def _is_indoor_anchor(descriptor: ActorDescriptor) -> bool:
    bounds = descriptor.bounds
    if (
        bounds is None
        or descriptor.renderable is not True
        or not descriptor.asset_path
    ):
        return False
    extents = tuple(float(value) for value in bounds.extent_cm)
    if min(extents) < _MIN_INDOOR_ANCHOR_EXTENT_CM:
        return False
    if max(extents) > _MAX_INDOOR_ANCHOR_EXTENT_CM:
        return False
    # Doors, wall panels, glazing and other structural sheets can be bounded
    # and renderable while still being poor room-side camera anchors.  Use a
    # prompt-independent geometry gate instead of fragile asset-name matching
    # (for example, ``door`` is also a substring of ``indoor``).
    if min(extents) / max(extents) < _MIN_INDOOR_ANCHOR_ASPECT_RATIO:
        return False
    text = " ".join(
        str(value or "").casefold()
        for value in (
            descriptor.actor_class,
            descriptor.asset_path,
            descriptor.actor_label,
            descriptor.unreal_name,
        )
    )
    return not any(token in text for token in _INDOOR_ANCHOR_EXCLUDED_TOKENS)


def shared_indoor_anchor_ids(
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
    inventory: ActorInventorySnapshot,
    *,
    count: int,
) -> tuple[str, ...]:
    """Choose deterministic, spatially distributed unedited interior props."""

    input_index = index_actors(input_scene)
    gt_index = index_actors(gt_scene)
    unchanged = {
        actor_id
        for actor_id in input_index.keys() & gt_index.keys()
        if _unchanged_signature(input_index[actor_id])
        == _unchanged_signature(gt_index[actor_id])
    }
    candidates = tuple(
        descriptor
        for descriptor in inventory.actors
        if descriptor.live_actor_id in unchanged and _is_indoor_anchor(descriptor)
    )
    if not candidates:
        raise ValueError(
            "indoor overview has no bounded, renderable, unedited prop anchor"
        )

    def prominence(value: ActorDescriptor) -> float:
        assert value.bounds is not None
        ex, ey, ez = value.bounds.extent_cm
        return max(ex, ey) * max(ez, 1.0)

    ranked = sorted(
        candidates,
        key=lambda value: (-prominence(value), value.live_actor_id.casefold()),
    )
    selected = [ranked[0]]
    while len(selected) < min(count, len(ranked)):
        remaining = [value for value in ranked if value not in selected]
        selected.append(
            max(
                remaining,
                key=lambda value: (
                    min(
                        math.dist(
                            value.bounds.center_cm[:2],
                            other.bounds.center_cm[:2],
                        )
                        for other in selected
                    ),
                    prominence(value),
                ),
            )
        )
    return tuple(value.live_actor_id for value in selected)


def _raw_actor_bounds(
    actor: Mapping[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    bounds = actor.get("bounds") if isinstance(actor.get("bounds"), Mapping) else {}
    center = bounds.get("origin_cm")
    extent = bounds.get("extent_cm")
    if (
        not isinstance(center, Sequence)
        or isinstance(center, (str, bytes))
        or not isinstance(extent, Sequence)
        or isinstance(extent, (str, bytes))
        or len(center) != 3
        or len(extent) != 3
    ):
        return None
    try:
        center_value = tuple(float(value) for value in center)
        extent_value = tuple(abs(float(value)) for value in extent)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (*center_value, *extent_value)):
        return None
    return center_value, extent_value


def _raw_actor_text(actor: Mapping[str, Any]) -> str:
    return " ".join(
        str(value or "").casefold()
        for value in (
            actor.get("class"),
            actor.get("asset_path"),
            actor.get("label"),
            actor.get("name"),
        )
    )


def _edited_focus_cm(
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
) -> tuple[float, float, float]:
    input_index = index_actors(input_scene)
    gt_index = index_actors(gt_scene)
    points: list[tuple[float, float, float]] = []
    for actor_id in sorted(input_index.keys() | gt_index.keys()):
        input_actor = input_index.get(actor_id)
        gt_actor = gt_index.get(actor_id)
        if (
            input_actor is not None
            and gt_actor is not None
            and _unchanged_signature(input_actor) == _unchanged_signature(gt_actor)
        ):
            continue
        actor = gt_actor if gt_actor is not None else input_actor
        if actor is None or (value := _raw_actor_bounds(actor)) is None:
            continue
        points.append(value[0])
    if not points:
        raise ValueError("indoor overview cannot locate the edited room")
    return tuple(
        sum(value[axis] for value in points) / len(points)
        for axis in range(3)
    )


def _shared_structural_actors(
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    input_index = index_actors(input_scene)
    gt_index = index_actors(gt_scene)
    return tuple(
        (actor_id, gt_index[actor_id])
        for actor_id in sorted(input_index.keys() & gt_index.keys())
        if _unchanged_signature(input_index[actor_id])
        == _unchanged_signature(gt_index[actor_id])
    )


def _infer_indoor_room_frame(
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
) -> _IndoorRoomFrame:
    """Infer the edited room from shared floor/wall geometry, never Candidate."""

    focus = _edited_focus_cm(input_scene, gt_scene)
    shared = _shared_structural_actors(input_scene, gt_scene)

    def contains_xy(
        value: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> bool:
        center, extent = value
        return (
            center[0] - extent[0] <= focus[0] <= center[0] + extent[0]
            and center[1] - extent[1] <= focus[1] <= center[1] + extent[1]
        )

    floor_candidates = []
    wall_candidates = []
    ceiling_candidates = []
    for actor_id, actor in shared:
        value = _raw_actor_bounds(actor)
        if value is None:
            continue
        center, extent = value
        text = _raw_actor_text(actor)
        horizontal_area = 4.0 * extent[0] * extent[1]
        if (
            "floor" in text
            and min(extent[0], extent[1]) >= _INDOOR_ROOM_MIN_SPAN_CM / 2.0
            and extent[2] <= 40.0
            and contains_xy(value)
        ):
            floor_candidates.append((horizontal_area, actor_id, center, extent))
        if (
            "wall" in text
            and min(extent[0], extent[1]) >= _INDOOR_ROOM_MIN_SPAN_CM / 2.0
            and extent[2] >= _INDOOR_ROOM_MIN_HEIGHT_CM / 2.0
            and contains_xy(value)
        ):
            wall_candidates.append((horizontal_area, actor_id, center, extent))
        if "ceiling" in text and contains_xy(value):
            ceiling_candidates.append((center[2] - extent[2], actor_id))

    if floor_candidates:
        _area, boundary_id, center, extent = min(floor_candidates)
        floor_z = center[2] + extent[2]
        boundary_kind = "shared_floor_bounds"
    elif wall_candidates:
        _area, boundary_id, center, extent = min(wall_candidates)
        floor_z = center[2] - extent[2]
        boundary_kind = "shared_wall_envelope"
    else:
        raise ValueError(
            "indoor overview has no shared floor or enclosing wall geometry "
            "for the edited room"
        )

    ceiling_values = sorted(
        value
        for value, _actor_id in ceiling_candidates
        if value >= floor_z + _INDOOR_ROOM_MIN_HEIGHT_CM
    )
    ceiling_z = (
        ceiling_values[0]
        if ceiling_values
        else max(floor_z + _INDOOR_ROOM_MIN_HEIGHT_CM, center[2] + extent[2])
    )
    frame = _IndoorRoomFrame(
        center[0] - extent[0],
        center[1] - extent[1],
        center[0] + extent[0],
        center[1] + extent[1],
        floor_z,
        ceiling_z,
        boundary_id,
        boundary_kind,
    )
    if (
        frame.max_x - frame.min_x < _INDOOR_ROOM_MIN_SPAN_CM
        or frame.max_y - frame.min_y < _INDOOR_ROOM_MIN_SPAN_CM
        or frame.ceiling_z - frame.floor_z < _INDOOR_ROOM_MIN_HEIGHT_CM
    ):
        raise ValueError(f"indoor room bounds are degenerate: {frame.to_dict()}")
    return frame


def _look_at_room_pose(
    location: tuple[float, float, float],
    target: tuple[float, float, float],
) -> CameraPose:
    dx = target[0] - location[0]
    dy = target[1] - location[1]
    dz = target[2] - location[2]
    horizontal = math.hypot(dx, dy)
    return CameraPose(
        location[0],
        location[1],
        location[2],
        math.degrees(math.atan2(dz, max(horizontal, 1e-6))),
        math.degrees(math.atan2(dy, dx)),
        0.0,
    )


def _room_corner_candidate_groups(
    frame: _IndoorRoomFrame,
) -> tuple[tuple[CameraPose, ...], ...]:
    width = frame.max_x - frame.min_x
    depth = frame.max_y - frame.min_y
    inset = min(
        _INDOOR_CAMERA_MAX_INSET_CM,
        max(
            _INDOOR_CAMERA_MIN_INSET_CM,
            _INDOOR_CAMERA_INSET_FRACTION * min(width, depth),
        ),
    )
    camera_z = min(
        frame.ceiling_z - 45.0,
        frame.floor_z + _INDOOR_CAMERA_HEIGHT_CM,
    )
    target = frame.center_cm
    groups = []
    for x_side, y_side in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        candidates = []
        for x_extra, y_extra in _INDOOR_CAMERA_FALLBACK_OFFSETS_CM:
            x = (
                frame.min_x + inset + x_extra
                if x_side < 0
                else frame.max_x - inset - x_extra
            )
            y = (
                frame.min_y + inset + y_extra
                if y_side < 0
                else frame.max_y - inset - y_extra
            )
            # Keep every fallback in its original room quadrant so four views
            # cannot silently collapse to one central camera.
            x = min(target[0] - 20.0, x) if x_side < 0 else max(target[0] + 20.0, x)
            y = min(target[1] - 20.0, y) if y_side < 0 else max(target[1] + 20.0, y)
            pose = _look_at_room_pose((x, y, camera_z), target)
            if pose not in candidates:
                candidates.append(pose)
        groups.append(tuple(candidates))
    return tuple(groups)


def _target_actor_bounds(
    target_actors: Sequence[Mapping[str, Any]],
) -> ActorBounds:
    values = tuple(
        value
        for actor in target_actors
        if (value := _raw_actor_bounds(actor)) is not None
    )
    if not values:
        raise ValueError("Indoor local planner needs bounded repair target Actors")
    minimum = tuple(
        min(center[axis] - extent[axis] for center, extent in values)
        for axis in range(3)
    )
    maximum = tuple(
        max(center[axis] + extent[axis] for center, extent in values)
        for axis in range(3)
    )
    return ActorBounds.from_min_max(minimum, maximum)


def _is_hanging_target(bounds: ActorBounds, room: _IndoorRoomFrame) -> bool:
    """Identify upper-room targets without using labels, prompts, or Candidate."""

    room_height = room.ceiling_z - room.floor_z
    center_fraction = (bounds.center_cm[2] - room.floor_z) / room_height
    ceiling_gap = room.ceiling_z - bounds.max_cm[2]
    return (
        center_fraction >= _INDOOR_HANGING_MIN_HEIGHT_FRACTION
        and ceiling_gap <= _INDOOR_HANGING_CEILING_GAP_CM
    )


def _is_vertical_thin_target(bounds: ActorBounds) -> bool:
    """Return whether GT bounds describe a wall-like thin repair target."""

    extent_x, extent_y, extent_z = (
        abs(float(value)) for value in bounds.extent_cm
    )
    thin_horizontal = min(extent_x, extent_y)
    return (
        0.0
        < thin_horizontal
        <= _INDOOR_THIN_TARGET_MAX_HALF_THICKNESS_CM
        and max(extent_x, extent_y)
        >= _INDOOR_THIN_TARGET_MIN_ASPECT_RATIO * thin_horizontal
        and extent_z
        >= _INDOOR_THIN_TARGET_MIN_ASPECT_RATIO * thin_horizontal
    )


def _thin_target_support_frame(
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
    target_bounds: ActorBounds,
) -> _ThinTargetSupportFrame | None:
    """Find the unchanged wall behind a vertical thin GT repair target."""

    if not _is_vertical_thin_target(target_bounds):
        return None
    target_center = target_bounds.center_cm
    target_extent = tuple(abs(float(value)) for value in target_bounds.extent_cm)
    thin_axis = 0 if target_extent[0] <= target_extent[1] else 1
    wide_axis = 1 - thin_axis
    input_index = index_actors(input_scene)
    gt_index = index_actors(gt_scene)
    candidates = []
    for actor_id in input_index.keys() & gt_index.keys():
        input_actor = input_index[actor_id]
        gt_actor = gt_index[actor_id]
        if _unchanged_signature(input_actor) != _unchanged_signature(gt_actor):
            continue
        if "wall" not in _raw_actor_text(gt_actor):
            continue
        raw_bounds = _raw_actor_bounds(gt_actor)
        if raw_bounds is None:
            continue
        center, extent = raw_bounds
        extent = tuple(abs(float(value)) for value in extent)
        if extent[thin_axis] > 25.0:
            continue
        if (
            extent[wide_axis] + 5.0 < 0.8 * target_extent[wide_axis]
            or extent[2] + 5.0 < 0.8 * target_extent[2]
        ):
            continue
        if any(
            abs(float(center[axis]) - float(target_center[axis]))
            > extent[axis] + target_extent[axis] + 5.0
            for axis in range(3)
        ):
            continue
        support_side = float(center[thin_axis]) - float(
            target_center[thin_axis]
        )
        if abs(support_side) <= 0.5:
            continue
        candidates.append(
            (
                abs(support_side),
                str(actor_id).casefold(),
                _ThinTargetSupportFrame(
                    str(actor_id),
                    thin_axis,
                    -1 if support_side > 0.0 else 1,
                    tuple(float(value) for value in center),
                ),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda value: (value[0], value[1]))
    return candidates[0][2]


def _shared_authored_camera_poses(
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
    target_bounds: ActorBounds,
    support: _ThinTargetSupportFrame,
    *,
    fov_deg: float,
) -> tuple[tuple[str, CameraPose], ...]:
    """Return framed free-space camera locations shared by Input and GT."""

    input_index = index_actors(input_scene)
    gt_index = index_actors(gt_scene)
    center = target_bounds.center_cm
    values = []
    seen: set[tuple[float, ...]] = set()
    for actor_id in input_index.keys() & gt_index.keys():
        input_actor = input_index[actor_id]
        gt_actor = gt_index[actor_id]
        if _unchanged_signature(input_actor) != _unchanged_signature(gt_actor):
            continue
        actor_class = str(gt_actor.get("class") or "").casefold()
        if "camera" not in actor_class:
            continue
        transform = gt_actor.get("transform")
        location = (
            transform.get("location_cm")
            if isinstance(transform, Mapping)
            else None
        )
        if (
            not isinstance(location, Sequence)
            or isinstance(location, (str, bytes))
            or len(location) != 3
        ):
            continue
        try:
            location_cm = tuple(float(value) for value in location)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in location_cm):
            continue
        if support.exposed_sign * (
            location_cm[support.thin_axis] - center[support.thin_axis]
        ) < _MIN_INDOOR_ANCHOR_EXTENT_CM * 10.0:
            continue
        pose = _look_at_room_pose(location_cm, center)
        if not assess_bounds_camera_pose(
            center,
            target_bounds.extent_cm,
            pose,
            fov_degrees=fov_deg,
        ).healthy:
            continue
        key = camera_pose_key(pose)
        if key in seen:
            continue
        seen.add(key)
        values.append(
            (
                math.dist(location_cm, center),
                str(actor_id).casefold(),
                str(actor_id),
                pose,
            )
        )
    values.sort(key=lambda value: (value[0], value[1]))
    return tuple((value[2], value[3]) for value in values)


def _shared_authored_camera_neighborhood(
    authored: Sequence[tuple[str, CameraPose]],
    target_bounds: ActorBounds,
    support: _ThinTargetSupportFrame,
    *,
    fov_deg: float,
) -> tuple[CameraPose, ...]:
    """Expand shared camera anchors only along the support-wall plane."""

    center = target_bounds.center_cm
    tangent_axis = 1 - support.thin_axis
    values = []
    seen: set[tuple[float, ...]] = set()
    for _actor_id, anchor in authored:
        for tangent_offset, height_offset in (
            _INDOOR_AUTHORED_CAMERA_NEIGHBORHOOD_OFFSETS_CM
        ):
            location = list(anchor.location_cm)
            location[tangent_axis] += tangent_offset
            location[2] += height_offset
            location_cm = tuple(location)
            if support.exposed_sign * (
                location_cm[support.thin_axis]
                - center[support.thin_axis]
            ) < _MIN_INDOOR_ANCHOR_EXTENT_CM * 10.0:
                continue
            pose = _look_at_room_pose(location_cm, center)
            if not assess_bounds_camera_pose(
                center,
                target_bounds.extent_cm,
                pose,
                fov_degrees=fov_deg,
            ).healthy:
                continue
            key = camera_pose_key(pose)
            if key in seen:
                continue
            seen.add(key)
            values.append(pose)
    return tuple(values)


def _thin_surface_candidate_groups(
    preferred_views: Sequence[render.Viewpoint],
    target_bounds: ActorBounds,
    scene_bounds: SceneBounds,
    support: _ThinTargetSupportFrame,
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
) -> tuple[
    tuple[tuple[CameraPose, ...], ...],
    tuple[tuple[str, CameraPose], ...],
    tuple[CameraPose, ...],
]:
    """Keep only framed cameras on the target side opposite its support wall."""

    center = target_bounds.center_cm
    authored = _shared_authored_camera_poses(
        input_scene,
        gt_scene,
        target_bounds,
        support,
        fov_deg=float(preferred_views[0].fov_deg),
    )
    authored_neighborhood = _shared_authored_camera_neighborhood(
        authored,
        target_bounds,
        support,
        fov_deg=float(preferred_views[0].fov_deg),
    )
    groups = []
    for view in preferred_views:
        candidates = visibility_pose_candidates(
            _pose(view),
            (target_bounds,),
            scene_bounds,
            fov_degrees=float(view.fov_deg),
        )
        generic_exposed = tuple(
            candidate
            for candidate in candidates
            if support.exposed_sign
            * (
                candidate.location_cm[support.thin_axis]
                - center[support.thin_axis]
            )
            >= _MIN_INDOOR_ANCHOR_EXTENT_CM * 10.0
        )
        exposed = tuple(
            dict.fromkeys((*authored_neighborhood, *generic_exposed))
        )
        if len(exposed) < len(preferred_views):
            raise ValueError(
                "vertical thin target has fewer than "
                f"{len(preferred_views)} framed exposed-side camera candidates"
            )
        groups.append(exposed)
    return tuple(groups), authored, authored_neighborhood


def _hanging_target_candidate_groups(
    preferred_views: Sequence[render.Viewpoint],
    target_bounds: ActorBounds,
    scene_bounds: SceneBounds,
    room: _IndoorRoomFrame,
) -> tuple[tuple[CameraPose, ...], ...]:
    """Add bounded room-side views below a ceiling-adjacent local target."""

    center = target_bounds.center_cm
    groups = []
    for view in preferred_views:
        preferred = _pose(view)
        candidates = list(
            visibility_pose_candidates(
                preferred,
                (target_bounds,),
                scene_bounds,
                fov_degrees=float(view.fov_deg),
            )
        )
        seen = {camera_pose_key(value) for value in candidates}
        dx = preferred.x - center[0]
        dy = preferred.y - center[1]
        horizontal = math.hypot(dx, dy)
        base_azimuth = (
            math.degrees(math.atan2(dy, dx))
            if horizontal > 1e-6
            else preferred.yaw + 180.0
        )
        base_radius = max(
            math.dist(preferred.location_cm, center),
            math.sqrt(sum(value * value for value in target_bounds.extent_cm))
            * 1.05,
            100.0,
        )
        added = 0
        for offset in _INDOOR_HANGING_AZIMUTH_OFFSETS_DEG:
            azimuth = math.radians(base_azimuth + offset)
            for scale, elevation_degrees in _INDOOR_HANGING_VIEW_VARIANTS:
                radius = base_radius * scale
                elevation = math.radians(elevation_degrees)
                horizontal_radius = radius * math.cos(elevation)
                location = (
                    center[0] + math.cos(azimuth) * horizontal_radius,
                    center[1] + math.sin(azimuth) * horizontal_radius,
                    center[2] + math.sin(elevation) * radius,
                )
                if not (
                    room.min_x + _INDOOR_HANGING_CAMERA_MARGIN_CM
                    <= location[0]
                    <= room.max_x - _INDOOR_HANGING_CAMERA_MARGIN_CM
                    and room.min_y + _INDOOR_HANGING_CAMERA_MARGIN_CM
                    <= location[1]
                    <= room.max_y - _INDOOR_HANGING_CAMERA_MARGIN_CM
                    and room.floor_z
                    + _INDOOR_HANGING_CAMERA_MIN_FLOOR_CLEARANCE_CM
                    <= location[2]
                    <= room.ceiling_z
                    - _INDOOR_HANGING_CAMERA_MIN_CEILING_CLEARANCE_CM
                ):
                    continue
                pose = _look_at_room_pose(location, center)
                if not assess_bounds_camera_pose(
                    center,
                    target_bounds.extent_cm,
                    pose,
                    fov_degrees=float(view.fov_deg),
                ).healthy:
                    continue
                key = camera_pose_key(pose)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(pose)
                added += 1
                if added >= _INDOOR_HANGING_MAX_ADDITIONAL_CANDIDATES:
                    break
            if added >= _INDOOR_HANGING_MAX_ADDITIONAL_CANDIDATES:
                break
        groups.append(tuple(candidates))
    return tuple(groups)


def _resolve(
    bridge: Any,
    preferred: scene_graph_capture.CameraPlan,
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    actor_ids_by_view: Sequence[Sequence[str]],
    *,
    planner_id: str,
    pairing_policy: str,
    planning_source: str,
    scene_environment: str,
    timeout_s: float,
    candidate_poses_by_view: Sequence[Sequence[CameraPose]] | None = None,
    overview_mode: bool = False,
    require_enclosure: bool = False,
    require_unique_camera_poses: bool = False,
    enclosure_room_floor_z_cm: float | None = None,
    enclosure_room_ceiling_z_cm: float | None = None,
    required_actor_clear_fraction: float | None = None,
    max_overview_candidates_per_view: int | None = None,
    allow_enclosed_overview_without_visible_actor: bool = False,
    minimum_visible_overview_views: int | None = None,
    allow_thin_target_surface_proxy: bool = False,
    visibility_policy: str = "ue-collision-navigation-target-los",
) -> scene_graph_capture.CameraPlan:
    resolutions = resolve_visibility_camera_poses(
        bridge,
        inventory,
        scene_bounds,
        tuple(_pose(value) for value in preferred.views),
        actor_ids_by_view,
        timeout_s=timeout_s,
        fov_degrees=render.DEFAULT_FOV_DEG,
        candidate_poses_by_pose=candidate_poses_by_view,
        overview_mode=overview_mode,
        require_enclosure=require_enclosure,
        require_unique_camera_poses=require_unique_camera_poses,
        enclosure_room_floor_z_cm=enclosure_room_floor_z_cm,
        enclosure_room_ceiling_z_cm=enclosure_room_ceiling_z_cm,
        required_actor_clear_fraction=required_actor_clear_fraction,
        max_overview_candidates_per_pose=max_overview_candidates_per_view,
        allow_enclosed_overview_without_visible_actor=(
            allow_enclosed_overview_without_visible_actor
        ),
        allow_thin_target_surface_proxy=allow_thin_target_surface_proxy,
    )
    failures = [
        {"view": preferred.views[index].name, **value.to_dict()}
        for index, value in enumerate(resolutions)
        if value.pose is None
    ]
    if failures:
        raise _CameraPortfolioUnavailable(
            f"no safe visible paired camera for: {failures}"
        )
    if minimum_visible_overview_views is not None:
        visible_view_count = sum(
            value.pose is not None and value.visible_actor_count > 0
            for value in resolutions
        )
        if visible_view_count < minimum_visible_overview_views:
            raise _CameraPortfolioUnavailable(
                "indoor overview fallback resolved only "
                f"{visible_view_count}/{len(resolutions)} shared-anchor-visible "
                "views; requires at least "
                f"{minimum_visible_overview_views}"
            )
    resolved_views = tuple(
        _view(source.name, resolution.pose, source.fov_deg)
        for source, resolution in zip(preferred.views, resolutions, strict=True)
        if resolution.pose is not None
    )
    return scene_graph_capture.freeze_visibility_camera_plan(
        preferred,
        resolved_views,
        planner_id=planner_id,
        pairing_policy=pairing_policy,
        planning_source=planning_source,
        scene_environment=scene_environment,
        visibility_audit=tuple(value.to_dict() for value in resolutions),
        visibility_policy=visibility_policy,
    )


def _resolve_case_camera_views(
    bridge: Any,
    portfolio: CaseCameraPortfolio,
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    target_bounds: ActorBounds,
    actor_ids: tuple[str, ...],
    *,
    count: int,
    room: _IndoorRoomFrame | None,
    timeout_s: float,
    input_gt_observability_by_view: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[
    tuple[render.Viewpoint, ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    """Keep current-task Input/GT-observable, runtime-safe authored poses."""

    if not portfolio.views:
        raise _CameraPortfolioUnavailable(
            "case camera portfolio must contain at least one trial view"
        )
    pose_keys = [camera_pose_key(_pose(value)) for value in portfolio.views]
    if len(set(pose_keys)) != len(pose_keys):
        raise _CameraPortfolioUnavailable("case camera poses must be unique")
    runtime_observability = input_gt_observability_by_view or {}

    accepted = []
    audits = []
    rejections = []
    for index, (view, authored_name) in enumerate(
        zip(portfolio.views, portfolio.authored_view_names, strict=True)
    ):
        if len(accepted) == count:
            break
        runtime_neighbor_source = (
            portfolio.runtime_neighbor_sources[index]
            if portfolio.runtime_neighbor_sources
            else None
        )
        pose = _pose(view)
        framing = assess_bounds_camera_pose(
            target_bounds.center_cm,
            target_bounds.extent_cm,
            pose,
            fov_degrees=float(view.fov_deg),
        )
        framing_audit = {
            "target_in_front": framing.target_in_front,
            "longest_viewport_fraction": framing.longest_viewport_fraction,
            "center_offset": framing.center_offset,
            "edge_overflow": framing.edge_overflow,
            "healthy": framing.healthy,
        }
        resolution = resolve_visibility_camera_poses(
            bridge,
            inventory,
            scene_bounds,
            (pose,),
            (actor_ids,),
            timeout_s=timeout_s,
            fov_degrees=float(view.fov_deg),
            candidate_poses_by_pose=((pose,),),
            require_enclosure=True,
            require_unique_camera_poses=True,
            enclosure_room_floor_z_cm=(
                room.floor_z if room is not None else None
            ),
            enclosure_room_ceiling_z_cm=(
                room.ceiling_z if room is not None else None
            ),
            required_actor_clear_fraction=(
                _INDOOR_CASE_CAMERA_REQUIRED_ACTOR_CLEAR_FRACTION
            ),
        )[0]
        resolution_audit = dict(resolution.to_dict())
        pixel_observability = runtime_observability.get(authored_name)
        if not isinstance(pixel_observability, Mapping):
            rejections.append(
                {
                    "view": f"view_{index + 1}",
                    "authored_view": authored_name,
                    "case_camera_origin": (
                        "runtime_neighbor"
                        if runtime_neighbor_source is not None
                        else "authored_manifest"
                    ),
                    "runtime_neighbor_source": runtime_neighbor_source,
                    "selection": "case_camera_rejected",
                    "rejection_stage": "runtime_input_gt_observability",
                    "reason": "current_task_input_gt_evidence_missing",
                    "fov_deg": float(view.fov_deg),
                    "requested_pose": pose.to_dict(),
                    "geometry_advisory": {
                        "framing": framing_audit,
                        "live_visibility": resolution_audit,
                    },
                }
            )
            continue
        pixel_audit = dict(pixel_observability)
        if pixel_audit.get("accepted") is not True:
            rejections.append(
                {
                    "view": f"view_{index + 1}",
                    "authored_view": authored_name,
                    "case_camera_origin": (
                        "runtime_neighbor"
                        if runtime_neighbor_source is not None
                        else "authored_manifest"
                    ),
                    "runtime_neighbor_source": runtime_neighbor_source,
                    "selection": "case_camera_rejected",
                    "rejection_stage": "runtime_input_gt_observability",
                    "reason": "current_task_input_gt_view_failed",
                    "fov_deg": float(view.fov_deg),
                    "requested_pose": pose.to_dict(),
                    "pixel_observability": pixel_audit,
                    "geometry_advisory": {
                        "framing": framing_audit,
                        "live_visibility": resolution_audit,
                    },
                }
            )
            continue
        camera_safe = (
            resolution_audit.get("camera_initial_overlap") is False
            and resolution_audit.get("camera_enclosure_ok") is True
        )
        if not camera_safe:
            rejections.append(
                {
                    "view": f"view_{index + 1}",
                    "authored_view": authored_name,
                    "case_camera_origin": (
                        "runtime_neighbor"
                        if runtime_neighbor_source is not None
                        else "authored_manifest"
                    ),
                    "runtime_neighbor_source": runtime_neighbor_source,
                    "selection": "case_camera_rejected",
                    "rejection_stage": "camera_safety",
                    "reason": "overlap_or_enclosure_not_proven",
                    "fov_deg": float(view.fov_deg),
                    "requested_pose": pose.to_dict(),
                    "pixel_observability": pixel_audit,
                    "geometry_advisory": {
                        "framing": framing_audit,
                        "live_visibility": resolution_audit,
                    },
                }
            )
            continue
        resolved_pose = getattr(resolution, "pose", None)
        exact_pose_resolver_match = (
            resolved_pose is not None
            and camera_pose_key(resolved_pose) == camera_pose_key(pose)
        )
        accepted.append(_view(view.name, pose, view.fov_deg))
        audits.append(
            {
                **resolution_audit,
                "view": f"view_{index + 1}",
                "authored_view": authored_name,
                "case_camera_origin": (
                    "runtime_neighbor"
                    if runtime_neighbor_source is not None
                    else "authored_manifest"
                ),
                "runtime_neighbor_source": runtime_neighbor_source,
                "selection": "case_camera_runtime_input_gt_exact_pose",
                "reason": "current_task_input_gt_observable_and_camera_safe",
                "resolved_pose": pose.to_dict(),
                "fov_deg": float(view.fov_deg),
                "pixel_observability": pixel_audit,
                "geometry_advisory": {
                    "framing": framing_audit,
                    "live_visibility": resolution_audit,
                    "exact_pose_resolver_match": exact_pose_resolver_match,
                },
            }
        )
    if not accepted:
        raise _CameraPortfolioUnavailable(
            f"no case camera passed its applicable acceptance gates: {rejections}"
        )
    return tuple(accepted), tuple(audits), tuple(rejections)


def _freeze_case_camera_plan(
    portfolio: CaseCameraPortfolio,
    case_views: Sequence[render.Viewpoint],
    case_audit: Sequence[Mapping[str, Any]],
    case_rejection_audit: Sequence[Mapping[str, Any]],
    fill_plan: scene_graph_capture.CameraPlan | None,
    target_bounds: ActorBounds,
    *,
    count: int,
    requested_count: int | None = None,
) -> scene_graph_capture.CameraPlan:
    case_camera_policy = (
        "runtime-input-gt-authored-neighborhood-full-hit-else-generic-fallback"
    )
    fill_views = fill_plan.views if fill_plan is not None else ()
    fill_audit = (
        tuple(fill_plan.audit.get("visibility_resolution") or ())
        if fill_plan is not None
        else ()
    )
    if len(fill_audit) != len(fill_views):
        raise _CameraPortfolioUnavailable(
            "generic fill plan lacks one visibility audit per fill view"
        )
    combined = tuple((*case_views, *fill_views))
    if len(combined) != count:
        raise _CameraPortfolioUnavailable(
            f"case camera portfolio filled {len(combined)}/{count} views"
        )
    if len({camera_pose_key(_pose(value)) for value in combined}) != count:
        raise _CameraPortfolioUnavailable(
            "case cameras overlap a verifier-planned fill camera"
        )
    views = tuple(
        _view(f"view_{index + 1}", _pose(value), value.fov_deg)
        for index, value in enumerate(combined)
    )
    visibility_audit = [
        {**dict(value), "view": f"view_{index}"}
        for index, value in enumerate(case_audit, start=1)
    ]
    if fill_plan is not None:
        for index, value in enumerate(
            fill_audit,
            start=len(case_views) + 1,
        ):
            visibility_audit.append(
                {
                    **dict(value),
                    "view": f"view_{index}",
                    "selection": "verifier_planned_fill",
                }
            )
    preferred = scene_graph_capture.CameraPlan(
        views,
        target_bounds.center_cm,
        {
            "planner_id": scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN,
            "pairing_policy": (
                "gt-repair-target-visibility-frozen-absolute-cameras"
            ),
            "planning_source": (
                "case_camera_manifest_then_gt_minus_input_target_fill"
            ),
            "case_camera_policy": case_camera_policy,
            "case_camera_acceptance_basis": (
                "current_task_input_gt_same_pose_rgb_observability"
            ),
            "case_camera_candidate_used_for_acceptance": False,
            "case_camera_release_qa_role": "optional_provenance_only",
            "case_camera_geometry_gate_mode": (
                "overlap_enclosure_hard_framing_los_advisory"
            ),
            "case_camera_attempt": "accepted_exact_pose",
            "case_camera_manifest_schema": portfolio.schema_version,
            "case_camera_manifest_sha256": portfolio.manifest_sha256,
            "case_camera_paired_render_qa_sha256": (
                portfolio.paired_render_qa_sha256
            ),
            "case_camera_authored_view_names": list(
                portfolio.authored_view_names
            ),
            "case_camera_view_count": len(case_views),
            "case_camera_rejected_view_count": len(case_rejection_audit),
            "case_camera_rejections": [
                dict(value) for value in case_rejection_audit
            ],
            "fill_view_count": len(fill_views),
            "fill_planner_id": (
                fill_plan.audit.get("planner_id")
                if fill_plan is not None
                else None
            ),
            "view_count": count,
            "requested_view_count": (
                requested_count if requested_count is not None else count
            ),
            "case_camera_direct_hit_required_view_count": count,
            "case_camera_direct_hit_satisfied": len(case_views) == count,
        },
    )
    return scene_graph_capture.freeze_visibility_camera_plan(
        preferred,
        views,
        planner_id=scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN,
        pairing_policy="gt-repair-target-visibility-frozen-absolute-cameras",
        planning_source="case_camera_manifest_then_gt_minus_input_target_fill",
        scene_environment="indoor",
        visibility_audit=visibility_audit,
        visibility_policy=(
            "case_camera_runtime_input_gt_observable_overlap_enclosure_hard_"
            "framing-los-advisory-two-view-direct-else-generic"
        ),
    )


def _with_case_camera_fallback_audit(
    plan: scene_graph_capture.CameraPlan,
    portfolio: CaseCameraPortfolio,
    reason: str,
) -> scene_graph_capture.CameraPlan:
    return scene_graph_capture.CameraPlan(
        plan.views,
        plan.anchor_cm,
        {
            **dict(plan.audit),
            "case_camera_policy": (
                "runtime_input_gt_authored_neighborhood_full_hit_"
                "else-generic-fallback"
            ),
            "case_camera_acceptance_basis": (
                "current_task_input_gt_same_pose_rgb_observability"
            ),
            "case_camera_candidate_used_for_acceptance": False,
            "case_camera_release_qa_role": "optional_provenance_only",
            "case_camera_geometry_gate_mode": (
                "overlap_enclosure_hard_framing_los_advisory"
            ),
            "case_camera_attempt": "full-generic-strategy-fallback",
            "case_camera_fallback_reason": reason,
            "case_camera_manifest_schema": portfolio.schema_version,
            "case_camera_manifest_sha256": portfolio.manifest_sha256,
            "case_camera_paired_render_qa_sha256": (
                portfolio.paired_render_qa_sha256
            ),
            "case_camera_authored_view_names": list(
                portfolio.authored_view_names
            ),
            "case_camera_view_count": 0,
            "fill_view_count": len(plan.views),
        },
    )


def _with_camera_fallback_audit(
    plan: scene_graph_capture.CameraPlan,
    *,
    stage: str,
    reason: str,
) -> scene_graph_capture.CameraPlan:
    """Retain the frozen preferred geometry while naming a bounded fallback."""

    return scene_graph_capture.CameraPlan(
        plan.views,
        plan.anchor_cm,
        {
            **dict(plan.audit),
            "indoor_camera_fallback_stage": stage,
            "indoor_camera_fallback_reason": reason,
        },
    )


def _shared_indoor_fallback_anchor_ids(
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
    inventory: ActorInventorySnapshot,
    *,
    count: int,
) -> tuple[str, ...]:
    """Return shared props, then shared renderable structure as a last resort.

    The ordinary selector intentionally rejects walls and thin modular pieces.
    That is right for the primary room-corner plan but leaves tiled interiors
    with no fallback anchor at all.  The second tier remains Candidate-blind:
    it only admits Actors whose Input and GT signatures are identical, and UE
    must still prove camera clearance, enclosure and line of sight.
    """

    input_index = index_actors(input_scene)
    gt_index = index_actors(gt_scene)
    focus = _edited_focus_cm(input_scene, gt_scene)
    unchanged = {
        actor_id
        for actor_id in input_index.keys() & gt_index.keys()
        if _unchanged_signature(input_index[actor_id])
        == _unchanged_signature(gt_index[actor_id])
    }
    candidates = []
    for descriptor in inventory.actors:
        bounds = descriptor.bounds
        if (
            descriptor.live_actor_id not in unchanged
            or bounds is None
            or descriptor.renderable is not True
            or not descriptor.asset_path
        ):
            continue
        extents = tuple(abs(float(value)) for value in bounds.extent_cm)
        if max(extents) <= 0.0 or max(extents) > _MAX_INDOOR_ANCHOR_EXTENT_CM:
            continue
        text = " ".join(
            str(value or "").casefold()
            for value in (
                descriptor.actor_class,
                descriptor.asset_path,
                descriptor.actor_label,
                descriptor.unreal_name,
            )
        )
        if any(
            token in text
            for token in (
                "camera",
                "fog",
                "light",
                "postprocess",
                "reflectioncapture",
                "sky",
                "volume",
                "worldsettings",
            )
        ):
            continue
        prominence = max(extents[0], extents[1]) * max(extents[2], 1.0)
        distance = math.dist(bounds.center_cm, focus)
        candidates.append(
            (
                0 if _is_indoor_anchor(descriptor) else 1,
                distance,
                -prominence,
                descriptor.live_actor_id,
                str(descriptor.asset_path).casefold(),
            )
        )
    candidates.sort(
        key=lambda value: (value[0], value[1], value[2], value[3].casefold())
    )
    selected = []
    selected_ids: set[str] = set()
    selected_assets: set[str] = set()
    for value in candidates:
        asset = value[4]
        if asset in selected_assets:
            continue
        selected.append(value[3])
        selected_ids.add(value[3].casefold())
        selected_assets.add(asset)
        if len(selected) >= count:
            return tuple(selected)
    for value in candidates:
        if value[3].casefold() in selected_ids:
            continue
        selected.append(value[3])
        if len(selected) >= count:
            break
    return tuple(selected)


def _resolve_indoor_anchor_overview_fallback(
    bridge: Any,
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    *,
    count: int,
    timeout_s: float,
    planner_id: str,
    primary_failure: str,
) -> scene_graph_capture.CameraPlan:
    """Search around shared interior anchors when room corners are unavailable."""

    preferred = scene_graph_capture.plan_gt_frozen_dense_core_aerial(
        tuple(gt_scene.get("actors") or ()),
        count=count,
    )
    anchors = _shared_indoor_fallback_anchor_ids(
        input_scene,
        gt_scene,
        inventory,
        count=max(count, 8),
    )
    if not anchors:
        raise ValueError(
            f"{primary_failure}; indoor anchor fallback has no shared live Actor"
        )
    by_id = {value.live_actor_id: value for value in inventory.actors}
    candidates_by_view_and_anchor = []
    candidate_counts_by_anchor = [0 for _actor_id in anchors]
    for view in preferred.views:
        candidates_by_anchor = []
        for anchor_index, actor_id in enumerate(anchors):
            descriptor = by_id.get(actor_id)
            if descriptor is None or descriptor.bounds is None:
                candidates_by_anchor.append(())
                continue
            candidates = visibility_pose_candidates(
                _pose(view),
                (descriptor.bounds,),
                scene_bounds,
                fov_degrees=float(view.fov_deg),
            )
            candidates_by_anchor.append(candidates)
            candidate_counts_by_anchor[anchor_index] = max(
                candidate_counts_by_anchor[anchor_index],
                len(candidates),
            )
        candidates_by_view_and_anchor.append(tuple(candidates_by_anchor))

    # Share one interleaved portfolio across all four output views. A modular
    # room may have no valid camera in one nominal compass direction even when
    # it has many safe interior viewpoints elsewhere. The live resolver keeps
    # a global used-pose set, so the shared portfolio still yields four unique
    # cameras. Candidate rank is outermost to preserve breadth within the hard
    # 24-pose budget instead of exhausting it on one view or one anchor.
    combined = []
    seen: set[tuple[float, float, float, float, float]] = set()
    maximum = max(
        (
            len(value)
            for groups in candidates_by_view_and_anchor
            for value in groups
        ),
        default=0,
    )
    for candidate_index in range(maximum):
        for anchor_index in range(len(anchors)):
            for view_groups in candidates_by_view_and_anchor:
                anchor_candidates = view_groups[anchor_index]
                if candidate_index >= len(anchor_candidates):
                    continue
                candidate = anchor_candidates[candidate_index]
                key = camera_pose_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                combined.append(candidate)
                if len(combined) >= _INDOOR_FALLBACK_MAX_CANDIDATES_PER_VIEW:
                    break
            if len(combined) >= _INDOOR_FALLBACK_MAX_CANDIDATES_PER_VIEW:
                break
        if len(combined) >= _INDOOR_FALLBACK_MAX_CANDIDATES_PER_VIEW:
            break
    if not combined:
        raise ValueError(
            f"{primary_failure}; indoor anchor fallback generated no "
            "framed camera candidates"
        )
    candidate_groups = tuple(
        tuple(combined) for _view_value in preferred.views
    )

    actor_ids_by_view = tuple(anchors for _view_value in preferred.views)
    preferred = _with_camera_fallback_audit(
        preferred,
        stage=scene_graph_capture.INDOOR_OVERVIEW_CAMERA_FALLBACK_STAGE,
        reason=primary_failure,
    )
    preferred = scene_graph_capture.CameraPlan(
        preferred.views,
        preferred.anchor_cm,
        {
            **dict(preferred.audit),
            "planning_source": "input_gt_shared_actor_geometry",
            "indoor_camera_position_policy": (
                "shared-anchor-pool-orbit-enclosed-visibility"
            ),
            "indoor_visibility_anchor_actor_ids": list(anchors),
            "anchor_candidate_count_by_actor": dict(
                zip(anchors, candidate_counts_by_anchor, strict=True)
            ),
            "fallback_candidate_count_per_view": [
                len(value) for value in candidate_groups
            ],
            "fallback_max_candidates_per_view": (
                _INDOOR_FALLBACK_MAX_CANDIDATES_PER_VIEW
            ),
        },
    )
    try:
        return _resolve(
            bridge,
            preferred,
            inventory,
            scene_bounds,
            actor_ids_by_view,
            planner_id=planner_id,
            pairing_policy="environment-routed-frozen-absolute-cameras",
            planning_source="input_gt_shared_actor_geometry",
            scene_environment="indoor",
            timeout_s=timeout_s,
            candidate_poses_by_view=candidate_groups,
            overview_mode=True,
            require_enclosure=True,
            require_unique_camera_poses=True,
            required_actor_clear_fraction=(
                _INDOOR_FALLBACK_REQUIRED_ACTOR_CLEAR_FRACTION
            ),
            max_overview_candidates_per_view=(
                _INDOOR_FALLBACK_MAX_CANDIDATES_PER_VIEW
            ),
            allow_enclosed_overview_without_visible_actor=True,
            minimum_visible_overview_views=max(1, count - 1),
            visibility_policy=(
                "indoor-shared-anchor-pool-overlap-enclosure-los-rgb-unique"
            ),
        )
    except _CameraPortfolioUnavailable as fallback_error:
        raise ValueError(
            f"{primary_failure}; indoor anchor fallback failed: {fallback_error}"
        ) from fallback_error


def resolve_repair_target_plan(
    bridge: Any,
    visibility_scene: Mapping[str, Any],
    target_actors: Sequence[Mapping[str, Any]],
    *,
    input_scene: Mapping[str, Any] | None = None,
    gt_scene: Mapping[str, Any] | None = None,
    count: int,
    scene_environment: str,
    half_extent_m: float | None,
    timeout_s: float,
    case_cameras: CaseCameraPortfolio | None = None,
    case_camera_input_gt_observability: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
) -> scene_graph_capture.CameraPlan:
    """Try exact authored Indoor views first; otherwise use the generic planner."""

    preferred = scene_graph_capture.plan_gt_repair_target_aerial(
        target_actors,
        count=count,
    )
    if scene_environment == "outdoor":
        if case_cameras is not None:
            raise ValueError("case camera manifests are only valid for Indoor repair")
        return scene_graph_capture.freeze_visibility_camera_plan(
            preferred,
            preferred.views,
            planner_id=scene_graph_capture.GT_REPAIR_TARGET_OUTDOOR_CAMERA_PLAN,
            pairing_policy=(
                "gt-repair-target-visibility-frozen-absolute-cameras"
            ),
            planning_source="gt_minus_input_target",
            scene_environment=scene_environment,
            visibility_policy="outdoor-target-local-preserved",
            visibility_audit=tuple(
                {
                    "view": value.name,
                    "reason": "outdoor-target-local-preserved",
                }
                for value in preferred.views
            ),
        )
    if scene_environment != "indoor":
        raise ValueError(f"unsupported scene environment {scene_environment!r}")
    if input_scene is None or gt_scene is None:
        raise ValueError(
            "Indoor local camera planning requires Input and GT scene snapshots"
        )
    planner_id = (
        scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN
        if case_cameras is not None
        else scene_graph_capture.GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN
    )
    inventory, bounds = adapt_scene_snapshot(
        visibility_scene,
        half_extent_m=half_extent_m,
    )
    room: _IndoorRoomFrame | None = None
    room_failure: str | None = None
    try:
        room = _infer_indoor_room_frame(input_scene, gt_scene)
    except ValueError as error:
        # Tiled and modular interiors may have no single floor/wall Actor that
        # describes the room. UE enclosure traces can still prove a safe
        # interior camera without inventing a room AABB.
        room_failure = str(error)
    target_bounds = _target_actor_bounds(target_actors)
    hanging_target = (
        _is_hanging_target(target_bounds, room) if room is not None else False
    )
    candidate_groups = (
        _hanging_target_candidate_groups(
            preferred.views,
            target_bounds,
            bounds,
            room,
        )
        if hanging_target and room is not None
        else None
    )
    preferred = scene_graph_capture.CameraPlan(
        preferred.views,
        preferred.anchor_cm,
        {
            **preferred.audit,
            "local_camera_position_policy": (
                "standard-then-ceiling-room-side"
            ),
            "indoor_room": room.to_dict() if room is not None else None,
            "indoor_room_inference_error": room_failure,
            "indoor_room_requirement": (
                "inferred_room_bounds"
                if room is not None
                else "ue_enclosure_trace_fallback"
            ),
            "hanging_target_detected": hanging_target,
            "hanging_target_detection_policy": (
                "upper-room-and-ceiling-gap-geometry"
            ),
            "hanging_candidate_count_per_view": (
                [
                    max(
                        0,
                        len(group)
                        - len(
                            visibility_pose_candidates(
                                _pose(view),
                                (target_bounds,),
                                bounds,
                                fov_degrees=float(view.fov_deg),
                            )
                        ),
                    )
                    for view, group in zip(
                        preferred.views,
                        candidate_groups,
                        strict=True,
                    )
                ]
                if candidate_groups is not None
                else [0 for _view_value in preferred.views]
            ),
        },
    )
    try:
        actor_ids = tuple(actor_identity(value) for value in target_actors)
    except ActorIdentityError as error:
        raise ValueError(f"repair target lacks live visibility identity: {error}") from error
    actor_ids_by_view = tuple(actor_ids for _value in preferred.views)
    case_camera_failure: str | None = None
    if case_cameras is not None:
        try:
            case_views, case_audit, case_rejection_audit = (
                _resolve_case_camera_views(
                    bridge,
                    case_cameras,
                    inventory,
                    bounds,
                    target_bounds,
                    actor_ids,
                    count=count,
                    room=room,
                    timeout_s=timeout_s,
                    input_gt_observability_by_view=(
                        case_camera_input_gt_observability
                    ),
                )
            )
        except _CameraPortfolioUnavailable as error:
            case_camera_failure = f"no authored view accepted: {error}"
        else:
            if len(case_views) == count:
                return _freeze_case_camera_plan(
                    case_cameras,
                    case_views,
                    case_audit,
                    case_rejection_audit,
                    None,
                    target_bounds,
                    count=len(case_views),
                    requested_count=count,
                )
            fill_count = count - len(case_views)
            try:
                fill_plan = (
                    resolve_repair_target_plan(
                        bridge,
                        visibility_scene,
                        target_actors,
                        input_scene=input_scene,
                        gt_scene=gt_scene,
                        count=max(2, fill_count),
                        scene_environment=scene_environment,
                        half_extent_m=half_extent_m,
                        timeout_s=timeout_s,
                    )
                    if fill_count
                    else None
                )
                return _freeze_case_camera_plan(
                    case_cameras,
                    case_views,
                    case_audit,
                    case_rejection_audit,
                    fill_plan,
                    target_bounds,
                    count=count,
                )
            except _CameraPortfolioUnavailable as error:
                case_camera_failure = (
                    f"accepted {len(case_views)} authored view(s), but generic fill "
                    f"failed: {error}"
                )

    def finish(plan: scene_graph_capture.CameraPlan) -> scene_graph_capture.CameraPlan:
        if case_cameras is None:
            return plan
        return _with_case_camera_fallback_audit(
            plan,
            case_cameras,
            case_camera_failure or "case camera fill was unavailable",
        )

    try:
        return finish(_resolve(
            bridge,
            preferred,
            inventory,
            bounds,
            actor_ids_by_view,
            planner_id=planner_id,
            pairing_policy="gt-repair-target-visibility-frozen-absolute-cameras",
            planning_source="gt_minus_input_target",
            scene_environment=scene_environment,
            timeout_s=timeout_s,
            candidate_poses_by_view=candidate_groups,
            require_enclosure=True,
            require_unique_camera_poses=True,
            enclosure_room_floor_z_cm=(
                room.floor_z if room is not None else None
            ),
            enclosure_room_ceiling_z_cm=(
                room.ceiling_z if room is not None else None
            ),
            required_actor_clear_fraction=(
                _INDOOR_HANGING_REQUIRED_ACTOR_CLEAR_FRACTION
                if hanging_target
                else None
            ),
            visibility_policy=(
                "indoor-local-overlap-enclosure-los-unique-hanging-room-side"
            ),
        ))
    except _CameraPortfolioUnavailable as primary_error:
        fallback = _with_camera_fallback_audit(
            preferred,
            stage="target-relaxed-clear-fraction",
            reason=str(primary_error),
        )
        try:
            return finish(_resolve(
                bridge,
                fallback,
                inventory,
                bounds,
                actor_ids_by_view,
                planner_id=planner_id,
                pairing_policy=(
                    "gt-repair-target-visibility-frozen-absolute-cameras"
                ),
                planning_source="gt_minus_input_target",
                scene_environment=scene_environment,
                timeout_s=timeout_s,
                require_enclosure=True,
                require_unique_camera_poses=True,
                enclosure_room_floor_z_cm=(
                    room.floor_z if room is not None else None
                ),
                enclosure_room_ceiling_z_cm=(
                    room.ceiling_z if room is not None else None
                ),
                required_actor_clear_fraction=(
                    _INDOOR_LOCAL_FALLBACK_REQUIRED_ACTOR_CLEAR_FRACTION
                ),
                visibility_policy=(
                    "indoor-local-enclosed-relaxed-los-unique"
                ),
            ))
        except _CameraPortfolioUnavailable as fallback_error:
            # Local target evidence may be captured from a doorway or window.
            # Keep collision and target LoS mandatory but stop claiming that
            # the camera itself is enclosed when UE cannot prove it.
            last_resort = _with_camera_fallback_audit(
                fallback,
                stage="target-visibility-without-enclosure",
                reason=str(fallback_error),
            )
            try:
                return finish(_resolve(
                    bridge,
                    last_resort,
                    inventory,
                    bounds,
                    actor_ids_by_view,
                    planner_id=planner_id,
                    pairing_policy=(
                        "gt-repair-target-visibility-frozen-absolute-cameras"
                    ),
                    planning_source="gt_minus_input_target",
                    scene_environment=scene_environment,
                    timeout_s=timeout_s,
                    require_enclosure=False,
                    require_unique_camera_poses=True,
                    required_actor_clear_fraction=(
                        _INDOOR_LOCAL_FALLBACK_REQUIRED_ACTOR_CLEAR_FRACTION
                    ),
                    visibility_policy=(
                        "indoor_local_overlap_los_unique_"
                        "enclosure-unverified"
                    ),
                ))
            except _CameraPortfolioUnavailable as line_of_sight_error:
                support = _thin_target_support_frame(
                    input_scene,
                    gt_scene,
                    target_bounds,
                )
                if support is None:
                    if case_camera_failure is not None:
                        raise _CameraPortfolioUnavailable(
                            "case camera rejected before full generic fallback: "
                            f"{case_camera_failure}; full generic fallback also "
                            f"failed: {line_of_sight_error}"
                        ) from line_of_sight_error
                    raise
                (
                    exposed_candidate_groups,
                    authored_camera_poses,
                    authored_camera_neighborhood,
                ) = _thin_surface_candidate_groups(
                    preferred.views,
                    target_bounds,
                    bounds,
                    support,
                    input_scene,
                    gt_scene,
                )
                # A renderable wall picture or sign may have no visibility
                # collision. Retry only this GT-frozen vertical-thin target
                # with an enclosed camera and a five-centimetre surface-reach
                # proof. Candidate and GT still receive the exact same plan.
                surface_reach = _with_camera_fallback_audit(
                    fallback,
                    stage="target-thin-surface-reach",
                    reason=str(line_of_sight_error),
                )
                surface_reach = scene_graph_capture.CameraPlan(
                    surface_reach.views,
                    surface_reach.anchor_cm,
                    {
                        **dict(surface_reach.audit),
                        "thin_target_support": support.to_dict(),
                        "thin_surface_candidate_count_per_view": [
                            len(value)
                            for value in exposed_candidate_groups
                        ],
                        "shared_authored_camera_actor_ids": [
                            value[0] for value in authored_camera_poses
                        ],
                        "shared_authored_camera_position_count": len(
                            authored_camera_poses
                        ),
                        "shared_authored_camera_neighborhood_candidate_count": len(
                            authored_camera_neighborhood
                        ),
                        "shared_authored_camera_neighborhood_offsets_cm": [
                            list(value)
                            for value in (
                                _INDOOR_AUTHORED_CAMERA_NEIGHBORHOOD_OFFSETS_CM
                            )
                        ],
                        "thin_surface_camera_side_policy": (
                            "shared-authored-neighborhood-then-opposite-"
                            "input-gt-support-wall"
                        ),
                    },
                )
                return finish(_resolve(
                    bridge,
                    surface_reach,
                    inventory,
                    bounds,
                    actor_ids_by_view,
                    planner_id=planner_id,
                    pairing_policy=(
                        "gt-repair-target-visibility-frozen-absolute-cameras"
                    ),
                    planning_source="gt_minus_input_target",
                    scene_environment=scene_environment,
                    timeout_s=timeout_s,
                    candidate_poses_by_view=exposed_candidate_groups,
                    require_enclosure=True,
                    require_unique_camera_poses=True,
                    enclosure_room_floor_z_cm=(
                        room.floor_z if room is not None else None
                    ),
                    enclosure_room_ceiling_z_cm=(
                        room.ceiling_z if room is not None else None
                    ),
                    required_actor_clear_fraction=(
                        _INDOOR_FALLBACK_REQUIRED_ACTOR_CLEAR_FRACTION
                    ),
                    allow_thin_target_surface_proxy=True,
                    visibility_policy=(
                        "indoor-local-enclosed-thin-surface-reach-unique"
                    ),
                ))


def resolve_environment_overview_plan(
    bridge: Any,
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
    *,
    count: int,
    scene_environment: str,
    half_extent_m: float | None,
    timeout_s: float,
    planner_id: str = scene_graph_capture.GT_VISUAL_CAMERA_PLAN,
) -> scene_graph_capture.CameraPlan:
    """Build an indoor free-space portfolio or preserve Outdoor framing."""

    if scene_environment == "outdoor":
        preferred = scene_graph_capture.plan_gt_frozen_dense_core_aerial(
            tuple(gt_scene.get("actors") or ()),
            count=count,
        )
        return scene_graph_capture.freeze_visibility_camera_plan(
            preferred,
            preferred.views,
            planner_id=planner_id,
            pairing_policy="environment-routed-frozen-absolute-cameras",
            planning_source="gt",
            scene_environment=scene_environment,
            visibility_policy="outdoor-dense-core-preserved",
            visibility_audit=tuple(
                {
                    "view": value.name,
                    "reason": "outdoor-dense-core-preserved",
                    "resolved_pose": {
                        "location_cm": list(value.location),
                        "rotation_degrees": [
                            value.rotation[1],
                            value.rotation[2],
                            value.rotation[0],
                        ],
                    },
                }
                for value in preferred.views
            ),
        )
    if scene_environment != "indoor":
        raise ValueError(f"unsupported scene environment {scene_environment!r}")
    if count != 4:
        raise ValueError("indoor overview requires exactly four room-corner views")

    inventory, bounds = adapt_scene_snapshot(gt_scene, half_extent_m=half_extent_m)
    try:
        room = _infer_indoor_room_frame(input_scene, gt_scene)
    except ValueError as error:
        return _resolve_indoor_anchor_overview_fallback(
            bridge,
            input_scene,
            gt_scene,
            inventory,
            bounds,
            count=count,
            timeout_s=timeout_s,
            planner_id=planner_id,
            primary_failure=str(error),
        )
    candidate_groups = _room_corner_candidate_groups(room)
    try:
        candidate_anchor_ids = shared_indoor_anchor_ids(
            input_scene,
            gt_scene,
            inventory,
            count=8,
        )
    except ValueError:
        candidate_anchor_ids = ()
    by_id = {value.live_actor_id: value for value in inventory.actors}
    anchors = tuple(
        actor_id
        for actor_id in candidate_anchor_ids
        if (
            (descriptor := by_id.get(actor_id)) is not None
            and descriptor.bounds is not None
            and room.min_x <= descriptor.bounds.center_cm[0] <= room.max_x
            and room.min_y <= descriptor.bounds.center_cm[1] <= room.max_y
        )
    )[:4]
    if not anchors and room.boundary_actor_id in by_id:
        anchors = (room.boundary_actor_id,)
    if not anchors:
        raise ValueError(
            "indoor room overview has no shared live Actor for visibility audit"
        )
    preferred_views = tuple(
        _view(f"view_{index + 1}", candidates[0], render.DEFAULT_FOV_DEG)
        for index, candidates in enumerate(candidate_groups)
    )
    preferred = scene_graph_capture.CameraPlan(
        preferred_views,
        room.center_cm,
        {
            "planner_id": planner_id,
            "pairing_policy": "environment-routed-frozen-absolute-cameras",
            "planning_source": "input_gt_shared_room_geometry",
            "scene_environment": scene_environment,
            "indoor_room": room.to_dict(),
            "indoor_camera_position_policy": "four-inset-room-corners",
            "indoor_visibility_anchor_actor_ids": list(anchors),
            "corner_candidate_count_per_view": [
                len(value) for value in candidate_groups
            ],
            "view_count": count,
        },
    )
    try:
        return _resolve(
            bridge,
            preferred,
            inventory,
            bounds,
            tuple(anchors for _value in preferred.views),
            planner_id=planner_id,
            pairing_policy="environment-routed-frozen-absolute-cameras",
            planning_source="input_gt_shared_room_geometry",
            scene_environment=scene_environment,
            timeout_s=timeout_s,
            candidate_poses_by_view=candidate_groups,
            overview_mode=True,
            require_enclosure=True,
            require_unique_camera_poses=True,
            enclosure_room_floor_z_cm=room.floor_z,
            visibility_policy=(
                "indoor-room-corners-overlap-enclosure-los-unique"
            ),
        )
    except _CameraPortfolioUnavailable as error:
        return _resolve_indoor_anchor_overview_fallback(
            bridge,
            input_scene,
            gt_scene,
            inventory,
            bounds,
            count=count,
            timeout_s=timeout_s,
            planner_id=planner_id,
            primary_failure=str(error),
        )


__all__ = [
    "CaseCameraPortfolio",
    "case_camera_portfolio",
    "require_scene_environment",
    "resolve_environment_overview_plan",
    "resolve_repair_target_plan",
    "shared_indoor_anchor_ids",
]
