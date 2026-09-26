"""Geometry-first camera placement for semantic Actor evidence.

RGB validation can detect a broken acquisition, but it cannot prove that the
requested Actor is visible. This module resolves a preferred camera pose
before rendering by sampling ordinary, scale-aware alternatives and asking
the loaded UE world whether each target AABB has an unobstructed line of sight.

Camera/AABB data plus the target's opaque live identity crosses the bridge so
the trace can distinguish the target from an overlapping wall or support.
Actor labels, assets, prompt text, retrieval ranks, and semantic verdicts never
enter the probe.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .actions import clamp_camera_pose
from .actor_inventory import (
    ActorBounds,
    ActorDescriptor,
    ActorInventorySnapshot,
)
from .asset_candidates import assess_bounds_camera_pose, plan_bounds_camera_poses
from .contracts import CameraPose, SceneBounds

_RESULT_KEY = "_SB_GEOMETRY_CAMERA_VISIBILITY"
_AZIMUTH_OFFSETS_DEG = (
    0.0,
    30.0,
    -30.0,
    60.0,
    -60.0,
    90.0,
    -90.0,
    120.0,
    -120.0,
    150.0,
    -150.0,
    180.0,
)
_VIEW_VARIANTS = (
    # Search from the target-side interior outward. A room divider can sit
    # between an otherwise healthy context camera and a small target; the
    # close variants let the camera occupy the same free-space compartment as
    # the target while the projection gate still prevents clipping.
    (0.25, 0.0),
    (0.30, 0.0),
    (0.35, 0.0),
    (0.50, 0.0),
    (0.72, 0.0),
    (1.00, 0.0),
    (0.72, 12.0),
    (1.20, 18.0),
)
_MAX_CANDIDATES = len(_AZIMUTH_OFFSETS_DEG) * len(_VIEW_VARIANTS) + 4
_MAX_OVERVIEW_CANDIDATES = 24
_MAX_OVERVIEW_ANCHOR_EXTENT_CM = 7_500.0
_MIN_OVERVIEW_ANCHOR_HEIGHT_CM = 25.0
_OVERVIEW_NON_CONTENT_CLASSES = (
    "Light",
    "Fog",
    "PostProcess",
    "Sky",
    "Camera",
    "Volume",
    "Landscape",
    "WorldSettings",
    "Brush",
)
# Keep the camera outside UE's ordinary near plane without forcing small
# props to occupy only a handful of pixels.  Every candidate is still gated
# by projected AABB framing and a live overlap trace before it can win.
_MIN_CAMERA_TARGET_DISTANCE_CM = 50.0
_MIN_VISIBILITY_SWEEP_RADIUS_CM = 0.5
_MAX_VISIBILITY_SWEEP_RADIUS_CM = 6.0
_VISIBILITY_SWEEP_EXTENT_FRACTION = 0.15
_ENCLOSURE_PROBE_DISTANCE_CM = 2_500.0
_MIN_HORIZONTAL_ENCLOSURE_HITS = 4
_THIN_SURFACE_PROXY_MAX_HALF_THICKNESS_CM = 5.0
_THIN_SURFACE_PROXY_MIN_ASPECT_RATIO = 4.0
_THIN_SURFACE_PROXY_REACH_TOLERANCE_CM = 5.0


@dataclass(frozen=True, slots=True)
class CameraVisibilityResolution:
    """One pre-render visibility result for a requested camera."""

    pose: CameraPose | None
    requested_pose: CameraPose
    actor_ids: tuple[str, ...]
    candidate_count: int
    selected_index: int | None
    visible_actor_count: int
    target_actor_count: int
    clear_ray_fraction: float
    minimum_actor_clear_fraction: float
    navigation_constrained: bool
    navigation_ok: bool
    scene_center_distance_cm: float
    reason: str
    required_actor_clear_fraction: float = 0.40
    camera_initial_overlap: bool = False
    camera_enclosure_ok: bool = True
    enclosure_up_hit: bool = False
    enclosure_up_trace_hit: bool = False
    enclosure_up_room_ceiling_fallback: bool = False
    enclosure_down_hit: bool = False
    enclosure_down_trace_hit: bool = False
    enclosure_down_room_floor_fallback: bool = False
    horizontal_enclosure_hits: int = 0
    required_horizontal_enclosure_hits: int = 0
    direct_target_ray_count: int = 0
    thin_surface_proxy_ray_count: int = 0
    thin_surface_open_ray_count: int = 0
    thin_surface_support_ray_count: int = 0
    thin_surface_support_actors: tuple[str, ...] = ()
    nearest_blocker_distance_to_sample_cm: float | None = None
    nearest_blocker_actor: str | None = None
    thin_surface_proxy_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_pose": self.requested_pose.to_dict(),
            "resolved_pose": self.pose.to_dict() if self.pose is not None else None,
            "actor_ids": list(self.actor_ids),
            "candidate_count": self.candidate_count,
            "selected_index": self.selected_index,
            "visible_actor_count": self.visible_actor_count,
            "target_actor_count": self.target_actor_count,
            "clear_ray_fraction": self.clear_ray_fraction,
            "minimum_actor_clear_fraction": self.minimum_actor_clear_fraction,
            "navigation_constrained": self.navigation_constrained,
            "navigation_ok": self.navigation_ok,
            "scene_center_distance_cm": self.scene_center_distance_cm,
            "reason": self.reason,
            "required_actor_clear_fraction": self.required_actor_clear_fraction,
            "camera_initial_overlap": self.camera_initial_overlap,
            "camera_enclosure_ok": self.camera_enclosure_ok,
            "enclosure_up_hit": self.enclosure_up_hit,
            "enclosure_up_trace_hit": self.enclosure_up_trace_hit,
            "enclosure_up_room_ceiling_fallback": (
                self.enclosure_up_room_ceiling_fallback
            ),
            "enclosure_down_hit": self.enclosure_down_hit,
            "enclosure_down_trace_hit": self.enclosure_down_trace_hit,
            "enclosure_down_room_floor_fallback": (
                self.enclosure_down_room_floor_fallback
            ),
            "horizontal_enclosure_hits": self.horizontal_enclosure_hits,
            "required_horizontal_enclosure_hits": (
                self.required_horizontal_enclosure_hits
            ),
            "direct_target_ray_count": self.direct_target_ray_count,
            "thin_surface_proxy_ray_count": self.thin_surface_proxy_ray_count,
            "thin_surface_open_ray_count": self.thin_surface_open_ray_count,
            "thin_surface_support_ray_count": self.thin_surface_support_ray_count,
            "thin_surface_support_actors": list(self.thin_surface_support_actors),
            "nearest_blocker_distance_to_sample_cm": (
                self.nearest_blocker_distance_to_sample_cm
            ),
            "nearest_blocker_actor": self.nearest_blocker_actor,
            "thin_surface_proxy_enabled": self.thin_surface_proxy_enabled,
        }


def _overview_anchor_descriptors(
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    *,
    limit: int = 3,
) -> tuple[ActorDescriptor, ...]:
    """Choose prompt-independent, sane scene subjects for overview grounding."""

    candidates = []
    for descriptor in inventory.actors:
        bounds = descriptor.bounds
        if bounds is None or not descriptor.eligible_for_stage1(scene_bounds):
            continue
        ex, ey, ez = bounds.extent_cm
        actor_class = str(descriptor.actor_class or "")
        if any(value in actor_class for value in _OVERVIEW_NON_CONTENT_CLASSES):
            continue
        if max(ex, ey, ez) > _MAX_OVERVIEW_ANCHOR_EXTENT_CM:
            continue
        if ez < _MIN_OVERVIEW_ANCHOR_HEIGHT_CM or max(ex, ey) <= 0.0:
            continue
        horizontal = max(ex, ey)
        prominence = horizontal * max(ez, _MIN_OVERVIEW_ANCHOR_HEIGHT_CM)
        candidates.append((prominence, descriptor.live_actor_id, descriptor))
    if not candidates:
        return ()
    candidates.sort(key=lambda value: (-value[0], value[1].casefold()))
    primary = candidates[0][2]
    assert primary.bounds is not None
    neighbours = sorted(
        (value[2] for value in candidates[1:]),
        key=lambda value: (
            math.dist(value.bounds.center_cm, primary.bounds.center_cm),
            value.live_actor_id.casefold(),
        ),
    )
    return (primary, *neighbours[: max(0, limit - 1)])


def _union_bounds(values: Sequence[ActorBounds]) -> ActorBounds:
    minimum = tuple(min(value.min_cm[axis] for value in values) for axis in range(3))
    maximum = tuple(max(value.max_cm[axis] for value in values) for axis in range(3))
    return ActorBounds.from_min_max(minimum, maximum)


def _look_at_pose(
    location: Sequence[float],
    target: Sequence[float],
    scene_bounds: SceneBounds,
) -> CameraPose:
    provisional = clamp_camera_pose(
        CameraPose(
            float(location[0]),
            float(location[1]),
            float(location[2]),
            0.0,
            0.0,
        ),
        scene_bounds,
    )
    dx = float(target[0]) - provisional.x
    dy = float(target[1]) - provisional.y
    dz = float(target[2]) - provisional.z
    horizontal = math.hypot(dx, dy)
    return clamp_camera_pose(
        CameraPose(
            provisional.x,
            provisional.y,
            provisional.z,
            math.degrees(math.atan2(dz, horizontal)),
            math.degrees(math.atan2(dy, dx)) if horizontal > 1e-6 else 0.0,
        ),
        scene_bounds,
    )


def camera_pose_key(pose: CameraPose) -> tuple[float, float, float, float, float]:
    """Return one rounded pose identity with yaw canonicalized to [0, 360)."""

    if not isinstance(pose, CameraPose):
        raise TypeError("pose must be a CameraPose")
    return (
        round(float(pose.x), 2),
        round(float(pose.y), 2),
        round(float(pose.z), 2),
        round(float(pose.pitch), 2),
        round(float(pose.yaw) % 360.0, 2) % 360.0,
    )


def visibility_pose_candidates(
    preferred: CameraPose,
    target_bounds: Sequence[ActorBounds],
    scene_bounds: SceneBounds,
    *,
    fov_degrees: float = 90.0,
) -> tuple[CameraPose, ...]:
    """Return framed orbital alternatives ordered around the preferred pose.

    The existing planner contributes its context/close/alternate portfolio.
    The additional orbit fills the directions that the three-view API
    intentionally discards, allowing the live world trace to choose the side
    of an indoor wall that actually contains the target.
    """

    targets = tuple(target_bounds)
    if not targets or any(not isinstance(value, ActorBounds) for value in targets):
        raise ValueError("target_bounds must contain at least one ActorBounds")
    if not isinstance(preferred, CameraPose):
        raise TypeError("preferred must be a CameraPose")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")

    bounds = _union_bounds(targets)
    center = bounds.center_cm
    dx = preferred.x - center[0]
    dy = preferred.y - center[1]
    dz = preferred.z - center[2]
    horizontal = math.hypot(dx, dy)
    distance = max(
        math.sqrt(dx * dx + dy * dy + dz * dz),
        math.sqrt(sum(value * value for value in bounds.extent_cm)) * 1.05,
        _MIN_CAMERA_TARGET_DISTANCE_CM,
    )
    base_azimuth = (
        math.degrees(math.atan2(dy, dx))
        if horizontal > 1e-6
        else (preferred.yaw + 180.0)
    )
    base_elevation = math.degrees(math.atan2(dz, max(horizontal, 1e-6)))
    standard = plan_bounds_camera_poses(
        center,
        bounds.extent_cm,
        scene_bounds,
        fov_degrees=fov_degrees,
    )

    candidates: list[CameraPose] = []
    seen: set[tuple[float, ...]] = set()

    def append(pose: CameraPose) -> None:
        if (
            math.dist(pose.location_cm, center)
            < _MIN_CAMERA_TARGET_DISTANCE_CM
        ):
            return
        assessment = assess_bounds_camera_pose(
            center,
            bounds.extent_cm,
            pose,
            fov_degrees=fov_degrees,
        )
        if not assessment.healthy:
            return
        key = camera_pose_key(pose)
        if key not in seen:
            seen.add(key)
            candidates.append(pose)

    append(preferred)
    for pose in sorted(
        standard,
        key=lambda value: (
            math.dist(value.location_cm, preferred.location_cm),
            abs((value.yaw - preferred.yaw + 180.0) % 360.0 - 180.0),
        ),
    ):
        append(pose)

    for offset in _AZIMUTH_OFFSETS_DEG:
        azimuth = math.radians(base_azimuth + offset)
        for scale, minimum_elevation in _VIEW_VARIANTS:
            elevation = math.radians(
                min(24.0, max(base_elevation, minimum_elevation))
            )
            radius = distance * scale
            horizontal_radius = radius * math.cos(elevation)
            append(
                _look_at_pose(
                    (
                        center[0] + math.cos(azimuth) * horizontal_radius,
                        center[1] + math.sin(azimuth) * horizontal_radius,
                        center[2] + math.sin(elevation) * radius,
                    ),
                    center,
                    scene_bounds,
                )
            )
            if len(candidates) >= _MAX_CANDIDATES:
                return tuple(candidates)
    return tuple(candidates)


def _probe_script(specs: Sequence[Mapping[str, Any]]) -> str:
    """Build one UE batch of visibility-only line traces."""

    return f"""
import math, unreal
_specs = {list(specs)!r}
_world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
_actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
_trace_type_query = getattr(unreal, "TraceTypeQuery", None)
_trace_channel = (
    getattr(_trace_type_query, "ECC_VISIBILITY", None)
    if _trace_type_query is not None else None
)
if _trace_channel is None:
    _trace_channel = unreal.TraceTypeQuery.TRACE_TYPE_QUERY1

_actors_by_stable_id = {{}}
_actors_by_unreal_name = {{}}
try:
    _level_actors = list(_actor_subsystem.get_all_level_actors() or [])
except Exception:
    _level_actors = []
for _actor in _level_actors:
    try:
        _actors_by_unreal_name[str(_actor.get_name()).casefold()] = _actor
    except Exception:
        pass
    try:
        _tags = list(_actor.tags or [])
    except Exception:
        _tags = []
    for _tag in _tags:
        _tag_text = str(_tag)
        _prefix = "simcodearena.stable_actor_id="
        if _tag_text.casefold().startswith(_prefix):
            _actors_by_stable_id[
                _tag_text[len(_prefix):].strip().casefold()
            ] = _actor

def _resolve_target_actor(_target):
    _actor_id = str(_target.get("actor_id") or "").strip()
    _stable_id = (
        _actor_id.split(":", 1)[1]
        if _actor_id.casefold().startswith("stable:")
        else ""
    )
    if _stable_id:
        _actor = _actors_by_stable_id.get(_stable_id.casefold())
        if _actor is not None:
            return _actor
    _unreal_name = str(_target.get("unreal_name") or "").strip()
    return _actors_by_unreal_name.get(_unreal_name.casefold())

def _project_nav(_point, _extent_values):
    try:
        _projected = unreal.NavigationSystemV1.project_point_to_navigation(
            _world,
            _point,
            None,
            None,
            unreal.Vector(*_extent_values),
        )
    except Exception:
        return None
    if _projected is None:
        return None
    for _axis, _allowed in zip(("x", "y", "z"), _extent_values):
        if (
            abs(float(getattr(_projected, _axis)) - float(getattr(_point, _axis)))
            > float(_allowed) + 1.0
        ):
            return None
    return _projected

def _path_connected(_start, _end):
    try:
        _path = unreal.NavigationSystemV1.find_path_to_location_synchronously(
            _world, _start, _end
        )
    except Exception:
        return False
    if _path is None:
        return False
    try:
        if _path.is_partial() or _path.is_valid() is False:
            return False
    except Exception:
        pass
    try:
        _points = list(_path.path_points or [])
    except Exception:
        _points = []
    if len(_points) < 2:
        return False
    _last = _points[-1]
    return math.sqrt(
        (_last.x - _end.x) ** 2
        + (_last.y - _end.y) ** 2
        + (_last.z - _end.z) ** 2
    ) <= 100.0

def _hit_values(_hit):
    try:
        return _hit.to_tuple()
    except Exception:
        return ()

def _hit_location(_values):
    # UE HitResult exposes location and impact_point next to each other. Prefer
    # impact_point, matching the repository's physics measurement parser.
    for _index in (5, 4):
        try:
            _value = _values[_index]
            if all(hasattr(_value, _axis) for _axis in "xyz"):
                return _value
        except Exception:
            pass
    return None

def _thin_surface_proxy_eligible(_target):
    _extent = [abs(float(_value)) for _value in _target["extent_cm"]]
    _thin_horizontal = min(_extent[0], _extent[1])
    if not 0.0 < _thin_horizontal <= {_THIN_SURFACE_PROXY_MAX_HALF_THICKNESS_CM!r}:
        return False
    _ratio = {_THIN_SURFACE_PROXY_MIN_ASPECT_RATIO!r}
    return (
        max(_extent[0], _extent[1]) >= _ratio * _thin_horizontal
        and _extent[2] >= _ratio * _thin_horizontal
    )

def _extend_to_thin_support(_start, _end):
    _dx = float(_end.x - _start.x)
    _dy = float(_end.y - _start.y)
    _dz = float(_end.z - _start.z)
    _length = math.sqrt(_dx * _dx + _dy * _dy + _dz * _dz)
    if _length <= 1.0e-6:
        return _end
    _extra = {_THIN_SURFACE_PROXY_REACH_TOLERANCE_CM!r} / _length
    return unreal.Vector(
        _end.x + _dx * _extra,
        _end.y + _dy * _extra,
        _end.z + _dz * _extra,
    )

def _hit_reaches_thin_target_surface(_values, _sample, _target):
    _location = _hit_location(_values)
    if _location is None:
        return False
    _tolerance = {_THIN_SURFACE_PROXY_REACH_TOLERANCE_CM!r}
    _distance = math.sqrt(
        (_location.x - _sample.x) ** 2
        + (_location.y - _sample.y) ** 2
        + (_location.z - _sample.z) ** 2
    )
    if _distance > _tolerance:
        return False
    _center = _target["center_cm"]
    _extent = _target["extent_cm"]
    return all(
        abs(float(getattr(_location, _axis_name)) - float(_center[_axis]))
        <= abs(float(_extent[_axis])) + _tolerance
        for _axis, _axis_name in enumerate(("x", "y", "z"))
    )

def _actor_name(_actor):
    if _actor is None:
        return None
    try:
        return str(_actor.get_name())
    except Exception:
        return None

def _blocking_actor_is_thin_target_support(_actor, _start, _target):
    if _actor is None:
        return False
    try:
        _origin, _extent_value = _actor.get_actor_bounds(False)
        _support_center = [
            float(_origin.x), float(_origin.y), float(_origin.z)
        ]
        _support_extent = [
            abs(float(_extent_value.x)),
            abs(float(_extent_value.y)),
            abs(float(_extent_value.z)),
        ]
    except Exception:
        return False
    _target_center = [float(_value) for _value in _target["center_cm"]]
    _target_extent = [
        abs(float(_value)) for _value in _target["extent_cm"]
    ]
    _thin_axis = 0 if _target_extent[0] <= _target_extent[1] else 1
    _wide_axis = 1 - _thin_axis
    # Reject whole-building proxies and small clutter. A valid supporting
    # surface is wall-like along the target's thin axis and spans the target
    # in both visible dimensions.
    if _support_extent[_thin_axis] > 25.0:
        return False
    if (
        _support_extent[_wide_axis] + 5.0
        < 0.8 * _target_extent[_wide_axis]
        or _support_extent[2] + 5.0 < 0.8 * _target_extent[2]
    ):
        return False
    _tolerance = {_THIN_SURFACE_PROXY_REACH_TOLERANCE_CM!r}
    if any(
        abs(_support_center[_axis] - _target_center[_axis])
        > _support_extent[_axis] + _target_extent[_axis] + _tolerance
        for _axis in range(3)
    ):
        return False
    _support_side = (
        _support_center[_thin_axis] - _target_center[_thin_axis]
    )
    _camera_side = (
        float((_start.x, _start.y)[_thin_axis])
        - _target_center[_thin_axis]
    )
    return (
        abs(_support_side) > 0.5
        and _camera_side * _support_side < 0.0
    )

def _hit_actor(_hit, _values):
    for _attribute in ("hit_actor", "actor"):
        try:
            _actor = getattr(_hit, _attribute)
        except Exception:
            _actor = None
        if _actor is not None:
            return _actor
    for _value in _values:
        if _value is None:
            continue
        try:
            if isinstance(_value, unreal.Actor):
                return _value
        except Exception:
            pass
        try:
            _owner = _value.get_owner()
        except Exception:
            _owner = None
        if _owner is not None:
            return _owner
    return None

def _same_actor(_left, _right):
    if _left is None or _right is None:
        return False
    if _left is _right:
        return True
    try:
        return str(_left.get_path_name()) == str(_right.get_path_name())
    except Exception:
        return False

def _initial_overlap(_values):
    try:
        return bool(_values[1])
    except Exception:
        return False

def _sphere_hit(_start, _end, _radius):
    _args = (
        _world,
        _start,
        _end,
        float(_radius),
        _trace_channel,
        True,
        [],
        unreal.DrawDebugTrace.NONE,
        True,
    )
    _raw = unreal.SystemLibrary.sphere_trace_single(*_args)
    if hasattr(_raw, "to_tuple"):
        return _raw
    if isinstance(_raw, tuple):
        for _item in _raw:
            if hasattr(_item, "to_tuple"):
                return _item
    return None

def _first_hit(_start, _end, _target):
    # A zero-width ray can pass through blinds, railings, or a narrow window
    # gap while the resulting photograph remains unusable.  The sweep must be
    # scale-aware, however: a fixed six-centimetre radius reaches behind a
    # thin wall-mounted sword, painting, or sign and reports its support wall
    # as the first hit.  Fifteen spatial samples already provide robustness;
    # this radius models only the local visual footprint of each sample.
    _extent = [max(0.0, float(_value)) for _value in _target["extent_cm"]]
    _radius = min(
        {_MAX_VISIBILITY_SWEEP_RADIUS_CM!r},
        max(
            {_MIN_VISIBILITY_SWEEP_RADIUS_CM!r},
            {_VISIBILITY_SWEEP_EXTENT_FRACTION!r} * min(_extent),
        ),
    )
    return _sphere_hit(_start, _end, _radius)

def _camera_overlaps_geometry(_start):
    _end = unreal.Vector(_start.x, _start.y, _start.z + 1.0)
    try:
        _hit = _sphere_hit(_start, _end, 20.0)
    except Exception:
        return False
    if _hit is None:
        return False
    _values = _hit_values(_hit)
    return bool(_values and (_values[0] or _initial_overlap(_values)))

def _has_blocking_hit(_start, _end):
    try:
        _hit = _sphere_hit(_start, _end, 1.0)
    except Exception:
        return False
    if _hit is None:
        return False
    _values = _hit_values(_hit)
    return bool(_values and _values[0])

def _camera_enclosure(_start, _distance):
    _up = _has_blocking_hit(
        _start, unreal.Vector(_start.x, _start.y, _start.z + _distance)
    )
    _down = _has_blocking_hit(
        _start, unreal.Vector(_start.x, _start.y, _start.z - _distance)
    )
    _horizontal_hits = 0
    for _dx, _dy in (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.70710678, 0.70710678), (0.70710678, -0.70710678),
        (-0.70710678, 0.70710678), (-0.70710678, -0.70710678),
    ):
        if _has_blocking_hit(
            _start,
            unreal.Vector(
                _start.x + _dx * _distance,
                _start.y + _dy * _distance,
                _start.z,
            ),
        ):
            _horizontal_hits += 1
    return _up, _down, _horizontal_hits

def _sample_points(_target):
    _center = _target["center_cm"]
    _extent = _target["extent_cm"]
    _points = [
        _center,
        [_center[0], _center[1], _center[2] + 0.72 * _extent[2]],
        [_center[0], _center[1], _center[2] - 0.72 * _extent[2]],
        [_center[0] + 0.72 * _extent[0], _center[1], _center[2]],
        [_center[0] - 0.72 * _extent[0], _center[1], _center[2]],
        [_center[0], _center[1] + 0.72 * _extent[1], _center[2]],
        [_center[0], _center[1] - 0.72 * _extent[1], _center[2]],
    ]
    for _sign_x in (-0.62, 0.62):
        for _sign_y in (-0.62, 0.62):
            for _sign_z in (-0.62, 0.62):
                _points.append([
                    _center[0] + _sign_x * _extent[0],
                    _center[1] + _sign_y * _extent[1],
                    _center[2] + _sign_z * _extent[2],
                ])
    return tuple(_points)

_results = []
_used_camera_pose_keys = set()
for _spec in _specs:
    _best = None
    _trace_errors = 0
    _overview_mode = _spec.get("mode") == "overview"
    _enclosure_mode = bool(_spec.get("require_enclosure"))
    _target_centers = [_value["center_cm"] for _value in _spec["targets"]]
    _target_anchor = unreal.Vector(
        *[
            sum(_value[_axis] for _value in _target_centers)
            / float(len(_target_centers))
            for _axis in range(3)
        ]
    )
    _target_nav = (
        None
        if _overview_mode
        else _project_nav(_target_anchor, (180.0, 180.0, 500.0))
    )
    _navigation_constrained = not _overview_mode and _target_nav is not None
    for _candidate_index, _candidate in enumerate(_spec["candidates"]):
        _candidate_pose_key = tuple(_candidate["pose_key"])
        if (
            _spec.get("require_unique_camera_pose")
            and _candidate_pose_key in _used_camera_pose_keys
        ):
            continue
        _start_values = _candidate["location_cm"]
        _start = unreal.Vector(*_start_values)
        _camera_nav = _project_nav(_start, (60.0, 60.0, 500.0))
        _navigation_ok = not _navigation_constrained
        if _navigation_constrained and _camera_nav is not None:
            _camera_height = float(_start.z - _camera_nav.z)
            _navigation_ok = (
                60.0 <= _camera_height <= 260.0
                and _path_connected(_camera_nav, _target_nav)
            )
        _visible_actor_count = 0
        _clear_rays = 0
        _total_rays = 0
        _direct_target_rays = 0
        _thin_surface_proxy_rays = 0
        _thin_surface_open_rays = 0
        _thin_surface_support_rays = 0
        _thin_surface_support_actors = set()
        _nearest_blocker_distance = None
        _nearest_blocker_actor = None
        _actor_clear_fractions = []
        _configured_required_actor_fraction = _spec.get(
            "required_actor_clear_fraction"
        )
        _required_actor_fraction = (
            float(_configured_required_actor_fraction)
            if _configured_required_actor_fraction is not None
            else 0.10
            if _overview_mode
            else 2.0 / 3.0
            if len(_spec["targets"]) > 1
            else 0.40
        )
        _camera_initial_overlap = _camera_overlaps_geometry(_start)
        _enclosure_up_hit = False
        _enclosure_up_trace_hit = False
        _enclosure_up_room_ceiling_fallback = False
        _enclosure_down_hit = False
        _enclosure_down_trace_hit = False
        _enclosure_down_room_floor_fallback = False
        _horizontal_enclosure_hits = 0
        _camera_enclosure_ok = True
        if _enclosure_mode:
            (
                _enclosure_up_trace_hit,
                _enclosure_down_trace_hit,
                _horizontal_enclosure_hits,
            ) = _camera_enclosure(
                _start,
                float(_spec["enclosure_probe_distance_cm"]),
            )
            _room_floor_z = _spec.get("enclosure_room_floor_z_cm")
            _room_ceiling_z = _spec.get("enclosure_room_ceiling_z_cm")
            _enclosure_up_room_ceiling_fallback = (
                _room_ceiling_z is not None
                and 0.0 <= float(_room_ceiling_z) - float(_start.z)
                <= float(_spec["enclosure_probe_distance_cm"])
            )
            _enclosure_up_hit = (
                _enclosure_up_trace_hit
                or _enclosure_up_room_ceiling_fallback
            )
            _enclosure_down_room_floor_fallback = (
                _room_floor_z is not None
                and 0.0 <= float(_start.z) - float(_room_floor_z)
                <= float(_spec["enclosure_probe_distance_cm"])
            )
            _enclosure_down_hit = (
                _enclosure_down_trace_hit
                or _enclosure_down_room_floor_fallback
            )
            _camera_enclosure_ok = (
                _enclosure_up_hit
                and _enclosure_down_hit
                and _horizontal_enclosure_hits
                >= int(_spec["required_horizontal_enclosure_hits"])
            )
        for _target in _spec["targets"]:
            _actor_clear = 0
            _target_actor = _resolve_target_actor(_target)
            _points = _sample_points(_target)
            _thin_surface_proxy = (
                bool(_spec.get("allow_thin_target_surface_proxy"))
                and _thin_surface_proxy_eligible(_target)
            )
            for _point in _points:
                _total_rays += 1
                _sample = unreal.Vector(*_point)
                _end = (
                    _extend_to_thin_support(_start, _sample)
                    if _thin_surface_proxy
                    else _sample
                )
                try:
                    _hit = _first_hit(_start, _end, _target)
                except Exception:
                    _trace_errors += 1
                    continue
                if _hit is None:
                    if _thin_surface_proxy:
                        # The GT target AABB is already projection-gated and
                        # the camera must later pass overlap and enclosure.
                        # For an explicitly eligible vertical thin target, no
                        # blocking hit to a point inside its bounds is positive
                        # open-ray evidence that a collisionless picture/sign
                        # is not occluded. Ordinary targets still fail closed.
                        _actor_clear += 1
                        _clear_rays += 1
                        _thin_surface_proxy_rays += 1
                        _thin_surface_open_rays += 1
                    continue
                _values = _hit_values(_hit)
                if _initial_overlap(_values):
                    _camera_initial_overlap = True
                    continue
                _blocking_actor = _hit_actor(_hit, _values)
                _hit_location_value = _hit_location(_values)
                if _hit_location_value is not None:
                    _blocker_distance = math.sqrt(
                        (_hit_location_value.x - _sample.x) ** 2
                        + (_hit_location_value.y - _sample.y) ** 2
                        + (_hit_location_value.z - _sample.z) ** 2
                    )
                    if (
                        _nearest_blocker_distance is None
                        or _blocker_distance < _nearest_blocker_distance
                    ):
                        _nearest_blocker_distance = _blocker_distance
                        _nearest_blocker_actor = _actor_name(_blocking_actor)
                if _same_actor(_blocking_actor, _target_actor):
                    _actor_clear += 1
                    _clear_rays += 1
                    _direct_target_rays += 1
                    continue
                if (
                    _thin_surface_proxy
                    and _blocking_actor_is_thin_target_support(
                        _blocking_actor, _start, _target
                    )
                ):
                    _actor_clear += 1
                    _clear_rays += 1
                    _thin_surface_proxy_rays += 1
                    _thin_surface_support_rays += 1
                    _support_name = _actor_name(_blocking_actor)
                    if _support_name:
                        _thin_surface_support_actors.add(_support_name)
                    continue
                if (
                    _thin_surface_proxy
                    and _hit_reaches_thin_target_surface(
                        _values, _sample, _target
                    )
                ):
                    # Some thin renderable meshes do not block UE's visibility
                    # channel. A wall immediately behind the GT target is then
                    # the first hit even though the target is in frame. This
                    # bounded surface-reach proof is only enabled by the
                    # GT-frozen indoor repair-target planner.
                    _actor_clear += 1
                    _clear_rays += 1
                    _thin_surface_proxy_rays += 1
                    continue
                # Bounds overlap is not identity.  A wall, shelf, table, or
                # support can occupy the target AABB and must remain an
                # occluder unless UE reports the actual target Actor as the
                # first hit.
            _actor_fraction = (
                float(_actor_clear) / float(len(_points))
                if _points else 0.0
            )
            _actor_clear_fractions.append(_actor_fraction)
            # A couple of rays through a railing or room-divider gap do not
            # make an Actor visually usable. A single supported prop may keep
            # its naturally hidden lower surface. A collection view is held
            # to a stronger per-Actor gate so an extreme longitudinal/sliver
            # view cannot masquerade as simultaneous multi-object evidence.
            if _actor_fraction >= _required_actor_fraction:
                _visible_actor_count += 1
        _target_count = len(_spec["targets"])
        _fraction = (
            float(_clear_rays) / float(_total_rays)
            if _total_rays else 0.0
        )
        _minimum_actor_fraction = (
            min(_actor_clear_fractions) if _actor_clear_fractions else 0.0
        )
        _value = {{
            "selected_index": _candidate_index,
            "visible_actor_count": _visible_actor_count,
            "target_actor_count": _target_count,
            "clear_ray_fraction": _fraction,
            "minimum_actor_clear_fraction": _minimum_actor_fraction,
            "direct_target_ray_count": _direct_target_rays,
            "thin_surface_proxy_ray_count": _thin_surface_proxy_rays,
            "thin_surface_open_ray_count": _thin_surface_open_rays,
            "thin_surface_support_ray_count": _thin_surface_support_rays,
            "thin_surface_support_actors": sorted(
                _thin_surface_support_actors
            ),
            "nearest_blocker_distance_to_sample_cm": (
                _nearest_blocker_distance
            ),
            "nearest_blocker_actor": _nearest_blocker_actor,
            "thin_surface_proxy_enabled": bool(
                _spec.get("allow_thin_target_surface_proxy")
            ),
            "required_actor_clear_fraction": _required_actor_fraction,
            "camera_initial_overlap": _camera_initial_overlap,
            "camera_enclosure_ok": _camera_enclosure_ok,
            "enclosure_up_hit": _enclosure_up_hit,
            "enclosure_up_trace_hit": _enclosure_up_trace_hit,
            "enclosure_up_room_ceiling_fallback": (
                _enclosure_up_room_ceiling_fallback
            ),
            "enclosure_down_hit": _enclosure_down_hit,
            "enclosure_down_trace_hit": _enclosure_down_trace_hit,
            "enclosure_down_room_floor_fallback": (
                _enclosure_down_room_floor_fallback
            ),
            "horizontal_enclosure_hits": _horizontal_enclosure_hits,
            "required_horizontal_enclosure_hits": int(
                _spec["required_horizontal_enclosure_hits"]
                if _enclosure_mode else 0
            ),
            "navigation_constrained": _navigation_constrained,
            "navigation_ok": _navigation_ok,
            "scene_center_distance_cm": float(
                _candidate["scene_center_distance_cm"]
            ),
            "target_distance_cm": float(_candidate["target_distance_cm"]),
        }}
        if (
            _best is None
            or (
                _navigation_ok,
                _camera_enclosure_ok,
                not _camera_initial_overlap,
                _visible_actor_count,
                _direct_target_rays,
                _thin_surface_proxy_rays,
                -(
                    _nearest_blocker_distance
                    if _nearest_blocker_distance is not None
                    else 1.0e30
                ),
                -_candidate_index if _overview_mode else -_value["target_distance_cm"],
                _minimum_actor_fraction,
                _fraction,
                -_value["scene_center_distance_cm"],
                -_value["target_distance_cm"] if _overview_mode else -_candidate_index,
            )
            > (
                _best["navigation_ok"],
                _best["camera_enclosure_ok"],
                not _best["camera_initial_overlap"],
                _best["visible_actor_count"],
                _best["direct_target_ray_count"],
                _best["thin_surface_proxy_ray_count"],
                -(
                    _best["nearest_blocker_distance_to_sample_cm"]
                    if _best["nearest_blocker_distance_to_sample_cm"] is not None
                    else 1.0e30
                ),
                -_best["selected_index"] if _overview_mode else -_best["target_distance_cm"],
                _best["minimum_actor_clear_fraction"],
                _best["clear_ray_fraction"],
                -_best["scene_center_distance_cm"],
                -_best["target_distance_cm"] if _overview_mode else -_best["selected_index"],
            )
        ):
            _best = _value
    if _best is None:
        _best = {{
            "selected_index": None,
            "visible_actor_count": 0,
            "target_actor_count": len(_spec["targets"]),
            "clear_ray_fraction": 0.0,
            "minimum_actor_clear_fraction": 0.0,
            "direct_target_ray_count": 0,
            "thin_surface_proxy_ray_count": 0,
            "thin_surface_open_ray_count": 0,
            "thin_surface_support_ray_count": 0,
            "thin_surface_support_actors": [],
            "nearest_blocker_distance_to_sample_cm": None,
            "nearest_blocker_actor": None,
            "thin_surface_proxy_enabled": bool(
                _spec.get("allow_thin_target_surface_proxy")
            ),
            "camera_initial_overlap": False,
            "camera_enclosure_ok": not _enclosure_mode,
            "enclosure_up_hit": False,
            "enclosure_up_trace_hit": False,
            "enclosure_up_room_ceiling_fallback": False,
            "enclosure_down_hit": False,
            "enclosure_down_trace_hit": False,
            "enclosure_down_room_floor_fallback": False,
            "horizontal_enclosure_hits": 0,
            "required_horizontal_enclosure_hits": (
                int(_spec["required_horizontal_enclosure_hits"])
                if _enclosure_mode else 0
            ),
            "navigation_constrained": _navigation_constrained,
            "navigation_ok": not _navigation_constrained,
            "scene_center_distance_cm": 0.0,
            "target_distance_cm": 0.0,
        }}
    _accepted = (
        _best["selected_index"] is not None
        and _best["navigation_ok"]
        and _best["camera_enclosure_ok"]
        and not _best["camera_initial_overlap"]
        and _best["target_actor_count"] > 0
        and (
            _best["visible_actor_count"] >= int(
                _spec.get("minimum_visible_actor_count")
                or _best["target_actor_count"]
            )
            or (
                _overview_mode
                and bool(
                    _spec.get(
                        "allow_enclosed_overview_without_visible_actor"
                    )
                )
            )
        )
    )
    _best["accepted"] = _accepted
    _best["trace_errors"] = _trace_errors
    if _accepted and _spec.get("require_unique_camera_pose"):
        _selected_candidate = _spec["candidates"][_best["selected_index"]]
        _used_camera_pose_keys.add(tuple(_selected_candidate["pose_key"]))
    _results.append(_best)
globals()[{_RESULT_KEY!r}] = {{"results": _results}}
"""


def resolve_visibility_camera_poses(
    bridge: Any,
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    poses: Sequence[CameraPose],
    actor_ids_by_pose: Sequence[Sequence[str]],
    *,
    timeout_s: float = 300.0,
    fov_degrees: float = 90.0,
    candidate_poses_by_pose: Sequence[Sequence[CameraPose]] | None = None,
    overview_mode: bool = False,
    require_enclosure: bool = False,
    require_unique_camera_poses: bool = False,
    enclosure_room_floor_z_cm: float | None = None,
    enclosure_room_ceiling_z_cm: float | None = None,
    required_actor_clear_fraction: float | None = None,
    max_overview_candidates_per_pose: int | None = None,
    allow_enclosed_overview_without_visible_actor: bool = False,
    allow_thin_target_surface_proxy: bool = False,
) -> tuple[CameraVisibilityResolution, ...]:
    """Resolve targeted poses before RGB acquisition.

    Targeted requests search alternate framed poses and fail closed when no
    line-of-sight pose exists. Actorless overview/grid requests are grounded to
    a small deterministic set of prominent, sane-bounds scene Actors. The live
    UE probe must find a view of at least one such subject before RGB capture;
    pixel health remains a second, independent acquisition gate.
    """

    requested = tuple(poses)
    actor_groups = tuple(tuple(value) for value in actor_ids_by_pose)
    if len(requested) != len(actor_groups):
        raise ValueError("poses and actor_ids_by_pose must have equal length")
    if any(not isinstance(value, CameraPose) for value in requested):
        raise TypeError("poses must contain CameraPose values")
    if not isinstance(require_enclosure, bool):
        raise TypeError("require_enclosure must be a boolean")
    if not isinstance(require_unique_camera_poses, bool):
        raise TypeError("require_unique_camera_poses must be a boolean")
    if enclosure_room_floor_z_cm is not None and (
        isinstance(enclosure_room_floor_z_cm, bool)
        or not isinstance(enclosure_room_floor_z_cm, (int, float))
        or not math.isfinite(float(enclosure_room_floor_z_cm))
    ):
        raise ValueError("enclosure_room_floor_z_cm must be finite when provided")
    if enclosure_room_ceiling_z_cm is not None and (
        isinstance(enclosure_room_ceiling_z_cm, bool)
        or not isinstance(enclosure_room_ceiling_z_cm, (int, float))
        or not math.isfinite(float(enclosure_room_ceiling_z_cm))
    ):
        raise ValueError(
            "enclosure_room_ceiling_z_cm must be finite when provided"
        )
    if required_actor_clear_fraction is not None and (
        isinstance(required_actor_clear_fraction, bool)
        or not isinstance(required_actor_clear_fraction, (int, float))
        or not math.isfinite(float(required_actor_clear_fraction))
        or not 0.0 < float(required_actor_clear_fraction) <= 1.0
    ):
        raise ValueError(
            "required_actor_clear_fraction must be in (0, 1] when provided"
        )
    if max_overview_candidates_per_pose is not None and (
        isinstance(max_overview_candidates_per_pose, bool)
        or not isinstance(max_overview_candidates_per_pose, int)
        or not 1 <= max_overview_candidates_per_pose <= _MAX_CANDIDATES
    ):
        raise ValueError(
            "max_overview_candidates_per_pose must be an integer in "
            f"[1, {_MAX_CANDIDATES}] when provided"
        )
    if not isinstance(allow_enclosed_overview_without_visible_actor, bool):
        raise TypeError(
            "allow_enclosed_overview_without_visible_actor must be a boolean"
        )
    if not isinstance(allow_thin_target_surface_proxy, bool):
        raise TypeError("allow_thin_target_surface_proxy must be a boolean")
    overview_candidate_limit = (
        max_overview_candidates_per_pose
        if max_overview_candidates_per_pose is not None
        else _MAX_OVERVIEW_CANDIDATES
    )
    explicit_candidates = (
        None
        if candidate_poses_by_pose is None
        else tuple(tuple(group) for group in candidate_poses_by_pose)
    )
    if explicit_candidates is not None:
        if len(explicit_candidates) != len(requested):
            raise ValueError(
                "candidate_poses_by_pose and poses must have equal length"
            )
        if any(not group for group in explicit_candidates):
            raise ValueError("explicit camera candidate groups must not be empty")
        if any(
            not isinstance(candidate, CameraPose)
            for group in explicit_candidates
            for candidate in group
        ):
            raise TypeError("explicit camera candidates must be CameraPose values")
    if (
        isinstance(fov_degrees, bool)
        or not isinstance(fov_degrees, (int, float))
        or not math.isfinite(float(fov_degrees))
        or not 1.0 <= float(fov_degrees) < 180.0
    ):
        raise ValueError("fov_degrees must be finite and between 1 and 180")
    by_id = {
        value.live_actor_id.casefold(): value
        for value in inventory.actors
        if value.bounds is not None
    }

    resolved: list[CameraVisibilityResolution | None] = [None] * len(requested)
    specs: list[dict[str, Any]] = []
    spec_indices: list[int] = []
    candidate_sets: list[tuple[CameraPose, ...]] = []
    target_sets: list[tuple[ActorBounds, ...]] = []
    resolution_actor_groups = list(actor_groups)
    spec_modes: list[str] = []
    overview_anchors = _overview_anchor_descriptors(inventory, scene_bounds)
    for index, (pose, actor_ids) in enumerate(zip(requested, actor_groups, strict=True)):
        mode = "overview" if overview_mode else "targeted"
        if not actor_ids:
            descriptors = overview_anchors
            if not descriptors:
                resolved[index] = CameraVisibilityResolution(
                    pose,
                    pose,
                    (),
                    1,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    False,
                    True,
                    round(math.dist(pose.location_cm, scene_bounds.center_cm), 6),
                    "overview_anchor_unavailable_rgb_health_deferred",
                    0.0,
                )
                continue
            actor_ids = tuple(value.live_actor_id for value in descriptors)
            resolution_actor_groups[index] = actor_ids
            mode = "overview"
        else:
            descriptors = tuple(
                descriptor
                for actor_id in actor_ids
                if (descriptor := by_id.get(str(actor_id).casefold())) is not None
                and descriptor.bounds is not None
            )
        targets = tuple(descriptor.bounds for descriptor in descriptors)
        if len(descriptors) != len(actor_ids):
            resolved[index] = CameraVisibilityResolution(
                None,
                pose,
                actor_ids,
                0,
                None,
                0,
                len(actor_ids),
                0.0,
                0.0,
                False,
                False,
                0.0,
                "target_bounds_unavailable",
            )
            continue
        candidates = (
            explicit_candidates[index]
            if explicit_candidates is not None
            else visibility_pose_candidates(
                pose,
                targets,
                scene_bounds,
                fov_degrees=float(fov_degrees),
            )
        )
        if mode == "overview":
            candidates = candidates[:overview_candidate_limit]
        if not candidates:
            resolved[index] = CameraVisibilityResolution(
                None,
                pose,
                actor_ids,
                0,
                None,
                0,
                len(actor_ids),
                0.0,
                0.0,
                False,
                False,
                0.0,
                "no_framed_camera_candidates",
            )
            continue
        spec_indices.append(index)
        candidate_sets.append(candidates)
        target_sets.append(targets)
        spec_modes.append(mode)
        target_center = _union_bounds(targets).center_cm
        specs.append(
            {
                "mode": mode,
                "require_enclosure": require_enclosure,
                "require_unique_camera_pose": require_unique_camera_poses,
                "enclosure_room_floor_z_cm": (
                    float(enclosure_room_floor_z_cm)
                    if enclosure_room_floor_z_cm is not None
                    else None
                ),
                "enclosure_room_ceiling_z_cm": (
                    float(enclosure_room_ceiling_z_cm)
                    if enclosure_room_ceiling_z_cm is not None
                    else None
                ),
                "enclosure_probe_distance_cm": (
                    _ENCLOSURE_PROBE_DISTANCE_CM if require_enclosure else 0.0
                ),
                "required_horizontal_enclosure_hits": (
                    _MIN_HORIZONTAL_ENCLOSURE_HITS if require_enclosure else 0
                ),
                "required_actor_clear_fraction": (
                    float(required_actor_clear_fraction)
                    if required_actor_clear_fraction is not None
                    else None
                ),
                "minimum_visible_actor_count": (
                    1 if mode == "overview" else len(descriptors)
                ),
                "allow_enclosed_overview_without_visible_actor": (
                    allow_enclosed_overview_without_visible_actor
                ),
                "allow_thin_target_surface_proxy": (
                    allow_thin_target_surface_proxy
                ),
                "candidates": [
                    {
                        "location_cm": list(value.location_cm),
                        "pose_key": list(camera_pose_key(value)),
                        "scene_center_distance_cm": math.dist(
                            value.location_cm,
                            scene_bounds.center_cm,
                        ),
                        "target_distance_cm": math.dist(
                            value.location_cm,
                            target_center,
                        ),
                    }
                    for value in candidates
                ],
                "targets": [
                    {
                        "actor_id": descriptor.live_actor_id,
                        "unreal_name": descriptor.unreal_name,
                        "center_cm": list(value.center_cm),
                        "extent_cm": list(value.extent_cm),
                    }
                    for descriptor, value in zip(
                        descriptors,
                        targets,
                        strict=True,
                    )
                ],
            }
        )

    if specs:
        payload = bridge.exec_python_result(
            _probe_script(specs),
            _RESULT_KEY,
            timeout=float(timeout_s),
        )
        values = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(values, Sequence) or len(values) != len(specs):
            raise RuntimeError("camera visibility probe returned an invalid result")
        for source_index, candidates, targets, mode, value in zip(
            spec_indices,
            candidate_sets,
            target_sets,
            spec_modes,
            values,
            strict=True,
        ):
            item = value if isinstance(value, Mapping) else {}
            selected_raw = item.get("selected_index")
            selected = (
                int(selected_raw)
                if isinstance(selected_raw, int)
                and not isinstance(selected_raw, bool)
                and 0 <= selected_raw < len(candidates)
                else None
            )
            accepted = item.get("accepted") is True and selected is not None
            resolved[source_index] = CameraVisibilityResolution(
                candidates[selected] if accepted and selected is not None else None,
                requested[source_index],
                resolution_actor_groups[source_index],
                len(candidates),
                selected if accepted else None,
                int(item.get("visible_actor_count") or 0),
                len(targets),
                round(float(item.get("clear_ray_fraction") or 0.0), 6),
                round(
                    float(item.get("minimum_actor_clear_fraction") or 0.0),
                    6,
                ),
                item.get("navigation_constrained") is True,
                item.get("navigation_ok") is not False,
                round(float(item.get("scene_center_distance_cm") or 0.0), 6),
                (
                    "overview_enclosed_rgb_health_fallback"
                    if (
                        accepted
                        and mode == "overview"
                        and int(item.get("visible_actor_count") or 0) == 0
                    )
                    else "overview_visibility_resolved"
                    if accepted and mode == "overview"
                    else "overview_scene_not_visible"
                    if mode == "overview"
                    else "line_of_sight_resolved"
                    if accepted
                    else "no_unoccluded_camera_pose"
                ),
                round(
                    float(item.get("required_actor_clear_fraction") or 0.40),
                    6,
                ),
                camera_initial_overlap=(
                    item.get("camera_initial_overlap") is True
                ),
                camera_enclosure_ok=(
                    item.get("camera_enclosure_ok") is not False
                ),
                enclosure_up_hit=item.get("enclosure_up_hit") is True,
                enclosure_up_trace_hit=(
                    item.get("enclosure_up_trace_hit") is True
                ),
                enclosure_up_room_ceiling_fallback=(
                    item.get("enclosure_up_room_ceiling_fallback") is True
                ),
                enclosure_down_hit=item.get("enclosure_down_hit") is True,
                enclosure_down_trace_hit=(
                    item.get("enclosure_down_trace_hit") is True
                ),
                enclosure_down_room_floor_fallback=(
                    item.get("enclosure_down_room_floor_fallback") is True
                ),
                horizontal_enclosure_hits=int(
                    item.get("horizontal_enclosure_hits") or 0
                ),
                required_horizontal_enclosure_hits=int(
                    item.get("required_horizontal_enclosure_hits") or 0
                ),
                direct_target_ray_count=int(
                    item.get("direct_target_ray_count") or 0
                ),
                thin_surface_proxy_ray_count=int(
                    item.get("thin_surface_proxy_ray_count") or 0
                ),
                thin_surface_open_ray_count=int(
                    item.get("thin_surface_open_ray_count") or 0
                ),
                thin_surface_support_ray_count=int(
                    item.get("thin_surface_support_ray_count") or 0
                ),
                thin_surface_support_actors=tuple(
                    str(value)
                    for value in (
                        item.get("thin_surface_support_actors") or ()
                    )
                ),
                nearest_blocker_distance_to_sample_cm=(
                    round(
                        float(
                            item["nearest_blocker_distance_to_sample_cm"]
                        ),
                        6,
                    )
                    if item.get(
                        "nearest_blocker_distance_to_sample_cm"
                    ) is not None
                    else None
                ),
                nearest_blocker_actor=(
                    str(item["nearest_blocker_actor"])
                    if item.get("nearest_blocker_actor") is not None
                    else None
                ),
                thin_surface_proxy_enabled=(
                    item.get("thin_surface_proxy_enabled") is True
                ),
            )

    if any(value is None for value in resolved):
        raise RuntimeError("camera visibility resolver left an input unresolved")
    return tuple(value for value in resolved if value is not None)


__all__ = [
    "CameraVisibilityResolution",
    "camera_pose_key",
    "resolve_visibility_camera_poses",
    "visibility_pose_candidates",
]
