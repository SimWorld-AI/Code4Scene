"""Deterministic scene-graph camera planning and acquisition-quality policy.

This module is adapted from an earlier stand-alone scene-reconstruction
capture script. Its measured capture lessons are exposed through one
verifier-owned policy surface:

* choose the densest bounded region rather than assuming world origin;
* infer local ground from the median underside of nearby standing objects;
* use aerial poses for holistic visual evidence;
* reject valid-but-unusable near-black PNGs before a consumer sees them.

No runtime imports that earlier script and no model chooses a
camera.  Each verifier still requests and caches its own screenshots: sharing
this deterministic planner never means sharing a RenderSet or image artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

from .render import (
    DEFAULT_FOV_DEG,
    HEIGHT_RADIUS_MULTIPLIER,
    RADIUS_HALF_EXTENT_MULTIPLIER,
    Viewpoint,
)


CAPTION_CAMERA_PLAN = "caption-environment-routed-visibility-frozen"
CAPTION_FRAME_QUALITY = "caption-near-black-relight"
REFERENCE_CAMERA_PLAN = "reference-grounded-clearance-overview"
REFERENCE_FRAME_QUALITY = "reference-occlusion-aware-relight"
GT_VISUAL_CAMERA_PLAN = "visual-gt-environment-routed-visibility-frozen"
GT_REPAIR_TARGET_OUTDOOR_CAMERA_PLAN = (
    "visual-gt-repair-target-outdoor-visibility-frozen-local"
)
GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN = (
    "visual-gt-repair-target-indoor-room-side-local"
)
GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN = (
    "visual-gt-repair-target-authored-camera-priority-local"
)
GT_REPAIR_TARGET_CASE_CAMERA_VIEW_COUNT = 4
GT_CAMERA_ADJUSTMENT_CONTRACT = (
    "candidate-gt-joint-camera-adjustment"
)
INDOOR_OVERVIEW_CAMERA_FALLBACK_STAGE = (
    "shared-anchor-pool-enclosed-visibility"
)
GT_VISUAL_FRAME_QUALITY = (
    "visual-gt-geometry-locked-per-view-independent-photometry"
)
GT_DENSE_CORE_AERIAL_SEED_PLAN = "visual-gt-dense-core-aerial-seed"
GT_REPAIR_TARGET_AERIAL_SEED_PLAN = "visual-gt-repair-target-aerial-seed"
SUPPORTED_CAMERA_PLAN_POLICIES = frozenset({
    CAPTION_CAMERA_PLAN,
    REFERENCE_CAMERA_PLAN,
    GT_VISUAL_CAMERA_PLAN,
    GT_REPAIR_TARGET_OUTDOOR_CAMERA_PLAN,
    GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN,
    GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN,
})
SUPPORTED_FRAME_QUALITY_POLICIES = frozenset({
    CAPTION_FRAME_QUALITY,
    REFERENCE_FRAME_QUALITY,
    GT_VISUAL_FRAME_QUALITY,
})

# Measured near-black guard.  A black PNG is structurally valid, so evidence
# validation must explicitly reject it before independent captioning.
NEAR_BLACK_LUMA = 12.0
AUTO_MIN_P95_LUMA = 24.0
AUTO_MAX_DARK_FRACTION = 0.80
AUTO_MAX_WHITE_CLIP_FRACTION = 0.20
AUTO_MAX_CHANNEL_CLIP_FRACTION = 0.45

CROP_CM = 20_000.0
GRID_CM = 2_500.0
MIN_REACH_CM = 2_500.0
MAX_SANE_EXTENT_CM = 20_000.0
GROUND_MAX_THICKNESS_CM = 300.0
WIDE_OBSTACLE_CM = CROP_CM * 0.4
GT_VISUAL_CORE_COVERAGE = 0.95
GT_VISUAL_MIN_REACH_CM = 1_500.0
REFERENCE_MIN_RADIUS_CM = 1_800.0
REFERENCE_MAX_NEAR_RADIUS_CM = 6_000.0
REFERENCE_MAX_WIDE_RADIUS_CM = 8_000.0
REFERENCE_MIN_CLEARANCE_RADIUS_CM = 6_000.0
REFERENCE_MAX_CLEARANCE_RADIUS_CM = 24_000.0
REFERENCE_MAX_CLEARANCE_HEIGHT_CM = 50_000.0
REFERENCE_CLEARANCE_RADIUS_MULTIPLIER = 1.75
REFERENCE_CLEARANCE_WIDE_RADIUS_MULTIPLIER = 1.50
REFERENCE_CLEARANCE_HEIGHT_RADIUS_MULTIPLIER = 0.75
REFERENCE_VERTICAL_CLEARANCE_MARGIN_CM = 2_500.0
REFERENCE_VERTICAL_CLEARANCE_RADIUS_MULTIPLIER = 0.25
REFERENCE_RADIAL_CLEARANCE_MARGIN_CM = 1_500.0
REFERENCE_RADIAL_CLEARANCE_REACH_MULTIPLIER = 0.75
REFERENCE_OVERVIEW_GROUND_QUANTILE = 0.10
REFERENCE_OVERVIEW_TOP_QUANTILE = 0.95
REFERENCE_OVERVIEW_FOCUS_HEIGHT_FRACTION = 0.35
REFERENCE_CORE_COVERAGE = 0.90
REFERENCE_MIN_DETAIL_EDGE_FRACTION = 0.060
REFERENCE_MIN_DETAILED_VIEWS = 2
REFERENCE_RECOVERY_MIN_DETAIL_EDGE_FRACTION = 0.025
REFERENCE_RECOVERY_MIN_P99_SPATIAL_GRADIENT = 12.0
REFERENCE_RECOVERY_MAX_FLAT_COMPONENT_FRACTION = 0.50
REFERENCE_RECOVERY_MAX_CENTRAL_FLAT_FRACTION = 0.50
REFERENCE_RECOVERY_MIN_STRUCTURED_VIEWS = 3
REFERENCE_MIN_OCCLUDING_FLAT_COMPONENT_FRACTION = 0.35
REFERENCE_MIN_OCCLUDING_CENTER_FLAT_FRACTION = 0.35
REFERENCE_MIN_OCCLUDING_DARK_FRACTION = 0.55
REFERENCE_MAX_OCCLUDED_VIEWS = 1
REFERENCE_SPARSE_VERTICAL_MIN_ASPECT_RATIO = 2.0

_VISUAL_PLANNING_PROXIES = (
    "ReflectionCapture",
)

_NON_CONTENT_CLASSES = (
    "Light",
    "Fog",
    "PostProcess",
    "Sky",
    "Player",
    "Camera",
    "Volume",
    "Landscape",
    "WorldSettings",
    "Brush",
)


@dataclass(frozen=True, slots=True)
class _Box:
    x: float
    y: float
    z: float
    ex: float
    ey: float
    ez: float

    @property
    def bottom(self) -> float:
        return self.z - self.ez

    @property
    def top(self) -> float:
        return self.z + self.ez


@dataclass(frozen=True, slots=True)
class SceneFrame:
    """One dense scene region expressed in UE centimetres."""

    anchor_cm: tuple[float, float, float]
    reach_cm: float
    content_count: int
    crop_min_cm: tuple[float, float]
    crop_max_cm: tuple[float, float]
    boxes: tuple[_Box, ...]

    def audit(self) -> dict[str, Any]:
        return {
            "anchor_cm": list(self.anchor_cm),
            "reach_cm": self.reach_cm,
            "content_count": self.content_count,
            "crop_min_cm": list(self.crop_min_cm),
            "crop_max_cm": list(self.crop_max_cm),
        }


@dataclass(frozen=True, slots=True)
class CameraPlan:
    views: tuple[Viewpoint, ...]
    anchor_cm: tuple[float, float, float]
    audit: Mapping[str, Any]


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _content_box(actor: Mapping[str, Any]) -> _Box | None:
    if actor.get("keep") is True:
        return None
    cls = str(actor.get("cls") or actor.get("class") or "")
    if any(value in cls for value in _NON_CONTENT_CLASSES):
        return None
    bounds = actor.get("bounds")
    bounds_origin = (
        bounds.get("origin_cm") if isinstance(bounds, Mapping) else None
    )
    location = bounds_origin
    if not isinstance(location, Sequence) or isinstance(location, (str, bytes)):
        location = actor.get("loc")
    if not isinstance(location, Sequence) or isinstance(location, (str, bytes)):
        transform = actor.get("transform")
        location = (
            transform.get("location_cm")
            if isinstance(transform, Mapping)
            else None
        )
    extent = actor.get("extent")
    if not isinstance(extent, Sequence) or isinstance(extent, (str, bytes)):
        extent = bounds.get("extent_cm") if isinstance(bounds, Mapping) else None
    if not isinstance(location, Sequence) or len(location) < 3:
        return None
    if not isinstance(extent, Sequence) or len(extent) < 3:
        extent = (50.0, 50.0, 50.0)
    values = tuple(_number(value) for value in (*location[:3], *extent[:3]))
    if any(value is None for value in values):
        return None
    x, y, z, ex, ey, ez = (float(value) for value in values)
    if min(ex, ey, ez) < 0.0:
        return None
    return _Box(x, y, z, ex, ey, ez)


def _densest_center(boxes: Sequence[_Box]) -> tuple[float, float, int]:
    if not boxes:
        return 0.0, 0.0, 0
    xs = [value.x for value in boxes]
    ys = [value.y for value in boxes]
    half = CROP_CM / 2.0
    best = (-1, min(xs), min(ys))
    cx = min(xs)
    while cx <= max(xs):
        cy = min(ys)
        while cy <= max(ys):
            count = sum(
                abs(value.x - cx) <= half and abs(value.y - cy) <= half
                for value in boxes
            )
            candidate = (count, -cx, -cy)
            current = (best[0], -best[1], -best[2])
            if candidate > current:
                best = (count, cx, cy)
            cy += GRID_CM
        cx += GRID_CM
    return best[1], best[2], best[0]


def _is_ground(box: _Box) -> bool:
    return box.ez < GROUND_MAX_THICKNESS_CM or max(box.ex, box.ey) >= WIDE_OBSTACLE_CM


def _local_ground(x: float, y: float, boxes: Sequence[_Box]) -> float:
    standing = [
        value
        for value in boxes
        if not _is_ground(value) and value.ez < MAX_SANE_EXTENT_CM
    ]
    near = sorted(standing, key=lambda value: math.hypot(x - value.x, y - value.y))[:12]
    if near:
        bottoms = sorted(value.bottom for value in near)
        base = bottoms[len(bottoms) // 2]
    else:
        sane = [value.bottom for value in boxes if value.ez < MAX_SANE_EXTENT_CM]
        base = min(sane, default=0.0)
    underfoot = [
        value.top
        for value in boxes
        if abs(x - value.x) <= value.ex
        and abs(y - value.y) <= value.ey
        and _is_ground(value)
        and base - 200.0 <= value.top <= base + 500.0
    ]
    return max(underfoot, default=base)


def scene_frame(actors: Sequence[Mapping[str, Any]]) -> SceneFrame:
    """Locate and size the densest content region in an inventory."""

    all_boxes = tuple(
        value for actor in actors if (value := _content_box(actor)) is not None
    )
    cx, cy, _count = _densest_center(all_boxes)
    half = CROP_CM / 2.0
    inside = tuple(
        value
        for value in all_boxes
        if abs(value.x - cx) <= half and abs(value.y - cy) <= half
    )
    ground = _local_ground(cx, cy, inside)
    horizontal_reaches = [
        max(abs(value.x - cx) + value.ex, abs(value.y - cy) + value.ey)
        for value in inside
        if max(value.ex, value.ey) < MAX_SANE_EXTENT_CM
    ]
    reach = max(MIN_REACH_CM, min(half, max(horizontal_reaches, default=MIN_REACH_CM)))
    return SceneFrame(
        anchor_cm=(round(cx, 6), round(cy, 6), round(ground, 6)),
        reach_cm=round(reach, 6),
        content_count=len(inside),
        crop_min_cm=(round(cx - half, 6), round(cy - half, 6)),
        crop_max_cm=(round(cx + half, 6), round(cy + half, 6)),
        boxes=inside,
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("median needs at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile needs at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("quantile fraction must be in [0, 1]")
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1),
    )
    return ordered[index]


def _gt_dense_core_frame(
    actors: Sequence[Mapping[str, Any]],
) -> tuple[SceneFrame, dict[str, Any]]:
    """Robust GT-only frame for a legible whole-scene visual comparison.

    The earlier planner sizes from the farthest bounded Actor around the first
    maximum-density grid point. When a 200 m crop contains the whole authored
    environment, many grid points tie and that first point can sit near a
    corner. One fringe Actor then pushes ``reach_cm`` to the 100 m crop cap,
    turning the developed scene into a small silhouette.

    The current planner keeps the same Candidate-independent dense-region gate, then recentres
    on the bounded GT Actor medians and fits the 95th-percentile Actor reach.
    The structured Candidate-to-GT comparison still covers every Actor; this
    robust crop controls only whether the visual evidence is discriminative.
    """

    bounded: list[tuple[Mapping[str, Any], _Box]] = []
    for actor in actors:
        box = _content_box(actor)
        if box is not None:
            bounded.append((actor, box))
    if not bounded:
        fallback = SceneFrame(
            anchor_cm=(0.0, 0.0, 0.0),
            reach_cm=GT_VISUAL_MIN_REACH_CM,
            content_count=0,
            crop_min_cm=(-GT_VISUAL_MIN_REACH_CM, -GT_VISUAL_MIN_REACH_CM),
            crop_max_cm=(GT_VISUAL_MIN_REACH_CM, GT_VISUAL_MIN_REACH_CM),
            boxes=(),
        )
        return fallback, {
            "coverage_target": GT_VISUAL_CORE_COVERAGE,
            "coverage_actor_count": 0,
            "coverage_fraction": 0.0,
            "bounded_actor_count": 0,
            "proxy_actor_count": 0,
            "full_reach_cm": 0.0,
            "fallback": "empty_gt_inventory",
        }

    initial_x, initial_y, _ = _densest_center([box for _, box in bounded])
    half = CROP_CM / 2.0
    region = [
        (actor, box)
        for actor, box in bounded
        if abs(box.x - initial_x) <= half and abs(box.y - initial_y) <= half
    ]
    visual_region = [
        (actor, box)
        for actor, box in region
        if not any(
            proxy in str(actor.get("cls") or actor.get("class") or "")
            for proxy in _VISUAL_PLANNING_PROXIES
        )
    ]
    if not visual_region:
        visual_region = region

    boxes = tuple(box for _, box in visual_region)
    center_x = _median([box.x for box in boxes])
    center_y = _median([box.y for box in boxes])
    ground = _local_ground(center_x, center_y, boxes)
    reaches = sorted(
        max(
            abs(box.x - center_x) + box.ex,
            abs(box.y - center_y) + box.ey,
        )
        for box in boxes
        if max(box.ex, box.ey) < MAX_SANE_EXTENT_CM
    )
    if reaches:
        coverage_index = max(
            0,
            min(
                len(reaches) - 1,
                math.ceil(len(reaches) * GT_VISUAL_CORE_COVERAGE) - 1,
            ),
        )
        percentile_reach = reaches[coverage_index]
        full_reach = reaches[-1]
        covered = sum(value <= percentile_reach for value in reaches)
    else:
        percentile_reach = GT_VISUAL_MIN_REACH_CM
        full_reach = 0.0
        covered = 0
    reach = max(
        GT_VISUAL_MIN_REACH_CM,
        min(half, percentile_reach),
    )
    anchor = (
        round(center_x, 6),
        round(center_y, 6),
        round(ground, 6),
    )
    frame = SceneFrame(
        anchor_cm=anchor,
        reach_cm=round(reach, 6),
        content_count=len(boxes),
        crop_min_cm=(round(center_x - reach, 6), round(center_y - reach, 6)),
        crop_max_cm=(round(center_x + reach, 6), round(center_y + reach, 6)),
        boxes=boxes,
    )
    return frame, {
        "coverage_target": GT_VISUAL_CORE_COVERAGE,
        "coverage_actor_count": covered,
        "coverage_fraction": (
            round(covered / len(reaches), 6) if reaches else 0.0
        ),
        "bounded_actor_count": len(bounded),
        "dense_region_actor_count": len(region),
        "proxy_actor_count": len(region) - len(visual_region),
        "full_reach_cm": round(full_reach, 6),
        "percentile_reach_cm": round(percentile_reach, 6),
        "initial_dense_grid_center_cm": [
            round(initial_x, 6),
            round(initial_y, 6),
        ],
        "anchor_policy": "bounded_gt_actor_xy_median",
        "reach_policy": "bounded_gt_actor_reach_p95",
    }
def _aerial_views(
    frame: SceneFrame,
    *,
    count: int,
    radius_cm: float,
    angle_offset_deg: float = 45.0,
) -> tuple[Viewpoint, ...]:
    if count < 2:
        raise ValueError("an aerial portfolio needs at least two views")
    cx, cy, cz = frame.anchor_cm
    height = radius_cm * HEIGHT_RADIUS_MULTIPLIER
    views = []
    for index in range(count):
        angle = math.radians(angle_offset_deg + 360.0 * index / count)
        x = radius_cm * math.cos(angle)
        y = radius_cm * math.sin(angle)
        yaw = math.degrees(math.atan2(-y, -x))
        pitch = -math.degrees(math.atan2(height, math.hypot(x, y)))
        views.append(
            Viewpoint(
                name=f"view_{index}",
                location=[round(cx + x, 6), round(cy + y, 6), round(cz + height, 6)],
                rotation=[0.0, round(pitch, 6), round(yaw, 6)],
                fov_deg=DEFAULT_FOV_DEG,
            )
        )
    return tuple(views)


def _aerial_portfolio(
    frame: SceneFrame,
    *,
    radii_cm: Sequence[float],
    heights_cm: Sequence[float] | None = None,
    angle_offset_deg: float = 45.0,
) -> tuple[Viewpoint, ...]:
    """Return a deterministic mixed-distance gallery around one anchor."""

    if len(radii_cm) < 2:
        raise ValueError("an aerial portfolio needs at least two radii")
    if heights_cm is not None and len(heights_cm) != len(radii_cm):
        raise ValueError("aerial portfolio heights must match its radii")
    cx, cy, cz = frame.anchor_cm
    views = []
    for index, raw_radius in enumerate(radii_cm):
        radius = float(raw_radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("aerial portfolio radii must be finite and positive")
        angle = math.radians(
            angle_offset_deg + 360.0 * index / len(radii_cm)
        )
        height = (
            radius * HEIGHT_RADIUS_MULTIPLIER
            if heights_cm is None
            else float(heights_cm[index])
        )
        if not math.isfinite(height) or height <= 0.0:
            raise ValueError(
                "aerial portfolio heights must be finite and positive"
            )
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        yaw = math.degrees(math.atan2(-y, -x))
        pitch = -math.degrees(math.atan2(height, math.hypot(x, y)))
        views.append(
            Viewpoint(
                name=f"view_{index}",
                location=[
                    round(cx + x, 6),
                    round(cy + y, 6),
                    round(cz + height, 6),
                ],
                rotation=[0.0, round(pitch, 6), round(yaw, 6)],
                fov_deg=DEFAULT_FOV_DEG,
            )
        )
    return tuple(views)


def _geometry_clearance_radii(
    frame: SceneFrame,
    *,
    base_radii_cm: Sequence[float],
    angle_offset_deg: float,
) -> tuple[tuple[float, ...], tuple[dict[str, float | bool], ...]]:
    """Keep every overview camera beyond content on its viewing ray.

    A radius derived only from the scene's centre can still put an oblique
    camera directly above a building near the edge of that scene.  The image
    then becomes a detailed roof/wall close-up, which pixel-quality gates
    cannot reliably distinguish from a useful overview.  Project every UE
    Actor bound onto each camera ray and move that camera beyond the outermost
    projected surface plus a deterministic margin.
    """

    if len(base_radii_cm) < 2:
        raise ValueError("geometry clearance needs at least two radii")
    cx, cy, _cz = frame.anchor_cm
    margin = max(
        REFERENCE_RADIAL_CLEARANCE_MARGIN_CM,
        frame.reach_cm * REFERENCE_RADIAL_CLEARANCE_REACH_MULTIPLIER,
    )
    radii: list[float] = []
    audit: list[dict[str, float | bool]] = []
    for index, raw_base_radius in enumerate(base_radii_cm):
        base_radius = float(raw_base_radius)
        angle_deg = angle_offset_deg + 360.0 * index / len(base_radii_cm)
        angle = math.radians(angle_deg)
        ux = math.cos(angle)
        uy = math.sin(angle)
        radial_content_front = max(
            (
                (box.x - cx) * ux
                + (box.y - cy) * uy
                + abs(ux) * box.ex
                + abs(uy) * box.ey
                for box in frame.boxes
            ),
            default=0.0,
        )
        required_radius = max(0.0, radial_content_front) + margin
        radius = min(
            REFERENCE_MAX_CLEARANCE_RADIUS_CM,
            max(base_radius, required_radius),
        )
        radii.append(radius)
        audit.append({
            "angle_deg": round(angle_deg, 6),
            "base_radius_cm": round(base_radius, 6),
            "radial_content_front_cm": round(radial_content_front, 6),
            "radial_clearance_margin_cm": round(margin, 6),
            "required_radius_cm": round(required_radius, 6),
            "geometry_clearance_applied": radius > base_radius + 1e-6,
            "radius_cm": round(radius, 6),
        })
    return tuple(radii), tuple(audit)


def _candidate_overview_frame(
    actors: Sequence[Mapping[str, Any]],
) -> tuple[SceneFrame, dict[str, Any]]:
    """Find a robust, visible Candidate core without prompt-specific names.

    Candidate worlds sometimes contain enormous terrain proxy cubes. Their
    bounds are useful to UE, but letting them choose a camera can put the real
    authored scene hundreds of metres away in fog. The current planner therefore selects the
    densest region from sane standing content, recentres on its median, and
    fits ninety percent of that content.
    """

    all_boxes = tuple(
        value for actor in actors if (value := _content_box(actor)) is not None
    )
    sane_standing = tuple(
        value
        for value in all_boxes
        if max(value.ex, value.ey, value.ez) < MAX_SANE_EXTENT_CM
        and not _is_ground(value)
    )
    planning_boxes = sane_standing or tuple(
        value
        for value in all_boxes
        if max(value.ex, value.ey, value.ez) < MAX_SANE_EXTENT_CM
    )
    if not planning_boxes:
        fallback = scene_frame(actors)
        return fallback, {
            "bounded_actor_count": len(all_boxes),
            "planning_actor_count": 0,
            "fallback": "no_sane_content_bounds",
        }

    initial_x, initial_y, _ = _densest_center(planning_boxes)
    half = CROP_CM / 2.0
    region = tuple(
        value
        for value in planning_boxes
        if abs(value.x - initial_x) <= half
        and abs(value.y - initial_y) <= half
    )
    center_x = _median([value.x for value in region])
    center_y = _median([value.y for value in region])
    # A city block can contain hundreds of rooftop props near its XY median.
    # The old nearest-object median therefore mistook the roof deck for the
    # ground and aimed every overview at the skyline's upper storeys.  Use a
    # robust lower standing-bottom quantile, then aim partway up the robust
    # vertical span.  This keeps ground, facades, and roofs in the same frame.
    ground = _quantile(
        [value.bottom for value in region],
        REFERENCE_OVERVIEW_GROUND_QUANTILE,
    )
    robust_top = max(
        ground,
        _quantile(
            [value.top for value in region],
            REFERENCE_OVERVIEW_TOP_QUANTILE,
        ),
    )
    focus_z = ground + (
        robust_top - ground
    ) * REFERENCE_OVERVIEW_FOCUS_HEIGHT_FRACTION
    reaches = sorted(
        max(
            abs(value.x - center_x) + value.ex,
            abs(value.y - center_y) + value.ey,
        )
        for value in region
    )
    coverage_index = max(
        0,
        min(
            len(reaches) - 1,
            math.ceil(len(reaches) * REFERENCE_CORE_COVERAGE) - 1,
        ),
    )
    percentile_reach = reaches[coverage_index]
    full_reach = reaches[-1]
    reach = max(
        MIN_REACH_CM,
        min(REFERENCE_MAX_NEAR_RADIUS_CM, percentile_reach),
    )
    frame = SceneFrame(
        anchor_cm=(
            round(center_x, 6),
            round(center_y, 6),
            round(focus_z, 6),
        ),
        reach_cm=round(reach, 6),
        content_count=len(region),
        crop_min_cm=(round(center_x - half, 6), round(center_y - half, 6)),
        crop_max_cm=(round(center_x + half, 6), round(center_y + half, 6)),
        boxes=region,
    )
    return frame, {
        "bounded_actor_count": len(all_boxes),
        "planning_actor_count": len(planning_boxes),
        "dense_region_actor_count": len(region),
        "coverage_target": REFERENCE_CORE_COVERAGE,
        "coverage_actor_count": coverage_index + 1,
        "coverage_fraction": round((coverage_index + 1) / len(reaches), 6),
        "percentile_reach_cm": round(percentile_reach, 6),
        "full_reach_cm": round(full_reach, 6),
        "ground_cm": round(ground, 6),
        "robust_top_cm": round(robust_top, 6),
        "focus_z_cm": round(focus_z, 6),
        "ground_policy": "standing_bottom_p10",
        "focus_policy": "ground_plus_35pct_robust_vertical_span",
        "initial_dense_grid_center_cm": [
            round(initial_x, 6),
            round(initial_y, 6),
        ],
        "anchor_policy": "sane_standing_dense_region_xy_median",
        "reach_policy": "sane_standing_actor_reach_p90",
    }


def _camera_specs_sha256(views: Sequence[Viewpoint]) -> str:
    value = [
        {
            "name": view.name,
            "location_cm": [round(float(item), 6) for item in view.location],
            "rotation_deg": [round(float(item), 6) for item in view.rotation],
            "fov_deg": round(float(view.fov_deg), 6),
        }
        for view in views
    ]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def freeze_visibility_camera_plan(
    preferred: CameraPlan,
    resolved_views: Sequence[Viewpoint],
    *,
    planner_id: str,
    pairing_policy: str,
    planning_source: str,
    scene_environment: str,
    visibility_audit: Sequence[Mapping[str, Any]],
    visibility_policy: str = "ue-collision-navigation-target-los",
) -> CameraPlan:
    """Freeze one UE-validated absolute portfolio for both compared scenes."""

    if planner_id not in SUPPORTED_CAMERA_PLAN_POLICIES:
        raise ValueError(f"unsupported camera planner {planner_id!r}")
    if scene_environment not in {"indoor", "outdoor"}:
        raise ValueError(f"unsupported scene environment {scene_environment!r}")
    views = tuple(resolved_views)
    if len(views) != len(preferred.views) or not views:
        raise ValueError("resolved camera portfolio must preserve the view count")
    if len({value.name for value in views}) != len(views):
        raise ValueError("resolved camera view names must be unique")
    audit = {
        **dict(preferred.audit),
        "planner_id": planner_id,
        "pairing_policy": pairing_policy,
        "planning_source": planning_source,
        "scene_environment": scene_environment,
        "camera_adjustment_contract": GT_CAMERA_ADJUSTMENT_CONTRACT,
        "camera_adjustment_scope": "candidate_gt_pair",
        "single_scene_reframing_allowed": False,
        "visibility_policy": visibility_policy,
        "preferred_camera_specs_sha256": _camera_specs_sha256(preferred.views),
        "visibility_resolution": [dict(value) for value in visibility_audit],
        "camera_specs_sha256": _camera_specs_sha256(views),
    }
    return CameraPlan(views, preferred.anchor_cm, audit)


def plan_candidate_grounded_overview(
    actors: Sequence[Mapping[str, Any]],
    *,
    count: int,
    clearance_retry: bool = False,
) -> CameraPlan:
    """Plan a legible Candidate overview portfolio for semantic VLM evidence."""

    if count < 2:
        raise ValueError("a candidate overview portfolio needs at least two views")
    frame, framing_audit = _candidate_overview_frame(actors)
    near_radius = max(
        REFERENCE_MIN_RADIUS_CM,
        min(REFERENCE_MAX_NEAR_RADIUS_CM, frame.reach_cm * 1.05),
    )
    wide_radius = max(
        near_radius,
        min(REFERENCE_MAX_WIDE_RADIUS_CM, frame.reach_cm * 1.40),
    )
    highest_content_top = max(
        (box.z + box.ez for box in frame.boxes),
        default=frame.anchor_cm[2],
    )
    if clearance_retry:
        coverage_reach = float(
            framing_audit.get("percentile_reach_cm", frame.reach_cm)
        )
        retry_radius = max(
            REFERENCE_MIN_CLEARANCE_RADIUS_CM,
            min(
                REFERENCE_MAX_CLEARANCE_RADIUS_CM,
                max(
                    wide_radius
                    * REFERENCE_CLEARANCE_WIDE_RADIUS_MULTIPLIER,
                    coverage_reach * REFERENCE_CLEARANCE_RADIUS_MULTIPLIER,
                ),
            ),
        )
        vertical_clearance_margin = max(
            REFERENCE_VERTICAL_CLEARANCE_MARGIN_CM,
            retry_radius
            * REFERENCE_VERTICAL_CLEARANCE_RADIUS_MULTIPLIER,
        )
        retry_height = min(
            REFERENCE_MAX_CLEARANCE_HEIGHT_CM,
            max(
                retry_radius
                * REFERENCE_CLEARANCE_HEIGHT_RADIUS_MULTIPLIER,
                highest_content_top
                - frame.anchor_cm[2]
                + vertical_clearance_margin,
            ),
        )
        base_radii = tuple(retry_radius for _ in range(count))
        heights = tuple(retry_height for _ in range(count))
        angle_offset_deg = 22.5
        portfolio = "clearance-retry"
    else:
        base_radii = tuple(
            near_radius if index % 2 == 0 else wide_radius
            for index in range(count)
        )
        heights = None
        angle_offset_deg = 45.0
        portfolio = "geometry-clearance-primary"
        vertical_clearance_margin = None
    radii, geometry_clearance_audit = _geometry_clearance_radii(
        frame,
        base_radii_cm=base_radii,
        angle_offset_deg=angle_offset_deg,
    )
    if heights is None:
        # A camera can be horizontally outside a tower and still look through
        # its upper storeys toward the anchor.  Keep every primary overview
        # above the robust content top as well as beyond the radial envelope.
        heights = tuple(
            min(
                REFERENCE_MAX_CLEARANCE_HEIGHT_CM,
                max(
                    radius * HEIGHT_RADIUS_MULTIPLIER,
                    highest_content_top
                    - frame.anchor_cm[2]
                    + max(
                        REFERENCE_VERTICAL_CLEARANCE_MARGIN_CM,
                        radius
                        * REFERENCE_VERTICAL_CLEARANCE_RADIUS_MULTIPLIER,
                    ),
                ),
            )
            for radius in radii
        )
    else:
        # Preserve the clearance retry's minimum elevation, while keeping a
        # consistent aerial angle if radial geometry clearance moved a view.
        heights = tuple(
            max(height, radius * REFERENCE_CLEARANCE_HEIGHT_RADIUS_MULTIPLIER)
            for height, radius in zip(heights, radii, strict=True)
        )
    views = _aerial_portfolio(
        frame,
        radii_cm=radii,
        heights_cm=heights,
        angle_offset_deg=angle_offset_deg,
    )
    return CameraPlan(
        views=views,
        anchor_cm=frame.anchor_cm,
        audit={
            "planner_id": REFERENCE_CAMERA_PLAN,
            "pairing_policy": "candidate-only-grounded-overview",
            "planning_source": "candidate",
            "scene": frame.audit(),
            "framing": framing_audit,
            "portfolio": portfolio,
            "base_radii_cm": [round(value, 6) for value in base_radii],
            "radii_cm": [round(value, 6) for value in radii],
            "geometry_clearance": list(geometry_clearance_audit),
            "heights_cm": [
                round(float(view.location[2]) - frame.anchor_cm[2], 6)
                for view in views
            ],
            "highest_content_top_cm": round(highest_content_top, 6),
            "vertical_clearance_margin_cm": (
                round(vertical_clearance_margin, 6)
                if clearance_retry
                else REFERENCE_VERTICAL_CLEARANCE_MARGIN_CM
            ),
            "angle_offset_deg": angle_offset_deg,
            "view_count": count,
            "camera_specs_sha256": _camera_specs_sha256(views),
        },
    )


def plan_candidate_near_visibility_recovery(
    actors: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> CameraPlan:
    """Recover a fogged/occluded overview with the near-camera portfolio.

    The primary planner moves cameras beyond the projected content envelope. That is
    the right response to a roof or facade close-up, but it can be the wrong
    direction in distance fog or a large atmospheric shell: every farther
    camera becomes less informative. This bounded fallback keeps the same
    prompt-blind scene anchor and framing, but returns to the nearer/lower
    mixed-distance portfolio used by the bounded recovery path. Frame quality, never a semantic
    score, decides whether the recovery is usable.
    """

    if count < 2:
        raise ValueError(
            "a candidate overview portfolio needs at least two views"
        )
    frame, framing_audit = _candidate_overview_frame(actors)
    near_radius = max(
        REFERENCE_MIN_RADIUS_CM,
        min(REFERENCE_MAX_NEAR_RADIUS_CM, frame.reach_cm * 1.05),
    )
    wide_radius = max(
        near_radius,
        min(REFERENCE_MAX_WIDE_RADIUS_CM, frame.reach_cm * 1.40),
    )
    radii = tuple(
        near_radius if index % 2 == 0 else wide_radius
        for index in range(count)
    )
    views = _aerial_portfolio(
        frame,
        radii_cm=radii,
        angle_offset_deg=45.0,
    )
    return CameraPlan(
        views=views,
        anchor_cm=frame.anchor_cm,
        audit={
            "planner_id": REFERENCE_CAMERA_PLAN,
            "pairing_policy": "candidate-only-grounded-overview",
            "planning_source": "candidate",
            "scene": frame.audit(),
            "framing": framing_audit,
            "portfolio": "near-visibility-recovery",
            "recovery_reason": "fog_or_occlusion_quality_failure",
            "radii_cm": [round(value, 6) for value in radii],
            "heights_cm": [
                round(float(view.location[2]) - frame.anchor_cm[2], 6)
                for view in views
            ],
            "angle_offset_deg": 45.0,
            "view_count": count,
            "camera_specs_sha256": _camera_specs_sha256(views),
        },
    )

def plan_gt_frozen_dense_core_aerial(
    gt_actors: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> CameraPlan:
    """Plan legible whole-scene cameras once from GT and freeze them."""

    frame, framing_audit = _gt_dense_core_frame(gt_actors)
    radius = frame.reach_cm * RADIUS_HALF_EXTENT_MULTIPLIER
    views = _aerial_views(frame, count=count, radius_cm=radius)
    return CameraPlan(
        views=views,
        anchor_cm=frame.anchor_cm,
        audit={
            "planner_id": GT_DENSE_CORE_AERIAL_SEED_PLAN,
            "pairing_policy": "gt-dense-core-frozen-absolute-cameras",
            "planning_source": "gt",
            "camera_adjustment_contract": (
                GT_CAMERA_ADJUSTMENT_CONTRACT
            ),
            "camera_adjustment_scope": "candidate_gt_pair",
            "single_scene_reframing_allowed": False,
            "scene": frame.audit(),
            "framing": framing_audit,
            "radius_cm": radius,
            "angle_offset_deg": 45.0,
            "view_count": count,
            "camera_specs_sha256": _camera_specs_sha256(views),
        },
    )


def plan_gt_repair_target_aerial(
    target_actors: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> CameraPlan:
    """Frame the immutable GT-minus-Input repair region, not the whole scene."""

    boxes = tuple(
        value
        for actor in target_actors
        if (value := _content_box(actor)) is not None
    )
    if not boxes:
        raise ValueError("repair-target visual planning needs bounded GT Actors")
    minimum = (
        min(value.x - value.ex for value in boxes),
        min(value.y - value.ey for value in boxes),
        min(value.z - value.ez for value in boxes),
    )
    maximum = (
        max(value.x + value.ex for value in boxes),
        max(value.y + value.ey for value in boxes),
        max(value.z + value.ez for value in boxes),
    )
    anchor = tuple(
        round((left + right) / 2.0, 6)
        for left, right in zip(minimum, maximum, strict=True)
    )
    half_x = (maximum[0] - minimum[0]) / 2.0
    half_y = (maximum[1] - minimum[1]) / 2.0
    # A small floor avoids clipping a single object; the cap prevents an
    # erroneous outlier bound from silently turning this back into overview.
    reach = round(max(250.0, min(3_000.0, max(half_x, half_y) * 1.35)), 6)
    frame = SceneFrame(
        anchor_cm=anchor,
        reach_cm=reach,
        content_count=len(boxes),
        crop_min_cm=(minimum[0], minimum[1]),
        crop_max_cm=(maximum[0], maximum[1]),
        boxes=boxes,
    )
    radius = reach * RADIUS_HALF_EXTENT_MULTIPLIER
    views = _aerial_views(frame, count=count, radius_cm=radius)
    return CameraPlan(
        views=views,
        anchor_cm=frame.anchor_cm,
        audit={
            "planner_id": GT_REPAIR_TARGET_AERIAL_SEED_PLAN,
            "pairing_policy": "gt-repair-target-frozen-absolute-cameras",
            "planning_source": "gt_minus_input_target",
            "camera_adjustment_contract": (
                GT_CAMERA_ADJUSTMENT_CONTRACT
            ),
            "camera_adjustment_scope": "candidate_gt_pair",
            "single_scene_reframing_allowed": False,
            "target_actor_count": len(target_actors),
            "bounded_target_actor_count": len(boxes),
            "target_bounds_min_cm": list(minimum),
            "target_bounds_max_cm": list(maximum),
            "scene": frame.audit(),
            "radius_cm": radius,
            "angle_offset_deg": 45.0,
            "view_count": count,
            "camera_specs_sha256": _camera_specs_sha256(views),
        },
    )


def frame_quality(
    paths: Mapping[str, str | Path],
    measure,
    *,
    policy: str = CAPTION_FRAME_QUALITY,
) -> dict[str, Any]:
    """Audit a complete RGB group using the measured near-black luma threshold.

    ``mean_luma`` is injected to keep this module independent of the renderer
    implementation (and easy to test).  Unreadable images are acquisition
    failures, never silently treated as bright enough.
    """

    if policy not in SUPPORTED_FRAME_QUALITY_POLICIES:
        raise ValueError(f"unsupported frame quality policy {policy!r}")
    if policy == REFERENCE_FRAME_QUALITY:
        values: dict[str, dict[str, float] | None] = {}
        required = {
            "mean_luma",
            "p05_luma",
            "p50_luma",
            "p95_luma",
            "dark_fraction",
            "white_clip_fraction",
            "channel_clip_fraction",
            "detail_edge_fraction",
            "p99_spatial_gradient",
        }
        required.update({
            "flat_tile_fraction",
            "largest_flat_component_fraction",
            "central_flat_tile_fraction",
        })
        for view, path in sorted(paths.items()):
            measured = measure(Path(path))
            if not isinstance(measured, Mapping) or not required.issubset(measured):
                values[str(view)] = None
                continue
            values[str(view)] = {
                key: round(float(measured[key]), 6) for key in sorted(required)
            }
        unreadable = [view for view, value in values.items() if value is None]
        too_dark = [
            view
            for view, value in values.items()
            if value is not None
            and (
                value["p95_luma"] < AUTO_MIN_P95_LUMA
                or value["dark_fraction"] > AUTO_MAX_DARK_FRACTION
            )
        ]
        overexposed = [
            view
            for view, value in values.items()
            if value is not None
            and (
                value["white_clip_fraction"] > AUTO_MAX_WHITE_CLIP_FRACTION
                or value["channel_clip_fraction"]
                > AUTO_MAX_CHANNEL_CLIP_FRACTION
            )
        ]
        low_detail = [
            view
            for view, value in values.items()
            if value is not None
            and value["detail_edge_fraction"]
            < REFERENCE_MIN_DETAIL_EDGE_FRACTION
        ]
        occluded = [
            view
            for view, value in values.items()
            if value is not None
            and value["central_flat_tile_fraction"]
            >= REFERENCE_MIN_OCCLUDING_CENTER_FLAT_FRACTION
            and (
                value["largest_flat_component_fraction"]
                >= REFERENCE_MIN_OCCLUDING_FLAT_COMPONENT_FRACTION
                or value["dark_fraction"]
                >= REFERENCE_MIN_OCCLUDING_DARK_FRACTION
            )
        ]
        detailed_count = len(values) - len(unreadable) - len(low_detail)
        required_detailed_count = min(
            REFERENCE_MIN_DETAILED_VIEWS,
            len(values),
        )
        return {
            "policy": policy,
            "thresholds": {
                "min_p95_luma": AUTO_MIN_P95_LUMA,
                "max_dark_fraction": AUTO_MAX_DARK_FRACTION,
                "max_white_clip_fraction": AUTO_MAX_WHITE_CLIP_FRACTION,
                "max_channel_clip_fraction": AUTO_MAX_CHANNEL_CLIP_FRACTION,
                "min_detail_edge_fraction": (
                    REFERENCE_MIN_DETAIL_EDGE_FRACTION
                ),
                "min_detailed_views": required_detailed_count,
                "min_occluding_flat_component_fraction": (
                    REFERENCE_MIN_OCCLUDING_FLAT_COMPONENT_FRACTION
                ),
                "min_occluding_center_flat_fraction": (
                    REFERENCE_MIN_OCCLUDING_CENTER_FLAT_FRACTION
                ),
                "min_occluding_dark_fraction": (
                    REFERENCE_MIN_OCCLUDING_DARK_FRACTION
                ),
                "max_occluded_views": REFERENCE_MAX_OCCLUDED_VIEWS,
            },
            "image_quality": values,
            "unreadable_views": unreadable,
            "near_black_views": too_dark,
            "too_dark_views": too_dark,
            "overexposed_views": overexposed,
            "low_detail_views": low_detail,
            "occluded_views": occluded,
            "detailed_view_count": detailed_count,
            "view_count": len(values),
            "accepted": (
                bool(values)
                and not unreadable
                and not too_dark
                and not overexposed
                and detailed_count >= required_detailed_count
                and len(occluded) <= REFERENCE_MAX_OCCLUDED_VIEWS
            ),
        }

    if policy == GT_VISUAL_FRAME_QUALITY:
        values: dict[str, dict[str, float] | None] = {}
        for view, path in sorted(paths.items()):
            measured = measure(Path(path))
            if not isinstance(measured, Mapping):
                values[str(view)] = None
                continue
            required = {
                "mean_luma", "p05_luma", "p50_luma", "p95_luma",
                "dark_fraction", "white_clip_fraction",
                "channel_clip_fraction",
            }
            if not required.issubset(measured):
                values[str(view)] = None
                continue
            values[str(view)] = {
                key: round(float(measured[key]), 6) for key in sorted(required)
            }
        unreadable = [view for view, value in values.items() if value is None]
        too_dark = [
            view for view, value in values.items()
            if value is not None and (
                value["p95_luma"] < AUTO_MIN_P95_LUMA
                or value["dark_fraction"] > AUTO_MAX_DARK_FRACTION
            )
        ]
        overexposed = [
            view for view, value in values.items()
            if value is not None and (
                value["white_clip_fraction"] > AUTO_MAX_WHITE_CLIP_FRACTION
                or value["channel_clip_fraction"]
                > AUTO_MAX_CHANNEL_CLIP_FRACTION
            )
        ]
        return {
            "policy": policy,
            "thresholds": {
                "min_p95_luma": AUTO_MIN_P95_LUMA,
                "max_dark_fraction": AUTO_MAX_DARK_FRACTION,
                "max_white_clip_fraction": AUTO_MAX_WHITE_CLIP_FRACTION,
                "max_channel_clip_fraction": AUTO_MAX_CHANNEL_CLIP_FRACTION,
            },
            "image_quality": values,
            "unreadable_views": unreadable,
            "too_dark_views": too_dark,
            "overexposed_views": overexposed,
            "view_count": len(values),
            "accepted": (
                bool(values) and not unreadable and not too_dark
                and not overexposed
            ),
        }

    values: dict[str, float | None] = {}
    for view, path in sorted(paths.items()):
        measured = measure(Path(path))
        values[str(view)] = None if measured is None else round(float(measured), 6)
    unreadable = [view for view, value in values.items() if value is None]
    near_black = [
        view
        for view, value in values.items()
        if value is not None and value < NEAR_BLACK_LUMA
    ]
    return {
        "policy": policy,
        "threshold_mean_luma": NEAR_BLACK_LUMA,
        "mean_luma": values,
        "unreadable_views": unreadable,
        "near_black_views": near_black,
        "near_black_count": len(near_black),
        "view_count": len(values),
        "majority_near_black": len(near_black) > len(values) / 2.0,
        "accepted": bool(values) and not unreadable and not near_black,
    }


def apply_indoor_overview_supplemental_quality_override(
    quality: Mapping[str, Any],
    *,
    plan_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept one dark view only when it is the audited fallback supplement.

    Indoor modular rooms can yield exactly three distinct enclosed viewpoints
    with positive line of sight to shared unedited geometry.  The fourth view
    remains collision-free, enclosed, unique, and subject to RGB health, but
    may depict a naturally dark wall.  This exception is portfolio-level and
    cannot turn an arbitrary dark gallery into valid evidence.
    """

    result = dict(quality)
    if (
        result.get("accepted") is True
        or result.get("unreadable_views")
        or plan_audit.get("indoor_camera_fallback_stage")
        != INDOOR_OVERVIEW_CAMERA_FALLBACK_STAGE
    ):
        return result
    resolutions = plan_audit.get("visibility_resolution")
    if not isinstance(resolutions, Sequence) or isinstance(
        resolutions, (str, bytes)
    ):
        return result
    resolution_values = [
        value for value in resolutions if isinstance(value, Mapping)
    ]
    if len(resolution_values) != 4:
        return result
    no_los_indices = {
        index
        for index, value in enumerate(resolution_values)
        if int(value.get("visible_actor_count") or 0) == 0
        and value.get("reason") == "overview_enclosed_rgb_health_fallback"
        and value.get("camera_initial_overlap") is not True
        and value.get("camera_enclosure_ok") is not False
    }
    near_black = result.get("near_black_views")
    if not isinstance(near_black, Sequence) or isinstance(near_black, (str, bytes)):
        return result
    dark_indices = set()
    for value in near_black:
        name = str(value)
        prefix = "view_"
        if not name.startswith(prefix) or not name[len(prefix) :].isdigit():
            return result
        dark_indices.add(int(name[len(prefix) :]))
    visible_view_count = sum(
        int(value.get("visible_actor_count") or 0) > 0
        for value in resolution_values
    )
    if (
        visible_view_count < 3
        or no_los_indices != dark_indices
        or len(dark_indices) != 1
    ):
        return result
    result.update(
        {
            "accepted": True,
            "accepted_by": (
                "indoor-overview-three-los-one-enclosed-supplement"
            ),
            "shared_anchor_visible_view_count": visible_view_count,
            "enclosed_supplemental_view": f"view_{next(iter(dark_indices))}",
        }
    )
    return result


def apply_candidate_near_visibility_quality_override(
    quality: Mapping[str, Any],
    *,
    plan_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept judgeable near evidence misclassified as flat or low-detail.

    A narrow tower over a legitimate ground plane can leave most coarse image
    tiles flat even when the structure is sharp and fully judgeable.  The current
    flat-component gate is still authoritative for every other portfolio.
    Architectural scenes with broad, low-texture facades can also fall just
    below the global edge-density threshold even though several near views
    contain strong structural gradients.  A near-visibility recovery may
    bypass either false-positive mode only after all exposure and acquisition
    checks pass; the relaxed path still requires three independently
    structured views.
    """

    result = dict(quality)
    if (
        result.get("accepted") is True
        or result.get("policy") != REFERENCE_FRAME_QUALITY
        or plan_audit.get("portfolio") != "near-visibility-recovery"
    ):
        return result
    if any(
        result.get(key)
        for key in (
            "unreadable_views",
            "near_black_views",
            "too_dark_views",
            "overexposed_views",
        )
    ):
        return result
    image_quality = result.get("image_quality")
    if isinstance(image_quality, Mapping):
        structured_views = []
        for view, raw_metrics in image_quality.items():
            if not isinstance(raw_metrics, Mapping):
                continue
            detail = _number(raw_metrics.get("detail_edge_fraction"))
            gradient = _number(raw_metrics.get("p99_spatial_gradient"))
            flat = _number(
                raw_metrics.get("largest_flat_component_fraction")
            )
            central_flat = _number(
                raw_metrics.get("central_flat_tile_fraction")
            )
            if (
                detail is not None
                and detail >= REFERENCE_RECOVERY_MIN_DETAIL_EDGE_FRACTION
                and gradient is not None
                and gradient >= REFERENCE_RECOVERY_MIN_P99_SPATIAL_GRADIENT
                and flat is not None
                and flat <= REFERENCE_RECOVERY_MAX_FLAT_COMPONENT_FRACTION
                and central_flat is not None
                and central_flat <= REFERENCE_RECOVERY_MAX_CENTRAL_FLAT_FRACTION
            ):
                structured_views.append(str(view))
        required_structured = min(
            REFERENCE_RECOVERY_MIN_STRUCTURED_VIEWS,
            len(image_quality),
        )
        if (
            required_structured >= 2
            and len(structured_views) >= required_structured
        ):
            result["accepted"] = True
            result["acceptance_override"] = {
                "policy": "structured-near-visibility",
                "reason": (
                    "near_gallery_has_multi_view_structural_gradients"
                ),
                "structured_views": sorted(structured_views),
                "minimum_structured_views": required_structured,
                "min_detail_edge_fraction": (
                    REFERENCE_RECOVERY_MIN_DETAIL_EDGE_FRACTION
                ),
                "min_p99_spatial_gradient": (
                    REFERENCE_RECOVERY_MIN_P99_SPATIAL_GRADIENT
                ),
                "max_flat_component_fraction": (
                    REFERENCE_RECOVERY_MAX_FLAT_COMPONENT_FRACTION
                ),
                "max_central_flat_fraction": (
                    REFERENCE_RECOVERY_MAX_CENTRAL_FLAT_FRACTION
                ),
                "preserved_low_detail_views": list(
                    result.get("low_detail_views") or []
                ),
                "preserved_occluded_views": list(
                    result.get("occluded_views") or []
                ),
            }
            return result

    framing = plan_audit.get("framing")
    if not isinstance(framing, Mapping):
        return result
    ground = _number(framing.get("ground_cm"))
    robust_top = _number(framing.get("robust_top_cm"))
    horizontal_reach = _number(framing.get("percentile_reach_cm"))
    if (
        ground is None
        or robust_top is None
        or horizontal_reach is None
        or horizontal_reach <= 0.0
    ):
        return result
    vertical_span = max(0.0, robust_top - ground)
    aspect_ratio = vertical_span / horizontal_reach
    if aspect_ratio < REFERENCE_SPARSE_VERTICAL_MIN_ASPECT_RATIO:
        return result
    if result.get("low_detail_views"):
        return result
    view_count = result.get("view_count")
    detailed_count = result.get("detailed_view_count")
    occluded = result.get("occluded_views")
    thresholds = result.get("thresholds")
    max_occluded = (
        thresholds.get("max_occluded_views")
        if isinstance(thresholds, Mapping)
        else None
    )
    if (
        isinstance(view_count, bool)
        or not isinstance(view_count, int)
        or view_count < 2
        or detailed_count != view_count
        or not isinstance(occluded, list)
        or isinstance(max_occluded, bool)
        or not isinstance(max_occluded, int)
        or len(occluded) <= max_occluded
    ):
        return result
    result["accepted"] = True
    result["acceptance_override"] = {
        "policy": "sparse-vertical-near-visibility",
        "reason": "bright_flat_background_with_detailed_vertical_subject",
        "vertical_span_cm": round(vertical_span, 6),
        "horizontal_percentile_reach_cm": round(horizontal_reach, 6),
        "vertical_to_horizontal_aspect_ratio": round(aspect_ratio, 6),
        "minimum_aspect_ratio": (
            REFERENCE_SPARSE_VERTICAL_MIN_ASPECT_RATIO
        ),
        "preserved_occluded_views": list(occluded),
    }
    return result


__all__ = [
    "CAPTION_CAMERA_PLAN",
    "CAPTION_FRAME_QUALITY",
    "CameraPlan",
    "GT_VISUAL_CAMERA_PLAN",
    "GT_REPAIR_TARGET_OUTDOOR_CAMERA_PLAN",
    "GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN",
    "GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN",
    "GT_REPAIR_TARGET_CASE_CAMERA_VIEW_COUNT",
    "GT_VISUAL_FRAME_QUALITY",
    "NEAR_BLACK_LUMA",
    "REFERENCE_CAMERA_PLAN",
    "REFERENCE_FRAME_QUALITY",
    "SUPPORTED_CAMERA_PLAN_POLICIES",
    "SUPPORTED_FRAME_QUALITY_POLICIES",
    "apply_candidate_near_visibility_quality_override",
    "frame_quality",
    "freeze_visibility_camera_plan",
    "plan_candidate_grounded_overview",
    "plan_candidate_near_visibility_recovery",
    "plan_gt_frozen_dense_core_aerial",
    "plan_gt_repair_target_aerial",
    "scene_frame",
]
