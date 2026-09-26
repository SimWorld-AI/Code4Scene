"""Frozen-world bounds pass: clamp oversized flat sheets, delete escapees.

Python port of ``boundsPass`` from PR #24's ``run_episode.js``. Deterministic
enforcement of a rectangular XY boundary: actors fully outside are deleted;
large flat sheets (ground/water) overshooting an edge are re-centred and
squashed back to the rectangle. Everything touched is counted and returned —
the agent is measured, never silently corrected.

``bounds_v2`` takes that rectangle from the canonical 3D GT content AABB when
one exists. Candidate geometry never defines its own evaluation region. An
origin-centred ``size_m`` plate remains only as a compatibility fallback for
tasks without 3D GT. The constants and label-prefix exemption list are frozen
scoring behaviour; they differ from measure_v1's infra list on purpose.

Known limitations carried over verbatim, all bounds_v2 candidates:
  - exemption is by label prefix, so an adversarial agent can escape the pass
    by naming an oversized sheet e.g. "SkyOcean"; class-based exemption is the
    fix, but it changes which actors are counted;
  - the clamp rescales an actor about its pivot and then moves it to the
    recomputed *bounds* midpoint, which only lands correctly when pivot and
    bounds centre coincide — off-centre pivots end up displaced.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from code4scene.core.bridge import Bridge, BridgeError
from code4scene.core.inventory import SCENERY_PREFIXES

METRIC_ID = "bounds_v2"
MARK = "BOUNDS_JSON"
BOUNDARY_SCHEMA_VERSION = "scenebenchmark-world-bounds.v1"

# Label prefixes exempt from the pass (raw case, startswith). Defined in
# `core.inventory` because the task generator measures the plate over the same
# actors this pass enforces it against, and a second copy is a second opinion.
EXEMPT_PREFIXES = SCENERY_PREFIXES
FLAT_EZ_CM = 25.0        # half-height below which a sheet counts as flat
FLAT_EXTENT_CM = 1000.0  # ...combined with a half-extent above this
OVERSHOOT_SLACK_CM = 50.0
MIN_SCALE = 0.05


class BoundsError(Exception):
    """The pass did not run or did not report (bridge/editor failure)."""


def boundary(
    *,
    center_cm: Sequence[float],
    extent_cm: Sequence[float],
    source: str,
    source_map: str | None = None,
) -> dict[str, Any]:
    """Build one validated, JSON-native XY evaluation boundary."""

    if len(center_cm) < 2 or len(extent_cm) < 2:
        raise ValueError("world bounds need two-dimensional center and extent")
    center = [float(center_cm[0]), float(center_cm[1])]
    extent = [float(extent_cm[0]), float(extent_cm[1])]
    if not all(math.isfinite(value) for value in (*center, *extent)):
        raise ValueError("world bounds center and extent must be finite")
    if any(value <= 0.0 for value in extent):
        raise ValueError("world bounds extent must be positive")
    return {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "source": str(source),
        **({"source_map": str(source_map)} if source_map else {}),
        "center_cm": center,
        "extent_cm": extent,
        "minimum_cm": [center[i] - extent[i] for i in range(2)],
        "maximum_cm": [center[i] + extent[i] for i in range(2)],
    }


def from_half_extent_m(half_extent_m: float) -> dict[str, Any]:
    """Represent the legacy origin-centred task plate in the new contract."""

    extent = float(half_extent_m) * 100.0
    return boundary(
        center_cm=(0.0, 0.0),
        extent_cm=(extent, extent),
        source="task_size_m",
    )


def from_inventory(
    actors: Sequence[Mapping[str, Any]],
    *,
    source_map: str,
) -> dict[str, Any]:
    """Measure the exact XY content AABB of a loaded canonical GT scene.

    The population matches the bounds pass: label-prefix scenery exemptions
    are excluded, while every remaining Actor contributes its full world AABB.
    Candidate geometry is never consulted, so an outlier cannot enlarge its
    own evaluation region.
    """

    minimum = [math.inf, math.inf]
    maximum = [-math.inf, -math.inf]
    measured = 0
    excluded = 0
    for actor in actors:
        label = str(actor.get("label") or "")
        if label.startswith(EXEMPT_PREFIXES):
            excluded += 1
            continue
        raw_center = actor.get("loc")
        raw_extent = actor.get("extent")
        if (
            not isinstance(raw_center, Sequence)
            or isinstance(raw_center, (str, bytes))
            or len(raw_center) < 2
            or not isinstance(raw_extent, Sequence)
            or isinstance(raw_extent, (str, bytes))
            or len(raw_extent) < 2
        ):
            continue
        try:
            center = [float(raw_center[0]), float(raw_center[1])]
            extent = [abs(float(raw_extent[0])), abs(float(raw_extent[1]))]
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in (*center, *extent)):
            continue
        for axis in range(2):
            minimum[axis] = min(minimum[axis], center[axis] - extent[axis])
            maximum[axis] = max(maximum[axis], center[axis] + extent[axis])
        measured += 1
    if measured == 0:
        raise BoundsError(
            f"canonical GT {source_map} has no measurable non-scenery Actors"
        )
    center = [(minimum[i] + maximum[i]) / 2.0 for i in range(2)]
    extent = [max((maximum[i] - minimum[i]) / 2.0, 1.0) for i in range(2)]
    result = boundary(
        center_cm=center,
        extent_cm=extent,
        source="canonical_gt_scene",
        source_map=source_map,
    )
    result["measured_actor_count"] = measured
    result["excluded_scenery_actor_count"] = excluded
    return result


def _validated_boundary(value: Mapping[str, Any]) -> dict[str, Any]:
    return boundary(
        center_cm=value.get("center_cm") or (),
        extent_cm=value.get("extent_cm") or (),
        source=str(value.get("source") or "unknown"),
        source_map=(str(value["source_map"]) if value.get("source_map") else None),
    ) | {
        key: value[key]
        for key in ("measured_actor_count", "excluded_scenery_actor_count")
        if key in value
    }


def payload_script(half_m: float) -> str:
    """Legacy origin-centred wrapper retained for agent-time task plates."""

    return boundary_payload_script(
        from_half_extent_m(half_m), legacy_half_m=half_m
    )


def boundary_payload_script(
    evaluation_bounds: Mapping[str, Any],
    *,
    legacy_half_m: float | None = None,
) -> str:
    """Editor Python for a GT-centred or legacy bounds pass."""

    evaluation_bounds = _validated_boundary(evaluation_bounds)
    minimum = evaluation_bounds["minimum_cm"]
    maximum = evaluation_bounds["maximum_cm"]
    # math.floor(x + 0.5), not round(): JavaScript's Math.round is
    # half-away-from-zero while python's round() is half-to-even, so a plate
    # landing on a half-centimetre came out 1 cm different from bounds_v1.
    prefix = []
    if legacy_half_m is not None:
        h = math.floor(float(legacy_half_m) * 100 + 0.5)
        prefix.append(f"H={h}.0")
    return "\n".join([
        "import unreal, json",
        *prefix,
        f"MINX={minimum[0]!r}; MAXX={maximum[0]!r}",
        f"MINY={minimum[1]!r}; MAXY={maximum[1]!r}",
        "clamped=0; deleted=0; details=[]",
        "for a in list(unreal.EditorLevelLibrary.get_all_level_actors()):",
        "    try:",
        "        lbl=a.get_actor_label()",
        f"        if lbl.startswith({EXEMPT_PREFIXES!r}): continue",
        "        o,e=a.get_actor_bounds(False)",
        "        minx,maxx=o.x-e.x,o.x+e.x; miny,maxy=o.y-e.y,o.y+e.y",
        "        if minx>MAXX or maxx<MINX or miny>MAXY or maxy<MINY:",
        "            details.append(('deleted',lbl)); unreal.EditorLevelLibrary.destroy_actor(a); deleted+=1; continue",
        f"        flat = e.z<{FLAT_EZ_CM!r} and (e.x>{FLAT_EXTENT_CM!r} or e.y>{FLAT_EXTENT_CM!r})",
        f"        over = maxx>MAXX+{OVERSHOOT_SLACK_CM:g} or minx<MINX-{OVERSHOOT_SLACK_CM:g} or maxy>MAXY+{OVERSHOOT_SLACK_CM:g} or miny<MINY-{OVERSHOOT_SLACK_CM:g}",
        "        if flat and over:",
        "            nx=(max(minx,MINX)+min(maxx,MAXX))/2.0; ny=(max(miny,MINY)+min(maxy,MAXY))/2.0",
        f"            sx=max({MIN_SCALE!r},(min(maxx,MAXX)-max(minx,MINX))/(2*e.x)) if e.x>1.0 else 1.0",
        f"            sy=max({MIN_SCALE!r},(min(maxy,MAXY)-max(miny,MINY))/(2*e.y)) if e.y>1.0 else 1.0",
        "            s=a.get_actor_scale3d(); loc=a.get_actor_location()",
        "            a.set_actor_scale3d(unreal.Vector(s.x*sx, s.y*sy, s.z))",
        "            a.set_actor_location(unreal.Vector(nx, ny, loc.z), False, False)",
        "            details.append(('clamped',lbl)); clamped+=1",
        "    except Exception: pass",
        "checked=0; remaining_oob=[]",
        "for a in list(unreal.EditorLevelLibrary.get_all_level_actors()):",
        "    try:",
        "        lbl=a.get_actor_label()",
        f"        if lbl.startswith({EXEMPT_PREFIXES!r}): continue",
        "        loc=a.get_actor_location(); checked+=1",
        "        if loc.x<MINX or loc.x>MAXX or loc.y<MINY or loc.y>MAXY: remaining_oob.append(lbl)",
        "    except Exception: pass",
        f"_boundary={evaluation_bounds!r}",
        "_report={'clamped':clamped,'deleted':deleted,'sample':[d[1] for d in details[:10]],",
        "         'checked':checked,'remaining_pivot_oob_count':len(remaining_oob),",
        "         'remaining_pivot_oob_sample':remaining_oob[:10],'boundary':_boundary}",
        f"print('{MARK} '+json.dumps(_report))",
        # Stored as well as printed: a marker printed at the end of a
        # pass over every actor falls outside the log window the call
        # returns (see Bridge.exec_python_result).
        "globals()['_SB_BOUNDS'] = _report",
    ])


def run(
    bridge: Bridge,
    half_m: float | None = None,
    timeout: float = 180.0,
    *,
    evaluation_bounds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one bounds pass; return ``{'clamped', 'deleted', 'sample'}``."""
    if evaluation_bounds is None:
        if half_m is None:
            raise BoundsError("a bounds pass needs GT bounds or a task half-extent")
        script = payload_script(half_m)
    else:
        script = boundary_payload_script(evaluation_bounds)
    try:
        bridge.exec_python("globals().pop('_SB_BOUNDS', None)", timeout=timeout)
        return bridge.exec_python_result(script, "_SB_BOUNDS", timeout=timeout)
    except BridgeError as e:
        raise BoundsError(f"bridge failure during bounds pass: {e}") from e


__all__ = [
    "BOUNDARY_SCHEMA_VERSION",
    "BoundsError",
    "METRIC_ID",
    "boundary",
    "boundary_payload_script",
    "from_half_extent_m",
    "from_inventory",
    "payload_script",
    "run",
]
