"""Where an Actor is, how big it is, and how two of them sit together.

Every spatial metric asks the same handful of questions of a scene graph —
what is this Actor's world AABB, do two boxes overlap, how far apart are their
surfaces, does one contain the other — and each one that answers them itself
answers them slightly differently. Four separate answers to "is this Actor
floating" is the mistake this repository already made once; this module exists
so the spatial half never repeats it.

The conventions are the exporter's, not a preference:

* ``bounds.origin_cm`` is the world-space CENTRE of the AABB and
  ``bounds.extent_cm`` is its half-size, so the box is origin ± extent. An
  Actor without bounds falls back to its transform location and a zero extent,
  which is a point — the honest reading of "we do not know how big it is".
* Z is up, so ``above`` and ``below`` are axis 2 and the footprint is XY.
* Distances are centimetres and angles are degrees, because that is what the
  editor reports; nothing here converts silently.

Rounding is half-up rather than Python's banker's rounding. It looks like a
cosmetic choice and is not: the relation checks compare the ROUNDED gap
against the declared margin, so a value landing exactly on ``.5`` at the last
kept digit decides a pass, and the JavaScript this was ported from rounds
half-up. Two implementations that disagree on that are two benchmarks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

#: Axis overlap beyond which two AABBs count as intersecting rather than
#: touching, in centimetres. `measure_v1` uses the same 5 cm, and the two must
#: agree: a scene-wide collision rate and a declared no-overlap assertion that
#: disagreed about what "touching" means would contradict each other in one
#: record.
TOUCH_TOLERANCE_CM = 5.0


def rounded(value: Any, digits: int = 6) -> float | None:
    """Half-up rounding, or None for anything that is not a finite number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    quantum = Decimal(1).scaleb(-digits)
    return float(Decimal(repr(number)).quantize(quantum, rounding=ROUND_HALF_UP))


def _triplet(value: Any, default: float = 0.0) -> list[float]:
    """Three finite floats out of whatever the exporter wrote."""
    items = list(value) if isinstance(value, (list, tuple)) else []
    out = []
    for index in range(3):
        try:
            number = float(items[index]) if index < len(items) else default
        except (TypeError, ValueError):
            number = default
        out.append(number if math.isfinite(number) else default)
    return out


def center_cm(actor: Mapping[str, Any]) -> list[float]:
    """The AABB centre, or the pivot when the Actor carries no bounds."""
    bounds = actor.get("bounds")
    if isinstance(bounds, Mapping) and bounds.get("origin_cm") is not None:
        return _triplet(bounds.get("origin_cm"))
    transform = actor.get("transform")
    location = transform.get("location_cm") if isinstance(transform, Mapping) else None
    return _triplet(location)


def extent_cm(actor: Mapping[str, Any]) -> list[float]:
    """The AABB half-size, or a point when the Actor carries no bounds.

    A POINT IS NOT A MEASUREMENT. Nothing intersects a point, nothing
    overlaps a doorway with one, and every geometric requirement over a scene
    of points passes. The exporter writes exactly this shape when its bounds
    read fails (`export_scene_snapshot.py`: origin = location, extent = 0, and
    an entry in `export_errors`), so it is a live state and not a
    hypothetical.

    So this stays permissive and `has_bounds` is what a metric asks first.
    Every verifier whose number depends on size WITHHOLDS when the population
    does not carry it, rather than scoring a scene of dimensionless dots.
    """
    bounds = actor.get("bounds")
    if isinstance(bounds, Mapping) and bounds.get("extent_cm") is not None:
        return _triplet(bounds.get("extent_cm"))
    return [0.0, 0.0, 0.0]


def has_bounds(actor: Mapping[str, Any]) -> bool:
    """Did the export actually measure this Actor's size.

    A zero extent counts as absent. It is indistinguishable from the
    exporter's failure fallback, and an Actor genuinely of zero size cannot
    intersect anything either — so treating the two alike costs nothing and
    closes the hole.
    """
    bounds = actor.get("bounds")
    if not isinstance(bounds, Mapping) or bounds.get("extent_cm") is None:
        return False
    return any(abs(value) > 0.0 for value in _triplet(bounds.get("extent_cm")))


def bounds_coverage(actors: Sequence[Mapping[str, Any]]) -> tuple[float, list[str]]:
    """What share of a population was measured, and which were not."""
    missing = [str(actor.get("label") or actor_key(actor) or "?")
               for actor in actors if not has_bounds(actor)]
    total = len(actors)
    return ((total - len(missing)) / total if total else 1.0), missing


def location_cm(actor: Mapping[str, Any]) -> list[float]:
    """The pivot, which is NOT the AABB centre — `out_of_bounds` uses this."""
    transform = actor.get("transform")
    return _triplet(transform.get("location_cm") if isinstance(transform, Mapping) else None)


def rotation_deg(actor: Mapping[str, Any]) -> list[float]:
    transform = actor.get("transform")
    return _triplet(transform.get("rotation_deg") if isinstance(transform, Mapping) else None)


def scale(actor: Mapping[str, Any]) -> list[float]:
    transform = actor.get("transform")
    return _triplet(transform.get("scale") if isinstance(transform, Mapping) else None, 1.0)


def aabb(actor: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """(minimum, maximum) in world space. Negative extents are read as sizes."""
    center, extent = center_cm(actor), extent_cm(actor)
    return ([center[i] - abs(extent[i]) for i in range(3)],
            [center[i] + abs(extent[i]) for i in range(3)])


def volume_cm3(actor: Mapping[str, Any]) -> float:
    extent = extent_cm(actor)
    return math.prod(max(0.0, extent[i] * 2) for i in range(3))


def overlap_on_axis(left_origin: float, left_extent: float,
                    right_origin: float, right_extent: float) -> float:
    """Signed overlap of two intervals given as centre ± half-size.

    Negative means a gap. `solid_penetration` and `spatial_overlap` both
    needed this and had two spellings of it; one of them wrote the operands in
    min/max form and the pair drifted.
    """
    return (min(left_origin + left_extent, right_origin + right_extent)
            - max(left_origin - left_extent, right_origin - right_extent))


def actor_key(actor: Mapping[str, Any]) -> str | None:
    """Whatever identifies this Actor to another metric, best first."""
    for field in ("stable_actor_id", "actor_path", "label"):
        value = actor.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def axis_overlaps_cm(left: Mapping[str, Any],
                     right: Mapping[str, Any]) -> list[float]:
    left_center, left_extent = center_cm(left), extent_cm(left)
    right_center, right_extent = center_cm(right), extent_cm(right)
    return [overlap_on_axis(left_center[i], left_extent[i],
                            right_center[i], right_extent[i]) for i in range(3)]


def aabb_overlaps(actors: Sequence[Mapping[str, Any]],
                  touch_tolerance_cm: float = TOUCH_TOLERANCE_CM,
                  ) -> list[dict[str, Any]]:
    """Every pair whose boxes intersect on ALL THREE axes beyond tolerance.

    All three, because two boxes that share a face overlap on two axes and
    touch on the third — a table against a wall, a poster on it — and calling
    that a collision would fail every scene that was assembled correctly.
    """
    found = []
    for i in range(len(actors)):
        for j in range(i + 1, len(actors)):
            overlaps = axis_overlaps_cm(actors[i], actors[j])
            if all(value > touch_tolerance_cm for value in overlaps):
                found.append({
                    "actor1": actors[i].get("label"),
                    "actor2": actors[j].get("label"),
                    "actor1_id": actor_key(actors[i]),
                    "actor2_id": actor_key(actors[j]),
                    "overlap_cm": [rounded(value, 3) for value in overlaps],
                    "penetration_depth_cm": rounded(
                        min(overlaps) - touch_tolerance_cm, 3),
                })
    return found


def aabb_overlaps_between(subjects: Sequence[Mapping[str, Any]],
                          others: Sequence[Mapping[str, Any]],
                          touch_tolerance_cm: float = TOUCH_TOLERANCE_CM,
                          ) -> list[dict[str, Any]]:
    """The same question asked of a population against the whole scene.

    An Actor is never paired with itself, and each unordered pair is reported
    once even when both sides are selected — otherwise a subject that is also
    in ``others`` doubles its own penetration count.
    """
    if subjects is others:
        return aabb_overlaps(subjects, touch_tolerance_cm)
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for left in subjects:
        for right in others:
            if left is right:
                continue
            left_key, right_key = actor_key(left), actor_key(right)
            if left_key is not None and left_key == right_key:
                continue
            # A keyless Actor still needs a distinct identity here: stringifying
            # None made every unidentified pair the SAME pair, so the first one
            # seen suppressed all the rest — overlapping actors with no ids
            # were deduplicated out of the report. Object identity is stable
            # for the duration of this scan and dedupes the same actor
            # appearing in both populations, which is all `seen` is for.
            pair = tuple(sorted((
                left_key if left_key is not None else f"@{id(left)}",
                right_key if right_key is not None else f"@{id(right)}",
            )))
            if pair in seen:
                continue
            seen.add(pair)
            overlaps = axis_overlaps_cm(left, right)
            if all(value > touch_tolerance_cm for value in overlaps):
                found.append({
                    "actor1": left.get("label"), "actor2": right.get("label"),
                    "actor1_id": left_key, "actor2_id": right_key,
                    "overlap_cm": [rounded(value, 3) for value in overlaps],
                    "penetration_depth_cm": rounded(
                        min(overlaps) - touch_tolerance_cm, 3),
                })
    return found


def center_distance_cm(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a, b = center_cm(left), center_cm(right)
    return math.dist(a, b)


def cyclic_degrees(left: float, right: float) -> float:
    """How far apart two angles are. 359 and 1 are two degrees apart."""
    delta = abs(float(left) - float(right)) % 360
    return min(delta, 360 - delta)


def contains(outer: Mapping[str, Any], inner: Mapping[str, Any],
             tolerance_cm: float = 0.0) -> bool:
    outer_min, outer_max = aabb(outer)
    inner_min, inner_max = aabb(inner)
    return all(inner_min[i] >= outer_min[i] - tolerance_cm
               and inner_max[i] <= outer_max[i] + tolerance_cm for i in range(3))


def xy_overlap_fraction(subject: Mapping[str, Any],
                        object_actor: Mapping[str, Any]) -> float:
    """How much of the SUBJECT's footprint sits over the object's, 0..1."""
    subject_min, subject_max = aabb(subject)
    other_min, other_max = aabb(object_actor)
    overlap_x = max(0.0, min(subject_max[0], other_max[0])
                    - max(subject_min[0], other_min[0]))
    overlap_y = max(0.0, min(subject_max[1], other_max[1])
                    - max(subject_min[1], other_min[1]))
    extent = extent_cm(subject)
    area = max(1e-9, abs(extent[0]) * 2 * abs(extent[1]) * 2)
    return max(0.0, min(1.0, overlap_x * overlap_y / area))


def pair_geometry(subject: Mapping[str, Any],
                  object_actor: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a relation rule might ask about one ordered pair.

    Computed once per pair even when the rule reads one field: the fields are
    cheap, and a relation that recomputed only what it needed is a relation
    whose evidence cannot be compared with its neighbour's.
    """
    subject_min, subject_max = aabb(subject)
    object_min, object_max = aabb(object_actor)
    gaps = [max(0.0, subject_min[i] - object_max[i], object_min[i] - subject_max[i])
            for i in range(3)]
    overlaps = [max(0.0, min(subject_max[i], object_max[i])
                    - max(subject_min[i], object_min[i])) for i in range(3)]
    lengths = [max(0.0, subject_max[i] - subject_min[i]) for i in range(3)]
    subject_volume = math.prod(lengths)
    intersection_volume = math.prod(overlaps)
    subject_area = max(1e-9, lengths[0] * lengths[1])
    return {
        "center_distance_cm": rounded(center_distance_cm(subject, object_actor)),
        "closest_surface_distance_cm": rounded(math.hypot(*gaps)),
        "axis_surface_gaps_cm": [rounded(value) for value in gaps],
        "axis_overlap_cm": [rounded(value) for value in overlaps],
        "intersecting_axis_count": sum(value > 0 for value in overlaps),
        "intersection_volume_cm3": rounded(intersection_volume),
        "overlap_volume_fraction_of_subject": rounded(
            intersection_volume / subject_volume if subject_volume > 0 else 0.0),
        "xy_overlap_fraction_of_subject": rounded(
            overlaps[0] * overlaps[1] / subject_area),
    }


def distance_to_polyline(x: float, y: float,
                         points: Sequence[Sequence[float]]) -> float:
    """Shortest distance from a point to a polyline, in the points' units."""
    best = math.inf
    for index in range(len(points) - 1):
        x1, y1 = float(points[index][0]), float(points[index][1])
        x2, y2 = float(points[index + 1][0]), float(points[index + 1][1])
        vx, vy = x2 - x1, y2 - y1
        length_squared = vx * vx + vy * vy
        raw = (((x - x1) * vx + (y - y1) * vy) / length_squared
               if length_squared > 0 else 0.0)
        t = max(0.0, min(1.0, raw))
        best = min(best, math.hypot(x - (x1 + t * vx), y - (y1 + t * vy)))
    return best


def placement(actor: Mapping[str, Any]) -> dict[str, Any]:
    """The Actor as a footprint in METRES, which is what routes are given in."""
    center, extent = center_cm(actor), extent_cm(actor)
    return {"x_m": center[0] / 100, "y_m": center[1] / 100,
            "radius_m": math.hypot(extent[0], extent[1]) / 100}


def intersects_rectangle(actor: Mapping[str, Any],
                         rectangle: Mapping[str, Any]) -> bool:
    """Does this Actor's footprint reach into a protected rectangle at all."""
    low = rectangle.get("min_xy_cm") or []
    high = rectangle.get("max_xy_cm") or []
    if len(low) < 2 or len(high) < 2:
        raise ValueError(
            f"protected rectangle {rectangle.get('id')!r} needs min_xy_cm and "
            f"max_xy_cm")
    minimum, maximum = aabb(actor)
    return (min(maximum[0], float(high[0])) - max(minimum[0], float(low[0])) > 0
            and min(maximum[1], float(high[1])) - max(minimum[1], float(low[1])) > 0)


def clearance_violations(actors: Sequence[Mapping[str, Any]],
                         rectangles: Sequence[Mapping[str, Any]] = (),
                         routes: Sequence[Mapping[str, Any]] = (),
                         ) -> list[dict[str, Any]]:
    """Every (Actor, protected region) pair the Actor intrudes on.

    Shared by the clearance verifier and the regression one, which must agree:
    a regression is "a violation the candidate introduced", and two different
    definitions of violation make that difference meaningless.
    """
    found: list[dict[str, Any]] = []
    for actor in actors:
        for rectangle in rectangles:
            if intersects_rectangle(actor, rectangle):
                found.append({"actor": actor.get("label"),
                              "actor_id": actor_key(actor),
                              "region": rectangle.get("id"),
                              "method": "aabb_rectangle"})
        spot = placement(actor)
        for route in routes:
            points = route.get("points") or []
            if len(points) < 2:
                continue
            required = float(route.get("width_m") or 0.0) / 2 + spot["radius_m"]
            distance = distance_to_polyline(spot["x_m"], spot["y_m"], points)
            if distance < required:
                found.append({"actor": actor.get("label"),
                              "actor_id": actor_key(actor),
                              "region": route.get("id"),
                              "method": "route_corridor",
                              "distance_m": rounded(distance, 3),
                              "required_clearance_m": rounded(required, 3)})
                break
    return found


__all__ = ["TOUCH_TOLERANCE_CM", "aabb", "aabb_overlaps", "aabb_overlaps_between",
           "actor_key", "axis_overlaps_cm", "bounds_coverage", "center_cm",
           "center_distance_cm", "has_bounds",
           "clearance_violations", "contains", "cyclic_degrees",
           "distance_to_polyline", "extent_cm", "intersects_rectangle",
           "location_cm", "overlap_on_axis", "pair_geometry", "placement",
           "rotation_deg", "rounded", "scale", "volume_cm3",
           "xy_overlap_fraction"]
