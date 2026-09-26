"""Is this Actor near / left of / on top of / against that one.

Fourteen named relations over pairs of world AABBs, each reduced to one
predicate and the geometry that decided it. The declared `spatial_relation`
assertions and RequirementGraph's internal object-relation family both read
this, and they must agree about what `on_top_of` means or the
benchmark asks two different questions under one word.

What the geometry commits to:

* `near` / `far` / `distance_range` measure between CENTRES by default. A rule
  may ask for `closest_surface` instead, which is the honest measure for large
  meshes — two buildings 5 m apart have centres 60 m apart — but it is opt-in,
  because changing the default would silently rescore every existing rule.
* `left_of` and friends are world-axis relations, not viewer-relative ones. A
  rule that means "left from the camera" has to say which camera, and none
  of them do; calling the +X axis "left" and saying so is better than a
  relation that is right about a third of the time.
* `on_top_of` needs BOTH a vertical touch and a footprint overlap. Vertical
  alone credits a lamp hovering over a table it misses entirely.
* `facing` uses yaw only. Pitch and roll on a placed prop are usually noise.
  A rule may add a subject-asset forward-axis yaw offset when the mesh's visual
  forward direction is not the Actor's local +X axis.

One deliberate departure from the JavaScript: an unrecognised relation name
raises instead of quietly failing. There, `pass` began as false and only a
matching branch could set it, so a typo in a rule name produced a confident
failure and a scene was marked wrong for a mistake in the case file.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .scene_geometry import (aabb, center_cm, contains, cyclic_degrees,
                             pair_geometry, rotation_deg, rounded,
                             xy_overlap_fraction)
from .scene_diff import actor_identity

#: Every relation this module can decide. A rule naming anything else is a
#: case-file error, reported as one.
RELATIONS = ("near", "far", "distance_range", "left_of", "right_of", "above",
             "below", "inside", "on_top_of", "facing", "parallel",
             "intersects", "against", "occupies", "blocks",
             "angular_coverage", "straddles", "collinear",
             "spacing_uniformity")

GROUP_RELATIONS = ("angular_coverage", "straddles", "collinear",
                   "spacing_uniformity")

#: How a rule's subjects and objects must line up for the rule to hold.
QUANTIFIERS = ("all_subjects_any_object", "any_pair", "all_pairs")

DEFAULT_QUANTIFIER = "all_subjects_any_object"


class RelationError(Exception):
    """A relation rule cannot be decided as written."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RelationError(f"{name} must be a JSON number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise RelationError(f"{name} must be finite")
    return number


def _optional(rule: Mapping[str, Any], name: str, default: float) -> float:
    value = rule.get(name)
    return default if value is None else _number(value, name)


def _xy_pair(value: Any, name: str) -> list[float]:
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes)) or len(value) != 2):
        raise RelationError(f"{name} must be an array of two JSON numbers")
    return [_number(value[index], f"{name}[{index}]") for index in range(2)]


def _object_surface(rule: Mapping[str, Any]) -> dict[str, Any] | None:
    """A case-private horizontal support plane inside a compound Actor.

    Some useful anchors have one Actor AABB for several shelves.  Their AABB
    top is not the particular shelf a task asks the agent to restore.  A rule
    may therefore declare that shelf's measured world-space plane and XY
    rectangle while still selecting the real anchor Actor on the object side.
    """
    value = rule.get("object_surface")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RelationError("object_surface must be an object")
    plane = _number(value.get("plane_z_cm"), "object_surface.plane_z_cm")
    minimum = _xy_pair(value.get("min_xy_cm"), "object_surface.min_xy_cm")
    maximum = _xy_pair(value.get("max_xy_cm"), "object_surface.max_xy_cm")
    if any(minimum[axis] >= maximum[axis] for axis in range(2)):
        raise RelationError(
            "object_surface.min_xy_cm must be strictly below "
            "object_surface.max_xy_cm on both axes")
    return {"plane_z_cm": plane, "min_xy_cm": minimum,
            "max_xy_cm": maximum}


def _xy_rectangle_overlap_fraction(
    subject: Mapping[str, Any],
    minimum: Sequence[float],
    maximum: Sequence[float],
) -> float:
    subject_min, subject_max = aabb(subject)
    overlap_x = max(0.0, min(subject_max[0], maximum[0])
                    - max(subject_min[0], minimum[0]))
    overlap_y = max(0.0, min(subject_max[1], maximum[1])
                    - max(subject_min[1], minimum[1]))
    subject_area = max(0.0, subject_max[0] - subject_min[0]) * max(
        0.0, subject_max[1] - subject_min[1])
    if subject_area <= 0.0:
        return 0.0
    return max(0.0, min(1.0, overlap_x * overlap_y / subject_area))


def observe(subject: Mapping[str, Any], object_actor: Mapping[str, Any],
            rule: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Decide one ordered pair, and return the geometry that decided it."""
    relation = str(rule.get("relation") or "")
    if relation not in RELATIONS:
        raise RelationError(
            f"unknown spatial relation {relation!r}; this build decides "
            f"{', '.join(RELATIONS)}")

    geometry = pair_geometry(subject, object_actor)
    measure = ("closest_surface" if rule.get("distance_measure") == "closest_surface"
               else "center")
    distance = (geometry["closest_surface_distance_cm"] if measure == "closest_surface"
                else geometry["center_distance_cm"])
    observed: dict[str, Any] = {**geometry, "relation_distance_cm": distance,
                                "distance_measure": measure}
    subject_min, subject_max = aabb(subject)
    object_min, object_max = aabb(object_actor)
    subject_center = center_cm(subject)
    object_center = center_cm(object_actor)

    if relation == "near":
        return distance <= _number(rule.get("maximum_distance_cm"),
                                   "maximum_distance_cm"), observed
    if relation == "far":
        return distance >= _number(rule.get("minimum_distance_cm"),
                                   "minimum_distance_cm"), observed
    if relation == "distance_range":
        low = _number(rule.get("minimum_distance_cm"), "minimum_distance_cm")
        high = _number(rule.get("maximum_distance_cm"), "maximum_distance_cm")
        return low <= distance <= high, observed

    margin = _optional(rule, "margin_cm", 0.0)
    tolerance = _optional(rule, "tolerance_cm", 0.0)

    # Each of the four is the clear gap between one box's near face and the
    # other's far face along one axis, so a negative value means they overlap
    # and only a gap of at least `margin_cm` counts as being on that side.
    separations = {
        "left_of": (object_min, subject_max, 0),     # subject at lower X
        "right_of": (subject_min, object_max, 0),    # subject at higher X
        "above": (subject_min, object_max, 2),       # subject at higher Z
        "below": (object_min, subject_max, 2),       # subject at lower Z
    }
    if relation in separations:
        lower, upper, axis = separations[relation]
        observed["axis_gap_cm"] = rounded(lower[axis] - upper[axis])
        return observed["axis_gap_cm"] >= margin, observed

    if relation == "inside":
        return contains(object_actor, subject, tolerance), observed

    if relation == "on_top_of":
        surface = _object_surface(rule)
        if surface is None:
            surface_z = object_max[2]
            overlap = xy_overlap_fraction(subject, object_actor)
            observed["support_surface_source"] = "object_actor_aabb"
        else:
            surface_z = surface["plane_z_cm"]
            overlap = _xy_rectangle_overlap_fraction(
                subject, surface["min_xy_cm"], surface["max_xy_cm"])
            observed.update({
                "support_surface_source": "declared_object_surface",
                "object_surface": {
                    "plane_z_cm": rounded(surface["plane_z_cm"]),
                    "min_xy_cm": [rounded(value)
                                  for value in surface["min_xy_cm"]],
                    "max_xy_cm": [rounded(value)
                                  for value in surface["max_xy_cm"]],
                },
            })
        observed["vertical_gap_cm"] = rounded(subject_min[2] - surface_z)
        observed["xy_overlap_fraction"] = rounded(overlap)
        minimum = _optional(rule, "minimum_overlap_fraction", 0.1)
        return (abs(observed["vertical_gap_cm"]) <= tolerance
                and observed["xy_overlap_fraction"] >= minimum), observed

    if relation == "facing":
        desired = math.degrees(math.atan2(object_center[1] - subject_center[1],
                                          object_center[0] - subject_center[0]))
        subject_yaw = rotation_deg(subject)[1]
        forward_offset = _optional(
            rule, "subject_forward_yaw_offset_deg", 0.0)
        heading = subject_yaw + forward_offset
        # Keep the reported heading in the same [-180, 180) convention as
        # atan2. The cyclic comparison itself accepts any equivalent angle.
        observed_heading = (heading + 180.0) % 360.0 - 180.0
        observed.update({
            "subject_yaw_deg": rounded(subject_yaw),
            "subject_forward_yaw_offset_deg": rounded(forward_offset),
            "observed_heading_yaw_deg": rounded(observed_heading),
            "desired_yaw_deg": rounded(desired),
        })
        observed["yaw_error_deg"] = rounded(
            cyclic_degrees(heading, desired))
        return observed["yaw_error_deg"] <= _number(
            rule.get("maximum_angle_deg"), "maximum_angle_deg"), observed

    if relation == "parallel":
        subject_yaw = rotation_deg(subject)[1]
        object_yaw = rotation_deg(object_actor)[1]
        observed.update({
            "subject_yaw_deg": rounded(subject_yaw),
            "object_yaw_deg": rounded(object_yaw),
            "yaw_error_deg": rounded(cyclic_degrees(subject_yaw, object_yaw)),
        })
        return observed["yaw_error_deg"] <= _number(
            rule.get("maximum_angle_deg"), "maximum_angle_deg"), observed

    if relation in GROUP_RELATIONS:
        raise RelationError(
            f"{relation} is a population relation; call evaluate with both "
            "Actor populations"
        )

    if relation == "intersects":
        return (geometry["intersecting_axis_count"] == 3
                and geometry["intersection_volume_cm3"] > 0), observed

    if relation == "against":
        maximum = rule.get("maximum_surface_gap_cm")
        if maximum is None:
            maximum = 10.0 if rule.get("tolerance_cm") is None else tolerance
        else:
            maximum = _number(maximum, "maximum_surface_gap_cm")
        observed["maximum_surface_gap_cm"] = maximum
        # Two axes, not three: a bookshelf against a wall touches on one axis
        # and overlaps on the other two, and requiring three would only accept
        # a bookshelf INSIDE the wall.
        return (geometry["closest_surface_distance_cm"] <= maximum
                and geometry["intersecting_axis_count"] >= 2), observed

    if relation == "occupies":
        minimum = _optional(rule, "minimum_overlap_fraction", 0.1)
        observed["occupancy_measure"] = "xy_overlap_fraction_of_subject"
        return geometry["xy_overlap_fraction_of_subject"] >= minimum, observed

    minimum = _optional(rule, "minimum_overlap_fraction", 0.01)      # blocks
    observed["blocking_measure"] = "overlap_volume_or_xy_overlap_of_subject"
    return (geometry["overlap_volume_fraction_of_subject"] >= minimum
            or geometry["xy_overlap_fraction_of_subject"] >= minimum), observed


def _key(actor: Mapping[str, Any]) -> str:
    value = actor.get("evaluator_object_id")
    return str(value) if value else actor_identity(actor)


def _summarize(actor: Mapping[str, Any]) -> dict[str, Any]:
    return {"actor_id": _key(actor), "label": actor.get("label"),
            "location_cm": center_cm(actor),
            "logical_object": actor.get("is_logical_object") is True,
            "raw_actor_count": actor.get("raw_actor_count") or 1}


def _quantifier_holds(rows: Sequence[Mapping[str, Any]], subject_count: int,
                      quantifier: str) -> bool:
    if not subject_count or not rows:
        return False
    if quantifier == "any_pair":
        return any(row["pass"] for row in rows)
    if quantifier == "all_pairs":
        return all(row["pass"] for row in rows)
    by_subject: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject_key"]), []).append(row)
    return (len(by_subject) == subject_count
            and all(any(row["pass"] for row in subject_rows)
                    for subject_rows in by_subject.values()))


def _integer(rule: Mapping[str, Any], name: str, default: int,
             minimum: int) -> int:
    number = _optional(rule, name, float(default))
    if not number.is_integer() or number < minimum:
        raise RelationError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _population_center(objects: Sequence[Mapping[str, Any]]) -> list[float]:
    centres = [center_cm(actor) for actor in objects]
    return [rounded(sum(point[axis] for point in centres) / len(centres))
            for axis in range(3)]


def _principal_axis_xy(
    subjects: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Fit the least-squares XY line through subject AABB centres.

    This is the two-dimensional principal component written out directly so
    the evaluator does not acquire a numerical-library dependency.  The axis
    sign is canonicalised to keep evidence stable across runs; line distance
    and adjacent spacing themselves are sign invariant.
    """
    points = [center_cm(actor)[:2] for actor in subjects]
    center = [sum(point[axis] for point in points) / len(points)
              for axis in range(2)]
    offsets = [[point[0] - center[0], point[1] - center[1]]
               for point in points]
    xx = sum(point[0] * point[0] for point in offsets)
    xy = sum(point[0] * point[1] for point in offsets)
    yy = sum(point[1] * point[1] for point in offsets)
    if xx + yy <= 1e-12:
        return None
    angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
    axis = [math.cos(angle), math.sin(angle)]
    if axis[0] < -1e-12 or (abs(axis[0]) <= 1e-12 and axis[1] < 0.0):
        axis = [-axis[0], -axis[1]]
    projections = [point[0] * axis[0] + point[1] * axis[1]
                   for point in offsets]
    deviations = [abs(-axis[1] * point[0] + axis[0] * point[1])
                  for point in offsets]
    return {
        "center_xy_cm": center,
        "axis_xy": axis,
        "axis_angle_deg": math.degrees(math.atan2(axis[1], axis[0])),
        "projections_cm": projections,
        "deviations_cm": deviations,
    }


def _layout_rows(
    subjects: Sequence[Mapping[str, Any]],
    fit: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if fit is None:
        return [{"subject": _summarize(actor), "projection_cm": None,
                 "perpendicular_deviation_cm": None}
                for actor in subjects]
    rows = [
        {"subject": _summarize(actor),
         "projection_cm": rounded(fit["projections_cm"][index]),
         "perpendicular_deviation_cm": rounded(
             fit["deviations_cm"][index])}
        for index, actor in enumerate(subjects)
    ]
    return sorted(rows, key=lambda row: (row["projection_cm"],
                                         row["subject"]["actor_id"]))


def _layout_observed(
    subjects: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    fit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    observed: dict[str, Any] = {
        "subject_count": len(subjects),
        "object_count": len(objects),
        "geometry_population": "subjects",
        "degenerate_xy": fit is None,
    }
    if fit is not None:
        observed.update({
            "subject_population_center_xy_cm": [
                rounded(value) for value in fit["center_xy_cm"]],
            "principal_axis_xy": [rounded(value) for value in fit["axis_xy"]],
            "principal_axis_angle_deg": rounded(fit["axis_angle_deg"]),
            "maximum_perpendicular_deviation_cm": rounded(
                max(fit["deviations_cm"])),
        })
    return observed


def _evaluate_group_relation(
    rule: Mapping[str, Any],
    subjects: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    minimum_subjects: int,
    minimum_objects: int,
) -> dict[str, Any]:
    relation = str(rule.get("relation"))
    missing = not subjects or not objects
    counts_ok = (len(subjects) >= minimum_subjects
                 and len(objects) >= minimum_objects)
    if missing:
        return {
            "pass": False,
            "expected": {"relation": relation,
                         "minimum_subject_count": minimum_subjects,
                         "minimum_object_count": minimum_objects},
            "observed": {"subject_count": len(subjects),
                         "object_count": len(objects)},
            "rows": [],
            "missing_selection": True,
        }

    if relation == "angular_coverage":
        center = _population_center(objects)
        sectors = _integer(rule, "angular_sector_count", 8, 2)
        minimum_coverage = _number(
            rule.get("minimum_angular_coverage"),
            "minimum_angular_coverage",
        )
        if not 0.0 <= minimum_coverage <= 1.0:
            raise RelationError("minimum_angular_coverage must be in [0, 1]")
        coverage = angular_coverage(subjects, center, sectors)
        holds = counts_ok and coverage["coverage"] >= minimum_coverage
        expected = {
            "relation": relation,
            "minimum_subject_count": minimum_subjects,
            "minimum_object_count": minimum_objects,
            "center_mode": "object_population_centroid",
            "angular_sector_count": sectors,
            "minimum_angular_coverage": minimum_coverage,
        }
        observed = {
            "subject_count": len(subjects),
            "object_count": len(objects),
            "object_population_center_cm": center,
            **coverage,
        }
        rows = [{"subject": _summarize(actor),
                 "angle_deg": rounded(math.degrees(math.atan2(
                     center_cm(actor)[1] - center[1],
                     center_cm(actor)[0] - center[0])) % 360)}
                for actor in subjects]
    elif relation == "straddles":
        center = _population_center(objects)
        axis_name = str(rule.get("axis") or "x").lower()
        axes = {"x": 0, "y": 1, "z": 2}
        if axis_name not in axes:
            raise RelationError("straddles axis must be one of x, y, z")
        axis = axes[axis_name]
        minimum_offset = _optional(rule, "minimum_offset_cm", 0.0)
        if minimum_offset < 0:
            raise RelationError("minimum_offset_cm must be non-negative")
        minimum_per_side = _integer(rule, "minimum_per_side", 1, 1)
        rows = []
        negative = positive = 0
        for actor in subjects:
            delta = rounded(center_cm(actor)[axis] - center[axis])
            side = ("negative" if delta <= -minimum_offset else
                    "positive" if delta >= minimum_offset else "center")
            negative += side == "negative"
            positive += side == "positive"
            rows.append({"subject": _summarize(actor),
                         "axis_delta_cm": delta, "side": side})
        holds = (counts_ok and negative >= minimum_per_side
                 and positive >= minimum_per_side)
        expected = {
            "relation": relation,
            "minimum_subject_count": minimum_subjects,
            "minimum_object_count": minimum_objects,
            "center_mode": "object_population_centroid",
            "axis": axis_name,
            "minimum_offset_cm": minimum_offset,
            "minimum_per_side": minimum_per_side,
        }
        observed = {
            "subject_count": len(subjects),
            "object_count": len(objects),
            "object_population_center_cm": center,
            "negative_side_count": negative,
            "positive_side_count": positive,
            "center_band_count": len(subjects) - negative - positive,
        }
    elif relation == "collinear":
        maximum = _number(rule.get("maximum_deviation_cm"),
                          "maximum_deviation_cm")
        if maximum < 0.0:
            raise RelationError("maximum_deviation_cm must be non-negative")
        fit = _principal_axis_xy(subjects)
        rows = _layout_rows(subjects, fit)
        expected = {
            "relation": relation,
            "minimum_subject_count": minimum_subjects,
            "minimum_object_count": minimum_objects,
            "geometry_population": "subjects",
            "fit": "xy_center_principal_axis",
            "maximum_deviation_cm": maximum,
        }
        observed = _layout_observed(subjects, objects, fit)
        holds = (counts_ok and fit is not None
                 and observed["maximum_perpendicular_deviation_cm"] <= maximum)
    else:  # spacing_uniformity
        minimum_spacing = _number(rule.get("minimum_spacing_cm"),
                                  "minimum_spacing_cm")
        maximum_spacing = _number(rule.get("maximum_spacing_cm"),
                                  "maximum_spacing_cm")
        maximum_relative = _number(
            rule.get("maximum_relative_deviation"),
            "maximum_relative_deviation")
        if minimum_spacing < 0.0:
            raise RelationError("minimum_spacing_cm must be non-negative")
        if maximum_spacing < minimum_spacing:
            raise RelationError(
                "maximum_spacing_cm must be >= minimum_spacing_cm")
        if maximum_relative < 0.0:
            raise RelationError(
                "maximum_relative_deviation must be non-negative")
        fit = _principal_axis_xy(subjects)
        rows = _layout_rows(subjects, fit)
        expected = {
            "relation": relation,
            "minimum_subject_count": minimum_subjects,
            "minimum_object_count": minimum_objects,
            "geometry_population": "subjects",
            "ordering": "projection_on_xy_center_principal_axis",
            "minimum_spacing_cm": minimum_spacing,
            "maximum_spacing_cm": maximum_spacing,
            "maximum_relative_deviation": maximum_relative,
        }
        observed = _layout_observed(subjects, objects, fit)
        if fit is None:
            spacings: list[float] = []
            mean_spacing = None
            relative_deviations: list[float] = []
            observed_maximum_relative = None
        else:
            ordered = sorted(fit["projections_cm"])
            spacings = [rounded(ordered[index + 1] - ordered[index])
                        for index in range(len(ordered) - 1)]
            mean_spacing = rounded(sum(spacings) / len(spacings))
            relative_deviations = ([rounded(abs(value - mean_spacing)
                                                    / mean_spacing)
                                    for value in spacings]
                                   if mean_spacing > 0.0 else [])
            observed_maximum_relative = (
                max(relative_deviations) if relative_deviations else None)
        observed.update({
            "adjacent_spacing_cm": spacings,
            "mean_spacing_cm": mean_spacing,
            "relative_deviations": relative_deviations,
            "maximum_relative_deviation": observed_maximum_relative,
        })
        holds = (
            counts_ok and fit is not None and len(spacings) >= 2
            and mean_spacing is not None and mean_spacing > 0.0
            and all(minimum_spacing <= value <= maximum_spacing
                    for value in spacings)
            and observed_maximum_relative is not None
            and observed_maximum_relative <= maximum_relative
        )
    return {"pass": holds, "expected": expected, "observed": observed,
            "rows": rows, "missing_selection": False}


def evaluate(rule: Mapping[str, Any], subjects: Sequence[Mapping[str, Any]],
             objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every subject against every object, reduced by the rule's quantifier.

    Returns the verdict plus the per-pair rows, which is what makes a failing
    relation debuggable: "the chairs are not around the table" is not a report,
    "chair 3 is 412 cm from the table and the rule allows 200" is.
    """
    relation = str(rule.get("relation") or "")
    if relation not in RELATIONS:
        raise RelationError(
            f"unknown spatial relation {relation!r}; this build decides "
            f"{', '.join(RELATIONS)}")
    intrinsic_minimum_subjects = {
        "collinear": 2,
        "spacing_uniformity": 3,
    }.get(relation, 1)
    minimum_subjects = _integer(
        rule, "minimum_subject_count", intrinsic_minimum_subjects,
        intrinsic_minimum_subjects)
    minimum_objects = _integer(rule, "minimum_object_count", 1, 1)
    if relation in GROUP_RELATIONS:
        return _evaluate_group_relation(
            rule, subjects, objects, minimum_subjects, minimum_objects,
        )

    quantifier = str(rule.get("quantifier") or DEFAULT_QUANTIFIER)
    if quantifier not in QUANTIFIERS:
        raise RelationError(
            f"unknown quantifier {quantifier!r}; this build decides "
            f"{', '.join(QUANTIFIERS)}")
    # Validate the new optional field even when selectors resolve to no pairs.
    # A malformed case declaration must never turn into an ordinary scene
    # failure or a vacuous missing-selection result.
    forward_offset = (_optional(rule, "subject_forward_yaw_offset_deg", 0.0)
                      if relation == "facing" else None)
    surface = _object_surface(rule) if relation == "on_top_of" else None
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        subject_key = _key(subject)
        for object_actor in objects:
            if _key(object_actor) == subject_key:
                continue
            ok, observed = observe(subject, object_actor, rule)
            rows.append({"subject_key": subject_key,
                         "subject": _summarize(subject),
                         "object": _summarize(object_actor),
                         "pass": ok, **observed})
    counts_ok = (len(subjects) >= minimum_subjects
                 and len(objects) >= minimum_objects)
    holds = counts_ok and _quantifier_holds(rows, len(subjects), quantifier)
    expected = {
        "relation": rule.get("relation"), "quantifier": quantifier,
        "minimum_subject_count": minimum_subjects,
        "minimum_object_count": minimum_objects,
        "distance_measure": rule.get("distance_measure") or "center",
        **{key: rule.get(key) for key in (
            "minimum_distance_cm", "maximum_distance_cm", "margin_cm",
            "tolerance_cm", "maximum_angle_deg", "minimum_overlap_fraction",
            "maximum_surface_gap_cm")},
    }
    if relation == "facing":
        expected["subject_forward_yaw_offset_deg"] = forward_offset
    if surface is not None:
        expected["object_surface"] = surface
    return {
        "pass": holds,
        "expected": expected,
        "observed": {"subject_count": len(subjects), "object_count": len(objects),
                     "evaluated_pair_count": len(rows),
                     "passing_pair_count": sum(row["pass"] for row in rows)},
        "rows": rows,
        "missing_selection": not subjects or not objects,
    }


def angular_coverage(subjects: Sequence[Mapping[str, Any]],
                     center: Sequence[float], sectors: int = 8) -> dict[str, Any]:
    """How much of the circle around a point the subjects occupy.

    What "around" means when a rubric says the chairs surround the table:
    counting is not enough — four chairs in a row on one side satisfy any
    count-and-distance rule and surround nothing.
    """
    occupied: set[int] = set()
    for subject in subjects:
        position = center_cm(subject)
        angle = math.degrees(math.atan2(position[1] - center[1],
                                        position[0] - center[0])) % 360
        occupied.add(int(angle // (360 / sectors)) % sectors)
    return {"sector_count": sectors, "occupied_sectors": sorted(occupied),
            "coverage": rounded(len(occupied) / sectors)}


__all__ = ["DEFAULT_QUANTIFIER", "GROUP_RELATIONS", "QUANTIFIERS", "RELATIONS",
           "RelationError", "aabb", "angular_coverage", "evaluate", "observe"]
