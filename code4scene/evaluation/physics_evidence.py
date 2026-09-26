"""Is the physics payload sound, and whose measurement is whose.

The probe answers about Actors; the atom asks about Actors it selected itself.
Matching the two is where a metric quietly goes wrong — a record reused for
two Actors, a label that matches more than one thing, an identity that
disagrees with the scene — so every match here either succeeds unambiguously
or names why it did not.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .values import as_text


def measurement_records(measurements: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(measurements, Mapping):
        return {}
    actors = measurements.get("actors")
    source = actors if isinstance(actors, Mapping) else {}
    return {
        str(key): value for key, value in source.items()
        if isinstance(value, Mapping)
    }


def _normalized_path(value: Any) -> str | None:
    text = as_text(value)
    return text.replace("\\", "/").rstrip("/") if text else None


def producer_summary(
    measurements: Any, collection_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = measurements if isinstance(measurements, Mapping) else {}
    provenance = payload.get("runtime_provenance")
    diagnostics = payload.get("diagnostics")
    capabilities = payload.get("capabilities")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    capabilities = capabilities if isinstance(capabilities, Mapping) else {}
    summary = {
        "evidence_type": "physics_measurement_producer",
        "present": measurements is not None,
        "schema_version": as_text(payload.get("schema_version")),
        "measurement_type": as_text(payload.get("measurement_type")),
        "map_path": as_text(payload.get("map_path")),
        "runtime_provenance": {
            "project_file_path": as_text(provenance.get("project_file_path")),
            "engine_version": as_text(provenance.get("engine_version")),
            "map_path": as_text(provenance.get("map_path")),
        },
        "diagnostics": {
            "status": as_text(diagnostics.get("status")),
            "requested_actor_count": diagnostics.get("requested_actor_count"),
            "measured_actor_count": diagnostics.get("measured_actor_count"),
            "unresolved_target_count": len(diagnostics.get("unresolved_targets") or [])
            if isinstance(diagnostics.get("unresolved_targets"), list) else None,
            "measurement_error_count": len(diagnostics.get("measurement_errors") or [])
            if isinstance(diagnostics.get("measurement_errors"), list) else None,
        },
        "capabilities": dict(capabilities),
    }
    if collection_provenance:
        summary["collection_provenance"] = dict(collection_provenance)
    return summary


def validate_measurement_envelope(
    candidate: Mapping[str, Any], measurements: Any,
    collection_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    producer = producer_summary(measurements, collection_provenance)
    if measurements is None:
        return producer, ()
    if not isinstance(measurements, Mapping):
        return producer, ("physics measurement payload must be an object",)
    errors: list[str] = []
    schema_version = measurements.get("schema_version")
    if not isinstance(schema_version, str) or not re.fullmatch(
        r"0\.(?:4|5)\.\d+", schema_version
    ):
        errors.append("schema_version must be a 0.4.x or 0.5.x string")
    if measurements.get("measurement_type") != "ue_editor_actor_physics":
        errors.append("measurement_type must be ue_editor_actor_physics")
    capabilities = measurements.get("capabilities")
    if not isinstance(capabilities, Mapping):
        errors.append("capabilities must be an object")
        capabilities = {}
    solid_capability = capabilities.get("solid_penetration")
    if not isinstance(solid_capability, Mapping):
        errors.append("capabilities.solid_penetration must be an object")
    else:
        required_capability = {
            "broad_phase": "actor_world_aabb",
            "overlap_method": "ue_component_overlap_components",
            "depth_method": "ue_fhitresult_initial_overlap_mtd",
            "aabb_role": "broad_phase_only",
        }
        for field, expected in required_capability.items():
            if solid_capability.get(field) != expected:
                errors.append(
                    f"capabilities.solid_penetration.{field} must be {expected}"
                )
    actors = measurements.get("actors")
    if not isinstance(actors, Mapping):
        errors.append("actors must be an object")
        actors = {}
    diagnostics = measurements.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        errors.append("diagnostics must be an object")
        diagnostics = {}
    elif diagnostics.get("status") != "success":
        errors.append("diagnostics.status must be success")
    for field in ("unresolved_targets", "measurement_errors"):
        value = diagnostics.get(field)
        if not isinstance(value, list):
            errors.append(f"diagnostics.{field} must be an array")
        elif value:
            errors.append(f"diagnostics.{field} must be empty")
    measured_count = diagnostics.get("measured_actor_count")
    if isinstance(measured_count, bool) or not isinstance(measured_count, int):
        errors.append("diagnostics.measured_actor_count must be an integer")
    elif measured_count != len(actors):
        errors.append("diagnostics.measured_actor_count must match actors size")
    requested_count = diagnostics.get("requested_actor_count")
    if (
        isinstance(requested_count, bool)
        or not isinstance(requested_count, int)
        or requested_count < 0
    ):
        errors.append("diagnostics.requested_actor_count must be a non-negative integer")
    elif isinstance(measured_count, int) and requested_count != measured_count:
        errors.append(
            "successful diagnostics must have equal requested_actor_count and "
            "measured_actor_count"
        )
    provenance = measurements.get("runtime_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("runtime_provenance must be an object")
        provenance = {}
    for field in ("project_file_path", "engine_version", "map_path"):
        if as_text(provenance.get(field)) is None:
            errors.append(f"runtime_provenance.{field} must be a non-empty string")
    payload_map = _normalized_path(measurements.get("map_path"))
    provenance_map = _normalized_path(provenance.get("map_path"))
    if payload_map is None:
        errors.append("map_path must be a non-empty string")
    elif provenance_map is not None and provenance_map != payload_map:
        errors.append("runtime_provenance.map_path must match map_path")
    candidate_map = _normalized_path(candidate.get("map_path"))
    if candidate_map is not None and payload_map is not None and candidate_map != payload_map:
        errors.append("physics measurement map_path must match candidate map_path")
    collection = collection_provenance or {}
    if (
        collection.get("independent_scoring_editor") is True
        and collection.get("measurements_from_independent_editor") is not True
    ):
        errors.append(
            "an independently exported candidate requires physics measurements "
            "from the same independent scoring editor"
        )
    return producer, tuple(errors)


def _record_identity_conflicts(
    actor: Mapping[str, Any], item: Mapping[str, Any]
) -> bool:
    for field in ("actor_path", "stable_actor_id"):
        measured = as_text(item.get(field))
        if measured is not None and measured != as_text(actor.get(field)):
            return True
    return False


def measurement_for_actor(
    actor: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    label_counts: Counter[str],
    used_keys: set[str],
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    """Match without ever assigning one ambiguous record to two actors."""
    strong_matches: set[str] = set()
    for field in ("actor_path", "stable_actor_id"):
        identity = as_text(actor.get(field))
        if identity is None:
            continue
        if identity in records:
            strong_matches.add(identity)
        strong_matches.update(
            key for key, item in records.items() if as_text(item.get(field)) == identity
        )
    if len(strong_matches) > 1:
        return None, None, "ambiguous_strong_identity"
    if strong_matches:
        key = next(iter(strong_matches))
        if _record_identity_conflicts(actor, records[key]):
            return None, key, "conflicting_strong_identity"
        if key in used_keys:
            return None, key, "measurement_record_reused"
        used_keys.add(key)
        return records[key], key, None

    label = as_text(actor.get("label"))
    if label is None:
        return None, None, "target_identity_missing"
    if label_counts[label] != 1:
        return None, label if label in records else None, "ambiguous_label"
    label_matches = {key for key, item in records.items() if as_text(item.get("label")) == label}
    if label in records:
        label_matches.add(label)
    if len(label_matches) > 1:
        return None, None, "ambiguous_label_measurement"
    if not label_matches:
        return None, None, "measurement_missing"
    key = next(iter(label_matches))
    if any(as_text(records[key].get(field)) for field in ("actor_path", "stable_actor_id")):
        return None, key, "conflicting_strong_identity"
    if key in used_keys:
        return None, key, "measurement_record_reused"
    used_keys.add(key)
    return records[key], key, None

# ── whether one Actor's measurement says it is placed badly ──────────────
#
# Read by `physics_regression`, which compares the SAME question asked of the
# input scene and of the candidate. Two spellings of "is this Actor grounded"
# would make a difference between two scenes meaningless, so the predicate
# lives beside the records it reads rather than in the verifier.

#: What counts as placed acceptably when a case does not say.
DEFAULT_PHYSICS_THRESHOLDS = {"maximum_ground_gap_cm": 5.0,
                              "maximum_penetration_cm": 5.0,
                              "minimum_support_fraction": 0.05}


def _finite(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def has_ground_measurement(item: Any) -> bool:
    """Is this record's ground contact usable, or only partly filled in.

    A record that found no surface is COMPLETE — it says so, and says the
    Actor is not grounded — but only if it agrees with itself. A record
    claiming both a gap and a penetration is describing an Actor that is
    simultaneously above and inside the ground, and is not trusted.
    """
    if not isinstance(item, Mapping) or not isinstance(item.get("grounded"), bool):
        return False
    if item.get("surface_detected") is False:
        return (item.get("grounded") is False and item.get("ground_gap_cm") is None
                and item.get("penetration_cm") is None
                and _finite(item.get("support_fraction")) == 0)
    gap = _finite(item.get("ground_gap_cm"))
    penetration = _finite(item.get("penetration_cm"))
    support = _finite(item.get("support_fraction"))
    return (gap is not None and gap >= 0 and penetration is not None
            and penetration >= 0 and support is not None and 0 <= support <= 1
            and not (gap > 0 and penetration > 0))


def has_bounds_measurement(item: Any) -> bool:
    return isinstance(item, Mapping) and isinstance(item.get("out_of_bounds"), bool)


def physics_violations(actors: Sequence[Mapping[str, Any]], measurements: Any,
                       thresholds: Mapping[str, float] | None = None,
                       ) -> dict[str, Any]:
    """Which of these Actors the measurement says are badly placed.

    Reports what it could NOT evaluate separately from what failed, because a
    regression comparison needs both sides complete: an Actor missing from
    either measurement makes the difference between them unknowable, not zero.
    """
    limits = {**DEFAULT_PHYSICS_THRESHOLDS, **(thresholds or {})}
    records = measurement_records(measurements)
    actors = list(actors)
    if not actors:
        return {"ground_evaluable": True, "bounds_evaluable": True,
                "missing_ground": [], "missing_bounds": [],
                "ground_failures": [], "out_of_bounds": [], "observed": []}
    if not records:
        missing = [as_text(actor.get("label")) for actor in actors]
        return {"ground_evaluable": False, "bounds_evaluable": False,
                "missing_ground": missing, "missing_bounds": missing,
                "ground_failures": [], "out_of_bounds": [], "observed": []}

    missing_ground, missing_bounds = [], []
    ground_failures, out_of_bounds, observed = [], [], []
    label_counts = Counter(label for actor in actors
                           if (label := as_text(actor.get("label"))) is not None)
    used: set[str] = set()
    for actor in actors:
        item, _key, _error = measurement_for_actor(actor, records, label_counts, used)
        ground_ok = has_ground_measurement(item)
        bounds_ok = has_bounds_measurement(item)
        label = as_text(actor.get("label"))
        if not ground_ok:
            missing_ground.append(label)
        if not bounds_ok:
            missing_bounds.append(label)
        item = item if isinstance(item, Mapping) else {}
        gap = _finite(item.get("ground_gap_cm"))
        penetration = _finite(item.get("penetration_cm"))
        support = _finite(item.get("support_fraction"))
        placed = bool(ground_ok and item.get("grounded") is not False
                      and (gap or 0.0) <= limits["maximum_ground_gap_cm"]
                      and (penetration or 0.0) <= limits["maximum_penetration_cm"]
                      and (support if support is not None else 0.0)
                      >= limits["minimum_support_fraction"])
        evidence = {
            "actor_id": (as_text(actor.get("stable_actor_id"))
                         or as_text(actor.get("actor_path")) or label),
            "label": label, "grounded": placed,
            "surface_detected": item.get("surface_detected")
            if isinstance(item.get("surface_detected"), bool) else None,
            "ground_gap_cm": gap, "penetration_cm": penetration,
            "support_fraction": support,
            "measurement_method": as_text(item.get("measurement_method"))}
        if ground_ok:
            observed.append(evidence)
            if not placed:
                ground_failures.append(evidence)
        if bounds_ok and item.get("out_of_bounds") is True:
            out_of_bounds.append({"actor_id": evidence["actor_id"], "label": label,
                                  "measurement_method": evidence["measurement_method"]})
    return {"ground_evaluable": not missing_ground,
            "bounds_evaluable": not missing_bounds,
            "missing_ground": missing_ground, "missing_bounds": missing_bounds,
            "ground_failures": ground_failures, "out_of_bounds": out_of_bounds,
            "observed": observed}


__all__ = ["DEFAULT_PHYSICS_THRESHOLDS", "has_bounds_measurement",
           "has_ground_measurement", "measurement_for_actor",
           "measurement_records", "physics_violations", "producer_summary",
           "validate_measurement_envelope"]
