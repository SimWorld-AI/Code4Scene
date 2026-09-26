"""Tolerance-based repair diagnostics; never used by a primary score.

Scan2CAD (CVPR 2019) motivates conjunctive pose/scale correctness, and BOP
(ECCV 2020 workshops) motivates thresholded pose recall. Our exact-asset,
attribute and signed-scale checks and 5 cm / 5 deg / 5% nominal tolerances
are a separate, initial benchmark policy, not their calibrated thresholds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import contracts
from .assignment import match
from .gt_geometry_compare import actor_pair_attribute_metrics
from .scene_diff import actor_identity, diff_scenes, index_actors


POLICY_ID = "repair-success-diagnostic.v1"
PROFILES = {
    "strict": {"position_cm": 2.5, "rotation_deg": 2.5, "relative_size": 0.025},
    "nominal": {"position_cm": 5.0, "rotation_deg": 5.0, "relative_size": 0.05},
    "relaxed": {"position_cm": 10.0, "rotation_deg": 10.0, "relative_size": 0.10},
}
_ATTRIBUTE_FIELDS = {
    "task_relevant_properties_match": "properties",
    "material_set_match": "material_paths",
    "component_material_slots_match": "component_material_slots",
}


def _vector(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (float, int))
           or not math.isfinite(x) for x in value):
        return None
    return tuple(float(x) for x in value)


def _rotation_quaternion(angles: Sequence[float]) -> tuple[float, ...]:
    """UE FRotator order: pitch, yaw, roll; preserve its handedness."""
    p, y, r = [math.radians(x) / 2.0 for x in angles]
    sp, sy, sr = math.sin(p), math.sin(y), math.sin(r)
    cp, cy, cr = math.cos(p), math.cos(y), math.cos(r)
    return (cr * sp * sy - sr * cp * cy,
            -cr * sp * cy - sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def _rotation_error(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(
        _rotation_quaternion(left), _rotation_quaternion(right), strict=True)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def _relative_error(left: Sequence[float], right: Sequence[float], *, scale: bool) -> float:
    # Per-axis maximum: averaging must not hide a bad axis or a reflection.
    errors = []
    for a, b in zip(left, right, strict=True):
        if b == 0:
            errors.append(math.inf if scale or a != 0 else 0.0)
        else:
            errors.append(abs(a / b - 1.0))
    return max(errors)


def pair_errors(candidate: Mapping[str, Any], gt: Mapping[str, Any]) -> dict[str, Any]:
    # Lazy import keeps the shared target-scope module free to call measure().
    from .repair_target_scope import _actor_descriptor

    left, right = _actor_descriptor(candidate), _actor_descriptor(gt)
    failures = []
    if not right["actor_type"] or left != right:
        failures.append("identity_mismatch_or_missing")
    errors: dict[str, float | None] = {}
    for field, name in (("location_cm", "position_cm"),
                        ("rotation_deg", "rotation_deg"), ("scale", "relative_scale")):
        a = _vector((candidate.get("transform") or {}).get(field))
        b = _vector((gt.get("transform") or {}).get(field))
        if a is None or b is None:
            failures.append(f"{field}_missing_or_invalid")
            errors[name] = None
        else:
            value = (math.dist(a, b) if field == "location_cm" else
                     _rotation_error(a, b) if field == "rotation_deg" else
                     _relative_error(a, b, scale=True))
            errors[name] = value if math.isfinite(value) else None
            if errors[name] is None:
                failures.append(f"{field}_invalid")
    # Applicability is determined by GT, never by what Candidate omitted.
    omitted = []
    for field, name in (("origin_cm", "bounds_position_cm"),
                        ("extent_cm", "relative_bounds_size")):
        gt_bounds = gt.get("bounds") or {}
        if field not in gt_bounds:
            omitted.append(f"bounds.{field}")
            continue
        a = _vector((candidate.get("bounds") or {}).get(field))
        b = _vector(gt_bounds[field])
        if a is None or b is None or (field == "extent_cm" and min(*a, *b) < 0):
            errors[name] = None
        else:
            value = math.dist(a, b) if field == "origin_cm" else _relative_error(a, b, scale=False)
            errors[name] = value if math.isfinite(value) else None
        if errors[name] is None:
            failures.append(f"bounds.{field}_missing_or_invalid")
    attributes = actor_pair_attribute_metrics(candidate, gt)
    for check, field in _ATTRIBUTE_FIELDS.items():
        if field not in gt:
            omitted.append(field)
        elif (not isinstance(gt[field], Mapping if field == "properties" else list)
              or not isinstance(candidate.get(field), Mapping if field == "properties" else list)
              or attributes[check] is not True):
            failures.append(f"{field}_mismatch_or_missing")
    return {"errors": errors, "hard_failures": failures, "gt_unavailable_fields": omitted}


def _failures(pair: Mapping[str, Any], thresholds: Mapping[str, float]) -> list[str]:
    failed = list(pair["hard_failures"])
    for name, value in pair["errors"].items():
        limit = (thresholds["position_cm"] if name.endswith("position_cm") else
                 thresholds["rotation_deg"] if name == "rotation_deg" else
                 thresholds["relative_size"])
        if value is not None and value > limit + 1e-9:
            failed.append(f"{name}_out_of_tolerance")
    return failed


def _counts(tp: int, candidates: int, desired: int) -> dict[str, Any]:
    applicable = desired > 0
    return {
        "status": "measured" if applicable else "not_applicable",
        "true_positive": tp, "false_positive": candidates - tp,
        "false_negative": desired - tp,
        "candidate_count": candidates, "desired_count": desired,
        "precision": (tp / candidates if candidates else 0.0) if applicable else None,
        "recall": tp / desired if applicable else None,
        "f1": 2.0 * tp / (candidates + desired) if applicable else None,
    }


def measure(scope: Any, candidate_scene: Mapping[str, Any]) -> dict[str, Any]:
    """Match only fully correct pairs, independently of continuous matching."""
    from .repair_target_scope import _candidate_resolution

    candidates, reasons = _candidate_resolution(scope, candidate_scene)
    desired = list(scope.desired_actors)
    pairs = [[pair_errors(c, g) for c in candidates] for g in desired]
    profiles = {}
    for name, thresholds in PROFILES.items():
        failures = [[_failures(pair, thresholds) for pair in row] for row in pairs]
        # A 0/1 assignment maximizes the number of passing pairs. Reject all
        # assigned 1-edges afterwards; no continuous assignment is reused.
        assigned = match([[float(bool(f)) for f in row] for row in failures]) if desired and candidates else []
        passed = [(g, c) for g, c in assigned if not failures[g][c]]
        matched_gt, matched_candidate = {g for g, _ in passed}, {c for _, c in passed}
        profiles[name] = {
            **_counts(len(passed), len(candidates), len(desired)),
            "thresholds": dict(thresholds),
            "matches": [{"gt": actor_identity(desired[g]),
                         "candidate": actor_identity(candidates[c]),
                         **pairs[g][c]} for g, c in passed],
            "unmatched_gt": [actor_identity(g) for i, g in enumerate(desired) if i not in matched_gt],
            "unmatched_candidates": [actor_identity(c) for i, c in enumerate(candidates) if i not in matched_candidate],
        }
        if name == "nominal":
            diagnostics = []
            for g in range(len(desired)):
                if g in matched_gt:
                    continue
                valid = [c for c in range(len(candidates)) if not failures[g][c]]
                detail = {"gt": actor_identity(desired[g]),
                          "reason": "one_to_one_conflict" if valid else "no_successful_candidate",
                          "valid_candidate_ids": [actor_identity(candidates[c]) for c in valid]}
                if candidates and not valid:
                    # Explain the most plausible failed pair, not an arbitrary
                    # incompatible edge that filled out the 0/1 assignment.
                    c = min(range(len(candidates)), key=lambda c: (
                        "identity_mismatch_or_missing" in pairs[g][c]["hard_failures"],
                        len(failures[g][c]),
                        sum(x / (1 + x) for x in pairs[g][c]["errors"].values() if x is not None),
                    ))
                    detail["closest_failed_candidate"] = {
                        "candidate": actor_identity(candidates[c]),
                        "failure_reasons": failures[g][c], **pairs[g][c]}
                diagnostics.append(detail)
            profiles[name]["unmatched_gt_diagnostics"] = diagnostics
    removals = [t for t in scope.targets if t.operation == "remove" and t.input_actor is not None]
    candidate_ids = index_actors(candidate_scene)
    removed = sum(actor_identity(t.input_actor) not in candidate_ids for t in removals)
    frozen_ids = {actor_identity(t.input_actor) for t in scope.targets if t.input_actor is not None}
    off_target_removals = [actor_identity(a) for a in diff_scenes(scope.input_scene, candidate_scene).removed
                           if actor_identity(a) not in frozen_ids]
    return {
        "policy_id": POLICY_ID, "status": "measured", "score_role": "report_only",
        "contributes_to_aggregate": False, "nominal_profile": "nominal",
        "coordinate_frame": "absolute_UE_world_no_candidate_fitted_alignment",
        "symmetry_policy": "exact_authored_orientation_no_unannotated_symmetry_equivalence",
        "candidate_scope": "all_added_moved_modified_and_surviving_frozen_target_actors_before_compatibility_filter",
        "candidate_reasons": reasons, "profiles": profiles,
        "gt_evidence": [{"gt": actor_identity(g), **pair_errors(g, g)} for g in desired],
        "removal": {"target_count": len(removals), "completed_count": removed,
                    "completion": removed / len(removals) if removals else None,
                    "identity_policy": "designated_input_actor_identity_absent"},
        "off_target_removed_actor_ids": off_target_removals,
        "references": ["https://arxiv.org/abs/1811.11187", "https://arxiv.org/abs/2009.07378"],
    }


def report(context: Any, measurement: Mapping[str, Any] | None) -> dict[str, Any]:
    """A diagnostic subtree: even failure/unavailability cannot affect parents."""
    data = dict(measurement or {})
    nominal = (data.get("profiles") or {}).get("nominal") or {}
    values = {key: nominal.get(key) for key in ("precision", "recall", "f1")}
    values["removal_completion"] = (data.get("removal") or {}).get("completion")
    measured = data.get("status") == "measured"
    children = []
    for key, value in values.items():
        children.append({
            **contracts.base(f"gt_repair.repair_target_diff.repair_success.{key}", context.ids),
            "leaf_id": key, "score": value,
            "status": "measured" if value is not None else "not_applicable" if measured else "not_evaluated",
            "contributes_to_aggregate": False, "score_role": "report_only",
            "metrics": {}, "evidence": {"diagnostic_policy_id": POLICY_ID},
            "failure_reason": None if value is not None else
                "no applicable target" if measured else data.get("failure_reason", "repair success measurements unavailable"),
        })
    return {
        **contracts.base("gt_repair.repair_target_diff.repair_success", context.ids),
        "leaf_id": "repair_success", "status": "measured" if measured else "not_evaluated",
        "score": None, "contributes_to_aggregate": False, "score_role": "report_only",
        "failure_reason": None if measured else data.get("failure_reason", "repair success measurements unavailable"),
        "metrics": {"repair_success": data, "leaf_results": children},
        "evidence": {"diagnostic_policy_id": POLICY_ID, "used_for_overall": False},
    }


def _valid_record(data: Any) -> bool:
    if not isinstance(data, Mapping) or data.get("policy_id") != POLICY_ID or data.get("status") != "measured":
        return False
    def count(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    profiles = data.get("profiles")
    if not isinstance(profiles, Mapping):
        return False
    for name, thresholds in PROFILES.items():
        row = profiles.get(name)
        if not isinstance(row, Mapping) or any(not count(row.get(key)) for key in
                ("true_positive", "candidate_count", "desired_count")):
            return False
        if row["true_positive"] > min(row["candidate_count"], row["desired_count"]):
            return False
        if row.get("thresholds", thresholds) != thresholds:
            return False
    removal = data.get("removal")
    return (isinstance(removal, Mapping) and count(removal.get("target_count"))
            and count(removal.get("completed_count"))
            and removal["completed_count"] <= removal["target_count"])


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool counts within a track. Do not mistake mean per-case P/R for micro P/R."""
    records = []
    def visit(node: Mapping[str, Any]) -> None:
        if not isinstance(node, Mapping):
            return
        metrics = node.get("metrics") or {}
        data = node.get("repair_success") or metrics.get("repair_success")
        if _valid_record(data):
            records.append(data)
            return
        for child in node.get("children") or metrics.get("leaf_results") or ():
            visit(child)
    for row in rows:
        before = len(records)
        for root in row.get("reports") or ():
            visit(root)
        if len(records) == before:
            for root in (row.get("score_breakdown") or {}).get("reports") or ():
                visit(root)
        if len(records) > before + 1:
            # Ambiguous duplicate diagnostic roots are unavailable, not twice
            # the sample count. This has no bearing on the primary score.
            del records[before:]
    profiles = {}
    for name in PROFILES:
        applicable = [r["profiles"][name] for r in records if r["profiles"][name]["desired_count"] > 0]
        profiles[name] = {
            **_counts(sum(r["true_positive"] for r in applicable),
                      sum(r["candidate_count"] for r in applicable),
                      sum(r["desired_count"] for r in applicable)),
            "observed_case_count": len(applicable),
        }
    removal_count = sum(r["removal"]["target_count"] for r in records)
    removed = sum(r["removal"]["completed_count"] for r in records)
    return {"policy_id": POLICY_ID, "score_role": "report_only", "profiles": profiles,
            "selected_case_count": len(rows), "measured_case_count": len(records),
            "unavailable_case_count": len(rows) - len(records),
            "aggregation": "micro_counts_on_measured_applicable_cases",
            "removal": {"target_count": removal_count, "completed_count": removed,
                        "completion": removed / removal_count if removal_count else None}}
