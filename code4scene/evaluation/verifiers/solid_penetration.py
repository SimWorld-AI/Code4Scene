"""``solid_penetration`` — is an Actor inside real UE collision geometry.

A world AABB says a lamppost intersects the church it stands beside, because
the church's box contains the space in front of its door. It is therefore only
an internal broad phase. The formal verdict comes from UE collision-body
overlap plus the native ``FHitResult.PenetrationDepth`` minimum-translation
distance (MTD).

The primary v3 score is the collision-free share of eligible solid Actors.
The current default policy derives a tolerance for each Actor from five
percent of its shortest full world-AABB span, clamped to 5--50 cm. A task can
still freeze a fixed tolerance explicitly. Any trusted native MTD above that
tolerance fails the target Actor once, so neither repeated contacts nor one
kilometre-scale MTD can contribute more than one unit of penalty per Actor.
Raw depths and the former size-normalized depth weighting remain in the report
as diagnostics; they are not the primary score.

Evidence still has three measurement outcomes, and the third is the point:

* the editor MEASURED every contact depth — that verdict stands either way;
* no trusted editor measurement — **not evaluated**. A box overlap is a reason
  to run the native probe, never a reason to fail. Inside a complete editor
  record, a threshold-independent clear broad phase can eliminate impossible
  collision-body intersections, but it never estimates a depth.

Incomplete editor coverage can still PROVE a violation but can never prove a
pass. Legacy mesh-surface depth and AABB overlap are diagnostic only.

Non-solid Actors are filtered before any of this — lights, fog, emitters,
group actors and water have bounds and nothing to collide with, and leaving
them in makes every scene look penetrated. Scene-relative deterministic roles
also separate support surfaces and oversized environment proxies from scored
target Actors. A trusted contact with an environment proxy is retained as one
actor-level containment failure rather than ignored or charged by raw depth.

Serves component `physics.absolute`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .. import assertions
from ..assertions import Case, Check
from ..context import Context
from ..scene_geometry import TOUCH_TOLERANCE_CM, actor_key, rounded
from ..physics_evidence import measurement_for_actor, measurement_records
from ..scene_semantics import (
    PHYSICS_ROLE_ENVIRONMENT_PROXY, PHYSICS_ROLE_NON_SOLID,
    PHYSICS_ROLE_SCORED_SOLID, PHYSICS_ROLE_SUPPORT_SURFACE,
    PHYSICS_ROLE_UNRESOLVED, actor_name, bounding_box,
    classify_physics_roles, is_irregular_environment, is_non_solid, names,
    overlap_depths,
)
from ..values import nonnegative_finite


#: Penetration beyond this contributes a non-zero penalty unless overridden.
DEFAULT_MAXIMUM_PENETRATION_CM = 5.0

#: Fraction of the shortest full Actor AABB span used by the default policy.
DEFAULT_RELATIVE_PENETRATION_TOLERANCE_FRACTION = 0.05

#: Upper clamp for the default Actor-scale tolerance.
DEFAULT_MAXIMUM_ADAPTIVE_PENETRATION_TOLERANCE_CM = 50.0

#: Public identifier for the per-Actor tolerance calculation.
ADAPTIVE_TOLERANCE_POLICY = "actor_shortest_full_aabb_span_clamped_v1"

#: Prevent degenerate or missing bounds from making the divisor vanish.
MIN_CHARACTERISTIC_SIZE_CM = 10.0

#: Strength of the logarithmic size-normalized depth term.
DEPTH_PENALTY_LAMBDA = 1.0

#: Public identifier for the continuous normalization used in observations.
DEPTH_WEIGHTING = "binary_plus_log1p_size_normalized_excess_v2"

# The primary v3 score is the collision-free share of eligible solid Actors.
# Native MTD remains in the report as severity evidence, but no single
# kilometre-scale proxy can contribute more than one failed target Actor.
SCORING_POLICY = "eligible_actor_collision_free_rate_v3"
ROLE_POLICY = "deterministic_scene_relative_physics_roles_v1"

#: Match failures that mean "the probe never answered about this Actor" rather
#: than "the answer cannot be trusted". Only these fall back to the boxes.
_NO_MEASUREMENT = frozenset({"measurement_missing", "target_identity_missing"})


def _actor_characteristic_size_cm(actor: Mapping[str, Any]) -> float:
    """The shortest full AABB span used to normalize an Actor's MTD."""
    bounds = bounding_box(actor)
    if bounds is None:
        return MIN_CHARACTERISTIC_SIZE_CM
    return max(
        MIN_CHARACTERISTIC_SIZE_CM,
        min(2.0 * float(extent) for extent in bounds["extent"]),
    )


def _penetration_tolerance_cm(
    actor: Mapping[str, Any],
    minimum_cm: float,
    relative_fraction: float,
    maximum_cm: float,
) -> tuple[float, float]:
    """Return the clamped Actor-scale tolerance and characteristic size."""
    characteristic_size = _actor_characteristic_size_cm(actor)
    tolerance = min(
        max(float(minimum_cm), float(relative_fraction) * characteristic_size),
        float(maximum_cm),
    )
    return tolerance, characteristic_size


def _penetration_depth_penalty(
        actor: Mapping[str, Any], depth_cm: float | None,
        tolerance_cm: float) -> tuple[float, float, float]:
    """Return incidence-plus-depth penalty, normalized excess, and size."""
    characteristic_size = _actor_characteristic_size_cm(actor)
    if depth_cm is None:
        return 0.0, 0.0, characteristic_size
    depth = max(0.0, float(depth_cm))
    tolerance = max(0.0, float(tolerance_cm))
    if depth <= tolerance:
        return 0.0, 0.0, characteristic_size
    normalized_excess = (depth - tolerance) / characteristic_size
    penalty = 1.0 + DEPTH_PENALTY_LAMBDA * math.log1p(normalized_excess)
    return penalty, normalized_excess, characteristic_size


def _contacts(measurement: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The per-collider records, when the probe reported any."""
    for field in ("solid_penetrations", "penetrating_with", "solid_overlaps"):
        source = measurement.get(field)
        if isinstance(source, list):
            break
    else:
        return None
    contacts = []
    for item in source:
        if isinstance(item, str):
            contacts.append({"collider": item, "confirmed_overlap": True,
                             "penetration_depth_cm": None, "valid": True,
                             "depth_method": None, "depth_status": None})
            continue
        value = item if isinstance(item, Mapping) else {}
        raw = value.get("penetration_depth_cm")
        depth = nonnegative_finite(raw if raw is not None else value.get("depth_cm"))
        # An explicitly negative depth is not a shallow one; it is a probe
        # that returned nonsense, and the record is untrusted from there.
        stated = raw if raw is not None else value.get("depth_cm")
        contacts.append({
            "collider": (value.get("collider") or value.get("actor")
                         or value.get("label") or value.get("actor_id")
                         or "unknown_solid"),
            "collider_path": value.get("collider_path") or value.get("actor_path"),
            "confirmed_overlap": value.get("confirmed_overlap") is not False,
            "penetration_depth_cm": depth,
            "valid": stated is None or depth is not None,
            "detection_method": value.get("detection_method"),
            "depth_method": value.get("depth_method"),
            "depth_status": value.get("depth_status"),
            "collision_response": value.get("collision_response"),
            "target_component": value.get("target_component"),
            "target_component_path": value.get("target_component_path"),
            "collider_component": value.get("collider_component"),
            "collider_component_path": value.get("collider_component_path"),
            "depth_probe_orientation": value.get("depth_probe_orientation"),
            "depth_probe_attempt_count": value.get("depth_probe_attempt_count"),
            "depth_error": value.get("depth_error"),
            "depth_forward_error": value.get("depth_forward_error"),
            "depth_reciprocal_error": value.get("depth_reciprocal_error")})
    return contacts


def _trusted(actor: Mapping[str, Any], measurement: Mapping[str, Any] | None,
             maximum_cm: float,
             collider_lookup: Mapping[str, Mapping[str, Any]],
             role_index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """The editor's verdict for one Actor, or None to fall back to boxes."""
    if measurement is None:
        return None
    contacts = _contacts(measurement)
    exact_evaluated = measurement.get("solid_penetration_evaluated")
    if contacts is None:
        return None
    if (not contacts
            and measurement.get("solid_penetration_method") ==
            "ue_aabb_broad_phase_no_candidates"):
        broad_phase_tolerance = nonnegative_finite(
            measurement.get("solid_broad_phase_tolerance_cm"))
        if (broad_phase_tolerance is None
                or broad_phase_tolerance > maximum_cm):
            return None
    external_contacts = []
    internal_contacts = []
    filtered_role_contacts = []
    for item in contacts or []:
        collider = (collider_lookup.get(str(item.get("collider_path")))
                    or collider_lookup.get(str(item.get("collider"))))
        left_object = actor.get("logical_object_id")
        right_object = collider.get("logical_object_id") if collider else None
        if left_object and right_object and str(left_object) == str(right_object):
            internal_contacts.append(item)
            continue
        role_record = (
            role_index.get(actor_name(collider)) if collider is not None else None
        )
        collider_role = (
            role_record.get("role") if isinstance(role_record, Mapping) else None
        )
        classified = {**item, "collider_physics_role": collider_role}
        if collider_role == PHYSICS_ROLE_NON_SOLID:
            filtered_role_contacts.append(classified)
            continue
        external_contacts.append(classified)
    exact_contacts = [
        item for item in external_contacts
        if item["confirmed_overlap"]
        and item["valid"]
        and item["penetration_depth_cm"] is not None
        and item["depth_method"] == "ue_fhitresult_initial_overlap_mtd"
    ]
    violating_contacts = [
        item for item in exact_contacts
        if item["penetration_depth_cm"] > maximum_cm
    ]
    containment_contacts = [
        item for item in violating_contacts
        if item.get("collider_physics_role") == PHYSICS_ROLE_ENVIRONMENT_PROXY
    ]
    ordinary_contacts = [
        item for item in violating_contacts
        if item.get("collider_physics_role") != PHYSICS_ROLE_ENVIRONMENT_PROXY
    ]
    unresolved_contacts = []
    for item in external_contacts:
        if not item["confirmed_overlap"]:
            continue
        if not item["valid"]:
            reason = "invalid_penetration_depth"
        elif item["penetration_depth_cm"] is None:
            reason = "confirmed_overlap_has_no_penetration_depth"
        elif item["depth_method"] != "ue_fhitresult_initial_overlap_mtd":
            reason = "penetration_depth_is_not_native_ue_mtd"
        else:
            continue
        unresolved_contacts.append({**item, "reason": reason})
    violating = bool(violating_contacts)
    if exact_evaluated is not True and not violating:
        return None
    measured_contact_depths = [
        item["penetration_depth_cm"] for item in exact_contacts
    ]
    decisive_depth = max(measured_contact_depths) if measured_contact_depths else None
    basis = (["ue_collision_body_native_mtd"] if violating_contacts else [])
    if containment_contacts:
        basis.append("environment_proxy_containment")
    return {"violating": violating, "depth_cm": decisive_depth,
            "unresolved": unresolved_contacts,
            "filtered_internal_contacts": internal_contacts,
            "filtered_role_contacts": filtered_role_contacts,
            "environment_containment": bool(containment_contacts),
            "ordinary_penetration": bool(ordinary_contacts),
            "evidence": {"actor": actor_name(actor),
                         "actor_label": actor.get("label"),
                         "method": (measurement.get("solid_penetration_method")
                                    or measurement.get("measurement_method")
                                    or "trusted_physics_measurement"),
                         "maximum_native_mtd_cm": decisive_depth,
                         "penetrating_with": violating_contacts,
                         "ordinary_penetrations": ordinary_contacts,
                         "environment_containment_failures": containment_contacts,
                         "verdict_basis": basis} if violating else None}


def _broad_phase(actor: Mapping[str, Any], scene: Sequence[Mapping[str, Any]],
                 tolerance: float,
                 maximum_span: float | None) -> dict[str, list[dict[str, Any]]]:
    bounds = bounding_box(actor)
    if bounds is None:
        return {"candidates": [], "uncertain": [
            {"actor": actor_name(actor), "actor_label": actor.get("label"),
             "reason": "missing_bounds"}]}
    candidates = []
    for collider in scene:
        collider_key = actor_key(collider)
        if (collider is actor or collider_key == actor_key(actor)
                or is_non_solid(collider)):
            continue
        collider_bounds = bounding_box(collider)
        if collider_bounds is None:
            continue
        depths = overlap_depths(bounds, collider_bounds)
        if not all(value > tolerance for value in depths):
            continue
        candidates.append({
            "actor": actor_name(actor), "actor_label": actor.get("label"),
            "collider": actor_name(collider), "collider_label": collider.get("label"),
            "aabb_overlap_cm": [rounded(value, 3) for value in depths],
            "estimated_minimum_penetration_cm": rounded(min(depths) - tolerance, 3),
            "method": "world_aabb_broad_phase_candidate",
            "irregular_environment": is_irregular_environment(collider, maximum_span),
            "reason": "precise_ue_collision_confirmation_required"})
    return {"candidates": candidates, "uncertain": []}


def _checks(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
    selector = assertion.get("target_selector")
    selector = selector if isinstance(selector, Mapping) else {}
    population_scope = str(
        selector.get("scope") or assertion.get("scope") or assertions.DEFAULT_SCOPE
    )
    # A no-op repair introduces no local collision defect. It is neutral in
    # Physical Safety while GT-repair target/preservation scoring rejects the
    # missing edit. Other scopes retain the strict empty-population refusal.
    actors = (
        case.population(assertion)
        if population_scope == "edited_actors"
        else assertions.actors_or_unresolved(case, assertion)
    )
    maximum_cm = assertion.get("maximum_penetration_cm")
    maximum_cm = (DEFAULT_MAXIMUM_PENETRATION_CM if maximum_cm is None
                  else float(maximum_cm))
    adaptive_tolerance = assertion.get("adaptive_penetration_tolerance") is True
    relative_fraction = float(assertion.get(
        "relative_penetration_tolerance_fraction",
        DEFAULT_RELATIVE_PENETRATION_TOLERANCE_FRACTION,
    ))
    maximum_adaptive_cm = float(assertion.get(
        "maximum_adaptive_penetration_tolerance_cm",
        DEFAULT_MAXIMUM_ADAPTIVE_PENETRATION_TOLERANCE_CM,
    ))
    allowed = int(assertion.get("maximum_penetrating_actor_count") or 0)
    tolerance = assertion.get("aabb_touch_tolerance_cm")
    tolerance = (TOUCH_TOLERANCE_CM if tolerance is None else float(tolerance))
    maximum_span = assertion.get("maximum_decisive_aabb_span_cm")
    role_index = classify_physics_roles(case.candidate, assertion)
    role_counts = Counter(
        str(record.get("role")) for record in role_index.values()
    )
    filtered_targets: list[dict[str, Any]] = []
    actor_failure_penalties: list[float] = []
    environment_containment_actors: set[str] = set()
    ordinary_penetration_actors: set[str] = set()
    violations: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    depths: list[float] = []
    depth_penalties: list[float] = []
    effective_tolerances: list[float] = []
    trusted_count = 0
    # Use guarded identity matching: a label-first lookup with no ambiguity,
    # reuse or identity-conflict check
    # hands ONE record to several Actors, clears all of them, and RAISES the
    # score — a duplicate actor is credited by the record belonging to its twin.
    records = measurement_records(case.evidence.measurements)
    candidate_labels = Counter(
        str(candidate.get("label")) for candidate in case.candidate
        if candidate.get("label") is not None
    )
    collider_lookup: dict[str, Mapping[str, Any]] = {}
    for candidate in case.candidate:
        for field in ("actor_path", "stable_actor_id"):
            if candidate.get(field):
                collider_lookup[str(candidate[field])] = candidate
        label = candidate.get("label")
        if label is not None and candidate_labels[str(label)] == 1:
            collider_lookup[str(label)] = candidate
    label_counts = Counter(label for actor in actors
                           if (label := actor.get("label")) is not None)
    used_keys: set[str] = set()
    for actor in actors:
        role_record = role_index.get(actor_name(actor), {
            "role": PHYSICS_ROLE_UNRESOLVED,
            "reasons": ["actor_missing_from_physics_role_index"],
        })
        target_role = role_record.get("role")
        if target_role == PHYSICS_ROLE_UNRESOLVED:
            unresolved.append({
                "actor": actor_name(actor),
                "actor_label": actor.get("label"),
                "reason": "physics_role_unresolved",
                "role_reasons": role_record.get("reasons") or [],
            })
            continue
        if target_role in {
            PHYSICS_ROLE_NON_SOLID,
            PHYSICS_ROLE_SUPPORT_SURFACE,
            PHYSICS_ROLE_ENVIRONMENT_PROXY,
        }:
            filtered_targets.append({
                "actor": actor_name(actor),
                "actor_label": actor.get("label"),
                "physics_role": target_role,
                "role_reasons": role_record.get("reasons") or [],
            })
            continue
        record, _key, match_error = measurement_for_actor(
            actor, records, label_counts, used_keys)
        if match_error and match_error not in _NO_MEASUREMENT:
            # An AMBIGUOUS match is not the same as an absent one. Absent
            # falls back to the broad phase, which is honest. Ambiguous means
            # one record could be handed to two Actors — and handing it over
            # CLEARS both and raises the score, so a duplicated Actor is
            # credited by its twin's measurement.
            unresolved.append({"actor": actor_name(actor),
                               "actor_label": actor.get("label"),
                               "reason": match_error})
            continue
        if adaptive_tolerance:
            effective_tolerance_cm, characteristic_size = \
                _penetration_tolerance_cm(
                    actor,
                    maximum_cm,
                    relative_fraction,
                    maximum_adaptive_cm,
                )
        else:
            effective_tolerance_cm = maximum_cm
            characteristic_size = _actor_characteristic_size_cm(actor)
        verdict = _trusted(
            actor, record, effective_tolerance_cm, collider_lookup, role_index
        )
        if verdict is not None:
            trusted_count += 1
            effective_tolerances.append(effective_tolerance_cm)
            depth_penalty, normalized_excess, _ = _penetration_depth_penalty(
                actor, verdict["depth_cm"], effective_tolerance_cm
            )
            depth_penalties.append(depth_penalty)
            actor_failure_penalties.append(
                1.0 if verdict["violating"] else 0.0
            )
            if verdict["environment_containment"]:
                environment_containment_actors.add(actor_name(actor))
            if verdict["ordinary_penetration"]:
                ordinary_penetration_actors.add(actor_name(actor))
            if verdict["depth_cm"] is not None:
                depths.append(verdict["depth_cm"])
            if verdict["violating"]:
                violations.append({
                    **verdict["evidence"],
                    "target_physics_role": target_role,
                    "actor_failure_penalty": 1.0,
                    "characteristic_size_cm": characteristic_size,
                    "effective_penetration_tolerance_cm":
                        effective_tolerance_cm,
                    "penetration_tolerance_policy": (
                        ADAPTIVE_TOLERANCE_POLICY
                        if adaptive_tolerance else "fixed_declared_tolerance"
                    ),
                    "normalized_excess_penetration": normalized_excess,
                    "penetration_depth_penalty": depth_penalty,
                })
            if verdict["unresolved"]:
                unresolved.append({
                    "actor": actor_name(actor),
                    "actor_label": actor.get("label"),
                    "reason": "confirmed_collision_overlap_depth_unavailable",
                    "contacts": verdict["unresolved"],
                })
            for contact in verdict["filtered_internal_contacts"]:
                filtered.append({
                    "actor": actor_name(actor),
                    "actor_label": actor.get("label"),
                    "collider": contact.get("collider"),
                    "reason": "same_logical_object_contact_filtered",
                })
            for contact in verdict["filtered_role_contacts"]:
                filtered.append({
                    "actor": actor_name(actor),
                    "actor_label": actor.get("label"),
                    "collider": contact.get("collider"),
                    "reason": "non_solid_collider_contact_filtered",
                })
            continue
        fallback = _broad_phase(actor, case.candidate, tolerance, maximum_span)
        broad_phase = fallback["candidates"] or fallback["uncertain"]
        if broad_phase:
            unresolved.extend(broad_phase)
        else:
            # Broad-phase clearance is useful diagnostic evidence but cannot
            # replace collision-body measurement.  Treating it as a pass made
            # candidate_all scenes look perfectly clean whenever the probe
            # omitted an Actor.
            unresolved.append({
                "actor": actor_name(actor),
                "actor_label": actor.get("label"),
                "reason": "trusted_collision_measurement_missing",
                "method": "world_aabb_broad_phase_clear_not_decisive",
            })

    penetrating = len(names(violations))
    unresolved_actors = names(unresolved)
    broad_phase_candidates = sum(
        item.get("method") == "world_aabb_broad_phase_candidate"
        for item in unresolved)
    depth_penalty_sum = sum(depth_penalties)
    mean_depth_penalty = (
        depth_penalty_sum / trusted_count if trusted_count else None)
    weighted_score = (
        max(0.0, 1.0 - mean_depth_penalty)
        if mean_depth_penalty is not None else None)
    actor_failure_penalty_sum = sum(actor_failure_penalties)
    non_penetration_rate = (
        max(0.0, 1.0 - actor_failure_penalty_sum / trusted_count)
        if trusted_count else None
    )
    filtered_target_counts = Counter(
        str(item.get("physics_role")) for item in filtered_targets
    )
    observed = {
        "population_scope": population_scope,
        "target_actor_count": len(actors),
        "evaluated_actor_count": trusted_count,
        "trusted_measurement_actor_count": trusted_count,
        "broad_phase_candidate_count": broad_phase_candidates,
        "physics_role_policy": ROLE_POLICY,
        "scene_physics_role_counts": dict(sorted(role_counts.items())),
        "eligible_target_actor_count": sum(
            role_index.get(actor_name(actor), {}).get("role")
            == PHYSICS_ROLE_SCORED_SOLID for actor in actors
        ),
        "filtered_non_solid_actor_count": filtered_target_counts.get(
            PHYSICS_ROLE_NON_SOLID, 0),
        "filtered_support_surface_actor_count": filtered_target_counts.get(
            PHYSICS_ROLE_SUPPORT_SURFACE, 0),
        "filtered_environment_proxy_actor_count": filtered_target_counts.get(
            PHYSICS_ROLE_ENVIRONMENT_PROXY, 0),
        "filtered_contact_count": len(filtered),
        "environment_containment_actor_count": len(environment_containment_actors),
        "ordinary_penetration_actor_count": len(ordinary_penetration_actors),
        "penetrating_actor_count": penetrating,
        "maximum_observed_penetration_cm": max(depths) if depths else None,
        "adaptive_penetration_tolerance": adaptive_tolerance,
        "penetration_tolerance_policy": (
            ADAPTIVE_TOLERANCE_POLICY
            if adaptive_tolerance else "fixed_declared_tolerance"
        ),
        "minimum_effective_penetration_tolerance_cm": (
            min(effective_tolerances) if effective_tolerances else None
        ),
        "maximum_effective_penetration_tolerance_cm": (
            max(effective_tolerances) if effective_tolerances else None
        ),
        "penetration_depth_weighting": DEPTH_WEIGHTING,
        "penetration_depth_penalty_sum": depth_penalty_sum,
        "mean_penetration_depth_penalty": mean_depth_penalty,
        "depth_weighted_score": weighted_score,
        "scoring_policy": SCORING_POLICY,
        "actor_failure_penalty_sum": actor_failure_penalty_sum,
        "mean_actor_failure_penalty": (
            actor_failure_penalty_sum / trusted_count if trusted_count else None
        ),
        "non_penetration_rate": non_penetration_rate,
        "unresolved_actor_count": len(unresolved_actors)}
    expected = {"maximum_penetrating_actor_count": allowed,
                "population_scope": population_scope,
                "minimum_penetration_tolerance_cm": maximum_cm,
                "relative_penetration_tolerance_fraction": (
                    relative_fraction if adaptive_tolerance else None),
                "maximum_penetration_tolerance_cm": (
                    maximum_adaptive_cm if adaptive_tolerance else maximum_cm),
                "adaptive_penetration_tolerance": adaptive_tolerance,
                "penetration_depth_weighting": DEPTH_WEIGHTING,
                "scoring_policy": SCORING_POLICY,
                "physics_role_policy": ROLE_POLICY}
    # Preserve the former conservative evidence rule: incomplete evidence can
    # contribute a score only when a trusted contact already proves non-zero
    # penetration. It must never manufacture a clean score from partial data.
    if unresolved and not violations:
        return [Check(id="physics.solid_penetration", status="not_evaluated",
                      expected=expected, observed=observed, evidence=unresolved,
                      failure_reason=(
                          "collision-body evidence is incomplete for "
                          f"{len(unresolved_actors)} Actor(s); an overlap without "
                          "penetration depth cannot be compared with the declared "
                          "depth tolerance, and an AABB overlap is not a penetration"))]
    if actors and not observed["evaluated_actor_count"]:
        # Infrastructure and support geometry are collider roles, not scored
        # target populations.  An empty eligible population is inapplicable,
        # never a perfect score.
        return [assertions.inapplicable(
            "physics.solid_penetration",
            f"all {len(actors)} selected Actor(s) were excluded from the scored "
            f"solid target population by {ROLE_POLICY}",
            observed)]
    return [Check(id="physics.solid_penetration", status="measured",
                  expected=expected, observed=observed, evidence=violations)]


def _score(checks: Sequence[Check]) -> float:
    # Only measured checks carry a continuous penalty; not-applicable checks do
    # not enter the denominator.
    decided = [item for item in checks if item.status == "measured"]
    actor_count = sum(
        int(item.observed.get("evaluated_actor_count") or 0)
        for item in decided)
    penalty_sum = sum(
        float(item.observed.get("actor_failure_penalty_sum") or 0.0)
        for item in decided)
    return 1.0 if not actor_count else max(
        0.0, 1.0 - penalty_sum / actor_count)


def verify(context: Context) -> dict[str, Any]:
    return assertions.run("solid_penetration", context, "solid_penetration",
                          "eligible Actor collision-free rate",
                          _checks, _score)


__all__ = [
    "ADAPTIVE_TOLERANCE_POLICY",
    "DEFAULT_MAXIMUM_ADAPTIVE_PENETRATION_TOLERANCE_CM",
    "DEFAULT_MAXIMUM_PENETRATION_CM",
    "DEFAULT_RELATIVE_PENETRATION_TOLERANCE_FRACTION",
    "DEPTH_PENALTY_LAMBDA",
    "DEPTH_WEIGHTING",
    "MIN_CHARACTERISTIC_SIZE_CM",
    "verify",
    "ROLE_POLICY",
    "SCORING_POLICY",
]
