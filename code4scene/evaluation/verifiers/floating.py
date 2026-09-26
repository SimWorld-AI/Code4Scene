"""``floating`` — the measured share of the selected Actors resting on nothing.

Text-to-scene measures the scene-wide 5 cm support rule (metric
``t2s-aabb-floating-contact-5cm.v2``) on the candidate snapshot the scorer
exported, so the recorded leaf equals an offline recomputation from that
snapshot. Only when no snapshot is available does it fall back to the
harness's in-editor measurement record (same rule, live bounds). Image-to-scene repair
policies may instead declare ``edited_actors``: those Actors are resolved from
Input -> Candidate and use the targeted independent UE physics records already
shared with solid-penetration scoring. The public result remains the defect
rate itself (lower is better), not ``1 - floating_rate``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ... import protocol
from .. import assertions, contracts, ue_evidence
from ..assertions import Case, Check
from ..context import Context, error
from ..physics_evidence import measurement_for_actor, measurement_records
from ..scene_rate import rate_report
from ..scene_semantics import (
    ground_gap_support_surface_evidence,
    is_non_solid_ground_gap_target,
)
from ..selection import actor_identifier
from ..ue_evidence import semantic_contract
from ..values import as_text


CLASS = "open_ended"

LOCAL_SCOPE = "edited_actors"
LOCAL_APPLICABILITY_FILTER = (
    "scene_semantics.i2s_local_ground_gap_applicability_v3"
)
GROUND_ONLY_SUPPORT_MODEL = "ground_only_v1"
LATERAL_SUPPORT_MODEL = "ground_or_lateral_v1"


def _declared_scope(context: Context) -> str:
    contract = semantic_contract(context)
    if not isinstance(contract, Mapping):
        return "candidate_all"
    for assertion in contract.get("assertions") or ():
        if not isinstance(assertion, Mapping) or assertion.get("primitive") != "physics":
            continue
        selector = assertion.get("target_selector")
        selector = selector if isinstance(selector, Mapping) else {}
        return (
            as_text(selector.get("scope") or assertion.get("scope"))
            or "primary_additions"
        )
    return "candidate_all"


def _local_checks(
    case: Case,
    assertion: Mapping[str, Any],
    *,
    population_audit: dict[str, Any] | None = None,
) -> list[Check]:
    # Empty means this Candidate introduced no Actor whose physical placement
    # can be judged. It is neutral here rather than a perfect whole-scene
    # claim; GT-repair preservation/target scoring still judges the no-op.
    selected = case.population(assertion, LOCAL_SCOPE)
    actors: list[Mapping[str, Any]] = []
    filtered_non_solid: list[Mapping[str, Any]] = []
    filtered_support: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for actor in selected:
        if is_non_solid_ground_gap_target(actor):
            filtered_non_solid.append(actor)
            continue
        support_evidence = ground_gap_support_surface_evidence(actor)
        if support_evidence is not None:
            filtered_support.append((actor, support_evidence))
            continue
        actors.append(actor)

    if population_audit is not None:
        population_audit["resolved_assertion_count"] += 1
        for bucket, population in (
            ("selected", selected),
            ("scored", actors),
            ("filtered_non_solid", filtered_non_solid),
        ):
            recorded = population_audit[bucket]
            for index, actor in enumerate(population):
                actor_id = actor_identifier(actor, index)
                recorded[actor_id] = {
                    "actor_id": actor_id,
                    "actor_label": actor.get("label"),
                    "actor_class": actor.get("class"),
                    "asset_path": actor.get("asset_path"),
                }
        recorded_support = population_audit["filtered_support_surface"]
        for index, (actor, support_evidence) in enumerate(filtered_support):
            actor_id = actor_identifier(actor, index)
            recorded_support[actor_id] = {
                "actor_id": actor_id,
                "actor_label": actor.get("label"),
                "actor_class": actor.get("class"),
                "asset_path": actor.get("asset_path"),
                **support_evidence,
            }

    if not actors:
        return [assertions.check(
            "physics.local_edit_floating",
            True,
            {"population_scope": LOCAL_SCOPE},
            {
                "target_actor_count": 0,
                "floating_actor_count": 0,
                "selected_actor_count_before_applicability_filter": len(selected),
                "selected_actor_count_before_non_solid_filter": len(selected),
                "filtered_non_solid_actor_count": len(filtered_non_solid),
                "filtered_support_surface_actor_count": len(filtered_support),
                "empty_scope_policy": "neutral_no_introduced_physics",
            },
        )]

    records = measurement_records(case.evidence.measurements)
    label_counts = Counter(
        str(actor.get("label")) for actor in actors
        if actor.get("label") is not None
    )
    used_keys: set[str] = set()
    checks: list[Check] = []
    for index, actor in enumerate(actors):
        actor_id = actor_identifier(actor, index)
        measurement, measurement_key, match_error = measurement_for_actor(
            actor,
            records,
            label_counts,
            used_keys,
        )
        check_id = f"physics.local_edit_floating:{actor_id}"
        if match_error or measurement is None:
            checks.append(assertions.unevaluated(
                check_id,
                f"targeted UE physics evidence unavailable: "
                f"{match_error or 'measurement_missing'}",
                {"actor_id": actor_id, "population_scope": LOCAL_SCOPE},
            ))
            continue
        support_model = assertion.get("support_model") or GROUND_ONLY_SUPPORT_MODEL
        verdict_field = (
            "supported"
            if support_model == LATERAL_SUPPORT_MODEL
            else "grounded"
        )
        supported = measurement.get(verdict_field)
        if not isinstance(supported, bool):
            checks.append(assertions.unevaluated(
                check_id,
                "targeted UE physics evidence has no boolean "
                f"{verdict_field} verdict for {support_model}",
                {
                    "actor_id": actor_id,
                    "measurement_key": measurement_key,
                    "population_scope": LOCAL_SCOPE,
                    "support_model": support_model,
                },
            ))
            continue
        checks.append(assertions.check(
            check_id,
            supported,
            {
                verdict_field: True,
                "population_scope": LOCAL_SCOPE,
                "support_model": support_model,
            },
            {
                "actor_id": actor_id,
                "actor_label": actor.get("label"),
                "grounded": measurement.get("grounded"),
                "supported": measurement.get("supported"),
                "support_mode": measurement.get("support_mode"),
                "support_model": support_model,
                "surface_detected": measurement.get("surface_detected"),
                "ground_gap_cm": measurement.get("ground_gap_cm"),
                "lateral_support_detected": measurement.get(
                    "lateral_support_detected"
                ),
                "lateral_support_distance_cm": measurement.get(
                    "lateral_support_distance_cm"
                ),
                "lateral_support_fraction": measurement.get(
                    "lateral_support_fraction"
                ),
                "lateral_trace_count": measurement.get(
                    "lateral_trace_count"
                ),
                "lateral_supporting_colliders": measurement.get(
                    "lateral_supporting_colliders"
                ),
                "measurement_method": measurement.get("measurement_method"),
                "measurement_key": measurement_key,
            },
            failure_reason="edited Actor rests on no support",
        ))
    return checks


def _floating_rate(checks: Sequence[Check]) -> float:
    decided = [
        item for item in checks
        if item.status in {contracts.PASS, contracts.FAIL}
    ]
    return (
        sum(item.status == contracts.FAIL for item in decided) / len(decided)
        if decided else 0.0
    )


def _local_report(context: Context) -> dict[str, Any]:
    population_audit: dict[str, Any] = {
        "resolved_assertion_count": 0,
        "selected": {},
        "scored": {},
        "filtered_non_solid": {},
        "filtered_support_surface": {},
    }

    def produce(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
        return _local_checks(
            case,
            assertion,
            population_audit=population_audit,
        )

    report = assertions.run(
        "floating",
        context,
        "physics",
        "share of edited Actors with no support beneath them",
        produce,
        _floating_rate,
    )
    if population_audit["resolved_assertion_count"]:
        selected = population_audit["selected"]
        scored = population_audit["scored"]
        filtered_non_solid = population_audit["filtered_non_solid"]
        filtered_support = population_audit["filtered_support_surface"]
        report.setdefault("metrics", {}).update(
            {
                "selected_actor_count_before_applicability_filter": len(selected),
                "selected_actor_count_before_non_solid_filter": len(selected),
                "scored_actor_count_after_applicability_filter": len(scored),
                "scored_actor_count_after_non_solid_filter": len(scored),
                "filtered_non_solid_actor_count": len(filtered_non_solid),
                "filtered_support_surface_actor_count": len(filtered_support),
            }
        )
        report.setdefault("evidence", {}).update(
            {
                "applicability_filter": LOCAL_APPLICABILITY_FILTER,
                "non_solid_filter": (
                    "scene_semantics.is_non_solid_ground_gap_target"
                ),
                "support_surface_filter": (
                    "scene_semantics.is_ground_support_ground_gap_target"
                ),
                "filtered_non_solid_actors": [
                    filtered_non_solid[key]
                    for key in sorted(filtered_non_solid)
                ],
                "filtered_support_surface_actors": [
                    filtered_support[key]
                    for key in sorted(filtered_support)
                ],
            }
        )
    if report.get("score") is not None:
        report["metadata"] = {
            "score_direction": "lower_is_better",
            "result_semantics": "direct_measured_rate",
        }
    report.setdefault("evidence", {})["population_scope"] = LOCAL_SCOPE
    report["evidence"]["global_coverage_owner"] = "gt_repair.scene_diff"
    return report


def snapshot_leaf(
    candidate: Mapping[str, Any],
    ids: Mapping[str, str],
    *,
    live_rate: float | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """The text-to-scene floating leaf measured on a candidate scene snapshot.

    Returns ``None`` when the snapshot has no eligible actor (the caller then
    keeps the in-editor measurement). The same builder serves the live
    verifier and offline rescoring, so both publish the identical leaf.
    """

    measured = protocol.physics.t2s_floating_rate(candidate)
    if measured["rate"] is None:
        return None
    return {
        **contracts.base("floating", dict(ids)),
        "status": contracts.MEASURED,
        "score": measured["rate"],
        "metrics": {"actors": measured["actor_count"], "floating_rate": measured["rate"],
                    "floating_count": measured["floating_count"],
                    "floating_labels": measured["floating_labels"][:500]},
        "evidence": {"metric_id": measured["metric"],
                     "source": "candidate_scene_snapshot",
                     "dimension": "share of Actors with no support beneath them",
                     "publication": "direct_measured_rate_without_pass_fail_threshold",
                     "live_measure_floating_rate": live_rate},
        "metadata": {"score_direction": "lower_is_better",
                     "result_semantics": "direct_measured_rate"},
        "artifacts": dict(artifacts or {}),
        "probes_used": ("candidate_scene_snapshot",),
    }


def _snapshot_report(context: Context) -> dict[str, Any] | None:
    """Scene-wide rate from the exported candidate snapshot, when there is one."""

    try:
        candidate = ue_evidence.collect(context).candidate
        return snapshot_leaf(
            candidate, context.ids,
            live_rate=(context.record.get("metrics") or {}).get("floating_rate"),
            artifacts=contracts.artifacts(context.record))
    except Exception:  # noqa: BLE001 - no usable snapshot: use the measure record
        return None


def verify(context: Context) -> dict[str, Any]:
    try:
        if _declared_scope(context) == LOCAL_SCOPE:
            return _local_report(context)
    except Exception as exc:  # noqa: BLE001 - missing local evidence is visible
        return error("floating", context, f"{type(exc).__name__}: {exc}")
    snapshot = _snapshot_report(context)
    if snapshot is not None:
        return snapshot
    return rate_report(
        "floating",
        context,
        rate_key="floating_rate",
        dimension="share of Actors with no support beneath them",
        publish_direct_rate=True,
    )


__all__ = ["LOCAL_APPLICABILITY_FILTER", "LOCAL_SCOPE", "snapshot_leaf", "verify"]
