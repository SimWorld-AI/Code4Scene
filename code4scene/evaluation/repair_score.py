"""GT repair with measured visual scores and structural fallback when absent."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from . import composite, contracts, missing_score


POLICY_ID = "gt-repair-arithmetic-indoor-structured-global"
LOCAL_POLICY_ID = "gt-repair-local-optional-visual"
GLOBAL_POLICY_ID = "gt-repair-global-environment-routed"
VISUAL_MISSING_POLICY_ID = "visual-missing-exclude-renormalize.v1"
LOCAL_RULE_WEIGHTS = {
    "target_recovery": 0.125,
    "target_identity": 0.125,
    "target_transform_diff": 0.34375,
    "target_geometry_diff": 0.21875,
    "target_attribute_diff": 0.1875,
}
LOCAL_WEIGHTS = {
    "target_recovery": 0.10,
    "target_identity": 0.10,
    "target_transform_diff": 0.275,
    "target_geometry_diff": 0.175,
    "target_attribute_diff": 0.15,
    "target_visual_diff": 0.20,
}
GLOBAL_WEIGHTS = {"structured_scene_diff": 0.5, "visual_semantic_diff": 0.5}
PREREQUISITES = ("target_recovery", "target_identity")
FINAL_WEIGHTS = {"repair_target_diff": 0.8, "scene_diff": 0.2}


def _select_visual(report: dict[str, Any], *, enabled: bool = True) -> None:
    used = enabled and missing_score.observed_score(report) is not None
    report.update(contributes_to_aggregate=used, aggregate_weight=0.0)
    report["score_role"] = "optional_visual_component" if enabled else "report_only"
    if used:
        report.pop("report_only_reason", None)
    else:
        report["report_only_reason"] = (
            "visual score unavailable; use same-head structural weights"
            if enabled else "pure removal uses recovery directly"
        )


def _leaves(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return (report.get("metrics") or {}).get("leaf_results") or []


def _refresh(report: dict[str, Any]) -> dict[str, Any]:
    """Refresh score and aggregate status after changing child participation."""
    metrics = report.setdefault("metrics", {})
    leaves = _leaves(report)
    weights = metrics["configured_score_weights"]
    optional = set(metrics.get("optional_score_leaf_ids") or [])
    active = [
        child for child in leaves
        if child.get("contributes_to_aggregate") is not False
        and weights.get(child.get("leaf_id"), 0) > 0
    ]
    report["status"] = composite._status(active)
    present = {child.get("leaf_id") for child in leaves}
    absent = [name for name, weight in weights.items()
              if weight > 0 and name not in present and name not in optional]
    if absent and report["status"] == "not_applicable":
        # Missing required rules are unknown evidence, not explicit N/A. In
        # particular, a visual-only head must not disappear from the GT mean.
        report["status"] = "not_evaluated"
    metrics["report_only_leaf_ids"] = [
        child["leaf_id"] for child in leaves if child.get("contributes_to_aggregate") is False
    ]
    projection = missing_score.project(
        report, path=str(report.get("report_id") or "gt_repair"), recompute=True,
    )
    score = projection["score"]
    report["score"] = (
        score if report["status"] in {contracts.MEASURED, contracts.PASS, contracts.FAIL}
        else None
    )
    reason = composite._reason(active, report["status"])
    if absent and report["status"] == "not_evaluated":
        reason = "required scoring leaves missing: " + ", ".join(absent)
    if reason:
        report["failure_reason"] = reason
    else:
        report.pop("failure_reason", None)
    metrics.update(
        report_only_leaf_ids=[
            child["leaf_id"] for child in leaves
            if child.get("contributes_to_aggregate") is False
        ],
        score_vector={child["leaf_id"]: child.get("score") for child in leaves},
        aggregate_score_vector={child["leaf_id"]: child.get("score") for child in active},
        normalized_aggregate_score_vector={
            child["leaf_id"]: child.get("score") for child in active
        },
        required_score_leaf_ids=[
            name for name in weights
            if name not in optional
            and not any(child.get("leaf_id") == name and child.get("status") == "not_applicable"
                       for child in active)
        ],
        effective_score_weights=projection.get("effective_weights", {}),
        score_coverage=projection["known_coverage"],
        missing_evidence=projection["missing_evidence"],
        incomplete_leaf_ids=[
            child["leaf_id"] for child in active
            if child.get("status") not in {"measured", "pass", "fail", "not_applicable"}
        ],
        all_required_scores_present=not projection["missing_evidence"],
        scored_leaf_count=sum(child.get("score") is not None for child in active),
        applicable_quality_leaf_count=sum(child.get("status") != "not_applicable" for child in active),
    )
    return report


def _visual_coverage(report: dict[str, Any], leaf_id: str) -> dict[str, Any]:
    visual = next((child for child in _leaves(report) if child.get("leaf_id") == leaf_id), {})
    metrics = report.get("metrics") or {}
    score = missing_score.observed_score(visual)
    configured = (metrics.get("configured_score_weights") or {}).get(leaf_id, 0.0)
    effective = (metrics.get("effective_score_weights") or {}).get(leaf_id, 0.0)
    if visual:
        visual["aggregate_weight"] = effective
    fallback = configured > 0 and score is None
    return {
        "status": visual.get("status", "missing"), "score": score,
        "available": score is not None, "used": effective > 0,
        "configured_weight": configured, "effective_weight": effective,
        "fallback_used": fallback,
        "fallback_reason": (
            visual.get("failure_reason") or ("no valid visual score" if visual else "visual report missing")
        ) if fallback else None,
    }


def _pure_removal(local: Mapping[str, Any]) -> bool:
    metrics = local.get("metrics") or {}
    if metrics.get("direct_score_leaf_id") == "target_recovery":
        return True
    counts = ((local.get("evidence") or {}).get("target_derivation") or {}).get("operation_counts") or {}
    if counts:
        return counts.get("remove", 0) > 0 and counts.get("add", 0) == counts.get("repair", 0) == 0
    recovery = next((child for child in _leaves(local) if child.get("leaf_id") == "target_recovery"), {})
    values = recovery.get("metrics") or {}
    return values.get("desired_actor_count") == 0 and (values.get("removal_target_count") or 0) > 0


def apply_local(local: dict[str, Any], *, pure_removal: bool | None = None) -> dict[str, Any]:
    if not _leaves(local):
        if local.get("score") is not None and (
            local.get("metrics") or {}
        ).get("score_weight_policy_id") != LOCAL_POLICY_ID:
            raise ValueError("gt_repair_local_rule_subscores_missing")
        return local
    pure_removal = _pure_removal(local) if pure_removal is None else pure_removal
    weights = {"target_recovery": 1.0} if pure_removal else dict(LOCAL_WEIGHTS)
    for child in _leaves(local):
        if child.get("leaf_id") == "target_visual_diff":
            _select_visual(child, enabled=not pure_removal)
    metrics = local.setdefault("metrics", {})
    metrics.update(
        configured_score_weights=weights,
        optional_score_leaf_ids=[] if pure_removal else ["target_visual_diff"],
        score_weight_groups={} if pure_removal else {
            "rules": {"weight": 0.8, "leaf_ids": list(LOCAL_RULE_WEIGHTS)},
            "visual": {"weight": 0.2, "leaf_ids": ["target_visual_diff"]},
        },
        score_weight_policy_id=LOCAL_POLICY_ID,
        score_aggregation=(
            "direct_measured_leaf_score" if pure_removal
            else "prerequisite_gate_times_weighted_mean_of_available_normalized_leaf_scores"
        ),
        prerequisite_leaf_ids=[] if pure_removal else list(PREREQUISITES),
        direct_score_leaf_id="target_recovery" if pure_removal else None,
        prerequisite_selection=(
            "pure_removal_uses_recovery_directly" if pure_removal
            else "desired_targets_require_recovery_and_identity"
        ),
    )
    _refresh(local)
    values = {child["leaf_id"]: missing_score.observed_score(child) for child in _leaves(local)}
    gate = 1.0 if pure_removal else min(values.get(name) or 0.0 for name in PREREQUISITES)
    effective = metrics["effective_score_weights"]
    contributions = {name: (values.get(name) or 0.0) * weight for name, weight in effective.items()}
    metrics.update(
        prerequisite_gate=gate,
        prerequisite_values={name: values.get(name) for name in metrics["prerequisite_leaf_ids"]},
        weighted_score_contributions=contributions,
        weighted_quality_mix=round(sum(contributions.values()), 6),
        visual_coverage=_visual_coverage(local, "target_visual_diff"),
    )
    local.setdefault("evidence", {}).update(
        score_weight_policy_id=LOCAL_POLICY_ID,
        configured_score_weights=weights,
        unavailable_leaf_policy="zero_for_missing_required_rule; renormalize_explicit_NA",
        visual_score_role="optional_visual_component",
        missing_visual_policy=VISUAL_MISSING_POLICY_ID,
        prerequisite_selection=metrics["prerequisite_selection"],
    )
    return local


def apply_global(scene: dict[str, Any], *, scene_environment: str | None = None) -> dict[str, Any]:
    indoor = scene_environment == "indoor" or (
        scene_environment is None and (scene.get("evidence") or {}).get("whole_scene_visual") == "omitted"
    )
    if not _leaves(scene):
        if scene.get("score") is not None and (
            (scene.get("metrics") or {}).get("score_weight_policy_id") != GLOBAL_POLICY_ID
            or (indoor and (scene.get("evidence") or {}).get("whole_scene_visual") != "omitted")
        ):
            raise ValueError("gt_repair_global_structured_subscore_missing")
        return scene
    for child in _leaves(scene):
        if child.get("leaf_id") == "visual_semantic_diff":
            _select_visual(child, enabled=not indoor)
            if indoor:
                child["report_only_reason"] = "indoor precision uses global structure only"
    scene.setdefault("metrics", {}).update(
        configured_score_weights={"structured_scene_diff": 1.0} if indoor else dict(GLOBAL_WEIGHTS),
        optional_score_leaf_ids=[] if indoor else ["visual_semantic_diff"],
        score_weight_groups={},
        score_weight_policy_id=GLOBAL_POLICY_ID,
        score_aggregation="structured_only" if indoor else "weighted_mean_with_optional_visual_fallback",
        direct_score_leaf_id=None,
        prerequisite_leaf_ids=[],
    )
    scene.setdefault("evidence", {}).update(
        score_weight_policy_id=GLOBAL_POLICY_ID,
        visual_score_role="optional_visual_component",
        missing_visual_policy=VISUAL_MISSING_POLICY_ID,
    )
    scene["evidence"]["whole_scene_visual"] = "omitted" if indoor else "score_when_available"
    _refresh(scene)
    scene["metrics"]["visual_coverage"] = _visual_coverage(scene, "visual_semantic_diff")
    return scene


#: What the gt_repair report publishes as its score (paper Appendix C.5).
PUBLISHED_SCORE = "actor_repair_f1"
LEGACY_DIAGNOSTIC_KEY = "legacy_gt_repair_composite"


def publish_actor_f1(report: dict[str, Any]) -> dict[str, Any]:
    """Make Actor Repair F1 the gt_repair report's score (in place).

    The report's leaves (repair-target diff, scene diff, locality audit) and
    their weighted composite remain as diagnostics: the composite value is kept
    only under ``metrics.diagnostics.legacy_gt_repair_composite`` and never
    enters a benchmark score. The published score is the paper's Repair F1,
    measured by :mod:`code4scene.protocol.actor_f1` and recorded under
    ``metrics.actor_repair_f1``. A report without that measurement (a result
    produced before it existed) is ``not_evaluated`` rather than silently
    falling back to the composite.
    """

    metrics = report.setdefault("metrics", {})
    diagnostics = dict(metrics.get("diagnostics") or {})
    if metrics.get("published_score") == PUBLISHED_SCORE and LEGACY_DIAGNOSTIC_KEY in diagnostics:
        # Already published and not recomputed since: the report's score holds
        # the F1, so restore the recorded composite before re-publishing.
        report["score"] = diagnostics[LEGACY_DIAGNOSTIC_KEY]
        report["status"] = diagnostics.get(LEGACY_DIAGNOSTIC_KEY + "_status", report.get("status"))
        reason = diagnostics.get(LEGACY_DIAGNOSTIC_KEY + "_failure_reason")
        if reason:
            report["failure_reason"] = reason
    diagnostics.update({
        LEGACY_DIAGNOSTIC_KEY: report.get("score"),
        LEGACY_DIAGNOSTIC_KEY + "_status": report.get("status"),
        LEGACY_DIAGNOSTIC_KEY + "_failure_reason": report.get("failure_reason"),
        LEGACY_DIAGNOSTIC_KEY + "_policy_id": POLICY_ID,
        "role": "diagnostic_not_scored",
    })
    metrics["diagnostics"] = diagnostics
    metrics["published_score"] = PUBLISHED_SCORE
    measured = metrics.get(PUBLISHED_SCORE)
    measured = measured if isinstance(measured, Mapping) else None
    f1 = (measured or {}).get("f1")
    if measured is not None and measured.get("status") == contracts.MEASURED and (
        isinstance(f1, (int, float)) and not isinstance(f1, bool) and 0.0 <= f1 <= 1.0
    ):
        report["status"] = contracts.MEASURED
        report["score"] = float(f1)
        report.pop("failure_reason", None)
    elif measured is None:
        report["status"] = "not_evaluated"
        report["score"] = None
        report["failure_reason"] = (
            "Actor Repair F1 was not recorded for this result; rescore it from the "
            "input, ground-truth and candidate scene snapshots"
        )
    else:
        report["status"] = contracts.ERROR
        report["score"] = None
        report["failure_reason"] = str(
            measured.get("failure_reason") or "Actor Repair F1 could not be measured"
        )
    return report


def carry_actor_f1(new_root: dict[str, Any], old_root: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a frozen Actor Repair F1 when a diagnostic leaf is recomputed.

    Recomputing a visual or caption leaf changes only the diagnostic composite;
    the published F1 is measured from the scene snapshots and is carried over.
    """

    measured = (old_root.get("metrics") or {}).get(PUBLISHED_SCORE)
    if isinstance(measured, Mapping):
        new_root.setdefault("metrics", {})[PUBLISHED_SCORE] = deepcopy(dict(measured))
    return new_root


def apply(report: Mapping[str, Any], *, scene_environment: str | None = None) -> dict[str, Any]:
    """Recompute both heads; never mislabel a legacy scalar as the current policy.

    The result carries the legacy weighted composite as its score; callers that
    publish a report apply :func:`publish_actor_f1` afterwards.
    """
    result = deepcopy(dict(report))
    routing = (result.get("evidence") or {}).get("visual_evidence_routing") or {}
    if scene_environment is None and routing.get("whole_scene_visual") == "omitted":
        scene_environment = "indoor"
    leaves = _leaves(result)
    if not leaves:
        if (result.get("metrics") or {}).get("published_score") == PUBLISHED_SCORE:
            return result
        if result.get("score") is not None and (
            result.get("metrics") or {}
        ).get("score_weight_policy_id") != POLICY_ID:
            raise ValueError("gt_repair_rule_subscores_missing")
        return result
    for child in leaves:
        name = child.get("leaf_id")
        if name == "repair_target_diff":
            apply_local(child)
        elif name == "scene_diff":
            apply_global(child, scene_environment=scene_environment)
        elif name == "locality_audit":
            child["contributes_to_aggregate"] = False
    metrics = result.setdefault("metrics", {})
    metrics.update(
        configured_score_weights=dict(FINAL_WEIGHTS),
        score_weight_policy_id=POLICY_ID,
        score_aggregation="weighted_arithmetic_mean(repair_target_diff, scene_diff)",
        optional_score_leaf_ids=[],
        score_weight_groups={},
        prerequisite_leaf_ids=[],
    )
    result.setdefault("evidence", {}).update(
        final_score_weight_policy_id=POLICY_ID,
        final_score_weights=dict(FINAL_WEIGHTS),
        final_score_formula="0.8 * repair_target_diff + 0.2 * scene_diff",
        visual_score_role="optional_visual_component",
        missing_visual_policy=VISUAL_MISSING_POLICY_ID,
    )
    if scene_environment in {"indoor", "outdoor"}:
        result["evidence"]["visual_evidence_routing"] = {
            **routing, "repair_target_visual": "score_when_available",
            "whole_scene_visual": "omitted" if scene_environment == "indoor" else "score_when_available",
            "whole_scene_structured_diff": "required",
        }
    _refresh(result)
    # The score now holds the freshly recomputed composite, not a published F1.
    metrics.pop("published_score", None)
    heads = {child["leaf_id"]: child.get("score") for child in leaves}
    metrics.update(
        repair_target_score=heads.get("repair_target_diff"),
        scene_diff_score=heads.get("scene_diff"),
        visual_coverage={
            scope: deepcopy((child.get("metrics") or {}).get("visual_coverage"))
            for scope, name in (("local", "repair_target_diff"), ("global", "scene_diff"))
            for child in leaves if child.get("leaf_id") == name
        },
    )
    return result
