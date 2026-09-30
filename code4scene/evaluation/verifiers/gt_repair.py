"""GT-backed scene-repair verifier.

The required evidence is exactly three scenes:

* Input, captured from the task's ``inputs.init_map`` in the scorer;
* Candidate, exported from the independent scoring editor;
* GT, captured from the task's canonical map in the same scorer.

GT-minus-Input freezes what should change before Candidate inspection.  One
Candidate-to-GT correspondence is then projected twice: locally onto those
repair targets and globally through ``scene_diff``. Both Indoor and Outdoor
combine structural evidence with measured visual scores. Missing visual scores
fall back to the same head's structural score without changing aggregate status.
Prompt text and task-provided images are optional audit context and never
change target derivation or the structured score.

The report's published score is the paper's Actor Repair F1 (``metrics.
actor_repair_f1``), measured from the same three snapshots. The local/global
weighted composite described above is retained only as a diagnostic
(``metrics.diagnostics.legacy_gt_repair_composite``) and is never scored.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .. import contracts, repair_score, repair_success, ue_evidence
from ...protocol import actor_f1 as actor_repair_f1
from ..composite import composite_report, run_leaf
from ..context import Context, error, read_label
from ..evaluation_policy import load_evaluation_policy, task_mode
from ..repair_target_scope import load_repair_target_scope
from ..render_evidence import EvidenceRequest
from ..scene_diff import actor_identity, actor_summary, diff_scenes
from . import caption_similarity, gt_geometry, scene_diff, visual_as_judge


CLASS = "gt"


TARGET_TRANSFORM_WEIGHT_POLICY_ID = "gt-repair-target-transform-weighted"
TARGET_TRANSFORM_WEIGHTS = {
    "position_similarity": 0.50,
    "rotation_similarity": 0.30,
    "scale_similarity": 0.20,
}
TARGET_GEOMETRY_WEIGHT_POLICY_ID = "gt-repair-target-geometry-weighted"
TARGET_GEOMETRY_WEIGHTS = {
    "footprint_similarity": 0.50,
    "bounds_size_similarity": 0.30,
    "pairwise_layout_similarity": 0.20,
}
LOCAL_REPAIR_WEIGHT_POLICY_ID = repair_score.LOCAL_POLICY_ID
MISSING_TARGET_SHORT_CIRCUIT_POLICY_ID = (
    "gt-repair-missing-candidate-target-short-circuit"
)
PURE_REMOVAL_VISUAL_SHORT_CIRCUIT_POLICY_ID = (
    "gt-repair-pure-removal-visual-short-circuit"
)
LOCAL_REPAIR_WEIGHTS = repair_score.LOCAL_WEIGHTS
LOCAL_REPAIR_PREREQUISITES = repair_score.PREREQUISITES
FINAL_SCORE_WEIGHT_POLICY_ID = repair_score.POLICY_ID
FINAL_SCORE_WEIGHTS = repair_score.FINAL_WEIGHTS
INDOOR_LOCAL_VISUAL_ROUTING_POLICY_ID = (
    "gt-repair-indoor-local-visual-only-structured-global"
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _unit(value: Any) -> float | None:
    number = _number(value)
    return None if number is None else max(0.0, min(1.0, number))


def _weighted_available(
    values: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> tuple[float | None, dict[str, float], dict[str, float]]:
    """Return a transparent weighted mean over the measured channels only."""

    unknown = sorted(set(values) - set(weights))
    if unknown:
        raise ValueError(f"weighted aggregation has no weights for {unknown}")
    measured = {name: value for name, value in values.items() if value is not None}
    if not measured:
        return None, {}, {}
    invalid = {
        name: weight
        for name, weight in weights.items()
        if _number(weight) is None or float(weight) <= 0.0
    }
    if invalid:
        raise ValueError(f"weighted aggregation needs positive weights: {invalid}")
    total = sum(float(weights[name]) for name in measured)
    effective = {name: round(float(weights[name]) / total, 6) for name in measured}
    contributions = {
        name: round(float(value) * float(weights[name]) / total, 6)
        for name, value in measured.items()
    }
    score = (
        sum(float(value) * float(weights[name]) for name, value in measured.items()) / total
    )
    return round(score, 6), effective, contributions


def _apply_weighted_leaf_policy(
    report: dict[str, Any],
    weights: Mapping[str, float],
    *,
    policy_id: str,
    prerequisite_leaf_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Replace a composite mean with a frozen prerequisite-gated score."""

    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{policy_id}: composite report has no metrics")
    leaves = metrics.get("leaf_results")
    if not isinstance(leaves, list):
        raise ValueError(f"{policy_id}: composite report has no leaf_results")
    leaf_ids = {str(value.get("leaf_id")) for value in leaves if isinstance(value, Mapping)}
    if leaf_ids != set(weights):
        raise ValueError(
            f"{policy_id}: weight keys {sorted(weights)} do not match "
            f"leaf ids {sorted(leaf_ids)}"
        )
    values = {
        str(value["leaf_id"]): (
            _unit(value.get("score"))
            if value.get("status") in {contracts.MEASURED, contracts.PASS, contracts.FAIL}
            else None
        )
        for value in leaves
        if isinstance(value, Mapping)
    }
    weighted_score, effective, contributions = _weighted_available(values, weights)
    unknown_prerequisites = sorted(set(prerequisite_leaf_ids) - set(values))
    if unknown_prerequisites:
        raise ValueError(
            f"{policy_id}: unknown prerequisite leaves {unknown_prerequisites}"
        )
    prerequisite_values = {name: values[name] for name in prerequisite_leaf_ids}
    prerequisite_gate = (
        min(float(value) for value in prerequisite_values.values())
        if prerequisite_values
        and all(value is not None for value in prerequisite_values.values())
        else (1.0 if not prerequisite_values else None)
    )
    score = (
        round(float(prerequisite_gate) * float(weighted_score), 6)
        if prerequisite_gate is not None and weighted_score is not None
        else None
    )
    metrics.update(
        {
            "score_aggregation": (
                "prerequisite_gate_times_weighted_mean_of_available_normalized_leaf_scores"
            ),
            "score_weight_policy_id": policy_id,
            "configured_score_weights": dict(weights),
            "effective_score_weights": effective,
            "weighted_score_contributions": contributions,
            "weighted_quality_mix": weighted_score,
            "prerequisite_leaf_ids": list(prerequisite_leaf_ids),
            "prerequisite_values": prerequisite_values,
            "prerequisite_gate": prerequisite_gate,
        }
    )
    evidence = report.setdefault("evidence", {})
    if isinstance(evidence, dict):
        evidence.update(
            {
                "score_weight_policy_id": policy_id,
                "configured_score_weights": dict(weights),
                "unavailable_leaf_policy": "renormalize_over_available_weights",
                "prerequisite_gate_formula": (
                    "min(target_recovery, target_identity)"
                    if prerequisite_leaf_ids
                    else None
                ),
            }
        )
    if report.get("status") in {
        contracts.MEASURED,
        contracts.PASS,
        contracts.FAIL,
    }:
        report["score"] = score
    return report


def _nonnegative_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_pure_removal_local_report(report: Mapping[str, Any]) -> bool:
    """Whether the frozen GT-minus-Input target contains only removals."""

    evidence = report.get("evidence")
    derivation = (
        evidence.get("target_derivation") if isinstance(evidence, Mapping) else None
    )
    counts = derivation.get("operation_counts") if isinstance(derivation, Mapping) else None
    if isinstance(counts, Mapping):
        parsed = {
            operation: _nonnegative_count(counts.get(operation))
            for operation in ("add", "remove", "repair")
        }
        if all(value is not None for value in parsed.values()):
            return parsed["remove"] > 0 and parsed["add"] == 0 and parsed["repair"] == 0

    # Backward-compatible fallback for frozen results produced before the
    # operation-count audit was recorded. These two counts are emitted on the
    # target-recovery leaf by every current repair-target measurement.
    metrics = report.get("metrics")
    leaves = metrics.get("leaf_results") if isinstance(metrics, Mapping) else None
    recovery = next(
        (
            value
            for value in leaves or ()
            if isinstance(value, Mapping) and value.get("leaf_id") == "target_recovery"
        ),
        None,
    )
    recovery_metrics = recovery.get("metrics") if isinstance(recovery, Mapping) else None
    if not isinstance(recovery_metrics, Mapping):
        return False
    desired_count = _nonnegative_count(recovery_metrics.get("desired_actor_count"))
    removal_count = _nonnegative_count(recovery_metrics.get("removal_target_count"))
    return desired_count == 0 and removal_count is not None and removal_count > 0


def _apply_local_repair_policy(report: dict[str, Any]) -> dict[str, Any]:
    """Apply prerequisites selected from the frozen repair operation set."""

    return repair_score.apply_local(
        report, pure_removal=_is_pure_removal_local_report(report),
    )


def _pick(metrics: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {name: metrics.get(name) for name in names}


def _visual_config(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = spec.get("visual_semantic_diff")
    return dict(value) if isinstance(value, Mapping) else {}


def _visual_spec(spec: Mapping[str, Any], scope: str) -> dict[str, Any]:
    return {
        "name": "visual_as_judge",
        "mode": "gt_paired",
        "ground_truth": spec.get("ground_truth"),
        "comparison_scope": scope,
    }


def _uses_indoor_local_visual_only(task: Any) -> bool:
    """Whether room-scale visual evidence is redundant for this repair."""

    return (
        str(getattr(task, "scene_environment", "") or "").strip().casefold()
        == "indoor"
    )


def _global_scene_diff_spec(task: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Disable room-scale visual leaves while retaining global structure."""

    result = copy.deepcopy(dict(spec))
    if not _uses_indoor_local_visual_only(task):
        return result
    configured = _visual_config(result)
    configured["caption_diff"] = False
    configured["paired_visual"] = False
    result["visual_semantic_diff"] = configured
    result["visual_evidence_routing"] = {
        "policy_id": INDOOR_LOCAL_VISUAL_ROUTING_POLICY_ID,
        "repair_target_visual": "score_when_available",
        "whole_scene_visual": "omitted",
        "whole_scene_structured_diff": "required",
    }
    return result


def evidence_requests(
    task: Any,
    spec: dict[str, Any],
) -> tuple[EvidenceRequest, ...]:
    """Request local-only visual evidence for Indoor repair tasks."""

    configured = _visual_config(spec)
    indoor_local_only = _uses_indoor_local_visual_only(task)
    requests: list[EvidenceRequest] = []
    if configured.get("caption_diff") is True and not indoor_local_only:
        requests.extend(caption_similarity.evidence_requests(task, spec))
    if configured.get("paired_visual") is True:
        if not indoor_local_only:
            requests.extend(
                visual_as_judge.evidence_requests(
                    task,
                    _visual_spec(spec, "whole_scene"),
                )
            )
        requests.extend(
            visual_as_judge.evidence_requests(task, _visual_spec(spec, "repair_target"))
        )
    return tuple(requests)


def _target_leaf(
    context: Context,
    leaf_id: str,
    *,
    score: float | None,
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    status = contracts.MEASURED if score is not None else "not_applicable"
    result = {
        **contracts.base(f"gt_repair.repair_target_diff.{leaf_id}", context.ids),
        "leaf_id": leaf_id,
        "status": status,
        "score": score,
        "metrics": dict(metrics),
        "evidence": dict(evidence),
        "artifacts": {},
        "probes_used": ("shared_gt_scene_correspondence",),
    }
    if score is None:
        result["failure_reason"] = reason or (
            f"{leaf_id} has no applicable measured channel for this target"
        )
    return result


def _target_visual_leaf(context: Context) -> dict[str, Any]:
    configured = _visual_config(context.spec)
    common = {
        "primary_scope": "gt_minus_input_repair_target",
        "camera_source": "gt_minus_input_target_bounds",
        "candidate_and_gt_reuse_identical_absolute_cameras": True,
    }
    if configured.get("paired_visual") is not True:
        return _target_leaf(
            context,
            "target_visual_diff",
            score=None,
            metrics={},
            evidence=common,
            reason="local paired visual is disabled by the frozen gt_repair spec",
        )
    report = run_leaf(
        context,
        visual_as_judge.verify,
        spec=_visual_spec(context.spec, "repair_target"),
    )
    score = _unit(report.get("score"))
    report_status = report.get("status")
    if score is not None:
        status = contracts.MEASURED
    elif report_status == contracts.ERROR:
        status = contracts.ERROR
    else:
        status = "not_evaluated"
    result = {
        **contracts.base("gt_repair.repair_target_diff.target_visual_diff", context.ids),
        "leaf_id": "target_visual_diff",
        "status": status,
        "score": score,
        "metrics": dict(report.get("metrics") or {}),
        "evidence": {**common, **dict(report.get("evidence") or {})},
        "artifacts": dict(report.get("artifacts") or {}),
        "probes_used": tuple(report.get("probes_used") or ()),
    }
    if score is None:
        result["failure_reason"] = str(
            report.get("failure_reason")
            or "local paired visual produced no calibrated score"
        )
    return result


def _missing_desired_target(target: Mapping[str, Any]) -> bool:
    """Whether no Candidate Actor matched any required desired target.

    The frozen GT-minus-Input derivation owns ``desired_actor_count`` and the
    Candidate-to-GT correspondence owns ``matched_actor_count``.  Their exact
    zero-pair state is therefore sufficient evidence that downstream target
    fidelity channels are inapplicable, not unavailable due to verifier
    failure.
    """

    desired_count = _nonnegative_count(target.get("desired_actor_count"))
    matched_count = _nonnegative_count(target.get("matched_actor_count"))
    return desired_count is not None and desired_count > 0 and matched_count == 0


def _pure_removal_without_desired_target(target: Mapping[str, Any]) -> bool:
    desired_count = _nonnegative_count(target.get("desired_actor_count"))
    removal_count = _nonnegative_count(target.get("removal_target_count"))
    return desired_count == 0 and removal_count is not None and removal_count > 0


def _target_visual_short_circuit_kind(
    target: Mapping[str, Any],
) -> str | None:
    if _missing_desired_target(target):
        return "missing_candidate_target"
    if _pure_removal_without_desired_target(target):
        return "pure_removal_without_desired_target"
    return None


def _short_circuited_target_visual_leaf(
    context: Context,
    target: Mapping[str, Any],
    kind: str,
) -> dict[str, Any]:
    """Record an intentional prerequisite-gated local visual skip."""

    if kind == "missing_candidate_target":
        gate = 0.0
        policy_id = MISSING_TARGET_SHORT_CIRCUIT_POLICY_ID
        decision = (
            "no_candidate_target_pair; recovery_and_identity_prerequisite_"
            "gate_determines_zero"
        )
        reason = (
            "local paired visual is not applicable because no Candidate Actor "
            "matched a required GT repair target; the recovery/identity "
            "prerequisite gate determines a zero local score"
        )
    elif kind == "pure_removal_without_desired_target":
        gate = _unit(target.get("actor_correspondence_score"))
        policy_id = PURE_REMOVAL_VISUAL_SHORT_CIRCUIT_POLICY_ID
        decision = (
            "pure_removal_has_no_desired_actor; recovery_prerequisite_"
            "determines_local_score"
        )
        reason = (
            "local paired visual is not applicable to a pure-removal target "
            "because no desired Actor remains; target_recovery determines the "
            "local score"
        )
    else:
        raise ValueError(f"unsupported target visual short circuit {kind!r}")

    return _target_leaf(
        context,
        "target_visual_diff",
        score=None,
        metrics={
            "desired_actor_count": target.get("desired_actor_count"),
            "matched_actor_count": target.get("matched_actor_count"),
            "missing_actor_count": target.get("missing_actor_count"),
            "removal_target_count": target.get("removal_target_count"),
            "remaining_removal_count": target.get("remaining_removal_count"),
            "prerequisite_gate": gate,
        },
        evidence={
            "primary_scope": "gt_minus_input_repair_target",
            "camera_source": "gt_minus_input_target_bounds",
            "candidate_and_gt_reuse_identical_absolute_cameras": True,
            "target_visual_short_circuit_kind": kind,
            "target_visual_short_circuit_policy_id": policy_id,
            "visual_evidence_required_for_score": False,
            "decision": decision,
        },
        reason=reason,
    )


def report_from_repair_target_measurement(
    context: Context,
    target_value: Mapping[str, Any],
    *,
    target_visual: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the production repair-target report from frozen measurements.

    Live verification leaves ``target_visual`` unset and captures the visual
    leaf normally. Audited offline recomputation may instead pass the exact
    visual leaf from the source result while rebuilding only deterministic
    correspondence-dependent leaves. Both paths therefore share every score
    formula, weight, and public evidence field.
    """

    if not isinstance(target_value, Mapping):
        return error(
            "gt_repair.repair_target_diff",
            context,
            "GT-minus-Input produced no repair target measurement",
        )
    target = dict(target_value)
    missing_desired_target = _missing_desired_target(target)
    target_visual_short_circuit_kind = _target_visual_short_circuit_kind(target)
    matching_audit = _pick(
        target,
        "matching_algorithm_version",
        "matching_cost_formula",
        "hard_compatibility_fields",
        "name_guid_path_used_as_hint_only",
        "target_actor_signatures",
        "target_resolution",
        "compatibility_candidates",
        "final_assignment",
        "unmatched_gt_actors",
        "unmatched_candidate_actors",
        "ambiguity",
        "failure_classification",
        "primary_classification",
        "permuted_target_identity_count",
    )
    common = {
        "primary_scope": "gt_minus_input_repair_target",
        "target_derivation": target.get("target_derivation"),
        "normalization_scale_cm": target.get("normalization_scale_cm"),
        "pair_coverage": target.get("pair_coverage"),
        "global_alignment": target.get("global_alignment"),
        "correspondence_source": (
            "shared_candidate_to_gt_actor_rows_with_target_local_rematch"
        ),
        "correspondence_policy": target.get("correspondence_policy"),
        "exchangeable_assignment_cost": target.get("exchangeable_assignment_cost"),
        "exchangeable_group_count": target.get("exchangeable_group_count"),
        "exchangeable_reassigned_actor_count": target.get(
            "exchangeable_reassigned_actor_count"
        ),
        "matching_algorithm_version": target.get("matching_algorithm_version"),
        "name_guid_path_used_as_hint_only": target.get("name_guid_path_used_as_hint_only"),
        "score_policy": "continuous_without_pass_fail_threshold",
    }
    transform_channels = {
        "position_similarity": _unit(target.get("position_similarity")),
        "rotation_similarity": _unit(target.get("rotation_similarity")),
        "scale_similarity": _unit(target.get("scale_similarity")),
    }
    geometry_channels = {
        "footprint_similarity": _unit(target.get("footprint_similarity")),
        "bounds_size_similarity": _unit(target.get("bounds_size_similarity")),
        "pairwise_layout_similarity": _unit(target.get("pairwise_layout_similarity")),
    }
    (
        transform_score,
        transform_effective_weights,
        transform_contributions,
    ) = _weighted_available(transform_channels, TARGET_TRANSFORM_WEIGHTS)
    (
        geometry_score,
        geometry_effective_weights,
        geometry_contributions,
    ) = _weighted_available(geometry_channels, TARGET_GEOMETRY_WEIGHTS)
    leaves = [
        _target_leaf(
            context,
            "target_recovery",
            score=_unit(target.get("actor_correspondence_score")),
            metrics=_pick(
                target,
                "desired_actor_count",
                "candidate_target_actor_count",
                "matched_actor_count",
                "missing_actor_count",
                "extra_actor_count",
                "removal_target_count",
                "remaining_removal_count",
                "actor_correspondence_score",
            ),
            evidence={
                **common,
                "score_formula": "actor_restoration_F1_and_removal_completion_mean",
            },
        ),
        _target_leaf(
            context,
            "target_identity",
            score=_unit(target.get("identity_score")),
            metrics=_pick(target, "identity_match_rate", "pair_coverage", "identity_score"),
            evidence={
                **common,
                "score_formula": "target_correspondence_score * identity_match_rate",
            },
        ),
        _target_leaf(
            context,
            "target_transform_diff",
            score=transform_score,
            metrics={
                **_pick(
                    target,
                    "position_rmse_cm",
                    "position_mean_cm",
                    "rotation_mean_deg",
                    "scale_log_rmse",
                    "pair_coverage",
                ),
                "normalized_channels": transform_channels,
                "configured_channel_weights": dict(TARGET_TRANSFORM_WEIGHTS),
                "effective_channel_weights": transform_effective_weights,
                "weighted_channel_contributions": transform_contributions,
            },
            evidence={
                **common,
                "score_formula": "weighted_mean(position, rotation, scale)",
                "score_weight_policy_id": (TARGET_TRANSFORM_WEIGHT_POLICY_ID),
                "unavailable_channel_policy": ("renormalize_over_available_weights"),
            },
        ),
        _target_leaf(
            context,
            "target_geometry_diff",
            score=geometry_score,
            metrics={
                **_pick(
                    target,
                    "mean_footprint_iou",
                    "bounds_size_log_rmse",
                    "pairwise_layout_error",
                    "pairwise_layout_pair_count",
                    "pair_coverage",
                ),
                "normalized_channels": geometry_channels,
                "configured_channel_weights": dict(TARGET_GEOMETRY_WEIGHTS),
                "effective_channel_weights": geometry_effective_weights,
                "weighted_channel_contributions": geometry_contributions,
            },
            evidence={
                **common,
                "score_formula": ("weighted_mean(footprint, bounds_size, layout)"),
                "score_weight_policy_id": (TARGET_GEOMETRY_WEIGHT_POLICY_ID),
                "unavailable_channel_policy": ("renormalize_over_available_weights"),
            },
        ),
        _target_leaf(
            context,
            "target_attribute_diff",
            score=_unit(target.get("attribute_similarity")),
            metrics=_pick(
                target,
                "task_property_match_rate",
                "material_set_match_rate",
                "material_slot_match_rate",
                "pair_coverage",
                "attribute_similarity",
            ),
            evidence={
                **common,
                "score_formula": "macro_mean_of_measured_target_attribute_rates",
            },
        ),
        (
            _short_circuited_target_visual_leaf(
                context,
                target,
                target_visual_short_circuit_kind,
            )
            if target_visual_short_circuit_kind is not None
            else (
                _target_visual_leaf(context)
                if target_visual is None
                else copy.deepcopy(dict(target_visual))
            )
        ),
    ]
    leaves.insert(-1, repair_success.report(context, target.get("repair_success")))
    report = composite_report(
        "gt_repair.repair_target_diff",
        context,
        leaves,
        evidence={
            **common,
            "actor_correspondence": matching_audit,
            "missing_target_short_circuit": missing_desired_target,
            "missing_target_short_circuit_policy_id": (
                MISSING_TARGET_SHORT_CIRCUIT_POLICY_ID if missing_desired_target else None
            ),
            "target_visual_short_circuit_kind": target_visual_short_circuit_kind,
            "local_dimensions": [
                "recovery",
                "identity",
                "transform",
                "geometry",
                "attribute",
                "visual",
            ],
        },
    )
    return _apply_local_repair_policy(report)


def _repair_target_report(
    context: Context,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    if comparison.get("status") not in {contracts.MEASURED, contracts.PASS}:
        return error(
            "gt_repair.repair_target_diff",
            context,
            str(comparison.get("failure_reason") or "GT correspondence failed"),
        )
    target_value = (comparison.get("metrics") or {}).get("repair_target")
    if not isinstance(target_value, Mapping):
        return error(
            "gt_repair.repair_target_diff",
            context,
            "GT-minus-Input produced no repair target measurement",
        )
    return report_from_repair_target_measurement(context, target_value)


def _summary_ids(value: Any) -> set[str]:
    if not isinstance(value, Mapping):
        return set()
    values = value.get("stable_actor_ids") or ()
    return {str(item) for item in values if str(item).strip()}


def _asset_key(actor: Mapping[str, Any]) -> str:
    return str(actor.get("asset_path") or "").strip().casefold()


def _locality_audit(
    context: Context,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Report off-target Input-to-Candidate edits without a second vote."""

    try:
        evidence = ue_evidence.collect(context)
        if evidence.input_scene is None:
            raise ValueError("Input scene snapshot was not loaded")
        if context.spec.get("canonical_scene") == {"runtime_task_ground_truth_map": True}:
            canonical_scene, _canonical_path = ue_evidence.capture_task_ground_truth(
                context,
                candidate_scene=evidence.candidate,
            )
            label = dict(read_label(context) or {})
            label["canonical_actors"] = canonical_scene.get("actors")
            scope = load_repair_target_scope(
                context.task,
                label=label,
                input_scene=evidence.input_scene,
            )
        else:
            scope = load_repair_target_scope(
                context.task,
                input_scene=evidence.input_scene,
            )
        if scope is None:
            raise ValueError("no runtime GT-minus-Input repair target scope")
        changed = diff_scenes(evidence.input_scene, evidence.candidate)
        source_keys = {actor_identity(actor) for actor in scope.source_actors}
        desired_assets = {
            value for actor in scope.desired_actors if (value := _asset_key(actor))
        }
        target_metrics = (comparison.get("metrics") or {}).get("repair_target")
        rows = (target_metrics or {}).get("matched_actor_rows") or []
        matched_candidate_ids = {
            actor_id
            for row in rows
            if isinstance(row, Mapping)
            for actor_id in _summary_ids(row.get("candidate"))
        }

        def allowed_actor(actor: Mapping[str, Any]) -> bool:
            stable = str(actor.get("stable_actor_id") or "")
            return (
                actor_identity(actor) in source_keys
                or stable in matched_candidate_ids
                or bool(_asset_key(actor) and _asset_key(actor) in desired_assets)
            )

        off_added = [actor for actor in changed.added if not allowed_actor(actor)]
        off_removed = [actor for actor in changed.removed if not allowed_actor(actor)]
        off_moved = [change for change in changed.moved if change.key not in source_keys]
        off_modified = [
            change for change in changed.modified if change.key not in source_keys
        ]
        all_keys = {actor_identity(actor) for actor in (*changed.added, *changed.removed)}
        all_keys.update(change.key for change in (*changed.moved, *changed.modified))
        off_keys = {actor_identity(actor) for actor in (*off_added, *off_removed)}
        off_keys.update(change.key for change in (*off_moved, *off_modified))
        locality_similarity = 1.0 - len(off_keys) / max(len(all_keys), 1)
        report = {
            **contracts.base("gt_repair.locality_audit", context.ids),
            "leaf_id": "locality_audit",
            "status": contracts.MEASURED,
            "score": round(max(0.0, min(1.0, locality_similarity)), 6),
            "metrics": {
                "changed_actor_count": len(all_keys),
                "off_target_actor_count": len(off_keys),
                "added_actor_count": len(changed.added),
                "removed_actor_count": len(changed.removed),
                "moved_actor_count": len(changed.moved),
                "modified_actor_count": len(changed.modified),
                "off_target_added_actor_count": len(off_added),
                "off_target_removed_actor_count": len(off_removed),
                "off_target_moved_actor_count": len(off_moved),
                "off_target_modified_actor_count": len(off_modified),
                "off_target_actors": [
                    actor_summary(actor) for actor in (*off_added, *off_removed)
                ],
            },
            "evidence": {
                "comparison": "Input_to_Candidate",
                "allowed_region": "frozen_GT_minus_Input_targets",
                "score_role": "diagnostic_only",
                "global_scene_diff_already_penalizes_off_target_changes": True,
            },
            "artifacts": evidence.artifacts(),
            "probes_used": ("input_candidate_scene_diff",),
            "contributes_to_aggregate": False,
            "score_role": "report_only",
            "report_only_reason": (
                "whole-scene scene_diff already measures off-target drift; "
                "locality is retained for diagnosis and must not vote twice"
            ),
        }
        return report
    except Exception as exc:  # noqa: BLE001 - report-only evidence stays visible
        report = error(
            "gt_repair.locality_audit",
            context,
            f"{type(exc).__name__}: {exc}",
        )
        report.update(
            {
                "leaf_id": "locality_audit",
                "contributes_to_aggregate": False,
                "score_role": "report_only",
                "report_only_reason": "locality audit evidence was unavailable",
            }
        )
        return report


def _optional_input_audit(context: Context) -> dict[str, Any]:
    prompt = str(getattr(context.task, "prompt", "") or "").strip()
    declared = getattr(context.task, "declared_reference_views", ())
    try:
        declared_count = len(list(declared))
    except TypeError:
        declared_count = 0
    return {
        "prompt": {
            "available": bool(prompt),
            "used_for_target_derivation": False,
            "used_for_score": False,
        },
        "gt_or_reference_images": {
            "declared_count": declared_count,
            "runtime_count": len(context.reference_images),
            "used_for_target_derivation": False,
            "used_for_score": False,
            "role": "optional_audit_or_camera_metadata_only",
        },
    }


def measure_actor_repair_f1(
    input_scene: Mapping[str, Any] | None,
    ground_truth_scene: Mapping[str, Any] | None,
    candidate_scene: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The published Repair F1 of this verifier, from three scene snapshots."""

    try:
        values, audit = actor_repair_f1.measure(input_scene, ground_truth_scene, candidate_scene)
    except Exception as exc:  # noqa: BLE001 - a measurement failure is reported
        return {"policy": actor_repair_f1.POLICY, "status": contracts.ERROR,
                "failure_reason": f"{type(exc).__name__}: {exc}"}
    return {**values, "status": contracts.MEASURED,
            "audit_summary": {
                key: audit[key] for key in (
                    "presence_matches", "presence_unmatched_gt",
                    "presence_unmatched_candidates", "correct_removal_ids",
                    "missing_removal_ids", "off_target_removal_ids",
                    "repurposed_background_ids",
                )
            }}


def _actor_repair_f1(context: Context) -> dict[str, Any]:
    """Collect the three snapshots the verifier already exported and measure F1."""

    try:
        evidence = ue_evidence.collect(context)
        if evidence.input_scene is None:
            raise ValueError("Input scene snapshot was not loaded")
        if context.spec.get("canonical_scene") == {"runtime_task_ground_truth_map": True}:
            ground_truth, _path = ue_evidence.capture_task_ground_truth(
                context, candidate_scene=evidence.candidate,
            )
        else:
            scope = load_repair_target_scope(context.task, input_scene=evidence.input_scene)
            if scope is None:
                raise ValueError("no runtime GT-minus-Input repair target scope")
            ground_truth = scope.canonical_scene
    except Exception as exc:  # noqa: BLE001 - reported, never scored
        return {"policy": actor_repair_f1.POLICY, "status": contracts.ERROR,
                "failure_reason": f"{type(exc).__name__}: {exc}"}
    return measure_actor_repair_f1(evidence.input_scene, ground_truth, evidence.candidate)


def finalize_report(
    report: dict[str, Any],
    local: Mapping[str, Any],
    global_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen local/global aggregation to a GT-repair report.

    Kept separate from live evidence collection so a failed external leaf can
    be recomputed from its already-frozen evidence without reopening either UE
    scene. Both paths therefore use exactly the same score formula and public
    provenance fields.
    """

    children = (report.get("metrics") or {}).get("leaf_results") or []
    others = [
        child for child in children
        if child.get("leaf_id") not in {"repair_target_diff", "scene_diff"}
    ]
    report.setdefault("metrics", {})["leaf_results"] = [
        {**dict(local), "leaf_id": "repair_target_diff"},
        {**dict(global_report), "leaf_id": "scene_diff"}, *others,
    ]
    rebuilt = repair_score.apply(report)
    report.clear()
    report.update(rebuilt)
    return repair_score.publish_actor_f1(report)


def _target_visual_coverage(local: Mapping[str, Any]) -> dict[str, Any]:
    """Lift the nested visual availability summary to the public report."""

    metrics = local.get("metrics")
    leaves = metrics.get("leaf_results") if isinstance(metrics, Mapping) else ()
    visual = next(
        (
            value
            for value in leaves or ()
            if isinstance(value, Mapping)
            and value.get("leaf_id") == "target_visual_diff"
        ),
        None,
    )
    if not isinstance(visual, Mapping):
        return {"status": "not_evaluated", "visual_score_available": False}
    evidence = visual.get("evidence")
    coverage = (
        evidence.get("visual_coverage")
        if isinstance(evidence, Mapping)
        else None
    )
    return {
        "status": visual.get("status"),
        "visual_score_available": visual.get("score") is not None,
        **(dict(coverage) if isinstance(coverage, Mapping) else {}),
    }


def verify(context: Context) -> dict[str, Any]:
    """Evaluate one Input/Candidate/GT repair with local and global scopes."""

    if task_mode(getattr(context.task, "kind", "")) != "repair":
        return error("gt_repair", context, "gt_repair accepts only scene-repair tasks")
    if (
        not isinstance(context.spec.get("ground_truth"), str)
        or not str(context.spec.get("ground_truth")).strip()
    ):
        return error("gt_repair", context, "gt_repair needs ground_truth")
    try:
        policy = load_evaluation_policy(context.task)
    except Exception as exc:  # noqa: BLE001 - frozen input contract refusal
        return error("gt_repair", context, f"{type(exc).__name__}: {exc}")
    if policy.source_snapshot is None:
        return error(
            "gt_repair",
            context,
            "gt_repair needs evaluation_policy.source_snapshot for the Input scene",
        )

    comparison_spec = policy.leaf_spec(context.task, "gt_repair", context.spec)
    if policy.source_snapshot == {"runtime_task_input_map": True}:
        comparison_spec["canonical_scene"] = {"runtime_task_ground_truth_map": True}
    working = replace(context, spec=comparison_spec)
    comparison = run_leaf(working, gt_geometry.verify, spec=comparison_spec)
    local = _repair_target_report(working, comparison)
    local["leaf_id"] = "repair_target_diff"
    global_spec = _global_scene_diff_spec(context.task, comparison_spec)
    global_working = replace(working, spec=global_spec)
    global_report = scene_diff._report_from_comparison(global_working, comparison)
    global_report["leaf_id"] = "scene_diff"
    indoor_local_only = _uses_indoor_local_visual_only(context.task)
    if indoor_local_only:
        global_evidence = dict(global_report.get("evidence") or {})
        global_evidence.update(
            {
                "visual_evidence_routing_policy_id": (
                    INDOOR_LOCAL_VISUAL_ROUTING_POLICY_ID
                ),
                "whole_scene_visual": "omitted",
                "whole_scene_structured_diff": "required",
            }
        )
        global_report["evidence"] = global_evidence
    locality = _locality_audit(working, comparison)
    runtime_capture = policy.source_snapshot == {"runtime_task_input_map": True}
    target_visual_coverage = _target_visual_coverage(local)

    report = composite_report(
        "gt_repair",
        context,
        [local, global_report, locality],
        evidence={
            "public_interface": "gt_repair",
            "report_schema_version": "gt-repair-public.v1",
            "required_inputs": {
                "input": (
                    "independent_runtime_export_of_task.inputs.init_map"
                    if runtime_capture
                    else "evaluation_policy.source_snapshot"
                ),
                "candidate": "independent_scoring_editor_export",
                "gt": (
                    "independent_runtime_export_of_task.ground_truth_map"
                    if runtime_capture
                    else "answer_key.canonical_actors_and_canonical_map"
                ),
            },
            "target_derivation": "deterministic_GT_minus_Input_before_Candidate",
            "shared_correspondence": (
                "one Candidate_to_GT actor correspondence reused by local and global"
            ),
            "visual_evidence_routing": (
                {
                    "policy_id": INDOOR_LOCAL_VISUAL_ROUTING_POLICY_ID,
                    "repair_target_visual": "score_when_available",
                    "whole_scene_visual": "omitted",
                    "whole_scene_structured_diff": "required",
                }
                if indoor_local_only
                else {
                    "repair_target_visual": "score_when_available",
                    "whole_scene_visual": "score_when_available",
                    "whole_scene_structured_diff": "required",
                }
            ),
            "optional_inputs": _optional_input_audit(context),
            "target_visual_coverage": target_visual_coverage,
        },
    )
    report["metrics"]["target_visual_coverage"] = target_visual_coverage
    report["metrics"][repair_score.PUBLISHED_SCORE] = _actor_repair_f1(working)
    return finalize_report(report, local, global_report)


__all__ = [
    "evidence_requests",
    "finalize_report",
    "measure_actor_repair_f1",
    "report_from_repair_target_measurement",
    "verify",
]
