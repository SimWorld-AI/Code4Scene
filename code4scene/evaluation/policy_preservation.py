"""Policy-aware Input-to-Candidate preservation measurements.

The canonical path performs one strict scene diff and then applies the frozen
edit scope. It is used only when the task starts from an editable Input scene.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import contracts, ue_evidence
from .context import Context, error
from .evaluation_policy import FrozenEvaluationPolicy
from .scene_diff import Change, actor_identity, actor_summary, diff_scenes
from .scene_geometry import cyclic_degrees, volume_cm3
from .selection import select_candidate_actors


_METRIC_VERSION = "source-preservation-policy-v1"
_TRANSFORM_AXES = (
    ("location", "location_cm", ("x", "y", "z")),
    ("rotation", "rotation_deg", ("pitch", "yaw", "roll")),
    ("scale", "scale", ("x", "y", "z")),
)


def _matches(actor: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    return bool(select_candidate_actors([actor], selector))


def _targets(
    actor: Mapping[str, Any], policy: FrozenEvaluationPolicy
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        value
        for value in policy.edit_scope.get("targets", ())
        if isinstance(value, Mapping)
        and _matches(actor, value.get("selector") or {})
    )


def _allows_operation(
    actor: Mapping[str, Any],
    operation: str,
    policy: FrozenEvaluationPolicy,
) -> bool:
    for target in _targets(actor, policy):
        allowed = (target.get("allow") or {}).get(operation, False)
        if allowed is True or isinstance(allowed, Mapping):
            return True
    return False


def _triplet(actor: Mapping[str, Any], key: str, default: float) -> tuple[float, ...]:
    raw = (actor.get("transform") or {}).get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return (default, default, default)
    return tuple(float(raw[index]) if index < len(raw) else default for index in range(3))


def _changed_transform_axes(
    change: Change, tolerances: Mapping[str, Any]
) -> tuple[str, ...]:
    changed: list[str] = []
    for group, key, axes in _TRANSFORM_AXES:
        default = 1.0 if group == "scale" else 0.0
        before = _triplet(change.before, key, default)
        after = _triplet(change.after, key, default)
        tolerance = float(tolerances.get(key, 0.0))
        for index, axis in enumerate(axes):
            delta = (
                cyclic_degrees(before[index], after[index])
                if group == "rotation"
                else abs(before[index] - after[index])
            )
            if delta > tolerance:
                changed.append(f"{group}.{axis}")
    return tuple(changed)


def _transform_authorized(
    change: Change, policy: FrozenEvaluationPolicy
) -> tuple[bool, tuple[str, ...]]:
    changed = _changed_transform_axes(
        change, policy.edit_scope.get("tolerances") or {}
    )
    for target in _targets(change.after, policy):
        allowed = (target.get("allow") or {}).get("transform", False)
        if allowed is True:
            return True, changed
        if isinstance(allowed, Mapping):
            axes = {str(value) for value in allowed.get("axes", ())}
            groups = {value.split(".", 1)[0] for value in axes if "." not in value}
            if all(axis in axes or axis.split(".", 1)[0] in groups for axis in changed):
                return True, changed
    return False, changed


def _slot_map(actor: Mapping[str, Any]) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    for value in actor.get("component_material_slots") or ():
        if not isinstance(value, Mapping):
            continue
        component = str(value.get("component_identity") or "").strip()
        index = value.get("slot_index")
        path = str(value.get("material_path") or "").strip()
        if component and isinstance(index, int) and not isinstance(index, bool) and path:
            result[(component, index)] = path
    return result


def _changed_attribute_tokens(change: Change) -> tuple[str, ...]:
    tokens: set[str] = set()
    for field in change.fields:
        if field == "properties":
            left = change.before.get("properties") or {}
            right = change.after.get("properties") or {}
            keys = set(left) | set(right) if isinstance(left, Mapping) and isinstance(right, Mapping) else set()
            tokens.update(
                f"property:{key}"
                for key in keys
                if left.get(key) != right.get(key)
            )
        elif field == "component_material_slots":
            left_slots = _slot_map(change.before)
            right_slots = _slot_map(change.after)
            tokens.update(
                f"material_slot:{component}:{index}"
                for component, index in set(left_slots) | set(right_slots)
                if left_slots.get((component, index)) != right_slots.get((component, index))
            )
        else:
            tokens.add(f"field:{field}")
    return tuple(sorted(tokens))


def _attribute_authorized(
    change: Change, policy: FrozenEvaluationPolicy
) -> tuple[bool, tuple[str, ...]]:
    tokens = _changed_attribute_tokens(change)
    for target in _targets(change.after, policy):
        allowed = (target.get("allow") or {}).get("attribute", False)
        if allowed is True:
            return True, tokens
        if not isinstance(allowed, Mapping):
            continue
        fields = {f"field:{value}" for value in allowed.get("fields", ())}
        properties = {f"property:{value}" for value in allowed.get("properties", ())}
        slots = {
            f"material_slot:{value.get('component_identity')}:{value.get('slot_index')}"
            for value in allowed.get("material_slots", ())
            if isinstance(value, Mapping)
        }
        if set(tokens) <= fields | properties | slots:
            return True, tokens
    return False, tokens


def _leaf(
    context: Context,
    leaf_id: str,
    *,
    within_policy_limit: bool,
    score: float,
    raw: Mapping[str, Any],
    required_evidence: tuple[str, ...],
    evidence: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    result = contracts.MetricResult(
        id=leaf_id,
        instance_id=f"{context.ids['episode_id']}:{leaf_id}",
        metric_version=_METRIC_VERSION,
        dimension="source_preservation",
        applicability_policy="frozen_evaluation_policy.edit_scope",
        required_evidence=required_evidence,
        applicable=True,
        status="measured",
        coverage=1.0,
        raw=dict(raw),
        score=max(0.0, min(1.0, float(score))),
        normalization_policy="one_minus_policy_violation_rate",
        normalization_parameters={"legacy_policy_limit_used_for_audit_only": True},
        calibration_status="continuous_score_without_pass_fail_threshold",
        contributes_to_aggregate=False,
        evidence=(dict(evidence),),
        failure_reason=None,
    )
    report = {
        **contracts.base(f"source_preservation.{leaf_id}", context.ids),
        "leaf_id": leaf_id,
        "status": contracts.MEASURED,
        "score": result.score,
        "metrics": {
            "result": result.to_json_dict(),
            "within_legacy_policy_limit": within_policy_limit,
            **dict(raw),
        },
        "evidence": dict(evidence),
        "artifacts": dict(artifacts),
        "probes_used": ("input_candidate_scene_diff",),
    }
    return report


def evaluate(
    context: Context, policy: FrozenEvaluationPolicy
) -> list[dict[str, Any]] | dict[str, Any]:
    spec = policy.leaf_spec(context.task, "source_preservation", context.spec)
    child = Context(
        record=context.record,
        task=context.task,
        ids=context.ids,
        bridge=context.bridge,
        scoring=context.scoring,
        images=context.images,
        reference_images=context.reference_images,
        renders=context.renders,
        visual_renders=context.visual_renders,
        artifacts_dir=context.artifacts_dir,
        out_dir=context.out_dir,
        judge_verdict=context.judge_verdict,
        spec=spec,
        cache=context.cache,
    )
    try:
        evidence = ue_evidence.collect(child)
        if evidence.input_scene is None:
            raise ValueError("the frozen policy source_snapshot was not available")
        source = evidence.input_actors()
        candidate = evidence.candidate_actors()
        if not source:
            raise ValueError("the frozen source_snapshot contains no Actors")
        tolerances = policy.edit_scope.get("tolerances") or {}
        diff = diff_scenes(evidence.input_scene, evidence.candidate, tolerances)
    except Exception as exc:  # noqa: BLE001 - evidence refusal is a report
        return error(
            "source_preservation",
            context,
            f"{type(exc).__name__}: {exc}",
        )

    limits = policy.edit_scope.get("limits") or {}
    unauthorized_added = [
        actor for actor in diff.added
        if not _allows_operation(actor, "add", policy)
    ]
    unauthorized_removed = [
        actor for actor in diff.removed
        if not _allows_operation(actor, "remove", policy)
    ]
    moved_details = [
        (change, *_transform_authorized(change, policy))
        for change in diff.moved
    ]
    modified_details = [
        (change, *_attribute_authorized(change, policy))
        for change in diff.modified
    ]
    unauthorized_moved = [value for value in moved_details if not value[1]]
    unauthorized_modified = [value for value in modified_details if not value[1]]
    changed_keys = {
        actor_identity(actor) for actor in (*unauthorized_added, *unauthorized_removed)
    }
    changed_keys.update(change.key for change, _, _ in unauthorized_moved)
    changed_keys.update(change.key for change, _, _ in unauthorized_modified)
    off_target_count = len(changed_keys)

    def cap(name: str) -> int:
        return int(limits.get(name, 0))

    actor_ok = (
        len(diff.added) <= cap("maximum_added_actor_count")
        and len(diff.removed) <= cap("maximum_removed_actor_count")
        and not unauthorized_added
        and not unauthorized_removed
    )
    transform_ok = (
        len(diff.moved) <= cap("maximum_moved_actor_count")
        and not unauthorized_moved
    )
    property_ok = (
        len(diff.modified) <= cap("maximum_modified_actor_count")
        and not unauthorized_modified
    )
    locality_ok = off_target_count <= cap("maximum_off_target_change_count")

    policy_evidence = {
        **policy.evidence(),
        "source_actor_count": len(source),
        "candidate_actor_count": len(candidate),
        "edit_scope_target_count": len(policy.edit_scope.get("targets", ())),
    }
    artifacts = evidence.artifacts()
    actor_denominator = max(len(source), len(candidate), 1)
    touched_ratio = off_target_count / max(len(source), 1)
    changed_source = {
        actor_identity(actor): actor
        for actor in source
        if actor_identity(actor) in changed_keys
    }
    source_volume = sum(volume_cm3(actor) for actor in source)
    changed_volume = sum(volume_cm3(actor) for actor in changed_source.values())
    leaves = [
        _leaf(
            context,
            "actor_set_preserved",
            within_policy_limit=actor_ok,
            score=1.0 - (len(unauthorized_added) + len(unauthorized_removed)) / actor_denominator,
            raw={
                "unit": "actor_count",
                "added_actor_count": len(diff.added),
                "removed_actor_count": len(diff.removed),
                "unauthorized_added_actor_count": len(unauthorized_added),
                "unauthorized_removed_actor_count": len(unauthorized_removed),
                "maximum_added_actor_count": cap("maximum_added_actor_count"),
                "maximum_removed_actor_count": cap("maximum_removed_actor_count"),
                "violations": [
                    {"operation": "add", **actor_summary(actor)}
                    for actor in unauthorized_added
                ] + [
                    {"operation": "remove", **actor_summary(actor)}
                    for actor in unauthorized_removed
                ],
            },
            required_evidence=("candidate_scene_graph", "source_snapshot", "edit_scope"),
            evidence=policy_evidence,
            artifacts=artifacts,
        ),
        _leaf(
            context,
            "transform_scope_preserved",
            within_policy_limit=transform_ok,
            score=1.0 - len(unauthorized_moved) / max(len(source), 1),
            raw={
                "unit": "actor_count",
                "moved_actor_count": len(diff.moved),
                "unauthorized_moved_actor_count": len(unauthorized_moved),
                "maximum_moved_actor_count": cap("maximum_moved_actor_count"),
                "violations": [
                    {
                        "stable_actor_id": change.before.get("stable_actor_id"),
                        "label": change.before.get("label"),
                        "changed_axes": list(axes),
                    }
                    for change, _, axes in unauthorized_moved
                ],
            },
            required_evidence=("candidate_scene_graph", "source_snapshot", "edit_scope"),
            evidence=policy_evidence,
            artifacts=artifacts,
        ),
        _leaf(
            context,
            "property_scope_preserved",
            within_policy_limit=property_ok,
            score=1.0 - len(unauthorized_modified) / max(len(source), 1),
            raw={
                "unit": "actor_count",
                "modified_actor_count": len(diff.modified),
                "unauthorized_modified_actor_count": len(unauthorized_modified),
                "maximum_modified_actor_count": cap("maximum_modified_actor_count"),
                "violations": [
                    {
                        "stable_actor_id": change.before.get("stable_actor_id"),
                        "label": change.before.get("label"),
                        "changed_attributes": list(tokens),
                    }
                    for change, _, tokens in unauthorized_modified
                ],
            },
            required_evidence=("candidate_scene_graph", "source_snapshot", "edit_scope"),
            evidence=policy_evidence,
            artifacts=artifacts,
        ),
        _leaf(
            context,
            "edit_locality",
            within_policy_limit=locality_ok,
            score=(
                1.0 - changed_volume / source_volume
                if source_volume > 0
                else 1.0 - touched_ratio
            ),
            raw={
                "unit": "ratio",
                "off_target_change_count": off_target_count,
                "maximum_off_target_change_count": cap("maximum_off_target_change_count"),
                "off_target_changed_actor_ratio": touched_ratio,
                "off_target_changed_volume_cm3": changed_volume,
                "off_target_changed_volume_ratio": (
                    changed_volume / source_volume if source_volume > 0 else 0.0
                ),
            },
            required_evidence=("candidate_scene_graph", "source_snapshot", "edit_scope"),
            evidence=policy_evidence,
            artifacts=artifacts,
        ),
        _leaf(
            context,
            "off_target_change",
            within_policy_limit=locality_ok,
            score=1.0 - touched_ratio,
            raw={
                "unit": "actor_count",
                "off_target_change_count": off_target_count,
                "maximum_off_target_change_count": cap("maximum_off_target_change_count"),
                "changed_actor_ids": sorted(changed_keys),
            },
            required_evidence=("candidate_scene_graph", "source_snapshot", "edit_scope"),
            evidence=policy_evidence,
            artifacts=artifacts,
        ),
    ]
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for leaf in leaves
        for value in (leaf.get("metrics") or {}).values()
    ):
        return error("source_preservation", context, "non-finite preservation metric")
    return leaves


__all__ = ["evaluate"]
