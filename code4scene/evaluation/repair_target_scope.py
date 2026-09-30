"""Task-aware repair target scope shared by scene_diff and render planning.

The scope is frozen before Candidate inspection: it is the deterministic
Input-to-GT Actor difference.  Candidate is used only after that boundary to
identify the population that attempted the repair and to project the already
computed full-scene correspondence onto those targets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .assignment import match
from .context import read_label
from .evaluation_policy import load_evaluation_policy, task_mode
from .gt_geometry_compare import (
    actor_pair_attribute_metrics,
    actor_pair_metrics,
)
from .requirement_graph.repair_target_authoring import (
    RepairTarget,
    derive_repair_targets,
)
from .scene_diff import actor_identity, diff_scenes, index_actors
from .scene_geometry import center_cm, extent_cm
from .ue_evidence import document


def _actors(scene: Mapping[str, Any], name: str) -> tuple[Mapping[str, Any], ...]:
    values = scene.get("actors")
    if not isinstance(values, list) or any(
        not isinstance(value, Mapping) for value in values
    ):
        raise ValueError(f"{name}.actors must be an array of Actor objects")
    return tuple(values)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _unit(value: float | None) -> float | None:
    return None if value is None else max(0.0, min(1.0, value))


def _mean(values: Sequence[float | None]) -> float | None:
    measured = [float(value) for value in values if value is not None]
    return sum(measured) / len(measured) if measured else None


def _rmse(values: Sequence[float]) -> float | None:
    return (
        math.sqrt(sum(value * value for value in values) / len(values))
        if values
        else None
    )


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _normalized_path(value: Any) -> str:
    text = str(value or "").strip()
    if "'" in text and text.endswith("'"):
        text = text.split("'", 1)[1][:-1]
    return text.casefold()


MATCHING_ALGORITHM_VERSION = "repair-target-actor-correspondence.v2"
_INCOMPATIBLE_COST = 1_000_000.0
_MATCH_COST_WEIGHTS = {
    "aligned_position": 0.40,
    "rotation": 0.10,
    "scale": 0.10,
    "bounds_size": 0.15,
    "footprint": 0.10,
    "local_relation": 0.13,
    "identity_hint": 0.02,
}
_HINT_FIELDS = (
    "stable_actor_id",
    "actor_guid",
    "name",
    "label",
    "actor_path",
)


def _normalized_paths(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        sorted(
            {
                normalized
                for item in value
                if (normalized := _normalized_path(item))
            }
        )
    )


def _actor_descriptor(actor: Mapping[str, Any]) -> dict[str, Any]:
    """Structural identity used only to open or reject assignment edges."""

    actor_type = _normalized_path(actor.get("class"))
    component_assets = _normalized_paths(actor.get("component_asset_paths"))
    asset = _normalized_path(actor.get("asset_path"))
    blueprint_class = actor_type if actor_type.startswith("/game/") else None
    static_mesh = (
        asset
        if asset and blueprint_class is None
        else None
    )
    return {
        "actor_type": actor_type or None,
        "static_mesh": static_mesh,
        "blueprint_class": blueprint_class,
        "component_asset_signature": list(component_assets),
    }


def _descriptor_key(actor: Mapping[str, Any]) -> tuple[Any, ...]:
    descriptor = _actor_descriptor(actor)
    return (
        descriptor["actor_type"] or "",
        descriptor["static_mesh"] or "",
        descriptor["blueprint_class"] or "",
        tuple(descriptor["component_asset_signature"]),
    )


def _actor_order_key(actor: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _descriptor_key(actor),
        *[round(value, 6) for value in center_cm(actor)],
        _normalized_path(actor.get("stable_actor_id")),
        _normalized_path(actor.get("actor_guid")),
        _normalized_path(actor.get("name")),
        _normalized_path(actor.get("label")),
        _normalized_path(actor.get("actor_path")),
    )


def _actor_object_id(actor: Mapping[str, Any], side: str, index: int) -> str:
    stable = str(actor.get("stable_actor_id") or "").strip()
    return stable or f"{side}:repair_actor:{index:04d}"


def _actor_summary(
    actor: Mapping[str, Any],
    *,
    side: str,
    index: int,
) -> dict[str, Any]:
    stable = str(actor.get("stable_actor_id") or "").strip()
    return {
        "object_id": _actor_object_id(actor, side, index),
        "stable_actor_ids": [stable] if stable else [],
        "label": actor.get("label"),
        "name": actor.get("name"),
        "actor_guid": actor.get("actor_guid"),
        "actor_path": actor.get("actor_path"),
        "asset_paths": (
            [str(actor.get("asset_path"))]
            if str(actor.get("asset_path") or "").strip()
            else []
        ),
        "class": actor.get("class"),
        "component_asset_paths": list(
            actor.get("component_asset_paths") or ()
        ),
        "center_cm": [_rounded(value) for value in center_cm(actor)],
        "size_cm": [
            _rounded(value * 2.0) for value in extent_cm(actor)
        ],
        "signature": _actor_descriptor(actor),
    }


def _identity_hints(actor: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in _HINT_FIELDS:
        value = _normalized_path(actor.get(field))
        if field == "actor_path" and ":persistentlevel." in value:
            value = value.rsplit(":persistentlevel.", 1)[1]
        if value:
            values[field] = value
    return values


def _hard_compatibility(
    candidate: Mapping[str, Any],
    canonical: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a provenance-rich hard compatibility decision.

    Structural asset and type evidence controls edge creation. Runtime
    identity fields never reject an edge and are intentionally absent here.
    """

    left = _actor_descriptor(candidate)
    right = _actor_descriptor(canonical)
    checks: dict[str, str] = {}
    conflicts: list[str] = []
    supporting: list[str] = []

    left_type = left["actor_type"]
    right_type = right["actor_type"]
    if left_type and right_type:
        if left_type == right_type:
            checks["actor_type"] = "equal"
        else:
            checks["actor_type"] = "conflict"
            conflicts.append("actor_type_mismatch")
    else:
        checks["actor_type"] = "missing_on_one_or_both_sides"

    for field in (
        "static_mesh",
        "blueprint_class",
        "component_asset_signature",
    ):
        left_value = left[field]
        right_value = right[field]
        left_present = bool(left_value)
        right_present = bool(right_value)
        if left_present and right_present:
            if left_value == right_value:
                checks[field] = "equal"
                supporting.append(field)
            else:
                checks[field] = "conflict"
                conflicts.append(f"{field}_mismatch")
        else:
            checks[field] = "missing_on_one_or_both_sides"

    structural_present = any(
        bool(left[field]) or bool(right[field])
        for field in (
            "static_mesh",
            "blueprint_class",
            "component_asset_signature",
        )
    )
    if conflicts:
        compatible = False
        reason = "hard_structural_conflict"
    elif supporting:
        compatible = True
        reason = "shared_structural_signature"
    elif not structural_present and left_type and left_type == right_type:
        compatible = True
        reason = "shared_actor_type_without_asset_evidence"
        supporting.append("actor_type")
    else:
        compatible = False
        reason = "insufficient_shared_structural_signature"
    return {
        "compatible": compatible,
        "reason": reason,
        "supporting_fields": supporting,
        "rejection_reasons": conflicts or ([] if compatible else [reason]),
        "checks": checks,
    }


def _candidate_resolution(
    scope: RepairTargetScope,
    candidate_scene: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, tuple[str, ...]]]:
    """Resolve Actors that actually attempted an Input-to-GT repair."""

    difference = diff_scenes(scope.input_scene, candidate_scene)
    selected: dict[str, Mapping[str, Any]] = {}
    reasons: dict[str, set[str]] = {}

    def add(actor: Mapping[str, Any], reason: str) -> None:
        key = actor_identity(actor)
        selected[key] = actor
        reasons.setdefault(key, set()).add(reason)

    for actor in difference.added:
        add(actor, "input_to_candidate_added")
    for change in difference.moved:
        add(change.after, "input_to_candidate_moved")
    for change in difference.modified:
        add(change.after, "input_to_candidate_modified")

    candidate_by_identity = index_actors(candidate_scene)
    for target in scope.targets:
        for source_name, actor in (
            ("input_target_identity", target.input_actor),
            ("gt_target_identity", target.desired_actor),
        ):
            if actor is None:
                continue
            current = candidate_by_identity.get(actor_identity(actor))
            if current is not None:
                add(current, source_name)

    actors = sorted(selected.values(), key=_actor_order_key)
    return actors, {
        actor_identity(actor): tuple(sorted(reasons[actor_identity(actor)]))
        for actor in actors
    }


def _bounded(value: float | None) -> float | None:
    return None if value is None else value / (1.0 + value)


def _relation_profile(
    actor: Mapping[str, Any],
    population: Sequence[Mapping[str, Any]],
    scale: float,
) -> tuple[float, ...]:
    center = center_cm(actor)
    return tuple(
        sorted(
            math.dist(center, center_cm(other)) / scale
            for other in population
            if other is not actor
        )
    )


def _relation_cost(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if not left and not right:
        return None
    paired = min(len(left), len(right))
    error = (
        math.sqrt(
            sum((left[index] - right[index]) ** 2 for index in range(paired))
            / paired
        )
        if paired
        else 1.0
    )
    count_penalty = abs(len(left) - len(right)) / max(len(left), len(right), 1)
    return _bounded(error) * 0.75 + count_penalty * 0.25


def _pair_cost(
    candidate: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    pair_metrics: Mapping[str, Any],
    candidate_relation: Sequence[float],
    canonical_relation: Sequence[float],
    target_scale: float,
) -> tuple[
    float,
    dict[str, float | None],
    list[str],
    dict[str, dict[str, str | None]],
]:
    candidate_hints = _identity_hints(candidate)
    canonical_hints = _identity_hints(canonical)
    hint_matches = sorted(
        field
        for field in set(candidate_hints) & set(canonical_hints)
        if candidate_hints[field] == canonical_hints[field]
    )
    hint_differences = {
        field: {
            "candidate": candidate_hints.get(field),
            "canonical": canonical_hints.get(field),
        }
        for field in sorted(set(candidate_hints) | set(canonical_hints))
        if candidate_hints.get(field) != canonical_hints.get(field)
    }
    hint_cost = (
        0.0 if hint_matches else 1.0
        if candidate_hints and canonical_hints
        else None
    )
    components = {
        "aligned_position": _bounded(
            float(pair_metrics["aligned_center_distance_cm"]) / target_scale
            if pair_metrics.get("aligned_center_distance_cm") is not None
            else None
        ),
        "rotation": (
            min(
                1.0,
                float(pair_metrics["aligned_rotation_error_deg"]) / 180.0,
            )
            if pair_metrics.get("aligned_rotation_error_deg") is not None
            else None
        ),
        "scale": _bounded(
            float(pair_metrics["scale_log_error"])
            if pair_metrics.get("scale_log_error") is not None
            else None
        ),
        "bounds_size": _bounded(
            float(pair_metrics["bounds_size_log_rmse"])
            if pair_metrics.get("bounds_size_log_rmse") is not None
            else None
        ),
        "footprint": (
            1.0 - float(pair_metrics["footprint_iou"])
            if pair_metrics.get("footprint_iou") is not None
            else None
        ),
        "local_relation": _relation_cost(
            candidate_relation, canonical_relation
        ),
        "identity_hint": hint_cost,
    }
    available = [
        (_MATCH_COST_WEIGHTS[name], value)
        for name, value in components.items()
        if value is not None
    ]
    denominator = sum(weight for weight, _ in available)
    cost = (
        sum(weight * float(value) for weight, value in available)
        / denominator
        if denominator
        else 1.0
    )
    return cost, components, hint_matches, hint_differences


def _independent_target_correspondence(
    scope: RepairTargetScope,
    candidate_scene: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    target_scale: float,
) -> tuple[
    list[Mapping[str, Any]],
    list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    list[Mapping[str, Any]],
    dict[str, Any],
]:
    """Match GT repair Actors directly to compatible Candidate attempts."""

    desired = sorted(scope.desired_actors, key=_actor_order_key)
    desired_operations = {
        actor_identity(target.desired_actor): target.operation
        for target in scope.targets
        if target.desired_actor is not None
    }
    resolution_pool, resolution_reasons = _candidate_resolution(
        scope, candidate_scene
    )
    alignment = (comparison.get("audit") or {}).get("alignment")
    alignment = alignment if isinstance(alignment, Mapping) else None

    compatibility: list[list[dict[str, Any]]] = []
    compatible_candidate_indexes: set[int] = set()
    for candidate_index, candidate in enumerate(resolution_pool):
        row: list[dict[str, Any]] = []
        for canonical in desired:
            decision = _hard_compatibility(candidate, canonical)
            row.append(decision)
            if decision["compatible"]:
                compatible_candidate_indexes.add(candidate_index)
        compatibility.append(row)
    candidates = [
        resolution_pool[index] for index in sorted(compatible_candidate_indexes)
    ]
    resolution_index = {
        id(actor): index for index, actor in enumerate(resolution_pool)
    }
    candidate_profiles = [
        _relation_profile(actor, candidates, target_scale)
        for actor in candidates
    ]
    desired_profiles = [
        _relation_profile(actor, desired, target_scale) for actor in desired
    ]
    pair_metrics_grid: list[list[dict[str, Any]]] = []
    cost_components: list[list[dict[str, float | None]]] = []
    hint_matches: list[list[list[str]]] = []
    hint_differences: list[list[dict[str, Any]]] = []
    cost_matrix: list[list[float]] = []
    compatibility_matrix: list[list[dict[str, Any]]] = []
    for candidate_index, candidate in enumerate(candidates):
        source_index = resolution_index[id(candidate)]
        metric_row: list[dict[str, Any]] = []
        component_row: list[dict[str, float | None]] = []
        hint_row: list[list[str]] = []
        hint_difference_row: list[dict[str, Any]] = []
        cost_row: list[float] = []
        compatible_row: list[dict[str, Any]] = []
        for canonical_index, canonical in enumerate(desired):
            decision = compatibility[source_index][canonical_index]
            metrics = actor_pair_metrics(
                candidate,
                canonical,
                alignment_audit=alignment,
            )
            cost, components, hints, differences = _pair_cost(
                candidate,
                canonical,
                pair_metrics=metrics,
                candidate_relation=candidate_profiles[candidate_index],
                canonical_relation=desired_profiles[canonical_index],
                target_scale=target_scale,
            )
            metric_row.append(metrics)
            component_row.append(components)
            hint_row.append(hints)
            hint_difference_row.append(differences)
            cost_row.append(cost if decision["compatible"] else _INCOMPATIBLE_COST)
            compatible_row.append(decision)
        pair_metrics_grid.append(metric_row)
        cost_components.append(component_row)
        hint_matches.append(hint_row)
        hint_differences.append(hint_difference_row)
        cost_matrix.append(cost_row)
        compatibility_matrix.append(compatible_row)

    assigned = [
        (candidate_index, canonical_index)
        for candidate_index, canonical_index in match(cost_matrix)
        if compatibility_matrix[candidate_index][canonical_index]["compatible"]
    ] if cost_matrix and desired else []
    actor_rows: list[Mapping[str, Any]] = []
    actor_pairs: list[
        tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = []
    assignment_audit: list[dict[str, Any]] = []
    for candidate_index, canonical_index in assigned:
        candidate = candidates[candidate_index]
        canonical = desired[canonical_index]
        metrics = pair_metrics_grid[candidate_index][canonical_index]
        attribute_metrics = actor_pair_attribute_metrics(candidate, canonical)
        candidate_summary = _actor_summary(
            candidate, side="candidate", index=candidate_index
        )
        canonical_summary = _actor_summary(
            canonical, side="canonical", index=canonical_index
        )
        row = {
            "candidate": candidate_summary,
            "canonical": canonical_summary,
            "identity_cost": 0.0,
            "identity_source": "hard_compatible_global_min_cost_assignment",
            "structured_identity_match": True,
            **metrics,
            **attribute_metrics,
            "matching_cost": _rounded(
                cost_matrix[candidate_index][canonical_index]
            ),
            "matching_cost_components": {
                key: _rounded(value)
                for key, value in cost_components[
                    candidate_index
                ][canonical_index].items()
            },
            "identity_hint_matches": hint_matches[
                candidate_index
            ][canonical_index],
            "identity_hint_differences": hint_differences[
                candidate_index
            ][canonical_index],
        }
        actor_rows.append(row)
        actor_pairs.append((candidate, canonical))
        assignment_audit.append({
            "candidate": candidate_summary,
            "canonical": canonical_summary,
            "matching_cost": row["matching_cost"],
            "matching_cost_components": row["matching_cost_components"],
            "identity_hint_matches": row["identity_hint_matches"],
            "identity_hint_differences": row[
                "identity_hint_differences"
            ],
            "compatibility": compatibility_matrix[
                candidate_index
            ][canonical_index],
        })

    assigned_candidate = {row for row, _ in assigned}
    assigned_canonical = {column for _, column in assigned}
    ambiguous_targets: list[dict[str, Any]] = []
    for canonical_index, canonical in enumerate(desired):
        options = sorted(
            (
                cost_matrix[candidate_index][canonical_index],
                candidate_index,
            )
            for candidate_index in range(len(candidates))
            if compatibility_matrix[candidate_index][canonical_index][
                "compatible"
            ]
        )
        if len(options) >= 2 and math.isclose(
            options[0][0], options[1][0], rel_tol=0.0, abs_tol=1e-9
        ):
            ambiguous_targets.append({
                "canonical": _actor_summary(
                    canonical, side="canonical", index=canonical_index
                ),
                "minimum_cost": _rounded(options[0][0]),
                "tied_candidate_count": sum(
                    math.isclose(
                        value[0],
                        options[0][0],
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    for value in options
                ),
            })

    compatibility_candidates: list[dict[str, Any]] = []
    for canonical_index, canonical in enumerate(desired):
        values = []
        for source_index, candidate in enumerate(resolution_pool):
            values.append({
                "candidate": _actor_summary(
                    candidate, side="candidate_pool", index=source_index
                ),
                "target_resolution_reasons": list(
                    resolution_reasons[actor_identity(candidate)]
                ),
                **compatibility[source_index][canonical_index],
            })
        compatibility_candidates.append({
            "target": _actor_summary(
                canonical, side="canonical", index=canonical_index
            ),
            "candidates": values,
        })

    all_scene_compatible_outside_resolution: list[dict[str, Any]] = []
    resolved_identities = {
        actor_identity(actor) for actor in resolution_pool
    }
    for scene_index, candidate in enumerate(
        sorted(_actors(candidate_scene, "candidate_scene"), key=_actor_order_key)
    ):
        if actor_identity(candidate) in resolved_identities:
            continue
        matching_targets = []
        for canonical_index, canonical in enumerate(desired):
            if not _hard_compatibility(candidate, canonical)["compatible"]:
                continue
            metrics = actor_pair_metrics(
                candidate,
                canonical,
                alignment_audit=alignment,
            )
            candidate_hints = _identity_hints(candidate)
            canonical_hints = _identity_hints(canonical)
            shared_hints = sorted(
                field
                for field in set(candidate_hints) & set(canonical_hints)
                if candidate_hints[field] == canonical_hints[field]
            )
            distance = metrics.get("aligned_center_distance_cm")
            plausible = bool(shared_hints) or (
                distance is not None
                and float(distance) <= target_scale * 3.0
            )
            # An unchanged pre-existing instance cannot satisfy a frozen
            # add target; doing so would erase the required Actor-count delta.
            if desired_operations.get(actor_identity(canonical)) == "add":
                plausible = False
            matching_targets.append({
                "target_index": canonical_index,
                "aligned_center_distance_cm": distance,
                "identity_hint_matches": shared_hints,
                "plausible_target_resolution_miss": plausible,
            })
        if matching_targets:
            all_scene_compatible_outside_resolution.append({
                "candidate": _actor_summary(
                    candidate, side="candidate_scene", index=scene_index
                ),
                "compatible_targets": matching_targets,
                "filter_reason": "unchanged_and_not_a_frozen_target_identity",
            })

    groups: dict[tuple[Any, ...], int] = {}
    for canonical in desired:
        key = _descriptor_key(canonical)
        groups[key] = groups.get(key, 0) + 1
    exchangeable_group_count = sum(value > 1 for value in groups.values())
    target_stable_ids = {
        str(actor.get("stable_actor_id") or "").strip()
        for actor in desired
        if str(actor.get("stable_actor_id") or "").strip()
    }
    reassigned_actor_count = sum(
        bool(target_stable_ids)
        and str(candidate.get("stable_actor_id") or "").strip()
        != str(canonical.get("stable_actor_id") or "").strip()
        for candidate, canonical in actor_pairs
    )
    permuted_target_identity_count = sum(
        str(candidate.get("stable_actor_id") or "").strip()
        in target_stable_ids
        and str(candidate.get("stable_actor_id") or "").strip()
        != str(canonical.get("stable_actor_id") or "").strip()
        for candidate, canonical in actor_pairs
    )
    classifications: set[str] = set()
    if reassigned_actor_count:
        classifications.add("identity_changed")
    if exchangeable_group_count:
        classifications.add("same_class_multi_instance")
    if exchangeable_group_count and permuted_target_identity_count:
        classifications.add("exchange_or_permutation")
    if len(actor_pairs) < len(desired):
        classifications.add("missing_or_wrong_asset")
    plausible_resolution_miss = any(
        target["plausible_target_resolution_miss"]
        for row in all_scene_compatible_outside_resolution
        for target in row["compatible_targets"]
    )
    if plausible_resolution_miss and len(actor_pairs) < len(desired):
        classifications.add("target_resolution_excluded_compatible_actor")

    near_incompatible_attempt = any(
        distance <= target_scale * 3.0
        for candidate in resolution_pool
        for canonical in desired
        if not _hard_compatibility(candidate, canonical)["compatible"]
        and (
            distance := _number(actor_pair_metrics(
                candidate,
                canonical,
                alignment_audit=alignment,
            ).get("aligned_center_distance_cm")
            )
        ) is not None
    )
    if len(actor_pairs) == len(desired):
        if exchangeable_group_count and permuted_target_identity_count:
            primary_classification = "exchange_or_permutation"
        elif exchangeable_group_count:
            primary_classification = "same_class_multi_instance"
        elif reassigned_actor_count:
            primary_classification = "identity_changed"
        else:
            primary_classification = "matched_without_identity_change"
    elif actor_pairs:
        primary_classification = "partial_missing_actor"
    elif plausible_resolution_miss:
        primary_classification = "target_resolution_selected_wrong_actor"
    elif near_incompatible_attempt:
        primary_classification = "wrong_asset"
    else:
        primary_classification = "missing_actor"

    audit = {
        "matching_algorithm_version": MATCHING_ALGORITHM_VERSION,
        "correspondence_policy": (
            "input_gt_frozen_targets_then_hard_compatibility_global_assignment"
        ),
        "matching_cost_formula": dict(_MATCH_COST_WEIGHTS),
        "hard_compatibility_fields": [
            "static_mesh",
            "blueprint_class",
            "component_asset_signature",
            "actor_type",
        ],
        "name_guid_path_used_as_hint_only": True,
        "target_actor_signatures": [
            _actor_summary(actor, side="canonical", index=index)
            for index, actor in enumerate(desired)
        ],
        "target_resolution": {
            "policy": (
                "input_to_candidate_changes_plus_frozen_target_identities"
            ),
            "candidate_count": len(resolution_pool),
            "candidates": [
                {
                    "actor": _actor_summary(
                        actor, side="candidate_pool", index=index
                    ),
                    "reasons": list(
                        resolution_reasons[actor_identity(actor)]
                    ),
                }
                for index, actor in enumerate(resolution_pool)
            ],
            "compatible_outside_resolution": (
                all_scene_compatible_outside_resolution
            ),
        },
        "compatibility_candidates": compatibility_candidates,
        "final_assignment": assignment_audit,
        "unmatched_gt_actors": [
            _actor_summary(actor, side="canonical", index=index)
            for index, actor in enumerate(desired)
            if index not in assigned_canonical
        ],
        "unmatched_candidate_actors": [
            _actor_summary(actor, side="candidate", index=index)
            for index, actor in enumerate(candidates)
            if index not in assigned_candidate
        ],
        "ambiguity": {
            "ambiguous": bool(ambiguous_targets),
            "target_count": len(ambiguous_targets),
            "targets": ambiguous_targets,
            "deterministic_tie_break": "canonical_actor_sort_then_hungarian",
        },
        "failure_classification": sorted(classifications),
        "primary_classification": primary_classification,
        # Backward-compatible provenance aliases for saved recompute readers.
        "exchangeable_assignment_cost": (
            "weighted_pose_size_relation_cost_with_identity_hint"
        ),
        "exchangeable_group_count": exchangeable_group_count,
        "exchangeable_reassigned_actor_count": reassigned_actor_count,
        "permuted_target_identity_count": permuted_target_identity_count,
    }
    return actor_rows, actor_pairs, candidates, audit


@dataclass(frozen=True, slots=True)
class RepairTargetScope:
    """The frozen Actor-level region that a repair task is supposed to change."""

    input_scene: Mapping[str, Any]
    canonical_scene: Mapping[str, Any]
    targets: tuple[RepairTarget, ...]

    @property
    def desired_actors(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            value.desired_actor
            for value in self.targets
            if value.desired_actor is not None
        )

    @property
    def source_actors(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            value.input_actor
            for value in self.targets
            if value.input_actor is not None
        )

    @property
    def focus_actors(self) -> tuple[Mapping[str, Any], ...]:
        """GT-space actors used to frame the local paired visual evidence."""

        return self.desired_actors or self.source_actors

    def audit(self) -> dict[str, Any]:
        counts = {
            operation: sum(value.operation == operation for value in self.targets)
            for operation in ("add", "remove", "repair")
        }
        return {
            "derivation": "deterministic_gt_minus_input_actor_diff",
            "target_count": len(self.targets),
            "operation_counts": counts,
            "targets": [value.to_dict() for value in self.targets],
        }


def is_repair_target_task(task: Any) -> bool:
    """Whether a task declares the evidence needed for a local repair scope."""

    declared = {
        str(value.get("name"))
        for value in (getattr(task, "verifiers", None) or ())
        if isinstance(value, Mapping)
    }
    if (
        getattr(task, "case_type", "prompt_to_scene") != "image_to_scene"
        and "gt_repair" not in declared
    ):
        return False
    if task_mode(getattr(task, "kind", "")) == "generation":
        return False
    try:
        return load_evaluation_policy(task).source_snapshot is not None
    except Exception:  # noqa: BLE001 - evidence planning reports the real error later
        return False


def load_repair_target_scope(
    task: Any,
    *,
    label: Mapping[str, Any] | None = None,
    input_scene: Mapping[str, Any] | None = None,
) -> RepairTargetScope | None:
    """Load and deterministically derive the task's GT-minus-Input target set."""

    if not is_repair_target_task(task):
        return None
    policy = load_evaluation_policy(task)
    if input_scene is None:
        loaded = document(
            SimpleNamespace(task=task),
            policy.source_snapshot,
            "repair target Input scene graph",
        )
        if not isinstance(loaded, Mapping):
            raise TypeError("repair target Input scene graph must be an object")
        input_scene = loaded
    if label is None:
        label = read_label(SimpleNamespace(task=task))
    if not isinstance(label, Mapping):
        raise ValueError("repair target scene_diff needs an answer-key label")
    canonical = label.get("canonical_actors")
    if not isinstance(canonical, list) or any(
        not isinstance(value, Mapping) for value in canonical
    ):
        raise ValueError("repair target label needs canonical_actors")
    canonical_scene = {
        "actors": canonical,
        "actor_count": len(canonical),
        "export_metadata": {
            "status": "success",
            "source": "scene_diff_answer_key",
        },
    }
    _actors(input_scene, "input_scene")
    targets = derive_repair_targets(input_scene, canonical_scene)
    if not targets:
        return None
    return RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=canonical_scene,
        targets=targets,
    )


def candidate_target_actors(
    scope: RepairTargetScope,
    candidate_scene: Mapping[str, Any],
    comparison: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return structurally compatible Actors that attempted the frozen repair.

    ``comparison`` remains accepted for API compatibility, but full-scene
    correspondence is deliberately not consulted.
    """

    del comparison
    pool, _reasons = _candidate_resolution(scope, candidate_scene)
    return tuple(
        actor
        for actor in pool
        if any(
            _hard_compatibility(actor, desired)["compatible"]
            for desired in scope.desired_actors
        )
    )


def _target_scale_cm(scope: RepairTargetScope) -> float:
    sizes = sorted(
        component * 2.0
        for actor in scope.focus_actors
        for component in extent_cm(actor)
        if component > 0.0
    )
    return max(1.0, sizes[len(sizes) // 2]) if sizes else 1.0


def _layout_error(rows: Sequence[Mapping[str, Any]], scale: float) -> float | None:
    if len(rows) < 2:
        return None
    errors: list[float] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            candidate_left = rows[left].get("candidate") or {}
            candidate_right = rows[right].get("candidate") or {}
            canonical_left = rows[left].get("canonical") or {}
            canonical_right = rows[right].get("canonical") or {}
            candidate_distance = math.dist(
                candidate_left.get("center_cm") or (0.0, 0.0, 0.0),
                candidate_right.get("center_cm") or (0.0, 0.0, 0.0),
            )
            canonical_distance = math.dist(
                canonical_left.get("center_cm") or (0.0, 0.0, 0.0),
                canonical_right.get("center_cm") or (0.0, 0.0, 0.0),
            )
            errors.append(abs(candidate_distance - canonical_distance) / scale)
    return sum(errors) / len(errors) if errors else None


def measure_repair_target(
    scope: RepairTargetScope,
    candidate_scene: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one full-scene correspondence onto the frozen repair targets."""

    global_metrics = comparison.get("metrics") or {}
    target_scale = _target_scale_cm(scope)
    (
        actor_rows,
        _actor_pairs,
        candidate_population,
        correspondence_audit,
    ) = _independent_target_correspondence(
        scope,
        candidate_scene,
        comparison,
        target_scale=target_scale,
    )
    desired_count = len(scope.desired_actors)
    candidate_count = len(candidate_population)
    matched_count = len(actor_rows)
    restoration_f1 = (
        2.0 * matched_count / (desired_count + candidate_count)
        if desired_count + candidate_count
        else None
    )

    candidate_by_identity = index_actors(candidate_scene)
    removal_targets = [
        value for value in scope.targets if value.operation == "remove"
    ]
    remaining_removals = sum(
        actor_identity(value.input_actor) in candidate_by_identity
        for value in removal_targets
        if value.input_actor is not None
    )
    removal_score = (
        1.0 - remaining_removals / len(removal_targets)
        if removal_targets
        else None
    )
    operation_count = desired_count + len(removal_targets)
    correspondence_score = (
        (
            (restoration_f1 or 0.0) * desired_count
            + (removal_score or 0.0) * len(removal_targets)
        )
        / operation_count
        if operation_count
        else None
    )
    identity_rate = _mean(
        [
            1.0 if value.get("structured_identity_match") is True else 0.0
            for value in actor_rows
        ]
    )
    if identity_rate is None and desired_count:
        identity_rate = 0.0
    pair_coverage = matched_count / desired_count if desired_count else None

    positions = [
        value
        for row in actor_rows
        if (value := _number(row.get("aligned_center_distance_cm"))) is not None
    ]
    rotations = [
        value
        for row in actor_rows
        if (value := _number(row.get("aligned_rotation_error_deg"))) is not None
    ]
    scale_errors = [
        value
        for row in actor_rows
        if (value := _number(row.get("scale_log_error"))) is not None
    ]
    position_rmse = _rmse(positions)
    rotation_mean = _mean(rotations)
    scale_rmse = _rmse(scale_errors)
    coverage = pair_coverage or 0.0
    position_similarity = (
        1.0 / (1.0 + position_rmse / target_scale) * coverage
        if position_rmse is not None
        else None
    )
    rotation_similarity = (
        max(0.0, 1.0 - rotation_mean / 180.0) * coverage
        if rotation_mean is not None
        else None
    )
    scale_similarity = (
        math.exp(-scale_rmse) * coverage if scale_rmse is not None else None
    )

    footprints = [
        value
        for row in actor_rows
        if (value := _number(row.get("footprint_iou"))) is not None
    ]
    bounds_errors = [
        value
        for row in actor_rows
        if (value := _number(row.get("bounds_size_log_rmse"))) is not None
    ]
    footprint_similarity = (
        _mean(footprints) * coverage if footprints else None
    )
    bounds_rmse = _rmse(bounds_errors)
    bounds_similarity = (
        math.exp(-bounds_rmse) * coverage if bounds_rmse is not None else None
    )
    # The frozen repair target is Actor-level. Never collapse repeated target
    # Actors into a logical object: correspondence already accounts for
    # missing and extra Actors, while layout measures their relative geometry.
    layout_error = _layout_error(actor_rows, target_scale)
    layout_similarity = (
        1.0 / (1.0 + layout_error) * coverage
        if layout_error is not None
        else None
    )

    property_rate = _mean(
        [
            1.0 if row.get("task_relevant_properties_match") is True else 0.0
            for row in actor_rows
            if row.get("task_relevant_properties_match") is not None
        ]
    )
    material_rate = _mean(
        [
            1.0 if row.get("material_set_match") is True else 0.0
            for row in actor_rows
            if row.get("material_set_match") is not None
        ]
    )
    slot_rate = _mean(
        [
            1.0 if row.get("component_material_slots_match") is True else 0.0
            for row in actor_rows
            if row.get("component_material_slots_match") is not None
        ]
    )
    attribute_similarity = _mean([property_rate, material_rate, slot_rate])
    if attribute_similarity is not None:
        attribute_similarity *= coverage

    # Diagnostic execution must never turn an otherwise valid continuous
    # measurement into an error or change any score used by the benchmark.
    from . import repair_success
    try:
        success = repair_success.measure(scope, candidate_scene)
    except Exception as exc:  # report-only evidence failure
        success = {"policy_id": repair_success.POLICY_ID, "status": "not_evaluated",
                   "failure_reason": f"{type(exc).__name__}: {exc}"}
    return {
        "repair_success": success,
        "scope_schema_version": "scene-diff-repair-target-v1",
        "target_derivation": scope.audit(),
        "normalization_scale_cm": round(target_scale, 3),
        "desired_actor_count": desired_count,
        "candidate_target_actor_count": candidate_count,
        "matched_actor_count": matched_count,
        "missing_actor_count": max(0, desired_count - matched_count),
        "extra_actor_count": max(0, candidate_count - matched_count),
        "removal_target_count": len(removal_targets),
        "remaining_removal_count": remaining_removals,
        "pair_coverage": _rounded(pair_coverage),
        "actor_correspondence_score": _rounded(_unit(correspondence_score)),
        "identity_match_rate": _rounded(identity_rate),
        "identity_score": _rounded(
            _unit(correspondence_score * identity_rate)
            if correspondence_score is not None and identity_rate is not None
            else None
        ),
        "position_rmse_cm": _rounded(position_rmse),
        "position_mean_cm": _rounded(_mean(positions)),
        "position_similarity": _rounded(_unit(position_similarity)),
        "rotation_mean_deg": _rounded(rotation_mean),
        "rotation_similarity": _rounded(_unit(rotation_similarity)),
        "scale_log_rmse": _rounded(scale_rmse),
        "scale_similarity": _rounded(_unit(scale_similarity)),
        "mean_footprint_iou": _rounded(_mean(footprints)),
        "footprint_similarity": _rounded(_unit(footprint_similarity)),
        "bounds_size_log_rmse": _rounded(bounds_rmse),
        "bounds_size_similarity": _rounded(_unit(bounds_similarity)),
        "pairwise_layout_error": _rounded(layout_error),
        "pairwise_layout_pair_count": (
            len(actor_rows) * (len(actor_rows) - 1) // 2
        ),
        "pairwise_layout_similarity": _rounded(_unit(layout_similarity)),
        "task_property_match_rate": _rounded(property_rate),
        "material_set_match_rate": _rounded(material_rate),
        "material_slot_match_rate": _rounded(slot_rate),
        "attribute_similarity": _rounded(_unit(attribute_similarity)),
        "global_alignment": {
            "anchor_count": global_metrics.get("alignment_anchor_count"),
            "translation_error_cm": global_metrics.get(
                "global_translation_error_cm"
            ),
            "yaw_error_deg": global_metrics.get("global_yaw_error_deg"),
            "policy": "reuse_full_scene_identity_anchored_alignment",
        },
        **correspondence_audit,
        "matched_actor_rows": actor_rows,
    }


__all__ = [
    "RepairTargetScope",
    "MATCHING_ALGORITHM_VERSION",
    "candidate_target_actors",
    "is_repair_target_task",
    "load_repair_target_scope",
    "measure_repair_target",
]
