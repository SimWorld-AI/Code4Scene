"""Frozen default evaluation policy, separate from task semantics.

RequirementGraph owns what the prompt or reference image asks for. This module
owns checks that run even when the prompt says nothing about them: candidate
integrity, physical safety, and conditional Input preservation. The executable
policy is code-owned and serialized as a schema-validated JSON object. Source
preservation is routed only
for image-guided edit/repair tasks that begin from an editable Input scene; it
is not a default check for from-scratch generation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from code4scene.tasks.evaluation_policy import (
    DEFAULT_POLICY_ID,
    IMAGE_TO_SCENE_POLICY_ID,
    SCHEMA_VERSION,
    default_physics_profile,
    image_to_scene_physics_profile,
)



def _json_native(value: Any, path: str = "evaluation_policy") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            result[key] = _json_native(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_native(item, f"{path}[]") for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")

_EDIT_LIMITS = (
    "maximum_added_actor_count",
    "maximum_removed_actor_count",
    "maximum_moved_actor_count",
    "maximum_modified_actor_count",
    "maximum_off_target_change_count",
)
_SELECTOR_INCLUSION_FIELDS = (
    "allowed_asset_paths",
    "allowed_categories",
    "allowed_classes",
    "labels",
    "stable_actor_ids",
    "logical_object_ids",
    "actor_roles",
    "actor_origins",
    "required_tags",
)
_SELECTOR_EXCLUSION_FIELDS = (
    "excluded_asset_paths",
    "excluded_categories",
    "excluded_classes",
    "excluded_labels",
    "excluded_stable_actor_ids",
    "excluded_logical_object_ids",
)
_SELECTOR_FIELDS = (*_SELECTOR_INCLUSION_FIELDS, *_SELECTOR_EXCLUSION_FIELDS)
_TRANSFORM_AXES = frozenset(
    {
        "location",
        "location.x",
        "location.y",
        "location.z",
        "rotation",
        "rotation.roll",
        "rotation.pitch",
        "rotation.yaw",
        "scale",
        "scale.x",
        "scale.y",
        "scale.z",
    }
)


def _nonnegative_number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{path} must be a finite non-negative number")
    return float(value)


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _string_array(value: Any, path: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{path} must be an array of non-empty strings")
    result = [item.strip() for item in value]
    if not allow_empty and not result:
        raise ValueError(f"{path} must not be empty")
    return result


def _selector(
    value: Any,
    path: str,
    *,
    allowed_scopes: frozenset[str] = frozenset({"candidate_all"}),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    unknown = sorted(set(value) - {"scope", *_SELECTOR_FIELDS})
    if unknown:
        raise ValueError(f"{path} has unsupported fields {unknown}")
    result: dict[str, Any] = {}
    scope = value.get("scope")
    if scope is not None:
        if scope not in allowed_scopes:
            expected = ", ".join(sorted(allowed_scopes))
            raise ValueError(f"{path}.scope must be one of {expected}")
        result["scope"] = scope
    for name in _SELECTOR_FIELDS:
        if name in value:
            result[name] = _string_array(value[name], f"{path}.{name}")
    if result.get("scope") is None and not any(
        result.get(name) for name in _SELECTOR_INCLUSION_FIELDS
    ):
        raise ValueError(f"{path} must identify at least one Actor population")
    return result


def _allow_policy(value: Any, path: str) -> bool | dict[str, Any]:
    if isinstance(value, bool):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be boolean or an object")
    if path.endswith(".transform"):
        unknown = sorted(set(value) - {"axes"})
        if unknown:
            raise ValueError(f"{path} has unsupported fields {unknown}")
        axes = _string_array(value.get("axes", ()), f"{path}.axes", allow_empty=False)
        invalid = sorted(set(axes) - _TRANSFORM_AXES)
        if invalid:
            raise ValueError(f"{path}.axes has unsupported values {invalid}")
        return {"axes": axes}
    unknown = sorted(set(value) - {"fields", "properties", "material_slots"})
    if unknown:
        raise ValueError(f"{path} has unsupported fields {unknown}")
    result: dict[str, Any] = {}
    for name in ("fields", "properties"):
        if name in value:
            result[name] = _string_array(value[name], f"{path}.{name}")
    slots = value.get("material_slots")
    if slots is not None:
        if not isinstance(slots, (list, tuple)):
            raise TypeError(f"{path}.material_slots must be an array")
        normalized_slots = []
        for index, slot in enumerate(slots):
            where = f"{path}.material_slots[{index}]"
            if not isinstance(slot, Mapping):
                raise TypeError(f"{where} must be an object")
            component = str(slot.get("component_identity") or "").strip()
            slot_index = slot.get("slot_index")
            if not component:
                raise ValueError(f"{where}.component_identity must be non-empty")
            normalized_slots.append(
                {
                    "component_identity": component,
                    "slot_index": _nonnegative_integer(
                        slot_index, f"{where}.slot_index"
                    ),
                }
            )
        result["material_slots"] = normalized_slots
    if not any(result.values()):
        raise ValueError(f"{path} must name allowed fields, properties or slots")
    return result


def _normalize_edit_scope(value: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(value) - {"targets", "tolerances", "limits"})
    if unknown:
        raise ValueError(f"evaluation_policy.edit_scope has unsupported fields {unknown}")
    targets = value.get("targets", ())
    if not isinstance(targets, (list, tuple)):
        raise TypeError("evaluation_policy.edit_scope.targets must be an array")
    normalized_targets = []
    seen: set[str] = set()
    for index, target in enumerate(targets):
        path = f"evaluation_policy.edit_scope.targets[{index}]"
        if not isinstance(target, Mapping):
            raise TypeError(f"{path} must be an object")
        unknown_target = sorted(set(target) - {"id", "selector", "allow"})
        if unknown_target:
            raise ValueError(f"{path} has unsupported fields {unknown_target}")
        target_id = str(target.get("id") or "").strip()
        if not target_id or target_id in seen:
            raise ValueError(f"{path}.id must be non-empty and unique")
        seen.add(target_id)
        allow = target.get("allow")
        if not isinstance(allow, Mapping):
            raise TypeError(f"{path}.allow must be an object")
        unknown_operations = sorted(
            set(allow) - {"add", "remove", "transform", "attribute"}
        )
        if unknown_operations:
            raise ValueError(f"{path}.allow has unsupported operations {unknown_operations}")
        normalized_allow = {
            operation: _allow_policy(configured, f"{path}.allow.{operation}")
            for operation, configured in allow.items()
        }
        if not any(
            configured is True or isinstance(configured, Mapping)
            for configured in normalized_allow.values()
        ):
            raise ValueError(f"{path}.allow authorizes no operation")
        normalized_targets.append(
            {
                "id": target_id,
                "selector": _selector(target.get("selector"), f"{path}.selector"),
                "allow": normalized_allow,
            }
        )

    tolerance_defaults = {
        "location_cm": 0.1,
        "rotation_deg": 0.01,
        "scale": 0.0001,
    }
    tolerances = value.get("tolerances") or {}
    if not isinstance(tolerances, Mapping):
        raise TypeError("evaluation_policy.edit_scope.tolerances must be an object")
    unknown_tolerances = sorted(set(tolerances) - set(tolerance_defaults))
    if unknown_tolerances:
        raise ValueError(
            "evaluation_policy.edit_scope.tolerances has unsupported fields "
            f"{unknown_tolerances}"
        )
    normalized_tolerances = {
        name: _nonnegative_number(
            tolerances.get(name, default),
            f"evaluation_policy.edit_scope.tolerances.{name}",
        )
        for name, default in tolerance_defaults.items()
    }

    limits = value.get("limits") or {}
    if not isinstance(limits, Mapping):
        raise TypeError("evaluation_policy.edit_scope.limits must be an object")
    unknown_limits = sorted(set(limits) - set(_EDIT_LIMITS))
    if unknown_limits:
        raise ValueError(
            f"evaluation_policy.edit_scope.limits has unsupported fields {unknown_limits}"
        )
    normalized_limits = {
        name: _nonnegative_integer(
            limits.get(name, 0),
            f"evaluation_policy.edit_scope.limits.{name}",
        )
        for name in _EDIT_LIMITS
    }
    return {
        "targets": normalized_targets,
        "tolerances": normalized_tolerances,
        "limits": normalized_limits,
    }


def _normalize_physics_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed_fields = {
        "profile_id",
        "selector",
        "maximum_ground_gap_cm",
        "adaptive_penetration_tolerance",
        "relative_penetration_tolerance_fraction",
        "maximum_penetration_cm",
        "maximum_adaptive_penetration_tolerance_cm",
        "minimum_support_fraction",
        "aabb_touch_tolerance_cm",
        "maximum_decisive_aabb_span_cm",
        "environment_proxy_minimum_span_cm",
        "environment_proxy_median_span_multiplier",
        "environment_proxy_minimum_contained_actor_count",
        "environment_proxy_minimum_contained_actor_fraction",
        "waterline_tolerance_cm",
        "minimum_surface_span_cm",
        "maximum_thin_surface_extent_cm",
        "maximum_penetrating_actor_count",
        "maximum_fully_submerged_actor_count",
        "maximum_partially_submerged_actor_count",
        "excluded_classes",
        "support_surface_classes",
        "allow_floating_categories",
        "allow_submerged_categories",
        "support_model",
        "evaluator_overrides",
        "input_measurements",
    }
    unknown = sorted(set(value) - allowed_fields)
    if unknown:
        raise ValueError(
            "evaluation_policy.physics_profile has unsupported fields "
            f"{unknown}"
        )
    result = _json_native(dict(value), "physics_profile")
    profile_id = str(result.get("profile_id") or "").strip()
    if not profile_id:
        raise ValueError("evaluation_policy.physics_profile.profile_id must be non-empty")
    result["profile_id"] = profile_id
    support_model = result.get("support_model")
    if support_model is not None:
        if support_model not in {"ground_only_v1", "ground_or_lateral_v1"}:
            raise ValueError(
                "evaluation_policy.physics_profile.support_model must be "
                "ground_only_v1 or ground_or_lateral_v1"
            )
    result["selector"] = _selector(
        result.get("selector") or {},
        "evaluation_policy.physics_profile.selector",
        allowed_scopes=frozenset({"candidate_all", "edited_actors"}),
    )
    if "adaptive_penetration_tolerance" in result:
        if not isinstance(result["adaptive_penetration_tolerance"], bool):
            raise TypeError(
                "evaluation_policy.physics_profile."
                "adaptive_penetration_tolerance must be boolean"
            )
    for name in (
        "maximum_ground_gap_cm",
        "relative_penetration_tolerance_fraction",
        "maximum_penetration_cm",
        "maximum_adaptive_penetration_tolerance_cm",
        "minimum_support_fraction",
        "aabb_touch_tolerance_cm",
        "maximum_decisive_aabb_span_cm",
        "environment_proxy_minimum_span_cm",
        "environment_proxy_median_span_multiplier",
        "environment_proxy_minimum_contained_actor_fraction",
        "waterline_tolerance_cm",
        "minimum_surface_span_cm",
        "maximum_thin_surface_extent_cm",
    ):
        if name in result:
            result[name] = _nonnegative_number(
                result[name], f"evaluation_policy.physics_profile.{name}"
            )
    if float(result.get("minimum_support_fraction", 0.0)) > 1.0:
        raise ValueError(
            "evaluation_policy.physics_profile.minimum_support_fraction must be <= 1"
        )
    if float(result.get("relative_penetration_tolerance_fraction", 0.0)) > 1.0:
        raise ValueError(
            "evaluation_policy.physics_profile."
            "relative_penetration_tolerance_fraction must be <= 1"
        )
    if float(result.get(
        "environment_proxy_minimum_contained_actor_fraction", 0.0
    )) > 1.0:
        raise ValueError(
            "evaluation_policy.physics_profile."
            "environment_proxy_minimum_contained_actor_fraction must be <= 1"
        )
    if result.get("adaptive_penetration_tolerance"):
        minimum_tolerance = float(result.get("maximum_penetration_cm", 5.0))
        maximum_tolerance = float(
            result.get("maximum_adaptive_penetration_tolerance_cm", 50.0)
        )
        if maximum_tolerance < minimum_tolerance:
            raise ValueError(
                "evaluation_policy.physics_profile."
                "maximum_adaptive_penetration_tolerance_cm must be >= "
                "maximum_penetration_cm"
            )
    for name in (
        "maximum_penetrating_actor_count",
        "environment_proxy_minimum_contained_actor_count",
        "maximum_fully_submerged_actor_count",
        "maximum_partially_submerged_actor_count",
    ):
        if name in result:
            result[name] = _nonnegative_integer(
                result[name], f"evaluation_policy.physics_profile.{name}"
            )
    for name in (
        "excluded_classes",
        "support_surface_classes",
        "allow_floating_categories",
        "allow_submerged_categories",
    ):
        if name in result:
            result[name] = _string_array(
                result[name], f"evaluation_policy.physics_profile.{name}"
            )
    overrides = result.get("evaluator_overrides", {})
    if not isinstance(overrides, Mapping):
        raise TypeError(
            "evaluation_policy.physics_profile.evaluator_overrides must be an object"
        )
    result["evaluator_overrides"] = _json_native(
        dict(overrides), "physics_profile.evaluator_overrides"
    )
    return result

def _normalize_source_snapshot(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping) and set(value) == {
        "runtime_task_input_map"
    }:
        if value.get("runtime_task_input_map") is not True:
            raise ValueError(
                "evaluation_policy.source_snapshot.runtime_task_input_map "
                "must be true"
            )
        return {"runtime_task_input_map": True}
    if isinstance(value, (str, Path)):
        path = str(value).strip()
        if not path:
            raise ValueError("evaluation_policy.source_snapshot path must be non-empty")
        return path
    if isinstance(value, Mapping) and set(value) == {
        "path",
        "sha256",
        "exclude_stable_actor_ids",
        "map_path",
    }:
        path = str(value.get("path") or "").strip()
        digest = str(value.get("sha256") or "").strip().casefold()
        map_path = str(value.get("map_path") or "").strip()
        excluded = value.get("exclude_stable_actor_ids")
        if not path:
            raise ValueError(
                "evaluation_policy.source_snapshot descriptor path must be non-empty"
            )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "evaluation_policy.source_snapshot descriptor sha256 must be "
                "64 hex characters"
            )
        if not map_path.startswith("/Game/"):
            raise ValueError(
                "evaluation_policy.source_snapshot descriptor map_path must be "
                "a /Game package"
            )
        if not isinstance(excluded, list) or not excluded:
            raise ValueError(
                "evaluation_policy.source_snapshot descriptor needs a non-empty "
                "exclude_stable_actor_ids array"
            )
        identifiers = [str(item).strip() for item in excluded]
        if any(not item for item in identifiers) or len(set(identifiers)) != len(
            identifiers
        ):
            raise ValueError(
                "evaluation_policy.source_snapshot descriptor stable Actor IDs "
                "must be non-empty and unique"
            )
        return {
            "path": path,
            "sha256": digest,
            "exclude_stable_actor_ids": identifiers,
            "map_path": map_path,
        }
    normalized = _json_native(value, "source_snapshot")
    if not isinstance(normalized, Mapping):
        raise TypeError(
            "evaluation_policy.source_snapshot must be an inline scene graph or path"
        )
    result = dict(normalized)
    actors = result.get("actors")
    if not isinstance(actors, list):
        raise ValueError(
            "inline evaluation_policy.source_snapshot must contain an actors array"
        )
    return result
def _normalize_asset_library_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    unknown = sorted(
        set(value) - {"manifest_id", "content_release"}
    )
    if unknown:
        raise ValueError(
            "evaluation_policy.asset_library_manifest has unsupported fields "
            f"{unknown}"
        )
    manifest_id = str(value.get("manifest_id") or "").strip()
    release = str(value.get("content_release") or "").strip()
    if not manifest_id:
        raise ValueError("asset_library_manifest.manifest_id must be non-empty")
    if not release:
        raise ValueError("asset_library_manifest.content_release must be non-empty")
    return {
        "manifest_id": manifest_id,
        "content_release": release,
    }


def task_mode(kind: str) -> str:
    normalized = str(kind).strip().casefold()
    if "repair" in normalized:
        return "repair"
    if "edit" in normalized or "completion" in normalized:
        return "edit"
    return "generation"


@dataclass(frozen=True, slots=True)
class FrozenEvaluationPolicy:
    policy_id: str
    source_snapshot: Any = None
    edit_scope: Mapping[str, Any] = field(default_factory=dict)
    physics_profile: Mapping[str, Any] = field(default_factory=default_physics_profile)
    asset_library_manifest: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    source: str = "code_owned_default"

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported evaluation_policy schema_version {self.schema_version!r}"
            )
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("evaluation_policy.policy_id must be non-empty")
        if not isinstance(self.edit_scope, Mapping):
            raise TypeError("evaluation_policy.edit_scope must be an object")
        if not isinstance(self.physics_profile, Mapping):
            raise TypeError("evaluation_policy.physics_profile must be an object")
        if not isinstance(self.asset_library_manifest, Mapping):
            raise TypeError(
                "evaluation_policy.asset_library_manifest must be an object"
            )
        object.__setattr__(
            self, "edit_scope", _normalize_edit_scope(self.edit_scope)
        )
        object.__setattr__(
            self, "physics_profile", _normalize_physics_profile(self.physics_profile)
        )
        object.__setattr__(
            self,
            "asset_library_manifest",
            _normalize_asset_library_manifest(self.asset_library_manifest),
        )
        source_snapshot = _normalize_source_snapshot(self.source_snapshot)
        object.__setattr__(self, "source_snapshot", source_snapshot)

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "source_snapshot": self.source_snapshot,
            "edit_scope": dict(self.edit_scope),
            "physics_profile": dict(self.physics_profile),
            "asset_library_manifest": dict(self.asset_library_manifest),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.content_dict()

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, source: str = "task"
    ) -> FrozenEvaluationPolicy:
        required = {"schema_version", "policy_id", "physics_profile"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"evaluation_policy is missing {missing}")
        return cls(
            schema_version=str(value["schema_version"]),
            policy_id=str(value["policy_id"]),
            source_snapshot=value.get("source_snapshot"),
            edit_scope=value.get("edit_scope") or {},
            physics_profile=value["physics_profile"],
            asset_library_manifest=value.get("asset_library_manifest") or {},
            source=source,
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "evaluation_policy_id": self.policy_id,
            "evaluation_policy_source": self.source,
        }

    def physics_contract(self, case_id: str) -> dict[str, Any]:
        profile = self.physics_profile
        base_selector = dict(
            profile.get("selector") or {"scope": "candidate_all"}
        )

        def selector(
            *, extra_classes: tuple[str, ...] = (),
            extra_categories: tuple[str, ...] = (),
        ) -> dict[str, Any]:
            result = dict(base_selector)
            classes = {
                *result.get("excluded_classes", ()),
                *profile.get("excluded_classes", ()),
                *extra_classes,
            }
            categories = {
                *result.get("excluded_categories", ()),
                *extra_categories,
            }
            if classes:
                result["excluded_classes"] = sorted(classes)
            if categories:
                result["excluded_categories"] = sorted(categories)
            return result

        ground_selector = selector(
            extra_classes=tuple(profile.get("support_surface_classes", ())),
            extra_categories=tuple(profile.get("allow_floating_categories", ())),
        )
        solid_selector = selector()
        support_model = profile.get("support_model")
        ground_assertion = {
            "id": "default-ground-contact",
            "primitive": "physics",
            "scope": "candidate_all",
            "target_selector": ground_selector,
            "maximum_ground_gap_cm": profile.get(
                "maximum_ground_gap_cm", 5.0
            ),
            "maximum_penetration_cm": profile.get(
                "maximum_penetration_cm", 5.0
            ),
            "minimum_support_fraction": profile.get(
                "minimum_support_fraction", 0.05
            ),
        }
        if support_model is not None:
            ground_assertion["support_model"] = support_model
        return {
            "schema_version": "0.2.0",
            "case_id": case_id,
            "policy_id": profile.get("profile_id"),
            "assertions": [
                ground_assertion,
                {
                    "id": "default-solid-penetration",
                    "primitive": "solid_penetration",
                    "scope": "candidate_all",
                    "target_selector": solid_selector,
                    "maximum_penetrating_actor_count": profile.get(
                        "maximum_penetrating_actor_count", 0
                    ),
                    "maximum_penetration_cm": profile.get(
                        "maximum_penetration_cm", 5.0
                    ),
                    "adaptive_penetration_tolerance": profile.get(
                        "adaptive_penetration_tolerance", False
                    ),
                    "relative_penetration_tolerance_fraction": profile.get(
                        "relative_penetration_tolerance_fraction", 0.05
                    ),
                    "maximum_adaptive_penetration_tolerance_cm": profile.get(
                        "maximum_adaptive_penetration_tolerance_cm", 50.0
                    ),
                    "aabb_touch_tolerance_cm": profile.get(
                        "aabb_touch_tolerance_cm", 5.0
                    ),
                    "maximum_decisive_aabb_span_cm": profile.get(
                        "maximum_decisive_aabb_span_cm", 3000.0
                    ),
                    "environment_proxy_minimum_span_cm": profile.get(
                        "environment_proxy_minimum_span_cm", 30000.0
                    ),
                    "environment_proxy_median_span_multiplier": profile.get(
                        "environment_proxy_median_span_multiplier", 10.0
                    ),
                    "environment_proxy_minimum_contained_actor_count": profile.get(
                        "environment_proxy_minimum_contained_actor_count", 3
                    ),
                    "environment_proxy_minimum_contained_actor_fraction": profile.get(
                        "environment_proxy_minimum_contained_actor_fraction", 0.05
                    ),
                },
            ],
        }

    def leaf_spec(
        self,
        task: Any,
        leaf_id: str,
        base_spec: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = dict(base_spec or {})
        spec["case_id"] = str(getattr(task, "id", "scene"))
        spec["evaluation_policy"] = self.to_dict()
        if leaf_id in {
            "floating",
            "ground_gap",
            "solid_penetration",
            "environment_consistency",
        }:
            spec["semantic_contract"] = self.physics_contract(spec["case_id"])
            overrides = self.physics_profile.get("evaluator_overrides") or {}
            if isinstance(overrides, Mapping):
                configured = overrides.get(leaf_id)
                if isinstance(configured, Mapping):
                    spec.update(configured)
        if leaf_id == "solid_penetration":
            # Floating is measured by its own scene-rate probe in Physical
            # Safety.  The solid leaf needs only exact collision evidence;
            # carrying this operational mode avoids repeating mesh/ground
            # traces without changing its assertions or thresholds.
            if (
                self.physics_profile.get("selector", {}).get("scope")
                != "edited_actors"
            ):
                spec["physics_measurement_mode"] = "solid_penetration"
        if self.source_snapshot is not None:
            spec["input_scene"] = self.source_snapshot
        input_measurements = self.physics_profile.get("input_measurements")
        if input_measurements is not None:
            spec["input_measurements"] = input_measurements
        return spec


def default_policy(task: Any) -> FrozenEvaluationPolicy:
    data = getattr(task, "data", {}) or {}
    source = data.get("source") if isinstance(data, Mapping) else None
    contract = (
        source.get("evaluation_contract")
        if isinstance(source, Mapping)
        else None
    )
    if contract is not None and not isinstance(contract, Mapping):
        raise TypeError("source.evaluation_contract must be an object")
    contract = dict(contract or {})
    unknown = sorted(
        set(contract) - {"source_snapshot", "edit_scope", "asset_library_manifest"}
    )
    if unknown:
        raise ValueError(
            "source.evaluation_contract has unsupported fields "
            f"{unknown}; implementation policy is code-owned"
        )
    if getattr(task, "case_type", None) == "image_to_scene":
        return FrozenEvaluationPolicy(
            policy_id=IMAGE_TO_SCENE_POLICY_ID,
            source_snapshot=contract.get(
                "source_snapshot", {"runtime_task_input_map": True}
            ),
            edit_scope=contract.get("edit_scope") or {},
            physics_profile=image_to_scene_physics_profile(),
            asset_library_manifest=contract.get("asset_library_manifest") or {},
            source=(
                "code_owned_image_to_scene_with_task_semantics"
                if contract
                else "code_owned_image_to_scene"
            ),
        )
    return FrozenEvaluationPolicy(
        policy_id=DEFAULT_POLICY_ID,
        source_snapshot=contract.get("source_snapshot"),
        edit_scope=contract.get("edit_scope") or {},
        physics_profile=default_physics_profile(),
        asset_library_manifest=contract.get("asset_library_manifest") or {},
        source=(
            "code_owned_default_with_task_semantics"
            if contract
            else "code_owned_default"
        ),
    )


def _load_document(task: Any, configured: Any) -> tuple[Mapping[str, Any], str]:
    if isinstance(configured, Mapping):
        return configured, "task_inline"
    if not isinstance(configured, (str, Path)):
        raise TypeError("evaluation_policy must be an object or YAML/JSON path")
    path = Path(configured)
    task_path = getattr(task, "path", None)
    if not path.is_absolute() and task_path:
        path = (Path(task_path).parent / path).resolve()
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read evaluation_policy from {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("evaluation_policy document must contain an object")
    return value, str(path)


def load_evaluation_policy(task: Any) -> FrozenEvaluationPolicy:
    data = getattr(task, "data", {}) or {}
    configured = data.get("evaluation_policy") if isinstance(data, Mapping) else None
    if configured is None:
        return default_policy(task)
    value, source = _load_document(task, configured)
    return FrozenEvaluationPolicy.from_dict(value, source=source)


__all__ = [
    "DEFAULT_POLICY_ID",
    "FrozenEvaluationPolicy",
    "SCHEMA_VERSION",
    "default_policy",
    "load_evaluation_policy",
    "task_mode",
]
