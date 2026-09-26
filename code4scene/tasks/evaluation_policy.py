"""Code-owned evaluation defaults and task-semantic edit-scope helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "1.0"
DEFAULT_POLICY_ID = "scenebenchmark-default-evaluation"
IMAGE_TO_SCENE_POLICY_ID = "image-to-scene-runtime-input-local-physics"


def default_physics_profile() -> dict[str, Any]:
    return {
        "profile_id": "absolute-physical-safety-candidate-all-v3",
        "selector": {"scope": "candidate_all"},
        "maximum_ground_gap_cm": 5.0,
        "adaptive_penetration_tolerance": True,
        "relative_penetration_tolerance_fraction": 0.05,
        "maximum_penetration_cm": 5.0,
        "maximum_adaptive_penetration_tolerance_cm": 50.0,
        "minimum_support_fraction": 0.05,
        "maximum_penetrating_actor_count": 0,
        "aabb_touch_tolerance_cm": 5.0,
        "maximum_decisive_aabb_span_cm": 3000.0,
        "environment_proxy_minimum_span_cm": 30000.0,
        "environment_proxy_median_span_multiplier": 10.0,
        "environment_proxy_minimum_contained_actor_count": 3,
        "environment_proxy_minimum_contained_actor_fraction": 0.05,
        "maximum_fully_submerged_actor_count": 0,
        "maximum_partially_submerged_actor_count": 0,
        "waterline_tolerance_cm": 5.0,
        "minimum_surface_span_cm": 200.0,
        "maximum_thin_surface_extent_cm": 100.0,
        "evaluator_overrides": {},
    }


def image_to_scene_physics_profile() -> dict[str, Any]:
    """Return the current edited-Actor safety profile for image repair.

    This is intentionally code-owned: every image-to-scene task receives the
    same behavior and cannot select an implementation generation in YAML.
    """

    physics_options = {
        "lateral_support_minimum_fraction": 0.05,
        "lateral_support_tolerance_cm": 5.0,
        "measurement_profile": "fast",
        "support_model": "ground_or_lateral_v1",
    }
    return {
        "profile_id": "image-to-scene-edited-actors-physical-safety",
        "selector": {"scope": "edited_actors"},
        "maximum_ground_gap_cm": 5.0,
        "maximum_penetration_cm": 5.0,
        "minimum_support_fraction": 0.05,
        "maximum_penetrating_actor_count": 0,
        "aabb_touch_tolerance_cm": 5.0,
        "maximum_decisive_aabb_span_cm": 3000.0,
        "maximum_fully_submerged_actor_count": 0,
        "maximum_partially_submerged_actor_count": 0,
        "waterline_tolerance_cm": 5.0,
        "minimum_surface_span_cm": 200.0,
        "maximum_thin_surface_extent_cm": 100.0,
        "evaluator_overrides": {
            name: {"physics_options": copy.deepcopy(physics_options)}
            for name in (
                "environment_consistency",
                "floating",
                "solid_penetration",
            )
        },
        "support_model": "ground_or_lateral_v1",
    }


def addition_edit_scope(removed_actors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Authorize only restoration additions matching frozen removed assets.

    Asset paths are used only when every removed Actor has one; otherwise an
    incomplete path list would reject a legitimate restoration.  Class paths
    provide the required population fallback.  Identity/count correctness is
    still a task/GT responsibility, while this scope prevents unrelated edits.
    """

    removed = [dict(value) for value in removed_actors]
    if not removed:
        raise ValueError("an addition edit scope needs at least one removed Actor")
    asset_paths = sorted(
        {str(value.get("asset_path")) for value in removed if value.get("asset_path")}
    )
    classes = sorted(
        {str(value.get("class")) for value in removed if value.get("class")}
    )
    selector: dict[str, Any] = {}
    if len(asset_paths) and all(value.get("asset_path") for value in removed):
        selector["allowed_asset_paths"] = asset_paths
    if classes and all(value.get("class") for value in removed):
        selector["allowed_classes"] = classes
    if not selector:
        raise ValueError(
            "removed Actors have neither complete asset paths nor complete class paths"
        )
    count = len(removed)
    return {
        "targets": [
            {
                "id": "restore-removed-actors",
                "selector": selector,
                "allow": {"add": True},
            }
        ],
        "tolerances": {
            "location_cm": 0.1,
            "rotation_deg": 0.01,
            "scale": 0.0001,
        },
        "limits": {
            "maximum_added_actor_count": count,
            "maximum_removed_actor_count": 0,
            "maximum_moved_actor_count": 0,
            "maximum_modified_actor_count": 0,
            "maximum_off_target_change_count": 0,
        },
    }


__all__ = [
    "DEFAULT_POLICY_ID",
    "IMAGE_TO_SCENE_POLICY_ID",
    "SCHEMA_VERSION",
    "addition_edit_scope",
    "default_physics_profile",
    "image_to_scene_physics_profile",
]
