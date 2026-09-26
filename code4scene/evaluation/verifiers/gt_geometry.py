"""gt_geometry — decomposed distance to a canonical scene.

This verifier deliberately separates three mistakes:

* global placement is measured before alignment and remains a deduction;
* raw-Actor fragmentation/fusion is measured before object aggregation;
* Actor identity, pose, properties, and material slots use stable-ID-first raw
  Actor correspondence;
* bounds and layout use deterministic logical aggregation after the same
  identity-anchored XY translation/yaw alignment.

The separation prevents one global offset or one tiled floor from being
charged again in every downstream metric. Material, colour, lighting, and
visible style remain the responsibility of scene_diff.visual_semantic_diff.

Exact errors and rates are published as decomposed continuous measurements;
the public structured-scene-diff layer derives transparent normalized scores
without a pass/fail cutoff.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import contracts, ue_evidence
from ..context import Context, LABEL_SUFFIX, error, read_label
from ..gt_geometry_compare import (
    AUTO_GROUP_GAP_CAP_CM,
    AUTO_GROUP_GAP_CM,
    AUTO_GROUP_GAP_RATIO,
    IDENTITY_GATE,
    MATCH_WEIGHTS,
    compare,
    compare_detailed,
)
from ..pairwise_identity_locator import LLMPairwiseIdentityLocatorBackend
from ..repair_target_scope import load_repair_target_scope, measure_repair_target

METRIC_VERSION = "gt-geometry-v5"


def _write_audit(context: Context, audit: Mapping[str, Any]) -> str | None:
    root_value = getattr(context, "out_dir", None)
    if root_value is None:
        return None
    root = Path(root_value) / "gt_geometry"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "correspondence.json"
    path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return str(path)


def _identity_locator_options(
    context: Context,
) -> tuple[LLMPairwiseIdentityLocatorBackend | None, str]:
    """Build an optional locator-only backend from this verifier's config."""

    spec = getattr(context, "spec", {})
    spec = spec if isinstance(spec, Mapping) else {}
    raw = spec.get("identity_retrieval")
    if raw is None:
        return None, "deterministic"
    if not isinstance(raw, Mapping):
        raise ValueError("identity_retrieval must be an object")
    mode = str(raw.get("mode", "deterministic")).strip().casefold()
    if mode in {"disabled", "deterministic", "lexical"}:
        return None, "deterministic"
    if mode != "llm":
        raise ValueError("identity_retrieval.mode must be deterministic or llm")
    top_k = raw.get("top_k", 5)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("identity_retrieval.top_k must be a positive integer")
    max_tokens = raw.get("max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise ValueError(
            "identity_retrieval.max_tokens must be a positive integer"
        )
    from ..requirement_graph.vlm_client import tool_client_from_env

    return (
        LLMPairwiseIdentityLocatorBackend(
            tool_client_from_env(),
            top_k=top_k,
            max_tokens=max_tokens,
        ),
        "llm_pairwise_locator",
    )


def verify(context: Context) -> dict[str, Any]:
    """Compare Candidate with frozen-label or live independent GT evidence."""

    runtime_gt = context.spec.get("canonical_scene") == {
        "runtime_task_ground_truth_map": True
    }
    label = read_label(context)
    if label is None:
        label = {}
    if not isinstance(label, Mapping):
        return error("gt_geometry", context, "the answer key must be an object")
    if not runtime_gt and not label:
        return error(
            "gt_geometry",
            context,
            f"no answer key beside {getattr(context.task, 'path', '?')} "
            f"({LABEL_SUFFIX}); geometry distance needs a frozen canonical "
            "Actor list",
        )
    canonical = label.get("canonical_actors")
    if not runtime_gt and (not isinstance(canonical, list) or not canonical):
        return error(
            "gt_geometry",
            context,
            "the answer key carries no canonical scene to compare against",
        )
    if not runtime_gt and any(
        not isinstance(actor, Mapping) for actor in canonical
    ):
        return error(
            "gt_geometry",
            context,
            "canonical_actors must be a non-empty array of Actor objects",
        )

    if context.scoring is None:
        return error(
            "gt_geometry",
            context,
            "geometry distance requires an independent scoring editor",
        )
    try:
        evidence = ue_evidence.collect(context)
        candidate = evidence.candidate_actors()
        canonical_path = None
        if runtime_gt:
            canonical_scene, canonical_path = (
                ue_evidence.capture_task_ground_truth(
                    context,
                    candidate_scene=evidence.candidate,
                )
            )
            canonical = canonical_scene.get("actors")
    except Exception as exception:  # noqa: BLE001 — reported as verifier evidence
        return error(
            "gt_geometry",
            context,
            f"{type(exception).__name__}: {exception}",
        )
    if not isinstance(canonical, list) or not canonical:
        return error(
            "gt_geometry",
            context,
            "the canonical scene carries no Actors to compare against",
        )
    if any(not isinstance(actor, Mapping) for actor in canonical):
        return error(
            "gt_geometry",
            context,
            "canonical Actors must be a non-empty array of Actor objects",
        )
    if (
        not isinstance(candidate, list)
        or any(not isinstance(actor, Mapping) for actor in candidate)
    ):
        return error(
            "gt_geometry",
            context,
            "the candidate export must be an array of Actor objects",
        )

    try:
        identity_locator, identity_retrieval_mode = _identity_locator_options(
            context
        )
        comparison = compare_detailed(
            candidate,
            canonical,
            identity_locator=identity_locator,
        )
        runtime_label = {**dict(label), "canonical_actors": canonical}
        repair_scope = load_repair_target_scope(
            context.task,
            label=runtime_label,
            input_scene=getattr(evidence, "input_scene", None),
        )
        repair_target = (
            measure_repair_target(
                repair_scope,
                {"actors": candidate},
                comparison,
            )
            if repair_scope is not None
            else None
        )
        audit_path = _write_audit(context, comparison["audit"])
    except Exception as exception:  # noqa: BLE001 — comparison failures are evidence
        return error(
            "gt_geometry",
            context,
            f"{type(exception).__name__}: {exception}",
        )
    artifacts = {"correspondence": audit_path} if audit_path else {}
    if canonical_path is not None:
        artifacts["ground_truth_scene_graph"] = str(canonical_path)
    return {
        **contracts.base("gt_geometry", context.ids),
        "status": contracts.MEASURED,
        "score": None,
        "metrics": {
            **comparison["metrics"],
            "metric_version": METRIC_VERSION,
            **(
                {"repair_target": repair_target}
                if repair_target is not None
                else {}
            ),
        },
        "evidence": {
            "gt_id": label.get("gt_id") or (
                (getattr(context.task, "data", {}).get("source") or {}).get(
                    "gt_id"
                )
            ),
            "candidate_exported_from": evidence.exported_from,
            "canonical_scene_from": (
                "independent_scoring_editor_runtime_task_ground_truth_map"
                if runtime_gt
                else "frozen_answer_key"
            ),
            "scoring_policy": "raw_decomposed_structured_measurement",
            "deduction_channels": [
                "global_translation_error_cm",
                "global_yaw_error_deg",
                "over_fragmented_actor_count",
                "under_segmented_actor_count",
            ],
            "local_metrics_frame": "identity_anchored_xy_translation_yaw_aligned",
            "correspondence_layers": {
                "actors": (
                    "stable_actor_id_first_then_identity_size_spatial_assignment"
                ),
                "logical_objects": (
                    "declared_or_connected_modular_geometry_aggregation"
                ),
            },
            "logical_object_policy": {
                "canonical_declared_ids_trusted": True,
                "candidate_declared_ids_trusted": False,
                "automatic_grouping": "connected_exact_asset_modular_parts",
                "minimum_connectivity_gap_cm": AUTO_GROUP_GAP_CM,
                "maximum_connectivity_gap_cm": AUTO_GROUP_GAP_CAP_CM,
                "connectivity_gap_scale_ratio": AUTO_GROUP_GAP_RATIO,
            },
            "identity_gate": IDENTITY_GATE,
            "match_weights": MATCH_WEIGHTS,
            "identity_retrieval_mode": identity_retrieval_mode,
            "identity_responsibility": {
                "shared_layer": "verdict_free_identity_catalog_and_locator",
                "gt_geometry": "correspondence_and_geometry_only",
                "semantic_requirements": (
                    "prompt_requirements_actor_focus_and_match_mismatch"
                ),
                "gt_repair": (
                    "shared_local_target_and_global_scene_diff_projection"
                ),
            },
            "visual_responsibility": "scene_diff.visual_semantic_diff",
            **(
                {
                    "repair_target_scope": repair_scope.audit(),
                    "repair_target_role": (
                        "primary_scoring_scope_for_image_to_scene_repair"
                    ),
                }
                if repair_scope is not None
                else {}
            ),
        },
        "artifacts": artifacts,
        "probes_used": (
            "scene_graph",
            "runtime_ground_truth_scene_snapshot" if runtime_gt else "answer_key",
        ),
    }


__all__ = ["MATCH_WEIGHTS", "METRIC_VERSION", "compare", "compare_detailed", "verify"]
