"""Recompute a failed paired-visual repair leaf from frozen render evidence.

The source score run already proved strict GT/candidate camera pairing before
calling the model.  This module reuses those immutable RGB, Base Color and
Scene Depth files, reruns only the four stateless VLM judgements, and rebuilds
the affected composite ancestors with the production aggregation functions.
It never opens UE, captures a scene, or edits the authoritative source result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from . import contracts, repair_score
from .composite import composite_report
from .context import Context
from .vlm_concurrency import (
    parallel_map as parallel_vlm_map,
    runtime_snapshot as vlm_runtime_snapshot,
)
from .verifiers import gt_repair
from .verifiers.vlm_as_judge import Judge, backend_from_policy, load_policy
from .verifiers.vlm_as_judge import paired
from .verifiers.vlm_as_judge.evidence import collect_paired


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _leaf(report: Mapping[str, Any], leaf_id: str) -> dict[str, Any]:
    leaves = (report.get("metrics") or {}).get("leaf_results") or []
    matches = [
        value
        for value in leaves
        if isinstance(value, Mapping) and value.get("leaf_id") == leaf_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{report.get('report_id', '<report>')}: expected one {leaf_id!r} "
            f"leaf, found {len(matches)}"
        )
    return copy.deepcopy(dict(matches[0]))


def _replace_top_report(
    reports: Sequence[Mapping[str, Any]],
    report_id: str,
    replacement: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    replaced = 0
    for report in reports:
        if report.get("report_id") == report_id:
            values.append(copy.deepcopy(dict(replacement)))
            replaced += 1
        else:
            values.append(copy.deepcopy(dict(report)))
    if replaced != 1:
        raise ValueError(
            f"expected one top-level {report_id!r} report, found {replaced}"
        )
    return values


def _saved_renders(evidence_dir: Path, channels: Sequence[str]) -> tuple[Any, Any]:
    views = tuple(f"view_{index}" for index in range(4))
    channel_paths: dict[str, dict[str, dict[str, str]]] = {}
    for scene in ("gt", "candidate"):
        scene_views: dict[str, dict[str, str]] = {}
        for view in views:
            scene_views[view] = {
                "rgb": str(evidence_dir / scene / f"{view}.png"),
                "base_color": str(
                    evidence_dir / scene / "base_color" / f"{view}.png"
                ),
                "scene_depth": str(
                    evidence_dir / scene / "scene_depth" / f"{view}.exr"
                ),
            }
        channel_paths[scene] = scene_views

    required = [
        Path(channel_paths[scene][view][channel])
        for scene in ("gt", "candidate")
        for view in views
        for channel in channels
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "saved paired-visual evidence is incomplete: " + ", ".join(missing)
        )
    pairing = SimpleNamespace(
        views=views,
        reference_scene="gt",
        pairs={
            channel: tuple(
                (
                    view,
                    channel_paths["gt"][view][channel],
                    channel_paths["candidate"][view][channel],
                )
                for view in views
            )
            for channel in channels
        },
    )
    return SimpleNamespace(images=channel_paths), pairing


def _prompt(task_path: Path) -> str:
    task = yaml.safe_load(task_path.read_text())
    prompt = str(((task or {}).get("inputs") or {}).get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"task {task_path} has no inputs.prompt")
    return prompt


def _visual_leaf(
    context: Context,
    raw: Mapping[str, Any],
    old_leaf: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = raw.get("metrics") or {}
    deterministic = metrics.get("deterministic") or {}
    vlm = metrics.get("vlm") or {}
    calibrated = metrics.get("calibrated") or {}
    evidence = {
        **dict(old_leaf.get("evidence") or {}),
        **dict(raw.get("evidence") or {}),
    }
    return {
        **contracts.base(
            "gt_repair.repair_target_diff.target_visual_diff", context.ids
        ),
        "leaf_id": "target_visual_diff",
        "status": contracts.MEASURED,
        "score": raw["score"],
        "metrics": {
            "paired_render_diff": {
                "status": "measured",
                "channels": ["rgb", "base_color", "scene_depth"],
                "metrics": deterministic,
                "units": {
                    "rgb_mae": "normalized_pixel_value",
                    "base_color_mae": "normalized_pixel_value",
                    "scene_depth_mae": "cm",
                    "foreground_silhouette_iou": "ratio",
                },
                "contributes_to_visual_score": True,
            },
            "visual_equivalence": {
                "status": "measured",
                "dimensions": vlm.get("dimensions") or {},
                "per_view": vlm.get("per_view") or {},
                "observations": evidence.get("observations") or [],
                "judgement_kind": "raw_vlm_semantic_equivalence",
                "contributes_to_visual_score": True,
            },
            "edit_region": {
                "status": "not_applicable",
                "reason": (
                    "canonical GT-paired capture contains Candidate and GT; an "
                    "edit-region diagnostic additionally requires aligned frozen "
                    "Input views"
                ),
                "contributes_to_visual_score": False,
            },
            "calibrated_visual_score": {
                "score": raw["score"],
                "dimensions": calibrated.get("dimensions") or {},
                "aggregation": "frozen_deterministic_vlm_blend",
                "contributes_to_visual_score": True,
            },
            "model_call_count": metrics.get("model_call_count"),
            "structured_output_recovery_count": metrics.get(
                "structured_output_recovery_count"
            ),
            "total_time_s": metrics.get("total_time_s"),
        },
        "evidence": evidence,
        "artifacts": dict(raw.get("artifacts") or {}),
        "probes_used": tuple(raw.get("probes_used") or ("vlm_as_judge",)),
    }


def measure_saved_visual(
    context: Context,
    *,
    prompt: str,
    policy_path: Path,
    evidence_dir: Path,
    source_result: Path,
) -> dict[str, Any]:
    """Run production measurements and judge calls over a frozen strict pair."""

    started = time.monotonic()
    policy = load_policy(policy_path)
    if policy.mode != "gt_paired":
        raise ValueError("saved visual recomputation needs a gt_paired policy")
    renders, pairing = _saved_renders(evidence_dir, policy.channels)
    output_evidence = (
        Path(context.out_dir)
        / "judge_evidence"
        / "gt_paired"
        / "repair_target"
    )
    bundle = collect_paired(
        renders,
        pairing,
        policy.rubric,
        output_evidence,
        near_cm=policy.depth_near_cm,
        far_cm=policy.depth_far_cm,
        base_color_encoding=policy.base_color_encoding,
    )
    judge = Judge(
        rubric=policy.rubric,
        backend=backend_from_policy(policy),
        model=policy.model,
        min_images=1,
    )

    def score_view(view: str) -> tuple[str, dict[str, Any]]:
        deterministic = paired.measure_view(renders, pairing, view, policy)
        verdict = judge.score(
            prompt=prompt,
            images=[],
            metrics={},
            evidence=bundle.images_by_view[view],
            evidence_layout=policy.evidence_layout,
            comparison_mode="gt_paired",
        )
        vlm_scores = {
            key: round(value / policy.rubric.scale_max, 6)
            for key, value in verdict.scores.items()
        }
        calibrated: dict[str, float] = {}
        calibration: dict[str, Any] = {}
        for dimension in policy.aggregation["dimension_weights"]:
            calibrated[dimension], calibration[dimension] = paired._blend_dimension(
                dimension, vlm_scores[dimension], deterministic, policy
            )
        return view, {
            "deterministic_metrics": deterministic,
            "vlm_scores": vlm_scores,
            "vlm_rationales": verdict.rationales,
            "vlm_raw_structured_response": verdict.raw_response,
            "structured_output_recovery": verdict.structured_output_recovery,
            "calibrated_dimension_scores": calibrated,
            "calibration": calibration,
            "paired_images": bundle.paired_images[view],
        }

    scored = dict(
        parallel_vlm_map(
            pairing.views,
            score_view,
            thread_name_prefix="code4scene-saved-visual-vlm",
        )
    )
    per_view = {view: scored[view] for view in pairing.views}
    vlm_dimensions = {
        dimension: paired._mean(
            [value["vlm_scores"][dimension] for value in per_view.values()]
        )
        for dimension in policy.aggregation["dimension_weights"]
    }
    calibrated_dimensions = {
        dimension: paired._mean(
            [
                value["calibrated_dimension_scores"][dimension]
                for value in per_view.values()
            ]
        )
        for dimension in policy.aggregation["dimension_weights"]
    }
    overall = round(
        sum(
            calibrated_dimensions[dimension] * weight
            for dimension, weight in policy.aggregation["dimension_weights"].items()
        ),
        6,
    )
    deterministic_aggregate = {
        name: paired._mean(
            [
                float(value["deterministic_metrics"][name])
                for value in per_view.values()
                if value["deterministic_metrics"].get(name) is not None
            ]
        )
        for name in (
            "rgb_perceptual_similarity",
            "base_color_similarity",
            "foreground_silhouette_iou",
            "depth_edge_similarity",
        )
    }
    observations = [
        {
            "dimension": dimension,
            "score": score,
            "reason": paired._vlm_dimension_reason(dimension, score, per_view),
        }
        for dimension, score in vlm_dimensions.items()
    ]
    policy_evidence = policy.evidence()
    policy_evidence["render_protocol"] = {
        **policy_evidence["render_protocol"],
        "camera_plan_policy": "visual-gt-dense-core-fill-frozen-aerial-v2",
    }
    return paired._json_finite(
        {
            **contracts.base("vlm_as_judge", context.ids),
            "status": contracts.MEASURED,
            "score": overall,
            "metrics": {
                "deterministic": {
                    "aggregate": deterministic_aggregate,
                    "per_view": {
                        view: value["deterministic_metrics"]
                        for view, value in per_view.items()
                    },
                },
                "vlm": {
                    "dimensions": vlm_dimensions,
                    "per_view": {
                        view: value["vlm_scores"]
                        for view, value in per_view.items()
                    },
                },
                "calibrated": {
                    "dimensions": calibrated_dimensions,
                    "overall_visual_similarity": overall,
                },
                "model_call_count": len(pairing.views),
                "structured_output_recovery_count": sum(
                    bool(value["structured_output_recovery"])
                    for value in per_view.values()
                ),
                "total_time_s": round(time.monotonic() - started, 6),
            },
            "evidence": {
                **policy_evidence,
                "artifact_namespace": "repair_target",
                "views_compared": len(pairing.views),
                "observations": observations,
                "per_view": per_view,
                "paired_images": bundle.paired_images,
                "base_color_normalization": bundle.base_color_normalization,
                "depth_visualization": bundle.depth_visualization,
                "saved_evidence_recomputation": {
                    "source_result": str(source_result),
                    "source_evidence_dir": str(evidence_dir),
                    "source_strict_pair_completed_before_original_judge_error": True,
                    "ue_recapture_performed": False,
                    "parallel_stateless_view_calls": len(pairing.views),
                },
                "vlm_runtime": vlm_runtime_snapshot(),
            },
            "artifacts": {
                f"paired_{view}_{channel}": paths["paired_image"]
                for view, channels in bundle.paired_images.items()
                for channel, paths in channels.items()
            },
            "probes_used": ("vlm_as_judge",),
        }
    )


def rebuild_result(
    source_result: Mapping[str, Any],
    target_visual: Mapping[str, Any],
    *,
    output_case_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    top_reports = source_result.get("reports")
    if not isinstance(top_reports, list):
        raise ValueError("source result has no report list")
    roots = [
        report
        for report in top_reports
        if isinstance(report, Mapping) and report.get("report_id") == "gt_repair"
    ]
    if len(roots) != 1:
        raise ValueError(f"source result needs one gt_repair report, found {len(roots)}")
    old_root = roots[0]
    old_local = _leaf(old_root, "repair_target_diff")
    scene = _leaf(old_root, "scene_diff")
    locality = _leaf(old_root, "locality_audit")

    leaves: list[dict[str, Any]] = []
    replaced = 0
    for value in (old_local.get("metrics") or {}).get("leaf_results") or []:
        if not isinstance(value, Mapping):
            continue
        if value.get("leaf_id") == "target_visual_diff":
            leaves.append(copy.deepcopy(dict(target_visual)))
            replaced += 1
        else:
            leaves.append(copy.deepcopy(dict(value)))
    if replaced != 1:
        raise ValueError(f"source local report needs one visual leaf, found {replaced}")

    ids = {
        "task_bundle_id": str(old_root["task_bundle_id"]),
        "episode_id": str(old_root["episode_id"]),
    }
    context = Context(record={}, task=None, ids=ids, out_dir=output_case_dir)
    local = composite_report(
        "gt_repair.repair_target_diff",
        context,
        leaves,
        evidence=old_local.get("evidence") or {},
    )
    local["leaf_id"] = "repair_target_diff"
    local = gt_repair._apply_local_repair_policy(local)
    root = composite_report(
        "gt_repair",
        context,
        [local, scene, locality],
        evidence=old_root.get("evidence") or {},
    )
    root["evidence"]["saved_evidence_recomputation"] = dict(provenance)
    repair_score.carry_actor_f1(root, old_root)
    root = gt_repair.finalize_report(root, local, scene)

    rebuilt = copy.deepcopy(dict(source_result))
    rebuilt["reports"] = _replace_top_report(top_reports, "gt_repair", root)
    error_count = sum(
        report.get("status") == contracts.ERROR
        for report in rebuilt["reports"]
        if isinstance(report, Mapping)
    )
    rebuilt["error_report_count"] = error_count
    rebuilt["batch_status"] = "complete_with_errors" if error_count else "complete"
    rebuilt["saved_evidence_recomputation"] = dict(provenance)
    from . import primary_score

    return primary_score.apply_to_result(rebuilt)


def _source_visual_leaf(source_result: Mapping[str, Any]) -> dict[str, Any]:
    root = next(
        report
        for report in source_result["reports"]
        if report.get("report_id") == "gt_repair"
    )
    local = _leaf(root, "repair_target_diff")
    visual = _leaf(local, "target_visual_diff")
    failure = str(visual.get("failure_reason") or "")
    if visual.get("status") != contracts.ERROR or "JudgeError" not in failure:
        raise ValueError(
            "offline visual replay accepts only a source target_visual_diff "
            "that failed in Judge validation"
        )
    return visual


def recompute(
    source_result_path: Path,
    task_path: Path,
    policy_path: Path,
    evidence_dir: Path,
    output_root: Path,
) -> Path:
    started = time.monotonic()
    source_result_path = source_result_path.resolve()
    task_path = task_path.resolve()
    policy_path = policy_path.resolve()
    evidence_dir = evidence_dir.resolve()
    source_result = json.loads(source_result_path.read_text())
    old_visual = _source_visual_leaf(source_result)
    task_id = str(source_result.get("task_id") or "")
    if not task_id:
        raise ValueError("source result has no task_id")
    output_case_dir = output_root.resolve() / "results" / task_id
    output_case_dir.mkdir(parents=True, exist_ok=True)
    root = next(
        report
        for report in source_result["reports"]
        if report.get("report_id") == "gt_repair"
    )
    ids = {
        "task_bundle_id": task_id,
        "episode_id": str(root["episode_id"]),
    }
    context = Context(record={}, task=None, ids=ids, out_dir=output_case_dir)
    raw_visual = measure_saved_visual(
        context,
        prompt=_prompt(task_path),
        policy_path=policy_path,
        evidence_dir=evidence_dir,
        source_result=source_result_path,
    )
    target_visual = _visual_leaf(context, raw_visual, old_visual)
    frozen_files = sorted(
        path
        for path in evidence_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".exr"}
    )
    code_files = [
        Path(paired.__file__).resolve(),
        Path(gt_repair.__file__).resolve(),
        Path(__file__).resolve(),
    ]
    provenance = {
        "schema_version": "saved-paired-visual-recomputation.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_result": str(source_result_path),
        "source_result_sha256": _sha256(source_result_path),
        "source_batch_status": source_result.get("batch_status"),
        "scope": "target_visual_diff_and_composite_ancestors_only",
        "source_failure_reason": old_visual.get("failure_reason"),
        "source_strict_pair_completed_before_original_judge_error": True,
        "ue_recapture_performed": False,
        "frozen_image_sha256": [
            {"path": str(path), "sha256": _sha256(path)} for path in frozen_files
        ],
        "task": str(task_path),
        "task_sha256": _sha256(task_path),
        "judge_policy": str(policy_path),
        "judge_policy_sha256": _sha256(policy_path),
        "code_sha256": {str(path): _sha256(path) for path in code_files},
        "target_visual_status": target_visual.get("status"),
        "target_visual_score": target_visual.get("score"),
    }
    rebuilt = rebuild_result(
        source_result,
        target_visual,
        output_case_dir=output_case_dir,
        provenance=provenance,
    )
    timings = dict(rebuilt.get("timings") or {})
    timings["offline_paired_visual_recompute_seconds"] = round(
        time.monotonic() - started, 3
    )
    rebuilt["timings"] = timings
    output_path = output_case_dir / "result.json"
    output_path.write_text(
        json.dumps(rebuilt, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute a failed repair target visual leaf from frozen strict-"
            "paired evidence without opening UE."
        )
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--judge-policy", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    output = recompute(
        args.source_result,
        args.task,
        args.judge_policy,
        args.evidence_dir,
        args.out_dir,
    )
    result = json.loads(output.read_text())
    gt_report = next(
        report for report in result["reports"] if report["report_id"] == "gt_repair"
    )
    local = _leaf(gt_report, "repair_target_diff")
    visual = _leaf(local, "target_visual_diff")
    print(
        json.dumps(
            {
                "result": str(output),
                "batch_status": result["batch_status"],
                "target_visual_score": visual["score"],
                "repair_target_score": local["score"],
                "gt_repair_score": gt_report["score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
