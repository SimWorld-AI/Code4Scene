"""Recompute a failed GT caption leaf from already-frozen render evidence.

This module deliberately cannot recapture a scene or recompute geometry. It
reuses the eight RGB views saved by the original authoritative score run,
reruns only the caption/embedding metric, and rebuilds its composite ancestors
with the same production aggregation functions.
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
from typing import Any

from . import contracts, render, repair_score
from .composite import composite_report
from .context import Context
from .render_evidence import CAPTION_ENVIRONMENT_RENDER_PROTOCOL
from .verifiers import gt_caption_similarity, gt_repair


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


def rebuild_result(
    source_result: Mapping[str, Any],
    caption_leaf: Mapping[str, Any],
    *,
    output_case_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace caption_diff and rebuild only its production ancestors."""

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
    local = _leaf(old_root, "repair_target_diff")
    old_scene = _leaf(old_root, "scene_diff")
    locality = _leaf(old_root, "locality_audit")
    structured = _leaf(old_scene, "structured_scene_diff")
    old_visual = _leaf(old_scene, "visual_semantic_diff")

    visual_leaves: list[dict[str, Any]] = []
    replaced_caption = 0
    for value in (old_visual.get("metrics") or {}).get("leaf_results") or []:
        if not isinstance(value, Mapping):
            continue
        if value.get("leaf_id") == "caption_diff":
            replacement = copy.deepcopy(dict(caption_leaf))
            replacement["leaf_id"] = "caption_diff"
            visual_leaves.append(replacement)
            replaced_caption += 1
        else:
            visual_leaves.append(copy.deepcopy(dict(value)))
    if replaced_caption != 1:
        raise ValueError(
            f"source visual report needs one caption_diff leaf, found {replaced_caption}"
        )

    ids = {
        "task_bundle_id": str(old_root["task_bundle_id"]),
        "episode_id": str(old_root["episode_id"]),
    }
    context = Context(
        record={},
        task=None,
        ids=ids,
        out_dir=output_case_dir,
    )
    visual = composite_report(
        "scene_diff.visual_semantic_diff",
        context,
        visual_leaves,
        evidence=old_visual.get("evidence") or {},
    )
    visual["leaf_id"] = "visual_semantic_diff"
    scene = composite_report(
        "scene_diff",
        context,
        [structured, visual],
        evidence=old_scene.get("evidence") or {},
    )
    scene["leaf_id"] = "scene_diff"
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
    rebuilt["reports"] = _replace_top_report(
        top_reports,
        "gt_repair",
        root,
    )
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


def _image_paths(evidence_dir: Path) -> dict[str, tuple[Path, ...]]:
    result = {
        scene: tuple(evidence_dir / scene / f"view_{index}.png" for index in range(4))
        for scene in ("gt", "candidate")
    }
    missing = [
        str(path)
        for paths in result.values()
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "saved caption evidence is incomplete: " + ", ".join(missing)
        )
    return result


def measure_saved_caption(
    context: Context,
    images: Mapping[str, Sequence[Path]],
    *,
    source_result: Path,
) -> dict[str, Any]:
    """Run the production caption clients over a frozen eight-image pair."""

    instance_id = (
        f"{gt_caption_similarity.EVALUATOR_ID}:"
        f"{context.ids['episode_id']}"
    )
    image_provenance = {
        scene: [
            {"path": str(path), "sha256": _sha256(path)}
            for path in paths
        ]
        for scene, paths in images.items()
    }
    try:
        client = gt_caption_similarity.QwenCaptionClient()
        gt_caption = gt_caption_similarity._caption_scene(
            client,
            images["gt"],
            scene="gt",
        )
        candidate_caption = gt_caption_similarity._caption_scene(
            client,
            images["candidate"],
            scene="candidate",
        )
        embedded = gt_caption_similarity.QwenEmbeddingClient().embed_pair(
            (gt_caption.caption, candidate_caption.caption)
        )
        similarity = gt_caption_similarity._cosine(*embedded.vectors)
        overview_luma = {
            scene: [render.mean_luma(path) for path in paths]
            for scene, paths in images.items()
        }
        artifact_payload = {
            "schema_version": "gt-caption-similarity.v1",
            "metric_version": gt_caption_similarity.METRIC_VERSION,
            "render_protocol": CAPTION_ENVIRONMENT_RENDER_PROTOCOL,
            "caption_protocol": gt_caption_similarity.CAPTION_PROTOCOL,
            "embedding_protocol": gt_caption_similarity.EMBEDDING_PROTOCOL,
            "caption_model": {
                "base_url": gt_caption_similarity.CAPTION_BASE_URL,
                "model": gt_caption_similarity.CAPTION_MODEL,
            },
            "embedding_model": {
                "base_url": gt_caption_similarity.EMBEDDING_BASE_URL,
                "model": gt_caption_similarity.EMBEDDING_MODEL,
            },
            "captions": {
                "gt": gt_caption.caption,
                "candidate": candidate_caption.caption,
            },
            "caption_request_ids": {
                "gt": gt_caption.request_id,
                "candidate": candidate_caption.request_id,
            },
            "caption_usage": {
                "gt": dict(gt_caption.usage),
                "candidate": dict(candidate_caption.usage),
            },
            "caption_attempts": {
                "gt": gt_caption.attempt_count,
                "candidate": candidate_caption.attempt_count,
            },
            "caption_structured_modes": {
                "gt": gt_caption.structured_mode,
                "candidate": candidate_caption.structured_mode,
            },
            "embedding_request_id": embedded.request_id,
            "embedding_usage": dict(embedded.usage),
            "embedding_dimension": len(embedded.vectors[0]),
            "cosine_similarity": similarity,
            "cosine_distance": 1.0 - similarity,
            "overview_mean_luma": overview_luma,
            "saved_evidence_recomputation": {
                "source_result": str(source_result),
                "images": image_provenance,
                "ue_recapture_performed": False,
            },
        }
        rounded_similarity = round(similarity, 6)
        rounded_distance = round(1.0 - similarity, 6)
        evidence = {
            "scoring_policy": "direct_continuous_cosine_similarity",
            "caption_protocol": gt_caption_similarity.CAPTION_PROTOCOL,
            "render_protocol": CAPTION_ENVIRONMENT_RENDER_PROTOCOL,
            "embedding_protocol": gt_caption_similarity.EMBEDDING_PROTOCOL,
            "caption_model": gt_caption_similarity.CAPTION_MODEL,
            "caption_base_url": gt_caption_similarity.CAPTION_BASE_URL,
            "embedding_model": gt_caption_similarity.EMBEDDING_MODEL,
            "embedding_base_url": gt_caption_similarity.EMBEDDING_BASE_URL,
            "independent_caption_requests": True,
            "saved_evidence_recomputation": True,
            "source_result": str(source_result),
            "image_sha256": image_provenance,
        }
        metrics = {
            "metric_version": gt_caption_similarity.METRIC_VERSION,
            "cosine_similarity": rounded_similarity,
            "cosine_distance": rounded_distance,
            "embedding_dimension": len(embedded.vectors[0]),
            "caption_character_count": {
                "gt": len(gt_caption.caption),
                "candidate": len(candidate_caption.caption),
            },
            "caption_attempt_count": {
                "gt": gt_caption.attempt_count,
                "candidate": candidate_caption.attempt_count,
            },
            "caption_structured_modes": {
                "gt": gt_caption.structured_mode,
                "candidate": candidate_caption.structured_mode,
            },
            "model_call_count": (
                gt_caption.attempt_count + candidate_caption.attempt_count + 1
            ),
        }
        measurement = contracts.RawMeasurement(
            id=gt_caption_similarity.EVALUATOR_ID,
            instance_id=instance_id,
            metric_version=gt_caption_similarity.METRIC_VERSION,
            applicable=True,
            status=contracts.MEASURED,
            coverage=1.0,
            raw={
                "metrics": metrics,
                "evidence": evidence,
                "artifact_payload": artifact_payload,
            },
            evidence=(evidence,),
        )
    except Exception as exception:  # noqa: BLE001 - failure remains publishable
        diagnostics = [
            dict(value)
            for value in getattr(exception, "diagnostics", ())
            if isinstance(value, Mapping)
        ]
        failure_path = gt_caption_similarity._write_artifact(
            context,
            {
                "schema_version": "gt-caption-failure.v1",
                "exception_type": type(exception).__name__,
                "failure_reason": str(exception),
                "attempts": diagnostics,
                "source_result": str(source_result),
                "images": image_provenance,
            },
            "gt_caption_similarity_failure",
        )
        failure_evidence = {
            "render_protocol": CAPTION_ENVIRONMENT_RENDER_PROTOCOL,
            "attempt_count": len(diagnostics),
            "failure_response_artifact": failure_path,
            "saved_evidence_recomputation": True,
            "source_result": str(source_result),
            "image_sha256": image_provenance,
        }
        measurement = contracts.RawMeasurement(
            id=gt_caption_similarity.EVALUATOR_ID,
            instance_id=instance_id,
            metric_version=gt_caption_similarity.METRIC_VERSION,
            applicable=True,
            status=contracts.ERROR,
            coverage=None,
            raw=None,
            evidence=(failure_evidence,),
            failure_reason=f"{type(exception).__name__}: {exception}",
        )

    normalized = gt_caption_similarity.normalize(measurement, {})
    leaf = gt_caption_similarity.report_from_atomic(
        context,
        measurement,
        normalized,
    )
    leaf["leaf_id"] = "caption_diff"
    leaf_evidence = dict(leaf.get("evidence") or {})
    leaf_evidence.update(
        {
            "independent_captioning": True,
            "comparison_space": "text_embedding_cosine_distance",
        }
    )
    if leaf.get("status") == contracts.ERROR:
        leaf_metrics = dict(leaf.get("metrics") or {})
        leaf_metrics.update(
            {
                "upstream_status": contracts.ERROR,
                "partial_aggregation_policy": "omit_with_coverage_penalty",
            }
        )
        leaf["metrics"] = leaf_metrics
        leaf["status"] = "not_evaluated"
        leaf_evidence["partial_aggregation_policy"] = (
            "caption failure remains visible but does not erase available "
            "paired-visual evidence"
        )
    leaf["evidence"] = leaf_evidence
    return leaf


def recompute(source_result_path: Path, output_root: Path) -> Path:
    started = time.monotonic()
    source_result_path = source_result_path.resolve()
    source_result = json.loads(source_result_path.read_text())
    task_id = str(source_result.get("task_id") or "")
    if not task_id:
        raise ValueError("source result has no task_id")
    protocol_dir = CAPTION_ENVIRONMENT_RENDER_PROTOCOL.replace(".", "_")
    evidence_dir = (
        source_result_path.parent
        / "render_evidence"
        / task_id
        / protocol_dir
    )
    images = _image_paths(evidence_dir)
    output_case_dir = output_root.resolve() / "results" / task_id
    output_case_dir.mkdir(parents=True, exist_ok=True)
    ids = {
        "task_bundle_id": task_id,
        "episode_id": next(
            str(report["episode_id"])
            for report in source_result["reports"]
            if report.get("report_id") == "gt_repair"
        ),
    }
    context = Context(
        record={},
        task=None,
        ids=ids,
        out_dir=output_case_dir,
    )
    caption_leaf = measure_saved_caption(
        context,
        images,
        source_result=source_result_path,
    )
    code_files = [
        Path(gt_caption_similarity.__file__).resolve(),
        Path(gt_repair.__file__).resolve(),
        Path(__file__).resolve(),
    ]
    provenance = {
        "schema_version": "saved-caption-recomputation.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_result": str(source_result_path),
        "source_result_sha256": _sha256(source_result_path),
        "source_batch_status": source_result.get("batch_status"),
        "scope": "caption_diff_and_composite_ancestors_only",
        "ue_recapture_performed": False,
        "frozen_image_sha256": {
            scene: [
                {"path": str(path), "sha256": _sha256(path)}
                for path in paths
            ]
            for scene, paths in images.items()
        },
        "code_sha256": {
            str(path): _sha256(path)
            for path in code_files
        },
        "caption_status": caption_leaf.get("status"),
        "caption_score": caption_leaf.get("score"),
    }
    rebuilt = rebuild_result(
        source_result,
        caption_leaf,
        output_case_dir=output_case_dir,
        provenance=provenance,
    )
    timings = dict(rebuilt.get("timings") or {})
    timings["offline_caption_recompute_seconds"] = round(
        time.monotonic() - started,
        3,
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
            "Recompute only a failed caption leaf and its composite ancestors "
            "from frozen score-run images."
        )
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    output = recompute(args.source_result, args.out_dir)
    value = json.loads(output.read_text())
    gt_report = next(
        report for report in value["reports"] if report["report_id"] == "gt_repair"
    )
    print(
        json.dumps(
            {
                "result": str(output),
                "batch_status": value["batch_status"],
                "gt_repair_status": gt_report["status"],
                "gt_repair_score": gt_report["score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
