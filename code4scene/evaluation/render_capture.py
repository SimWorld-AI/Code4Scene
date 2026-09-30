"""Capture a task's candidate-only render evidence in the scoring editor.

The executor the paper's scorer ran before the verifiers: for every
``candidate_render`` request of ``verifiers.render_evidence_requests(task)`` it
plans the views from the open level's inventory with the reference camera
planner, captures them through the bridge's screenshot commands, and enforces
the request's frame-quality policy (camera recovery plans, then one relit
recapture of the whole set). Text-to-scene tasks request one such protocol:
the four-view overview gallery that Overview Alignment judges.

The candidate level must already be open in the scoring editor; nothing here
loads a level. ``paired_render`` requests (the image-to-scene paired views) are
report-only and are not captured.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core import inventory
from . import render, scene_graph_capture

CANDIDATE = "candidate"


def request_settings(request: Any) -> dict[str, Any]:
    return {
        "protocol": request.protocol,
        "channels": request.channels,
        "count": request.view_count,
        "width": request.width,
        "height": request.height,
        "timeout": request.timeout_s,
        "lighting_policy": request.lighting_policy,
        "exposure_policy": request.exposure_policy,
        "camera_protocol": request.camera_protocol,
        "scene_alignment": request.scene_alignment,
        "camera_plan_policy": request.camera_plan_policy,
        "frame_quality_policy": request.frame_quality_policy,
        "scene_environment": request.scene_environment,
        "mode": "gt_paired" if request.kind == "paired_render" else "candidate_only",
    }


def capture_scene_group(
    bridge: Any,
    request: Any,
    result: render.RenderSet,
    scene_name: str,
    scene_root: Path,
    views: Sequence[Any],
    anchor: Any,
    *,
    camera_retry_plans: Sequence[scene_graph_capture.CameraPlan] = (),
) -> None:
    """Capture one protocol-owned group and enforce its acquisition policy."""

    settings = request_settings(request)
    active_views = list(views)
    active_anchor = anchor

    def capture_once() -> None:
        render.capture_set(
            bridge,
            scene_root,
            scene_name,
            views=active_views,
            channels=tuple(settings["channels"]),
            renders=result,
            width=int(settings["width"]),
            height=int(settings["height"]),
            timeout=float(settings["timeout"]),
            lighting_policy=settings["lighting_policy"],
            exposure_policy=settings.get("exposure_policy"),
            camera_protocol=settings.get("camera_protocol", render.RING_PROTOCOL),
            scene_anchor_cm=active_anchor,
            # The whole-group guard owns its retry. The renderer fallback
            # operates per frame and would mix lighting within a set.
            allow_relight_fallback=(request.frame_quality_policy is None),
        )

    capture_once()
    policy = request.frame_quality_policy
    if policy is None:
        return
    if policy not in scene_graph_capture.SUPPORTED_FRAME_QUALITY_POLICIES:
        raise render.RenderError(f"unsupported frame quality policy {policy!r}")
    if policy == scene_graph_capture.CAPTION_FRAME_QUALITY:
        owner = "caption"
    elif policy == scene_graph_capture.REFERENCE_FRAME_QUALITY:
        owner = "reference"
    else:
        raise render.RenderError(f"frame quality policy {policy!r} has no current group executor")
    audit_path = scene_root / f"{owner}_frame_quality.json"

    def rgb_paths() -> dict[str, str]:
        return {
            str(view): str(channels["rgb"])
            for view, channels in sorted((result.images.get(scene_name) or {}).items())
            if channels.get("rgb")
        }

    quality_measure = render.image_quality if owner == "reference" else render.mean_luma
    initial = scene_graph_capture.frame_quality(rgb_paths(), quality_measure, policy=policy)
    if owner == "caption":
        initial = scene_graph_capture.apply_indoor_overview_supplemental_quality_override(
            initial, plan_audit=result.camera_plan_audit.get(scene_name) or {})
    quality: dict[str, Any] = {
        "policy": policy,
        "protocol": request.protocol,
        "scene": scene_name,
        "source": "code4scene.evaluation.render_capture",
        "initial": initial,
        "relight_attempted": False,
        "saved_level": False,
    }

    def write_audit() -> None:
        audit_path.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")

    current = initial
    retry_records: list[dict[str, Any]] = []
    if owner == "reference" and not current["accepted"] and camera_retry_plans:
        def preserve_current(suffix: str) -> dict[str, str]:
            preserved_paths: dict[str, str] = {}
            for view, value in rgb_paths().items():
                path = Path(value)
                preserved = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
                preserved.unlink(missing_ok=True)
                if path.exists():
                    path.replace(preserved)
                    preserved_paths[view] = str(preserved)
            return preserved_paths

        initial_plan_audit = dict(result.camera_plan_audit.get(scene_name) or {})
        quality["camera_retry_attempted"] = True
        quality["primary_camera_paths"] = preserve_current("primary_camera")
        for retry_index, retry_plan in enumerate(camera_retry_plans, start=1):
            if retry_records:
                previous_record = retry_records[-1]
                previous_portfolio = str(previous_record["plan"].get("portfolio") or "camera")
                safe_portfolio = "".join(
                    value if value.isalnum() or value in "-_" else "_" for value in previous_portfolio)
                previous_record["paths"] = preserve_current(
                    f"camera_retry_{retry_index - 1}_{safe_portfolio}")
            active_views = list(retry_plan.views)
            active_anchor = retry_plan.anchor_cm
            capture_once()
            current = scene_graph_capture.frame_quality(rgb_paths(), quality_measure, policy=policy)
            current = scene_graph_capture.apply_candidate_near_visibility_quality_override(
                current, plan_audit=retry_plan.audit)
            retry_records.append({
                "attempt": retry_index,
                "plan": dict(retry_plan.audit),
                "quality": current,
                "paths": rgb_paths(),
            })
            quality["camera_retry"] = current
            result.camera_plan_audit[scene_name] = {
                **dict(retry_plan.audit),
                "retry_from_camera_specs_sha256": initial_plan_audit.get("camera_specs_sha256"),
                "camera_recovery_attempt_index": retry_index,
            }
            if current["accepted"]:
                break
        quality["camera_retries"] = retry_records
        quality["camera_retry_count"] = len(retry_records)
        quality["selected_camera_portfolio"] = (
            retry_records[-1]["plan"].get("portfolio") if current["accepted"] and retry_records else None)
        result.camera_plan_audit[scene_name]["camera_recovery_attempts"] = [
            {
                "attempt": record["attempt"],
                "portfolio": record["plan"].get("portfolio"),
                "camera_specs_sha256": record["plan"].get("camera_specs_sha256"),
                "accepted": bool(record["quality"]["accepted"]),
            }
            for record in retry_records
        ]
    else:
        quality["camera_retry_attempted"] = False

    if current["unreadable_views"]:
        write_audit()
        raise render.RenderError(
            f"{owner} {scene_name} evidence contains unreadable RGB views: {current['unreadable_views']}")
    if current["accepted"]:
        quality["accepted"] = True
        result.capture_overrides.append(quality)
        write_audit()
        return

    if (
        owner == "reference"
        and scene_name == CANDIDATE
        and not current.get("unreadable_views")
        and not current.get("too_dark_views")
        and not current.get("overexposed_views")
        and (current.get("low_detail_views") or current.get("occluded_views"))
    ):
        # Low structural detail and visible occlusion are Candidate quality
        # signals once the bounded camera portfolios have been exhausted. The
        # RGB files are still valid evidence for the overview judge.
        quality["accepted"] = True
        quality["selected_camera_portfolio"] = (
            retry_records[-1]["plan"].get("portfolio")
            if retry_records
            else (result.camera_plan_audit.get(scene_name) or {}).get("portfolio")
        )
        quality["acceptance_override"] = {
            "policy": "scorable-readable-low-quality-candidate",
            "reason": "readable_low_detail_or_occlusion_is_candidate_quality",
            "frame_quality_accepted": False,
            "preserved_low_detail_views": list(current.get("low_detail_views") or []),
            "preserved_occluded_views": list(current.get("occluded_views") or []),
        }
        result.capture_overrides.append(quality)
        write_audit()
        return

    # Preserve the native complete set, then recapture every view under one
    # temporary neutral rig. A partially relit gallery is invalid evidence.
    native_paths: dict[str, str] = {}
    for view, value in rgb_paths().items():
        path = Path(value)
        native = path.with_name(f"{path.stem}.native{path.suffix}")
        native.unlink(missing_ok=True)
        path.replace(native)
        native_paths[view] = str(native)
    quality["native_paths"] = native_paths
    quality["relight_attempted"] = True
    spawned = render.add_relight_rig(bridge, timeout=float(settings["timeout"]))
    quality["relight_spawned_actors"] = spawned
    if spawned <= 0:
        quality["relight_removed_actors"] = render.remove_relight_rig(
            bridge, timeout=float(settings["timeout"]))
        write_audit()
        raise render.RenderError(
            f"{owner} {scene_name} evidence is near-black and the temporary relight rig could not be created")
    try:
        capture_once()
    finally:
        quality["relight_removed_actors"] = render.remove_relight_rig(
            bridge, timeout=float(settings["timeout"]))
    final = scene_graph_capture.frame_quality(rgb_paths(), quality_measure, policy=policy)
    if owner == "caption":
        final = scene_graph_capture.apply_indoor_overview_supplemental_quality_override(
            final, plan_audit=result.camera_plan_audit.get(scene_name) or {})
    quality["final"] = final
    result.capture_overrides.append(quality)
    write_audit()
    if quality["relight_removed_actors"] != spawned:
        raise render.RenderError(
            f"{owner} temporary relight cleanup was incomplete: spawned {spawned}, "
            f"removed {quality['relight_removed_actors']}")
    if not final["accepted"]:
        raise render.RenderError(
            f"{owner} {scene_name} evidence remains unusable after relight: "
            f"near_black={final['near_black_views']}, unreadable={final['unreadable_views']}")


def capture_candidate_render(bridge: Any, request: Any, root: Path) -> render.RenderSet:
    """Execute one candidate-only protocol on the open level."""

    if request.camera_plan_policy != scene_graph_capture.REFERENCE_CAMERA_PLAN:
        raise render.RenderError("candidate-only render evidence requires the current reference camera planner")
    candidate_inventory = inventory.read(bridge)
    plan = scene_graph_capture.plan_candidate_grounded_overview(candidate_inventory, count=request.view_count)
    near_recovery = scene_graph_capture.plan_candidate_near_visibility_recovery(
        candidate_inventory, count=request.view_count)
    clearance_retry = scene_graph_capture.plan_candidate_grounded_overview(
        candidate_inventory, count=request.view_count, clearance_retry=True)
    result = render.RenderSet()
    result.camera_plan_audit[CANDIDATE] = dict(plan.audit)
    capture_scene_group(bridge, request, result, CANDIDATE, root, list(plan.views), plan.anchor_cm,
                        camera_retry_plans=(near_recovery, clearance_retry))
    return result


def capture_render_evidence(
    task: Any, bridge: Any, out_dir: Path,
) -> tuple[dict[str, render.RenderSet], dict[str, str]]:
    """Capture every candidate-only protocol the task's verifiers request.

    Returns the render sets by protocol and, by protocol, why a capture failed.
    """

    from .verifiers import render_evidence_requests

    store: dict[str, render.RenderSet] = {}
    errors: dict[str, str] = {}
    for request in render_evidence_requests(task):
        if request.kind != "candidate_render":
            continue
        protocol_dir = "".join(value if value.isalnum() or value in "-_" else "_" for value in request.protocol)
        root = Path(out_dir) / "render_evidence" / protocol_dir / CANDIDATE
        try:
            store[request.protocol] = capture_candidate_render(bridge, request, root)
        except Exception as exception:  # noqa: BLE001 - the verifier sees the absence
            errors[request.protocol] = f"{type(exception).__name__}: {exception}"
    return store, errors


__all__ = [
    "capture_candidate_render",
    "capture_render_evidence",
    "capture_scene_group",
    "request_settings",
]
