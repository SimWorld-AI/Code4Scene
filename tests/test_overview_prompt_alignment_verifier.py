from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image

from code4scene.evaluation import render, scene_graph_capture
from code4scene.evaluation.context import Context
from code4scene.evaluation.verifiers import overview_prompt_alignment


IDS = {"task_bundle_id": "bundle", "episode_id": "episode"}


def _renders(tmp_path):
    value = render.RenderSet()
    value.scene_anchors_cm["candidate"] = [100.0, 200.0, 300.0]
    value.camera_specs["candidate"] = {}
    for index in range(4):
        path = tmp_path / f"view_{index}.png"
        Image.new("RGB", (64, 36), (80 + index, 100, 120)).save(path)
        name = f"view_{index}"
        value.add("candidate", name, "rgb", str(path))
        value.camera_specs["candidate"][name] = {
            "relative_location_cm": [float(index), 2.0, 3.0],
            "rotation_deg": [0.0, -30.0, float(index * 90)],
        }
    return value


def _measured(frames):
    return {
        "schema_version": "overview-prompt-alignment.v6",
        "metric_version": "direct-multiview-overview-alignment",
        "status": "measured",
        "model_label": "fake",
        "score": 0.42,
        "overview_alignment_score": 0.6,
        "structural_integrity_score": 0.7,
        "structural_integrity_status": "degraded",
        "structural_adjustment_multiplier": 0.925,
        "overview_score_before_severe_cap": 0.555,
        "severe_structural_cap_eligible": False,
        "severe_structural_cap_applied": False,
        "severe_structural_cap": 0.4,
        "structural_adjustment_policy": {
            "soft_floor": 0.75,
            "structural_score_weight": 0.25,
            "severe_score_cap": 0.4,
            "severe_minimum_confidence": 0.8,
            "severe_minimum_evidence_views": 2,
            "severe_requires_issue_text_and_category": True,
        },
        "score_policy": "soft-intrinsic-structural-adjustment-overview-alignment",
        "calibration_status": "not_human_calibrated",
        "explanation_protocol": "non-scoring-explanation",
        "alignment_protocol": "direct-alignment",
        "dimension_weights": {"global_prompt_alignment": 1.0},
        "dimensions": {
            "global_prompt_alignment": {
                "score": 0.6,
                "weight": 1.0,
                "rationale": "partial match",
            }
        },
        "summary": "partial",
        "visible_scene_summary": "A visible generated scene.",
        "matched_prompt_elements": ["a town"],
        "missing_or_unsupported_prompt_elements": ["requested detail"],
        "allowed_nonstandard_geometry": [],
        "structural_geometry_integrity": {
            "score": 0.7,
            "status": "degraded",
            "issues": ["A wall is visibly stretched."],
            "issue_categories": ["mesh_deformation"],
            "evidence_view_indices": [0, 1],
            "severe_intrinsic_corruption": False,
            "confidence": 0.9,
            "rationale": "Localized intrinsic mesh deformation is visible.",
        },
        "prompt": "Build a town.",
        "prompt_sha256": "abc",
        "frames": [dict(value) for value in frames],
        "model": {"name": "fake"},
        "calls": {"count": 2},
        "computed_at": "2026-08-29T00:00:00+00:00",
    }


def test_evidence_request_is_candidate_only_scene_graph_gallery():
    request = overview_prompt_alignment.evidence_requests(None, {})[0]

    assert request.kind == "candidate_render"
    assert request.channels == ("rgb",)
    assert request.view_count == 4
    assert request.protocol == overview_prompt_alignment.RENDER_PROTOCOL
    assert request.camera_protocol == render.REFERENCE_OVERVIEW_SCENE_GRAPH_PROTOCOL
    assert request.camera_plan_policy == scene_graph_capture.REFERENCE_CAMERA_PLAN
    assert request.frame_quality_policy == scene_graph_capture.REFERENCE_FRAME_QUALITY


def test_report_is_measured_but_excluded_from_aggregate(monkeypatch, tmp_path):
    renders = _renders(tmp_path)
    observed = {}

    def evaluate(prompt, frames, **_kwargs):
        observed["prompt"] = prompt
        observed["frames"] = frames
        return _measured(frames)

    monkeypatch.setattr(
        overview_prompt_alignment,
        "evaluate_overview_frames",
        evaluate,
    )
    context = Context(
        record={},
        task=SimpleNamespace(prompt="Build a town."),
        ids=IDS,
        out_dir=tmp_path / "result",
        spec={"name": "overview_prompt_alignment"},
        render_evidence={overview_prompt_alignment.RENDER_PROTOCOL: renders},
    )

    report = overview_prompt_alignment.verify(context)

    assert report["status"] == "measured"
    assert report["score"] == 0.42
    assert report["contributes_to_aggregate"] is False
    assert report["score_role"] == "report_only"
    assert observed["prompt"] == "Build a town."
    assert [value["frame_id"] for value in observed["frames"]] == [
        "formal_view_0",
        "formal_view_1",
        "formal_view_2",
        "formal_view_3",
    ]
    artifact = report["artifacts"]["overview_prompt_alignment"]
    payload = json.loads(open(artifact, encoding="utf-8").read())
    assert payload["offline"] is False
    assert payload["ue_recapture_performed"] is True


def test_missing_addressed_evidence_is_an_unscored_error(tmp_path):
    context = Context(
        record={},
        task=SimpleNamespace(prompt="Build a town."),
        ids=IDS,
        out_dir=tmp_path,
        spec={"name": "overview_prompt_alignment"},
    )

    report = overview_prompt_alignment.verify(context)

    assert report["status"] == "error"
    assert report["score"] is None
    assert overview_prompt_alignment.RENDER_PROTOCOL in report["failure_reason"]
