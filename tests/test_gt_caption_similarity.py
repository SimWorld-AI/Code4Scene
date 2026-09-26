"""Independent GT/candidate captions followed by deterministic cosine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from code4scene.evaluation import render, scene_graph_capture
from code4scene.evaluation.context import Context
from code4scene.evaluation.render_evidence import CAPTION_ENVIRONMENT_RENDER_PROTOCOL
from code4scene.evaluation.verifiers import gt_caption_similarity


IDS = {"task_bundle_id": "bundle", "episode_id": "episode"}


def _renders(
    tmp_path,
    *,
    camera_mismatch=False,
    camera_precision_jitter=False,
    blank_view=False,
):
    shots = render.RenderSet()
    views = tuple(f"view_{index}" for index in range(4))
    for scene in ("gt", "candidate"):
        for view in views:
            path = tmp_path / scene / f"{view}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (16, 16),
                (0, 0, 0) if blank_view and scene == "gt" and view == "view_0"
                else (128, 128, 128),
            ).save(path)
            shots.add(scene, view, "rgb", str(path))
        shots.capture_settings[scene] = {
            "camera_protocol": render.GT_CAPTION_ENVIRONMENT_SCENE_GRAPH_PROTOCOL,
            "width": 1280,
            "height": 720,
            "lighting_policy": render.LIGHTING_NORMALIZATION_POLICY,
            "exposure_policy": render.EXPOSURE_NORMALIZATION_POLICY,
        }
        shots.camera_plan_audit[scene] = {
            "planner_id": scene_graph_capture.CAPTION_CAMERA_PLAN,
            "pairing_policy": "environment-routed-frozen-absolute-cameras",
            "planning_source": "gt",
            "scene_environment": "outdoor",
            "camera_specs_sha256": "camera-plan",
        }
        shots.scene_anchors_cm[scene] = [100.0, 200.0, 0.0]
        shots.camera_specs[scene] = {
            view: {
                "relative_location_cm": [
                    20_000.0,
                    0.0,
                    11_000.0,
                ],
                "rotation_deg": [0.0, -28.8, 180.0],
                "fov_deg": 55.0,
                "width": 1280,
                "height": 720,
                "camera_protocol": render.GT_CAPTION_ENVIRONMENT_SCENE_GRAPH_PROTOCOL,
                "lighting_policy": render.LIGHTING_NORMALIZATION_POLICY,
                "exposure_policy": render.EXPOSURE_NORMALIZATION_POLICY,
            }
            for view in views
        }
        shots.capture_overrides.append(
            {
                "policy": scene_graph_capture.CAPTION_FRAME_QUALITY,
                "scene": scene,
                "accepted": True,
            }
        )
    if camera_mismatch:
        shots.camera_specs["candidate"]["view_0"]["fov_deg"] = 60.0
    if camera_precision_jitter:
        for spec in shots.camera_specs["candidate"].values():
            spec["relative_location_cm"][2] = 11_000.02
    return shots


def _context(tmp_path, shots, *, answer_key=True):
    task_path = tmp_path / "task.yaml"
    task_path.write_text("placeholder")
    if answer_key:
        task_path.with_suffix(".label.json").write_text(
            json.dumps({"gt_id": "gt-west", "canonical_map": "/Game/Canonical"})
        )
    return Context(
        record={},
        task=SimpleNamespace(
            path=task_path,
            prompt="a western town",
            scene_environment="outdoor",
        ),
        ids=IDS,
        render_evidence={CAPTION_ENVIRONMENT_RENDER_PROTOCOL: shots},
        out_dir=tmp_path / "run",
        spec={"name": "scene_diff"},
    )


class _FakeCaptionClient:
    def __init__(self, captions):
        self.captions = iter(captions)
        self.calls = []

    def caption(self, images):
        self.calls.append(tuple(images))
        number = len(self.calls)
        return gt_caption_similarity.CaptionResult(
            next(self.captions), f"caption-{number}", {"total_tokens": 12}
        )


class _FakeEmbeddingClient:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed_pair(self, captions):
        self.calls.append(tuple(captions))
        return gt_caption_similarity.EmbeddingResult(
            self.vectors,
            "embedding-1",
            {"total_tokens": 20},
        )


def _caption_images(tmp_path: Path) -> tuple[Path, ...]:
    images = []
    for index in range(4):
        path = tmp_path / "caption-client" / f"view_{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (128, 128, 128)).save(path)
        images.append(path)
    return tuple(images)


def _tool_response(caption: str) -> dict:
    return {
        "id": "chatcmpl-tool",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "record_scene_caption",
                                "arguments": json.dumps({"caption": caption}),
                            }
                        }
                    ],
                },
            }
        ],
        "usage": {"total_tokens": 123},
    }


def test_qwen_caption_preserves_tool_call_as_first_attempt(
    tmp_path,
    monkeypatch,
):
    bodies = []

    def post(_base_url, _path, body, *, timeout_s):
        assert timeout_s == gt_caption_similarity.CAPTION_TIMEOUT_S
        bodies.append(body)
        return _tool_response("A valid scene caption.")

    monkeypatch.setattr(gt_caption_similarity, "_post_json", post)

    result = gt_caption_similarity.QwenCaptionClient().caption(
        _caption_images(tmp_path)
    )

    assert result.caption == "A valid scene caption."
    assert result.attempt_count == 1
    assert result.structured_mode == "tool_call"
    assert "tools" in bodies[0]
    assert "tool_choice" in bodies[0]
    assert "response_format" not in bodies[0]


def test_qwen_caption_retries_format_drift_with_json_schema(
    tmp_path,
    monkeypatch,
):
    bodies = []
    responses = iter(
        (
            {
                "id": "chatcmpl-malformed",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "A plain unstructured caption.",
                            "tool_calls": [],
                        },
                    }
                ],
                "usage": {"total_tokens": 100},
            },
            {
                "id": "chatcmpl-schema",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"caption": "A schema-conformant caption."}
                            )
                        },
                    }
                ],
                "usage": {"total_tokens": 110},
            },
        )
    )

    def post(_base_url, _path, body, *, timeout_s):
        assert timeout_s == gt_caption_similarity.CAPTION_TIMEOUT_S
        bodies.append(body)
        return next(responses)

    monkeypatch.setattr(gt_caption_similarity, "_post_json", post)

    result = gt_caption_similarity.QwenCaptionClient().caption(
        _caption_images(tmp_path)
    )

    assert result.caption == "A schema-conformant caption."
    assert result.attempt_count == 2
    assert result.structured_mode == "json_schema"
    assert "tools" in bodies[0]
    assert "response_format" not in bodies[0]
    assert bodies[1]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "scene_caption",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"caption": {"type": "string"}},
                "required": ["caption"],
            },
        },
    }
    assert "tools" not in bodies[1]


def test_qwen_caption_exhaustion_records_compact_response_diagnostics(
    tmp_path,
    monkeypatch,
):
    calls = []

    def post(_base_url, _path, body, *, timeout_s):
        calls.append(body)
        return {
            "id": f"chatcmpl-bad-{len(calls)}",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "still not structured",
                        "tool_calls": [],
                        "reasoning": "hidden reasoning",
                    },
                }
            ],
            "usage": {"total_tokens": 99},
        }

    monkeypatch.setattr(gt_caption_similarity, "_post_json", post)

    with pytest.raises(gt_caption_similarity.CaptionSimilarityError) as caught:
        gt_caption_similarity.QwenCaptionClient().caption(
            _caption_images(tmp_path)
        )

    error = caught.value
    assert len(calls) == gt_caption_similarity.CAPTION_MAX_ATTEMPTS
    assert len(error.diagnostics) == gt_caption_similarity.CAPTION_MAX_ATTEMPTS
    assert error.diagnostics[0]["structured_mode"] == "tool_call"
    assert error.diagnostics[1]["structured_mode"] == "json_schema"
    assert error.diagnostics[0]["response"]["content"] == "still not structured"
    assert error.diagnostics[0]["response"]["reasoning_character_count"] == 16


def test_caption_failure_response_is_persisted(tmp_path, monkeypatch):
    class BrokenCaption:
        def caption(self, _images):
            raise gt_caption_similarity.CaptionSimilarityError(
                "malformed response",
                diagnostics=(
                    {
                        "attempt": 1,
                        "structured_mode": "tool_call",
                        "response": {"content": "plain text"},
                    },
                ),
            )

    monkeypatch.setattr(
        gt_caption_similarity,
        "_caption_client",
        lambda: BrokenCaption(),
    )

    report = gt_caption_similarity.verify(_context(tmp_path, _renders(tmp_path)))

    assert report["status"] == "error"
    artifact = Path(report["artifacts"]["failure_response"])
    payload = json.loads(artifact.read_text())
    assert payload["schema_version"] == "gt-caption-failure.v1"
    assert payload["attempts"][0]["scene"] == "gt"
    assert payload["attempts"][0]["response"]["content"] == "plain text"


def test_independent_captions_feed_one_embedding_request_and_cosine(
    tmp_path,
    monkeypatch,
    infra,
):
    caption_client = _FakeCaptionClient(
        ("A desert town with wooden buildings.", "A frontier town on sandy ground.")
    )
    embedding_client = _FakeEmbeddingClient(((1.0, 0.0), (0.8, 0.6)))
    monkeypatch.setattr(gt_caption_similarity, "_caption_client", lambda: caption_client)
    monkeypatch.setattr(
        gt_caption_similarity, "_embedding_client", lambda: embedding_client
    )

    report = gt_caption_similarity.verify(_context(tmp_path, _renders(tmp_path)))

    parsed = infra.VerifierReport.from_json_dict(report)
    assert parsed.status == "measured"
    assert report["score"] == 0.8
    assert report["metrics"]["cosine_similarity"] == 0.8
    assert report["metrics"]["cosine_distance"] == 0.2
    assert report["metrics"]["model_call_count"] == 3
    assert len(caption_client.calls) == 2
    assert len(caption_client.calls[0]) == len(caption_client.calls[1]) == 4
    assert all("/gt/" in str(path) for path in caption_client.calls[0])
    assert all("/candidate/" in str(path) for path in caption_client.calls[1])
    assert embedding_client.calls == [
        (
            "A desert town with wooden buildings.",
            "A frontier town on sandy ground.",
        )
    ]
    artifact = json.loads(Path(report["artifacts"]["comparison"]).read_text())
    assert artifact["captions"]["gt"].startswith("A desert town")
    assert artifact["captions"]["candidate"].startswith("A frontier town")
    assert artifact["embedding_dimension"] == 2
    assert "embedding_vector_sha256" not in artifact
    assert "overview_image_sha256" not in artifact
    assert "vectors" not in artifact


def test_missing_answer_key_is_an_error_without_model_calls(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        gt_caption_similarity,
        "_caption_client",
        lambda: calls.append("caption"),
    )

    report = gt_caption_similarity.verify(
        _context(tmp_path, _renders(tmp_path), answer_key=False)
    )

    assert report["status"] == "error"
    assert report["score"] is None
    assert "answer key" in report["failure_reason"]
    assert calls == []


def test_camera_mismatch_is_withheld_before_captioning(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        gt_caption_similarity,
        "_caption_client",
        lambda: calls.append("caption"),
    )

    report = gt_caption_similarity.verify(
        _context(tmp_path, _renders(tmp_path, camera_mismatch=True))
    )

    assert report["status"] == "error"
    assert report["score"] is None
    assert "cameras differ" in report["failure_reason"]
    assert calls == []


def test_camera_serialization_difference_is_a_pairing_mismatch(
    tmp_path,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        gt_caption_similarity,
        "_caption_client",
        lambda: calls.append("caption"),
    )

    report = gt_caption_similarity.verify(
        _context(tmp_path, _renders(tmp_path, camera_precision_jitter=True))
    )

    assert report["status"] == "error"
    assert report["score"] is None
    assert "cameras differ" in report["failure_reason"]
    assert calls == []


def test_embedding_failure_is_not_a_zero(tmp_path, monkeypatch):
    caption_client = _FakeCaptionClient(("GT caption", "Candidate caption"))

    class BrokenEmbedding:
        def embed_pair(self, captions):
            raise ConnectionError("embedding service unavailable")

    monkeypatch.setattr(gt_caption_similarity, "_caption_client", lambda: caption_client)
    monkeypatch.setattr(
        gt_caption_similarity, "_embedding_client", lambda: BrokenEmbedding()
    )

    report = gt_caption_similarity.verify(_context(tmp_path, _renders(tmp_path)))

    assert report["status"] == "error"
    assert report["score"] is None
    assert "embedding service unavailable" in report["failure_reason"]


def test_blank_overview_is_withheld_before_captioning(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        gt_caption_similarity,
        "_caption_client",
        lambda: calls.append("caption"),
    )

    report = gt_caption_similarity.verify(
        _context(tmp_path, _renders(tmp_path, blank_view=True))
    )

    assert report["status"] == "error"
    assert report["score"] is None
    assert "blank overview" in report["failure_reason"]
    assert calls == []


def test_models_are_fixed_and_endpoints_are_user_supplied():
    assert gt_caption_similarity.CAPTION_MODEL == "Qwen/Qwen3.8-27B"
    assert gt_caption_similarity.EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    with pytest.raises(gt_caption_similarity.CaptionSimilarityError,
                       match="CODE4SCENE_EMBED_BASE_URL"):
        gt_caption_similarity._post_json("", "/embeddings", {}, timeout_s=1.0)
