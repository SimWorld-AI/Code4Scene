"""Current RGB/Base Color/Scene Depth visual judging contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from code4scene.evaluation import render, verifiers, vlm_model_config
from code4scene.evaluation.verifiers.vlm_as_judge import (
    EvidenceImage, Judge, JudgeError, JudgeRequest, backend_from_policy,
    load_policy, load_rubric,
)
from code4scene.evaluation.verifiers.vlm_as_judge import evidence, openai_compat
from code4scene.tasks import task as task_mod
from code4scene.resources import config_file


RUBRIC = config_file("rubrics", "visual-gt-paired.yaml")
POLICY = config_file("judge-policies", "visual-gt-paired-formal.yaml")


def _raw_files(tmp_path, views=("view_0", "view_1")):
    items = []
    for view in views:
        for channel in ("rgb", "base_color", "scene_depth"):
            path = tmp_path / f"{view}_{channel}.png"
            path.write_bytes(b"image")
            items.append(EvidenceImage(path=path, view=view, channel=channel))
    return items


def _all_scores(request, *, overrides=None, other=4):
    scores = {criterion.id: {"score": other, "rationale": "visible evidence"}
              for criterion in request.rubric.criteria}
    for criterion, score in (overrides or {}).items():
        scores[criterion]["score"] = score
    return scores


@pytest.mark.parametrize("override", ["", "   ", "/", "///"])
def test_blank_vlm_base_url_means_not_configured(monkeypatch, override):
    monkeypatch.setenv(vlm_model_config.BASE_URL_ENV, override)

    assert vlm_model_config.base_url() == ""
    with pytest.raises(vlm_model_config.JudgeEndpointNotConfigured):
        vlm_model_config.require_base_url()


def test_no_judge_endpoint_is_shipped(monkeypatch):
    monkeypatch.delenv(vlm_model_config.BASE_URL_ENV, raising=False)

    assert vlm_model_config.DEFAULT_BASE_URL == ""
    assert vlm_model_config.base_url() == ""
    monkeypatch.setenv(vlm_model_config.BASE_URL_ENV, "http://judge.example:9000/v1/")
    assert vlm_model_config.require_base_url() == "http://judge.example:9000/v1"


def test_current_rubric_declares_three_channels_and_channel_specific_questions():
    rubric = load_rubric(RUBRIC)

    assert rubric.name == "visual-gt-paired"
    assert rubric.channels == ("rgb", "base_color", "scene_depth")
    by_id = {criterion.id: criterion for criterion in rubric.criteria}
    assert by_id["material_fidelity"].channels == ("base_color",)
    assert by_id["silhouette_geometry"].channels == ("scene_depth", "rgb")


def test_current_rubric_separates_material_geometry_and_lighting_evidence():
    rubric = load_rubric(RUBRIC)
    by_id = {criterion.id: criterion for criterion in rubric.criteria}

    assert by_id["material_fidelity"].channels == ("base_color",)
    assert by_id["lighting_fidelity"].channels == ("rgb",)
    assert "deterministic" in by_id["silhouette_geometry"].guidance


def test_formal_policy_freezes_the_whole_visual_protocol():
    policy = load_policy(POLICY)

    assert policy.name == "visual-gt-paired-formal"
    assert policy.rubric.name == "visual-gt-paired"
    assert policy.backend == "openai_compat"
    # The serving location is transport, not policy: nothing is frozen.
    assert "base_url" not in yaml.safe_load(POLICY.read_text())["judge"]
    assert policy.base_url == ""
    assert policy.model == "Qwen/Qwen3.8-27B"
    assert policy.enable_thinking is False
    assert policy.view_count == policy.minimum_complete_viewpoints == 4
    assert (policy.width, policy.height) == (1280, 720)
    assert policy.channels == ("rgb", "base_color", "scene_depth")
    assert (policy.depth_near_cm, policy.depth_far_cm) == (0.0, 26000.0)


def test_policy_backend_takes_the_endpoint_from_the_caller_or_environment(monkeypatch):
    policy = load_policy(POLICY)

    backend = backend_from_policy(
        policy,
        base_url_override="http://judge.example:8013/v1/",
    )
    assert backend.base_url == "http://judge.example:8013/v1"

    monkeypatch.setenv("CODE4SCENE_VLM_BASE_URL", "http://judge.example:8011/v1")
    assert backend_from_policy(policy).base_url == "http://judge.example:8011/v1"
    monkeypatch.delenv("CODE4SCENE_VLM_BASE_URL")
    with pytest.raises(Exception, match="CODE4SCENE_VLM_BASE_URL"):
        backend_from_policy(policy)
    assert backend.model == policy.model
    assert backend.temperature == policy.temperature
    assert backend.seed == policy.seed


def test_formal_policy_loads_its_frozen_rubric_copy(tmp_path):
    policy_dir = tmp_path / "judge-policies"
    rubric_dir = tmp_path / "rubrics"
    policy_dir.mkdir()
    rubric_dir.mkdir()
    (policy_dir / POLICY.name).write_bytes(POLICY.read_bytes())
    changed = RUBRIC.read_text() + "\n# copied into frozen artifact\n"
    (rubric_dir / RUBRIC.name).write_text(changed)

    policy = load_policy(policy_dir / POLICY.name)
    assert policy.rubric.name == "visual-gt-paired"


def test_formal_policy_does_not_require_a_task_side_content_hash(tmp_path):
    changed_policy = tmp_path / POLICY.name
    changed_policy.write_text(
        POLICY.read_text().replace(
            "../rubrics/visual-gt-paired.yaml", str(RUBRIC)
        )
        + "\n# edited in place\n"
    )

    assert load_policy(changed_policy).name == "visual-gt-paired-formal"


def test_calibration_rubric_preserves_raw_semantic_scores(tmp_path):
    rubric = load_rubric(RUBRIC)
    judge = Judge(
        rubric=rubric,
        backend=lambda request: _all_scores(
            request, overrides={"composition_layout": 1}
        ),
        model="judge-model")

    verdict = judge.score(
        "a Korean palace courtyard", [], {"floating_rate": 0.0},
        evidence=_raw_files(tmp_path))

    assert verdict.judged > 0.7
    assert verdict.ceiling == 1.0
    assert verdict.final == verdict.judged
    assert verdict.scores["composition_layout"] == 1


def test_three_channels_from_one_camera_are_still_only_one_view(tmp_path):
    judge = Judge(
        rubric=load_rubric(RUBRIC),
        backend=lambda request: _all_scores(request))

    with pytest.raises(JudgeError, match="1 viewpoint"):
        judge.score(
            "a palace", [], {"floating_rate": 0.0},
            evidence=_raw_files(tmp_path, views=("view_0",)))


def test_backend_labels_view_and_channel_and_explains_depth(tmp_path):
    rubric = load_rubric(RUBRIC)
    addressed = _raw_files(tmp_path)
    addressed[2] = EvidenceImage(
        path=addressed[2].path, view="view_0", channel="scene_depth",
        description="fixed 0..26000 cm; white near, black far")
    request = JudgeRequest(
        prompt="a palace", rubric=rubric,
        images=[item.path for item in addressed], metrics={},
        evidence=tuple(addressed))

    parts = openai_compat.content_parts(request)
    labels = [part["text"] for part in parts if part["type"] == "text"]

    assert labels[0].startswith("BEGIN CHANNEL BLOCK")
    assert any("CHANNEL=SCENE_DEPTH / VIEW=view_0" in label and "26000 cm" in label
               for label in labels)
    assert "Do not treat the channels as extra viewpoints" in labels[-1]
    assert "material_fidelity [use: base_color]" in labels[-1]


def test_channel_major_layout_sends_non_interleaved_labelled_blocks(tmp_path):
    rubric = load_rubric(RUBRIC)
    request = JudgeRequest(
        prompt="a palace",
        rubric=rubric,
        images=[],
        metrics={},
        evidence=tuple(_raw_files(tmp_path)),
        evidence_layout="channel_major_blocks",
    )

    parts = openai_compat.content_parts(request)
    labels = [part["text"] for part in parts if part["type"] == "text"]
    image_labels = [label for label in labels if label.startswith("CHANNEL=")]
    assert image_labels == [
        "CHANNEL=RGB / VIEW=view_0: view_0_rgb.png",
        "CHANNEL=RGB / VIEW=view_1: view_1_rgb.png",
        "CHANNEL=BASE_COLOR / VIEW=view_0: view_0_base_color.png",
        "CHANNEL=BASE_COLOR / VIEW=view_1: view_1_base_color.png",
        "CHANNEL=SCENE_DEPTH / VIEW=view_0: view_0_scene_depth.png",
        "CHANNEL=SCENE_DEPTH / VIEW=view_1: view_1_scene_depth.png",
    ]
    headers = [label for label in labels if label.startswith("BEGIN CHANNEL BLOCK")]
    assert ["CHANNEL=RGB.", "CHANNEL=BASE_COLOR.", "CHANNEL=SCENE_DEPTH."] == [
        next(token for token in header.split() if token.startswith("CHANNEL="))
        for header in headers
    ]
    assert "non-interleaved channel blocks" in request.instructions()


def _render_set(tmp_path, *, omit=()):
    result = render.RenderSet()
    for view in ("view_0", "view_1"):
        for channel, suffix in (("rgb", ".png"), ("base_color", ".png"),
                                ("scene_depth", ".exr")):
            if (view, channel) in omit:
                continue
            path = tmp_path / channel / f"{view}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"render")
            result.add("candidate", view, channel, str(path))
    return result


def test_collect_converts_depth_but_keeps_raw_exr_as_source(tmp_path, monkeypatch):
    def fake_visualize(source, target, **bounds):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"png")
        return {"source_exr": str(source), "visualization_png": str(target),
                **bounds}

    monkeypatch.setattr(evidence, "visualize_depth", fake_visualize)
    bundle = evidence.collect(
        _render_set(tmp_path), load_rubric(RUBRIC), tmp_path / "judge",
        near_cm=0, far_cm=26000)

    assert len(bundle.images) == 6
    depths = [item for item in bundle.images if item.channel == "scene_depth"]
    assert all(item.path.suffix == ".png" for item in depths)
    assert all(Path(record["source_exr"]).suffix == ".exr"
               for record in bundle.depth_visualization["views"].values())
    assert bundle.depth_visualization["far_cm"] == 26000


def test_transparent_base_color_is_sent_as_opaque_rgb_without_compositing(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    source = tmp_path / "raw.png"
    target = tmp_path / "judge" / "base_color.png"
    Image.new("RGBA", (2, 2), (91, 47, 13, 0)).save(source)

    record = evidence.normalize_base_color(
        source, target, policy="rgb8-opaque-png",
    )

    with Image.open(target) as normalized:
        assert normalized.mode == "RGB"
        assert normalized.getpixel((0, 0)) == (91, 47, 13)
    assert record["source_alpha_extrema"] == [0, 0]
    assert record["output_mode"] == "RGB"


def test_missing_channel_is_an_evidence_error_not_a_partial_score(tmp_path):
    with pytest.raises(JudgeError, match="only 1 complete"):
        evidence.collect(
            _render_set(tmp_path, omit=(("view_1", "scene_depth"),)),
            load_rubric(RUBRIC), tmp_path / "judge",
            near_cm=0, far_cm=26000)


def test_image_to_scene_task_uses_current_render_settings_without_selector(tmp_path):
    from synthetic_tasks import image_to_scene

    task = task_mod.load(image_to_scene(tmp_path, "outdoor"))
    requests = verifiers.render_evidence_requests(task)
    request = next(
        value
        for value in requests
        if value.kind == "paired_render" and value.channels == render.CHANNELS
    )
    assert request.kind == "paired_render"
    assert request.channels == render.CHANNELS
    assert request.view_count == 4
    assert request.width == 1280 and request.height == 720
    assert request.lighting_policy == render.PAIRED_PHOTOMETRY_LIGHTING_POLICY


@pytest.mark.parametrize("bad,message", [
    ({"channels": ["rgb", "normal"]}, "unknown render channel"),
    ({"score_ceilings": [{"criterion": "missing", "at_or_below": 1}]},
     "unknown criterion"),
])
def test_bad_multichannel_rubric_policy_is_refused(tmp_path, bad, message):
    body = yaml.safe_load(RUBRIC.read_text())
    body.update(bad)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(body))

    with pytest.raises(Exception, match=message):
        load_rubric(path)
