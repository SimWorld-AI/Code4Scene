"""The command line reports bad input in one line, and bundles are read defensively."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from code4scene import bundle as bundle_mod
from code4scene import cli, scoring
from code4scene.evaluation import vlm_model_config
from code4scene.protocol import constants, t2s
from test_bundle_cli import _decision, _i2s_bundle, _integrity, _t2s_bundle
from test_offline_overview_prompt_alignment import FakeClient
from test_protocol import _actor, _physics_report


def _error(capsys) -> str:
    err = capsys.readouterr().err
    assert "Traceback" not in err
    return err


def _t2s_task(tmp_path):
    from synthetic_tasks import text_to_scene

    root = tmp_path / "task"
    root.mkdir(exist_ok=True)
    return text_to_scene(root, task_id="synthetic-t2s")


def _scores(tmp_path, capsys) -> Path:
    scores = tmp_path / "scores"
    scores.mkdir()
    bundle = _i2s_bundle(tmp_path / "b")
    assert cli.main(["score", str(bundle.root), "--no-vlm", "-o", str(scores / "a.json")]) == 0
    capsys.readouterr()
    return scores


def _complete_t2s_bundle(root: Path):
    """A text-to-scene bundle a live judge scores completely: four views, structured rows."""

    writer = bundle_mod.BundleWriter(root)
    crate = _actor("crate", 0.0, asset="crate")
    crate["transform"]["location_cm"][2] = crate["bounds"]["origin_cm"][2] = 20.0
    views = []
    for i in range(4):
        image = root.parent / "frames" / f"view_{i + 1}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (16, 24, 3), dtype=np.uint8)).save(image)
        views.append(writer.add_file(f"renders/overview/view_{i + 1}.png", image))
    return writer.finish({
        "task": {"id": "synthetic-t2s", "setting": constants.SETTING_T2S},
        "run": {"model": "model-a"},
        "candidate_integrity": _integrity(),
        "scenes": {"candidate": writer.add_json("scenes/candidate.scene.json", {"actors": [crate]})},
        "physics": {"report": writer.add_json("physics/physical_safety.json", _physics_report())},
        "semantic": {"decisions": writer.add_json("semantic/decisions.json", {
            "report_status": "measured",
            "decisions": [_decision("content_quantity", 0, 1.0, "stage1")]})},
        "overview": {"views": views},
    })


# -- command line ---------------------------------------------------------------

def test_a_malformed_task_file_is_a_one_line_error(tmp_path, capsys):
    bundle = _i2s_bundle(tmp_path / "b")
    task = tmp_path / "task.yaml"
    task.write_text("- not\n- a task\n")
    assert cli.main(["score", str(bundle.root), "--no-vlm", "--task", str(task)]) == 2
    assert "code4scene score: error:" in _error(capsys)


def test_live_scoring_without_a_judge_endpoint_stops_before_the_first_call(tmp_path, capsys):
    bundle = _t2s_bundle(tmp_path / "b", views=4)
    assert cli.main(["score", str(bundle.root), "--task", str(_t2s_task(tmp_path))]) == 2
    assert vlm_model_config.BASE_URL_ENV in _error(capsys)


def test_aggregate_refuses_a_schedule_folder_without_all_three_lists(tmp_path, capsys):
    scores = _scores(tmp_path, capsys)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "public-outdoor-cases.txt").write_text("synthetic-repair\n")
    for folder in (partial, tmp_path / "no-such-folder"):
        assert cli.main(["aggregate", str(scores), "--schedule", str(folder)]) == 2
        assert "public-t2s-cases.txt" in _error(capsys)


def test_aggregate_refuses_rows_for_a_setting_without_a_case_list(tmp_path, capsys):
    scores = _scores(tmp_path, capsys)
    t2s_cases = tmp_path / "t2s.txt"
    t2s_cases.write_text("synthetic-t2s\n")
    assert cli.main(["aggregate", str(scores), "--t2s-cases", str(t2s_cases)]) == 2
    assert "no case list" in _error(capsys)


def test_aggregate_skips_what_is_not_a_case_score_in_a_folder(tmp_path, capsys):
    scores = _scores(tmp_path, capsys)
    (scores / "models.json").write_text(json.dumps([{"model": "model-a", "model_score": 0.5}]))
    (scores / "models.csv").write_text("model,model_score\nmodel-a,0.5\n")
    assert cli.main(["aggregate", str(scores)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)[0]["model"] == "model-a"
    assert captured.err.count("skipped") == 2 and "not the paper's schedule" in captured.err


def test_aggregate_without_case_scores_is_an_error(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["aggregate", str(empty)]) == 2
    assert "no case scores" in _error(capsys)
    table = tmp_path / "scores.csv"
    table.write_text("model,setting,score\nm,t2s,1.0\n")
    assert cli.main(["aggregate", str(table)]) == 2
    assert "case_id" in _error(capsys)


# -- the judge ------------------------------------------------------------------

def test_another_judge_is_not_comparable_to_the_paper(tmp_path, monkeypatch):
    from code4scene.tasks import task as task_mod

    task = task_mod.load(_t2s_task(tmp_path))
    bundle = _complete_t2s_bundle(tmp_path / "b")
    paper = scoring.score_bundle(bundle, task, judge="live", client_factory=lambda: FakeClient())
    assert paper["status"] == "measured" and paper["comparable_to_paper"]
    assert "differs_from_paper" not in paper["judge"]

    monkeypatch.setenv(vlm_model_config.MODEL_ENV, "some-other-vlm")
    other = scoring.score_bundle(bundle, task, judge="live", client_factory=lambda: FakeClient())
    assert other["score"] == paper["score"] and not other["comparable_to_paper"]
    assert other["judge"]["differs_from_paper"] == ["model"]
    # Image-to-scene case scores never call the judge.
    assert scoring.score_bundle(_i2s_bundle(tmp_path / "i2s"), judge="live")["comparable_to_paper"]


def test_the_judge_key_comes_only_from_its_own_variable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-this-endpoint")
    assert vlm_model_config.api_key() == ""
    monkeypatch.setenv(vlm_model_config.API_KEY_ENV, "judge-key")
    assert vlm_model_config.api_key() == "judge-key"


# -- bundles --------------------------------------------------------------------

def test_a_compressed_file_that_expands_too_far_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "scene.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"actors": [0] * 500}).encode()))
    assert bundle_mod.read_json(path) == {"actors": [0] * 500}
    monkeypatch.setattr(bundle_mod, "MAX_JSON_BYTES", 100)
    with pytest.raises(bundle_mod.BundleError, match="expands"):
        bundle_mod.read_json(path)


def test_deeply_nested_json_is_a_bundle_error(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text("[" * 200_000 + "]" * 200_000)
    with pytest.raises(bundle_mod.BundleError, match="nested"):
        bundle_mod.read_json(path)


def test_manifest_sections_must_be_objects(tmp_path):
    manifest = json.loads((_i2s_bundle(tmp_path / "b").root / "bundle.json").read_text())
    for key, value, message in (("run", ["model-a"], "run must be"),
                                ("renders", "view.png", "renders must be"),
                                ("overview", {"views": "view.png"}, "overview.views")):
        with pytest.raises(bundle_mod.BundleError, match=message):
            bundle_mod.validate_manifest({**manifest, key: value})


def test_a_physics_report_that_is_not_an_object_is_refused(tmp_path, capsys):
    bundle = _i2s_bundle(tmp_path / "b")
    manifest = json.loads((bundle.root / "bundle.json").read_text())
    report = bundle.root / manifest["physics"]["report"]
    report.write_text("[]")
    manifest["files"][manifest["physics"]["report"]] = bundle_mod.sha256_file(report)
    (bundle.root / "bundle.json").write_text(json.dumps(manifest))
    assert cli.main(["score", str(bundle.root), "--no-vlm"]) == 2
    assert "physics.report must be a JSON object" in _error(capsys)


def test_stage3_frames_are_covered_by_the_manifest(tmp_path):
    def build(root: Path, *, hashed: bool):
        writer = bundle_mod.BundleWriter(root)
        frame = root / "renders" / "stage3" / "frame.png"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"png")
        if hashed:
            writer.files["renders/stage3/frame.png"] = bundle_mod.sha256_file(frame)
        plan = writer.add_json("semantic/stage3_plan.json", {
            "frames": {"f1": {"path": "renders/stage3/frame.png"}}, "claims": []})
        decisions = writer.add_json("semantic/decisions.json", {"report_status": "measured",
                                                                "decisions": []})
        return writer.finish({"task": {"id": "synthetic-t2s", "setting": constants.SETTING_T2S},
                              "candidate_integrity": _integrity(),
                              "physics": {"report": writer.add_json(
                                  "physics/physical_safety.json", _physics_report())},
                              "semantic": {"decisions": decisions, "stage3_plan": plan}})

    assert build(tmp_path / "hashed", hashed=True).stage3_plan()["frames"]
    with pytest.raises(bundle_mod.BundleError, match="Stage 3 plan"):
        build(tmp_path / "unhashed", hashed=False)


# -- recorded decisions ---------------------------------------------------------

def _row(stored, exact=None):
    row = {"node_id": "n", "semantic_family": "content_quantity", "clause_index": 0,
           "included": True, "predicate_weight": 1.0, "effective_score": stored}
    if exact is not None:
        row["unrounded_effective_score"] = exact
    return row


def test_offline_detailed_uses_the_exact_row_score_only_when_it_matches():
    family = "content_quantity"
    exact = t2s.detailed_from_decisions([_row(0.3333, 1 / 3), _row(1.0, 1.0)])
    assert exact["families"][family]["unrounded_score"] == pytest.approx((1 / 3 + 1.0) / 2, abs=1e-15)
    stored = t2s.detailed_from_decisions([_row(0.3333), _row(1.0)])
    assert stored["families"][family]["unrounded_score"] == pytest.approx((0.3333 + 1.0) / 2, abs=1e-15)
    # A row whose stored score was changed after it was recorded keeps the stored value.
    stale = t2s.detailed_from_decisions([_row(0.0, 1 / 3), _row(1.0, 1.0)])
    assert stale["families"][family]["unrounded_score"] == pytest.approx(0.5, abs=1e-15)


def test_a_dropped_visual_row_does_not_keep_its_exact_score():
    row = {**_row(0.3333, 1 / 3), "resolved_by": "stage3_visual"}
    (dropped,) = scoring._drop_visual([row])
    assert dropped["effective_score"] == 0.0 and "unrounded_effective_score" not in dropped
