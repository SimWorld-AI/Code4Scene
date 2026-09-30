"""The saved-level scorer: the answer scene's footprint, the frozen baseline's missing packages, a failed export."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from code4scene.core import inventory

REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "benchmark/public/image-to-scene/indoor/indoor-atomic-v1-013/task.yaml"
DUNGEON_TASK = REPO / "benchmark/public/image-to-scene/indoor/indoor-atomic-v1-041/task.yaml"
GT = "/Game/Code4SceneGT/archviz-apartment/GT"
CANDIDATE = "/Game/SavedScenes/run_013"
WORKING = CANDIDATE + "__c4s_scoring"


def _tool():
    spec = importlib.util.spec_from_file_location("score_saved_level", REPO / "tools" / "score_saved_level.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Editor:
    """Answers the tool's editor calls and remembers what it was asked to do."""

    def __init__(self, gt_present=True):
        self.gt_present = gt_present
        self.opened, self.saved, self.passes = [], [], []

    def ping(self, timeout=90.0):
        return None

    def exec_python(self, script, timeout=120.0):
        return {}

    def exec_python_result(self, script, key, timeout=120.0):
        if key == "_SB_LOADED":
            level = script.split("_p = ", 1)[1].split("\n", 1)[0].strip("'")
            self.opened.append(level)
            return {"loaded": not level.startswith("/Game/Code4SceneGT/") or self.gt_present, "level": level}
        if key == inventory.INVENTORY_KEY:
            assert self.opened[-1].startswith("/Game/Code4SceneGT/")  # measured on the answer scene
            return {"actors": [{"label": "SM_Floor", "loc": [0, 0, 0], "extent": [1000, 800, 5]},
                               {"label": "SkySphere", "loc": [0, 0, 0], "extent": [90000, 90000, 90000]}]}
        if key == "_SB_BOUNDS":
            self.passes.append((self.opened[-1], script))
            return {"clamped": 0, "deleted": 1, "sample": ["SM_Stray"], "checked": 5}
        if key == "_SB_SAVED":
            self.saved.append(script.split("_p = ", 1)[1].split("\n", 1)[0].strip("'"))
            return {"saved": True}
        raise AssertionError(key)


def _score(monkeypatch, tmp_path, editor, task=TASK, export=None):
    tool = _tool()
    seen = {}
    monkeypatch.setattr(tool.Bridge, "unix", classmethod(lambda cls, path: editor))
    monkeypatch.setattr(tool.ue_evidence, "_run_editor_export",
                        lambda *a, **k: export or {"status": "success", "unresolved": []})
    monkeypatch.setattr(tool.verifiers, "run", lambda task, record, ids, **k: seen.setdefault("record", record) and [])
    monkeypatch.setattr(tool.primary_score, "apply_to_result", lambda result: result)
    assert tool.main(["--task", str(task), "--candidate-map", CANDIDATE, "--bridge", "sock",
                      "--out", str(tmp_path)]) == 0
    return seen["record"], json.loads((tmp_path / "result.json").read_text())


def test_the_pass_runs_on_the_candidate_and_only_a_working_copy_is_saved(monkeypatch, tmp_path):
    editor = _Editor()
    record, result = _score(monkeypatch, tmp_path, editor)
    assert editor.opened == [GT, CANDIDATE, WORKING]
    assert editor.saved == [WORKING]
    level, script = editor.passes[0]
    assert level == CANDIDATE
    assert "MINX=-1000.0; MAXX=1000.0" in script and "MINY=-800.0; MAXY=800.0" in script
    assert record["scene_map"] == record["official"]["level"] == WORKING
    assert result["candidate_map"] == CANDIDATE and result["scored_map"] == WORKING
    assert result["edge_discipline"]["total_deleted"] == 1
    assert result["evaluation_bounds"]["source_map"] == GT


def test_without_the_answer_scene_there_is_no_pass(monkeypatch, tmp_path):
    editor = _Editor(gt_present=False)
    record, result = _score(monkeypatch, tmp_path, editor)
    assert editor.opened == [GT, CANDIDATE]
    assert editor.saved == [] and editor.passes == []
    assert record["scene_map"] == CANDIDATE
    assert result["evaluation_bounds"] is None and result["edge_discipline"] is None


def test_a_candidate_the_editor_cannot_open_stops_the_run(monkeypatch, tmp_path):
    editor = _Editor()
    original = editor.exec_python_result

    def refuse_candidate(script, key, timeout=120.0):
        answer = original(script, key, timeout)
        return {**answer, "loaded": False} if key == "_SB_LOADED" and editor.opened[-1] == CANDIDATE else answer

    editor.exec_python_result = refuse_candidate
    with pytest.raises(SystemExit):
        _score(monkeypatch, tmp_path, editor)
    assert editor.saved == []


def test_the_scenes_missing_packages_are_allowed_for_its_cases_only(monkeypatch, tmp_path):
    record, _ = _score(monkeypatch, tmp_path, _Editor(gt_present=False), task=DUNGEON_TASK)
    allowance = record["dependency_integrity_allowance"]
    assert allowance["mode"] == "shared_baseline_missing_dependencies.v2"
    assert allowance["allowed_unresolved"] == ["/Game/Mannequin/Character/Mesh/SK_Mannequin_Female"]
    other, _ = _score(monkeypatch, tmp_path / "other", _Editor(gt_present=False))
    assert other["dependency_integrity_allowance"] is None


def test_a_failed_dependency_export_is_not_recorded_as_a_clean_manifest(monkeypatch, tmp_path):
    failed = {"status": "error", "error": "RuntimeError: asset registry unavailable", "unresolved": []}
    record, _ = _score(monkeypatch, tmp_path, _Editor(gt_present=False), export=failed)
    assert record["scene_dependencies"] is None

