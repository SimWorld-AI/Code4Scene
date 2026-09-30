"""The dataset builder reports failures instead of hiding them, and a dry run changes nothing."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # dataset_builder is run from the repository root, not installed
    sys.path.insert(0, str(REPO))

from dataset_builder import build, catalog  # noqa: E402
from dataset_builder import recipe as rc  # noqa: E402
from dataset_builder import verify as verify_mod  # noqa: E402

# Any text-to-scene case: the steps these tests run do not depend on which one.
T2S_CASE = (REPO / "benchmark" / "public-t2s-cases.txt").read_text(encoding="utf-8").split()[0]

FAKE_EDITOR = """#!{python}
import json, os, sys
calls = os.environ.get("FAKE_CALLS")
if calls:
    with open(calls, "a") as handle:
        handle.write(json.dumps(sys.argv[1:]) + "\\n")
job = json.load(open(os.environ["C4S_JOB"], encoding="utf-8"))
status = os.environ.get("FAKE_STATUS", "ok")
tasks = [{{"id": t["id"], "status": status, "error": None if status == "ok" else "boom"}} for t in job["tasks"]]
json.dump({{"tasks": tasks, "finished": True}}, open(job["result_path"], "w"))
"""
PLUGINS = [{"Name": "PythonScriptPlugin", "Enabled": True}, {"Name": "EditorScriptingUtilities", "Enabled": True}]


def _project(tmp_path, plugins=PLUGINS):
    project = tmp_path / "Proj" / "Proj.uproject"
    (project.parent / "Content").mkdir(parents=True)
    project.write_text(json.dumps({"FileVersion": 3, "Plugins": plugins}) + "\n")
    return project


def _editor(tmp_path, monkeypatch, status="ok"):
    editor = tmp_path / "UnrealEditor-Cmd"
    editor.write_text(FAKE_EDITOR.format(python=sys.executable))
    editor.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    monkeypatch.setenv("FAKE_STATUS", status)
    return editor, calls


def _run(project, editor, dataset, *extra):
    return build.main(["--project", str(project), "--editor", str(editor), "--dataset", str(dataset), *extra])


def test_commandlet_mode_keeps_the_extra_editor_arguments():
    args = Namespace(mode="commandlet", editor="UE", project=Path("/p/P.uproject"),
                     editor_arg=["-notrace", "-LocalDataCachePath=/tmp/ddc"])
    command = build.editor_command(args, Path("/d/jobs/gt.job.json"), render=False)
    assert "-run=pythonscript" in command and command[-2:] == ["-notrace", "-LocalDataCachePath=/tmp/ddc"]


def test_a_failed_task_makes_the_run_fail(tmp_path, monkeypatch):
    project = _project(tmp_path)
    editor, _ = _editor(tmp_path, monkeypatch, status="error")
    assert _run(project, editor, tmp_path / "ds", "--steps", "blank", "--cases", T2S_CASE) == 1
    editor, _ = _editor(tmp_path, monkeypatch, status="ok")
    assert _run(project, editor, tmp_path / "ds", "--steps", "blank", "--cases", T2S_CASE) == 0


def test_missing_packs_block_only_the_steps_that_use_them(tmp_path, monkeypatch):
    project = _project(tmp_path)  # no pack installed
    editor, calls = _editor(tmp_path, monkeypatch)
    assert _run(project, editor, tmp_path / "ds", "--steps", "blank", "--cases", T2S_CASE) == 0
    assert _run(project, editor, tmp_path / "ds", "--steps", "gt",
                "--cases", "a20-s01-extra-chair-removal") == 2
    assert len(calls.read_text().splitlines()) == 1


def test_the_editor_is_not_started_without_the_scripting_plugins(tmp_path, monkeypatch):
    project = _project(tmp_path, plugins=[])
    editor, calls = _editor(tmp_path, monkeypatch)
    assert _run(project, editor, tmp_path / "ds", "--steps", "blank", "--cases", T2S_CASE) == 2
    assert not calls.exists()
    assert _run(project, editor, tmp_path / "ds", "--steps", "init-project", "blank", "--cases", T2S_CASE) == 0
    assert calls.exists()


def test_a_dry_run_changes_nothing(tmp_path):
    project = _project(tmp_path, plugins=[])
    before = project.read_text()
    dataset = tmp_path / "ds"
    assert build.main(["--project", str(project), "--dataset", str(dataset), "--dry-run", "--cases", T2S_CASE,
                       "--steps", "init-project", "check", "blank", "verify", "package"]) == 0
    assert project.read_text() == before and not project.with_suffix(".uproject.bak").exists()
    assert not (project.parent / "Config").exists()
    assert not (dataset / "reports" / "verify.json").exists() and not (dataset / "manifest.json").exists()
    assert not (dataset / "agent").exists() and (dataset / "jobs" / "blank.job.json").is_file()


@pytest.mark.parametrize("extra", [["--cases", "a20-s01-extra-chair-removal", "a20-s01-extra-chair-remvoal"],
                                   ["--settings", "indoors"]], ids=["case", "setting"])
def test_an_unknown_case_or_setting_is_an_error(tmp_path, extra):
    project = _project(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        build.main(["--project", str(project), "--dataset", str(tmp_path / "ds"), "--steps", "package", *extra])
    assert stopped.value.code == 2


def test_a_missing_project_is_a_clean_error(tmp_path):
    with pytest.raises(SystemExit) as stopped:
        build.main(["--project", str(tmp_path / "nope.uproject"), "--steps", "init-project"])
    assert stopped.value.code == 2


def test_paths_the_editor_command_line_would_mangle():
    assert build.script_path_problem(Path("/src/Code4Scene/dataset_builder/ue/c4s_job.py"), render=True) is None
    assert build.script_path_problem(Path("/src/my.pyprojects/dataset_builder/ue/c4s_job.py"), render=False)
    assert build.script_path_problem(Path("/src/a,b/dataset_builder/ue/c4s_job.py"), render=True)
    assert build.script_path_problem(Path("/src/a,b/dataset_builder/ue/c4s_job.py"), render=False) is None


def _write_views(group):
    from PIL import Image

    for task in group:
        out = Path(task["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        for view in task["cameras"]["views"]:
            Image.new("RGB", (64, 36), (120, 120, 120)).save(out / f"{view['name']}.png")


def test_one_failed_render_does_not_stop_the_other_levels(tmp_path, monkeypatch):
    cases = catalog.load_cases(selected=["a20-s01-extra-chair-removal", "indoor-atomic-v1-041"])
    calls = []

    def fake_run_job(args, name, group, render=False):
        calls.append(name)
        if len(calls) == 1:
            raise SystemExit(f"{name}: the editor produced no result file")
        _write_views(group)

    monkeypatch.setattr(build, "run_job", fake_run_job)
    args = Namespace(dataset=tmp_path, dry_run=False, problems=[])
    build.step_render(args, cases, {})
    assert len(calls) == 2
    first = next(c for c in cases if "render-" + c.recipe["ground_truth_map"].rstrip("/").split("/")[-2] == calls[0])
    second = next(c for c in cases if c is not first)
    for view in second.cameras["views"]:
        assert (build.agent_dir(tmp_path, second) / view["publish"]["file"]).is_file()
    assert any("produced no result file" in p for p in args.problems)
    assert sum(first.case_id in p and "missing" in p for p in args.problems) == len(first.cameras["views"])


def test_render_needs_pillow_or_ffmpeg(tmp_path, monkeypatch):
    cases = catalog.load_cases(selected=["a20-s01-extra-chair-removal"])
    monkeypatch.setattr(build, "_have_pillow", lambda: False)
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    monkeypatch.setattr(build, "run_job", lambda *a, **k: pytest.fail("rendered without a way to publish"))
    with pytest.raises(SystemExit, match="Pillow"):
        build.step_render(Namespace(dataset=tmp_path, dry_run=False, problems=[]), cases, {})


def test_without_pillow_ffmpeg_resizes_png_views(tmp_path, monkeypatch):
    raw = tmp_path / "view.png"
    raw.write_bytes(b"\x89PNG\r\n\x1a\n" + b"data" + b"IEND\xaeB`\x82")
    commands = []
    monkeypatch.setattr(build, "_have_pillow", lambda: False)
    monkeypatch.setattr(build.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(build.subprocess, "run", lambda command, check: commands.append(command))
    how = build._publish(raw, {"width": 1280, "height": 720, "file": "references/view-01.png"},
                         tmp_path / "out" / "view-01.png")
    assert how == "ffmpeg" and "scale=1280:720" in commands[0] and "-q:v" not in commands[0]


def test_verify_reports_a_recipe_target_missing_from_the_local_gt(tmp_path, monkeypatch):
    cases = catalog.load_cases(selected=["a20-s01-extra-chair-removal", "indoor-atomic-v1-041"])
    for case in cases:
        for kind, key in (("gt", case.scene_id), ("input", case.case_id)):
            path = tmp_path / "snapshots" / kind / f"{key}.scene.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"actors": []}))
    broken = cases[0]

    def apply_offline(gt, recipe, sublevel_rename=None):
        if recipe is broken.recipe:
            raise rc.RecipeError("set_transform: target sca_x not in level")
        return {"actors": []}

    mismatch = {"match": False, "mismatched": [], "missing": ["sca_x"], "extra": [], "expected_root": "r",
                "expected_actor_count": 1, "local_actor_count": 0, "matched_after_float_probe": 0}
    monkeypatch.setattr(verify_mod.fp, "compare", lambda local, expected: dict(mismatch))
    monkeypatch.setattr(verify_mod.fp, "expand_delta", lambda base, delta: {})
    monkeypatch.setattr(verify_mod.rc, "apply_offline", apply_offline)
    report = verify_mod.verify(tmp_path, cases)
    assert report["cases"][broken.case_id]["recipe_consistency"] == {
        "status": "error", "reason": "RecipeError: set_transform: target sca_x not in level"}
    other = next(c for c in cases if c is not broken)
    assert report["cases"][other.case_id]["recipe_consistency"]["status"] == "mismatch"
    assert "recipe consistency: RecipeError" in verify_mod.render_text(report)


def test_verify_reads_snapshots_as_utf8(tmp_path):
    path = tmp_path / "gt.scene.json"
    path.write_bytes(json.dumps({"actors": [{"label": "Café"}]}, ensure_ascii=False).encode("utf-8"))
    assert verify_mod._load(path)["actors"][0]["label"] == "Café"
    assert os.path.exists(path)
