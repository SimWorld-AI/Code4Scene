"""The rendering bridge and the saved text-to-scene scorer."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from code4scene.core import inventory
from code4scene.core import scene as core_scene
from code4scene.evaluation import case_outcome

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "tools" / "c4s_render_bridge.py"
TASK = REPO / "benchmark/public/text-to-scene/case_014/task.yaml"
CANDIDATE = "/Game/SavedScenes/run_bazaar"
WORKING = CANDIDATE + "__c4s_scoring"


# -- the bridge, run against a stand-in `unreal` --------------------------------

class _Vector:
    def __init__(self, *xyz):
        self.xyz = xyz


class _Component:
    def __init__(self):
        self.properties = {}

    def set_editor_property(self, name, value):
        self.properties[name] = value

    def get_editor_property(self, name):
        return self.properties.setdefault(name, types.SimpleNamespace(set_editor_property=lambda *a: None))


class _Camera:
    def __init__(self):
        self.label, self.component, self.poses = "", _Component(), []

    def set_actor_label(self, label):
        self.label = label

    def get_actor_label(self):
        return self.label

    def get_component_by_class(self, _cls):
        return self.component

    def set_actor_location(self, location, *_):
        self.poses.append(location.xyz)

    def set_actor_rotation(self, *_):
        pass


def _fake_unreal(shots_taken):
    fake = types.ModuleType("unreal")
    actors = []

    class EditorActorSubsystem:
        def get_all_level_actors(self):
            return list(actors)

        def spawn_actor_from_class(self, *_):
            actors.append(_Camera())
            return actors[-1]

        def destroy_actor(self, actor):
            actors.remove(actor)

    def take_high_res_screenshot(width, height, filepath, camera=None):
        shots_taken.append(filepath)
        Path(filepath).write_bytes(b"png")  # lands at once; the real editor writes it frames later

    fake.EditorActorSubsystem = EditorActorSubsystem
    fake.get_editor_subsystem = lambda cls: cls()
    fake.CameraActor = fake.CameraComponent = object
    fake.Vector = fake.Rotator = _Vector
    fake.AutoExposureMethod = types.SimpleNamespace(AEM_MANUAL="manual")
    fake.AutomationLibrary = types.SimpleNamespace(take_high_res_screenshot=take_high_res_screenshot,
                                                   finish_loading_before_screenshot=lambda: None)
    fake.SystemLibrary = types.SimpleNamespace(quit_editor=lambda: None)
    fake.log = fake.log_error = fake.log_warning = lambda *_: None
    fake.register_slate_post_tick_callback = lambda fn: "handle"
    fake.unregister_slate_post_tick_callback = lambda handle: None
    return fake


@pytest.fixture
def bridge(monkeypatch, tmp_path):
    shots_taken = []
    monkeypatch.setitem(sys.modules, "unreal", _fake_unreal(shots_taken))
    monkeypatch.setenv("C4S_BRIDGE_SOCK", str(tmp_path / "bridge.sock"))
    source = BRIDGE.read_text(encoding="utf-8")
    body = source[: source.index("\nif START_MAP:")]  # everything but the serving tail
    module = types.ModuleType("c4s_render_bridge")
    exec(compile(body, str(BRIDGE), "exec"), module.__dict__)
    module.shots_taken = shots_taken
    return module


class _Conn:
    def __init__(self):
        self.sent = b""

    def sendall(self, data):
        self.sent += data

    def close(self):
        pass

    def reply(self):
        return json.loads(self.sent.decode())


def _pose(tmp_path, name, **extra):
    return {"filepath": str(tmp_path / f"{name}.png"), "width": 1280, "height": 720,
            "camera_location": [1.0, 2.0, 3.0], "camera_rotation": [0.0, -30.0, 45.0],
            "field_of_view": 55.0, **extra}


def test_level_changes_are_found_in_top_level_code_only(bridge):
    exporter = (REPO / "code4scene" / "ue_scripts" / "export_scene_snapshot.py").read_text()
    assert bridge._changes_level(core_scene.load_script("/Game/Maps/Other"))
    assert bridge._changes_level(core_scene.new_canvas_script("/Game/Maps/Other"))
    assert not bridge._changes_level("SCENE_DISTANCE_MAP = ''\n" + exporter)
    assert not bridge._changes_level(inventory.script())


def test_shot_requests_are_validated(bridge, tmp_path):
    shots, error = bridge._shots({"shots": [_pose(tmp_path, "a"), _pose(tmp_path, "b")]}, batch=True)
    assert error is None and [s["exposure_mode"] for s in shots] == [None, None]
    _, error = bridge._shots({"shots": []}, batch=True)
    assert "between 1 and 16" in error
    _, error = bridge._shots({"shots": [_pose(tmp_path, "a", exposure_bias_ev=1.0), _pose(tmp_path, "b")]},
                             batch=True)
    assert "one exposure_bias_ev" in error
    _, error = bridge._shots({"filepath": str(tmp_path / "c.png")}, batch=False)
    assert "camera_location" in error
    shots, _ = bridge._shots(_pose(tmp_path, "d", exposure_bias_ev=0.5), batch=False)
    assert shots[0]["exposure_mode"] == bridge.POST_PROCESS_BIAS


def test_a_batch_is_answered_when_every_picture_landed_and_the_first_is_warmed_up(bridge, tmp_path):
    conn = _Conn()
    poses = [_pose(tmp_path, f"view_{i}") for i in range(3)]
    bridge._screenshot(conn, "take_screenshot_batch", {"shots": poses})
    while bridge.STATE["job"] is not None:
        bridge._advance_job()
    reply = conn.reply()
    assert reply["status"] == "success" and reply["missing"] == []
    assert reply["camera_policy"] == "single-camera-slate-tick-batch"
    warm = str(tmp_path / "view_0.warmup.png")
    assert bridge.shots_taken == [warm] + [p["filepath"] for p in poses]
    assert not Path(warm).exists()
    # Later requests are not warmed up again.
    second = _Conn()
    bridge._screenshot(second, "take_screenshot", _pose(tmp_path, "close_up", exposure_bias_ev=1.5))
    while bridge.STATE["job"] is not None:
        bridge._advance_job()
    reply = second.reply()
    assert reply["status"] == "success" and reply["camera_exposure_bias_ev"] == 1.5
    assert bridge.shots_taken[-1] == str(tmp_path / "close_up.png") and len(bridge.shots_taken) == 5


def test_a_level_change_is_refused_once_rendering_began(bridge):
    conn = _Conn()
    request = {"type": "execute_python_script", "params": {"script": core_scene.load_script("/Game/Maps/Other")}}
    chunks = iter([json.dumps(request).encode() + b"\n", b""])
    conn.recv = lambda _size: next(chunks)
    conn.settimeout = lambda _t: None
    assert bridge._serve_tick(conn) is True
    assert conn.reply()["status"] == "error"


# -- the saved text-to-scene scorer ---------------------------------------------

def _tool():
    spec = importlib.util.spec_from_file_location("score_saved_build", REPO / "tools" / "score_saved_build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Editor:
    def __init__(self, renderer=None):
        self.calls = []
        self.renderer = renderer

    def ping(self, timeout=90.0):
        return None

    def exec_python(self, script, timeout=120.0):
        return {}

    def exec_python_result(self, script, key, timeout=120.0):
        if key == "_SB_LOADED":
            level = script.split("_p = ", 1)[1].split("\n", 1)[0].strip("'")
            self.calls.append(("open", level))
            return {"loaded": True, "level": level}
        if key == "_SB_BOUNDS":
            self.calls.append(("bounds", script))
            return {"clamped": 1, "deleted": 2, "sample": [], "checked": 9}
        if key == "_SB_SAVED":
            self.calls.append(("save", script.split("_p = ", 1)[1].split("\n", 1)[0].strip("'")))
            return {"saved": True}
        if key == "_C4S_RENDERER":
            names = ast.literal_eval(script.split("for name in ", 1)[1].split("}", 1)[0])
            expected = {"True": 1.0, "False": 0.0}
            settings = dict(line.split("=", 1) for line in (REPO / "benchmark" / "renderer_settings.txt").read_text().splitlines()
                            if line and not line.startswith("#"))
            values = {name: expected.get(settings[name], None) if settings[name] in expected else float(settings[name])
                      for name in names}
            return {**values, **(self.renderer or {})}
        raise AssertionError(key)

    def command(self, kind, params, timeout=60.0):
        self.calls.append((kind,))
        return {"status": "ok"}


class _Renders:
    def as_dict(self):
        return {"views": ["view_0"]}


def _score(monkeypatch, tmp_path, reports, classification="resolved", render_errors=None, renderer=None,
           export=None, seen=None):
    tool = _tool()
    editor = _Editor(renderer)
    monkeypatch.setattr(tool.Bridge, "unix", classmethod(lambda cls, path: editor))
    monkeypatch.setattr(tool.ue_evidence, "_run_editor_export", lambda *a, **k: export or {"status": "success"})
    monkeypatch.setattr(tool.render_capture, "capture_render_evidence",
                        lambda task, bridge, out: (editor.calls.append(("capture",)) or
                                                   ({"overview_prompt_alignment.candidate_clearance_gallery": _Renders()},
                                                    render_errors or {})))
    monkeypatch.setattr(tool.verifiers, "run", lambda task, record, *a, **k: (
        editor.calls.append(("verify",)), (seen if seen is not None else {}).update(record=record))[0] or reports)
    monkeypatch.setattr(tool.score_policy, "apply_to_result",
                        lambda result, policy: {**result, "case_outcome": {"classification": classification},
                                                "overall_score": 0.5})
    code = tool.main(["--task", str(TASK), "--candidate-map", CANDIDATE, "--bridge", "sock", "--out", str(tmp_path)])
    return code, editor


MEASURED = [{"report_id": "semantic_requirements", "status": "measured"},
            {"report_id": "overview_prompt_alignment", "status": "measured"}]


def test_the_plate_is_enforced_on_a_working_copy_before_rendering(monkeypatch, tmp_path):
    code, editor = _score(monkeypatch, tmp_path, MEASURED)
    assert code == 0
    steps = [call[0] for call in editor.calls]
    assert steps == ["open", "bounds", "save", "open", "begin_rendering", "capture", "verify"]
    assert editor.calls[0] == ("open", CANDIDATE) and editor.calls[2] == ("save", WORKING)
    assert "MINX=-6500.0; MAXX=6500.0" in editor.calls[1][1]  # case_014: size_m 130
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["scored_map"] == WORKING and result["edge_discipline"]["total_deleted"] == 2


def test_an_unjudged_scene_is_not_published_as_scored(monkeypatch, tmp_path):
    reports = [MEASURED[0], {"report_id": "overview_prompt_alignment", "status": "error",
                             "failure_reason": "JudgeEndpointNotConfigured"}]
    with pytest.raises(SystemExit, match="not scored"):
        _score(monkeypatch, tmp_path, reports)
    assert not (tmp_path / "result.json").exists() and (tmp_path / "result.unscored.json").exists()


def test_an_invalid_candidate_still_scores_zero(monkeypatch, tmp_path):
    reports = [{"report_id": "semantic_requirements", "status": "not_evaluated"},
               {"report_id": "overview_prompt_alignment", "status": "not_evaluated"}]
    code, _ = _score(monkeypatch, tmp_path, reports, classification=case_outcome.MODEL_INVALID)
    assert code == 0 and (tmp_path / "result.json").exists()


def test_missing_pictures_stop_the_run(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="could not photograph"):
        _score(monkeypatch, tmp_path, MEASURED, render_errors={"overview": "RenderError: black"})
    assert not (tmp_path / "result.json").exists()


def test_a_project_without_the_benchmark_renderer_settings_is_refused(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="renderer settings"):
        _score(monkeypatch, tmp_path, MEASURED, renderer={"r.DynamicGlobalIlluminationMethod": 0.0})
    assert not (tmp_path / "result.json").exists()


def test_a_failed_dependency_export_is_not_recorded_as_a_clean_manifest(monkeypatch, tmp_path):
    seen = {}
    failed = {"status": "error", "error": "RuntimeError: asset registry unavailable", "unresolved": []}
    _score(monkeypatch, tmp_path, MEASURED, export=failed, seen=seen)
    assert seen["record"]["scene_dependencies"] is None

