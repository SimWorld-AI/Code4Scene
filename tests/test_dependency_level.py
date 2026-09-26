"""The dependency manifest must describe the level being SHIPPED."""

from code4scene.evaluation import ue_evidence


class Bridge:
    """Captures the prelude the probe would send to the editor."""

    def __init__(self):
        self.script = ""

    def exec_python(self, script, timeout=120.0):
        self.script += script
        return {}

    def exec_python_result(self, script, key, timeout=300.0):
        self.script += script
        return {"status": "success"}


def prelude_for(level, tmp_path, monkeypatch):
    bridge = Bridge()
    monkeypatch.setattr(ue_evidence, "_run_editor_export",
                        lambda b, s, prelude, *a, **k: (
                            setattr(b, "script", prelude) or {"status": "success"}))
    ue_evidence.scene_dependencies(bridge, tmp_path, level=level)
    return bridge.script


def test_the_named_level_reaches_the_probe(tmp_path, monkeypatch):
    """Without a name the probe walks whatever world the editor is open on.
    That is not always the artifact: a task whose init_map is a real map
    leaves the editor there, and an agent calling `save_scene_as` moves it
    onto its own copy. One run's manifest described `/Game/Maps/empty` and
    reported a clean closure for a scene it never looked at — a false pass."""
    level = "/Game/SavedScenes/scenebench_paris__20260827__paris"
    assert level in prelude_for(level, tmp_path, monkeypatch)


def test_no_name_still_means_the_open_level(tmp_path, monkeypatch):
    """Scoring an already-saved artifact opens it first and has no separate
    name to pass; that path must keep working."""
    assert "SCENE_DEPENDENCIES_MAP = ''" in prelude_for("", tmp_path, monkeypatch)
