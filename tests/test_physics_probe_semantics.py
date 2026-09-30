"""The in-editor physics probe classifies an actor by the actor's own names."""

from __future__ import annotations

import sys
import types

import pytest

from code4scene.resources import ue_script


class _Named:
    def __init__(self, path):
        self._path = path

    def get_name(self):
        return self._path.rsplit(".", 1)[-1].rsplit("/", 1)[-1]

    def get_path_name(self):
        return self._path


class _Component:
    def __init__(self, mesh):
        self._mesh = _Named(mesh)

    def get_editor_property(self, name):
        if name == "static_mesh":
            return self._mesh
        raise AttributeError(name)


class _Actor:
    def __init__(self, label, level="/Game/Maps/Test", mesh="/Game/Props/SM_Box",
                 cls="/Script/Engine.StaticMeshActor", tags=()):
        self._label, self._level, self._cls = label, level, _Named(cls)
        self.tags = list(tags)
        self._components = [_Component(mesh)]

    def get_actor_label(self):
        return self._label

    def get_name(self):
        return "StaticMeshActor_7"

    def get_path_name(self):
        leaf = self._level.rsplit("/", 1)[-1]
        return f"{self._level}.{leaf}:PersistentLevel.StaticMeshActor_7"

    def get_class(self):
        return self._cls

    def get_components_by_class(self, _cls):
        return list(self._components)

    def get_editor_property(self, name):
        if name == "tags":
            return list(self.tags)
        raise AttributeError(name)


@pytest.fixture(scope="module")
def probe():
    fake = types.ModuleType("unreal")
    fake.ActorComponent = object
    fake.PrimitiveComponent = object
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = fake
    try:
        source = ue_script("measure_scene_physics.py").read_text(encoding="utf-8").rstrip()
        assert source.endswith("main()")
        module = types.ModuleType("measure_scene_physics_probe")
        exec(compile(source[: -len("main()")], "measure_scene_physics.py", "exec"), module.__dict__)
        yield module
    finally:
        if saved is None:
            sys.modules.pop("unreal", None)
        else:
            sys.modules["unreal"] = saved


@pytest.mark.parametrize("actor", [
    _Actor("Crate", level="/Game/Maps/River_Town"),
    _Actor("Crate", level="/Game/Maps/Sky_Harbor"),
    _Actor("Crate", level="/Game/Maps/Pool_Hall"),
    _Actor("Crate", mesh="/Game/Lake_Pack/Meshes/SM_Crate.SM_Crate"),
    _Actor("Rock", cls="/Game/River_Assets/BP_Rock.BP_Rock_C"),
], ids=["river-level", "sky-level", "pool-level", "lake-folder-mesh", "river-folder-class"])
def test_level_and_folder_names_do_not_make_an_actor_non_solid(probe, actor):
    assert probe._is_non_solid(actor) is False


@pytest.mark.parametrize("actor", [
    _Actor("Water_Surface"),
    _Actor("Plane", mesh="/Game/Props/SM_Water_Plane.SM_Water_Plane"),
    _Actor("Sparks", tags=("vfx",)),
    _Actor("Smoke", cls="/Script/Niagara.NiagaraActor"),
    _Actor("Sun", cls="/Script/Engine.DirectionalLight"),
], ids=["water-label", "water-mesh", "vfx-tag", "niagara-class", "light-class"])
def test_non_solid_actors_are_still_recognised(probe, actor):
    assert probe._is_non_solid(actor) is True
