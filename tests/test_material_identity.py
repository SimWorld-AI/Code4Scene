"""Level-local dynamic material instances keep their identity across a save."""

from __future__ import annotations

from code4scene.protocol import actor_f1


def _light(level: str, mid: str, dynamic: bool = True) -> dict:
    path = f"{level}:PersistentLevel.BP_LightStudio_143.{mid}"
    return {
        "stable_actor_id": "light",
        "label": "BP_LightStudio",
        "class": "/Game/Lab/BP_LightStudio.BP_LightStudio_C",
        "asset_path": "/Game/Lab/BP_LightStudio.BP_LightStudio_C",
        "transform": {"location_cm": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        "material_paths": [path],
        "component_material_slots": [{
            "component_identity": "Skybox|/Script/Engine.StaticMeshComponent", "slot_index": 0,
            "material_path": path, "is_dynamic": dynamic,
            "material_class_path": "/Script/Engine.MaterialInstanceDynamic",
        }],
    }


INPUT = "/Game/Inputs/case/Input.Input"
SAVED = "/Game/Saved/run.run"


def test_a_renumbered_dynamic_instance_is_not_an_edit():
    before = _light(INPUT, "MID_MI_LightStage_Skybox_HDRI_0")
    after = _light(SAVED, "MID_MI_LightStage_Skybox_HDRI_1")
    assert actor_f1.changes(before, after) == ()


def test_a_different_dynamic_instance_is_still_an_edit():
    before = _light(INPUT, "MID_MI_LightStage_Skybox_HDRI_0")
    after = _light(SAVED, "MID_MI_Neon_Red_0")
    assert "property.material_paths" in actor_f1.changes(before, after)


def test_a_renumbered_static_material_is_still_an_edit():
    before = _light(INPUT, "MI_Panel_0", dynamic=False)
    after = _light(SAVED, "MI_Panel_1", dynamic=False)
    assert actor_f1.changes(before, after)
