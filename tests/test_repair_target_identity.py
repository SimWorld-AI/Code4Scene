"""Repair targets come from what changed in the scene, not from actor identifiers."""

from __future__ import annotations

import copy

from code4scene.evaluation.requirement_graph.repair_target_authoring import derive_repair_targets
from code4scene.protocol import actor_f1


def _actor(stable_id, label, asset, location=(0.0, 0.0, 100.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
    return {
        "stable_actor_id": stable_id,
        "actor_path": f"/Game/Test.Test:PersistentLevel.{label}",
        "label": label,
        "name": f"{label}_1",
        "class": "/Script/Engine.StaticMeshActor",
        "asset_path": f"/Game/Test/{asset}.{asset}",
        "actor_origin": "source",
        "actor_tags": [],
        "component_asset_paths": [f"/Game/Test/{asset}.{asset}"],
        "component_material_slots": [],
        "material_paths": [],
        "properties": {},
        "transform": {"location_cm": list(location), "rotation_deg": list(rotation), "scale": list(scale)},
        "bounds": {"origin_cm": list(location), "extent_cm": [50.0, 20.0, 80.0]},
    }


def _scene(*actors):
    return {"schema_version": "0.3.0", "map_path": "/Game/Test", "units": "cm",
            "actors": [copy.deepcopy(a) for a in actors], "actor_count": len(actors),
            "export_metadata": {"status": "success"}}


def _ops(targets):
    return sorted((t.operation, (t.desired_actor or t.input_actor)["label"]) for t in targets)


def test_an_actor_that_only_changes_identifier_is_not_a_target():
    wall = _actor("wall-gt-scope", "Wall", "SM_Wall", location=(500.0, 0.0, 0.0))
    wall_input = dict(wall, stable_actor_id="wall-input-scope")
    bench_gt = _actor("bench", "Bench", "SM_Bench")
    bench_moved = _actor("bench", "Bench", "SM_Bench", location=(300.0, 0.0, 100.0))
    targets = derive_repair_targets(_scene(wall_input, bench_moved), _scene(wall, bench_gt))
    assert _ops(targets) == [("repair", "Bench")]


def test_identifier_pairing_never_hides_a_real_edit():
    wall = _actor("wall-gt-scope", "Wall", "SM_Wall", location=(500.0, 0.0, 0.0))
    moved = _actor("wall-input-scope", "Wall", "SM_Wall", location=(520.0, 0.0, 0.0))
    swapped = _actor("wall-input-scope", "Wall", "SM_Wall_B", location=(500.0, 0.0, 0.0))
    for wrong in (moved, swapped):
        assert _ops(derive_repair_targets(_scene(wrong), _scene(wall))) == [("add", "Wall"), ("remove", "Wall")]


def test_a_real_removal_and_addition_remain_targets():
    extra = _actor("extra", "Crate_Extra", "SM_Crate", location=(0.0, 400.0, 0.0))
    lost = _actor("lost", "Lamp", "SM_Lamp", location=(0.0, -400.0, 0.0))
    keep = _actor("keep", "Table", "SM_Table")
    targets = derive_repair_targets(_scene(keep, extra), _scene(keep, lost))
    assert _ops(targets) == [("add", "Lamp"), ("remove", "Crate_Extra")]


def test_an_exact_repair_scores_full_marks_despite_identifier_changes():
    wall = _actor("wall-gt-scope", "Wall", "SM_Wall", location=(500.0, 0.0, 0.0))
    bench_gt = _actor("bench", "Bench", "SM_Bench")
    inp = _scene(dict(wall, stable_actor_id="wall-input-scope"),
                 _actor("bench", "Bench", "SM_Bench", location=(300.0, 0.0, 100.0)))
    gt = _scene(wall, bench_gt)
    counts, _ = actor_f1.measure(inp, gt, gt)
    assert (counts["true_positive"], counts["false_positive"], counts["false_negative"]) == (1, 0, 0)


def _pair_case():
    """The original moved, turned and rescaled; an injected copy keeps the correct turn and scale."""
    original_gt = _actor("bench", "Bench", "SM_Bench", location=(0.0, 0.0, 0.0))
    original_moved = _actor("bench", "Bench", "SM_Bench", location=(-400.0, 150.0, 0.0),
                            rotation=(0.0, 40.0, 0.0), scale=(1.3, 1.3, 1.3))
    copy_ = _actor("bench-copy", "Bench_Copy", "SM_Bench", location=(260.0, 40.0, 0.0))
    other = _actor("table", "Table", "SM_Table", location=(900.0, 0.0, 0.0))
    return _scene(original_moved, copy_, other), _scene(original_gt, other), original_gt, copy_, other


def _counts(inp, gt, cand):
    c, _ = actor_f1.measure(inp, gt, cand)
    return c["true_positive"], c["false_positive"], c["false_negative"]


def test_an_exact_repair_with_a_nearby_copy_scores_full_marks():
    inp, gt, *_ = _pair_case()
    assert _counts(inp, gt, gt) == (2, 0, 0)


def test_leaving_the_copy_in_place_is_still_a_missed_removal():
    inp, gt, original_gt, copy_, other = _pair_case()
    assert _counts(inp, gt, _scene(original_gt, copy_, other)) == (1, 0, 1)


def test_deleting_the_original_and_keeping_the_copy_is_not_a_repair():
    inp, gt, _, copy_, other = _pair_case()
    tp, _, fn = _counts(inp, gt, _scene(copy_, other))
    assert tp == 0 and fn == 2
