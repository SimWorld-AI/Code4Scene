"""The ways a verifier can report a clean scene it never measured.

Every case here was a live hole, found by running all verifiers over a good
and a wrecked scene and then trying to refute the claim that each number moved
for the reason its verifier names. They are collected in one file because they
are one failure mode wearing different clothes — "a clean bill of health for
work that never ran" — and a reader who fixes one should see the other twelve.

The pattern in every fix is the same: the metric refuses. Not a low score,
which is indistinguishable from a scene that measured badly; a refusal that
says what was missing.
"""

from __future__ import annotations

import copy
import json

from declared_cases import actor, case, context, scene

from code4scene.evaluation import contracts
from code4scene.evaluation.verifiers import (environment_consistency,
                                             ground_gap, gt_geometry,
                                             physics_regression,
                                             solid_penetration,
                                             spatial_clearance, spatial_overlap,
                                             spatial_relations,
                                             structure_concepts)

CANDIDATE_ALL = {"scope": "candidate_all"}
CHAIR = "/Game/Props/SM_Chair"


def blind(actors: list[dict]) -> dict:
    """A scene whose export failed to read bounds — the exporter's fallback.

    `export_scene_snapshot.py` writes origin = location and extent = 0 when
    the bounds read raises, and notes it in `export_errors` that no verifier
    reads. Every box test over this scene finds nothing.
    """
    out = copy.deepcopy(actors)
    for item in out:
        item["bounds"]["extent_cm"] = [0.0, 0.0, 0.0]
    return scene(out)


def colliding() -> list[dict]:
    return [actor("wall", location=(0.0, 0.0, 150.0), extent=(500.0, 10.0, 150.0)),
            actor("bench", location=(0.0, 0.0, 100.0), extent=(100.0, 60.0, 40.0))]


# ── geometry that was never read ─────────────────────────────────────────

def test_a_scene_of_dimensionless_points_is_refused_not_passed(tmp_path):
    spec = case([{"id": "props-apart", "primitive": "no_overlap",
                  "target_selector": CANDIDATE_ALL}])
    seen = spatial_overlap.verify(context(
        tmp_path, candidate=scene(colliding()), case_spec=spec))
    assert seen["status"] == contracts.MEASURED      # the score records the defect

    unseen = spatial_overlap.verify(context(
        tmp_path, candidate=blind(colliding()), case_spec=spec))
    assert unseen["status"] == "not_evaluated"
    assert "no usable bounds" in unseen["failure_reason"]


def test_a_crate_in_the_doorway_without_bounds_is_refused(tmp_path):
    doorway = {"id": "doorway", "min_xy_cm": [-100.0, -100.0],
               "max_xy_cm": [100.0, 100.0]}
    spec = case([{"id": "keep-clear", "primitive": "clearance",
                  "target_selector": CANDIDATE_ALL,
                  "protected_rectangles": [doorway]}])
    crate = [actor("crate", location=(0.0, 0.0, 40.0), extent=(60.0, 60.0, 40.0))]
    assert spatial_clearance.verify(context(
        tmp_path, candidate=scene(crate), case_spec=spec))["status"] == contracts.MEASURED
    refused = spatial_clearance.verify(context(
        tmp_path, candidate=blind(crate), case_spec=spec))
    assert refused["status"] == "not_evaluated"
    assert "no usable bounds" in refused["failure_reason"]


def test_a_route_that_is_not_a_line_is_refused_not_skipped(tmp_path):
    spec = case([{"id": "keep-clear", "primitive": "clearance",
                  "target_selector": CANDIDATE_ALL,
                  "routes": [{"id": "walkway", "width_m": 2.0,
                              "points": [[0.0, 0.0]]}]}])
    report = spatial_clearance.verify(context(
        tmp_path, candidate=scene([actor("crate")]), case_spec=spec))
    assert report["status"] == "not_evaluated"
    assert "fewer than two points" in report["failure_reason"]


def test_an_unusable_touch_tolerance_cannot_clear_the_scene(tmp_path):
    spec = case([{"id": "props-apart", "primitive": "no_overlap",
                  "target_selector": CANDIDATE_ALL,
                  "touch_tolerance_cm": float("inf")}])
    report = spatial_overlap.verify(context(
        tmp_path, candidate=scene(colliding()), case_spec=spec))
    assert report["status"] == "not_evaluated"
    assert "finite non-negative" in report["failure_reason"]


# ── a contract that says nothing ─────────────────────────────────────────

def test_a_concept_rule_with_no_count_is_refused_not_read_as_zero(tmp_path):
    """`int(None or 0)` turned "four chairs" into "exactly zero chairs"."""
    spec = case([{"id": "dining", "primitive": "structure",
                  "expected_raw_added_count": 1,
                  "concepts": [{"concept": "chair",
                                "allowed_asset_paths": [CHAIR]}]}])
    source = scene([actor("floor")])
    chairless = scene([actor("floor"), actor("statue")])
    report = structure_concepts.verify(context(
        tmp_path, candidate=chairless, input_scene=source, case_spec=spec))
    assert report["status"] == contracts.ERROR
    assert "must declare count" in report["failure_reason"]


def test_a_concept_rule_that_is_not_an_object_is_refused(tmp_path):
    """It used to be dropped — from the checks AND from the denominator."""
    spec = case([{"id": "dining", "primitive": "structure",
                  "expected_raw_added_count": 1,
                  "concepts": [{"concept": "chair", "count": 4,
                                "allowed_asset_paths": [CHAIR]},
                               "exactly one table"]}])
    report = structure_concepts.verify(context(
        tmp_path, candidate=scene([actor("floor")]),
        input_scene=scene([actor("floor")]), case_spec=spec))
    assert report["status"] == contracts.ERROR
    assert "must be an object" in report["failure_reason"]


# ── evidence that was never collected ────────────────────────────────────

def test_a_task_declaring_no_physics_assertion_does_not_pass_ground_gap(tmp_path):
    """It used to return status pass carrying score 0.0 — incoherent, and read
    as a clean bill of health for a check that never ran."""
    report = ground_gap.verify(context(
        tmp_path, candidate=scene([actor("crate")]),
        case_spec=case([{"id": "x", "primitive": "no_overlap"}])))
    assert report["status"] == contracts.ERROR
    assert "nobody stated" in report["failure_reason"]


def test_one_physics_record_cannot_clear_two_actors(tmp_path):
    """A duplicated label used to be credited by its twin's measurement."""
    twins = [actor("crate", location=(0.0, 0.0, 40.0), asset_path="/Game/C"),
             actor("crate", location=(500.0, 0.0, 40.0), asset_path="/Game/C")]
    twins[1]["actor_path"] = "/Game/Test.Test:PersistentLevel.crate_2"
    measurements = tmp_path / "m.json"
    measurements.write_text(json.dumps({"actors": {"crate": {
        "solid_penetration_cm": 0.0, "solid_penetrations": []}}}))
    spec = case([{"id": "no-clipping", "primitive": "solid_penetration",
                  "target_selector": CANDIDATE_ALL}])
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene(twins), case_spec=spec,
        spec={"candidate_measurements": str(measurements)}))
    assert report["status"] == "not_evaluated"
    assert "ambiguous_label" in json.dumps(report["metrics"])


def test_a_water_body_with_unusable_bounds_does_not_dry_the_lake(tmp_path):
    lake = actor("BP_WaterBody_Lake_01", location=(0.0, 0.0, 0.0),
                 extent=(1000.0, 1000.0, 5.0))
    lake["bounds"]["extent_cm"] = ["not", "a", "number"]
    drowned = actor("bench", location=(0.0, 0.0, -300.0), extent=(50.0, 25.0, 40.0))
    report = environment_consistency.verify(context(
        tmp_path, candidate=scene([lake, drowned]),
        case_spec=case([{"id": "dry", "primitive": "environment_consistency",
                         "target_selector": {"scope": "candidate_all",
                                             "labels": ["bench"]}}])))
    assert report["status"] == "not_evaluated"
    assert "named like water" in report["failure_reason"]


# ── numbers that were about the wrong thing ──────────────────────────────

def test_a_region_marker_is_not_a_collision_the_edit_introduced(tmp_path):
    """A dining zone is a box the furniture is asked to be placed inside."""
    zone = actor("dining_zone", location=(0.0, 0.0, 5.0),
                 extent=(400.0, 400.0, 5.0),
                 actor_class="/Script/Engine.TriggerVolume")
    source = scene([zone])
    complied = scene([zone, actor("table", location=(0.0, 0.0, 40.0),
                                  extent=(80.0, 80.0, 40.0))])
    report = physics_regression.verify(context(
        tmp_path, candidate=complied, input_scene=source,
        case_spec=case([{"id": "no-worse", "primitive": "physics_regression",
                         "scope": "candidate_all"}])))
    collision = next(item for item in report["metrics"]["checks"]
                     if item["id"] == "physics.regression.collision")
    assert collision["status"] == contracts.MEASURED
    assert collision["observed"]["new_collision_pairs"] == 0


def test_a_relation_about_objects_that_are_not_there_is_withheld(tmp_path):
    """It used to score as a failed check — a zero in the denominator that
    read as "the layout is wrong" rather than "the rule was not answered"."""
    spec = case([{"id": "layout", "primitive": "spatial_relation", "relations": [
        {"id": "bench-near-wall", "relation": "near", "maximum_distance_cm": 100.0,
         "subject": {"scope": "candidate_all", "allowed_categories": ["bench"]},
         "object": {"scope": "candidate_all", "allowed_categories": ["wall"]}}]}])
    report = spatial_relations.verify(context(
        tmp_path, candidate=scene([actor("floor")]), case_spec=spec))
    assert report["status"] == "not_evaluated"
    assert "did not resolve both sides" in report["failure_reason"]


def test_the_geometry_distance_does_not_depend_on_export_order():
    """A clamped match cost made most pairs tie, so the assignment — and the
    headline RMSE — fell out of the order the actors happened to arrive in."""
    canonical = [actor(f"a_{i}", location=(100.0 * i, 0.0, 0.0),
                       extent=(25.0, 25.0, 25.0)) for i in range(8)]
    wrecked = [actor(f"a_{i}", location=(4000.0 * i, 900.0, 0.0),
                     extent=(25.0, 25.0, 25.0)) for i in range(8)]
    baseline = gt_geometry.compare(wrecked, canonical)["position_rmse_cm"]
    for shift in range(1, 8):
        rotated = wrecked[shift:] + wrecked[:shift]
        assert gt_geometry.compare(rotated, canonical)["position_rmse_cm"] == baseline


def test_the_geometry_distance_reports_no_footprint_without_bounds():
    """`read_scene_graph` writes no bounds, so every Actor is a point and two
    zero footprints "match" perfectly. Absent, not 1.0."""
    plain = [{k: v for k, v in item.items() if k != "bounds"}
             for item in (actor("a", location=(0.0, 0.0, 0.0)),
                          actor("b", location=(50.0, 0.0, 0.0)))]
    result = gt_geometry.compare(plain, plain)
    assert result["bounds_measured"] is False
    assert result["mean_footprint_iou"] is None
    assert result["bounds_size_log_rmse"] is None
