"""Absolute spatial constraints: overlap, clearance, cluster span, relations."""

from __future__ import annotations

import math

import pytest
from declared_cases import actor, case, check_by_id, context, scene

from code4scene.evaluation import contracts
from code4scene.evaluation import spatial_relations as relations
from code4scene.evaluation.verifiers import (spatial_clearance, spatial_cluster,
                                             spatial_overlap, spatial_relations)

CANDIDATE_ALL = {"scope": "candidate_all"}


def overlap_case(tolerance: float | None = None) -> dict:
    assertion = {"id": "props-apart", "primitive": "no_overlap",
                 "target_selector": CANDIDATE_ALL}
    if tolerance is not None:
        assertion["touch_tolerance_cm"] = tolerance
    return case([assertion])


def test_two_boxes_sharing_a_face_are_touching_not_colliding(tmp_path):
    """Two axes overlapping and one touching is a table against a wall."""
    left = actor("left", location=(0.0, 0.0, 50.0), extent=(50.0, 50.0, 50.0))
    right = actor("right", location=(100.0, 0.0, 50.0), extent=(50.0, 50.0, 50.0))
    report = spatial_overlap.verify(context(
        tmp_path, candidate=scene([left, right]), case_spec=overlap_case()))
    assert report["status"] == contracts.MEASURED


def test_a_box_driven_into_another_is_a_collision(tmp_path):
    left = actor("left", location=(0.0, 0.0, 50.0), extent=(50.0, 50.0, 50.0))
    inside = actor("inside", location=(20.0, 0.0, 50.0), extent=(50.0, 50.0, 50.0))
    report = spatial_overlap.verify(context(
        tmp_path, candidate=scene([left, inside]), case_spec=overlap_case()))
    assert report["status"] == contracts.MEASURED
    check = check_by_id(report, "spatial.no_overlap.props-apart")
    assert check["observed"]["overlapping_pairs"] == 1
    assert check["evidence"][0]["penetration_depth_cm"] == pytest.approx(75.0)


def test_the_touch_tolerance_is_the_same_number_the_scene_rate_uses(tmp_path):
    """A 6 cm interpenetration is over the shared 5 cm tolerance."""
    left = actor("left", location=(0.0, 0.0, 50.0), extent=(50.0, 50.0, 50.0))
    right = actor("right", location=(94.0, 0.0, 50.0), extent=(50.0, 50.0, 50.0))
    report = spatial_overlap.verify(context(
        tmp_path, candidate=scene([left, right]), case_spec=overlap_case()))
    assert report["status"] == contracts.MEASURED


def test_an_empty_population_withholds_rather_than_passing(tmp_path):
    """Every requirement over nothing holds; that is not a measurement."""
    report = spatial_overlap.verify(context(
        tmp_path, candidate=scene([]), case_spec=overlap_case()))
    assert report["status"] == "not_evaluated"
    assert "matched nothing" in report["failure_reason"]


def test_a_prop_in_the_doorway_fails_clearance(tmp_path):
    doorway = {"id": "doorway", "min_xy_cm": [-100.0, -100.0],
               "max_xy_cm": [100.0, 100.0]}
    blocking = actor("crate", location=(50.0, 0.0, 50.0), extent=(60.0, 60.0, 50.0))
    clear = actor("crate", location=(500.0, 0.0, 50.0), extent=(60.0, 60.0, 50.0))
    spec = case([{"id": "keep-clear", "primitive": "clearance",
                  "target_selector": CANDIDATE_ALL,
                  "protected_rectangles": [doorway]}])
    blocked = spatial_clearance.verify(context(
        tmp_path, candidate=scene([blocking]), case_spec=spec))
    assert blocked["status"] == contracts.MEASURED
    assert check_by_id(blocked, "spatial.clearance.keep-clear")["evidence"][0][
        "region"] == "doorway"

    assert spatial_clearance.verify(context(
        tmp_path, candidate=scene([clear]), case_spec=spec))["status"] == contracts.MEASURED


def test_a_route_corridor_is_measured_in_metres(tmp_path):
    """A 2 m-wide corridor along X; the prop's centre is 150 cm off it."""
    route = {"id": "walkway", "width_m": 2.0, "points": [[0.0, 0.0], [10.0, 0.0]]}
    spec = case([{"id": "keep-walkable", "primitive": "clearance",
                  "target_selector": CANDIDATE_ALL, "routes": [route]}])
    # 1.5 m away with a ~0.14 m radius clears the 1.0 m half-width.
    beside = actor("bench", location=(200.0, 150.0, 0.0), extent=(10.0, 10.0, 50.0))
    assert spatial_clearance.verify(context(
        tmp_path, candidate=scene([beside]), case_spec=spec))["status"] == contracts.MEASURED
    on_it = actor("bench", location=(200.0, 50.0, 0.0), extent=(10.0, 10.0, 50.0))
    assert spatial_clearance.verify(context(
        tmp_path, candidate=scene([on_it]), case_spec=spec))["status"] == contracts.MEASURED


def test_cluster_span_is_measured_between_centres(tmp_path):
    spec = case([{"id": "one-set", "primitive": "compact_cluster",
                  "target_selector": CANDIDATE_ALL,
                  "maximum_center_span_cm": [300.0, 300.0]}])
    tight = [actor("a", location=(0.0, 0.0, 0.0)),
             actor("b", location=(200.0, 100.0, 0.0))]
    assert spatial_cluster.verify(context(
        tmp_path, candidate=scene(tight), case_spec=spec))["status"] == contracts.MEASURED
    scattered = [actor("a", location=(0.0, 0.0, 0.0)),
                 actor("b", location=(900.0, 0.0, 0.0))]
    report = spatial_cluster.verify(context(
        tmp_path, candidate=scene(scattered), case_spec=spec))
    assert report["status"] == contracts.MEASURED
    assert check_by_id(report, "spatial.compact_cluster.one-set")["observed"][
        "center_span_cm"] == [900.0, 0.0]


def test_a_cluster_without_a_declared_span_withholds(tmp_path):
    spec = case([{"id": "one-set", "primitive": "compact_cluster",
                  "target_selector": CANDIDATE_ALL}])
    report = spatial_cluster.verify(context(
        tmp_path, candidate=scene([actor("a")]), case_spec=spec))
    assert report["status"] == "not_evaluated"
    assert "maximum_center_span_cm" in report["failure_reason"]


def relation_case(rule: dict) -> dict:
    return case([{"id": "relations", "primitive": "spatial_relation",
                  "relations": [rule]}])


def test_the_share_of_holding_relations_is_the_score(tmp_path):
    table = actor("table", location=(0.0, 0.0, 40.0), extent=(80.0, 80.0, 40.0),
                  label_hint="table")
    near_chair = actor("chair_1", location=(120.0, 0.0, 40.0), extent=(25.0, 25.0, 40.0))
    far_chair = actor("chair_2", location=(2000.0, 0.0, 40.0), extent=(25.0, 25.0, 40.0))
    spec = case([{"id": "relations", "primitive": "spatial_relation", "relations": [
        {"id": "chairs-near-table", "relation": "near", "maximum_distance_cm": 200.0,
         "subject": {"scope": "candidate_all", "labels": ["chair_1"]},
         "object": {"scope": "candidate_all", "labels": ["table"]}},
        {"id": "all-chairs-near-table", "relation": "near",
         "maximum_distance_cm": 200.0,
         "subject": {"scope": "candidate_all", "labels": ["chair_1", "chair_2"]},
         "object": {"scope": "candidate_all", "labels": ["table"]}},
    ]}])
    report = spatial_relations.verify(context(
        tmp_path, candidate=scene([table, near_chair, far_chair]), case_spec=spec))
    assert report["status"] == contracts.MEASURED
    assert report["score"] == 0.5


def test_on_top_of_needs_the_footprint_and_not_only_the_height(tmp_path):
    table = actor("table", location=(0.0, 0.0, 70.0), extent=(80.0, 80.0, 5.0))
    on_it = actor("vase", location=(0.0, 0.0, 85.0), extent=(10.0, 10.0, 10.0))
    beside_it = actor("vase", location=(400.0, 0.0, 85.0), extent=(10.0, 10.0, 10.0))
    rule = {"id": "vase-on-table", "relation": "on_top_of", "tolerance_cm": 1.0,
            "subject": {"scope": "candidate_all", "labels": ["vase"]},
            "object": {"scope": "candidate_all", "labels": ["table"]}}
    assert spatial_relations.verify(context(
        tmp_path, candidate=scene([table, on_it]),
        case_spec=relation_case(rule)))["status"] == contracts.MEASURED
    assert spatial_relations.verify(context(
        tmp_path, candidate=scene([table, beside_it]),
        case_spec=relation_case(rule)))["status"] == contracts.MEASURED


def test_on_top_of_can_use_a_declared_surface_inside_a_compound_anchor():
    cabinet = actor("cabinet", location=(0.0, 0.0, 150.0),
                    extent=(100.0, 100.0, 150.0))
    books = actor("books", location=(0.0, 0.0, 110.0),
                  extent=(10.0, 10.0, 10.0))
    base = {"relation": "on_top_of", "tolerance_cm": 2.0,
            "minimum_overlap_fraction": 0.8}
    surface = {"plane_z_cm": 100.0, "min_xy_cm": [-80.0, -80.0],
               "max_xy_cm": [80.0, 80.0]}

    without_override = relations.evaluate(base, [books], [cabinet])
    with_override = relations.evaluate(
        {**base, "object_surface": surface}, [books], [cabinet])

    assert without_override["pass"] is False
    assert with_override["pass"] is True
    assert with_override["rows"][0]["support_surface_source"] == (
        "declared_object_surface")
    assert with_override["rows"][0]["vertical_gap_cm"] == 0.0
    assert with_override["rows"][0]["xy_overlap_fraction"] == 1.0
    assert with_override["expected"]["object_surface"] == surface


def test_declared_on_top_surface_rejects_height_and_footprint_failures():
    cabinet = actor("cabinet", location=(0.0, 0.0, 150.0),
                    extent=(100.0, 100.0, 150.0))
    rule = {
        "relation": "on_top_of",
        "tolerance_cm": 2.0,
        "minimum_overlap_fraction": 0.8,
        "object_surface": {
            "plane_z_cm": 100.0,
            "min_xy_cm": [-80.0, -80.0],
            "max_xy_cm": [80.0, 80.0],
        },
    }
    raised = actor("raised", location=(0.0, 0.0, 120.0),
                   extent=(10.0, 10.0, 10.0))
    outside = actor("outside", location=(200.0, 0.0, 110.0),
                    extent=(10.0, 10.0, 10.0))

    raised_result = relations.evaluate(rule, [raised], [cabinet])
    outside_result = relations.evaluate(rule, [outside], [cabinet])

    assert raised_result["pass"] is False
    assert raised_result["rows"][0]["vertical_gap_cm"] == 10.0
    assert outside_result["pass"] is False
    assert outside_result["rows"][0]["xy_overlap_fraction"] == 0.0


@pytest.mark.parametrize("surface", [
    [],
    {"plane_z_cm": math.inf, "min_xy_cm": [0.0, 0.0],
     "max_xy_cm": [1.0, 1.0]},
    {"plane_z_cm": 0.0, "min_xy_cm": [0.0],
     "max_xy_cm": [1.0, 1.0]},
    {"plane_z_cm": 0.0, "min_xy_cm": [1.0, 0.0],
     "max_xy_cm": [1.0, 1.0]},
])
def test_declared_on_top_surface_requires_finite_ordered_geometry(surface):
    rule = {"relation": "on_top_of", "object_surface": surface}
    with pytest.raises(relations.RelationError, match="object_surface"):
        relations.evaluate(rule, [actor("subject")], [actor("object")])


def test_facing_defaults_to_the_actor_yaw():
    chair = actor("chair", location=(0.0, 0.0, 0.0),
                  rotation=(0.0, 45.0, 0.0))
    table = actor("table", location=(100.0, 100.0, 0.0))
    base = {"relation": "facing", "maximum_angle_deg": 1.0}

    implicit = relations.evaluate(base, [chair], [table])
    explicit = relations.evaluate(
        {**base, "subject_forward_yaw_offset_deg": 0.0}, [chair], [table])

    assert implicit["pass"] is True
    assert implicit["pass"] == explicit["pass"]
    assert implicit["expected"]["subject_forward_yaw_offset_deg"] == 0.0
    assert implicit["rows"][0]["observed_heading_yaw_deg"] == 45.0
    assert implicit["rows"][0]["yaw_error_deg"] == 0.0


def test_facing_applies_the_subject_assets_forward_axis_offset():
    chair = actor("chair", location=(0.0, 0.0, 0.0),
                  rotation=(0.0, -45.0, 0.0))
    table = actor("table", location=(100.0, 100.0, 0.0))
    result = relations.evaluate({
        "relation": "facing",
        "maximum_angle_deg": 1.0,
        "subject_forward_yaw_offset_deg": 90.0,
    }, [chair], [table])

    assert result["pass"] is True
    assert result["expected"]["subject_forward_yaw_offset_deg"] == 90.0
    evidence = result["rows"][0]
    assert evidence["subject_yaw_deg"] == -45.0
    assert evidence["subject_forward_yaw_offset_deg"] == 90.0
    assert evidence["observed_heading_yaw_deg"] == 45.0
    assert evidence["desired_yaw_deg"] == 45.0
    assert evidence["yaw_error_deg"] == 0.0


def test_angular_coverage_requires_subjects_around_the_anchor():
    anchor = actor("support", location=(0.0, 0.0, 0.0))
    ring = [
        actor(f"candle_{index}", location=(
            100.0 * math.cos(index * math.pi / 4),
            100.0 * math.sin(index * math.pi / 4),
            0.0,
        ))
        for index in range(8)
    ]
    rule = {
        "relation": "angular_coverage",
        "minimum_subject_count": 8,
        "angular_sector_count": 8,
        "minimum_angular_coverage": 0.75,
    }

    around = relations.evaluate(rule, ring, [anchor])
    one_side = relations.evaluate(
        rule,
        [actor(f"line_{index}", location=(100.0 + index, 0.0, 0.0))
         for index in range(8)],
        [anchor],
    )

    assert around["pass"] is True
    assert around["observed"]["coverage"] >= 0.75
    assert one_side["pass"] is False
    assert one_side["observed"]["coverage"] < 0.75


def test_straddles_requires_subjects_on_both_sides_of_the_anchor_population():
    anchors = [actor("left_anchor", location=(-10.0, 0.0, 0.0)),
               actor("right_anchor", location=(10.0, 0.0, 0.0))]
    rule = {"relation": "straddles", "axis": "y",
            "minimum_offset_cm": 50.0, "minimum_subject_count": 2}

    opposite = relations.evaluate(
        rule,
        [actor("a", location=(0.0, -80.0, 0.0)),
         actor("b", location=(0.0, 80.0, 0.0))],
        anchors,
    )
    same_side = relations.evaluate(
        rule,
        [actor("a", location=(0.0, 80.0, 0.0)),
         actor("b", location=(0.0, 120.0, 0.0))],
        anchors,
    )

    assert opposite["pass"] is True
    assert opposite["observed"]["negative_side_count"] == 1
    assert opposite["observed"]["positive_side_count"] == 1
    assert same_side["pass"] is False


def test_collinear_fits_the_subject_centres_and_rejects_a_bent_row():
    anchor = actor("anchor")
    rule = {"relation": "collinear", "minimum_subject_count": 3,
            "maximum_deviation_cm": 5.0}
    straight = [actor("left", location=(0.0, 0.0, 0.0)),
                actor("middle", location=(300.0, 0.0, 0.0)),
                actor("right", location=(600.0, 0.0, 0.0))]
    bent = [actor("left", location=(0.0, 0.0, 0.0)),
            actor("middle", location=(300.0, 60.0, 0.0)),
            actor("right", location=(600.0, 0.0, 0.0))]

    straight_result = relations.evaluate(rule, straight, [anchor])
    bent_result = relations.evaluate(rule, bent, [anchor])

    assert straight_result["pass"] is True
    assert straight_result["observed"][
        "maximum_perpendicular_deviation_cm"] == 0.0
    assert bent_result["pass"] is False
    assert bent_result["observed"][
        "maximum_perpendicular_deviation_cm"] > 5.0


def test_spacing_uniformity_uses_adjacent_principal_axis_gaps():
    anchor = actor("anchor")
    rule = {"relation": "spacing_uniformity", "minimum_subject_count": 3,
            "minimum_spacing_cm": 150.0, "maximum_spacing_cm": 450.0,
            "maximum_relative_deviation": 0.05}
    uniform = [actor("left", location=(0.0, 0.0, 0.0)),
               actor("middle", location=(300.0, 0.0, 0.0)),
               actor("right", location=(600.0, 0.0, 0.0))]
    nonuniform = [actor("left", location=(0.0, 0.0, 0.0)),
                  actor("middle", location=(200.0, 0.0, 0.0)),
                  actor("right", location=(600.0, 0.0, 0.0))]

    uniform_result = relations.evaluate(rule, uniform, [anchor])
    nonuniform_result = relations.evaluate(rule, nonuniform, [anchor])

    assert uniform_result["pass"] is True
    assert uniform_result["observed"]["adjacent_spacing_cm"] == [300.0, 300.0]
    assert uniform_result["observed"]["maximum_relative_deviation"] == 0.0
    assert nonuniform_result["pass"] is False
    assert nonuniform_result["observed"]["adjacent_spacing_cm"] == [200.0, 400.0]
    assert nonuniform_result["observed"][
        "maximum_relative_deviation"] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_parallel_compares_actor_yaw_cyclically():
    left = actor("left", rotation=(0.0, -179.0, 0.0))
    nearly_parallel = actor("right", rotation=(0.0, 179.0, 0.0))
    perpendicular = actor("wrong", rotation=(0.0, -89.0, 0.0))
    rule = {"relation": "parallel", "maximum_angle_deg": 5.0}

    assert relations.evaluate(rule, [left], [nearly_parallel])["pass"] is True
    assert relations.evaluate(rule, [left], [perpendicular])["pass"] is False


@pytest.mark.parametrize("subject_yaw", [45.0, 135.0])
def test_facing_offset_rejects_wrong_and_outward_orientations(subject_yaw):
    chair = actor("chair", location=(0.0, 0.0, 0.0),
                  rotation=(0.0, subject_yaw, 0.0))
    table = actor("table", location=(100.0, 100.0, 0.0))
    result = relations.evaluate({
        "relation": "facing",
        "maximum_angle_deg": 10.0,
        "subject_forward_yaw_offset_deg": 90.0,
    }, [chair], [table])

    assert result["pass"] is False
    assert result["rows"][0]["yaw_error_deg"] >= 90.0


def test_a_non_numeric_facing_offset_is_withheld(tmp_path):
    chair = actor("chair", location=(0.0, 0.0, 0.0),
                  rotation=(0.0, -45.0, 0.0))
    table = actor("table", location=(100.0, 100.0, 0.0))
    rule = {
        "id": "chair-facing-table",
        "relation": "facing",
        "maximum_angle_deg": 10.0,
        "subject_forward_yaw_offset_deg": "ninety",
        "subject": {"scope": "candidate_all", "labels": ["chair"]},
        "object": {"scope": "candidate_all", "labels": ["table"]},
    }
    report = spatial_relations.verify(context(
        tmp_path, candidate=scene([chair, table]),
        case_spec=relation_case(rule)))

    assert report["status"] == "not_evaluated"
    assert "subject_forward_yaw_offset_deg must be a JSON number" in report[
        "failure_reason"]


def test_an_unknown_relation_is_withheld_and_not_failed(tmp_path):
    """A typo in a case file must not read as a wrong scene."""
    rule = {"id": "typo", "relation": "leftt_of",
            "subject": CANDIDATE_ALL, "object": CANDIDATE_ALL}
    report = spatial_relations.verify(context(
        tmp_path, candidate=scene([actor("a"), actor("b")]),
        case_spec=relation_case(rule)))
    assert report["status"] == "not_evaluated"
    assert "unknown spatial relation" in report["failure_reason"]


def test_a_rule_that_is_not_an_object_does_not_leave_the_denominator(tmp_path):
    """A malformed relation used to be skipped silently, so the score became
    the share of the SURVIVING rules that hold — 1.0 here, over one rule of a
    declared two. case_spec does not deep-validate relations, so this path is
    live, and it withholds like every other unanswerable check."""
    table = actor("table", location=(0.0, 0.0, 40.0), extent=(80.0, 80.0, 40.0))
    chair = actor("chair", location=(120.0, 0.0, 40.0), extent=(25.0, 25.0, 40.0))
    holding = {"id": "chair-near-table", "relation": "near",
               "maximum_distance_cm": 200.0,
               "subject": {"scope": "candidate_all", "labels": ["chair"]},
               "object": {"scope": "candidate_all", "labels": ["table"]}}
    spec = case([{"id": "relations", "primitive": "spatial_relation",
                  "relations": [holding, "not-a-rule"]}])
    report = spatial_relations.verify(context(
        tmp_path, candidate=scene([table, chair]), case_spec=spec))

    assert report["status"] == "not_evaluated" and report["score"] is None
    assert "not an object" in report["failure_reason"]


@pytest.mark.parametrize(("relation", "holds"), [
    ("left_of", True), ("right_of", False), ("above", False), ("below", False)])
def test_the_world_axis_relations_agree_about_which_side_is_which(relation, holds):
    subject = actor("s", location=(0.0, 0.0, 0.0), extent=(10.0, 10.0, 10.0))
    other = actor("o", location=(100.0, 0.0, 0.0), extent=(10.0, 10.0, 10.0))
    ok, observed = relations.observe(subject, other, {"relation": relation})
    assert ok is holds
    if relation == "left_of":
        assert observed["axis_gap_cm"] == pytest.approx(80.0)


def test_against_accepts_a_shelf_flat_on_a_wall():
    wall = actor("wall", location=(0.0, 0.0, 150.0), extent=(500.0, 10.0, 150.0))
    shelf = actor("shelf", location=(0.0, 30.0, 100.0), extent=(50.0, 20.0, 10.0))
    ok, observed = relations.observe(shelf, wall, {"relation": "against"})
    assert ok
    assert observed["closest_surface_distance_cm"] == 0.0
    # Flush against, not driven into: X and Z overlap, Y only touches.
    assert observed["intersecting_axis_count"] == 2

    detached = actor("shelf", location=(0.0, 200.0, 100.0), extent=(50.0, 20.0, 10.0))
    assert not relations.observe(detached, wall, {"relation": "against"})[0]
