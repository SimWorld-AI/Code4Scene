from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from code4scene.evaluation import primary_score, repair_score, repair_success
from code4scene.evaluation.gt_geometry_compare import compare_detailed
from code4scene.evaluation.repair_target_scope import RepairTargetScope, measure_repair_target
from code4scene.evaluation.requirement_graph.repair_target_authoring import derive_repair_targets
from code4scene.evaluation.verifiers import gt_repair


def actor(name, x=0.0, *, asset="chair"):
    return {"stable_actor_id": name, "class": "/Script/Engine.StaticMeshActor",
            "asset_path": f"/Game/{asset}.{asset}", "properties": {},
            "transform": {"location_cm": [x, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0],
                          "scale": [1.0, 1.0, 1.0]},
            "bounds": {"origin_cm": [x, 0.0, 0.0], "extent_cm": [20.0, 20.0, 20.0]}}


def scope(desired, initial=()):
    before, gt = {"actors": list(initial)}, {"actors": list(desired)}
    return RepairTargetScope(input_scene=before, canonical_scene=gt,
                             targets=derive_repair_targets(before, gt))


def evaluate(desired, candidates, initial=()):
    return repair_success.measure(scope(desired, initial), {"actors": candidates})


def test_identity_correspondence_does_not_imply_repair_success():
    gt, candidate = actor("gt"), actor("prediction", 200.0)
    target_scope = scope([gt])
    measured = measure_repair_target(target_scope, {"actors": [candidate]},
                                    compare_detailed([candidate], [gt]))
    assert measured["matched_actor_count"] == 1
    assert measured["actor_correspondence_score"] == 1.0
    assert measured["repair_success"]["profiles"]["nominal"]["true_positive"] == 0
    # Continuous matching may fit away a global shift; this diagnostic uses
    # the fixed world frame, not a Candidate-dependent registration.
    assert measured["repair_success"]["profiles"]["nominal"]["recall"] == 0.0


@pytest.mark.parametrize("distance,expected", [(0, 1), (5, 1), (5.001, 0)])
def test_position_boundary(distance, expected):
    data = evaluate([actor("gt")], [actor("candidate", distance)])
    assert data["profiles"]["nominal"]["true_positive"] == expected


@pytest.mark.parametrize("field,value,expected", [
    ("rotation_deg", [0, 5, 0], 1),
    ("rotation_deg", [0, 6, 0], 0),  # Do not dilute one bad axis by averaging.
    ("rotation_deg", [0, 360, 0], 1),
    ("rotation_deg", [180, 180, 180], 1),  # Same SO(3) orientation.
    ("scale", [1.05, 1, 1], 1),
    ("scale", [1.051, 1, 1], 0),
    ("scale", [-1, 1, 1], 0),  # Reflection cannot pass through abs(scale).
    ("scale", [0, 1, 1], 0),
    ("scale", [float("nan"), 1, 1], 0),
])
def test_transform_checks_are_conjunctive(field, value, expected):
    candidate = actor("candidate")
    candidate["transform"][field] = value
    assert evaluate([actor("gt")], [candidate])["profiles"]["nominal"]["true_positive"] == expected


@pytest.mark.parametrize("change", ["asset", "properties", "material", "slots", "bounds", "missing_scale"])
def test_required_discrete_and_geometry_checks(change):
    gt, candidate = actor("gt"), actor("candidate")
    if change == "asset":
        candidate["asset_path"] = "/Game/Other.Other"
    elif change == "properties":
        candidate["properties"] = {"intensity": 500}
    elif change == "material":
        gt["material_paths"] = ["/Game/Wood.Wood"]
        # Missing Candidate evidence must fail, not become not-applicable.
    elif change == "slots":
        gt["component_material_slots"] = []
    elif change == "bounds":
        candidate["bounds"]["extent_cm"][0] = 25
    else:
        del candidate["transform"]["scale"]
    assert evaluate([gt], [candidate])["profiles"]["nominal"]["f1"] == 0


def test_optional_evidence_applicability_is_frozen_by_gt():
    gt, candidate = actor("gt"), actor("candidate")
    for value in (gt, candidate):
        del value["bounds"]
        del value["properties"]
    data = evaluate([gt], [candidate])["profiles"]["nominal"]
    assert data["f1"] == 1
    assert "properties" in data["matches"][0]["gt_unavailable_fields"]
    gt["properties"] = {}
    assert evaluate([gt], [candidate])["profiles"]["nominal"]["f1"] == 0


def test_duplicates_and_incompatible_extras_are_false_positives():
    candidates = [actor("one"), actor("duplicate"), actor("wrong_asset", asset="lamp")]
    nominal = evaluate([actor("gt")], candidates)["profiles"]["nominal"]
    assert (nominal["true_positive"], nominal["false_positive"], nominal["false_negative"]) == (1, 2, 0)
    assert nominal["precision"] == pytest.approx(1 / 3)
    assert nominal["recall"] == 1
    assert nominal["f1"] == 0.5


def test_maximum_cardinality_matches_only_success_edges():
    desired = [actor("left", 0), actor("right", 8)]
    candidates = [actor("shared", 4), actor("left_only", -4)]
    for ordering in (candidates, list(reversed(candidates))):
        nominal = evaluate(desired, ordering)["profiles"]["nominal"]
        assert nominal["true_positive"] == 2
        assert {(r["gt"], r["candidate"]) for r in nominal["matches"]} == {
            ("stable:left", "stable:left_only"), ("stable:right", "stable:shared")}


def test_no_candidate_is_zero_and_pure_removal_is_separate():
    missing = evaluate([actor("gt")], [])["profiles"]["nominal"]
    assert (missing["precision"], missing["recall"], missing["f1"]) == (0, 0, 0)
    removed = evaluate([], [], [actor("remove")])
    assert removed["profiles"]["nominal"]["status"] == "not_applicable"
    assert removed["profiles"]["nominal"]["precision"] is None
    assert removed["removal"]["completion"] == 1
    retained = evaluate([], [actor("remove")], [actor("remove")])
    assert retained["removal"]["completion"] == 0


def test_unchanged_background_excluded_but_off_target_edits_audited():
    keep, removed = actor("keep", 100), actor("wrongly_removed", 200)
    data = evaluate([keep, removed, actor("gt")], [keep, actor("prediction")], [keep, removed])
    assert data["profiles"]["nominal"]["candidate_count"] == 1
    assert data["off_target_removed_actor_ids"] == ["stable:wrongly_removed"]
    moved = actor("keep", 110)
    data = evaluate([keep, actor("gt")], [moved, actor("prediction")], [keep])
    assert data["profiles"]["nominal"]["false_positive"] == 1


def test_threshold_profiles_are_monotonic():
    data = evaluate([actor("gt")], [actor("candidate", 7)])
    assert [data["profiles"][name]["true_positive"] for name in repair_success.PROFILES] == [0, 0, 1]


def test_diagnostic_failure_cannot_break_continuous_measurement(monkeypatch):
    gt = actor("gt")
    def fail(*args):
        raise ValueError("diagnostic unavailable")
    monkeypatch.setattr(repair_success, "measure", fail)
    target = measure_repair_target(scope([gt]), {"actors": [gt]}, compare_detailed([gt], [gt]))
    assert target["actor_correspondence_score"] == 1
    assert target["repair_success"]["status"] == "not_evaluated"


@pytest.mark.parametrize("diagnostic", [None, {"status": "not_evaluated"}, "success", "failure"])
def test_local_gt_and_i2s_primary_scores_unchanged(diagnostic):
    gt = actor("gt")
    target = measure_repair_target(scope([gt]), {"actors": [gt]}, compare_detailed([gt], [gt]))
    if diagnostic in ("success", "failure"):
        target["repair_success"] = evaluate([gt], [actor("c", 0 if diagnostic == "success" else 200)])
    else:
        target["repair_success"] = diagnostic
    context = SimpleNamespace(ids={"task_bundle_id": "bundle", "episode_id": "episode"})
    local = gt_repair.report_from_repair_target_measurement(
        context, target, target_visual={"leaf_id": "target_visual_diff", "status": "not_evaluated", "score": None})
    baseline = copy.deepcopy(local)
    baseline["metrics"]["leaf_results"] = [r for r in baseline["metrics"]["leaf_results"] if r["leaf_id"] != "repair_success"]
    repair_score.apply_local(baseline)
    assert (local["score"], local["status"]) == (baseline["score"], baseline["status"])
    roots = []
    for value in (local, baseline):
        root = {"report_id": "gt_repair", "metrics": {"leaf_results": [
            {**value, "leaf_id": "repair_target_diff"},
            {"leaf_id": "scene_diff", "status": "measured", "score": 0.7,
             "metrics": {"score_weight_policy_id": repair_score.GLOBAL_POLICY_ID}},
        ]}}
        roots.append(repair_score.apply(root))
    assert roots[0]["score"] == roots[1]["score"]
    physics = {"report_id": "physical_safety", "status": "measured", "score": 0.8}
    integrity = {"report_id": "candidate_integrity", "status": "valid"}
    scores = [primary_score.from_reports([integrity, root, physics]) for root in roots]
    for key in ("score", "status", "source_id", "known_coverage", "reason_codes", "policy"):
        assert scores[0].get(key) == scores[1].get(key)


def test_aggregate_reports_micro_counts_and_missing_coverage():
    context = SimpleNamespace(ids={"task_bundle_id": "bundle", "episode_id": "episode"})
    first = repair_success.report(context, evaluate([actor("gt")], [actor("candidate")]))
    second = repair_success.report(context, evaluate([actor("gt")], [actor("c1"), actor("c2"), actor("c3")]))
    rows = [{"reports": [first]}, {"score_breakdown": primary_score.breakdown([second], {})}, {}]
    aggregate = repair_success.aggregate(rows)
    nominal = aggregate["profiles"]["nominal"]
    assert nominal["precision"] == 0.5  # 2/4, not the per-case mean 2/3.
    assert nominal["recall"] == 1
    assert nominal["f1"] == pytest.approx(2 / 3)
    assert aggregate["measured_case_count"] == 2
    assert aggregate["unavailable_case_count"] == 1


def test_malformed_or_duplicate_diagnostics_do_not_break_model_aggregation():
    context = SimpleNamespace(ids={"task_bundle_id": "bundle", "episode_id": "episode"})
    measured = repair_success.report(context, evaluate([actor("gt")], [actor("candidate")]))
    malformed = copy.deepcopy(measured)
    malformed["metrics"]["repair_success"]["profiles"]["nominal"]["true_positive"] = -1
    rows = [{"reports": {"gt_repair": 1}}, {"reports": [malformed]}, {"reports": [measured, measured]}]
    data = repair_success.aggregate(rows)
    assert data["measured_case_count"] == 0
    assert data["unavailable_case_count"] == 3
