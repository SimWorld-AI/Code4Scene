"""The paper scoring protocol on synthetic evidence (no benchmark content)."""

from __future__ import annotations

import copy
import math

import pytest

from code4scene.evaluation import physics_score
from code4scene.protocol import actor_f1, aggregate, i2s, physics, t2s
from code4scene.protocol.constants import (
    SETTING_INDOOR,
    SETTING_OUTDOOR,
    SETTING_T2S,
)


def _actor(actor_id: str, x_cm: float, *, asset: str, y_cm: float = 0.0,
           yaw: float = 0.0) -> dict:
    return {
        "stable_actor_id": actor_id,
        "label": actor_id,
        "actor_path": f"/Game/Synthetic.Synthetic:PersistentLevel.{actor_id}",
        "class": "/Script/Engine.StaticMeshActor",
        "asset_path": f"/Game/Synthetic/{asset}.{asset}",
        "actor_tags": [],
        "transform": {"location_cm": [x_cm, y_cm, 50.0], "rotation_deg": [0.0, yaw, 0.0],
                      "scale": [1.0, 1.0, 1.0]},
        "bounds": {"origin_cm": [x_cm, y_cm, 50.0], "extent_cm": [20.0, 20.0, 20.0]},
        "properties": {},
    }


def _scene(*actors: dict) -> dict:
    return {"actors": list(actors), "actor_count": len(actors),
            "export_metadata": {"status": "success"}}


def _worked_example():
    """Paper Appendix C.5: restore a globe, reposition a chair, delete a box."""
    wall, plant = _actor("wall", 0.0, asset="wall"), _actor("plant", -300.0, asset="plant")
    chair_input = _actor("chair", 300.0, asset="chair")
    box = _actor("box", 500.0, asset="box")
    globe = _actor("globe", 700.0, asset="globe")
    chair_gt = _actor("chair", 100.0, asset="chair")
    initial = _scene(wall, plant, chair_input, box)
    ground_truth = _scene(wall, plant, chair_gt, globe)
    # Restores the globe and deletes the box, leaves the chair misplaced and
    # deletes an unrelated plant.
    candidate = _scene(wall, chair_input, globe)
    return initial, ground_truth, candidate


def test_paper_worked_example_counts_and_f1():
    values, audit = actor_f1.measure(*_worked_example())

    assert (values["true_positive"], values["false_positive"], values["false_negative"]) == (
        2, 2, 1)
    assert values["precision"] == pytest.approx(0.5)
    assert values["recall"] == pytest.approx(2 / 3)
    assert values["f1"] == pytest.approx(4 / 7)
    assert audit["off_target_removal_ids"] == ["stable:plant"]
    assert values["true_positive"] + values["false_negative"] == values["desired_count"] == 3


def test_perfect_repair_scores_one_and_no_op_scores_zero():
    initial, ground_truth, _ = _worked_example()
    perfect, _ = actor_f1.measure(initial, ground_truth, copy.deepcopy(ground_truth))
    assert (perfect["true_positive"], perfect["false_positive"],
            perfect["false_negative"]) == (3, 0, 0)
    assert perfect["f1"] == 1.0

    untouched, _ = actor_f1.measure(initial, ground_truth, copy.deepcopy(initial))
    assert untouched["true_positive"] == 0
    assert untouched["f1"] == 0.0


def test_actor_names_and_ids_never_change_the_counts():
    initial, ground_truth, candidate = _worked_example()
    renamed = copy.deepcopy(candidate)
    for i, actor in enumerate(renamed["actors"]):
        actor["stable_actor_id"] = f"renamed_{i}"
        actor["label"] = f"Renamed{i}"
        actor["actor_path"] = f"/Game/Synthetic.Synthetic:PersistentLevel.Renamed{i}"
    renamed["actors"].reverse()

    original, _ = actor_f1.measure(initial, ground_truth, candidate)
    moved, _ = actor_f1.measure(initial, ground_truth, renamed)
    assert {k: original[k] for k in actor_f1.COUNTS} == {k: moved[k] for k in actor_f1.COUNTS}


def test_repurposing_a_background_actor_to_fill_a_target_is_charged():
    wall = _actor("wall", 0.0, asset="wall")
    lamp_a = _actor("lamp_a", -400.0, asset="lamp")
    initial = _scene(wall, lamp_a)
    lamp_b = _actor("lamp_b", 400.0, asset="lamp")
    ground_truth = _scene(wall, lamp_a, lamp_b)
    # The candidate moves the existing lamp onto the missing one's position.
    moved = copy.deepcopy(lamp_a)
    moved["transform"]["location_cm"][0] = 400.0
    moved["bounds"]["origin_cm"][0] = 400.0
    values, audit = actor_f1.measure(initial, ground_truth, _scene(wall, moved))
    assert values["true_positive"] == 1
    assert values["false_positive"] == 1  # the lost background lamp
    assert audit["repurposed_background_ids"] == ["stable:lamp_a"]


def test_repair_tolerances_are_five_cm():
    initial = _scene(_actor("wall", 0.0, asset="wall"))
    ground_truth = _scene(_actor("wall", 0.0, asset="wall"), _actor("vase", 200.0, asset="vase"))
    near = _scene(_actor("wall", 0.0, asset="wall"), _actor("x", 204.9, asset="vase"))
    far = _scene(_actor("wall", 0.0, asset="wall"), _actor("x", 205.1, asset="vase"))
    assert actor_f1.measure(initial, ground_truth, near)[0]["f1"] == 1.0
    assert actor_f1.measure(initial, ground_truth, far)[0]["f1"] == 0.0


def test_a_case_without_targets_is_refused():
    scene = _scene(_actor("wall", 0.0, asset="wall"))
    with pytest.raises(actor_f1.ActorF1Error):
        actor_f1.measure(scene, copy.deepcopy(scene), copy.deepcopy(scene))


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


def _physics_report(floating=("measured", 0.2), penetration=("measured", 0.9)):
    leaves = []
    for name, (status, score) in (("floating", floating), ("solid_penetration", penetration)):
        direction = "lower_is_better" if name == "floating" else "higher_is_better"
        leaves.append({"leaf_id": name, "report_id": f"physical_safety.{name}",
                       "status": status, "score": score if status == "measured" else None,
                       "metadata": {"score_direction": direction},
                       **({} if status == "measured" else {"failure_reason": status})})
    return {"report_id": "physical_safety", "status": "measured",
            "metrics": {"leaf_results": leaves,
                        "leaf_score_directions": {"floating": "lower_is_better",
                                                  "solid_penetration": "higher_is_better"}}}


def test_physics_is_the_fixed_weight_mean_of_the_two_safety_leaves():
    scored = physics.score_from_report(_physics_report())
    assert scored["score"] == round(0.5 * 0.8 + 0.5 * 0.9, 4)


@pytest.mark.parametrize("status", ["not_applicable", "not_evaluated", "error"])
def test_an_unavailable_physics_leaf_scores_zero_at_its_weight(status):
    report = _physics_report(penetration=(status, None))
    assert physics.score_from_report(report)["score"] == 0.4
    # Same answer as the verifier layer's projection.
    assert physics_score.project(report)["score"] == 0.4


def test_t2s_support_rule_uses_a_five_cm_gap():
    def box(name, bottom):
        return {"label": name, "class": "/Script/Engine.StaticMeshActor",
                "transform": {"location_cm": [0.0, 0.0, bottom + 10.0]},
                "bounds": {"origin_cm": [0.0, 0.0, bottom + 10.0],
                           "extent_cm": [10.0, 10.0, 10.0]}}
    table = {"label": "table", "class": "/Script/Engine.StaticMeshActor",
             "transform": {"location_cm": [0.0, 0.0, 50.0]},
             "bounds": {"origin_cm": [0.0, 0.0, 50.0], "extent_cm": [100.0, 100.0, 50.0]}}
    for gap, floating in ((0.0, False), (5.0, False), (5.01, True)):
        rate = physics.t2s_floating_rate({"actors": [table, box("cup", 100.0 + gap)]})
        assert (rate["floating_count"] == 1) is floating
    # The world ground plane supports anything within 5 cm of z = 0.
    assert physics.t2s_floating_rate({"actors": [box("crate", 5.0)]})["rate"] == 0.0
    assert physics.t2s_floating_rate({"actors": [box("crate", 5.5)]})["rate"] == 1.0


def test_t2s_physics_recomputes_floating_from_the_snapshot():
    report = _physics_report(floating=("measured", 0.9), penetration=("measured", 1.0))
    grounded = {"label": "crate", "class": "/Script/Engine.StaticMeshActor",
                "transform": {"location_cm": [0.0, 0.0, 10.0]},
                "bounds": {"origin_cm": [0.0, 0.0, 10.0], "extent_cm": [10.0, 10.0, 10.0]}}
    scored = physics.t2s_score({"actors": [grounded]}, report)
    assert scored["floating_source"] == "recomputed_from_snapshot"
    assert scored["score"] == 1.0
    assert physics.t2s_score(None, report)["score"] == round(0.5 * 0.1 + 0.5, 4)


# ---------------------------------------------------------------------------
# Text-to-scene
# ---------------------------------------------------------------------------


def _row(family, clause, score, status="MATCH", included=True):
    return {"semantic_family": family, "clause_index": clause, "effective_score": score,
            "evaluation_status": status, "included": included, "predicate_weight": 1.0,
            "score_known": status in {"MATCH", "MISMATCH"}}


def test_detailed_alignment_is_a_clause_macro_family_weighted_mean():
    rows = [
        _row("identity_environment", 0, 1.0),
        _row("content_quantity", 0, 1.0), _row("content_quantity", 0, 0.0),  # clause 0: 0.5
        _row("content_quantity", 1, 1.0),                                     # clause 1: 1.0
        _row("spatial_composition", 1, 0.0, status="NOT_EVALUATED"),  # unresolved: zero
        _row("attributes_materials", 2, 1.0, included=False),  # excluded summary
    ]
    result = t2s.detailed_from_decisions(rows)
    families = result["families"]
    assert families["content_quantity"]["score"] == 0.75
    assert families["spatial_composition"]["score"] == 0.0
    assert families["attributes_materials"]["applicable"] is False
    # Attributes are absent: renormalize over 0.25 + 0.40 + 0.20.
    expected = (0.25 * 1.0 + 0.40 * 0.75 + 0.20 * 0.0) / 0.85
    assert result["score"] == round(expected, 4)


def test_legacy_interval_decisions_use_their_conservative_endpoint():
    row = _row("content_quantity", 0, None)
    del row["effective_score"]
    row.update(effective_lower=0.25, effective_upper=0.75)
    assert t2s.detailed_from_decisions([row])["score"] == 0.25


def test_overview_alignment_formula_and_severe_cap():
    dims = {"global_prompt_alignment": 0.8, "composition_and_layout": 0.6,
            "style_atmosphere_coherence": 1.0, "completeness_and_polish": 0.4}
    prompt = 0.40 * 0.8 + 0.25 * 0.6 + 0.20 * 1.0 + 0.15 * 0.4
    soft = t2s.overview_from_judgement(dims, 0.5)
    assert soft["score"] == round(prompt * (0.75 + 0.25 * 0.5), 4)
    capped = t2s.overview_from_judgement(dims, 0.5, severe_cap_eligible=True)
    assert capped["score"] == 0.40 and capped["severe_cap_applied"]


def test_t2s_case_score_weights_rounding_and_zero_rules():
    case = t2s.case_score(valid=True, detailed=0.5, overview=0.8, physics=0.9)
    assert case["score"] == round(0.2 * 0.5 + 0.2 * 0.9 + 0.6 * 0.8, 4)
    missing = t2s.case_score(valid=True, detailed=None, overview=0.8, physics=0.9)
    assert missing["score"] == round(0.2 * 0.9 + 0.6 * 0.8, 4)
    assert missing["missing_components"] == ["detailed"]
    assert t2s.case_score(valid=False, detailed=1, overview=1, physics=1)["score"] == 0.0


def test_unusable_reports_contribute_nothing():
    assert t2s.detailed_from_report({"status": "error", "metrics": {
        "semantic_requirement_aggregation": [_row("content_quantity", 0, 1.0)]}})["score"] is None
    assert t2s.overview_from_report({"status": "not_evaluated"})["score"] is None


# ---------------------------------------------------------------------------
# Image-to-scene and the model score
# ---------------------------------------------------------------------------


def test_i2s_case_score_is_point_eight_f1_plus_point_two_physics():
    initial, ground_truth, candidate = _worked_example()
    scored = i2s.score_case(valid=True, input_scene=initial, ground_truth_scene=ground_truth,
                            candidate_scene=candidate,
                            physical_safety_report=_physics_report())
    phys = round(0.5 * 0.8 + 0.5 * 0.9, 4)
    assert scored["score"] == round(math.fsum((0.8 * 4 / 7, 0.2 * phys)), 6)
    invalid = i2s.score_case(valid=False, input_scene=None, ground_truth_scene=None,
                             candidate_scene=None, physical_safety_report=None)
    assert invalid["score"] == 0.0 and invalid["actor_f1"]["f1"] == 0.0


def _rows():
    rows = []
    for model, base in (("model-a", 0.6), ("model-b", 0.3)):
        rows += [{"model": model, "setting": SETTING_T2S, "case_id": f"t{i}",
                  "score": base} for i in range(2)]
        rows += [{"model": model, "setting": SETTING_INDOOR, "case_id": "in0",
                  "score": base, "repair_f1": base, "physics": 1.0}]
        rows += [{"model": model, "setting": SETTING_OUTDOOR, "case_id": f"out{i}",
                  "score": base / 2, "repair_f1": base / 2, "physics": 1.0} for i in range(2)]
    return rows


def test_model_score_weights_every_i2s_case_equally():
    by_model = {r["model"]: r for r in aggregate.aggregate(_rows())}
    a = by_model["model-a"]
    assert a["t2s_score"] == pytest.approx(0.6)
    # One indoor case at 0.6 and two outdoor cases at 0.3 -> (0.6 + 0.6) / 3.
    assert a["i2s_score"] == pytest.approx(0.4)
    assert a["model_score"] == pytest.approx(0.5)
    assert a["model_rank"] == 1 and by_model["model-b"]["model_rank"] == 2


def test_a_scheduled_case_without_a_row_is_a_zero_or_blocks_the_aggregate():
    rows = [r for r in _rows() if not (r["model"] == "model-a" and r["case_id"] == "t1")]
    schedule = {SETTING_T2S: ["t0", "t1"], SETTING_INDOOR: ["in0"],
                SETTING_OUTDOOR: ["out0", "out1"]}
    zero = {r["model"]: r for r in aggregate.aggregate(rows, schedule=schedule)}
    assert zero["model-a"]["t2s_score"] == pytest.approx(0.3)
    assert zero["model-a"]["t2s_zero_cases"] == 1
    strict = {r["model"]: r for r in aggregate.aggregate(rows, schedule=schedule,
                                                          missing_as_zero=False)}
    assert strict["model-a"]["model_score"] is None
    assert strict["model-a"]["unresolved_cases"] == ["text-to-scene:t1"]


def test_invalid_cases_stay_in_the_denominator():
    rows = _rows()
    rows[0] = {**rows[0], "status": "zero_invalid_candidate", "score": 0.9}
    a = {r["model"]: r for r in aggregate.aggregate(rows)}["model-a"]
    assert a["t2s_score"] == pytest.approx(0.3)


def test_the_gt_repair_verifier_measures_actor_f1_from_its_own_exports(monkeypatch):
    from types import SimpleNamespace

    from code4scene.evaluation import ue_evidence
    from code4scene.evaluation.verifiers import gt_repair

    initial, ground_truth, candidate = _worked_example()
    monkeypatch.setattr(ue_evidence, "collect", lambda _context: SimpleNamespace(
        input_scene=initial, candidate=candidate))
    monkeypatch.setattr(gt_repair, "load_repair_target_scope",
                        lambda _task, input_scene: SimpleNamespace(canonical_scene=ground_truth))
    context = SimpleNamespace(spec={"ground_truth": "/Game/Synthetic/GT"}, task=None)
    measured = gt_repair._actor_repair_f1(context)
    assert measured["status"] == "measured"
    assert (measured["true_positive"], measured["false_positive"],
            measured["false_negative"]) == (2, 2, 1)
    assert measured["f1"] == pytest.approx(4 / 7)
