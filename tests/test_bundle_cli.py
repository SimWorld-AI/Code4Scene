"""Evidence bundles and the ``code4scene`` CLI on synthetic evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from code4scene import bundle as bundle_mod
from code4scene import cli, scoring
from code4scene.protocol import constants
from test_offline_overview_prompt_alignment import FakeClient
from test_protocol import _actor, _physics_report, _worked_example


def _integrity(status="valid"):
    return {"status": status, "leaves": {leaf: status for leaf in bundle_mod.INTEGRITY_LEAVES},
            "failure_reason": None if status == "valid" else "synthetic failure"}


def _i2s_bundle(root: Path, *, integrity="valid", model="model-a", case="synthetic-repair"):
    initial, ground_truth, candidate = _worked_example()
    writer = bundle_mod.BundleWriter(root)
    scenes = {role: writer.add_json(f"scenes/{role}.scene.json", scene)
              for role, scene in (("input", initial), ("ground_truth", ground_truth),
                                  ("candidate", candidate))}
    physics = writer.add_json("physics/physical_safety.json", _physics_report())
    return writer.finish({
        "task": {"id": case, "setting": constants.SETTING_OUTDOOR},
        "run": {"model": model, "submission": "submitted", "batch_status": "complete",
                "authoritative": True},
        "candidate_integrity": _integrity(integrity),
        "scenes": scenes, "physics": {"report": physics},
    })


def _decision(family, clause, score, stage):
    return {"node_id": f"n_{family}_{clause}_{stage}", "semantic_family": family,
            "clause_index": clause, "effective_score": score, "included": True,
            "evaluation_status": "MATCH" if score == 1.0 else "MISMATCH",
            "predicate_weight": 1.0, "score_known": True, "resolved_by": stage}


def _t2s_bundle(root: Path, *, views: int = 0):
    writer = bundle_mod.BundleWriter(root)
    grounded = _actor("crate", 0.0, asset="crate")
    grounded["transform"]["location_cm"][2] = 20.0
    grounded["bounds"]["origin_cm"][2] = 20.0  # bottom on the ground plane
    manifest = {
        "task": {"id": "synthetic-t2s", "setting": constants.SETTING_T2S},
        "run": {"model": "model-a"},
        "candidate_integrity": _integrity(),
        "scenes": {"candidate": writer.add_json("scenes/candidate.scene.json",
                                                {"actors": [grounded]})},
        "physics": {"report": writer.add_json("physics/physical_safety.json",
                                              _physics_report(floating=("measured", 0.5),
                                                              penetration=("measured", 1.0)))},
        "semantic": {"decisions": writer.add_json("semantic/decisions.json", {
            "report_status": "measured",
            "decisions": [_decision("content_quantity", 0, 1.0, "stage1"),
                          _decision("content_quantity", 0, 1.0, "stage3"),
                          _decision("spatial_composition", 1, 0.0, "stage3")]})},
        "overview": {"judgement": writer.add_json("overview/judgement.json", {
            "dimension_scores": {"global_prompt_alignment": 1.0, "composition_and_layout": 0.5,
                                 "style_atmosphere_coherence": 1.0,
                                 "completeness_and_polish": 1.0},
            "structural_integrity_score": 1.0, "severe_structural_cap_eligible": False})},
    }
    if views:
        paths = []
        for i in range(views):
            image = root.parent / "frames" / f"view_{i + 1}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            rgb = np.random.default_rng(i).integers(0, 256, (16, 24, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(image)
            paths.append(writer.add_file(f"renders/overview/view_{i + 1}.png", image))
        manifest["overview"]["views"] = paths
    return writer.finish(manifest)


def test_i2s_bundle_scores_offline_without_any_model(tmp_path):
    loaded = _i2s_bundle(tmp_path / "b")
    record = scoring.score_bundle(loaded, judge="none")
    phys = round(0.5 * 0.8 + 0.5 * 0.9, 4)
    assert record["status"] == "measured" and record["comparable_to_paper"]
    assert record["score"] == round(math.fsum((0.8 * 4 / 7, 0.2 * phys)), 6)
    assert record["components"]["repair_f1"]["true_positive"] == 2


def test_invalid_candidate_scores_zero(tmp_path):
    record = scoring.score_bundle(_i2s_bundle(tmp_path / "b", integrity="invalid"), judge="none")
    assert record["score"] == 0.0 and record["status"] == "zero_invalid_candidate"


def test_bundle_hashes_and_paths_are_enforced(tmp_path):
    loaded = _i2s_bundle(tmp_path / "b")
    (loaded.root / "physics/physical_safety.json").write_text("{}")
    with pytest.raises(bundle_mod.BundleError, match="changed"):
        bundle_mod.load(loaded.root)
    manifest = json.loads((loaded.root / "bundle.json").read_text())
    manifest["physics"]["report"] = "../outside.json"
    with pytest.raises(bundle_mod.BundleError, match="inside the bundle"):
        bundle_mod.validate_manifest(manifest)
    manifest["physics"]["report"] = "/abs/physics.json"
    with pytest.raises(bundle_mod.BundleError, match="inside the bundle"):
        bundle_mod.validate_manifest(manifest)


def test_t2s_recorded_judgements_reproduce_the_protocol(tmp_path):
    record = scoring.score_bundle(_t2s_bundle(tmp_path / "b"), judge="recorded")
    detailed = round((0.40 * 1.0 + 0.20 * 0.0) / 0.60, 4)
    overview = round(0.40 + 0.25 * 0.5 + 0.20 + 0.15, 4)
    assert record["components"]["physical_safety"]["score"] == 1.0  # recomputed support
    assert record["score"] == round(0.2 * detailed + 0.2 * 1.0 + 0.6 * overview, 4)
    assert record["comparable_to_paper"]


def test_no_vlm_scores_structured_leaves_only_and_flags_partial(tmp_path):
    record = scoring.score_bundle(_t2s_bundle(tmp_path / "b"), judge="none")
    assert record["status"] == "partial" and not record["comparable_to_paper"]
    assert set(record["not_evaluated_components"]) == {"detailed_alignment",
                                                        "overview_alignment"}
    # Only the Stage 1 (structured) decision keeps its score.
    assert record["components"]["detailed_alignment"]["score"] == round(0.40 * 0.5 / 0.60, 4)
    assert record["components"]["overview_alignment"]["score"] is None


def test_live_overview_uses_the_judge_on_the_bundled_views(tmp_path):
    from synthetic_tasks import text_to_scene

    from code4scene.tasks import task as task_mod

    task = task_mod.load(text_to_scene(tmp_path, task_id="synthetic-t2s"))
    loaded = _t2s_bundle(tmp_path / "b", views=4)
    record = scoring.score_bundle(loaded, task, judge="live",
                                  client_factory=lambda: FakeClient())
    overview = record["components"]["overview_alignment"]
    assert overview["status"] == "measured" and overview["judge_record"]["calls"]["count"] == 2
    # No Stage 3 schedule in this bundle: visual requirements are not re-judged.
    assert record["components"]["detailed_alignment"]["status"] == "partial"


def test_cli_score_rescore_and_aggregate(tmp_path, capsys):
    out_dir = tmp_path / "scores"
    out_dir.mkdir()
    for model, integrity in (("model-a", "valid"), ("model-b", "invalid")):
        b = _i2s_bundle(tmp_path / model, integrity=integrity, model=model)
        assert cli.main(["validate-bundle", str(b.root)]) == 0
        assert cli.main(["score", str(b.root), "--no-vlm", "--out",
                         str(out_dir / f"{model}.json")]) == 0
    capsys.readouterr()
    schedule = tmp_path / "schedule"
    schedule.mkdir()
    (schedule / "public-outdoor-cases.txt").write_text("synthetic-repair\nsynthetic-other\n")
    assert cli.main(["aggregate", str(out_dir), "--schedule", str(schedule),
                     "--format", "csv", "--out", str(tmp_path / "models.csv")]) == 0
    rows = {r["model"]: r for r in csv.DictReader((tmp_path / "models.csv").open())}
    a = json.loads((out_dir / "model-a.json").read_text())["score"]
    # The second scheduled case has no row: a missing result counts as zero.
    assert float(rows["model-a"]["outdoor_i2s_score"]) == pytest.approx(a / 2)
    assert float(rows["model-b"]["outdoor_i2s_score"]) == 0.0
    assert rows["model-a"]["model_score"] == ""  # no text-to-scene cases scheduled


def test_cli_rescore_reads_a_saved_verifier_result(tmp_path):
    initial, ground_truth, candidate = _worked_example()
    case_dir = tmp_path / "results" / "synthetic-repair"
    evidence = case_dir / "scene_evidence" / "run-1"
    evidence.mkdir(parents=True)
    for role, scene in (("input", initial), ("ground_truth", ground_truth),
                        ("candidate", candidate)):
        (evidence / f"{role}.scene.json").write_text(json.dumps(scene))
    result = {
        "schema_version": "scenebenchmark-artifact-score.v1",
        "task_id": "synthetic-repair", "benchmark_track": "image_to_scene_outdoor",
        "scene_environment": "outdoor", "batch_status": "complete", "authoritative": True,
        "reports": [
            {"report_id": "candidate_integrity", "status": "valid", "metrics": {
                "leaf_results": [{"leaf_id": leaf, "status": "valid"}
                                 for leaf in bundle_mod.INTEGRITY_LEAVES]}},
            _physics_report(),
            {"report_id": "gt_repair", "status": "measured", "score": 0.5},
        ],
    }
    (case_dir / "result.json").write_text(json.dumps(result))
    out = tmp_path / "case.json"
    assert cli.main(["rescore", str(case_dir / "result.json"), "--model", "model-a",
                     "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["setting"] == constants.SETTING_OUTDOOR
    assert record["components"]["repair_f1"]["f1"] == pytest.approx(4 / 7)


# ---------------------------------------------------------------------------
# One overall number: the verifier layer publishes the protocol case score
# ---------------------------------------------------------------------------


def _saved_i2s_result():
    return {
        "task_id": "synthetic-repair", "benchmark_track": "image_to_scene_outdoor",
        "scene_environment": "outdoor", "batch_status": "complete", "authoritative": True,
        "reports": [
            {"report_id": "candidate_integrity", "status": "valid", "metrics": {
                "leaf_results": [{"leaf_id": leaf, "status": "valid"}
                                 for leaf in bundle_mod.INTEGRITY_LEAVES]}},
            _physics_report(),
            {"report_id": "gt_repair", "status": "measured", "score": 0.9,
             "task_bundle_id": "b", "episode_id": "e"},
        ],
    }


def test_rebuilt_i2s_result_has_the_paper_score_as_its_only_overall(tmp_path):
    loaded = _i2s_bundle(tmp_path / "b")
    record = scoring.score_bundle(loaded, judge="none")
    rebuilt = scoring.rebuild_result(_saved_i2s_result(), loaded)
    assert rebuilt["overall_score"] == record["score"] == rebuilt["primary_score"]["score"]
    assert rebuilt["primary_score"]["source_id"] == constants.I2S_CASE_POLICY
    repair = next(r for r in rebuilt["reports"] if r["report_id"] == "gt_repair")
    assert repair["score"] == pytest.approx(4 / 7)
    # The legacy composite survives only as a named diagnostic.
    assert repair["metrics"]["diagnostics"]["legacy_gt_repair_composite"] == 0.9
    assert rebuilt["primary_score"]["diagnostics"]["legacy_gt_repair_composite"] == 0.9
    assert scoring.verifier_layer_summary(rebuilt, record)["consistent"]


def test_publishing_the_repair_f1_is_idempotent():
    from code4scene.evaluation import repair_score

    report = {"report_id": "gt_repair", "status": "measured", "score": 0.9,
              "metrics": {"actor_repair_f1": {"status": "measured", "f1": 0.25}}}
    once = repair_score.publish_actor_f1(json.loads(json.dumps(report)))
    twice = repair_score.publish_actor_f1(json.loads(json.dumps(once)))
    assert once == twice
    assert twice["score"] == 0.25
    assert twice["metrics"]["diagnostics"]["legacy_gt_repair_composite"] == 0.9


def test_rebuilt_t2s_result_matches_the_protocol_case_score(tmp_path):
    loaded = _t2s_bundle(tmp_path / "b")
    record = scoring.score_bundle(loaded, judge="recorded")
    decisions = loaded.decisions()
    judgement = loaded.overview_judgement()
    physics_report = _physics_report(floating=("measured", 0.5), penetration=("measured", 1.0))
    result = {
        "task_id": "synthetic-t2s", "benchmark_track": "text_to_scene",
        "batch_status": "complete", "authoritative": True,
        "reports": [
            {"report_id": "candidate_integrity", "status": "valid", "metrics": {
                "leaf_results": [{"leaf_id": leaf, "status": "valid"}
                                 for leaf in bundle_mod.INTEGRITY_LEAVES]}},
            physics_report,
            {"report_id": "semantic_requirements", "status": "measured",
             "score": record["components"]["detailed_alignment"]["score"],
             "metrics": {"semantic_requirement_aggregation": decisions,
                         "semantic_subscores": {
                             family: {"applicable": value["applicable"],
                                      "score": value.get("score"),
                                      "known_coverage": value.get("known_coverage")}
                             for family, value in
                             record["components"]["detailed_alignment"]["families"].items()}}},
            {"report_id": "overview_prompt_alignment", "status": "measured",
             "score": record["components"]["overview_alignment"]["score"],
             "metrics": {"dimensions": {k: {"score": v} for k, v in
                                        judgement["dimension_scores"].items()},
                         "structural_integrity_score": 1.0}},
        ],
    }
    rebuilt = scoring.rebuild_result(result, loaded)
    floating_leaf = next(
        leaf for leaf in next(r for r in rebuilt["reports"]
                              if r["report_id"] == "physical_safety")["metrics"]["leaf_results"]
        if leaf["leaf_id"] == "floating")
    assert floating_leaf["evidence"]["source"] == "candidate_scene_snapshot"
    assert floating_leaf["score"] == 0.0
    assert rebuilt["overall_score"] == record["score"]


def test_the_t2s_floating_leaf_is_measured_on_the_exported_snapshot(monkeypatch):
    from types import SimpleNamespace

    from code4scene.evaluation import ue_evidence
    from code4scene.evaluation.context import Context
    from code4scene.evaluation.verifiers import floating

    grounded = _actor("crate", 0.0, asset="crate")
    lifted = _actor("lamp", 300.0, asset="lamp")
    lifted["bounds"]["origin_cm"][2] = 200.0
    lifted["transform"]["location_cm"][2] = 200.0
    grounded["bounds"]["origin_cm"][2] = 20.0
    monkeypatch.setattr(ue_evidence, "collect", lambda _context: SimpleNamespace(
        candidate={"actors": [grounded, lifted]}))
    context = Context(record={"metrics": {"actors": 2, "floating_rate": 0.0}},
                      task=SimpleNamespace(kind="scene_generation", case_type="prompt_to_scene",
                                           data={}),
                      ids={"task_bundle_id": "b", "episode_id": "e"}, spec={"name": "floating"})
    report = floating.verify(context)
    assert report["evidence"]["source"] == "candidate_scene_snapshot"
    assert report["score"] == 0.5
    assert report["evidence"]["live_measure_floating_rate"] == 0.0
