"""Acceptance tests for the five public verifier entry points."""

from __future__ import annotations

import json

import pytest
import yaml

from code4scene.tasks import task as task_mod
from code4scene.evaluation import contracts, verifiers
from code4scene.tasks.verifier_schema import AUTOMATIC_POLICY_KINDS


def write_task(tmp_path, specs, name="t.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump({
        "id": "t-1",
        "kind": "scene_generation",
        "inputs": {"prompt": "build a scene", "init_map": "/Game/Maps/empty"},
        "assets": {"packs": ["Maps"]},
        "verifiers": specs,
    }, sort_keys=False))
    return path


RECORD = {"metrics": {"actors": 200, "structural_collision_rate": 0.05,
                      "floating_rate": 0.0, "oob_rate": 0.0},
          "edge_discipline": {"passes": [], "total_clamped": 0, "total_deleted": 0},
          "tool_calls": 12}
IDS = {"task_bundle_id": "b", "episode_id": "e"}
def _report(kind, context, *, status=None):
    status = (
        contracts.VALID
        if status is None and kind == "candidate_integrity"
        else contracts.MEASURED
        if status is None
        else status
    )
    return {
        **contracts.base(kind, context.ids),
        "status": status,
        "score": 1.0 if status == contracts.MEASURED else None,
        "metrics": {},
        "evidence": {},
        "artifacts": {},
        "probes_used": (),
    }


def test_automatic_policies_run_before_the_declared_task_goal(tmp_path, monkeypatch):
    for kind in (*AUTOMATIC_POLICY_KINDS, "scene_diff"):
        monkeypatch.setitem(
            verifiers.REGISTRY,
            kind,
            lambda context, report_id=kind: _report(report_id, context),
        )
    task = task_mod.load(write_task(tmp_path, [{
        "name": "scene_diff",
        "ground_truth": "/Game/GT/Canonical",
    }]))
    reports = verifiers.run(task, RECORD, IDS)
    assert [report["report_id"] for report in reports] == [
        *AUTOMATIC_POLICY_KINDS,
        "scene_diff",
    ]


def test_no_task_goal_still_runs_the_two_automatic_policies(tmp_path, monkeypatch):
    for kind in AUTOMATIC_POLICY_KINDS:
        monkeypatch.setitem(
            verifiers.REGISTRY,
            kind,
            lambda context, report_id=kind: _report(report_id, context),
        )
    task = task_mod.load(write_task(tmp_path, []))
    assert [report["report_id"] for report in verifiers.run(task, RECORD, IDS)] == [
        *AUTOMATIC_POLICY_KINDS
    ]


def test_selected_recompute_keeps_integrity_and_skips_automatic_physics(
    tmp_path, monkeypatch
):
    for kind in ("candidate_integrity", "overview_prompt_alignment"):
        monkeypatch.setitem(
            verifiers.REGISTRY,
            kind,
            lambda context, report_id=kind: _report(report_id, context),
        )
    task = task_mod.load(write_task(tmp_path, [{
        "name": "overview_prompt_alignment",
    }]))

    reports = verifiers.run(
        task,
        RECORD,
        IDS,
        only_verifiers=("overview_prompt_alignment",),
    )

    assert [report["report_id"] for report in reports] == [
        "candidate_integrity",
        "overview_prompt_alignment",
    ]


def test_selected_evidence_requests_only_include_requested_task_verifier(
    tmp_path,
):
    task = task_mod.load(write_task(tmp_path, [
        {"name": "overview_prompt_alignment"},
    ]))

    requests = verifiers.render_evidence_requests(
        task,
        only_verifiers=("overview_prompt_alignment",),
    )

    assert len(requests) == 1
    assert requests[0].protocol.endswith("candidate_clearance_gallery")


def test_image_guided_repair_routes_the_complete_five_verifier_surface(
    tmp_path, monkeypatch
):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    path = tmp_path / "repair.yaml"
    path.write_text(yaml.safe_dump({
        "id": "repair-1",
        "kind": "scene_repair",
        "case_type": "image_to_scene",
        "inputs": {"prompt": "repair it", "init_map": "/Game/Input/Repair"},
        "assets": {"packs": ["Input"]},
        "source": {"reference_views": [str(reference)]},
        "verifiers": [
            {
                "name": "semantic_requirements",
                "verification_bundle": "repair.verification.json",
            },
            {
                "name": "scene_diff",
                "ground_truth": "/Game/GT/Repair",
            },
        ],
    }, sort_keys=False))
    for kind in (
        "candidate_integrity",
        "source_preservation",
        "physical_safety",
        "semantic_requirements",
        "scene_diff",
    ):
        monkeypatch.setitem(
            verifiers.REGISTRY,
            kind,
            lambda context, report_id=kind: _report(report_id, context),
        )

    reports = verifiers.run(task_mod.load(path), RECORD, IDS)

    assert [report["report_id"] for report in reports] == [
        "candidate_integrity",
        "source_preservation",
        "physical_safety",
        "semantic_requirements",
        "scene_diff",
    ]


def test_image_guided_generation_with_3d_gt_does_not_route_preservation(
    tmp_path, monkeypatch
):
    """Image guidance is not evidence of a pre-existing scene to preserve."""

    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    path = tmp_path / "image-generation.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "image-generation-1",
                "kind": "scene_generation",
                "case_type": "image_to_scene",
                "inputs": {
                    "prompt": "build it",
                    "init_map": "/Game/Maps/empty",
                },
                "assets": {"packs": ["Maps"]},
                "source": {"reference_views": [str(reference)]},
                "verifiers": [
                    {
                        "name": "semantic_requirements",
                        "verification_bundle": "generation.verification.json",
                    },
                    {
                        "name": "scene_diff",
                        "ground_truth": "/Game/GT/Generation",
                    },
                ],
            },
            sort_keys=False,
        )
    )
    expected = (
        "candidate_integrity",
        "physical_safety",
        "semantic_requirements",
        "scene_diff",
    )
    for kind in expected:
        monkeypatch.setitem(
            verifiers.REGISTRY,
            kind,
            lambda context, report_id=kind: _report(report_id, context),
        )

    reports = verifiers.run(task_mod.load(path), RECORD, IDS)

    assert [report["report_id"] for report in reports] == list(expected)
    assert all(
        report["report_id"] != "source_preservation" for report in reports
    )


def test_gt_repair_integrates_locality_without_duplicate_source_preservation(
    tmp_path, monkeypatch
):
    path = tmp_path / "gt-repair.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "gt-repair-1",
                "kind": "scene_repair",
                "case_type": "image_to_scene",
                "source": {"pack": "Dungeon"},
                "inputs": {
                    "prompt": "optional",
                    "init_map": "/Game/Input/Repair",
                },
                "assets": {"packs": ["Input"]},
                "verifiers": [
                    {
                        "name": "gt_repair",
                        "ground_truth": "/Game/GT/Repair",
                    }
                ],
            },
            sort_keys=False,
        )
    )
    expected = (
        "candidate_integrity",
        "physical_safety",
        "gt_repair",
    )
    for kind in expected:
        monkeypatch.setitem(
            verifiers.REGISTRY,
            kind,
            lambda context, report_id=kind: _report(report_id, context),
        )

    reports = verifiers.run(task_mod.load(path), RECORD, IDS)

    assert [report["report_id"] for report in reports] == list(expected)
    assert all(
        report["report_id"] != "source_preservation" for report in reports
    )


def test_a_kind_nothing_implements_is_refused_at_load(tmp_path):
    """Before an environment is leased, not after an agent has spent an hour.

    Ignoring one silently is how a task got scored by something other than
    what it asked for.
    """
    path = write_task(tmp_path, [{"name": "vibes_check"}])

    with pytest.raises(task_mod.TaskError, match="which nothing implements"):
        task_mod.load(path)


def test_a_verifier_that_raises_still_produces_a_report(tmp_path, monkeypatch):
    def explode(_context):
        raise RuntimeError("verifier exploded")

    for kind in AUTOMATIC_POLICY_KINDS:
        implementation = explode if kind == "physical_safety" else (
            lambda context, report_id=kind: _report(report_id, context)
        )
        monkeypatch.setitem(verifiers.REGISTRY, kind, implementation)
    task = task_mod.load(write_task(tmp_path, []))
    report = next(
        value
        for value in verifiers.run(task, RECORD, IDS)
        if value["report_id"] == "physical_safety"
    )
    assert report["status"] == "error"
    assert "verifier exploded" in report["failure_reason"]


def test_reports_are_json_native(tmp_path):
    """Every report can be written into the episode JSON record."""
    task = task_mod.load(write_task(tmp_path, []))
    reports = verifiers.run(task, RECORD, IDS)

    assert isinstance(json.loads(json.dumps(reports)), list)


#: Every verifier that opens the answer key. Pinned as a literal rather than
#: derived, because the whole point is that adding one is a decision: a
#: verifier moving into this set can only run in the scoring environment, and
#: a verifier moving out of it is claiming it stopped reading the answer.
GT_VERIFIERS = {
    "gt_repair",
    "scene_diff",
}


def test_the_answer_key_is_what_separates_the_two_classes():
    """Only the verifiers that open a `.label.json` are GT.

    Everything else answers from the candidate, the prompt, or the scene the
    agent was handed to edit — which is why an open-ended verifier still works
    on a task that ships no answer key, and most do not. Note which side
    `physics_regression` and the `preservation_*` family land on: they compare
    against another scene, and that scene is the INPUT, which the agent had.
    """
    classes = {k: verifiers.classify(k) for k in verifiers.kinds()}
    assert {k for k, v in classes.items() if v == "gt"} == GT_VERIFIERS
    assert all(value in ("gt", "open_ended") for value in classes.values())
    for kind in ("candidate_integrity", "source_preservation", "physical_safety"):
        assert classes[kind] == "open_ended", (
            f"{kind} does not read the hidden answer key")


def test_a_report_cannot_talk_its_way_into_the_other_column():
    """The directory is the whole answer; a report saying otherwise is noise.

    There used to be a kind whose class its own report declared. A scorer that
    names its own column is a scorer that can put the answer in the answerless
    one, so `classify` ignores the report entirely."""
    assert verifiers.classify(
        "physical_safety", {"metrics": {"scored_against": "gt"}}
    ) == "open_ended"
    assert verifiers.classify(
        "scene_diff", {"metrics": {"scored_against": "open_ended"}}
    ) == "gt"
