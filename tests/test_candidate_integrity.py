"""The gate every other verifier assumes has already been passed.

The execution-failure half of `candidate_integrity` had no behaviour test,
which is how it silently failed: the gate checked ``exit_reason == "timeout"``
while the harness has only ever written ``time_cap``, so the stated policy
that a timed-out run is an execution failure never fired for anyone.
"""

from __future__ import annotations

from dataclasses import replace

from declared_cases import actor, context, scene

from code4scene.evaluation import contracts
from code4scene.evaluation.evaluation_policy import default_policy
from code4scene.evaluation.verifiers import candidate_integrity


def _report(tmp_path, **record_overrides):
    ctx = context(tmp_path, candidate=scene([actor("crate")]))
    ctx.record["scene_dependencies"] = {"unresolved": []}
    ctx.record.update(record_overrides)
    return candidate_integrity.verify(ctx)


def test_a_clean_completed_run_passes_the_gate(tmp_path):
    report = _report(tmp_path, exit_reason="completed")
    assert report["status"] == contracts.VALID and report["score"] is None
    assert report["evidence"]["case_outcome"]["classification"] == "valid"
    snapshot = report["metrics"]["leaf_results"][0]
    assert snapshot["metrics"]["execution_failure_count"] == 0


def test_a_run_the_wall_clock_killed_is_an_execution_failure(tmp_path):
    """`time_cap` is the value the harness actually writes — see the exit
    classification in the harness. The gate used to look for "timeout",
    which nothing produces, so this policy was dead on arrival."""
    report = _report(tmp_path, exit_reason="time_cap")

    assert report["status"] == contracts.INVALID and report["score"] is None
    assert "time_cap" in report["failure_reason"]
    assert report["evidence"]["case_outcome"] == {
        "schema_version": "scenebenchmark.case_outcome.v1",
        "classification": "model_invalid",
        "failure_owner": "model",
        "score_disposition": "fixed_zero",
        "reason_codes": ["generation_time_cap"],
        "source": "candidate_integrity_leaves",
    }
    snapshot = report["metrics"]["leaf_results"][0]
    assert snapshot["metrics"]["execution_failure_count"] == 1


def test_a_verified_checkpoint_is_valid_without_claiming_generation_completed(
    tmp_path,
):
    report = _report(
        tmp_path,
        exit_reason="time_cap",
        candidate_provenance={
            "classification": "incomplete_checkpoint",
            "completion_status": "incomplete",
            "termination_reason": "time_cap",
            "artifact_role": "diagnostic_partial",
            "checkpoint_identity_verified": True,
            "integrity_errors": [],
        },
    )

    assert report["status"] == contracts.VALID
    assert report["evidence"]["candidate_classification"] == (
        "incomplete_checkpoint"
    )
    assert report["evidence"]["generation_completed"] is False
    snapshot = report["metrics"]["leaf_results"][0]
    assert snapshot["metrics"]["is_incomplete_checkpoint"] is True
    assert snapshot["metrics"]["execution_failure_count"] == 0


def test_provider_diagnostic_does_not_invalidate_a_verified_checkpoint(tmp_path):
    report = _report(
        tmp_path,
        candidate_provenance={
            "classification": "incomplete_checkpoint",
            "completion_status": "incomplete",
            "termination_reason": "tool_cap",
            "artifact_role": "diagnostic_partial",
            "checkpoint_identity_verified": True,
            "provider_error_observed": True,
            "integrity_errors": [],
        },
    )

    assert report["status"] == contracts.VALID
    snapshot = report["metrics"]["leaf_results"][0]
    assert snapshot["evidence"]["candidate_provenance"][
        "provider_error_observed"
    ] is True


def test_an_unsaved_candidate_dependency_is_model_invalid(tmp_path):
    report = _report(
        tmp_path,
        exit_reason="completed",
        scene_dependencies={
            "unresolved": ["/Game/Generated/M_Missing"],
        },
    )

    assert report["status"] == contracts.INVALID
    assert report["evidence"]["case_outcome"]["classification"] == (
        "model_invalid"
    )
    assert report["evidence"]["case_outcome"]["reason_codes"] == [
        "candidate_dependency_missing"
    ]


def test_contradictory_checkpoint_provenance_is_invalid(tmp_path):
    report = _report(
        tmp_path,
        candidate_provenance={
            "classification": "candidate_invalid",
            "completion_status": "incomplete",
            "integrity_errors": ["checkpoint SHA-256 mismatch"],
        },
    )

    assert report["status"] == contracts.INVALID
    assert "checkpoint SHA-256 mismatch" in report["failure_reason"]
    assert report["evidence"]["case_outcome"]["classification"] == (
        "invalid"
    )
    assert report["evidence"]["case_outcome"]["score_disposition"] == "fixed_zero"
    assert all(
        not value["eligible"]
        for value in report["evidence"]["verifier_eligibility"].values()
    )


def test_a_recorded_infra_error_is_withheld_from_model_scoring(tmp_path):
    """`infra_error` is the one failure FIELD the harness writes; the fields
    this gate once also read (agent_error, compile_error, runtime_error) have
    no producer anywhere and were removed rather than left looking live."""
    report = _report(tmp_path, exit_reason="completed",
                     infra_error="editor died mid-round")

    assert report["status"] == contracts.ERROR
    assert "infra_error" in report["failure_reason"]
    assert report["evidence"]["case_outcome"]["classification"] == (
        "infrastructure_error"
    )


def test_an_unpublishable_exit_withholds_rather_than_failing(tmp_path):
    report = _report(tmp_path, exit_reason="tampered")
    assert report["status"] == contracts.ERROR and report["score"] is None


def test_release_label_drift_is_audit_only_without_frozen_manifest(tmp_path):
    report = _report(
        tmp_path,
        exit_reason="completed",
        provenance={
            "content_release": "generation-release",
            "scoring_content_release": "scoring-release",
        },
    )

    assert report["status"] == contracts.VALID
    manifest = report["metrics"]["leaf_results"][2]
    assert manifest["status"] == contracts.VALID
    assert manifest["metrics"]["release_label_drift_count"] == 1
    assert manifest["evidence"]["release_label_drift"] == {
        "observed": True,
        "gated_by_frozen_manifest": False,
        "policy": "audit_only_without_frozen_manifest",
    }
    assert all(
        value["eligible"]
        for value in report["evidence"]["verifier_eligibility"].values()
    )


def test_frozen_release_contract_still_rejects_runtime_mismatch(
    tmp_path, monkeypatch
):
    ctx = context(tmp_path, candidate=scene([actor("crate")]))
    ctx.record.update(
        {
            "exit_reason": "completed",
            "scene_dependencies": {"unresolved": [], "release": "frozen-release"},
            "provenance": {
                "content_release": "generation-release",
                "scoring_content_release": "frozen-release",
            },
        }
    )
    policy = replace(
        default_policy(ctx.task),
        asset_library_manifest={
            "manifest_id": "frozen-test",
            "content_release": "frozen-release",
        },
    )
    monkeypatch.setattr(candidate_integrity, "load_evaluation_policy", lambda _task: policy)

    report = candidate_integrity.verify(ctx)

    assert report["status"] == contracts.ERROR
    manifest = report["metrics"]["leaf_results"][2]
    assert manifest["status"] == contracts.ERROR
    assert manifest["metrics"]["release_mismatch_count"] == 1
    assert manifest["evidence"]["release_label_drift"] == {
        "observed": True,
        "gated_by_frozen_manifest": True,
        "policy": "frozen_contract",
    }
    assert all(
        not value["eligible"]
        and value["source_leaf"] == "asset_library_manifest_parity"
        for value in report["evidence"]["verifier_eligibility"].values()
    )
