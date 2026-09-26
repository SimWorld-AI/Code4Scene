"""``candidate_integrity`` — is there a scene here worth scoring at all.

The gate every other verifier consults before reading scene evidence. It asks
four things of the episode and the export, in this order:

* the run ENDED in a way that may be published — an infrastructure failure or
  a detected tamper is not a badly built scene, and scoring it would put a
  number on the harness rather than on the agent;
* the agent's own execution produced an attributable artifact — a verified
  checkpoint from a tool/time-capped run is measurable without pretending the
  generation completed;
* the exported Candidate is a well-formed scene graph — the fields every
  population is selected on are present and the right shape;
* and there is enough of it to measure, against the minimum the task declares.

The minimum-actor check is the one that earns its place. An empty level
measures perfectly: zero collisions, zero floating, zero out of bounds. Every
rate-based verifier already refuses a zero-actor scene for that reason, and
this states the same rule once, where a task can raise it.

Unlike its siblings this verifier needs no case specification: a task that
declares it wants the gate, not a contract.

Serves component `execution.candidate_integrity`.
"""

from __future__ import annotations

from typing import Any

from .. import case_outcome, contracts, ue_evidence
from ..case_spec import validate_candidate_contract
from ..composite import composite_report, run_leaf
from ..context import Context, error
from ..evaluation_policy import load_evaluation_policy
from . import content_parity


CLASS = "open_ended"

#: Record fields that carry an execution failure, and what each one means.
#: Only fields the harness actually writes belong here: a field nothing
#: produces is a check that can never fire, wearing the look of one that can.
_INFRASTRUCTURE_FAILURES = (
    ("infra_error", "the harness failed"),
)


def _verify_snapshot(context: Context) -> dict[str, Any]:
    record = context.record
    candidate_provenance = record.get("candidate_provenance")
    candidate_provenance = (
        candidate_provenance if isinstance(candidate_provenance, dict) else {}
    )
    candidate_classification = str(
        candidate_provenance.get("classification") or "unclassified_candidate"
    )
    checkpoint = candidate_classification == "incomplete_checkpoint"
    report = {
        **contracts.base("candidate_integrity", context.ids),
        "artifacts": contracts.artifacts(record),
        "probes_used": ("ue_scene_snapshot",),
    }
    unpublishable = contracts.unpublishable(record)
    if unpublishable:
        return {**report, "status": contracts.ERROR, "score": None,
                "metrics": {}, "evidence": {"exit_reason": unpublishable},
                "failure_reason": (
                    f"this run exited {unpublishable!r} and must not be scored, "
                    f"ranked or published; a number here would describe the "
                    f"harness, not the scene")}

    infrastructure_failures = [
        f"{field}: {record[field]} ({why})"
        for field, why in _INFRASTRUCTURE_FAILURES
        if record.get(field)
    ]
    if infrastructure_failures:
        return {
            **report,
            "status": contracts.ERROR,
            "score": None,
            "metrics": {},
            "evidence": {
                "exit_reason": record.get("exit_reason"),
                "infrastructure_failures": infrastructure_failures,
                "failure_attribution": {
                    "owner": "evaluation_infrastructure",
                    "reason_codes": ["harness_infrastructure_failure"],
                },
            },
            "failure_reason": "; ".join(infrastructure_failures),
        }

    failures = []
    model_reason_codes: list[str] = []
    infrastructure_reason_codes: list[str] = []
    # A normal episode killed at the wall clock remains invalid. An
    # independently exported diagnostic checkpoint is different: the launcher
    # already proved byte identity to the saved diagnostic UMAP and preserves
    # the incomplete outcome below instead of relabeling it completed.
    if record.get("exit_reason") == "time_cap" and not checkpoint:
        failures.append("exit_reason: time_cap (the agent ran out of wall clock)")
        model_reason_codes.append("generation_time_cap")

    try:
        evidence = ue_evidence.collect(context)
    except Exception as exc:  # noqa: BLE001 - fatal export refusal
        return error(
            "candidate_integrity",
            context,
            "the Candidate could not be exported, so there is no scene to "
            f"check: {type(exc).__name__}: {exc}",
        )
    schema_errors = validate_candidate_contract(evidence.candidate)
    provenance_errors = [
        str(value)
        for value in (candidate_provenance.get("integrity_errors") or [])
    ]
    actors = len(evidence.candidate_actors())
    minimum = context.spec.get("minimum_actor_count")
    minimum = 1 if minimum is None or isinstance(minimum, bool) else int(minimum)
    metrics = {
        "actor_count": actors,
        "minimum_actor_count": minimum,
        "execution_failure_count": len(failures),
        "schema_error_count": len(schema_errors),
        "provenance_error_count": len(provenance_errors),
        "is_incomplete_checkpoint": checkpoint,
    }
    body = {
        **report,
        "metrics": metrics,
        "evidence": {
            **evidence.evidence(),
            "execution_failures": failures,
            "candidate_schema_errors": schema_errors,
            "exit_reason": record.get("exit_reason"),
            "candidate_classification": candidate_classification,
            "candidate_provenance": candidate_provenance or None,
            "generation_completed": (
                candidate_provenance.get("completion_status") == "completed"
            ),
        },
    }
    reasons = []
    if failures:
        reasons.append("; ".join(failures))
    if schema_errors:
        model_reason_codes.append("candidate_schema_invalid")
        reasons.append(
            "the exported Candidate is not a well-formed scene graph: "
            + "; ".join(schema_errors[:4])
        )
    if provenance_errors or candidate_classification == "candidate_invalid":
        infrastructure_reason_codes.append("candidate_provenance_invalid")
        reasons.append(
            "the mapped Candidate contradicts its generation provenance: "
            + "; ".join(provenance_errors[:4] or ["candidate_invalid"])
        )
    if actors < minimum:
        model_reason_codes.append(
            "empty_scene" if actors == 0 else "candidate_below_minimum_actor_count"
        )
        reasons.append(
            f"the Candidate holds {actors} Actor(s), under the declared minimum "
            f"of {minimum}; an empty level measures perfectly on every rate"
        )
    if reasons:
        attribution = {
            "owner": (
                "evaluation_infrastructure"
                if infrastructure_reason_codes
                else "model"
            ),
            "reason_codes": [
                *model_reason_codes,
                *infrastructure_reason_codes,
            ],
        }
        return {
            **body,
            "status": contracts.INVALID,
            "score": None,
            "evidence": {
                **body["evidence"],
                "failure_attribution": attribution,
            },
            "failure_reason": "; ".join(reasons),
        }
    return {**body, "status": contracts.VALID, "score": None}


def _gate_evidence(
    snapshot: dict[str, Any],
    parity: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Lower leaf findings into explicit eligibility for each consumer."""

    downstream = content_parity.DOWNSTREAM_VERIFIER_KINDS
    blocked: dict[str, dict[str, Any]] = {}
    if snapshot.get("status") != contracts.VALID:
        for verifier_id in downstream:
            blocked[verifier_id] = {
                "reason_code": "candidate_snapshot_invalid",
                "source_leaf": "candidate_snapshot_integrity",
                "reason": snapshot.get("failure_reason"),
            }
    else:
        parity_evidence = parity.get("evidence")
        parity_evidence = parity_evidence if isinstance(parity_evidence, dict) else {}
        for verifier_id in parity_evidence.get("blocked_verifiers") or []:
            if verifier_id in downstream:
                blocked[verifier_id] = {
                    "reason_code": parity_evidence.get("reason_code")
                    or "dependency_not_ready",
                    "source_leaf": "content_and_dependency_parity",
                    "reason": parity.get("failure_reason"),
                }
        if manifest.get("status") != contracts.VALID:
            for verifier_id in downstream:
                blocked[verifier_id] = {
                    "reason_code": "asset_library_manifest_not_ready",
                    "source_leaf": "asset_library_manifest_parity",
                    "reason": manifest.get("failure_reason"),
                }

    eligibility = {
        verifier_id: (
            {
                "eligible": False,
                **blocked[verifier_id],
            }
            if verifier_id in blocked
            else {
                "eligible": True,
                "reason_code": None,
                "source_leaf": None,
                "reason": None,
            }
        )
        for verifier_id in downstream
    }
    parity_evidence = parity.get("evidence")
    parity_evidence = parity_evidence if isinstance(parity_evidence, dict) else {}
    snapshot_evidence = snapshot.get("evidence")
    snapshot_evidence = snapshot_evidence if isinstance(snapshot_evidence, dict) else {}
    return {
        "schema_version": "scenebenchmark.candidate_integrity.v2",
        "gate_mode": "per_verifier_eligibility.v2",
        "candidate_classification": snapshot_evidence.get(
            "candidate_classification", "unclassified_candidate"
        ),
        "generation_completed": snapshot_evidence.get("generation_completed"),
        "environment_readiness": {
            "status": parity_evidence.get("environment_readiness", "unknown"),
            "dependency_classification": parity_evidence.get(
                "dependency_classification", "unknown"
            ),
            "candidate_unresolved": parity_evidence.get("candidate_unresolved", []),
            "environment_unresolved": parity_evidence.get(
                "environment_unresolved", []
            ),
            "asset_library_manifest_status": manifest.get("status"),
        },
        "verifier_eligibility": eligibility,
        "fatal_gate": True,
        "requires_all_leaves": True,
    }


def _verify_asset_library(context: Context, policy: Any) -> dict[str, Any]:
    manifest = dict(policy.asset_library_manifest)
    dependencies = context.record.get("scene_dependencies")
    dependencies = dependencies if isinstance(dependencies, dict) else {}
    provenance = context.record.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    observed = {
        "candidate_dependencies": dependencies.get("release"),
        "candidate_environment": provenance.get("content_release"),
        "scoring_environment": provenance.get("scoring_content_release"),
    }
    expected = manifest.get("content_release")
    concrete = {
        str(value)
        for value in observed.values()
        if isinstance(value, str) and value
    }
    release_label_drift = len(concrete) > 1
    missing = []
    if expected and not (
        observed["candidate_dependencies"] or observed["candidate_environment"]
    ):
        missing.append("candidate content release")
    if expected and context.scoring is not None and not observed["scoring_environment"]:
        missing.append("scoring content release")
    mismatches = sorted(
        value for value in concrete if expected and value != expected
    )
    report = {
        **contracts.base("asset_library_manifest_parity", context.ids),
        "score": None,
        "metrics": {
            "observed_release_count": len(concrete),
            "release_label_drift_count": int(release_label_drift),
            "release_mismatch_count": len(mismatches),
            "missing_required_manifest_count": len(missing),
        },
        "evidence": {
            "frozen_asset_library_manifest": manifest or None,
            "observed_content_releases": observed,
            "release_label_drift": {
                "observed": release_label_drift,
                "gated_by_frozen_manifest": bool(expected),
                "policy": (
                    "frozen_contract"
                    if expected
                    else "audit_only_without_frozen_manifest"
                ),
            },
            "manifest_applies_to": ["Candidate", "Input", "GT"],
        },
        "artifacts": contracts.artifacts(context.record),
        "probes_used": ("scene_dependencies", "evaluation_policy"),
    }
    # A release label is provenance, not package-level dependency evidence.
    # Without a frozen expected release there is no contract for these opaque
    # labels to satisfy, and content_and_dependency_parity has already checked
    # whether every referenced package is available in the scorer. Keep label
    # drift visible in the audit, but do not block otherwise complete artifacts.
    if mismatches or missing:
        reasons = []
        if mismatches:
            reasons.append(
                f"runtime release(s) {mismatches} differ from frozen {expected!r}"
            )
        if missing:
            reasons.append("missing " + ", ".join(missing))
        return {
            **report,
            "status": contracts.ERROR,
            "failure_reason": "; ".join(reasons),
        }
    return {**report, "status": contracts.VALID}


def verify(context: Context) -> dict[str, Any]:
    try:
        policy = load_evaluation_policy(context.task)
    except Exception as exc:  # noqa: BLE001 - frozen policy refusal
        return error("candidate_integrity", context, f"{type(exc).__name__}: {exc}")
    snapshot = run_leaf(context, _verify_snapshot, spec=context.spec)
    snapshot["leaf_id"] = "candidate_snapshot_integrity"
    parity = run_leaf(context, content_parity.verify, spec=context.spec)
    parity["leaf_id"] = "content_and_dependency_parity"
    manifest = _verify_asset_library(context, policy)
    manifest["leaf_id"] = "asset_library_manifest_parity"
    outcome = case_outcome.from_integrity_leaves(snapshot, parity, manifest)
    return composite_report(
        "candidate_integrity",
        context,
        [snapshot, parity, manifest],
        evidence={
            **policy.evidence(),
            **_gate_evidence(snapshot, parity, manifest),
            "case_outcome": outcome,
        },
    )


__all__ = ["verify"]
