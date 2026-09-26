"""Frozen legacy-assertion leaves executed as internal atomic evaluators.

The migration tool copies a validated assertion into an EvaluationBinding.
Runtime scoring reads only that frozen binding; it never reopens the legacy
contract or reparses the task prompt.  Existing geometry algorithms are
called through their old modules so migration changes orchestration, not the
measurement definition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from code4scene.evaluation import atomic_registry, contracts
from code4scene.evaluation.context import Context
from code4scene.evaluation.evaluation_policy import load_evaluation_policy

from .bundle import FrozenVerificationBundle, UnknownReason
from .contracts import ClaimVerdict
from .deterministic import DeterministicAssessment


EVALUATOR_ID = "semantic_requirements.legacy_contract_atomic"
EVALUATOR_VERSION = "legacy-contract-atomic-v1"

_VERIFIERS = frozenset(
    {
        "structure_count",
        "structure_concepts",
        "structure_additions",
        "spatial_overlap",
        "spatial_clearance",
        "spatial_cluster",
        "spatial_relations",
    }
)


def _report_check(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(report.get("report_id") or "legacy_contract_atomic"),
        "status": str(report.get("status") or contracts.ERROR),
        "expected": {},
        "observed": {
            "score": report.get("score"),
            "metrics": dict(report.get("metrics") or {}),
        },
        "evidence": [
            {
                "legacy_atomic_report": {
                    "report_id": report.get("report_id"),
                    "status": report.get("status"),
                    "score": report.get("score"),
                    "metrics": dict(report.get("metrics") or {}),
                    "evidence": dict(report.get("evidence") or {}),
                    "artifacts": dict(report.get("artifacts") or {}),
                    "probes_used": list(report.get("probes_used") or ()),
                }
            }
        ],
        "failure_reason": report.get("failure_reason"),
    }


def _unknown_reason(report: Mapping[str, Any]) -> UnknownReason:
    checks = (report.get("metrics") or {}).get("checks")
    if isinstance(checks, list) and any(
        isinstance(value, Mapping)
        and value.get("status") in {"not_evaluated", "not_applicable"}
        for value in checks
    ):
        return UnknownReason.TARGETED_EVIDENCE_MISSING
    status = str(report.get("status") or "")
    if status in {"not_evaluated", "not_applicable"}:
        return UnknownReason.TARGETED_EVIDENCE_MISSING
    reason = str(report.get("failure_reason") or "").casefold()
    if any(
        token in reason
        for token in (
            "input_scene",
            "matched nothing",
            "no usable bounds",
            "could not be measured",
            "missing evidence",
            "population",
            "scope",
        )
    ):
        return UnknownReason.TARGETED_EVIDENCE_MISSING
    return UnknownReason.EVALUATION_ERROR


def evaluate_legacy_contract_bindings(
    context: Context,
    bundle: FrozenVerificationBundle,
) -> tuple[DeterministicAssessment, ...]:
    """Evaluate migrated leaves from their frozen binding parameters."""

    policy = load_evaluation_policy(context.task)
    results: list[DeterministicAssessment] = []
    for binding in bundle.evaluations:
        if binding.evaluator_id != EVALUATOR_ID:
            continue
        if binding.evaluator_version != EVALUATOR_VERSION:
            raise ValueError(
                f"{binding.binding_id}: unsupported legacy atomic version "
                f"{binding.evaluator_version!r}"
            )
        parameters = binding.parameters
        verifier_name = str(parameters.get("legacy_verifier") or "")
        assertion = parameters.get("assertion")
        if verifier_name not in _VERIFIERS:
            raise ValueError(
                f"{binding.binding_id}: unsupported legacy verifier "
                f"{verifier_name!r}"
            )
        if not isinstance(assertion, Mapping):
            raise ValueError(f"{binding.binding_id}: assertion must be an object")
        case_id = str(getattr(context.task, "id", "scene"))
        spec = policy.leaf_spec(context.task, "legacy_contract_atomic", context.spec)
        spec.pop("case_spec", None)
        spec["case_id"] = case_id
        spec["semantic_contract"] = {
            "schema_version": "0.2.0",
            "case_id": case_id,
            "policy_id": "offline-legacy-contract-migration-v1",
            "assertions": [dict(assertion)],
        }
        report = atomic_registry.get(verifier_name).evaluate_report(
            replace(context, spec=spec),
            assertion,
        )
        status = str(report.get("status") or contracts.ERROR)
        score = report.get("score")
        if status == contracts.MEASURED and score == 1.0:
            verdict = ClaimVerdict.MATCH
            unknown_reason = None
        elif status == contracts.MEASURED and isinstance(score, (int, float)):
            verdict = ClaimVerdict.MISMATCH
            unknown_reason = None
        elif status == contracts.PASS:
            verdict = ClaimVerdict.MATCH
            unknown_reason = None
        elif status == contracts.FAIL:
            verdict = ClaimVerdict.MISMATCH
            unknown_reason = None
        else:
            verdict = ClaimVerdict.UNKNOWN
            unknown_reason = _unknown_reason(report)
        results.append(
            DeterministicAssessment(
                node_id=binding.node_id,
                requirement_id=binding.requirement_id,
                rule_id=binding.binding_id,
                family=f"LEGACY_{assertion.get('primitive', 'ASSERTION')}",
                verdict=verdict,
                check=_report_check(report),
                unknown_reason=unknown_reason,
                rationale=str(
                    report.get("failure_reason")
                    or "the frozen legacy assertion matched"
                ),
            )
        )
    return tuple(results)


__all__ = [
    "EVALUATOR_ID",
    "EVALUATOR_VERSION",
    "evaluate_legacy_contract_bindings",
]
