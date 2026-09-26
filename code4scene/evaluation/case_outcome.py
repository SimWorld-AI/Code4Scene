"""Classify whether one generated scene may receive a quality score.

Candidate validity and evaluator availability are separate. An invalid
Candidate receives fixed zero even when failure ownership is unknown. An
evaluator error without an invalid verdict remains unresolved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "scenebenchmark.case_outcome.v1"

VALID = "valid"
MODEL_INVALID = "model_invalid"
INVALID = "invalid"
INFRASTRUCTURE_ERROR = "infrastructure_error"
UNKNOWN = "unknown"

_CLASSIFICATIONS = frozenset(
    {VALID, MODEL_INVALID, INVALID, INFRASTRUCTURE_ERROR, UNKNOWN}
)


def _positive_count(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and value > 0
    )


def make(
    classification: str,
    *,
    reason_codes: Sequence[str] = (),
    source: str,
) -> dict[str, Any]:
    """Build one normalized, JSON-native case-outcome record."""

    if classification not in _CLASSIFICATIONS:
        raise ValueError(f"unsupported case-outcome classification {classification!r}")
    owner = {
        VALID: None,
        MODEL_INVALID: "model",
        INVALID: "unknown",
        INFRASTRUCTURE_ERROR: "evaluation_infrastructure",
        UNKNOWN: "unknown",
    }[classification]
    disposition = {
        VALID: "quality_score",
        MODEL_INVALID: "fixed_zero",
        INVALID: "fixed_zero",
        INFRASTRUCTURE_ERROR: "withheld",
        UNKNOWN: "withheld",
    }[classification]
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "failure_owner": owner,
        "score_disposition": disposition,
        "reason_codes": sorted(
            {str(value) for value in reason_codes if str(value).strip()}
        ),
        "source": source,
    }


def _explicit(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    classification = value.get("classification")
    if classification not in _CLASSIFICATIONS:
        return None
    reasons = value.get("reason_codes")
    reasons = reasons if isinstance(reasons, (list, tuple)) else ()
    normalized = make(
        str(classification),
        reason_codes=[str(reason) for reason in reasons],
        source=str(value.get("source") or "explicit_case_outcome"),
    )
    return {**dict(value), **normalized}


def from_integrity_leaves(
    snapshot: Mapping[str, Any],
    parity: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Attribute a Candidate Integrity composite from its three leaves."""

    leaves = (snapshot, parity, manifest)
    statuses = [str(leaf.get("status") or "") for leaf in leaves]
    if "error" in statuses:
        codes = [
            f"{leaf.get('leaf_id') or leaf.get('report_id') or 'integrity_leaf'}_error"
            for leaf in leaves
            if leaf.get("status") == "error"
        ]
        return make(
            INFRASTRUCTURE_ERROR,
            reason_codes=codes,
            source="candidate_integrity_leaves",
        )
    if statuses and all(status == "valid" for status in statuses):
        return make(VALID, source="candidate_integrity_leaves")

    reason_codes: list[str] = []
    owners: list[str] = []
    if snapshot.get("status") == "invalid":
        evidence = snapshot.get("evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        attribution = evidence.get("failure_attribution")
        attribution = attribution if isinstance(attribution, Mapping) else {}
        owner = str(attribution.get("owner") or "unknown")
        raw_codes = attribution.get("reason_codes")
        if isinstance(raw_codes, (list, tuple)):
            reason_codes.extend(str(code) for code in raw_codes)
        else:
            metrics = snapshot.get("metrics")
            metrics = metrics if isinstance(metrics, Mapping) else {}
            execution_failures = evidence.get("execution_failures")
            execution_failures = (
                execution_failures if isinstance(execution_failures, list) else []
            )
            if _positive_count(metrics.get("provenance_error_count")):
                owner = "evaluation_infrastructure"
                reason_codes.append("candidate_provenance_invalid")
            elif any(
                str(failure).startswith("infra_error:")
                for failure in execution_failures
            ):
                owner = "evaluation_infrastructure"
                reason_codes.append("harness_infrastructure_failure")
            else:
                model_reasons = []
                if _positive_count(metrics.get("execution_failure_count")):
                    model_reasons.append("generation_budget_exhausted")
                if _positive_count(metrics.get("schema_error_count")):
                    model_reasons.append("candidate_schema_invalid")
                actor_count = metrics.get("actor_count")
                minimum = metrics.get("minimum_actor_count")
                if (
                    isinstance(actor_count, int)
                    and isinstance(minimum, int)
                    and actor_count < minimum
                ):
                    model_reasons.append(
                        "empty_scene"
                        if actor_count == 0
                        else "candidate_below_minimum_actor_count"
                    )
                if model_reasons:
                    owner = "model"
                    reason_codes.extend(model_reasons)
                else:
                    reason_codes.append("candidate_snapshot_invalid")
        owners.append(owner)

    if parity.get("status") == "invalid":
        evidence = parity.get("evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        classification = str(
            evidence.get("dependency_classification") or "unknown"
        )
        reason_codes.append(
            str(evidence.get("reason_code") or "candidate_dependency_invalid")
        )
        owners.append(
            "model" if classification == "candidate_dependency_missing" else "mixed"
        )

    if manifest.get("status") == "invalid":
        owners.append("unknown")
        reason_codes.append("asset_library_manifest_invalid")

    if owners and all(owner == "model" for owner in owners):
        return make(
            MODEL_INVALID,
            reason_codes=reason_codes,
            source="candidate_integrity_leaves",
        )
    if owners:
        return make(
            INVALID,
            reason_codes=reason_codes,
            source="candidate_integrity_leaves",
        )
    return make(
        UNKNOWN,
        reason_codes=("candidate_integrity_status_unclassified",),
        source="candidate_integrity_leaves",
    )


def from_integrity_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read an explicit outcome, with a conservative legacy fallback."""

    if not isinstance(report, Mapping):
        return make(
            UNKNOWN,
            reason_codes=("candidate_integrity_report_missing",),
            source="candidate_integrity_report",
        )
    evidence = report.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    explicit = _explicit(evidence.get("case_outcome"))
    status = str(report.get("status") or "").casefold()
    if explicit is not None:
        if status == "invalid" and explicit["classification"] not in {MODEL_INVALID, INVALID}:
            return {**make(INVALID, reason_codes=explicit["reason_codes"],
                          source="candidate_integrity_invalid_fixed_zero"),
                    "failure_owner": explicit["failure_owner"],
                    "previous_classification": explicit["classification"]}
        return explicit

    if status == "valid":
        return make(VALID, source="candidate_integrity_status")
    if status == "error":
        return make(
            INFRASTRUCTURE_ERROR,
            reason_codes=("candidate_integrity_error",),
            source="candidate_integrity_status",
        )
    if status == "invalid":
        metrics = report.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        raw_leaves = metrics.get("leaf_results")
        leaves = raw_leaves if isinstance(raw_leaves, list) else []
        if len(leaves) >= 3 and all(isinstance(leaf, Mapping) for leaf in leaves[:3]):
            outcome = from_integrity_leaves(leaves[0], leaves[1], leaves[2])
            if outcome["classification"] in {MODEL_INVALID, INVALID}:
                return outcome
        return make(
            INVALID,
            reason_codes=("legacy_candidate_invalid_without_attribution",),
            source="candidate_integrity_status",
        )
    return make(
        UNKNOWN,
        reason_codes=("candidate_integrity_status_unclassified",),
        source="candidate_integrity_status",
    )


def from_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    integrity = next(
        (
            report
            for report in reports
            if isinstance(report, Mapping)
            and report.get("report_id") == "candidate_integrity"
        ),
        None,
    )
    return from_integrity_report(integrity)


def from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    reports = result.get("reports")
    if isinstance(reports, list):
        integrity = next((report for report in reports if isinstance(report, Mapping)
                          and report.get("report_id") == "candidate_integrity"), None)
        if integrity is not None and integrity.get("status") == "invalid":
            return from_integrity_report(integrity)
    explicit = _explicit(result.get("case_outcome"))
    if explicit is not None:
        return explicit
    reports = result.get("reports")
    reports = reports if isinstance(reports, list) else []
    return from_reports(reports)


def has_integrity_outcome(result: Mapping[str, Any]) -> bool:
    """Whether a result actually contains a Candidate Integrity decision."""

    if _explicit(result.get("case_outcome")) is not None:
        return True
    reports = result.get("reports")
    return bool(
        isinstance(reports, list)
        and any(
            isinstance(report, Mapping)
            and report.get("report_id") == "candidate_integrity"
            for report in reports
        )
    )


__all__ = [
    "INFRASTRUCTURE_ERROR",
    "MODEL_INVALID",
    "INVALID",
    "SCHEMA_VERSION",
    "UNKNOWN",
    "VALID",
    "from_integrity_leaves",
    "from_integrity_report",
    "from_reports",
    "from_result",
    "has_integrity_outcome",
    "make",
]
