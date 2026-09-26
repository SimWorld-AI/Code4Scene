"""Helpers for canonical composite verifiers.

A composite owns orchestration and status visibility, never a hidden average.
Every child report is retained verbatim as a named leaf. When all applicable
children were evaluated, the composite publishes their transparent unweighted
macro mean as a convenience summary and retains the complete score vector.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from . import contracts
from .context import Context


Verifier = Callable[[Context], dict[str, Any]]


def iter_report_tree(
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return a composite and all nested leaf reports in public-address order."""

    values: list[dict[str, Any]] = [dict(report)]
    leaves = (report.get("metrics") or {}).get("leaf_results") or []
    if not isinstance(leaves, list):
        return tuple(values)
    for leaf in leaves:
        if isinstance(leaf, Mapping):
            values.extend(iter_report_tree(leaf))
    return tuple(values)


def run_leaf(
    context: Context,
    verifier: Verifier,
    *,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    child = replace(context, spec=dict(spec))
    try:
        result = verifier(child)
    except Exception as exc:  # noqa: BLE001 - every leaf must be visible
        name = getattr(verifier, "__module__", "leaf").rsplit(".", 1)[-1]
        result = {
            **contracts.base(name, context.ids),
            "status": contracts.ERROR,
            "score": None,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "metrics": {},
            "evidence": {},
            "artifacts": {},
            "probes_used": (name,),
        }
    status = result.get("status")
    if status in contracts.WITHHELD_STATUSES and result.get("score") is not None:
        result = {**result, "score": None}
    return result


def _status(leaves: Sequence[dict[str, Any]]) -> str:
    statuses = [str(value.get("status")) for value in leaves]
    if contracts.ERROR in statuses:
        return contracts.ERROR
    if contracts.INVALID in statuses:
        return contracts.INVALID
    if statuses and all(status == contracts.VALID for status in statuses):
        return contracts.VALID
    if any(
        status in {contracts.MEASURED, contracts.PASS, contracts.FAIL}
        for status in statuses
    ):
        return contracts.MEASURED
    if "not_evaluated" in statuses:
        return "not_evaluated"
    return "not_applicable"


def _reason(leaves: Sequence[dict[str, Any]], status: str) -> str | None:
    if status in {contracts.PASS, contracts.MEASURED, contracts.VALID}:
        return None
    reasons = [
        f"{value['leaf_id']}: "
        f"{value.get('failure_reason') or value.get('status')}"
        for value in leaves
        if value.get("status") not in {
            contracts.PASS,
            contracts.MEASURED,
            contracts.VALID,
        }
    ]
    if reasons:
        return "; ".join(reasons[:6])
    return "no leaf was applicable under the frozen evaluation policy"


def _contributes_to_aggregate(value: Mapping[str, Any]) -> bool:
    """Whether a diagnostic leaf participates in score and parent status.

    Report-only leaves remain fully visible in ``leaf_results`` and the raw
    ``score_vector``. They do not vote on the composite score or turn an
    otherwise measurable composite into an error while their calibration is
    still being studied.
    """
    return value.get("contributes_to_aggregate") is not False


def composite_report(
    report_id: str,
    context: Context,
    leaves: Sequence[dict[str, Any]],
    *,
    evidence: Mapping[str, Any] | None = None,
    require_all_scored: bool = False,
    exclude_not_applicable_from_required_scores: bool = False,
) -> dict[str, Any]:
    values = [dict(value) for value in leaves]
    leaf_ids: list[str] = []
    for index, value in enumerate(values):
        leaf_id = value.get("leaf_id")
        if not isinstance(leaf_id, str) or not leaf_id.strip():
            raise ValueError(
                f"{report_id}: leaf {index} needs a non-empty stable leaf_id"
            )
        if leaf_id in leaf_ids:
            raise ValueError(f"{report_id}: duplicate leaf_id {leaf_id!r}")
        leaf_ids.append(leaf_id)
        # The public address is canonical and parent-owned. A child verifier's
        # old top-level report name must never leak into a canonical composite.
        value["report_id"] = f"{report_id}.{leaf_id}"
    aggregate_values = [
        value for value in values if _contributes_to_aggregate(value)
    ]
    excluded_not_applicable_values = [
        value
        for value in aggregate_values
        if (
            exclude_not_applicable_from_required_scores
            and value.get("status") == "not_applicable"
        )
    ]
    required_aggregate_values = [
        value
        for value in aggregate_values
        if not (
            exclude_not_applicable_from_required_scores
            and value.get("status") == "not_applicable"
        )
    ]
    status = _status(aggregate_values)
    scored_values = [
        value
        for value in aggregate_values
        if value.get("status") in {contracts.MEASURED, contracts.PASS, contracts.FAIL}
        and isinstance(value.get("score"), (int, float))
        and not isinstance(value.get("score"), bool)
    ]
    numeric_scores = [
        score
        for value in scored_values
        if (score := contracts.score_for_aggregate(value)) is not None
    ]
    all_required_scores_present = bool(required_aggregate_values) and (
        len(numeric_scores) == len(required_aggregate_values)
    )
    composite_score = (
        round(sum(numeric_scores) / len(numeric_scores), 4)
        if status in {contracts.MEASURED, contracts.PASS, contracts.FAIL}
        and numeric_scores
        and (not require_all_scored or all_required_scores_present)
        else None
    )
    applicable_quality_leaves = [
        value
        for value in aggregate_values
        if value.get("status")
        not in {"not_applicable", contracts.VALID, contracts.INVALID}
    ]
    completeness_values = (
        required_aggregate_values
        if require_all_scored
        else applicable_quality_leaves
    )
    incomplete_leaf_ids = [
        value["leaf_id"]
        for value in completeness_values
        if not (
            value.get("status") in {
                contracts.MEASURED,
                contracts.PASS,
                contracts.FAIL,
            }
            and isinstance(value.get("score"), (int, float))
            and not isinstance(value.get("score"), bool)
        )
    ]
    counts = {
        name: sum(value.get("status") == name for value in values)
        for name in (
            contracts.PASS,
            contracts.FAIL,
            contracts.MEASURED,
            contracts.VALID,
            contracts.INVALID,
            contracts.ERROR,
            "not_evaluated",
            "not_applicable",
        )
    }
    result = {
        **contracts.base(report_id, context.ids),
        "status": status,
        "score": composite_score,
        "metrics": {
            "leaf_results": values,
            "leaf_count": len(values),
            "status_counts": counts,
            "score_vector": {
                value["leaf_id"]: value.get("score")
                for value in values
                if value.get("score") is not None
            },
            "aggregate_score_vector": {
                value["leaf_id"]: value.get("score")
                for value in aggregate_values
                if value.get("score") is not None
            },
            "normalized_aggregate_score_vector": {
                value["leaf_id"]: contracts.score_for_aggregate(value)
                for value in scored_values
            },
            "leaf_score_directions": {
                value["leaf_id"]: (
                    (value.get("metadata") or {}).get(
                        "score_direction", "higher_is_better"
                    )
                )
                for value in values
                if value.get("score") is not None
            },
            "report_only_leaf_ids": [
                value["leaf_id"]
                for value in values
                if not _contributes_to_aggregate(value)
            ],
            "score_aggregation": (
                "unweighted_macro_mean_after_score_direction_normalization"
                if composite_score is not None
                else None
            ),
            "scored_leaf_count": len(numeric_scores),
            "requires_all_scored_leaves": require_all_scored,
            "exclude_not_applicable_from_required_scores": (
                exclude_not_applicable_from_required_scores
            ),
            "required_score_leaf_ids": [
                value["leaf_id"] for value in required_aggregate_values
            ],
            "excluded_not_applicable_leaf_ids": [
                value["leaf_id"] for value in excluded_not_applicable_values
            ],
            "all_required_scores_present": all_required_scores_present,
            "applicable_quality_leaf_count": len(applicable_quality_leaves),
            "score_coverage": (
                round(
                    len(numeric_scores) / len(applicable_quality_leaves),
                    4,
                )
                if applicable_quality_leaves
                else None
            ),
            "incomplete_leaf_ids": incomplete_leaf_ids,
        },
        "evidence": dict(evidence or {}),
        "artifacts": {
            f"{value['leaf_id']}.{key}": str(path)
            for value in values
            for key, path in (value.get("artifacts") or {}).items()
        },
        "probes_used": tuple(
            dict.fromkeys(
                probe
                for value in values
                for probe in (value.get("probes_used") or ())
            )
        ),
    }
    reason = _reason(aggregate_values, status)
    if reason is not None:
        result["failure_reason"] = reason
    return result


def not_applicable_report(
    report_id: str,
    context: Context,
    reason: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **contracts.base(report_id, context.ids),
        "status": "not_applicable",
        "score": None,
        "failure_reason": reason,
        "metrics": {
            "leaf_results": [],
            "leaf_count": 0,
            "status_counts": {"not_applicable": 1},
        },
        "evidence": dict(evidence or {}),
        "artifacts": {},
        "probes_used": (),
    }


def blocked_report(
    report_id: str,
    context: Context,
    gate: Mapping[str, Any],
    *,
    eligibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    eligibility = dict(eligibility or {})
    reason = eligibility.get("reason")
    return {
        **contracts.base(report_id, context.ids),
        "status": "not_evaluated",
        "score": None,
        "failure_reason": (
            str(reason)
            if reason
            else "candidate_integrity did not authorize this scene-dependent "
            "verifier, so invalid or unavailable evidence cannot produce a score"
        ),
        "metrics": {"leaf_results": [], "leaf_count": 0},
        "evidence": {
            "blocked_by": "candidate_integrity",
            "gate_status": gate.get("status"),
            "gate_failure_reason": gate.get("failure_reason"),
            "gate_mode": (gate.get("evidence") or {}).get("gate_mode"),
            "eligibility": eligibility or None,
            "reason_code": eligibility.get("reason_code"),
            "source_leaf": eligibility.get("source_leaf"),
        },
        "artifacts": {},
        "probes_used": (),
    }


__all__ = [
    "blocked_report",
    "composite_report",
    "iter_report_tree",
    "not_applicable_report",
    "run_leaf",
]
