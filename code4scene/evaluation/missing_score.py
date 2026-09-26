"""Score missing required evidence as zero while preserving the raw reports."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from . import contracts


POLICY_ID = "required-evidence-missing-zero.v1"


def observed_score(report: Mapping[str, Any] | None) -> float | None:
    """Accept an actual measured score, including zero, without imputing it."""
    try:
        return contracts.score_for_aggregate(report) if report else None
    except ValueError:
        return None


def project(
    report: Mapping[str, Any] | None, *, path: str, recompute: bool = False,
) -> dict[str, Any]:
    """Reapply the report's recorded formula with zero for unavailable leaves.

    Only scoring composites are recombined. Atomic detector formulas and their
    evidence stay untouched. Explicit N/A and report-only children do not enter
    the denominator.
    """
    report = report or {}
    metrics = report.get("metrics") or {}
    status = report.get("status", "missing")
    observed = observed_score(report)
    result = {
        "path": path, "source_status": status, "reported_score": report.get("score"),
        "score": observed, "known_coverage": 1.0 if observed is not None else 0.0,
        "missing_evidence": [], "children": [],
    }
    if status == "not_applicable":
        return {**result, "applicable": False}

    raw_children = metrics.get("leaf_results") or []
    configured = metrics.get("configured_score_weights")
    optional = set(metrics.get("optional_score_leaf_ids") or [])
    report_only = set(metrics.get("report_only_leaf_ids") or [])
    active = {}
    for child in raw_children:
        name = str(child.get("leaf_id") or child.get("report_id"))
        contributes = (
            child.get("contributes_to_aggregate") is not False and name not in report_only
            and (not isinstance(configured, Mapping) or configured.get(name, 0) > 0)
        )
        projected = project(child, path=f"{path}.{name}")
        if name in optional:
            # Optional visual composites contribute their measured result.
            # Their diagnostic children must not cause hidden zero imputation.
            value = observed_score(child)
            projected.update(score=value, uses_observed_score=True)
            if value is None:
                contributes = False
            else:
                projected.update(known_coverage=1.0, missing_evidence=[])
        projected["contributes_to_aggregate"] = contributes
        result["children"].append(projected)
        if contributes and projected.get("applicable", True):
            active[name] = projected
    if raw_children and isinstance(configured, Mapping):
        present = {str(child.get("leaf_id") or child.get("report_id")) for child in raw_children}
        for name, weight in configured.items():
            if weight > 0 and name not in present and name not in report_only:
                child = project(None, path=f"{path}.{name}")
                if name in optional:
                    child.update(score=None, contributes_to_aggregate=False)
                else:
                    active[name] = child
                result["children"].append(child)

    missing = [item for child in active.values() for item in child["missing_evidence"]]
    # With complete evidence retain the detector's exact result, including its
    # normalization, gates and rounding.
    if observed is not None and not missing and not recompute:
        return {**result, "applicable": True}
    aggregation = metrics.get("score_aggregation") or ""
    can_recombine = active and (
        isinstance(configured, Mapping)
        or aggregation == "unweighted_macro_mean_after_score_direction_normalization"
        or "required_score_leaf_ids" in metrics
    )
    if can_recombine:
        weights = {name: float(configured[name]) if isinstance(configured, Mapping) else 1.0
                   for name in active}
        # Keep the rule/visual mixture fixed when an individual rule is N/A.
        # A wholly unavailable optional group is then excluded below.
        for group in (metrics.get("score_weight_groups") or {}).values():
            names = [name for name in group["leaf_ids"] if name in weights]
            group_total = math.fsum(weights[name] for name in names)
            if group_total > 0:
                for name in names:
                    weights[name] = float(group["weight"]) * weights[name] / group_total
        total = math.fsum(weights.values())
        weights = {name: weight / total for name, weight in weights.items()}
        scores = {name: child["score"] for name, child in active.items()}
        if aggregation.startswith("weighted_geometric_mean"):
            value = math.prod(scores[name] ** weight for name, weight in weights.items())
            precision = 4
        else:
            value = math.fsum(scores[name] * weight for name, weight in weights.items())
            prerequisites = metrics.get("prerequisite_leaf_ids") or ()
            if prerequisites:
                value *= min(scores.get(name, 0.0) for name in prerequisites)
            precision = 6 if isinstance(configured, Mapping) else 4
        result.update(
            score=round(value, precision), applicable=True,
            known_coverage=math.fsum(weights[name] * active[name]["known_coverage"] for name in active),
            effective_weights=weights, aggregation=aggregation,
            missing_evidence=missing,
        )
        return result

    reason = report.get("failure_reason") or ("report missing" if not report else "no valid score")
    result.update(
        score=0.0, applicable=True, known_coverage=0.0,
        missing_evidence=[{"path": path, "status": status, "reason": str(reason)}],
    )
    return result
