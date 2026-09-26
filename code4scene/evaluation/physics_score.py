"""Canonical leaf-wise Physics score with fixed zero-filled leaf weights."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from . import missing_score

POLICY_ID = "physics-all-leaves-zero.v2"
WEIGHTS = {"floating": 0.5, "solid_penetration": 0.5}


def project(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize direction and score every unavailable Physics leaf as zero.

    Old incomplete physical_safety reports may have score_aggregation=None.
    Their two declared leaves still use the canonical equal-weight formula.
    A complete legacy scalar without leaf evidence retains its measured value.
    Both a missing required leaf and an explicitly ``not_applicable`` leaf keep
    their fixed 0.5 weight and contribute zero. Raw verifier reports are never
    modified; the status conversion happens only in the projection copy.
    """
    prepared = deepcopy(report or {})
    metrics = prepared.setdefault("metrics", {})
    children = metrics.get("leaf_results") or []
    raw_not_applicable_leaf_ids: list[str] = []
    if children or missing_score.observed_score(prepared) is None:
        by_id = {str(c.get("leaf_id") or c.get("report_id", "").split(".")[-1]): c
                 for c in children}
        canonical = []
        for name in WEIGHTS:
            child = deepcopy(by_id.get(
                name,
                {"status": "not_evaluated", "score": None,
                 "failure_reason": "required physics leaf missing"},
            ))
            if child.get("status") == "not_applicable":
                raw_not_applicable_leaf_ids.append(name)
                child["scoring_policy_original_status"] = "not_applicable"
                child["status"] = "not_evaluated"
                child["score"] = None
                child["failure_reason"] = (
                    "explicitly not-applicable physics leaf receives zero under "
                    "the fixed-weight public scoring policy"
                )
            child["leaf_id"] = name
            child["contributes_to_aggregate"] = True
            canonical.append(child)
        metrics.update(leaf_results=canonical, configured_score_weights=dict(WEIGHTS),
                       required_score_leaf_ids=list(WEIGHTS), report_only_leaf_ids=[],
                       optional_score_leaf_ids=[],
                       score_aggregation="weighted_arithmetic_mean_after_score_direction_normalization")
    result = missing_score.project(prepared, path="physical_safety")
    result["policy_id"] = POLICY_ID
    result["raw_not_applicable_leaf_ids"] = raw_not_applicable_leaf_ids if children else []
    return result
