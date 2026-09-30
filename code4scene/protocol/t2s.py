"""Text-to-scene case score (paper protocol, Appendix C.4 and C.6).

``S_T2S = 0.20 * S_detailed + 0.60 * S_overview + 0.20 * S_physics``

rounded to four decimals. An invalid or missing candidate scores zero, and an
unavailable required component keeps its weight and contributes zero.

Detailed Alignment (Eq. 1). Each accepted atomic decision has a score
``q_j`` in [0, 1]; an unresolved applicable requirement scores zero while
keeping its weight. Within a family ``f`` the score is the mean over the
family's non-empty prompt clauses of the mean over the clause's active
predicates::

    S_f = 1/|C_f| * sum_c 1/|R_fc| * sum_j q_j
    S_detailed = sum_f w_f S_f / sum_f w_f      (applicable families only)

(family scores enter the weighted mean at their published four-decimal
precision; the case value is rounded to four decimals)

with adopted family weights identity/environment 0.25, content/quantity
0.40, attributes/materials 0.15, spatial composition 0.20.

Overview Alignment (Eq. 2). With prompt-aware dimension scores and weights
0.40 (global prompt alignment), 0.25 (composition and layout), 0.20 (style and
atmosphere) and 0.15 (completeness and polish)::

    S_prompt = sum_d w_d s_d
    S~ = S_prompt * (0.75 + 0.25 * S_struct)
    S_overview = min(S~, 0.40)  if a corroborated severe structural defect
                                (confidence >= 0.8, >= 2 evidence views)
               = S~             otherwise

The judge outputs (atomic decisions, dimension scores, structural score) are
inputs to this module; producing them needs the VLM judge, re-aggregating
them does not.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from ..evaluation import scalar_score
from .constants import (
    DETAILED_DECIMALS,
    DETAILED_FAMILY_WEIGHTS,
    OVERVIEW_DECIMALS,
    OVERVIEW_DIMENSION_WEIGHTS,
    OVERVIEW_SEVERE_CAP,
    OVERVIEW_STRUCTURAL_FLOOR,
    OVERVIEW_STRUCTURAL_WEIGHT,
    T2S_CASE_POLICY,
    T2S_DECIMALS,
    T2S_WEIGHTS,
)

MEASURED = "measured"
#: Report statuses whose score is unavailable (zero credit at fixed weight).
UNUSABLE_STATUSES = frozenset(
    {"blocked", "error", "invalid", "not_evaluated", "refused", "unavailable"})


def usable(report: Mapping[str, Any] | None) -> bool:
    """Whether a saved verifier report may contribute its measurement."""

    return bool(report) and str(report.get("status") or "").casefold() not in UNUSABLE_STATUSES
#: Aggregation order of the Semantic verifier (it fixes float rounding ties).
FAMILY_ORDER = (
    "identity_environment", "content_quantity", "attributes_materials", "spatial_composition",
)


def _unit(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


# ---------------------------------------------------------------------------
# Detailed Alignment
# ---------------------------------------------------------------------------


def _row_score(row: Mapping[str, Any]) -> float:
    """A decision's score, at full precision when the row records it.

    Rows store ``effective_score`` at four decimals; newer rows also keep the
    unrounded value, which is used only while it still rounds to the stored one.
    """

    rounded = _unit(row.get("effective_score")) or 0.0
    exact = _unit(row.get("unrounded_effective_score"))
    return exact if exact is not None and round(exact, 4) == rounded else rounded


def detailed_from_decisions(
    decisions: Iterable[Mapping[str, Any]],
    *,
    family_weights: Mapping[str, float] = DETAILED_FAMILY_WEIGHTS,
) -> dict[str, Any]:
    """Aggregate frozen per-requirement decisions into ``S_detailed`` (Eq. 1).

    Each decision needs ``semantic_family``, ``clause_index``, ``included``,
    ``predicate_weight`` (1 for every atomic predicate) and
    ``effective_score``: the requirement's score after conjunction capping,
    with an unresolved requirement scored zero. This is the row shape the
    Semantic verifier stores as ``semantic_requirement_aggregation``; rows
    saved in the older interval format are migrated to their conservative
    (lower) endpoint first. The summation order follows the verifier, so the
    four-decimal result is reproduced exactly.
    """

    rows = [dict(r) for r in scalar_score.without_intervals(list(decisions))
            if r.get("included", True)]
    families: dict[str, dict[str, Any]] = {}
    applicable = []
    for family in FAMILY_ORDER:
        weight = float(family_weights[family])
        members = [r for r in rows if r.get("semantic_family") == family]
        clause_ids = sorted({r.get("clause_index") for r in members},
                            key=lambda v: (v is None, v if v is not None else 0))
        clause_scores = []
        for clause in clause_ids:
            group = [r for r in members if r.get("clause_index") == clause]
            denominator = sum(float(r.get("predicate_weight", 1.0)) for r in group)
            value = sum(
                float(r.get("predicate_weight", 1.0)) * _row_score(r)
                for r in group) / denominator
            known = sum(
                float(r.get("predicate_weight", 1.0)) for r in group
                if r.get("score_known", True)) / denominator
            clause_scores.append({
                "clause_index": clause, "score": value, "known_coverage": known,
                "clause_weight": max(float(r.get("predicate_weight", 1.0)) for r in group),
                "active_requirement_count": len(group)})
        if not clause_scores:
            families[family] = {"applicable": False, "configured_weight": weight,
                                "score": None, "clause_count": 0}
            continue
        cw = sum(c["clause_weight"] for c in clause_scores)
        value = sum(c["clause_weight"] * c["score"] for c in clause_scores) / cw
        known = sum(c["clause_weight"] * c["known_coverage"] for c in clause_scores) / cw
        families[family] = {
            "applicable": True, "configured_weight": weight, "score": round(value, 4),
            "unrounded_score": value, "known_coverage": round(known, 4),
            "clause_count": len(clause_scores), "active_requirement_count": len(members)}
        applicable.append((weight, value, known))
    if not applicable:
        return {"status": "not_evaluated", "score": None, "families": families}
    # Family scores are published at four decimals and the family-weighted
    # mean is taken over those published values (renormalized over the
    # applicable families), accumulated in the verifier's family order.
    denominator = sum(w for w, _, _ in applicable)
    score = 0.0
    coverage = 0.0
    for weight, value, known in applicable:
        effective = weight / denominator
        score += effective * round(value, 4)
        coverage += effective * round(known, 4)
    for value in families.values():
        if value["applicable"]:
            value["effective_weight"] = value["configured_weight"] / denominator
    return {"status": MEASURED, "score": round(score, DETAILED_DECIMALS),
            "unrounded_score": score, "known_coverage": round(coverage, 4),
            "families": families}


# ---------------------------------------------------------------------------
# Overview Alignment
# ---------------------------------------------------------------------------


def overview_from_judgement(
    dimension_scores: Mapping[str, float],
    structural_score: float,
    *,
    severe_cap_eligible: bool = False,
    dimension_weights: Mapping[str, float] = OVERVIEW_DIMENSION_WEIGHTS,
) -> dict[str, Any]:
    """Combine the prompt-aware and prompt-free judge outputs (Eq. 2)."""

    missing = sorted(set(dimension_weights) - set(dimension_scores))
    if missing:
        raise ValueError(f"overview judgement lacks dimensions: {missing}")
    values = {k: _unit(dimension_scores[k]) for k in dimension_weights}
    struct = _unit(structural_score)
    if any(v is None for v in values.values()) or struct is None:
        raise ValueError("overview dimension and structural scores must lie in [0, 1]")
    prompt = math.fsum(values[k] * w for k, w in dimension_weights.items())
    multiplier = OVERVIEW_STRUCTURAL_FLOOR + OVERVIEW_STRUCTURAL_WEIGHT * struct
    before_cap = prompt * multiplier
    score = min(before_cap, OVERVIEW_SEVERE_CAP) if severe_cap_eligible else before_cap
    return {
        "status": MEASURED, "score": round(score, OVERVIEW_DECIMALS),
        "prompt_alignment_score": round(prompt, 4),
        "structural_integrity_score": round(struct, 4),
        "structural_multiplier": round(multiplier, 4),
        "score_before_severe_cap": round(before_cap, 4),
        "severe_cap_eligible": bool(severe_cap_eligible),
        "severe_cap_applied": bool(severe_cap_eligible and before_cap > OVERVIEW_SEVERE_CAP),
    }


def overview_from_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Re-aggregate a saved ``overview_prompt_alignment`` report."""

    report = report or {}
    metrics = report.get("metrics") or {}
    dims = metrics.get("dimensions") or {}
    if not usable(report) or not dims:
        return {"status": str(report.get("status") or "missing"), "score": None}
    return overview_from_judgement(
        {k: (v or {}).get("score") for k, v in dims.items()},
        metrics.get("structural_integrity_score"),
        severe_cap_eligible=bool(metrics.get("severe_structural_cap_eligible")),
    )


def detailed_from_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Re-aggregate a saved ``semantic_requirements`` report from its decisions."""

    report = report or {}
    rows = (report.get("metrics") or {}).get("semantic_requirement_aggregation")
    if not usable(report) or not rows:
        return {"status": str(report.get("status") or "missing"), "score": None}
    return detailed_from_decisions(rows)


# ---------------------------------------------------------------------------
# Case score
# ---------------------------------------------------------------------------


def case_score(
    *,
    valid: bool,
    detailed: float | None,
    overview: float | None,
    physics: float | None,
) -> dict[str, Any]:
    """``round(0.2 * detailed + 0.2 * physics + 0.6 * overview, 4)``; invalid is 0.

    The expression order matches the reference implementation so that exact
    rounding ties resolve identically.
    """

    if not valid:
        return {"policy": T2S_CASE_POLICY, "status": "zero_invalid_candidate", "score": 0.0,
                "detailed": 0.0, "overview": 0.0, "physics": 0.0}
    d = _unit(detailed) or 0.0
    o = _unit(overview) or 0.0
    p = _unit(physics) or 0.0
    total = T2S_WEIGHTS["detailed"] * d + T2S_WEIGHTS["physics"] * p + T2S_WEIGHTS["overview"] * o
    missing = [name for name, value in
               (("detailed", detailed), ("overview", overview), ("physics", physics))
               if _unit(value) is None]
    return {"policy": T2S_CASE_POLICY,
            "status": "measured" if not missing else "measured_with_missing_zero",
            "missing_components": missing, "score": round(total, T2S_DECIMALS),
            "detailed": d, "overview": o, "physics": p}


__all__ = [
    "case_score", "detailed_from_decisions", "detailed_from_report",
    "overview_from_judgement", "overview_from_report",
]
