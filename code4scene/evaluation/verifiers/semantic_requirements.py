"""Unified RequirementGraph semantic verifier with a Stage 1--3 evidence ladder."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .. import contracts
from ..context import Context, error
from ..requirement_graph.bundle import RequirementEvaluationStatus
from ..requirement_graph.contracts import ClaimVerdict
from ..requirement_graph.pipeline import run_pipeline
from ..semantic_scoring import (
    SEMANTIC_SCORE_POLICY,
    attach_semantic_families,
    score_semantic_case,
)
from .overview_prompt_alignment import RENDER_PROTOCOL as OVERVIEW_RENDER_PROTOCOL

CLASS = "open_ended"


def _requirement_score(value: Mapping[str, Any]) -> float | None:
    evaluation_status = RequirementEvaluationStatus(value["evaluation_status"])
    if evaluation_status not in {
        RequirementEvaluationStatus.MATCH,
        RequirementEvaluationStatus.MISMATCH,
    }:
        return None
    check = value.get("check")
    raw = check.get("score") if isinstance(check, Mapping) else None
    if not isinstance(raw, bool) and isinstance(raw, (int, float)):
        score = float(raw)
        if math.isfinite(score) and 0.0 <= score <= 1.0:
            return score
    return 1.0 if evaluation_status is RequirementEvaluationStatus.MATCH else 0.0


def _semantic_prompt(result: Mapping[str, Any]) -> str:
    bundle = result.get("bundle")
    prompt = getattr(bundle, "prompt", None)
    if prompt is None:
        prompt = getattr(getattr(bundle, "graph", None), "prompt", None)
    if prompt is not None:
        return str(prompt)
    max_end = max(
        (
            int(value["source_span"][1])
            for value in result.get("requirements", ())
            if isinstance(value.get("source_span"), (list, tuple))
            and len(value["source_span"]) == 2
        ),
        default=1,
    )
    return " " * max_end


def _unplanned_score_rows(
    result: Mapping[str, Any],
    unplanned: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep unsupported Stage 0 requirements visible in the v2 interval."""

    bundle = result.get("bundle")
    prompt = _semantic_prompt(result)
    rows = []
    for index, value in enumerate(unplanned):
        node_id = str(value.get("node_id") or value.get("requirement_id") or "")
        text = str(value.get("text") or "")
        start = prompt.find(text) if text else -1
        source_span = [start, start + len(text)] if start >= 0 else [0, max(1, len(prompt))]
        population_scope = "candidate_all"
        entity_scopes: dict[str, Any] = {}
        evaluation_binding: dict[str, Any] = {}
        weight = 1.0
        try:
            binding = bundle.binding_for(node_id)
            source_span = list(binding.source_span)
            population_scope = binding.population_scope.value
            entity_scopes = {
                key: scope.value for key, scope in binding.entity_scopes.items()
            }
            weight = float(bundle.graph.effective_weights().get(node_id, 1.0))
            evaluation_binding = bundle.evaluation_for(node_id).to_dict()
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        rows.append(
            {
                "node_id": node_id or f"unplanned_{index}",
                "requirement_id": str(
                    value.get("requirement_id") or node_id or f"unplanned_{index}"
                ),
                "verdict": ClaimVerdict.UNKNOWN.value,
                "evaluation_status": RequirementEvaluationStatus.NOT_EVALUATED.value,
                "resolved_by": None,
                "unknown_reason": "unsupported",
                "rationale": "Stage 0 requirement has no declared verifier capability",
                "text": text,
                "source_span": source_span,
                "population_scope": population_scope,
                "entity_scopes": entity_scopes,
                "weight": weight,
                "evaluation_binding": evaluation_binding,
            }
        )
    return rows


def verify(context: Context) -> dict[str, Any]:
    shared_overviews = context.renders_for(OVERVIEW_RENDER_PROTOCOL)
    if shared_overviews is not None:
        # The formal overview verifier owns this addressed protocol. Semantic
        # Stage 2 may reuse the same Candidate pixels and exact stored poses,
        # avoiding a second overview camera sweep while keeping target closeups
        # and Stage 3 judgement inside the requirement verifier.
        context = replace(context, visual_renders=shared_overviews)
    try:
        result = run_pipeline(context)
    except Exception as exc:  # noqa: BLE001 - verifier failures are reports
        return error(
            "semantic_requirements", context, f"{type(exc).__name__}: {exc}"
        )

    requirements = result["requirements"]
    non_supported = result.get("non_supported_requirements") or result.get(
        "waived_requirements", []
    )
    unplanned = [
        value for value in non_supported if value.get("status") == "unplanned"
    ]
    if not requirements and not unplanned:
        return {
            **contracts.base("semantic_requirements", context.ids),
            "status": "not_evaluated",
            "score": None,
            "failure_reason": "the frozen bundle delegates every requirement to another verifier",
            "metrics": {
                "checks": [],
                "check_count": 0,
                "coverage": 0.0,
                "unplanned_requirement_count": len(unplanned),
                "capability_coverage": 0.0 if unplanned else 1.0,
            },
            "evidence": {"non_supported_requirements": non_supported},
            "artifacts": {},
            "probes_used": ("semantic_requirements",),
        }
    unplanned_rows = _unplanned_score_rows(result, unplanned)
    scoring_requirements = attach_semantic_families(
        result["bundle"].graph,
        [*requirements, *unplanned_rows],
    )
    semantic_aggregation = score_semantic_case(
        _semantic_prompt(result), scoring_requirements
    )
    semantic_audit_by_node = {
        str(value["node_id"]): value
        for value in semantic_aggregation["requirements"]
    }
    checks = []
    for value in scoring_requirements:
        verdict = ClaimVerdict.coerce(value["verdict"])
        evaluation_status = RequirementEvaluationStatus(value["evaluation_status"])
        score_audit = semantic_audit_by_node[str(value["node_id"])]
        check_status = {
            RequirementEvaluationStatus.MATCH: contracts.MEASURED,
            RequirementEvaluationStatus.MISMATCH: contracts.MEASURED,
            RequirementEvaluationStatus.NOT_EVALUATED: "not_evaluated",
            RequirementEvaluationStatus.ERROR: contracts.ERROR,
        }[evaluation_status]
        checks.append(
            {
                "id": value["requirement_id"],
                "status": check_status,
                "score": _requirement_score(value),
                "expected": {
                    "text": value["text"],
                    "source_span": value["source_span"],
                    "population_scope": value.get("population_scope"),
                    "entity_scopes": value.get("entity_scopes", {}),
                },
                "observed": {
                    "verdict": verdict.value,
                    "evaluation_status": evaluation_status.value,
                    "resolved_by": value.get("resolved_by"),
                    "unknown_reason": value.get("unknown_reason"),
                    "score_included": score_audit["included"],
                    "score_exclusion_reason": score_audit[
                        "exclusion_reason"
                    ],
                },
                "evidence": (
                    value.get("check", {}).get("evidence", [])
                    if isinstance(value.get("check"), dict)
                    else []
                ),
                "rationale": value.get("rationale"),
            }
        )

    active_scoring_requirements = [
        value
        for value in scoring_requirements
        if semantic_audit_by_node[str(value["node_id"])]["included"]
    ]
    score_excluded_requirements = [
        value
        for value in scoring_requirements
        if not semantic_audit_by_node[str(value["node_id"])]["included"]
    ]
    raw_not_evaluated = [
        value
        for value in scoring_requirements
        if value["evaluation_status"]
        == RequirementEvaluationStatus.NOT_EVALUATED.value
    ]
    raw_errors = [
        value
        for value in scoring_requirements
        if value["evaluation_status"] == RequirementEvaluationStatus.ERROR.value
    ]
    raw_decided = [
        value
        for value in scoring_requirements
        if value["evaluation_status"]
        in {
            RequirementEvaluationStatus.MATCH.value,
            RequirementEvaluationStatus.MISMATCH.value,
        }
    ]
    not_evaluated = [
        value
        for value in active_scoring_requirements
        if value["evaluation_status"]
        == RequirementEvaluationStatus.NOT_EVALUATED.value
    ]
    errors = [
        value
        for value in active_scoring_requirements
        if value["evaluation_status"] == RequirementEvaluationStatus.ERROR.value
    ]
    decided = [
        value
        for value in active_scoring_requirements
        if value["evaluation_status"]
        in {
            RequirementEvaluationStatus.MATCH.value,
            RequirementEvaluationStatus.MISMATCH.value,
        }
    ]
    matched = [
        value
        for value in decided
        if value["evaluation_status"] == RequirementEvaluationStatus.MATCH.value
    ]
    legacy_decided_score = (
        round(
            sum(float(_requirement_score(value)) for value in raw_decided)
            / len(raw_decided),
            4,
        )
        if raw_decided
        else None
    )
    coverage = (
        round(len(decided) / len(active_scoring_requirements), 4)
        if active_scoring_requirements
        else 0.0
    )
    total_weight = sum(float(value["weight"]) for value in requirements)
    supported_decided = [
        value
        for value in requirements
        if value["evaluation_status"]
        in {
            RequirementEvaluationStatus.MATCH.value,
            RequirementEvaluationStatus.MISMATCH.value,
        }
    ]
    decided_weight = sum(float(value["weight"]) for value in supported_decided)
    scored_weight = sum(
        float(value["weight"]) * float(_requirement_score(value))
        for value in supported_decided
    )
    legacy_weighted_semantic_score = (
        round(scored_weight / decided_weight, 4) if decided_weight else None
    )
    legacy_weighted_coverage = (
        round(decided_weight / total_weight, 4) if total_weight else 0.0
    )
    semantic_score = semantic_aggregation["score"]
    weighted_coverage = semantic_aggregation["known_coverage"]
    identity_grounding = result.get("identity_grounding")
    identity_bindings = (
        tuple(identity_grounding.bindings)
        if identity_grounding is not None
        else ()
    )
    structured = result.get("structured")
    structured_assessments = (
        tuple(structured.assessments)
        if structured is not None
        else ()
    )
    base = {
        **contracts.base("semantic_requirements", context.ids),
        "score": (
            semantic_score
            if decided and not result.get("visual_error")
            else None
        ),
        "metrics": {
            "checks": checks,
            "check_count": len(checks),
            "satisfied_requirement_count": len(matched),
            "stage1_resolved_count": sum(
                value.get("resolved_by") == "stage1" for value in requirements
            ),
            "structured_resolved_count": sum(
                value.get("resolved_by") == "stage1_structured"
                for value in requirements
            ),
            "stage2_resolved_count": sum(
                value.get("resolved_by") == "stage2" for value in requirements
            ),
            "stage3_resolved_count": sum(
                value.get("resolved_by") == "stage3" for value in requirements
            ),
            "identity_grounded_requirement_count": sum(
                value.get("resolved_by") == "identity_grounding"
                for value in requirements
            ),
            "identity_bound_scene_graph_count": sum(
                value.get("resolved_by") == "identity_bound_scene_graph"
                for value in requirements
            ),
            "identity_binding_count": len(identity_bindings),
            "exact_identity_actor_count": sum(
                len(value.exact_actor_ids) for value in identity_bindings
            ),
            "visual_identity_actor_count": sum(
                len(value.visual_actor_ids) for value in identity_bindings
            ),
            "identity_capture_count": (
                identity_grounding.capture_count
                if identity_grounding is not None
                else 0
            ),
            "identity_judge_call_count": (
                identity_grounding.judge_call_count
                if identity_grounding is not None
                else 0
            ),
            "not_evaluated_count": len(not_evaluated),
            "unknown_count": len(not_evaluated) + len(errors),
            "error_count": len(errors),
            "decided_requirement_count": len(decided),
            "raw_not_evaluated_count": len(raw_not_evaluated),
            "raw_unknown_count": len(raw_not_evaluated) + len(raw_errors),
            "raw_error_count": len(raw_errors),
            "raw_decided_requirement_count": len(raw_decided),
            "score_excluded_not_evaluated_count": sum(
                value["evaluation_status"]
                == RequirementEvaluationStatus.NOT_EVALUATED.value
                for value in score_excluded_requirements
            ),
            "semantic_score": semantic_score,
            "coverage": coverage,
            "weighted_semantic_score": semantic_score,
            "weighted_coverage": weighted_coverage,
            "known_coverage": weighted_coverage,
            "score_policy": SEMANTIC_SCORE_POLICY,
            "semantic_family_score_policy": semantic_aggregation[
                "family_score_policy"
            ],
            "semantic_family_weights": semantic_aggregation["family_weights"],
            "semantic_subscores": semantic_aggregation["families"],
            "legacy_decided_only_semantic_score": legacy_decided_score,
            "legacy_weighted_decided_score": legacy_weighted_semantic_score,
            "legacy_weighted_decided_coverage": legacy_weighted_coverage,
            "source_requirement_count": semantic_aggregation[
                "source_requirement_count"
            ],
            "active_requirement_count": semantic_aggregation[
                "active_requirement_count"
            ],
            "excluded_requirement_count": semantic_aggregation[
                "excluded_requirement_count"
            ],
            "pre_evidence_excluded_requirement_count": len(
                result.get("pre_evidence_exclusions", {})
            ),
            "parent_adjustment_count": semantic_aggregation[
                "parent_adjustment_count"
            ],
            "semantic_clause_scores": semantic_aggregation["clauses"],
            "semantic_requirement_aggregation": semantic_aggregation[
                "requirements"
            ],
            "partial_score": bool(decided) and (
                weighted_coverage < 1.0 or bool(errors) or bool(unplanned)
            ),
            "unplanned_requirement_count": len(unplanned),
            "capability_coverage": round(
                len(requirements) / (len(requirements) + len(unplanned)), 4
            ),
        },
        "evidence": {
            **result["scene_evidence"].evidence(),
            "bundle_id": result["bundle"].bundle_id,
            "task_mode": result["bundle"].task_mode.value,
            "delegated_requirements": result["delegated_requirements"],
            # Bindings the bundle declares out of the score. Recorded here so
            # the reader can see the denominator shrank by declaration, not
            # by accident.
            "waived_requirements": result["waived_requirements"],
            "non_supported_requirements": non_supported,
            "visual_error": result["visual_error"],
            "pre_evidence_exclusions": result.get(
                "pre_evidence_exclusions", {}
            ),
        },
        "artifacts": result["artifacts"],
        "probes_used": tuple(
            dict.fromkeys(
                [
                    *result["scene_evidence"].probes_used(),
                    "requirement_graph_stage1",
                    *(
                        ["requirement_graph_structured"]
                        if structured_assessments
                        else []
                    ),
                    *(
                        ["requirement_graph_stage2"]
                        if any(value.get("resolved_by") == "stage2" for value in requirements)
                        else []
                    ),
                    *(
                        ["requirement_graph_stage3"]
                        if any(value.get("resolved_by") == "stage3" for value in requirements)
                        else []
                    ),
                    *(
                        ["semantic_identity_grounding"]
                        if identity_grounding is not None
                        else []
                    ),
                ]
            )
        ),
    }
    stress_policy = str(
        getattr(context, "spec", {}).get("semantic_stress_policy") or ""
    ).strip().casefold()
    if stress_policy == "overview-holistic":
        stage3 = result.get("stage3")
        holistic = stage3.get("holistic") if isinstance(stage3, Mapping) else None
        holistic = holistic if isinstance(holistic, Mapping) else {}
        raw_score = holistic.get("overall_score")
        holistic_score = (
            float(raw_score)
            if not isinstance(raw_score, bool)
            and isinstance(raw_score, (int, float))
            and math.isfinite(float(raw_score))
            and 0.0 <= float(raw_score) <= 1.0
            else None
        )
        holistic_error = holistic.get("evaluation_error") or holistic.get("error")
        base["score"] = (
            round(holistic_score, 4) if holistic_score is not None else None
        )
        base["metrics"].update(
            {
                "score_policy": stress_policy,
                "holistic_overview_score": base["score"],
                "holistic_dimensions": holistic.get("dimensions") or [],
                "atomic_semantic_score": semantic_score,
                "atomic_coverage": coverage,
            }
        )
        base["evidence"].update(
            {
                "semantic_stress_policy": stress_policy,
                "holistic_status": holistic.get("status"),
                "holistic_summary": holistic.get("summary"),
                "atomic_results_are_diagnostic": True,
            }
        )
        if holistic_score is not None and not holistic_error:
            return {**base, "status": contracts.MEASURED}
        return {
            **base,
            "status": contracts.ERROR if holistic_error else "not_evaluated",
            "failure_reason": str(
                holistic_error
                or "overview-holistic did not produce a valid holistic score"
            ),
        }
    unresolved = [*errors, *not_evaluated]
    if result.get("visual_error"):
        return {
            **base,
            "status": contracts.ERROR,
            "score": None,
            "failure_reason": str(result["visual_error"]),
        }
    if not decided:
        reasons = [
            f"{value['requirement_id']}: "
            f"{value.get('evaluation_status')} "
            f"[{value.get('unknown_reason') or 'unspecified'}] "
            f"({value.get('rationale')})"
            for value in unresolved[:4]
        ]
        return {
            **base,
            "status": "not_evaluated",
            "failure_reason": (
                "; ".join(reasons)
                or "no semantic requirement produced a MATCH/MISMATCH decision"
            ),
        }
    return {**base, "status": contracts.MEASURED}


__all__ = ["verify"]
