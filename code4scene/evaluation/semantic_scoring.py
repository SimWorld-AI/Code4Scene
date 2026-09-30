"""Case-normalized semantic scoring for RequirementGraph results.

The atomic judge answers one candidate on one frozen task.  This module turns
those answers into a case score without rewarding parser verbosity and turns
case scores into a suite score without pooling their requirement rows.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any
from .scalar_score import without_intervals


SEMANTIC_SCORE_POLICY = "case-family-clause-macro-missing-zero.v4"
SEMANTIC_FAMILY_SCORE_POLICY = "family-clause-macro-equal-atomic"
CASE_AGGREGATE_POLICY = "weighted-macro-mean-of-case-scores"

# Atomic predicates are deliberately equal inside a semantic family.  Any
# relative importance belongs at the family layer, where it can be calibrated
# against the corresponding human judgement dimension without conflating
# parser predicate types with evaluation priorities.
PREDICATE_WEIGHTS: dict[str, float] = {
    "scene_identity": 1.0,
    "atmosphere": 1.0,
    "environment": 1.0,
    "spatial_relation": 1.0,
    "distribution": 1.0,
    "composition": 1.0,
    "boundary": 1.0,
    "style_bundle": 1.0,
    "count": 1.0,
    "quantity": 1.0,
    "set": 1.0,
    "logic": 1.0,
    "material": 1.0,
    "attribute": 1.0,
    "existence": 1.0,
}
DEFAULT_PREDICATE_WEIGHT = 1.0

SEMANTIC_FAMILY_ORDER = (
    "identity_environment",
    "content_quantity",
    "attributes_materials",
    "spatial_composition",
)
SEMANTIC_FAMILY_PREDICATE_TYPES: dict[str, frozenset[str]] = {
    "identity_environment": frozenset(
        {"scene_identity", "atmosphere", "environment", "style_bundle"}
    ),
    "content_quantity": frozenset({"existence", "count", "quantity", "set"}),
    "attributes_materials": frozenset({"attribute", "material"}),
    "spatial_composition": frozenset(
        {"spatial_relation", "distribution", "composition", "boundary"}
    ),
}
PREDICATE_FAMILY: dict[str, str] = {
    predicate_type: family
    for family, predicate_types in SEMANTIC_FAMILY_PREDICATE_TYPES.items()
    for predicate_type in predicate_types
}

# These weights are intentionally equal until human calibration supplies a
# pre-registered alternative.  Their scale is arbitrary; only ratios matter.
SEMANTIC_FAMILY_WEIGHTS: dict[str, float] = {
    family: 1.0 for family in SEMANTIC_FAMILY_ORDER
}

# These nodes normally restate their scored descendants.  Once descendants
# exist they remain in the audit but do not receive another independent vote.
REPLACED_COMPOSITE_TYPES = frozenset({"scene_identity", "set", "logic"})

# A conjunctive parent's positive score cannot hide a failed or unknown child.
CONJUNCTIVE_TYPES = frozenset(
    {
        "scene_identity",
        "spatial_relation",
        "distribution",
        "composition",
        "boundary",
        "set",
        "logic",
    }
)

_KNOWN_STATUSES = frozenset({"MATCH", "MISMATCH"})
_UNKNOWN_STATUSES = frozenset({"ERROR", "NOT_EVALUATED"})


def _round4(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value)).strip()


def _identifier(value: Mapping[str, Any], index: int) -> str:
    return str(
        value.get("node_id")
        or value.get("requirement_id")
        or value.get("id")
        or f"requirement_{index}"
    )


def _source_span(
    value: Mapping[str, Any],
    *,
    prompt_length: int,
) -> tuple[int, int]:
    raw = value.get("source_span")
    if isinstance(raw, Mapping):
        raw = (raw.get("start"), raw.get("end"))
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in raw)
    ):
        start, end = int(raw[0]), int(raw[1])
        if 0 <= start < end:
            return (start, end)
    return (0, max(1, prompt_length))


def _predicate_type(value: Mapping[str, Any]) -> str:
    raw = value.get("predicate_type")
    binding = value.get("evaluation_binding")
    if raw is None and isinstance(binding, Mapping):
        parameters = binding.get("parameters")
        if isinstance(parameters, Mapping):
            raw = parameters.get("predicate_type")
    normalized = _enum_text(raw or "unknown").casefold()
    return "existence" if normalized == "presence" else normalized


def _semantic_family(
    value: Mapping[str, Any], predicate_type: str
) -> str | None:
    raw = value.get("semantic_family")
    if raw is not None:
        normalized = _enum_text(raw).casefold()
        if normalized not in SEMANTIC_FAMILY_ORDER:
            raise ValueError(f"unknown semantic family: {raw!r}")
        return normalized
    return PREDICATE_FAMILY.get(predicate_type)


def semantic_family_map_for_graph(graph: Any) -> dict[str, str]:
    """Resolve graph predicates to the four human-aligned semantic families.

    Ordinary predicate types have a direct mapping.  ``logic`` predicates use
    their explicit scope descendants and are assigned only when every mapped
    descendant belongs to the same family.  Cross-family logic wrappers remain
    unassigned so they cannot silently double-count multiple family scores.
    """

    def field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    raw_nodes = field(graph, "nodes", ()) or ()
    raw_edges = field(graph, "edges", ()) or ()
    predicate_types: dict[str, str] = {}
    for node in raw_nodes:
        node_id = field(node, "id")
        predicate_type = field(node, "predicate_type")
        if node_id is None or predicate_type is None:
            continue
        predicate_types[str(node_id)] = _enum_text(predicate_type).casefold()

    scope_children: dict[str, list[str]] = {}
    for edge in raw_edges:
        if _enum_text(field(edge, "edge_type", "")).casefold() != "scope":
            continue
        source_id = field(edge, "source_id")
        target_id = field(edge, "target_id")
        if source_id is None or target_id is None:
            continue
        scope_children.setdefault(str(source_id), []).append(str(target_id))

    cache: dict[str, frozenset[str]] = {}

    def families_for(node_id: str, visiting: frozenset[str]) -> frozenset[str]:
        if node_id in cache:
            return cache[node_id]
        if node_id in visiting:
            return frozenset()
        predicate_type = predicate_types.get(node_id)
        direct = PREDICATE_FAMILY.get(predicate_type or "")
        if direct is not None:
            result = frozenset({direct})
        elif predicate_type == "logic":
            values: set[str] = set()
            for child_id in scope_children.get(node_id, ()):
                values.update(families_for(child_id, visiting | {node_id}))
            result = frozenset(values)
        else:
            result = frozenset()
        cache[node_id] = result
        return result

    result: dict[str, str] = {}
    for node_id in predicate_types:
        families = families_for(node_id, frozenset())
        if len(families) == 1:
            result[node_id] = next(iter(families))
    return result


def attach_semantic_families(
    graph: Any,
    requirements: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy score rows and attach graph-derived family metadata."""

    def field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    family_by_node = semantic_family_map_for_graph(graph)
    scoped_parents = {
        str(field(edge, "source_id"))
        for edge in (field(graph, "edges", ()) or ())
        if _enum_text(field(edge, "edge_type", "")).casefold() == "scope"
        and field(edge, "source_id") is not None
    }
    result: list[dict[str, Any]] = []
    for index, value in enumerate(requirements):
        row = dict(value)
        node_id = _identifier(value, index)
        family = family_by_node.get(node_id)
        if family is not None:
            row["semantic_family"] = family
        if node_id in scoped_parents:
            row["semantic_replaced_by_children"] = True
        result.append(row)
    return result


def _dedup_identity(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Stable semantic target cues used only to prove an exact duplicate."""

    result: list[str] = []
    binding = value.get("evaluation_binding")
    if isinstance(binding, Mapping):
        parameters = binding.get("parameters")
        if isinstance(parameters, Mapping):
            for key, item in sorted(parameters.items()):
                if key != "predicate_type":
                    result.append(f"parameter:{key}={item!r}")
        binding_scopes = binding.get("entity_scopes")
        if isinstance(binding_scopes, Mapping):
            result.extend(
                f"binding_scope:{key}={item!r}"
                for key, item in sorted(binding_scopes.items())
            )
    entity_scopes = value.get("entity_scopes")
    if isinstance(entity_scopes, Mapping):
        result.extend(
            f"scope:{key}={item!r}" for key, item in sorted(entity_scopes.items())
        )
    return tuple(result)


def _evaluation_status(value: Mapping[str, Any]) -> str:
    raw = value.get("evaluation_status", value.get("verdict", "NOT_EVALUATED"))
    normalized = _enum_text(raw).upper()
    if normalized in _KNOWN_STATUSES | _UNKNOWN_STATUSES:
        return normalized
    return "NOT_EVALUATED"


def _measured_score(value: Mapping[str, Any], status: str) -> float | None:
    if status not in _KNOWN_STATUSES:
        return None
    candidates: list[Any] = [value.get("score")]
    check = value.get("check")
    if isinstance(check, Mapping):
        candidates.append(check.get("score"))
    for raw in candidates:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        score = float(raw)
        if math.isfinite(score) and 0.0 <= score <= 1.0:
            return score
    return 1.0 if status == "MATCH" else 0.0


def _strictly_contains(parent: Mapping[str, Any], child: Mapping[str, Any]) -> bool:
    return bool(
        parent["index"] != child["index"]
        and parent["start"] <= child["start"]
        and parent["end"] >= child["end"]
        and (
            parent["start"] < child["start"]
            or parent["end"] > child["end"]
        )
    )


def _direct_children(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {int(row["index"]): [] for row in rows}
    for parent in rows:
        candidates = [child for child in rows if _strictly_contains(parent, child)]
        for child in candidates:
            has_intermediate = any(
                other["index"] not in {parent["index"], child["index"]}
                and _strictly_contains(parent, other)
                and _strictly_contains(other, child)
                for other in rows
            )
            if not has_intermediate:
                children[int(parent["index"])].append(int(child["index"]))
    return children


def _clauses(prompt: str, max_end: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for match in re.finditer(r"[^.!?;]+(?:[.!?;]+|$)", prompt):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        start = match.start() + leading
        end = match.start() + trailing
        if end > start:
            result.append(
                {
                    "index": len(result),
                    "start": start,
                    "end": end,
                    "text": prompt[start:end],
                }
            )
    if not result:
        result.append(
            {
                "index": 0,
                "start": 0,
                "end": max(1, len(prompt), max_end),
                "text": prompt,
            }
        )
    return result


def _assign_clause(row: Mapping[str, Any], clauses: Sequence[Mapping[str, Any]]) -> int:
    def overlap(clause: Mapping[str, Any]) -> int:
        return max(
            0,
            min(int(row["end"]), int(clause["end"]))
            - max(int(row["start"]), int(clause["start"])),
        )

    return int(max(clauses, key=lambda clause: (overlap(clause), -clause["index"]))["index"])


def score_semantic_case(
    prompt: str,
    requirements: Sequence[Mapping[str, Any]],
    *,
    family_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return a conservative, parser-density-resistant score for one case.

    ``ERROR`` and ``NOT_EVALUATED`` receive zero without leaving the
    denominator. Evidence coverage is diagnostic and never rescales scores.
    """

    prompt = str(prompt or "")
    configured_family_weights = dict(SEMANTIC_FAMILY_WEIGHTS)
    if family_weights is not None:
        unknown = set(family_weights) - set(SEMANTIC_FAMILY_ORDER)
        if unknown:
            raise ValueError(f"unknown semantic family weights: {sorted(unknown)}")
        configured_family_weights.update(family_weights)
    for family, raw_weight in configured_family_weights.items():
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"semantic family weight for {family} must be numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"semantic family weight for {family} must be positive and finite"
            )
        configured_family_weights[family] = weight

    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(requirements):
        if not isinstance(raw, Mapping):
            raise TypeError("requirements must contain mapping values")
        status = _evaluation_status(raw)
        start, end = _source_span(raw, prompt_length=len(prompt))
        predicate_type = _predicate_type(raw)
        score = _measured_score(raw, status)
        rows.append(
            {
                "index": index,
                "node_id": _identifier(raw, index),
                "text": str(raw.get("text") or ""),
                "start": start,
                "end": end,
                "predicate_type": predicate_type,
                "semantic_family": _semantic_family(raw, predicate_type),
                "graph_replaced_by_children": bool(
                    raw.get("semantic_replaced_by_children", False)
                ),
                "dedup_identity": _dedup_identity(raw),
                "predicate_weight": PREDICATE_WEIGHTS.get(
                    predicate_type, DEFAULT_PREDICATE_WEIGHT
                ),
                "evaluation_status": status,
                "base_score": float(score) if score is not None else 0.0,
                "score_known": score is not None,
            }
        )

    if not rows:
        return {
            "schema_version": "scenebenchmark-semantic-case-score.v4",
            "score_policy": SEMANTIC_SCORE_POLICY,
            "family_score_policy": SEMANTIC_FAMILY_SCORE_POLICY,
            "family_weights": configured_family_weights,
            "score": None,
            "known_coverage": 0.0,
            "source_requirement_count": 0,
            "active_requirement_count": 0,
            "excluded_requirement_count": 0,
            "parent_adjustment_count": 0,
            "clauses": [],
            "families": {},
            "requirements": [],
        }

    by_index = {int(row["index"]): row for row in rows}
    child_map = _direct_children(rows)
    exclusion_reasons: dict[int, str] = {}
    exact: dict[tuple[int, int, str, str, tuple[str, ...]], int] = {}
    for row in rows:
        normalized_text = re.sub(r"\s+", " ", row["text"].strip().casefold())
        key = (
            int(row["start"]),
            int(row["end"]),
            normalized_text,
            str(row["predicate_type"]),
            tuple(row["dedup_identity"]),
        )
        if key in exact:
            exclusion_reasons[int(row["index"])] = "exact_duplicate"
        else:
            exact[key] = int(row["index"])

    for row in rows:
        index = int(row["index"])
        if (
            index not in exclusion_reasons
            and (child_map[index] or row["graph_replaced_by_children"])
            and row["predicate_type"] in REPLACED_COMPOSITE_TYPES
        ):
            exclusion_reasons[index] = "composite_replaced_by_children"

    for row in rows:
        index = int(row["index"])
        if index not in exclusion_reasons and row["semantic_family"] is None:
            exclusion_reasons[index] = "unassigned_semantic_family"

    active = [row for row in rows if row["index"] not in exclusion_reasons]
    if not active:
        raise ValueError("no active semantic requirement maps to a declared family")
    # Only requirements that are themselves scored can cap a conjunction.
    active_children = _direct_children(active)

    score_cache: dict[int, tuple[float, bool]] = {}

    def effective_score(index: int) -> tuple[float, bool]:
        if index in score_cache:
            return score_cache[index]
        row = by_index[index]
        score = float(row["base_score"])
        known = bool(row["score_known"])
        children = active_children.get(index, [])
        if children and row["predicate_type"] in CONJUNCTIVE_TYPES:
            values = [(score, known), *(effective_score(child) for child in children)]
            score = min(value for value, _ in values)
            # A confirmed false conjunct decides the conjunction even if
            # another conjunct is unknown. No optimistic score is calculated.
            known = all(resolved for _, resolved in values) or any(
                resolved and value == 0.0 for value, resolved in values
            )
        score_cache[index] = (score, known)
        return score_cache[index]

    clauses = _clauses(prompt, max(int(row["end"]) for row in rows))
    for row in active:
        clause_index = _assign_clause(row, clauses)
        row["clause_index"] = clause_index

    clause_scores: list[dict[str, Any]] = []
    family_scores: dict[str, dict[str, Any]] = {}
    applicable_families: list[dict[str, Any]] = []
    for family in SEMANTIC_FAMILY_ORDER:
        family_rows = [row for row in active if row["semantic_family"] == family]
        family_clause_scores: list[dict[str, Any]] = []
        for clause in clauses:
            clause_members = [
                row
                for row in family_rows
                if row.get("clause_index") == int(clause["index"])
            ]
            if not clause_members:
                continue
            denominator = sum(
                float(row["predicate_weight"]) for row in clause_members
            )
            clause_value = sum(
                float(row["predicate_weight"])
                * effective_score(int(row["index"]))[0]
                for row in clause_members
            ) / denominator
            clause_known = sum(
                float(row["predicate_weight"])
                for row in clause_members
                if effective_score(int(row["index"]))[1]
            ) / denominator
            clause_score = {
                "index": int(clause["index"]),
                "text": clause["text"],
                "semantic_family": family,
                "clause_weight": max(
                    float(row["predicate_weight"]) for row in clause_members
                ),
                "score": clause_value,
                "known_coverage": clause_known,
                "active_requirement_count": len(clause_members),
            }
            family_clause_scores.append(clause_score)
            clause_scores.append(clause_score)

        family_weight = float(configured_family_weights[family])
        if not family_clause_scores:
            family_scores[family] = {
                "score_policy": SEMANTIC_FAMILY_SCORE_POLICY,
                "family_weight": family_weight,
                "applicable": False,
                "score": None,
                "known_coverage": 0.0,
                "active_requirement_count": 0,
                "clause_count": 0,
                "clauses": [],
            }
            continue

        family_clause_denominator = sum(
            float(value["clause_weight"]) for value in family_clause_scores
        )
        family_value = sum(
            float(value["clause_weight"]) * float(value["score"])
            for value in family_clause_scores
        ) / family_clause_denominator
        family_coverage = sum(
            float(value["clause_weight"]) * float(value["known_coverage"])
            for value in family_clause_scores
        ) / family_clause_denominator
        family_result = {
            "score_policy": SEMANTIC_FAMILY_SCORE_POLICY,
            "family_weight": family_weight,
            "applicable": True,
            "score": _round4(family_value),
            "known_coverage": _round4(family_coverage),
            "active_requirement_count": len(family_rows),
            "clause_count": len(family_clause_scores),
            "clauses": family_clause_scores,
        }
        family_scores[family] = family_result
        applicable_families.append(
            {
                "family_weight": family_weight,
                "score": family_value,
                "known_coverage": family_coverage,
            }
        )

    family_denominator = sum(
        float(value["family_weight"]) for value in applicable_families
    )
    score = sum(
        float(value["family_weight"]) * float(value["score"])
        for value in applicable_families
    ) / family_denominator
    coverage = sum(
        float(value["family_weight"]) * float(value["known_coverage"])
        for value in applicable_families
    ) / family_denominator

    audit_rows = []
    parent_adjustments = 0
    for row in rows:
        index = int(row["index"])
        effective_value, effective_known = effective_score(index)
        constrained = not (
            math.isclose(effective_value, float(row["base_score"]), abs_tol=1e-12)
            and effective_known == row["score_known"]
        )
        included = index not in exclusion_reasons
        if included and constrained:
            parent_adjustments += 1
        audit_rows.append(
            {
                "node_id": row["node_id"],
                "text": row["text"],
                "source_span": [row["start"], row["end"]],
                "predicate_type": row["predicate_type"],
                "semantic_family": row["semantic_family"],
                "predicate_weight": row["predicate_weight"],
                "evaluation_status": row["evaluation_status"],
                "effective_score": _round4(effective_value),
                "score_known": effective_known,
                "included": included,
                "exclusion_reason": exclusion_reasons.get(index),
                "clause_index": row.get("clause_index"),
                "parent_constrained": constrained,
            }
        )

    return {
        "schema_version": "scenebenchmark-semantic-case-score.v4",
        "score_policy": SEMANTIC_SCORE_POLICY,
        "family_score_policy": SEMANTIC_FAMILY_SCORE_POLICY,
        "family_weights": configured_family_weights,
        "score": _round4(score),
        "known_coverage": _round4(coverage),
        "source_requirement_count": len(rows),
        "active_requirement_count": len(active),
        "excluded_requirement_count": len(exclusion_reasons),
        "parent_adjustment_count": parent_adjustments,
        "applicable_family_count": len(applicable_families),
        "clauses": [
            {
                **value,
                "score": _round4(float(value["score"])),
                "known_coverage": _round4(float(value["known_coverage"])),
            }
            for value in clause_scores
        ],
        "families": {
            family: {
                **value,
                "clauses": [
                    {
                        **clause,
                        "score": _round4(float(clause["score"])),
                        "known_coverage": _round4(
                            float(clause["known_coverage"])
                        ),
                    }
                    for clause in value["clauses"]
                ],
            }
            for family, value in family_scores.items()
        },
        "requirements": audit_rows,
    }


def semantic_exclusion_reasons(
    prompt: str,
    requirements: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Return the score policy's evidence-independent exclusion plan.

    Exact-duplicate and graph-replaced composite detection depends only on
    frozen requirement metadata, not on a candidate verdict.  The controller
    can therefore call this before visual acquisition and avoid spending
    capture/VLM budget on rows that the same scoring policy will later exclude.
    Calling the authoritative scorer here deliberately keeps the planning and
    final aggregation policies from drifting apart.
    """

    result = score_semantic_case(prompt, requirements)
    return {
        str(value["node_id"]): str(value["exclusion_reason"])
        for value in result["requirements"]
        if value["exclusion_reason"] is not None
    }


def macro_average_case_scores(
    cases: Sequence[Mapping[str, Any]],
    *,
    case_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate cases without pooling their requirement rows.

    The default gives every case one vote.  Optional weights must be explicit
    task/category policy; this function never derives them from requirement
    count. A missing whole case leaves the suite score unresolved; it is not
    silently relabelled as a confirmed failure or removed from the denominator.
    """

    if not cases:
        raise ValueError("at least one case is required")
    weights = dict(case_weights or {})
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(cases):
        case_id = str(
            value.get("case_id")
            or value.get("case")
            or value.get("task_id")
            or f"case_{index}"
        )
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        raw_weight = weights.get(case_id, value.get("case_weight", 1.0))
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"case weight for {case_id} must be numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"case weight for {case_id} must be positive and finite")

        score = without_intervals(value).get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (float, int))
                                  or not math.isfinite(score) or not 0.0 <= score <= 1.0):
            raise ValueError(f"invalid score for {case_id}")
        rows.append(
            {
                "case_id": case_id,
                "case_weight": weight,
                "score": score,
            }
        )

    denominator = sum(float(value["case_weight"]) for value in rows)
    score = sum(
        float(value["case_weight"]) * float(value["score"])
        for value in rows
    ) / denominator if all(value["score"] is not None for value in rows) else None
    complete_weight = sum(
        float(value["case_weight"])
        for value in rows
        if value["score"] is not None
    )
    return {
        "schema_version": "scenebenchmark-semantic-suite-score.v2",
        "score_policy": CASE_AGGREGATE_POLICY,
        "score": _round4(score) if score is not None else None,
        "complete_case_weight_coverage": _round4(complete_weight / denominator),
        "case_count": len(rows),
        "case_weight_sum": _round4(denominator),
        "cases": [
            {
                **value,
                "score": _round4(float(value["score"])) if value["score"] is not None else None,
            }
            for value in rows
        ],
    }


__all__ = [
    "CASE_AGGREGATE_POLICY",
    "CONJUNCTIVE_TYPES",
    "DEFAULT_PREDICATE_WEIGHT",
    "PREDICATE_FAMILY",
    "PREDICATE_WEIGHTS",
    "REPLACED_COMPOSITE_TYPES",
    "SEMANTIC_FAMILY_ORDER",
    "SEMANTIC_FAMILY_PREDICATE_TYPES",
    "SEMANTIC_FAMILY_SCORE_POLICY",
    "SEMANTIC_FAMILY_WEIGHTS",
    "SEMANTIC_SCORE_POLICY",
    "attach_semantic_families",
    "macro_average_case_scores",
    "score_semantic_case",
    "semantic_exclusion_reasons",
    "semantic_family_map_for_graph",
]
