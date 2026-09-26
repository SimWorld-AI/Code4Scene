"""Full-prompt Stage 0 authoring merged with Code4Scene review/freeze."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from .bundle import (
    EvidenceClass,
    FrozenVerificationBundle,
    PopulationScope,
    RequirementBinding,
    TaskMode,
    make_evaluation_binding,
)
from .contracts import (
    ArgumentEdge,
    ComparisonOperator,
    EntityGroundingMode,
    EntityNode,
    MemberRole,
    NumericConstraint,
    Polarity,
    PredicateNode,
    PredicateType,
    ReferentKind,
    RequirementGraph,
    RequirementMemberEdge,
    RequirementNode,
    RootRequirement,
    SourceSpan,
)
from .existing_llm import LLMClient
from . import rule_compiler
from .stage0 import Stage0Compiler

HYBRID_COMPILER_NAME = "scenebench-requirement-graph-hybrid"
HYBRID_COMPILER_VERSION = "2.5.1"
AUTO_APPROVAL_POLICY_NAME = "validated-hybrid-compilation"
AUTO_APPROVAL_POLICY_VERSION = "2.4.0"


@dataclass(frozen=True, slots=True)
class HybridCompilation:
    prompt: str
    task_mode: TaskMode
    graph: RequirementGraph | None
    bindings: tuple[RequirementBinding, ...]
    rule_draft: Mapping[str, Any]
    clause_routes: tuple[Mapping[str, Any], ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    request_manifest: tuple[Mapping[str, Any], ...] = ()
    raw_records: tuple[Mapping[str, Any], ...] = ()

    @property
    def can_freeze(self) -> bool:
        return self.graph is not None and not self.diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiler": {
                "name": HYBRID_COMPILER_NAME,
                "version": HYBRID_COMPILER_VERSION,
            },
            "prompt": self.prompt,
            "task_mode": self.task_mode.value,
            "can_freeze": self.can_freeze,
            "graph": self.graph.to_dict() if self.graph is not None else None,
            "requirements": [value.to_dict() for value in self.bindings],
            "rule_draft": dict(self.rule_draft),
            "clause_routes": [dict(value) for value in self.clause_routes],
            "diagnostics": [dict(value) for value in self.diagnostics],
            "request_manifest": [dict(value) for value in self.request_manifest],
            "raw_records": [dict(value) for value in self.raw_records],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HybridCompilation:
        compiler = value.get("compiler")
        if not isinstance(compiler, Mapping) or compiler.get("name") != HYBRID_COMPILER_NAME:
            raise ValueError("document is not a hybrid requirement compilation")
        graph = value.get("graph")
        requirements = value.get("requirements")
        if graph is not None and not isinstance(graph, Mapping):
            raise ValueError("compilation graph must be an object or null")
        if not isinstance(requirements, Sequence) or isinstance(requirements, (str, bytes)):
            raise ValueError("compilation requirements must be a list")
        return cls(
            prompt=str(value.get("prompt") or ""),
            task_mode=TaskMode(value.get("task_mode")),
            graph=RequirementGraph.from_dict(graph) if graph is not None else None,
            bindings=tuple(
                RequirementBinding.from_dict(item)
                for item in requirements
                if isinstance(item, Mapping)
            ),
            rule_draft=value.get("rule_draft") or {},
            clause_routes=tuple(value.get("clause_routes") or ()),
            diagnostics=tuple(value.get("diagnostics") or ()),
            request_manifest=tuple(value.get("request_manifest") or ()),
            raw_records=tuple(value.get("raw_records") or ()),
        )


def _safe_id(prefix: str, value: Any) -> str:
    token = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().casefold()).strip("_")
    token = token or "node"
    if not token[0].isalpha():
        token = "n_" + token
    result = f"{prefix}_{token}"
    if len(result) > 64:
        result = result[:48].rstrip("_") + "_" + result[-15:].lstrip("_")
    return result


def _selector_name(selector: Any, name: str) -> tuple[str, tuple[str, ...]]:
    if not isinstance(selector, Mapping):
        raise ValueError(f"{name} selector must be an object")
    categories = tuple(
        str(value).strip()
        for value in selector.get("allowed_categories") or ()
        if str(value).strip()
    )
    if not categories:
        raise ValueError(f"{name} selector needs allowed_categories")
    return categories[0], categories[1:]


def _constraint(expected: Mapping[str, Any]) -> NumericConstraint:
    if expected.get("exact_count") is not None:
        return NumericConstraint(ComparisonOperator.EQ, int(expected["exact_count"]))
    minimum = expected.get("min_count")
    maximum = expected.get("max_count")
    if minimum is not None and maximum is not None:
        return NumericConstraint(
            ComparisonOperator.BETWEEN, int(minimum), int(maximum)
        )
    if maximum is not None:
        return NumericConstraint(ComparisonOperator.LTE, int(maximum))
    return NumericConstraint(ComparisonOperator.GTE, int(minimum or 1))


def _rule_clause_graph(
    prompt: str,
    clause: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> tuple[RequirementGraph, Mapping[str, str]]:
    span_values = tuple(clause["source_span"])
    span = SourceSpan(int(span_values[0]), int(span_values[1]))
    text = prompt[span.start:span.end]
    prefix = _safe_id("clause", clause["id"])
    requirement = RequirementNode(
        id=_safe_id(prefix, "requirement"), text=text, source_span=span
    )
    nodes: list[Any] = [requirement]
    edges: list[Any] = []
    leaves: list[str] = []
    supports: list[str] = []
    rule_nodes: dict[str, str] = {}

    def entity(selector: Any, suffix: str, *, collection: bool = False) -> EntityNode:
        name, aliases = _selector_name(selector, suffix)
        value = EntityNode(
            id=_safe_id(prefix, suffix),
            text=text,
            name=name,
            aliases=aliases,
            referent_kind=(ReferentKind.COLLECTION if collection else ReferentKind.INDIVIDUAL),
            source_span=span,
        )
        nodes.append(value)
        return value

    for index, item in enumerate(items, start=1):
        kind = str(item.get("type") or "")
        item_prefix = _safe_id(prefix, f"item_{index}_{item.get('id') or kind}")
        expected = item.get("expected") or {}
        if not isinstance(expected, Mapping):
            raise ValueError(f"{kind} expected must be an object")
        subject = entity(
            item.get("subject"),
            f"{item_prefix}_subject",
            collection=kind in {"object_count", "around"},
        )
        if kind == "object_presence":
            leaves.append(subject.id)
            rule_nodes[str(item["id"])] = subject.id
            continue

        if kind == "forbidden_object":
            predicate_type = PredicateType.EXISTENCE
            predicate_name = "exists"
            polarity = Polarity.NEGATED
            constraint = None
        elif kind == "object_count":
            predicate_type = PredicateType.COUNT
            predicate_name = "count"
            polarity = Polarity.AFFIRMATIVE
            constraint = _constraint(expected)
        elif kind == "attribute_value":
            field_name = str(expected.get("field") or "attribute")
            predicate_type = (
                PredicateType.MATERIAL
                if "material" in field_name.casefold()
                else PredicateType.ATTRIBUTE
            )
            predicate_name = f"{field_name}={expected.get('equals')}"
            polarity = Polarity.AFFIRMATIVE
            constraint = None
        elif kind in {
            "object_object_relation",
            "object_architecture_relation",
            "around",
        }:
            predicate_type = PredicateType.SPATIAL_RELATION
            predicate_name = str(expected.get("relation") or kind)
            polarity = Polarity.AFFIRMATIVE
            constraint = None
        else:
            raise ValueError(f"rule item type {kind!r} has no graph lowering")

        predicate = PredicateNode(
            id=_safe_id(item_prefix, "predicate"),
            text=text,
            name=predicate_name,
            predicate_type=predicate_type,
            polarity=polarity,
            constraint=constraint,
            source_span=span,
        )
        nodes.append(predicate)
        role = "collection" if predicate_type is PredicateType.COUNT else "subject"
        edges.append(ArgumentEdge(predicate.id, subject.id, role, 0))
        supports.append(subject.id)
        if predicate_type is PredicateType.SPATIAL_RELATION:
            object_node = entity(item.get("object"), f"{item_prefix}_object")
            edges.append(ArgumentEdge(predicate.id, object_node.id, "reference", 1))
            supports.append(object_node.id)
        leaves.append(predicate.id)
        rule_nodes[str(item["id"])] = predicate.id

    if not leaves:
        raise ValueError(f"resolved clause {clause['id']} produced no graph leaves")
    weight = 1.0 / len(leaves)
    edges.extend(
        RequirementMemberEdge(
            requirement.id,
            leaf,
            MemberRole.SCORED_FACET,
            weight,
        )
        for leaf in leaves
    )
    edges.extend(
        RequirementMemberEdge(requirement.id, support, MemberRole.SUPPORT_ONLY)
        for support in dict.fromkeys(supports)
        if support not in leaves
    )
    return (
        RequirementGraph(
            prompt=prompt,
            nodes=tuple(nodes),
            edges=tuple(edges),
            roots=(RootRequirement(requirement.id, 1.0),),
        ),
        rule_nodes,
    )


def _relocated_payload(
    graph: RequirementGraph,
    *,
    prefix: str,
    offset: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Mapping[str, str],
]:
    payload = graph.to_dict()
    mapping = {
        node["id"]: _safe_id(prefix, node["id"])
        for node in payload["nodes"]
    }
    for node in payload["nodes"]:
        node["id"] = mapping[node["id"]]
        node["source_span"]["start"] += offset
        node["source_span"]["end"] += offset
        if node.get("member_ids"):
            node["member_ids"] = [mapping[value] for value in node["member_ids"]]
        semantic_parameters = node.get("semantic_parameters")
        if isinstance(semantic_parameters, dict):
            scope = semantic_parameters.get("scope")
            if isinstance(scope, dict) and scope.get("entity_id") is not None:
                scope["entity_id"] = mapping[scope["entity_id"]]
            logic = semantic_parameters.get("logic")
            if isinstance(logic, dict) and logic.get("requirement_ids") is not None:
                logic["requirement_ids"] = [
                    mapping[value] for value in logic["requirement_ids"]
                ]
    for edge in payload["edges"]:
        edge["source_id"] = mapping[edge["source_id"]]
        edge["target_id"] = mapping[edge["target_id"]]
    for root in payload["roots"]:
        root["requirement_id"] = mapping[root["requirement_id"]]
    return payload["nodes"], payload["edges"], payload["roots"], mapping


def _merge_graphs(
    prompt: str,
    values: Sequence[tuple[RequirementGraph, int, Mapping[str, str]]],
) -> tuple[RequirementGraph, Mapping[str, str]]:
    if not values:
        raise ValueError("cannot merge an empty graph sequence")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    merged_rule_nodes: dict[str, str] = {}
    for index, (graph, offset, rule_nodes) in enumerate(values, start=1):
        child_nodes, child_edges, child_roots, node_mapping = _relocated_payload(
            graph, prefix=f"c{index:02d}", offset=offset
        )
        nodes.extend(child_nodes)
        edges.extend(child_edges)
        roots.extend(child_roots)
        merged_rule_nodes.update(
            {rule_id: node_mapping[node_id] for rule_id, node_id in rule_nodes.items()}
        )

    start = len(prompt) - len(prompt.lstrip())
    end = len(prompt.rstrip())
    top = RequirementNode(
        id="merged_requirement_root",
        text=prompt[start:end],
        source_span=SourceSpan(start, end),
    )
    nodes.append(top.to_dict())
    total_weight = sum(float(value["weight"]) for value in roots)
    for root in roots:
        edges.append(
            RequirementMemberEdge(
                top.id,
                root["requirement_id"],
                MemberRole.SCORED_FACET,
                float(root["weight"]) / total_weight,
            ).to_dict()
        )
    return (
        RequirementGraph.from_dict(
            {
                "schema_version": "1.0",
                "prompt": prompt,
                "nodes": nodes,
                "edges": edges,
                "roots": [RootRequirement(top.id, 1.0).to_dict()],
            }
        ),
        merged_rule_nodes,
    )


_ACTION_SCOPE = re.compile(
    r"\b(?:add|create|build|place|put|insert|spawn|do\s+not\s+add)\b",
    re.IGNORECASE,
)
_GT_LANGUAGE = re.compile(
    r"\b(?:restore|recover|match\s+the\s+original|return\s+to\s+the\s+original)\b",
    re.IGNORECASE,
)
_PHYSICS_LANGUAGE = re.compile(
    r"\b(?:grounded|on\s+the\s+ground|floating|collision|colliding|"
    r"interpenetrat(?:e|ing|ion)|physically\s+stable)\b",
    re.IGNORECASE,
)


def _binding_for_node(
    graph: RequirementGraph,
    node_id: str,
    task_mode: TaskMode,
    deterministic_rule_id: str | None = None,
) -> RequirementBinding:
    node = graph.node(node_id)
    source_span = (node.source_span.start, node.source_span.end)
    source_text = graph.prompt[source_span[0]:source_span[1]]
    owner = "requirement_graph"
    evidence_class = EvidenceClass.OPEN_ENDED
    action_scoped = (
        (
            graph.schema_version == "1.0"
            or task_mode is not TaskMode.GENERATION
        )
        and bool(_ACTION_SCOPE.search(source_text))
    )
    scope = PopulationScope.ADDITIONS if action_scoped else PopulationScope.CANDIDATE_ALL
    if _PHYSICS_LANGUAGE.search(source_text):
        owner = "physics"
    if task_mode is TaskMode.REPAIR and _GT_LANGUAGE.search(source_text):
        owner = "gt_repair"
        evidence_class = EvidenceClass.GT
        scope = PopulationScope.GT_TARGETS
    reference_names = {
        graph.node(argument.target_id).name.strip().casefold()
        for candidate in graph.nodes
        if isinstance(candidate, PredicateNode)
        for argument in graph.arguments_for(candidate.id)
        if argument.role in {"object", "reference", "target", "anchor"}
        and isinstance(graph.node(argument.target_id), EntityNode)
    }
    entity_scopes: dict[str, PopulationScope] = {}
    if isinstance(node, EntityNode):
        entity_scopes[node.id] = scope
    elif isinstance(node, PredicateNode):
        arguments = graph.arguments_for(node.id)
        for argument in arguments:
            argument_scope = scope
            argument_node = graph.node(argument.target_id)
            if (
                action_scoped
                and evidence_class is EvidenceClass.OPEN_ENDED
                and (
                    argument.role in {"object", "reference", "target", "anchor"}
                    or (
                        node.predicate_type is PredicateType.COUNT
                        and isinstance(argument_node, EntityNode)
                        and argument_node.name.strip().casefold() in reference_names
                    )
                )
            ):
                argument_scope = PopulationScope.CANDIDATE_ALL
            entity_scopes[argument.target_id] = argument_scope
        if arguments:
            # The first argument is the scored subject/collection population.
            scope = entity_scopes[arguments[0].target_id]
    binding = RequirementBinding(
        requirement_id=f"atomic_{node_id}",
        node_id=node_id,
        source_span=source_span,
        source_text=source_text,
        primary_owner=owner,
        evidence_class=evidence_class,
        population_scope=scope,
        entity_scopes=entity_scopes,
        deterministic_rule_id=deterministic_rule_id,
    )
    from .capabilities import capability_for_node

    decision = capability_for_node(graph, node)
    return replace(binding, status=decision.status)


def _v2_grounding_diagnostics(
    graph: RequirementGraph,
) -> tuple[Mapping[str, Any], ...]:
    """Reject newly authored graphs that lack an executable grounding plan.

    This validation is deliberately Candidate-independent. It checks only the
    semantic acquisition contract frozen by Stage 0; Actor ids and asset paths
    remain runtime evidence and must never enter an authored bundle.
    """

    if graph.schema_version != "2.0":
        return ()

    from .actor_inventory import normalize_identity_term
    from .capabilities import GLOBAL_VLM_PLANNER_ROUTE, capability_for_node
    from .stage1 import Stage1Result
    from .stage2_tasks import build_stage2_queries, build_stage2_tasks

    diagnostics: list[Mapping[str, Any]] = []
    for entity in (
        node for node in graph.nodes if isinstance(node, EntityNode)
    ):
        if entity.grounding_mode is EntityGroundingMode.AUTO:
            diagnostics.append(
                {
                    "clause_id": "full_prompt",
                    "kind": "missing_entity_grounding_mode",
                    "node_id": entity.id,
                    "reason": (
                        f"entity {entity.id!r} needs an explicit grounding_mode; "
                        "AUTO is accepted only when loading legacy bundles"
                    ),
                    "blocking": True,
                }
            )

    tasks = build_stage2_tasks(graph, Stage1Result((), "partial"))
    feature_overview_entity_ids = {
        entity_id
        for task in tasks
        if task.allows_overview_fallback
        for entity_id in task.visual_feature_entity_ids
    }
    changed = True
    while changed:
        changed = False
        for entity in (
            node for node in graph.nodes if isinstance(node, EntityNode)
        ):
            if entity.id in feature_overview_entity_ids:
                before = len(feature_overview_entity_ids)
                feature_overview_entity_ids.update(entity.member_ids)
                changed = changed or len(feature_overview_entity_ids) != before
            elif entity.member_ids and all(
                member_id in feature_overview_entity_ids
                for member_id in entity.member_ids
            ):
                feature_overview_entity_ids.add(entity.id)
                changed = True
    query_entity_ids = {
        query.query_id for query in build_stage2_queries(graph, tasks)
    }
    query_path_entity_ids = set(query_entity_ids)
    pending_query_paths = list(query_entity_ids)
    while pending_query_paths:
        entity_id = pending_query_paths.pop(0)
        entity = graph.node(entity_id)
        if not isinstance(entity, EntityNode):
            continue
        for member_id in entity.member_ids:
            if member_id in query_path_entity_ids:
                continue
            query_path_entity_ids.add(member_id)
            pending_query_paths.append(member_id)
    for entity in (
        node for node in graph.nodes if isinstance(node, EntityNode)
    ):
        if (
            entity.effective_grounding_mode
            in {
                EntityGroundingMode.ACTOR,
                EntityGroundingMode.ACTOR_COLLECTION,
            }
            and entity.id not in query_path_entity_ids
            and entity.id not in feature_overview_entity_ids
        ):
            diagnostics.append(
                {
                    "clause_id": "full_prompt",
                    "kind": "missing_actor_identity_query_path",
                    "node_id": entity.id,
                    "reason": (
                        f"Actor-groundable entity {entity.id!r} has no active "
                        "Stage 2 identity-query path"
                    ),
                    "blocking": True,
                }
            )
            continue
        if entity.effective_grounding_mode in {
            EntityGroundingMode.ACTOR,
            EntityGroundingMode.ACTOR_COLLECTION,
        } and not any(
            entity.id in feature_overview_entity_ids or
            normalize_identity_term(term)
            for term in (entity.name, *entity.aliases)
        ):
            diagnostics.append(
                {
                    "clause_id": "full_prompt",
                    "kind": "missing_actor_identity_terms",
                    "node_id": entity.id,
                    "reason": (
                        f"Actor-groundable entity {entity.id!r} has no valid "
                        "candidate-independent identity term"
                    ),
                    "blocking": True,
                }
            )

    for node_id in graph.effective_weights():
        node = graph.node(node_id)
        if not isinstance(node, PredicateNode):
            continue
        dependency_ids = {
            edge.target_id for edge in graph.arguments_for(node.id)
        }
        scope = node.semantic_parameters.get("scope")
        if isinstance(scope, Mapping) and scope.get("entity_id") is not None:
            dependency_ids.add(str(scope["entity_id"]))
        dependencies = tuple(
            graph.node(entity_id) for entity_id in sorted(dependency_ids)
        )
        needs_global_evidence = any(
            isinstance(entity, EntityNode)
            and entity.effective_grounding_mode
            in {
                EntityGroundingMode.DERIVED_REGION,
                EntityGroundingMode.SCENE_GLOBAL,
            }
            for entity in dependencies
        )
        decision = capability_for_node(graph, node)
        if (
            needs_global_evidence
            and decision.planner_route != GLOBAL_VLM_PLANNER_ROUTE
        ):
            diagnostics.append(
                {
                    "clause_id": "full_prompt",
                    "kind": "incompatible_grounding_route",
                    "node_id": node.id,
                    "reason": (
                        f"requirement {node.id!r} depends on a derived/global "
                        "entity but is not routed to scene-level evidence"
                    ),
                    "blocking": True,
                }
            )
    return tuple(diagnostics)


def compile_hybrid(
    prompt: str,
    *,
    task_mode: TaskMode | str,
    llm_client: LLMClient | None = None,
    stage0_max_tokens: int | None = None,
    stage0_validation_feedback: str | None = None,
    previous_stage0_draft: Mapping[str, Any] | None = None,
    taxonomy: Mapping[str, Sequence[str]] | None = None,
    attributes: Mapping[str, Mapping[str, Any]] | None = None,
    relation_policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> HybridCompilation:
    """Use one full-prompt LLM parse, with rules retained as explicit legacy mode."""

    mode = TaskMode(task_mode)
    try:
        rule_draft = rule_compiler.compile_prompt(
            prompt,
            taxonomy=taxonomy,
            attributes=attributes,
            relation_policies=relation_policies,
        )
        rule_validation = rule_compiler.validate_draft(rule_draft)
    except rule_compiler.RuleCompilationError as exc:
        if llm_client is None:
            raise
        # The full-prompt LLM route does not consume the legacy rule draft.
        # Preserve a failed audit as provenance instead of preventing the
        # semantic compiler from seeing the prompt at all.
        rule_draft = {
            "schema_version": "0.1.0",
            "draft_id": "rule-draft-audit-failed",
            "prompt": prompt,
            "compiler": {
                "name": rule_compiler.COMPILER_NAME,
                "version": rule_compiler.COMPILER_VERSION,
                "mode": "rule_based_audit_failed",
            },
            "clauses": [],
            "items": [],
            "assumptions": [],
            "unresolved": [],
            "unsupported": [],
            "audit_error": f"{type(exc).__name__}: {exc}",
            "world_assumption": "open_world",
        }
        rule_validation = {"conflicts": [], "duplicates": []}
    if llm_client is not None:
        compiler = Stage0Compiler(
            llm_client,
            max_tokens=(
                int(stage0_max_tokens)
                if stage0_max_tokens is not None
                else int(getattr(llm_client, "max_tokens", 4096))
            ),
        )
        evaluation = compiler.compile(
            prompt,
            validation_feedback=stage0_validation_feedback,
            previous_draft=previous_stage0_draft,
        )
        if evaluation.graph is None:
            graph = None
            diagnostics: tuple[Mapping[str, Any], ...] = (
                {
                    "clause_id": "full_prompt",
                    "kind": "llm_semantic_draft_failed",
                    "reason": evaluation.result.evaluation_error,
                    "blocking": True,
                },
            )
            bindings: tuple[RequirementBinding, ...] = ()
            route = "blocked"
        else:
            graph = evaluation.graph
            if evaluation.draft is not None:
                rule_draft = {
                    **dict(rule_draft),
                    "stage0_semantic_ir": dict(evaluation.draft),
                }
            diagnostics = _v2_grounding_diagnostics(graph)
            bindings = tuple(
                _binding_for_node(graph, node_id, mode)
                for node_id in graph.effective_weights()
            )
            route = (
                "blocked"
                if diagnostics
                else "llm_semantic_draft_full_prompt"
            )
        route_record: dict[str, Any] = {
            "clause_id": "full_prompt",
            "route": route,
        }
        if graph is not None and graph.schema_version == "2.0":
            from .capabilities import capability_summary

            route_record["capability_audit"] = capability_summary(graph)
        return HybridCompilation(
            prompt=prompt,
            task_mode=mode,
            graph=graph,
            bindings=bindings,
            rule_draft=rule_draft,
            clause_routes=(
                route_record,
            ),
            diagnostics=diagnostics,
            request_manifest=tuple(evaluation.request_manifest),
            raw_records=tuple(evaluation.raw_records),
        )

    items_by_clause: dict[str, list[Mapping[str, Any]]] = {}
    for item in rule_draft.get("items") or ():
        trace = item.get("traceability") or {}
        for clause_id in trace.get("source_clause_ids") or ():
            items_by_clause.setdefault(str(clause_id), []).append(item)

    compiler = Stage0Compiler(llm_client) if llm_client is not None else None
    graphs: list[tuple[RequirementGraph, int, Mapping[str, str]]] = []
    routes: list[Mapping[str, Any]] = []
    diagnostics: list[Mapping[str, Any]] = []
    manifests: list[Mapping[str, Any]] = []
    raw_records: list[Mapping[str, Any]] = []
    diagnostics.extend(
        {
            "kind": str(value.get("kind") or "rule_validation_failed"),
            "reason": str(value),
            "blocking": True,
            **({"item_ids": list(value["item_ids"])} if value.get("item_ids") else {}),
        }
        for field in ("conflicts", "duplicates")
        for value in rule_validation[field]
    )
    for clause in rule_draft["clauses"]:
        clause_id = str(clause["id"])
        if clause.get("status") == "resolved":
            graph, rule_nodes = _rule_clause_graph(
                prompt, clause, items_by_clause.get(clause_id, ())
            )
            graphs.append((graph, 0, rule_nodes))
            routes.append({"clause_id": clause_id, "route": "rule_compiler"})
            continue
        if compiler is None:
            diagnostics.append(
                {
                    "clause_id": clause_id,
                    "kind": "llm_fallback_unavailable",
                    "reason": "rule compiler did not fully parse the clause",
                    "blocking": True,
                }
            )
            routes.append({"clause_id": clause_id, "route": "blocked"})
            continue
        evaluation = compiler.compile(str(clause["text"]))
        manifests.extend(evaluation.request_manifest)
        raw_records.extend(evaluation.raw_records)
        if evaluation.graph is None:
            diagnostics.append(
                {
                    "clause_id": clause_id,
                    "kind": "llm_semantic_draft_failed",
                    "reason": evaluation.result.evaluation_error,
                    "blocking": True,
                }
            )
            routes.append({"clause_id": clause_id, "route": "blocked"})
            continue
        graphs.append((evaluation.graph, int(clause["source_span"][0]), {}))
        routes.append({"clause_id": clause_id, "route": "llm_semantic_draft"})

    graph, rule_nodes = (
        _merge_graphs(prompt, graphs)
        if graphs and not diagnostics
        else (None, {})
    )
    node_rules = {node_id: rule_id for rule_id, node_id in rule_nodes.items()}
    bindings = (
        tuple(
            _binding_for_node(
                graph,
                node_id,
                mode,
                deterministic_rule_id=node_rules.get(node_id),
            )
            for node_id in graph.effective_weights()
        )
        if graph is not None
        else ()
    )
    return HybridCompilation(
        prompt=prompt,
        task_mode=mode,
        graph=graph,
        bindings=bindings,
        rule_draft=rule_draft,
        clause_routes=tuple(routes),
        diagnostics=tuple(diagnostics),
        request_manifest=tuple(manifests),
        raw_records=tuple(raw_records),
    )


def freeze_compilation(
    compilation: HybridCompilation,
    review: Mapping[str, Any] | None = None,
    *,
    bundle_id: str | None = None,
) -> FrozenVerificationBundle:
    if not compilation.can_freeze or compilation.graph is None:
        raise ValueError("hybrid compilation has blocking diagnostics")
    if review is None:
        approval = {
            "approved": True,
            "review_method": "automated_policy",
            "human_reviewed": False,
            "reviewer": f"{HYBRID_COMPILER_NAME}:{HYBRID_COMPILER_VERSION}",
            "policy_name": AUTO_APPROVAL_POLICY_NAME,
            "policy_version": AUTO_APPROVAL_POLICY_VERSION,
            "basis": [
                "hybrid_compilation_has_no_blocking_diagnostics",
                "requirement_graph_contract_valid",
                "prompt_grounding_valid",
            ],
        }
    else:
        approval = dict(review)
        if approval.get("approved") is not True:
            raise ValueError("freeze requires review.approved=true")
        reviewer = str(approval.get("reviewer") or "").strip()
        reviewed_at = str(approval.get("reviewed_at") or "").strip()
        if not reviewer:
            raise ValueError("freeze requires review.reviewer")
        try:
            datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("freeze requires ISO-8601 review.reviewed_at") from exc
        approval.setdefault("review_method", "human")
        approval.setdefault("human_reviewed", True)
    provenance = {
        "compiler_name": HYBRID_COMPILER_NAME,
        "compiler_version": HYBRID_COMPILER_VERSION,
        "clause_routes": [dict(value) for value in compilation.clause_routes],
        "deterministic_rule_draft": dict(compilation.rule_draft),
        "authoring_parser": "llm_full_prompt_with_rule_audit",
    }
    semantic_ir = compilation.rule_draft.get("stage0_semantic_ir")
    if isinstance(semantic_ir, Mapping):
        provenance["stage0_semantic_ir"] = dict(semantic_ir)
    evaluations = tuple(
        make_evaluation_binding(value, compilation.graph, provenance)
        for value in compilation.bindings
    )
    return FrozenVerificationBundle(
        schema_version="1.0",
        bundle_id=bundle_id or _safe_id(
            "verification", compilation.prompt.splitlines()[0][:48]
        ),
        task_mode=compilation.task_mode,
        graph=compilation.graph,
        requirements=compilation.bindings,
        evaluations=evaluations,
        provenance=provenance,
        review=approval,
    )


__all__ = [
    "HYBRID_COMPILER_NAME",
    "HYBRID_COMPILER_VERSION",
    "AUTO_APPROVAL_POLICY_NAME",
    "AUTO_APPROVAL_POLICY_VERSION",
    "HybridCompilation",
    "compile_hybrid",
    "freeze_compilation",
]
