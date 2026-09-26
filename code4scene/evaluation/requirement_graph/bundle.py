"""Frozen ownership and scope ledger around a RequirementGraph."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .contracts import EntityNode, PredicateNode, RequirementGraph


class TaskMode(str, Enum):
    REPAIR = "repair"
    EDIT = "edit"
    GENERATION = "generation"


class EvidenceClass(str, Enum):
    OPEN_ENDED = "open_ended"
    GT = "gt"


class PopulationScope(str, Enum):
    CANDIDATE_ALL = "candidate_all"
    ADDITIONS = "additions"
    GT_TARGETS = "gt_targets"


class RequirementStatus(str, Enum):
    SUPPORTED = "supported"
    WAIVED = "waived"
    UNPLANNED = "unplanned"


class UnknownReason(str, Enum):
    VISUAL_EVIDENCE_INCOMPLETE = "visual_evidence_incomplete"
    TARGET_NOT_LOCALIZED = "target_not_localized"
    TARGETED_EVIDENCE_MISSING = "targeted_evidence_missing"
    SCOPE_UNRESOLVED = "scope_unresolved"
    PROVENANCE_UNTRUSTED = "provenance_untrusted"
    UNSUPPORTED = "unsupported"
    NON_VISUAL = "non_visual"
    EVALUATION_ERROR = "evaluation_error"

    @property
    def stage3_eligible(self) -> bool:
        return self is UnknownReason.VISUAL_EVIDENCE_INCOMPLETE


class RequirementEvaluationStatus(str, Enum):
    """Public terminal outcome.

    UNKNOWN exists only in Stage 1--3 routing. Healthy terminal assessments
    are binary; unavailable prerequisites and runtime failure remain distinct.
    """

    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    NOT_EVALUATED = "NOT_EVALUATED"
    ERROR = "ERROR"


def _text(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclass(frozen=True, slots=True)
class RequirementBinding:
    requirement_id: str
    node_id: str
    source_span: tuple[int, int]
    source_text: str
    primary_owner: str
    evidence_class: EvidenceClass | str
    population_scope: PopulationScope | str
    status: RequirementStatus | str = RequirementStatus.SUPPORTED
    # A relation can intentionally compare two different populations.  For
    # example, "add chairs near the table" scopes the subject chairs to the
    # additions while resolving the reference table from the whole Candidate.
    # ``population_scope`` remains the scored/subject scope; this map is the
    # authoritative per-argument scope ledger.
    entity_scopes: Mapping[str, PopulationScope | str] = field(default_factory=dict)
    deterministic_rule_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requirement_id", _text(self.requirement_id, "requirement_id")
        )
        rule_id = _text(self.deterministic_rule_id, "deterministic_rule_id") \
            if self.deterministic_rule_id is not None else None
        object.__setattr__(self, "deterministic_rule_id", rule_id)
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        span = tuple(self.source_span)
        if (
            len(span) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
            or span[0] < 0
            or span[1] <= span[0]
        ):
            raise ValueError("source_span must satisfy 0 <= start < end")
        object.__setattr__(self, "source_span", span)
        object.__setattr__(self, "source_text", _text(self.source_text, "source_text"))
        object.__setattr__(
            self, "primary_owner", _text(self.primary_owner, "primary_owner")
        )
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(
            self, "population_scope", PopulationScope(self.population_scope)
        )
        object.__setattr__(self, "status", RequirementStatus(self.status))
        object.__setattr__(
            self,
            "entity_scopes",
            {
                _text(entity_id, "entity_scopes key"): PopulationScope(scope)
                for entity_id, scope in dict(self.entity_scopes).items()
            },
        )

    def scope_for_entity(self, entity_id: str) -> PopulationScope:
        return self.entity_scopes.get(str(entity_id), self.population_scope)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "node_id": self.node_id,
            "source_span": list(self.source_span),
            "source_text": self.source_text,
            "primary_owner": self.primary_owner,
            "evidence_class": self.evidence_class.value,
            "population_scope": self.population_scope.value,
            "status": self.status.value,
            "entity_scopes": {
                key: value.value for key, value in sorted(self.entity_scopes.items())
            },
            "deterministic_rule_id": self.deterministic_rule_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RequirementBinding:
        required = {
            "requirement_id",
            "node_id",
            "source_span",
            "source_text",
            "primary_owner",
            "evidence_class",
            "population_scope",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"requirement binding is missing {missing}")
        return cls(
            requirement_id=value["requirement_id"],
            node_id=value["node_id"],
            source_span=tuple(value["source_span"]),
            source_text=value["source_text"],
            primary_owner=value["primary_owner"],
            evidence_class=value["evidence_class"],
            population_scope=value["population_scope"],
            status=value.get("status", RequirementStatus.SUPPORTED.value),
            entity_scopes=value.get("entity_scopes") or {},
            deterministic_rule_id=value.get("deterministic_rule_id"),
        )

class BindingSourceKind(str, Enum):
    TEXT_SPAN = "text_span"
    REFERENCE_IMAGE = "reference_image"
    GT_INPUT_DIFF = "gt_input_diff"
    LEGACY_CONTRACT = "legacy_contract"


def _json_native(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            result[key] = _json_native(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_native(item, f"{path}[]") for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class EvaluationBinding:
    """Frozen evaluator choice and evidence contract for one graph leaf."""

    binding_id: str
    requirement_id: str
    node_id: str
    evaluator_id: str
    evaluator_version: str
    parameters: Mapping[str, Any]
    population_scope: PopulationScope | str
    entity_scopes: Mapping[str, PopulationScope | str]
    required_evidence: tuple[str, ...]
    source_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "binding_id",
            "requirement_id",
            "node_id",
            "evaluator_id",
            "evaluator_version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "parameters", _json_native(dict(self.parameters), "parameters")
        )
        object.__setattr__(
            self, "population_scope", PopulationScope(self.population_scope)
        )
        object.__setattr__(
            self,
            "entity_scopes",
            {
                _text(key, "entity_scopes key"): PopulationScope(value)
                for key, value in dict(self.entity_scopes).items()
            },
        )
        evidence = tuple(
            _text(value, "required_evidence item")
            for value in self.required_evidence
        )
        if not evidence or len(evidence) != len(set(evidence)):
            raise ValueError(
                "EvaluationBinding.required_evidence must be non-empty and unique"
            )
        object.__setattr__(self, "required_evidence", evidence)
        provenance = _json_native(
            dict(self.source_provenance), "source_provenance"
        )
        kind = BindingSourceKind(provenance.get("kind"))
        if kind is BindingSourceKind.TEXT_SPAN:
            span = provenance.get("source_span")
            if (
                not isinstance(span, list)
                or len(span) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in span
                )
                or span[0] < 0
                or span[1] <= span[0]
            ):
                raise ValueError(
                    "text_span source_provenance needs a valid source_span"
                )
        elif kind is BindingSourceKind.REFERENCE_IMAGE:
            reference_ids = provenance.get("reference_ids")
            if (
                not isinstance(reference_ids, list)
                or not reference_ids
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in reference_ids
                )
            ):
                raise ValueError(
                    "reference_image source_provenance needs reference_ids"
                )
        elif kind is BindingSourceKind.GT_INPUT_DIFF:
            target_id = provenance.get("repair_target_id")
            operation = provenance.get("operation")
            if not isinstance(target_id, str) or not target_id.strip():
                raise ValueError(
                    "gt_input_diff source_provenance needs repair_target_id"
                )
            if operation not in {"add", "remove", "repair"}:
                raise ValueError(
                    "gt_input_diff source_provenance needs a valid operation"
                )
        object.__setattr__(self, "source_provenance", provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "requirement_id": self.requirement_id,
            "node_id": self.node_id,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "parameters": dict(self.parameters),
            "population_scope": self.population_scope.value,
            "entity_scopes": {
                key: value.value
                for key, value in sorted(self.entity_scopes.items())
            },
            "required_evidence": list(self.required_evidence),
            "source_provenance": dict(self.source_provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationBinding:
        required = {
            "binding_id",
            "requirement_id",
            "node_id",
            "evaluator_id",
            "evaluator_version",
            "parameters",
            "population_scope",
            "entity_scopes",
            "required_evidence",
            "source_provenance",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"evaluation binding is missing {missing}")
        unknown = sorted(set(value) - required)
        if unknown:
            raise ValueError(f"evaluation binding has unsupported fields {unknown}")
        return cls(
            binding_id=value["binding_id"],
            requirement_id=value["requirement_id"],
            node_id=value["node_id"],
            evaluator_id=value["evaluator_id"],
            evaluator_version=value["evaluator_version"],
            parameters=value["parameters"],
            population_scope=value["population_scope"],
            entity_scopes=value["entity_scopes"],
            required_evidence=tuple(value["required_evidence"]),
            source_provenance=value["source_provenance"],
        )

def make_evaluation_binding(
    requirement: RequirementBinding,
    graph: RequirementGraph,
    provenance: Mapping[str, Any],
) -> EvaluationBinding:
    """Deterministically bind one semantic leaf to its atomic evaluator."""

    node = graph.node(requirement.node_id)
    parameters: dict[str, Any] = {
        "graph_node_kind": "entity" if isinstance(node, EntityNode) else "predicate",
    }
    if isinstance(node, PredicateNode):
        parameters["predicate_type"] = node.predicate_type.value
    if requirement.deterministic_rule_id is not None:
        parameters["deterministic_rule_id"] = requirement.deterministic_rule_id

    owner = requirement.primary_owner
    if owner == "requirement_graph":
        evaluator_id = (
            "semantic_requirements.atomic_rule"
            if requirement.deterministic_rule_id is not None
            else "semantic_requirements.evidence_ladder"
        )
        required_evidence = (
            "candidate_scene_graph",
            "requirement_graph",
            "evaluation_binding",
        )
    elif owner == "physics":
        evaluator_id = "physical_safety.task_binding"
        required_evidence = (
            "candidate_scene_graph",
            "ue_physics_measurement",
            "evaluation_binding",
        )
    elif owner == "gt_repair":
        evaluator_id = "scene_diff.task_binding"
        required_evidence = (
            "candidate_scene_graph",
            "gt_scene_graph",
            "evaluation_binding",
        )
    else:
        raise ValueError(
            f"{requirement.requirement_id}: unsupported primary_owner {owner!r}"
        )

    repair_sources = provenance.get("repair_target_by_node")
    repair_source = (
        repair_sources.get(requirement.node_id)
        if isinstance(repair_sources, Mapping)
        else None
    )
    references = provenance.get("reference_images")
    ledger = str(provenance.get("reference_observation_ledger") or "")
    if isinstance(repair_source, Mapping):
        source_provenance = {
            "kind": BindingSourceKind.GT_INPUT_DIFF.value,
            "repair_target_id": str(repair_source.get("target_id") or ""),
            "operation": str(repair_source.get("operation") or ""),
            "input_actor_identity": repair_source.get("input_actor_identity"),
            "gt_actor_identity": repair_source.get("gt_actor_identity"),
        }
    elif (
        provenance.get("case_type") == "image_to_scene"
        and requirement.source_text in ledger
        and isinstance(references, list)
    ):
        reference_ids = [
            str(value.get("reference_id"))
            for value in references
            if isinstance(value, Mapping) and value.get("reference_id")
        ]
        source_provenance = {
            "kind": BindingSourceKind.REFERENCE_IMAGE.value,
            "reference_ids": reference_ids,
        }
    elif provenance.get("authoring_parser") == "offline_legacy_contract_migration":
        source_provenance = {
            "kind": BindingSourceKind.LEGACY_CONTRACT.value,
            "migration_source": "offline_legacy_contract",
        }
    else:
        source_provenance = {
            "kind": BindingSourceKind.TEXT_SPAN.value,
            "source_span": list(requirement.source_span),
        }
    return EvaluationBinding(
        binding_id=f"evaluation_{requirement.requirement_id}",
        requirement_id=requirement.requirement_id,
        node_id=requirement.node_id,
        evaluator_id=evaluator_id,
        evaluator_version="1.0",
        parameters=parameters,
        population_scope=requirement.population_scope,
        entity_scopes=requirement.entity_scopes,
        required_evidence=required_evidence,
        source_provenance=source_provenance,
    )

@dataclass(frozen=True, slots=True)
class FrozenVerificationBundle:
    bundle_id: str
    task_mode: TaskMode | str
    graph: RequirementGraph
    requirements: tuple[RequirementBinding, ...]
    provenance: Mapping[str, Any]
    review: Mapping[str, Any]
    evaluations: tuple[EvaluationBinding, ...]
    schema_version: str = "1.0"
    _evaluation_by_node: Mapping[str, EvaluationBinding] = field(init=False, repr=False)
    _by_node: Mapping[str, RequirementBinding] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported frozen bundle schema_version")
        object.__setattr__(self, "bundle_id", _text(self.bundle_id, "bundle_id"))
        object.__setattr__(self, "task_mode", TaskMode(self.task_mode))
        if not isinstance(self.graph, RequirementGraph):
            raise TypeError("graph must be a RequirementGraph")
        requirements = tuple(self.requirements)
        if any(not isinstance(value, RequirementBinding) for value in requirements):
            raise TypeError("requirements must contain RequirementBinding values")
        evaluations = tuple(self.evaluations)
        if any(not isinstance(value, EvaluationBinding) for value in evaluations):
            raise TypeError("evaluations must contain EvaluationBinding values")
        if len(evaluations) != len(requirements):
            raise ValueError(
                "bundle requires exactly one EvaluationBinding per requirement"
            )
        evaluation_nodes = [value.node_id for value in evaluations]
        evaluation_requirements = [value.requirement_id for value in evaluations]
        if set(evaluation_nodes) != {value.node_id for value in requirements}:
            raise ValueError(
                "EvaluationBinding node IDs must equal requirement node IDs"
            )
        if set(evaluation_requirements) != {
            value.requirement_id for value in requirements
        }:
            raise ValueError(
                "EvaluationBinding requirement IDs must equal requirement IDs"
            )
        if len(evaluation_nodes) != len(set(evaluation_nodes)):
            raise ValueError("EvaluationBinding node IDs must be unique")
        node_ids = [value.node_id for value in requirements]
        requirement_ids = [value.requirement_id for value in requirements]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("each scored graph leaf must have exactly one binding")
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("requirement_id values must be unique")
        scored = set(self.graph.effective_weights())
        if set(node_ids) != scored:
            raise ValueError(
                "binding node ids must equal scored graph leaves: "
                f"missing={sorted(scored - set(node_ids))}, "
                f"extra={sorted(set(node_ids) - scored)}"
            )
        for binding in requirements:
            node = self.graph.node(binding.node_id)
            span = node.source_span
            if binding.source_span != (span.start, span.end):
                raise ValueError(f"{binding.node_id} binding span does not match graph")
            if self.graph.prompt[span.start:span.end] != binding.source_text:
                raise ValueError(f"{binding.node_id} binding text does not match prompt")
            if (
                binding.evidence_class is EvidenceClass.OPEN_ENDED
                and binding.population_scope is PopulationScope.GT_TARGETS
            ):
                raise ValueError("open-ended requirements cannot use gt_targets scope")
            if isinstance(node, EntityNode):
                relevant_entities = {node.id}
            elif isinstance(node, PredicateNode):
                relevant_entities = {
                    edge.target_id for edge in self.graph.arguments_for(node.id)
                }
            else:
                relevant_entities = set()
            declared_entities = set(binding.entity_scopes)
            if declared_entities != relevant_entities:
                raise ValueError(
                    f"{binding.node_id} entity scopes must equal its direct entity "
                    f"arguments: missing={sorted(relevant_entities - declared_entities)}, "
                    f"extra={sorted(declared_entities - relevant_entities)}"
                )
            entity_scope_values = set(binding.entity_scopes.values())
            if (
                binding.evidence_class is EvidenceClass.OPEN_ENDED
                and PopulationScope.GT_TARGETS in entity_scope_values
            ):
                raise ValueError(
                    "open-ended requirement entity scopes cannot use gt_targets"
                )
        scope_by_entity: dict[str, PopulationScope] = {}
        for binding in requirements:
            for entity_id, scope in binding.entity_scopes.items():
                previous = scope_by_entity.setdefault(entity_id, scope)
                if previous is not scope:
                    raise ValueError(
                        f"entity {entity_id} has conflicting population scopes "
                        f"({previous.value}, {scope.value}); split it into distinct "
                        "graph entity nodes before freezing"
                    )
        requirements_by_node = {value.node_id: value for value in requirements}
        for evaluation in evaluations:
            requirement = requirements_by_node[evaluation.node_id]
            if evaluation.requirement_id != requirement.requirement_id:
                raise ValueError(
                    "EvaluationBinding requirement/node pair is inconsistent"
                )
            if evaluation.population_scope is not requirement.population_scope:
                raise ValueError("EvaluationBinding population_scope does not match")
            if dict(evaluation.entity_scopes) != dict(requirement.entity_scopes):
                raise ValueError("EvaluationBinding entity_scopes do not match")

        if self.review.get("approved") is not True:
            raise ValueError("frozen bundle requires an approved freeze decision")
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(
            self,
            "_evaluation_by_node",
            {value.node_id: value for value in evaluations},
        )
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "review", dict(self.review))
        object.__setattr__(self, "_by_node", {value.node_id: value for value in requirements})

    @property
    def prompt(self) -> str:
        return self.graph.prompt

    def binding_for(self, node_id: str) -> RequirementBinding:
        return self._by_node[str(node_id)]
    def evaluation_for(self, node_id: str) -> EvaluationBinding:
        return self._evaluation_by_node[str(node_id)]

    def semantic_nodes(self) -> tuple[str, ...]:
        supported = {
            value.node_id
            for value in self.requirements
            if value.status is RequirementStatus.SUPPORTED
        }
        return tuple(
            value.node_id
            for value in self.evaluations
            if value.evaluator_id.startswith("semantic_requirements.")
            and value.node_id in supported
        )

    def semantic_entity_scopes(self) -> Mapping[str, PopulationScope]:
        result: dict[str, PopulationScope] = {}
        for node_id in self.semantic_nodes():
            result.update(self.binding_for(node_id).entity_scopes)
        return result

    def _content_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "task_mode": self.task_mode.value,
            "graph": self.graph.to_dict(),
            "requirements": [value.to_dict() for value in self.requirements],
            "provenance": dict(self.provenance),
            "review": dict(self.review),
        }
        result["evaluations"] = [
            value.to_dict() for value in self.evaluations
        ]
        return result

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FrozenVerificationBundle:
        schema_version = str(value.get("schema_version") or "")
        graph = value.get("graph")
        requirements = value.get("requirements")
        evaluations = value.get("evaluations")
        if (
            not isinstance(graph, Mapping)
            or not isinstance(requirements, Sequence)
            or isinstance(requirements, (str, bytes))
            or any(not isinstance(item, Mapping) for item in requirements)
        ):
            raise ValueError("frozen bundle requires graph and requirement objects")
        if schema_version != "1.0":
            raise ValueError("unsupported frozen bundle schema_version")
        allowed = {
            "schema_version",
            "bundle_id",
            "task_mode",
            "graph",
            "requirements",
            "evaluations",
            "provenance",
            "review",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"bundle has unsupported fields {unknown}")
        if (
            not isinstance(evaluations, Sequence)
            or isinstance(evaluations, (str, bytes))
            or any(not isinstance(item, Mapping) for item in evaluations)
        ):
            raise ValueError("bundle evaluations must be an array of objects")
        return cls(
            schema_version=schema_version,
            bundle_id=value.get("bundle_id", ""),
            task_mode=value.get("task_mode", ""),
            graph=RequirementGraph.from_dict(graph),
            requirements=tuple(
                RequirementBinding.from_dict(item) for item in requirements
            ),
            evaluations=tuple(
                EvaluationBinding.from_dict(item)
                for item in evaluations
            ),
            provenance=value.get("provenance") or {},
            review=value.get("review") or {},
        )


__all__ = [
    "EvidenceClass",
    "FrozenVerificationBundle",
    "BindingSourceKind",
    "EvaluationBinding",
    "RequirementEvaluationStatus",
    "PopulationScope",
    "RequirementBinding",
    "RequirementStatus",
    "TaskMode",
    "UnknownReason",
    "make_evaluation_binding",
]
