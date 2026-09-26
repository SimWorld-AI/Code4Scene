"""Offline migration from legacy assertion contracts to frozen artifacts.

This module is authoring-only.  The scorer consumes the emitted
``verification_bundle`` and ``evaluation_policy``; it never reads the legacy
contract again.  Semantic assertions become typed graph leaves whose binding
contains the validated atomic parameters.  Preservation and physics settings
move to the separate default-policy artifact.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from code4scene.evaluation.case_spec import validate_case_contract
from code4scene.evaluation.evaluation_policy import (
    FrozenEvaluationPolicy,
    default_policy,
)

from .bundle import (
    BindingSourceKind,
    EvaluationBinding,
    EvidenceClass,
    FrozenVerificationBundle,
    PopulationScope,
    RequirementBinding,
    TaskMode,
)
from .contracts import (
    ArgumentEdge,
    ComparisonOperator,
    EntityNode,
    MemberRole,
    NumericConstraint,
    PredicateNode,
    PredicateType,
    ReferentKind,
    RequirementGraph,
    RequirementMemberEdge,
    RequirementNode,
    RootRequirement,
    SourceSpan,
)
from .legacy_atomic import EVALUATOR_ID, EVALUATOR_VERSION


MIGRATION_ALGORITHM = "legacy-contract-to-reviewed-bundle-v1"

_SEMANTIC_PRIMITIVES = frozenset(
    {"structure", "no_overlap", "clearance", "compact_cluster", "spatial_relation"}
)
_PRESERVATION_PRIMITIVES = frozenset({"source_preservation", "edit_locality"})
_PHYSICS_PRIMITIVES = frozenset(
    {
        "physics",
        "environment_consistency",
        "solid_penetration",
        "physics_regression",
    }
)


@dataclass(frozen=True, slots=True)
class LegacyMigrationResult:
    verification_bundle: FrozenVerificationBundle | None
    evaluation_policy: FrozenEvaluationPolicy
    migration_manifest: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "migration_algorithm": MIGRATION_ALGORITHM,
            "verification_bundle": (
                self.verification_bundle.to_dict()
                if self.verification_bundle is not None
                else None
            ),
            "evaluation_policy": self.evaluation_policy.to_dict(),
            "migration_manifest": dict(self.migration_manifest),
        }


@dataclass(frozen=True, slots=True)
class _Leaf:
    leaf_id: str
    label: str
    verifier: str
    assertion: Mapping[str, Any]
    subject_selector: Mapping[str, Any]
    object_selector: Mapping[str, Any] | None = None
    count: int | None = None


def _safe_id(prefix: str, value: Any) -> str:
    token = re.sub(r"[^a-z0-9_]+", "_", str(value).casefold()).strip("_")
    token = token or "leaf"
    if not token[0].isalpha():
        token = "n_" + token
    value = f"{prefix}_{token}"
    if len(value) > 64:
        value = value[:48].rstrip("_") + "_" + value[-15:].lstrip("_")
    return value


def _copy_assertion(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(dict(value))
    # Canonical JSON is both a finite-number check and a guarantee that the
    # frozen EvaluationBinding carries no YAML-specific Python objects.
    json.dumps(copied, allow_nan=False, sort_keys=True)
    return copied


def _structure_leaves(assertion: Mapping[str, Any]) -> list[_Leaf]:
    identifier = str(assertion.get("id") or "structure")
    leaves: list[_Leaf] = []
    raw_count = assertion.get("expected_raw_added_count")
    if isinstance(raw_count, int) and not isinstance(raw_count, bool):
        leaves.append(
            _Leaf(
                f"{identifier}_actor_count",
                "raw added Actor count",
                "structure_count",
                _copy_assertion(assertion),
                {"scope": "all_additions"},
                count=raw_count,
            )
        )
    for field, label_field in (
        ("concepts", "concept"),
        ("companion_rules", "id"),
        ("compound_rules", "logical_object_id"),
    ):
        for index, rule in enumerate(assertion.get(field) or ()):
            if not isinstance(rule, Mapping):
                continue
            isolated = _copy_assertion(assertion)
            isolated["concepts"] = []
            isolated["companion_rules"] = []
            isolated["compound_rules"] = []
            isolated[field] = [_copy_assertion(rule)]
            label = str(rule.get(label_field) or f"{field}_{index + 1}")
            selector = {
                key: copy.deepcopy(rule[key])
                for key in (
                    "allowed_asset_paths",
                    "allowed_categories",
                    "allowed_classes",
                )
                if rule.get(key)
            }
            selector["scope"] = "primary_additions"
            expected = rule.get("count")
            count = (
                expected
                if isinstance(expected, int) and not isinstance(expected, bool)
                else None
            )
            leaves.append(
                _Leaf(
                    f"{identifier}_{field}_{label}",
                    label,
                    "structure_concepts",
                    isolated,
                    selector,
                    count=count,
                )
            )
    leaves.append(
        _Leaf(
            f"{identifier}_allowed_additions",
            "required and forbidden additions",
            "structure_additions",
            _copy_assertion(assertion),
            {"scope": "all_additions"},
        )
    )
    return leaves


def _semantic_leaves(assertions: Sequence[Mapping[str, Any]]) -> list[_Leaf]:
    leaves: list[_Leaf] = []
    verifier_by_primitive = {
        "no_overlap": "spatial_overlap",
        "clearance": "spatial_clearance",
        "compact_cluster": "spatial_cluster",
    }
    for assertion in assertions:
        primitive = str(assertion.get("primitive") or "")
        identifier = str(assertion.get("id") or primitive)
        if primitive == "structure":
            leaves.extend(_structure_leaves(assertion))
        elif primitive in verifier_by_primitive:
            leaves.append(
                _Leaf(
                    identifier,
                    identifier,
                    verifier_by_primitive[primitive],
                    _copy_assertion(assertion),
                    copy.deepcopy(
                        assertion.get("target_selector")
                        or {"scope": assertion.get("scope", "primary_additions")}
                    ),
                )
            )
        elif primitive == "spatial_relation":
            for index, relation in enumerate(assertion.get("relations") or ()):
                if not isinstance(relation, Mapping):
                    continue
                relation_id = str(relation.get("id") or f"relation_{index + 1}")
                isolated = _copy_assertion(assertion)
                isolated["relations"] = [_copy_assertion(relation)]
                leaves.append(
                    _Leaf(
                        f"{identifier}_{relation_id}",
                        f"{relation.get('relation', 'spatial relation')}: {relation_id}",
                        "spatial_relations",
                        isolated,
                        copy.deepcopy(relation.get("subject") or {}),
                        copy.deepcopy(relation.get("object") or {}),
                    )
                )
    seen: set[str] = set()
    unique: list[_Leaf] = []
    for index, leaf in enumerate(leaves, start=1):
        leaf_id = _safe_id("legacy", leaf.leaf_id)
        if leaf_id in seen:
            leaf_id = _safe_id("legacy", f"{leaf.leaf_id}_{index}")
        seen.add(leaf_id)
        unique.append(
            _Leaf(
                leaf_id,
                leaf.label,
                leaf.verifier,
                leaf.assertion,
                leaf.subject_selector,
                leaf.object_selector,
                leaf.count,
            )
        )
    return unique


def _population_scope(selector: Mapping[str, Any]) -> PopulationScope:
    scope = str(selector.get("scope") or "candidate_all")
    if scope in {
        "all_additions",
        "primary_additions",
        "companion_additions",
        "unexpected_additions",
    }:
        return PopulationScope.ADDITIONS
    return PopulationScope.CANDIDATE_ALL


def _selector_name(selector: Mapping[str, Any], fallback: str) -> tuple[str, ...]:
    values: list[str] = []
    for field in (
        "allowed_categories",
        "labels",
        "allowed_classes",
        "allowed_asset_paths",
        "stable_actor_ids",
        "logical_object_ids",
    ):
        for value in selector.get(field) or ():
            text = str(value).strip()
            if text and text not in values:
                values.append(text)
    return tuple(values or (fallback,))


def _predicate_type(leaf: _Leaf) -> PredicateType:
    if leaf.count is not None:
        return PredicateType.COUNT
    if leaf.verifier == "spatial_relations":
        return PredicateType.SPATIAL_RELATION
    return PredicateType.ATTRIBUTE


def _build_bundle(
    *,
    contract: Mapping[str, Any],
    prompt: str,
    task_mode: TaskMode,
    bundle_id: str,
    leaves: Sequence[_Leaf],
) -> FrozenVerificationBundle:
    start = len(prompt) - len(prompt.lstrip())
    end = len(prompt.rstrip())
    if end <= start:
        raise ValueError("migration prompt must be non-empty")
    span = SourceSpan(start, end)
    text = prompt[start:end]
    root = RequirementNode(
        id="legacy_contract_requirement",
        text=text,
        source_span=span,
    )
    nodes: list[Any] = [root]
    edges: list[Any] = []
    requirements: list[RequirementBinding] = []
    evaluations: list[EvaluationBinding] = []
    weight = 1.0 / len(leaves)

    for index, leaf in enumerate(leaves, start=1):
        predicate_id = _safe_id("predicate", leaf.leaf_id)
        subject_id = _safe_id("entity", f"{leaf.leaf_id}_subject")
        subject_names = _selector_name(leaf.subject_selector, leaf.label)
        subject = EntityNode(
            id=subject_id,
            text=text,
            name=subject_names[0],
            aliases=subject_names[1:],
            referent_kind=ReferentKind.COLLECTION,
            source_span=span,
        )
        predicate_type = _predicate_type(leaf)
        predicate = PredicateNode(
            id=predicate_id,
            text=text,
            name=leaf.label,
            predicate_type=predicate_type,
            constraint=(
                NumericConstraint(ComparisonOperator.EQ, leaf.count)
                if leaf.count is not None
                else None
            ),
            source_span=span,
        )
        nodes.extend((predicate, subject))
        edges.append(
            ArgumentEdge(
                predicate.id,
                subject.id,
                "collection" if predicate_type is PredicateType.COUNT else "subject",
                0,
            )
        )
        edges.append(
            RequirementMemberEdge(
                root.id, predicate.id, MemberRole.SCORED_FACET, weight
            )
        )
        edges.append(
            RequirementMemberEdge(root.id, subject.id, MemberRole.SUPPORT_ONLY)
        )
        entity_scopes = {subject.id: _population_scope(leaf.subject_selector)}
        if leaf.object_selector is not None:
            object_id = _safe_id("entity", f"{leaf.leaf_id}_object")
            object_names = _selector_name(leaf.object_selector, "reference object")
            object_node = EntityNode(
                id=object_id,
                text=text,
                name=object_names[0],
                aliases=object_names[1:],
                referent_kind=ReferentKind.COLLECTION,
                source_span=span,
            )
            nodes.append(object_node)
            edges.append(ArgumentEdge(predicate.id, object_node.id, "reference", 1))
            edges.append(
                RequirementMemberEdge(
                    root.id, object_node.id, MemberRole.SUPPORT_ONLY
                )
            )
            entity_scopes[object_node.id] = _population_scope(leaf.object_selector)
        population_scope = entity_scopes[subject.id]
        requirement_id = _safe_id("requirement", f"{index}_{leaf.leaf_id}")
        requirement = RequirementBinding(
            requirement_id=requirement_id,
            node_id=predicate.id,
            source_span=(span.start, span.end),
            source_text=text,
            primary_owner="requirement_graph",
            evidence_class=EvidenceClass.OPEN_ENDED,
            population_scope=population_scope,
            entity_scopes=entity_scopes,
        )
        required_evidence = [
            "candidate_scene_graph",
            "requirement_graph",
            "evaluation_binding",
        ]
        if population_scope is PopulationScope.ADDITIONS or any(
            scope is PopulationScope.ADDITIONS for scope in entity_scopes.values()
        ):
            required_evidence.append("source_snapshot")
        evaluation = EvaluationBinding(
            binding_id=_safe_id("evaluation", f"{index}_{leaf.leaf_id}"),
            requirement_id=requirement.requirement_id,
            node_id=requirement.node_id,
            evaluator_id=EVALUATOR_ID,
            evaluator_version=EVALUATOR_VERSION,
            parameters={
                "legacy_verifier": leaf.verifier,
                "assertion": dict(leaf.assertion),
                "measurement_semantics": "exact_legacy_atomic_algorithm",
            },
            population_scope=requirement.population_scope,
            entity_scopes=requirement.entity_scopes,
            required_evidence=tuple(required_evidence),
            source_provenance={
                "kind": BindingSourceKind.LEGACY_CONTRACT.value,
                "migration_source": "offline_legacy_contract",
            },
        )
        requirements.append(requirement)
        evaluations.append(evaluation)

    graph = RequirementGraph(
        prompt=prompt,
        nodes=tuple(nodes),
        edges=tuple(edges),
        roots=(RootRequirement(root.id, 1.0),),
    )
    provenance = {
        "authoring_parser": "offline_legacy_contract_migration",
        "migration_algorithm": MIGRATION_ALGORITHM,
        "legacy_contract_case_id": contract.get("case_id"),
    }
    return FrozenVerificationBundle(
        schema_version="1.0",
        bundle_id=bundle_id,
        task_mode=task_mode,
        graph=graph,
        requirements=tuple(requirements),
        evaluations=tuple(evaluations),
        provenance=provenance,
        review={
            "approved": True,
            "decision": "deterministic_offline_migration",
            "migration_algorithm": MIGRATION_ALGORITHM,
        },
    )


def _physics_profile(assertions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    profile = dict(default_policy(None).physics_profile)
    relevant = [
        value
        for value in assertions
        if value.get("primitive") in _PHYSICS_PRIMITIVES
    ]
    if not relevant:
        return profile
    selectors = []
    for assertion in relevant:
        selector = assertion.get("target_selector") or {"scope": "candidate_all"}
        if not isinstance(selector, Mapping):
            raise ValueError("legacy physics target_selector must be an object")
        if selector not in selectors:
            selectors.append(selector)
    if len(selectors) != 1:
        raise ValueError(
            "legacy physics assertions use different target selectors; one "
            "frozen physics_profile cannot preserve their populations"
        )
    profile["selector"] = copy.deepcopy(dict(selectors[0]))
    field_names = {
        "maximum_ground_gap_cm",
        "maximum_penetration_cm",
        "minimum_support_fraction",
        "aabb_touch_tolerance_cm",
        "maximum_decisive_aabb_span_cm",
        "maximum_penetrating_actor_count",
        "maximum_fully_submerged_actor_count",
        "maximum_partially_submerged_actor_count",
        "waterline_tolerance_cm",
        "minimum_surface_span_cm",
        "maximum_thin_surface_extent_cm",
    }
    observed: dict[str, Any] = {}
    for assertion in relevant:
        for name in field_names:
            if name not in assertion:
                continue
            previous = observed.setdefault(name, assertion[name])
            if previous != assertion[name]:
                raise ValueError(
                    f"legacy physics assertions disagree on {name}; migration "
                    "requires one unambiguous frozen physics_profile"
                )
    profile.update(observed)
    policy_ids = {
        str(value)
        for value in (assertion.get("policy_id") for assertion in relevant)
        if value
    }
    profile["profile_id"] = (
        next(iter(policy_ids))
        if len(policy_ids) == 1
        else "legacy-contract-physics-profile-v1"
    )
    return profile


def _edit_scope(
    assertions: Sequence[Mapping[str, Any]],
    explicit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    relevant = [
        value
        for value in assertions
        if value.get("primitive") in _PRESERVATION_PRIMITIVES
    ]
    if explicit is not None:
        return copy.deepcopy(dict(explicit))
    nonzero_locality = []
    for assertion in relevant:
        if assertion.get("primitive") != "edit_locality":
            continue
        nonzero_locality.extend(
            name
            for name, value in assertion.items()
            if name.startswith("maximum_")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) != 0.0
        )
    if nonzero_locality:
        raise ValueError(
            "legacy edit_locality allowances do not identify authorized target "
            "Actors or operations; provide explicit_edit_scope rather than "
            f"guessing from {sorted(set(nonzero_locality))}"
        )
    return {}


def migrate_legacy_contract(
    contract: Mapping[str, Any],
    *,
    prompt: str,
    task_mode: TaskMode | str,
    bundle_id: str,
    source_snapshot: Mapping[str, Any] | str | None = None,
    explicit_edit_scope: Mapping[str, Any] | None = None,
    asset_library_manifest: Mapping[str, Any] | None = None,
) -> LegacyMigrationResult:
    """Validate once and emit the only two artifacts runtime scoring needs."""

    if not isinstance(contract, Mapping):
        raise TypeError("legacy contract must be an object")
    case_id = str(contract.get("case_id") or "")
    errors, assertions = validate_case_contract(contract, case_id or None)
    if errors:
        raise ValueError("invalid legacy contract: " + "; ".join(errors))
    unsupported = sorted(
        {
            str(value.get("primitive"))
            for value in assertions
            if value.get("primitive")
            not in _SEMANTIC_PRIMITIVES
            | _PRESERVATION_PRIMITIVES
            | _PHYSICS_PRIMITIVES
        }
    )
    if unsupported:
        raise ValueError(
            f"legacy contract contains primitives with no migration: {unsupported}"
        )
    mode = TaskMode(task_mode)
    if mode in {TaskMode.EDIT, TaskMode.REPAIR} and source_snapshot is None:
        raise ValueError(
            f"{mode.value} migration requires a frozen source_snapshot"
        )
    contract_copy = _copy_assertion(contract)
    leaves = _semantic_leaves(assertions)
    bundle = (
        _build_bundle(
            contract=contract_copy,
            prompt=prompt,
            task_mode=mode,
            bundle_id=bundle_id,
            leaves=leaves,
        )
        if leaves
        else None
    )
    policy = FrozenEvaluationPolicy(
        policy_id=_safe_id("legacy_contract", case_id or bundle_id),
        source_snapshot=source_snapshot,
        edit_scope=_edit_scope(assertions, explicit_edit_scope),
        physics_profile=_physics_profile(assertions),
        asset_library_manifest=asset_library_manifest or {},
        source="offline_legacy_contract_migration",
    )
    semantic_count = sum(
        value.get("primitive") in _SEMANTIC_PRIMITIVES for value in assertions
    )
    preservation_count = sum(
        value.get("primitive") in _PRESERVATION_PRIMITIVES for value in assertions
    )
    physics_count = sum(
        value.get("primitive") in _PHYSICS_PRIMITIVES for value in assertions
    )
    return LegacyMigrationResult(
        verification_bundle=bundle,
        evaluation_policy=policy,
        migration_manifest={
            "legacy_assertion_count": len(assertions),
            "semantic_assertion_count": semantic_count,
            "semantic_leaf_count": len(leaves),
            "preservation_assertion_count": preservation_count,
            "physics_assertion_count": physics_count,
            "runtime_legacy_contract_read_allowed": False,
            "canonical_task_verifiers": (
                ["semantic_requirements"] if bundle is not None else []
            ),
        },
    )


__all__ = [
    "LegacyMigrationResult",
    "MIGRATION_ALGORITHM",
    "migrate_legacy_contract",
]
