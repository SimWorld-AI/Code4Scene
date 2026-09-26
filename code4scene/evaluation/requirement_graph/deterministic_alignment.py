"""Conservative runtime alignment of frozen rule items to atomic v2 leaves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .actor_inventory import normalize_identity_term
from .contracts import (
    EntityNode,
    Polarity,
    PredicateNode,
    PredicateType,
    RequirementGraph,
)


def _entity_identity_terms(
    graph: RequirementGraph,
    entity: EntityNode,
) -> set[str]:
    values = {entity.name, *entity.aliases}
    for member_id in entity.member_ids:
        member = graph.node(member_id)
        if isinstance(member, EntityNode):
            values.update((member.name, *member.aliases))
    return {
        normalized
        for value in values
        if (normalized := normalize_identity_term(value))
    }


def _selector_category(selector: Any) -> str | None:
    if not isinstance(selector, Mapping):
        return None
    categories = tuple(
        normalize_identity_term(value)
        for value in selector.get("allowed_categories") or ()
    )
    categories = tuple(value for value in categories if value)
    return categories[0] if len(categories) == 1 else None


def _item_contains_node(item: Mapping[str, Any], node: PredicateNode) -> bool:
    trace = item.get("traceability")
    span = trace.get("source_span") if isinstance(trace, Mapping) else None
    return (
        isinstance(span, Sequence)
        and not isinstance(span, (str, bytes))
        and len(span) == 2
        and int(span[0]) <= node.source_span.start
        and node.source_span.end <= int(span[1])
    )


def _argument_entity(
    graph: RequirementGraph,
    node: PredicateNode,
    roles: set[str],
    category: str | None,
) -> EntityNode | None:
    if category is None:
        return None
    matches = []
    for argument in graph.arguments_for(node.id):
        if argument.role not in roles:
            continue
        entity = graph.node(argument.target_id)
        if (
            isinstance(entity, EntityNode)
            and category in _entity_identity_terms(graph, entity)
        ):
            matches.append(entity)
    return matches[0] if len(matches) == 1 else None


def _is_unqualified(node: PredicateNode) -> bool:
    semantic = node.semantic_parameters
    return (
        not semantic.get("qualifiers")
        and semantic.get("logic") is None
        and semantic.get("scope") is None
    )


def _quantity_matches_rule(
    node: PredicateNode,
    expected: Mapping[str, Any],
) -> bool:
    quantity = node.semantic_parameters.get("quantity")
    if not isinstance(quantity, Mapping):
        return False
    mode = str(quantity.get("mode") or "")
    if mode == "exact":
        return expected.get("exact_count") == quantity.get("value")
    if mode == "minimum":
        return expected.get("min_count") == quantity.get("value")
    if mode == "maximum":
        return expected.get("max_count") == quantity.get("value")
    if mode == "range":
        return (
            expected.get("min_count") == quantity.get("lower")
            and expected.get("max_count") == quantity.get("upper")
        )
    return False


def _rule_matches_node(
    graph: RequirementGraph,
    item: Mapping[str, Any],
    node: PredicateNode,
) -> bool:
    if not _item_contains_node(item, node) or not _is_unqualified(node):
        return False
    item_type = str(item.get("type") or "")
    subject = _argument_entity(
        graph,
        node,
        {"subject", "collection"},
        _selector_category(item.get("subject")),
    )
    if subject is None:
        return False
    semantic = node.semantic_parameters
    requirement_type = str(
        semantic.get("requirement_type") or node.predicate_type.value
    )
    if item_type in {"object_presence", "forbidden_object"}:
        wanted_polarity = (
            Polarity.NEGATED
            if item_type == "forbidden_object"
            else Polarity.AFFIRMATIVE
        )
        return (
            node.predicate_type is PredicateType.EXISTENCE
            and requirement_type == "presence"
            and node.polarity is wanted_polarity
            and semantic.get("relation") is None
            and semantic.get("quantity") is None
        )
    if item_type == "object_count":
        expected = item.get("expected")
        return (
            node.predicate_type in {PredicateType.COUNT, PredicateType.QUANTITY}
            and requirement_type == "quantity"
            and isinstance(expected, Mapping)
            and _quantity_matches_rule(node, expected)
            and semantic.get("relation") is None
        )
    if item_type not in {
        "object_object_relation",
        "object_architecture_relation",
        "around",
    }:
        return False
    expected = item.get("expected")
    if not isinstance(expected, Mapping):
        return False
    relation = str(expected.get("relation") or item_type).strip().casefold()
    authored_relation = str(semantic.get("relation") or "").strip().casefold()
    if (
        node.predicate_type is not PredicateType.SPATIAL_RELATION
        or requirement_type != "spatial_relation"
        or relation != authored_relation
    ):
        return False
    return _argument_entity(
        graph,
        node,
        {"object", "reference", "target", "anchor"},
        _selector_category(item.get("object")),
    ) is not None


def align_deterministic_rules(
    graph: RequirementGraph,
    rule_draft: Mapping[str, Any],
) -> Mapping[str, str]:
    """Return unique, lossless runtime rule bindings for atomic v2 leaves."""

    if graph.schema_version != "2.0":
        return {}
    items = tuple(
        value
        for value in rule_draft.get("items") or ()
        if isinstance(value, Mapping) and value.get("id")
    )
    predicates = tuple(
        value for value in graph.nodes if isinstance(value, PredicateNode)
    )
    rules_by_node: dict[str, list[str]] = {}
    for item in items:
        if (
            item.get("type") == "attribute_value"
            and item.get("preferred_evaluator") == "vlm"
        ):
            continue
        matches = tuple(
            node.id
            for node in predicates
            if _rule_matches_node(graph, item, node)
        )
        if len(matches) == 1:
            rules_by_node.setdefault(matches[0], []).append(str(item["id"]))
    return {
        node_id: rule_ids[0]
        for node_id, rule_ids in rules_by_node.items()
        if len(rule_ids) == 1
    }


__all__ = ["align_deterministic_rules"]
