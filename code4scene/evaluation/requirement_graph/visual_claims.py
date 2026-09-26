"""Strict claim-payload boundary shared by visual RequirementGraph judges.

Only prompt-grounded semantic fields are accepted.  Unknown keys are rejected
instead of being dropped so controller metadata cannot silently cross into a
VLM request.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

JSON = dict[str, Any]

_PREDICATE_TYPES = frozenset(
    {
        "existence",
        "attribute",
        "material",
        "spatial_relation",
        "count",
        "atmosphere",
        "scene_identity",
        "quantity",
        "set",
        "distribution",
        "composition",
        "style_bundle",
        "environment",
        "logic",
        "boundary",
    }
)
_POLARITIES = frozenset({"affirmative", "negated"})
_CONSTRAINT_OPERATORS = frozenset({"eq", "gte", "lte", "between"})


def _strict_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    path: str,
) -> None:
    keys = {str(key) for key in value}
    missing = required - keys
    unknown = keys - allowed
    if missing:
        raise ValueError(f"{path} is missing keys {sorted(missing)!r}")
    if unknown:
        raise ValueError(f"{path} contains forbidden keys {sorted(unknown)!r}")


def _required_text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _json_number(value: Any, *, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a finite JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be a finite JSON number")
    return value


def _sanitize_argument(value: Any, *, path: str) -> JSON:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    _strict_keys(
        value,
        allowed={"role", "ordinal", "claim_text", "entity_name"},
        required={"role", "ordinal", "claim_text", "entity_name"},
        path=path,
    )
    ordinal = value["ordinal"]
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError(f"{path}.ordinal must be a non-negative integer")
    return {
        "role": _required_text(value["role"], path=f"{path}.role"),
        "ordinal": ordinal,
        "claim_text": _required_text(value["claim_text"], path=f"{path}.claim_text"),
        "entity_name": _required_text(value["entity_name"], path=f"{path}.entity_name"),
    }


def _sanitize_constraint(value: Any, *, path: str) -> JSON:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    _strict_keys(
        value,
        allowed={"operator", "value", "upper_value"},
        required={"operator", "value", "upper_value"},
        path=path,
    )
    operator = _required_text(value["operator"], path=f"{path}.operator")
    if operator not in _CONSTRAINT_OPERATORS:
        raise ValueError(f"{path}.operator is invalid")
    upper = value["upper_value"]
    if upper is not None:
        upper = _json_number(upper, path=f"{path}.upper_value")
    return {
        "operator": operator,
        "value": _json_number(value["value"], path=f"{path}.value"),
        "upper_value": upper,
    }


def _nullable_text(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, path=path)


def _sanitize_semantic_dsl(value: Any, *, path: str) -> JSON:
    """Validate the identifier-free projection of one Stage0 v2 requirement."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    keys = {
        "requirement_type",
        "relation",
        "quantity",
        "scope",
        "logic",
        "qualifiers",
    }
    _strict_keys(value, allowed=keys, required=keys, path=path)
    requirement_type = _required_text(
        value["requirement_type"], path=f"{path}.requirement_type"
    )
    if requirement_type not in _PREDICATE_TYPES | {"presence"}:
        raise ValueError(f"{path}.requirement_type is invalid")

    quantity = value["quantity"]
    if quantity is not None:
        if not isinstance(quantity, Mapping):
            raise TypeError(f"{path}.quantity must be an object or null")
        quantity_keys = {
            "mode",
            "value",
            "lower",
            "upper",
            "qualitative",
            "unit",
        }
        _strict_keys(
            quantity,
            allowed=quantity_keys,
            required=quantity_keys,
            path=f"{path}.quantity",
        )
        quantity = {
            "mode": _required_text(
                quantity["mode"], path=f"{path}.quantity.mode"
            ),
            **{
                key: (
                    None
                    if quantity[key] is None
                    else _json_number(quantity[key], path=f"{path}.quantity.{key}")
                )
                for key in ("value", "lower", "upper")
            },
            "qualitative": _nullable_text(
                quantity["qualitative"], path=f"{path}.quantity.qualitative"
            ),
            "unit": _nullable_text(
                quantity["unit"], path=f"{path}.quantity.unit"
            ),
        }

    scope = value["scope"]
    if scope is not None:
        if not isinstance(scope, Mapping):
            raise TypeError(f"{path}.scope must be an object or null")
        _strict_keys(
            scope,
            allowed={"quantifier", "entity_claim_text"},
            required={"quantifier", "entity_claim_text"},
            path=f"{path}.scope",
        )
        scope = {
            "quantifier": _required_text(
                scope["quantifier"], path=f"{path}.scope.quantifier"
            ),
            "entity_claim_text": _nullable_text(
                scope["entity_claim_text"], path=f"{path}.scope.entity_claim_text"
            ),
        }

    logic = value["logic"]
    if logic is not None:
        if not isinstance(logic, Mapping):
            raise TypeError(f"{path}.logic must be an object or null")
        _strict_keys(
            logic,
            allowed={"operator", "operand_claims"},
            required={"operator", "operand_claims"},
            path=f"{path}.logic",
        )
        operands = logic["operand_claims"]
        if not isinstance(operands, list):
            raise TypeError(f"{path}.logic.operand_claims must be a list")
        logic = {
            "operator": _required_text(
                logic["operator"], path=f"{path}.logic.operator"
            ),
            "operand_claims": [
                _required_text(item, path=f"{path}.logic.operand_claims[{index}]")
                for index, item in enumerate(operands)
            ],
        }

    raw_qualifiers = value["qualifiers"]
    if not isinstance(raw_qualifiers, list):
        raise TypeError(f"{path}.qualifiers must be a list")
    qualifiers = []
    for index, qualifier in enumerate(raw_qualifiers):
        qualifier_path = f"{path}.qualifiers[{index}]"
        if not isinstance(qualifier, Mapping):
            raise TypeError(f"{qualifier_path} must be an object")
        _strict_keys(
            qualifier,
            allowed={"kind", "name", "source_text"},
            required={"kind", "name", "source_text"},
            path=qualifier_path,
        )
        qualifiers.append(
            {
                key: _required_text(qualifier[key], path=f"{qualifier_path}.{key}")
                for key in ("kind", "name", "source_text")
            }
        )
    return {
        "requirement_type": requirement_type,
        "relation": _nullable_text(value["relation"], path=f"{path}.relation"),
        "quantity": quantity,
        "scope": scope,
        "logic": logic,
        "qualifiers": qualifiers,
    }


def _sanitize_predicate(value: Mapping[str, Any], *, path: str) -> JSON:
    allowed = {
        "node_type",
        "claim_text",
        "predicate_name",
        "predicate_type",
        "polarity",
        "arguments",
        "constraint",
        "scopes",
        "semantic_dsl",
    }
    required = {
        "node_type",
        "claim_text",
        "predicate_name",
        "predicate_type",
        "polarity",
        "arguments",
    }
    _strict_keys(value, allowed=allowed, required=required, path=path)
    if value["node_type"] != "predicate":
        raise ValueError(f"{path}.node_type must be 'predicate'")
    predicate_type = _required_text(
        value["predicate_type"], path=f"{path}.predicate_type"
    )
    if predicate_type not in _PREDICATE_TYPES:
        raise ValueError(f"{path}.predicate_type is invalid")
    polarity = _required_text(value["polarity"], path=f"{path}.polarity")
    if polarity not in _POLARITIES:
        raise ValueError(f"{path}.polarity is invalid")
    raw_arguments = value["arguments"]
    if not isinstance(raw_arguments, list):
        raise TypeError(f"{path}.arguments must be a list")
    arguments = [
        _sanitize_argument(argument, path=f"{path}.arguments[{index}]")
        for index, argument in enumerate(raw_arguments)
    ]
    payload: JSON = {
        "node_type": "predicate",
        "claim_text": _required_text(value["claim_text"], path=f"{path}.claim_text"),
        "predicate_name": _required_text(
            value["predicate_name"], path=f"{path}.predicate_name"
        ),
        "predicate_type": predicate_type,
        "polarity": polarity,
        "arguments": arguments,
    }
    if "constraint" in value:
        payload["constraint"] = _sanitize_constraint(
            value["constraint"], path=f"{path}.constraint"
        )
    if "scopes" in value:
        raw_scopes = value["scopes"]
        if not isinstance(raw_scopes, list):
            raise ValueError(f"{path}.scopes must be a list")
        payload["scopes"] = [
            _sanitize_predicate(scope, path=f"{path}.scopes[{index}]")
            if isinstance(scope, Mapping)
            else _raise_value_error(f"{path}.scopes[{index}] must be an object")
            for index, scope in enumerate(raw_scopes)
        ]
    if "semantic_dsl" in value:
        payload["semantic_dsl"] = _sanitize_semantic_dsl(
            value["semantic_dsl"], path=f"{path}.semantic_dsl"
        )
    return payload


def _raise_value_error(message: str) -> Any:
    raise ValueError(message)


def sanitize_visual_claim_payload(value: Mapping[str, Any]) -> JSON:
    """Return a deep, strict copy of a graph-derived visual claim payload."""

    if not isinstance(value, Mapping):
        raise TypeError("claim_payload must be a mapping")
    node_type = value.get("node_type")
    if node_type == "entity":
        _strict_keys(
            value,
            allowed={"node_type", "claim_text", "entity_name"},
            required={"node_type", "claim_text", "entity_name"},
            path="claim_payload",
        )
        return {
            "node_type": "entity",
            "claim_text": _required_text(
                value["claim_text"], path="claim_payload.claim_text"
            ),
            "entity_name": _required_text(
                value["entity_name"], path="claim_payload.entity_name"
            ),
        }
    if node_type == "predicate":
        return _sanitize_predicate(value, path="claim_payload")
    raise ValueError("claim_payload.node_type must be 'entity' or 'predicate'")


__all__ = ["sanitize_visual_claim_payload"]
