"""Is the case specification well-formed, and is the scene it names?

These rules were the JavaScript runtime's `assertSceneGraph` and
`assertCaseSpec`. They fail CLOSED: a malformed assertion, a duplicate id, a
case id that belongs to another task, or a scene missing the fields a
population is selected on all produce one non-scoring result. None of them can
fail open into a measured pass — a validator that waves through what it does
not understand is how an unmeasured scene gets a score.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .values import is_finite_json_number, validate_triplet


_PRIMITIVES = frozenset({
    "structure", "source_preservation", "no_overlap", "compact_cluster",
    "clearance", "physics", "spatial_relation", "edit_locality",
    "physics_regression", "environment_consistency", "solid_penetration",
    "restoration_match",
})
#: Actor populations an assertion may name. `candidate_all` resolves from the
#: candidate alone; the rest need the input scene, and an assertion that names
#: one without a task supplying it is `not_evaluated` — see `assertions.py`.
_ACTOR_SCOPES = frozenset({
    "all_additions", "primary_additions", "companion_additions",
    "unexpected_additions", "candidate_all", "edited_actors", "source_actors",
})
_SELECTOR_ARRAY_FIELDS = (
    "allowed_asset_paths", "allowed_categories", "allowed_classes", "labels",
    "stable_actor_ids", "logical_object_ids", "actor_roles", "actor_origins",
    "required_tags", "excluded_asset_paths", "excluded_categories",
    "excluded_classes", "excluded_labels", "excluded_stable_actor_ids",
    "excluded_logical_object_ids",
)

def validate_candidate_contract(candidate: Any) -> list[str]:
    """Validate the scene fields that define the metric actor population.

    The rules intentionally match the existing JS scene-graph boundary while
    being stricter about JSON numbers (numeric strings are not evidence).
    """
    errors: list[str] = []
    if not isinstance(candidate, Mapping):
        return ["candidate must be an object"]
    actors = candidate.get("actors")
    if not isinstance(actors, list):
        return ["candidate.actors must be an array"]
    actor_count = candidate.get("actor_count")
    if actor_count is not None and (
        isinstance(actor_count, bool)
        or not isinstance(actor_count, int)
        or actor_count != len(actors)
    ):
        errors.append("candidate.actor_count must equal candidate.actors length")
    for index, actor in enumerate(actors):
        name = f"candidate.actors[{index}]"
        if not isinstance(actor, Mapping):
            errors.append(f"{name} must be an object")
            continue
        for field in ("label", "class"):
            if not isinstance(actor.get(field), str) or not actor[field]:
                errors.append(f"{name}.{field} must be a non-empty string")
        for field in ("actor_path", "stable_actor_id", "logical_object_id"):
            if actor.get(field) is not None and (
                not isinstance(actor[field], str) or not actor[field]
            ):
                errors.append(f"{name}.{field} must be a non-empty string when present")
        tags = actor.get("actor_tags")
        if tags is not None and (
            not isinstance(tags, list)
            or any(not isinstance(tag, str) or not tag for tag in tags)
        ):
            errors.append(f"{name}.actor_tags must be an array of non-empty strings")
        transform = actor.get("transform")
        if not isinstance(transform, Mapping):
            errors.append(f"{name}.transform is required")
        else:
            for field in ("location_cm", "rotation_deg", "scale"):
                validate_triplet(transform.get(field), f"{name}.transform.{field}", errors)
        bounds = actor.get("bounds")
        if not isinstance(bounds, Mapping):
            errors.append(f"{name}.bounds is required")
        else:
            for field in ("origin_cm", "extent_cm"):
                validate_triplet(bounds.get(field), f"{name}.bounds.{field}", errors)
        properties = actor.get("properties")
        if properties is not None and not isinstance(properties, Mapping):
            errors.append(f"{name}.properties must be an object when present")
        slots = actor.get("component_material_slots")
        if slots is not None:
            if not isinstance(slots, list):
                errors.append(f"{name}.component_material_slots must be an array")
            else:
                seen_slots: set[tuple[str, int]] = set()
                for slot_index, slot in enumerate(slots):
                    where = f"{name}.component_material_slots[{slot_index}]"
                    if not isinstance(slot, Mapping):
                        errors.append(f"{where} must be an object")
                        continue
                    component = slot.get("component_identity")
                    index_value = slot.get("slot_index")
                    if not isinstance(component, str) or not component:
                        errors.append(f"{where}.component_identity must be non-empty")
                    if (
                        isinstance(index_value, bool)
                        or not isinstance(index_value, int)
                        or index_value < 0
                    ):
                        errors.append(f"{where}.slot_index must be non-negative")
                    elif isinstance(component, str):
                        key = (component, index_value)
                        if key in seen_slots:
                            errors.append(f"{where} duplicates component/slot identity")
                        seen_slots.add(key)
                    material = slot.get("material_path")
                    if material is not None and (
                        not isinstance(material, str) or not material
                    ):
                        errors.append(f"{where}.material_path must be non-empty or null")
                    if not isinstance(slot.get("is_dynamic"), bool):
                        errors.append(f"{where}.is_dynamic must be boolean")
    return errors


def _validate_selector_contract(
    selector: Any, name: str, errors: list[str]
) -> None:
    if not isinstance(selector, Mapping):
        errors.append(f"{name} must be an object")
        return
    scope = selector.get("scope", "candidate_all")
    if scope not in _ACTOR_SCOPES:
        errors.append(f"{name}.scope has unsupported value {scope!r}")
    for field in _SELECTOR_ARRAY_FIELDS:
        value = selector.get(field)
        if value is not None and (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
        ):
            errors.append(f"{name}.{field} must be an array of non-empty strings")


def _validate_structure_contract(
    assertion: Mapping[str, Any], name: str, errors: list[str],
) -> None:
    """A structure assertion's rules must say what they require.

    This was the hole the validator's own docstring promised to close and did
    not. `concepts`, `companion_rules` and `compound_rules` went unread, and
    the classifier downstream reads a missing `count` as `int(None or 0)` — so
    a case-authoring typo turned "exactly four chairs" into "exactly zero
    chairs" and scored a chairless scene 1.0. A rule that is not an object was
    dropped from the check list AND from the score's denominator, which
    inflated the rate for the rules that remained.

    Fails closed instead, on the whole assertion: a case whose requirements
    cannot be read is not a case whose requirements were met.
    """
    raw = assertion.get("expected_raw_added_count")
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int)):
        errors.append(f"{name}.expected_raw_added_count must be a JSON integer")
    for field, required in (("concepts", ("concept", "count")),
                            ("companion_rules", ("id", "actor_class", "count")),
                            ("compound_rules", ("logical_object_id",
                                                "required_roles"))):
        rules = assertion.get(field)
        if rules is None:
            continue
        if not isinstance(rules, list):
            errors.append(f"{name}.{field} must be an array")
            continue
        for index, rule in enumerate(rules):
            where = f"{name}.{field}[{index}]"
            if not isinstance(rule, Mapping):
                errors.append(f"{where} must be an object")
                continue
            for key in required:
                if key not in rule:
                    errors.append(f"{where} must declare {key}")
                    continue
                value = rule[key]
                if key == "count" and (isinstance(value, bool)
                                       or not isinstance(value, int)
                                       or value < 0):
                    errors.append(
                        f"{where}.count must be a non-negative JSON integer; a "
                        f"missing or coerced count reads as `exactly zero "
                        f"required`, which passes for a scene containing none")
                elif key == "required_roles" and (
                        not isinstance(value, list) or not value
                        or any(not isinstance(item, str) or not item
                               for item in value)):
                    errors.append(
                        f"{where}.required_roles must be a non-empty array of "
                        f"non-empty strings")
                elif key in ("concept", "id", "actor_class",
                             "logical_object_id") and (
                        not isinstance(value, str) or not value.strip()):
                    errors.append(f"{where}.{key} must be a non-empty string")


def validate_case_contract(
    case_spec: Any, expected_case_id: str | None = None,
) -> tuple[list[str], list[Mapping[str, Any]]]:
    """Fail closed before a case can select or normalize an metric result."""
    if case_spec is None:
        return [], []
    errors: list[str] = []
    if not isinstance(case_spec, Mapping):
        return ["case spec must be an object"], []
    if not isinstance(case_spec.get("schema_version"), str) or not case_spec["schema_version"]:
        errors.append("case spec schema_version must be a non-empty string")
    case_id = case_spec.get("case_id")
    if (
        not isinstance(case_id, str)
        or not case_id
        or re.fullmatch(r"[A-Za-z0-9._-]+", case_id) is None
    ):
        errors.append("case spec case_id must be a non-empty path-safe string")
    elif expected_case_id is not None and case_id != expected_case_id:
        errors.append(
            f"case spec case_id {case_id!r} does not match current case "
            f"{expected_case_id!r}"
        )
    assertions_value = case_spec.get("assertions")
    if not isinstance(assertions_value, list) or not assertions_value:
        errors.append("case spec assertions must be a non-empty array")
        return errors, []
    assertions: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for index, assertion in enumerate(assertions_value):
        name = f"case spec assertions[{index}]"
        if not isinstance(assertion, Mapping):
            errors.append(f"{name} must be an object")
            continue
        assertions.append(assertion)
        requirement_id = assertion.get("id")
        if not isinstance(requirement_id, str) or not requirement_id:
            errors.append(f"{name}.id must be a non-empty string")
        elif requirement_id in seen_ids:
            errors.append(f"duplicate assertion id {requirement_id}")
        else:
            seen_ids.add(requirement_id)
        primitive = assertion.get("primitive")
        if primitive not in _PRIMITIVES:
            errors.append(f"{name}.primitive has unsupported value {primitive!r}")
            continue
        # Every primitive names its population the same way, so the population
        # contract is checked the same way for all of them. This used to admit
        # `target_selector` only on the two primitives that had a runtime,
        # which meant a case could not even DECLARE a selector for the rest —
        # the validator was enforcing what was implemented rather than what is
        # well-formed, and it would have had to be edited for each port.
        selector = assertion.get("target_selector")
        scope = assertion.get("scope", "primary_additions")
        if scope not in _ACTOR_SCOPES:
            errors.append(f"{name}.scope has unsupported value {scope!r}")
        if selector is not None:
            _validate_selector_contract(selector, f"{name}.target_selector", errors)
        if primitive == "structure":
            _validate_structure_contract(assertion, name, errors)
        if primitive != "physics":
            continue
        support_model = assertion.get("support_model")
        if support_model is not None and support_model not in {
            "ground_only_v1", "ground_or_lateral_v1"
        }:
            errors.append(
                f"{name}.support_model must be ground_only_v1 or "
                "ground_or_lateral_v1"
            )
        for field in ("maximum_penetration_cm", "minimum_support_fraction"):
            value = assertion.get(field)
            if not is_finite_json_number(value) or float(value) < 0:
                errors.append(f"{name}.{field} must be a non-negative JSON number")
            elif field == "minimum_support_fraction" and float(value) > 1:
                errors.append(f"{name}.{field} must be <= 1")
    return errors, assertions


__all__ = ["validate_candidate_contract", "validate_case_contract"]
