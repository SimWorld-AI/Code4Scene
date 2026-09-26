"""Which added Actor answers which part of what the task asked for.

A `structure` assertion names the CONCEPTS a scene must contain — "exactly one
dining table", "exactly four chairs" — and says, per concept, which asset
paths, semantic categories or Actor classes count as that concept. Sorting the
edit's additions into those concepts is the step three verifiers and the scope
resolver all need, so it happens once, here.

Two rules in the sorting decide scores and are easy to read past:

* a concept rule matches an Actor if ANY of its three selectors does. The
  selectors are alternative spellings of one identity — this asset, or this
  category, or this class — not conditions to be met together.
* an Actor matching MORE THAN ONE concept rule counts for neither. Ambiguity
  is the case author's problem to fix, and crediting an Actor to whichever
  rule was listed first would make the score depend on declaration order.

Everything left over is either a declared companion (a component Actor of a
compound object, matched by class) or unexpected. Unexpected additions are the
closed-world half of the contract: a scene that contains what was asked for
AND a hundred things that were not is not the scene that was asked for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .selection import (category_from_actor, normalize_asset_path,
                        normalize_category, normalize_class)


def slug(value: Any) -> str:
    """A check-id fragment: lowercase, and nothing that needs escaping."""
    return re.sub(r"^_|_$", "", re.sub(r"[^a-z0-9]+", "_", str(value), flags=re.I)).lower()


@dataclass(frozen=True)
class ConceptRule:
    concept: str
    count: int
    allowed_asset_paths: tuple[str, ...] = ()
    allowed_categories: tuple[str, ...] = ()
    allowed_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompanionRule:
    id: str
    actor_class: str
    count: int


@dataclass(frozen=True)
class CompoundRule:
    logical_object_id: str
    required_roles: tuple[str, ...]


@dataclass(frozen=True)
class StructureSpec:
    """A `structure` assertion, read into the shape the classifier wants."""

    concepts: tuple[ConceptRule, ...] = ()
    companion_rules: tuple[CompanionRule, ...] = ()
    compound_rules: tuple[CompoundRule, ...] = ()
    expected_raw_added_count: int | None = None


@dataclass
class Classified:
    """Every addition, sorted. The lists are disjoint and cover the input."""

    concepts: dict[str, list[Mapping[str, Any]]] = field(default_factory=dict)
    companions: dict[str, list[Mapping[str, Any]]] = field(default_factory=dict)
    primary: list[Mapping[str, Any]] = field(default_factory=list)
    companion: list[Mapping[str, Any]] = field(default_factory=list)
    unexpected: list[Mapping[str, Any]] = field(default_factory=list)


def _strings(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, (list, tuple)) else ()


def structure_spec(assertions: Sequence[Mapping[str, Any]]) -> StructureSpec | None:
    """The first declared `structure` assertion, or None when there is none."""
    for assertion in assertions:
        if assertion.get("primitive") not in (None, "structure"):
            continue
        concepts = tuple(
            ConceptRule(
                concept=str(rule.get("concept")),
                count=int(rule.get("count") or 0),
                allowed_asset_paths=_strings(rule.get("allowed_asset_paths")),
                allowed_categories=_strings(rule.get("allowed_categories")),
                allowed_classes=_strings(rule.get("allowed_classes")))
            for rule in assertion.get("concepts") or [] if isinstance(rule, Mapping))
        companions = tuple(
            CompanionRule(id=str(rule.get("id")),
                          actor_class=str(rule.get("actor_class")),
                          count=int(rule.get("count") or 0))
            for rule in assertion.get("companion_rules") or []
            if isinstance(rule, Mapping))
        compounds = tuple(
            CompoundRule(logical_object_id=str(rule.get("logical_object_id")),
                         required_roles=_strings(rule.get("required_roles")))
            for rule in assertion.get("compound_rules") or []
            if isinstance(rule, Mapping))
        raw = assertion.get("expected_raw_added_count")
        return StructureSpec(
            concepts=concepts, companion_rules=companions, compound_rules=compounds,
            expected_raw_added_count=int(raw) if isinstance(raw, int)
            and not isinstance(raw, bool) else None)
    return None


def concept_rule_matches(actor: Mapping[str, Any], rule: ConceptRule) -> bool:
    """ANY of asset path, category or class — see the module docstring."""
    for observed, allowed, normalize in (
        (actor.get("asset_path"), rule.allowed_asset_paths, normalize_asset_path),
        (category_from_actor(actor), rule.allowed_categories, normalize_category),
        (actor.get("class"), rule.allowed_classes, normalize_class),
    ):
        value = normalize(observed)
        if value is not None and any(normalize(item) == value for item in allowed):
            return True
    return False


def classify_additions(added: Sequence[Mapping[str, Any]],
                       spec: StructureSpec) -> Classified:
    """Sort the edit's additions into concepts, companions and the rest."""
    result = Classified(concepts={rule.concept: [] for rule in spec.concepts},
                        companions={rule.id: [] for rule in spec.companion_rules})
    for actor in added:
        matching = [rule for rule in spec.concepts
                    if concept_rule_matches(actor, rule)]
        if len(matching) == 1:
            result.concepts[matching[0].concept].append(actor)
            result.primary.append(actor)
            continue
        companion = next((rule for rule in spec.companion_rules
                          if rule.actor_class == actor.get("class")), None)
        if companion is not None:
            result.companions[companion.id].append(actor)
            result.companion.append(actor)
            continue
        result.unexpected.append(actor)
    return result


__all__ = ["Classified", "CompanionRule", "CompoundRule", "ConceptRule",
           "StructureSpec", "classify_additions", "concept_rule_matches",
           "slug", "structure_spec"]
