"""Stage 0 v2: prompt-faithful semantic IR and deterministic normalization.

This module deliberately does not ask whether a verifier can execute a claim.
It preserves the prompt as typed semantic requirements; capability planning is
a later concern.  The public RequirementGraph lowering retains the complete v2
payload on every predicate so a later planner never needs to reinterpret the
natural-language prompt.
"""

from __future__ import annotations

import math
import copy
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    ArgumentEdge,
    EntityEvaluationRoute,
    EntityGroundingMode,
    EntityNode,
    EntityType,
    MemberRole,
    Polarity,
    PredicateNode,
    PredicateType,
    ReferentKind,
    RequirementGraph,
    RequirementMemberEdge,
    RequirementNode,
    RootRequirement,
    ScopeEdge,
    SourceSpan,
)

JSON = dict[str, Any]

DRAFT_VERSION = "2.0"
TOOL_SCHEMA_VERSION = "2.5"
SYSTEM_PROMPT_REVISION = "2.5.1"
WEIGHT_POLICY = "equal_atomic_top_level_requirements_v4"
MAX_SCORED_REQUIREMENTS = 96

REQUIREMENT_TYPES = (
    "presence",
    "quantity",
    "set",
    "spatial_relation",
    "distribution",
    "composition",
    "style_bundle",
    "environment",
    "logic",
    "boundary",
    "attribute",
    "material",
    "scene_identity",
)
QUANTITY_MODES = (
    "exact",
    "approximately",
    "at_least",
    "at_most",
    "range",
    "more_than",
    "less_than",
    "qualitative",
)
QUALITATIVE_QUANTITIES = (
    "none",
    "almost_none",
    "few",
    "several",
    "many",
    "dense",
    "sparse",
)
SCOPE_QUANTIFIERS = (
    "global",
    "each",
    "every",
    "all",
    "some",
    "most",
    "few",
    "per",
)
LOGIC_OPERATORS = ("all_of", "any_of", "one_of", "not", "if_then", "unless")
QUALIFIER_KINDS = (
    "attribute",
    "material",
    "style",
    "weather",
    "lighting",
    "atmosphere",
    "terrain",
    "density",
    "layout",
    "boundary",
    "condition",
    "color",
    "shape",
)

_MAX_ENTITIES = 96
_MAX_REQUIREMENTS = 128
_MAX_ARGUMENTS = 24
_MAX_REFERENCES = 24
_MAX_QUALIFIERS = 32
_MAX_ALIASES = 12

_FUNCTION_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "build", "but", "by",
        "add", "arrange", "composed", "create", "define", "enough", "for", "formed", "from", "give", "has",
        "have", "include", "including", "into", "is", "it", "keep", "make",
        "in", "me", "of", "on", "or", "organize", "place", "put", "scene", "should", "the",
        "let", "mood", "plus", "resembling", "that", "their", "them", "then", "these", "this", "to", "use", "while", "with", "yet",
    }
)

_TYPE_TO_PREDICATE = {
    "presence": PredicateType.EXISTENCE,
    "quantity": PredicateType.QUANTITY,
    "set": PredicateType.SET,
    "spatial_relation": PredicateType.SPATIAL_RELATION,
    "distribution": PredicateType.DISTRIBUTION,
    "composition": PredicateType.COMPOSITION,
    "style_bundle": PredicateType.STYLE_BUNDLE,
    "environment": PredicateType.ENVIRONMENT,
    "logic": PredicateType.LOGIC,
    "boundary": PredicateType.BOUNDARY,
    "attribute": PredicateType.ATTRIBUTE,
    "material": PredicateType.MATERIAL,
    "scene_identity": PredicateType.SCENE_IDENTITY,
}

_ROLE_ALIASES = {
    "object": "reference",
    "target": "reference",
    "anchor": "reference",
    "items": "member",
    "item": "member",
    "members": "member",
    "set": "collection",
    "group": "collection",
    "thing": "subject",
}


class Stage0V2Error(ValueError):
    """A v2 syntax, grounding, reference, or semantic coverage error."""

    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = str(code)
        self.path = str(path)
        self.message = str(message)
        super().__init__(f"{self.code} at {self.path}: {self.message}")


SYSTEM_PROMPT = """You translate an untrusted text-to-scene description into Stage0 v2 Semantic IR.

Stage0 is semantic parsing only. Ignore the current verifier implementation and preserve every
explicit visual constraint. Never use an unsupported/waiver bucket. Return exactly one call to the
provided tool and no prose. Do not invent objects, attributes, counts, relations, or quality bars.

Produce entities for concrete referents, regions, surfaces, masses, and collections. A collection
uses referent_kind=collection. For an explicit coordinated list, set list_mode to and/or and either
provide member_keys or let the deterministic normalizer split its exact source span.

Separate an entity's base category from independently judgeable modifiers. Keep color, material,
shape, size, style, and other visible qualifiers in requirements instead of folding them into the
identity used for retrieval. Add only safe, category-equivalent, candidate-independent aliases
that a scene-local Actor or asset vocabulary may use. Never emit Candidate Actor ids or asset
paths.

Produce one requirement for every independently judgeable constraint. One top-level requirement
must express one independently falsifiable visual proposition: failure of one colour, material,
shape, condition, motion/flow appearance, layout fact, distribution fact, or spatial relation must
not force an otherwise satisfied proposition to zero. Split existence from appearance, split each
independently visible appearance facet, and split distinct spatial/layout consequences even when
they occur in the same sentence. Reuse the same entity arguments and exact source span when atomic
requirements need the same evidence. A coordinated palette or inseparable named style may remain
one claim only when its members are jointly defined by the prompt rather than independently
judgeable facts. Judge material by visible appearance; do not invent hidden physical composition.
For example, a coloured fast-flowing river that is winding, passes through a settlement, divides
dense districts, and has those districts on both banks requires separate claims for colour, visible
flow, winding shape, passing through the settlement, dividing the districts, district density, and
the both-banks relation. Shared words or evidence never make these one all-or-nothing proposition.

Every entity needs a grounding_mode:
- actor: one discrete visible object;
- actor_collection: a collection of objects or placeable surface/decal instances;
- derived_region: an area, extent, layout component, or boundary inferred from multiple scene
  elements rather than a same-named Actor;
- scene_global: the entire scene is the only meaningful target.
Grounding mode describes the evidence representation, not entity_type alone. A semantic region or
surface uses actor_collection when category-equivalent placeable instances can be retrieved and
their union grounds the referent. Use derived_region only when grounding requires geometric or
topological inference and no directly retrievable constituent category defines the referent.
Resolve anaphoric references to their intended entity or scene/layout referent in the requirements.
Do not model an abstract region as a same-named Actor merely to make it locatable.

Requirement count follows prompt content rather than a target count. A detailed scene will normally
produce more requirements than a simple scene, but never exceed 96 top-level scored requirements or
128 total requirements including logic operands. Give an independent requirement to exact or
bounded quantities, explicit negation or exclusion, only/exactly constraints, overall scene
identity, object groups, visible attributes/materials, layout/boundary structure, and spatial
relations. Requirements that share a subject or camera view must remain separate scored claims and
reuse their entity arguments; evidence sharing is handled later and is not a reason to merge scores.
Preserve every prompt constraint in entities, arguments, qualifiers, quantity, scope, logic, and
exact source spans.

Use this requirement_type vocabulary:
- presence: object/region presence or explicit absence via polarity=negated
- quantity: exact, approximate, lower/upper/range, or qualitative quantities such as few/almost none
- set: an object collection or explicit and/or list
- spatial_relation: around, along, inside, adjacent, connected, intersecting, facing, etc.; relation
  is a concise canonical relation and arguments carry subject/reference/path/region roles
- distribution: dense, sparse, uniform, clustered, scattered, interleaved, perimeter, network layout
- composition: high-level structures such as blocks, road networks, courtyards, street canyons
- style_bundle: a named style plus every explicitly listed observable feature in qualifiers
- environment: weather, lighting, atmosphere, terrain, and global scene conditions
- logic: any_of/one_of/not/if_then/unless over referenced requirement keys
- boundary: outer edge, termination, enclosure, zoning, or scene-limit requirements
- attribute/material/scene_identity: atomic visible properties, substances, and overall scene identity

Represent a coordinated collection as a set plus atomic member-presence claims when each named
member is an independently required visible object or group. Keep the concrete members as entities
or arguments for retrieval. A collection-level distribution, composition, count, or spatial fact
is a separate requirement from member presence. Global claims keep
their concrete arguments available for scene-local identity retrieval and targeted evidence. The
operands of any_of/one_of must be semantically distinct after normalization; never duplicate one
operand merely to cover both sides of an “or”. The owning any_of/one_of requirement must use
requirement_type=logic. Expand shared-affix alternatives semantically: for example, “L- or
cross-shaped” means distinct L-shaped and cross-shaped operands, and “two- or three-story” means
distinct two-story and three-story operands. Keep source_text exact and contiguous; when an
expanded form is not itself contiguous in the prompt, its operand may share the complete original
alternative phrase with its sibling while its qualifier name states the distinct canonical form.
Copy common observable modifiers into both operands.

Use quantity.mode=approximately for “about/roughly”, qualitative for few/several/many/dense/sparse/
almost none, and preserve numeric values or bounds when stated. Use scope for each/every/some/most/
few/per-group meanings. Represent “A or B” as atomic operands plus a logic requirement referencing
them; operands are dependencies of the logic claim, not independent top-level obligations.

source_text for every entity, requirement, and qualifier must be an exact case-sensitive contiguous
substring of the description; occurrence is zero-based. A requirement source span may be a complete
clause or sentence and may contain all of its argument mentions. Qualifiers preserve exact feature
phrases. IDs are local references only; the deterministic normalizer owns final IDs, source offsets,
alias de-duplication, role ordinals, list expansion, and reference rewriting.

Every meaningful prompt constraint must be represented by at least one entity, argument,
qualifier, quantity, scope, logic expression, or requirement span. Coverage does not require a
separate scored node for every content word. Do not emit scored_facets or unsupported_semantics.
Do not collapse a multi-sentence prompt into one generic scene_identity requirement.
"""


def _object(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage0V2Error("invalid_type", path, "must be an object")
    return dict(value)


def _list(value: Any, *, path: str, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise Stage0V2Error("invalid_type", path, "must be a list")
    if len(value) > maximum:
        raise Stage0V2Error("budget_exceeded", path, f"contains more than {maximum} items")
    return value


def _text(value: Any, *, path: str, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise Stage0V2Error("invalid_type", path, "must be a string")
    result = value.strip()
    if not result:
        raise Stage0V2Error("empty_text", path, "must be non-empty")
    if len(result) > maximum:
        raise Stage0V2Error("text_too_long", path, f"must be at most {maximum} codepoints")
    return result


def _enum(value: Any, allowed: Sequence[str], *, path: str) -> str:
    result = _text(value, path=path, maximum=64).casefold().replace("-", "_").replace(" ", "_")
    if result not in allowed:
        raise Stage0V2Error("invalid_enum", path, f"must be one of {list(allowed)!r}")
    return result


def _normalized_with_offsets(value: str) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    characters: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    pending_space = False
    for index, character in enumerate(value):
        piece = unicodedata.normalize("NFKC", character).casefold()
        for normalized in piece:
            if normalized.isspace():
                pending_space = bool(characters)
                continue
            if pending_space:
                characters.append(" ")
                starts.append(index)
                ends.append(index + 1)
                pending_space = False
            characters.append(normalized)
            starts.append(index)
            ends.append(index + 1)
    return "".join(characters), tuple(starts), tuple(ends)


def _all_starts(haystack: str, needle: str) -> list[int]:
    starts: list[int] = []
    cursor = 0
    while True:
        start = haystack.find(needle, cursor)
        if start < 0:
            return starts
        starts.append(start)
        cursor = start + 1


def _occurrence(prompt: str, source_text: str, start: int) -> int:
    starts = _all_starts(prompt, source_text)
    try:
        return starts.index(start)
    except ValueError as exc:  # pragma: no cover - internal invariant
        raise Stage0V2Error("ungrounded_text", "source_text", "normalizer lost source alignment") from exc


def _align_source(
    prompt: str,
    value: Any,
    occurrence: Any,
    *,
    path: str,
) -> tuple[str, int, SourceSpan, bool]:
    source = _text(value, path=f"{path}.source_text", maximum=len(prompt))
    if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 0:
        occurrence = 0
    exact = _all_starts(prompt, source)
    if occurrence < len(exact):
        start = exact[occurrence]
        return source, occurrence, SourceSpan(start, start + len(source)), False

    normalized_prompt, starts, ends = _normalized_with_offsets(prompt)
    normalized_source, _, _ = _normalized_with_offsets(source)
    matches = _all_starts(normalized_prompt, normalized_source)
    if not matches:
        raise Stage0V2Error(
            "ungrounded_text",
            f"{path}.source_text",
            "does not align to the prompt after Unicode/case/whitespace normalization",
        )
    chosen = matches[occurrence] if occurrence < len(matches) else matches[0]
    start = starts[chosen]
    end = ends[chosen + len(normalized_source) - 1]
    canonical = prompt[start:end].strip()
    start += len(prompt[start:end]) - len(prompt[start:end].lstrip())
    end = start + len(canonical)
    return canonical, _occurrence(prompt, canonical, start), SourceSpan(start, end), True


def _words(value: str) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize("NFKC", match.group(0)).casefold()
        for match in re.finditer(r"[^\W_]+", value, flags=re.UNICODE)
    )


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")[:40]


def _relation_name(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    text = _text(value, path=path, maximum=96)
    return re.sub(r"[^\w]+", "_", unicodedata.normalize("NFKC", text).casefold()).strip("_")


def _split_list_parts(source: str) -> tuple[tuple[int, int, str], ...]:
    parts: list[tuple[int, int, str]] = []
    cursor = 0
    for match in re.finditer(r"\s*(?:,|;|\band\b|\bor\b)\s*", source, re.IGNORECASE):
        start, end = cursor, match.start()
        while start < end and source[start].isspace():
            start += 1
        while end > start and source[end - 1].isspace():
            end -= 1
        if end > start:
            parts.append((start, end, source[start:end]))
        cursor = match.end()
    start, end = cursor, len(source)
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    if end > start:
        parts.append((start, end, source[start:end]))
    return tuple(parts) if len(parts) >= 2 else ()


def _parse_quantity(value: Any, *, path: str) -> JSON | None:
    if value is None:
        return None
    raw = _object(value, path=path)
    mode = _enum(raw.get("mode"), QUANTITY_MODES, path=f"{path}.mode")

    def number(name: str) -> int | float | None:
        item = raw.get(name)
        if item is None:
            return None
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise Stage0V2Error("invalid_number", f"{path}.{name}", "must be finite or null")
        return item

    result = {
        "mode": mode,
        "value": number("value"),
        "lower": number("lower"),
        "upper": number("upper"),
        "qualitative": None,
        "unit": None,
    }
    qualitative = raw.get("qualitative")
    if qualitative is not None:
        result["qualitative"] = _enum(
            qualitative, QUALITATIVE_QUANTITIES, path=f"{path}.qualitative"
        )
    unit = raw.get("unit")
    if unit is not None:
        result["unit"] = _text(unit, path=f"{path}.unit", maximum=64)
    if mode == "qualitative" and result["qualitative"] is None:
        raise Stage0V2Error("missing_qualitative", path, "qualitative mode requires a qualitative value")
    if mode == "range" and (result["lower"] is None or result["upper"] is None):
        raise Stage0V2Error("missing_range", path, "range mode requires lower and upper")
    if mode not in {"range", "qualitative"} and result["value"] is None:
        raise Stage0V2Error("missing_quantity", path, f"{mode} mode requires value")
    return result


def _parse_entity(prompt: str, value: Any, index: int, actions: list[JSON]) -> JSON:
    path = f"entities[{index}]"
    raw = _object(value, path=path)
    source, occurrence, span, realigned = _align_source(
        prompt, raw.get("source_text"), raw.get("occurrence", 0), path=path
    )
    if realigned:
        actions.append({"action": "source_span_realigned", "path": path})
    name_raw = str(raw.get("name") or "").strip()
    name = name_raw if name_raw else source
    aliases: list[str] = []
    seen = {unicodedata.normalize("NFKC", name).casefold()}
    for alias in _list(raw.get("aliases", []), path=f"{path}.aliases", maximum=_MAX_ALIASES):
        candidate = _text(alias, path=f"{path}.aliases", maximum=96)
        key = unicodedata.normalize("NFKC", candidate).casefold()
        if key in seen:
            actions.append({"action": "duplicate_alias_removed", "path": path, "value": candidate})
            continue
        seen.add(key)
        aliases.append(candidate)
    entity_type = _enum(
        raw.get("entity_type", EntityType.OBJECT.value),
        tuple(item.value for item in EntityType),
        path=f"{path}.entity_type",
    )
    referent = _enum(
        raw.get("referent_kind", ReferentKind.INDIVIDUAL.value),
        tuple(item.value for item in ReferentKind),
        path=f"{path}.referent_kind",
    )
    if entity_type == EntityType.REGION.value:
        default_grounding = EntityGroundingMode.DERIVED_REGION.value
    elif entity_type == EntityType.SURFACE.value:
        default_grounding = EntityGroundingMode.ACTOR_COLLECTION.value
    elif referent == ReferentKind.COLLECTION.value:
        default_grounding = EntityGroundingMode.ACTOR_COLLECTION.value
    else:
        default_grounding = EntityGroundingMode.ACTOR.value
    grounding_mode = _enum(
        raw.get("grounding_mode", default_grounding),
        tuple(
            item.value
            for item in EntityGroundingMode
            if item is not EntityGroundingMode.AUTO
        ),
        path=f"{path}.grounding_mode",
    )
    if (
        grounding_mode == EntityGroundingMode.ACTOR.value
        and referent == ReferentKind.COLLECTION.value
    ):
        grounding_mode = EntityGroundingMode.ACTOR_COLLECTION.value
        actions.append(
            {
                "action": "collection_grounding_promoted",
                "path": path,
                "grounding_mode": grounding_mode,
            }
        )
    list_mode = _enum(raw.get("list_mode", "none"), ("none", "and", "or"), path=f"{path}.list_mode")
    members = [str(item).strip() for item in _list(raw.get("member_keys", []), path=f"{path}.member_keys", maximum=24)]
    key = str(raw.get("key") or f"entity_{index + 1}").strip() or f"entity_{index + 1}"
    return {
        "_old_key": key,
        "source_text": source,
        "occurrence": occurrence,
        "source_span": {"start": span.start, "end": span.end},
        "name": name,
        "aliases": aliases,
        "entity_type": entity_type,
        "referent_kind": referent,
        "grounding_mode": grounding_mode,
        "list_mode": list_mode,
        "member_keys": members,
    }


def _expand_tagged_lists(prompt: str, entities: list[JSON], actions: list[JSON]) -> None:
    generated: list[JSON] = []
    for entity in entities:
        if entity["list_mode"] == "none" or entity["member_keys"]:
            continue
        parts = _split_list_parts(entity["source_text"])
        if not parts:
            continue
        parent_start = int(entity["source_span"]["start"])
        head = _words(parts[-1][2])[-1] if _words(parts[-1][2]) else "item"
        member_keys: list[str] = []
        for member_index, (local_start, local_end, text) in enumerate(parts, start=1):
            start = parent_start + local_start
            end = parent_start + local_end
            member_key = f"{entity['_old_key']}__member_{member_index}"
            member_keys.append(member_key)
            words = _words(text)
            name = text if len(words) > 1 or words[-1:] == (head,) else f"{text} {head}"
            generated.append(
                {
                    "_old_key": member_key,
                    "source_text": prompt[start:end],
                    "occurrence": _occurrence(prompt, prompt[start:end], start),
                    "source_span": {"start": start, "end": end},
                    "name": name,
                    "aliases": [],
                    "entity_type": entity["entity_type"],
                    "referent_kind": ReferentKind.INDIVIDUAL.value,
                    "grounding_mode": (
                        EntityGroundingMode.ACTOR.value
                        if entity["grounding_mode"]
                        == EntityGroundingMode.ACTOR_COLLECTION.value
                        else entity["grounding_mode"]
                    ),
                    "list_mode": "none",
                    "member_keys": [],
                }
            )
        entity["member_keys"] = member_keys
        entity["referent_kind"] = ReferentKind.COLLECTION.value
        actions.append(
            {
                "action": "coordinated_list_split",
                "entity_key": entity["_old_key"],
                "member_count": len(member_keys),
            }
        )
    entities.extend(generated)


def _parse_requirement(prompt: str, value: Any, index: int, actions: list[JSON]) -> JSON:
    path = f"requirements[{index}]"
    raw = _object(value, path=path)
    source, occurrence, span, realigned = _align_source(
        prompt, raw.get("source_text"), raw.get("occurrence", 0), path=path
    )
    if realigned:
        actions.append({"action": "source_span_realigned", "path": path})
    requirement_type = _enum(raw.get("requirement_type"), REQUIREMENT_TYPES, path=f"{path}.requirement_type")
    polarity = _enum(raw.get("polarity", "affirmative"), tuple(item.value for item in Polarity), path=f"{path}.polarity")
    arguments = []
    for argument_index, value in enumerate(
        _list(raw.get("arguments", []), path=f"{path}.arguments", maximum=_MAX_ARGUMENTS)
    ):
        argument = _object(value, path=f"{path}.arguments[{argument_index}]")
        entity_key = _text(argument.get("entity_key"), path=f"{path}.arguments[{argument_index}].entity_key", maximum=128)
        role_raw = str(argument.get("role") or "").strip().casefold().replace("-", "_").replace(" ", "_")
        role = _ROLE_ALIASES.get(role_raw, role_raw or "subject")
        arguments.append({"entity_key": entity_key, "role": role})
    relation = _relation_name(raw.get("relation"), path=f"{path}.relation")
    quantity = _parse_quantity(raw.get("quantity"), path=f"{path}.quantity")
    if requirement_type == "quantity" and quantity is None:
        raise Stage0V2Error("missing_quantity", path, "quantity requirement needs quantity")

    scope_raw = raw.get("scope")
    scope = None
    if scope_raw is not None:
        scope_value = _object(scope_raw, path=f"{path}.scope")
        scope = {
            "quantifier": _enum(scope_value.get("quantifier"), SCOPE_QUANTIFIERS, path=f"{path}.scope.quantifier"),
            "entity_key": str(scope_value.get("entity_key") or "").strip() or None,
        }

    logic_raw = raw.get("logic")
    logic = None
    if logic_raw is not None:
        logic_value = _object(logic_raw, path=f"{path}.logic")
        logic = {
            "operator": _enum(logic_value.get("operator"), LOGIC_OPERATORS, path=f"{path}.logic.operator"),
            "requirement_keys": [
                _text(item, path=f"{path}.logic.requirement_keys", maximum=128)
                for item in _list(logic_value.get("requirement_keys", []), path=f"{path}.logic.requirement_keys", maximum=_MAX_REFERENCES)
            ],
        }
    if requirement_type == "logic" and logic is None:
        raise Stage0V2Error("missing_logic", path, "logic requirement needs logic")

    qualifiers = []
    for qualifier_index, value in enumerate(
        _list(raw.get("qualifiers", []), path=f"{path}.qualifiers", maximum=_MAX_QUALIFIERS)
    ):
        qualifier_path = f"{path}.qualifiers[{qualifier_index}]"
        qualifier = _object(value, path=qualifier_path)
        q_text, q_occurrence, q_span, q_realigned = _align_source(
            prompt, qualifier.get("source_text"), qualifier.get("occurrence", 0), path=qualifier_path
        )
        if q_realigned:
            actions.append({"action": "source_span_realigned", "path": qualifier_path})
        qualifiers.append(
            {
                "kind": _enum(qualifier.get("kind"), QUALIFIER_KINDS, path=f"{qualifier_path}.kind"),
                "source_text": q_text,
                "occurrence": q_occurrence,
                "source_span": {"start": q_span.start, "end": q_span.end},
                "name": str(qualifier.get("name") or q_text).strip() or q_text,
            }
        )
    return {
        "_old_key": str(raw.get("key") or f"requirement_{index + 1}").strip() or f"requirement_{index + 1}",
        "source_text": source,
        "occurrence": occurrence,
        "source_span": {"start": span.start, "end": span.end},
        "name": str(raw.get("name") or source).strip() or source,
        "requirement_type": requirement_type,
        "polarity": polarity,
        "arguments": arguments,
        "relation": relation,
        "quantity": quantity,
        "scope": scope,
        "logic": logic,
        "qualifiers": qualifiers,
    }


def _set_source_start(prompt: str, value: JSON, start: int) -> None:
    """Move an already-aligned source value to another exact occurrence."""

    source_text = str(value["source_text"])
    value["occurrence"] = _occurrence(prompt, source_text, start)
    value["source_span"] = {
        "start": start,
        "end": start + len(source_text),
    }


def _contained_source_starts(
    prompt: str,
    source_text: str,
    spans: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    end_offset = len(source_text)
    return tuple(
        start
        for start in _all_starts(prompt, source_text)
        if any(
            int(span["start"]) <= start
            and start + end_offset <= int(span["end"])
            for span in spans
        )
    )


def _realign_requirement_local_sources(
    prompt: str,
    requirements: Sequence[JSON],
    entities: Sequence[JSON],
    actions: list[JSON],
) -> None:
    """Disambiguate repeated source text using semantic ownership context.

    Tool models commonly emit ``occurrence=0`` for every short qualifier. A
    globally valid span is not sufficient when words such as ``red``, ``blue``
    or ``paved`` occur more than once: the selected occurrence must belong to
    the requirement's argument entity or local clause. This correction is
    Candidate-independent and uses only exact prompt spans already present in
    the Stage 0 draft.
    """

    entities_by_key = {value["_old_key"]: value for value in entities}
    for requirement in requirements:
        argument_spans = [
            entities_by_key[argument["entity_key"]]["source_span"]
            for argument in requirement["arguments"]
            if argument["entity_key"] in entities_by_key
        ]
        current_start = int(requirement["source_span"]["start"])
        contextual = _contained_source_starts(
            prompt,
            requirement["source_text"],
            argument_spans,
        )
        if len(contextual) == 1 and contextual[0] != current_start:
            _set_source_start(prompt, requirement, contextual[0])
            actions.append(
                {
                    "action": "requirement_source_context_realigned",
                    "requirement_key": requirement["_old_key"],
                    "from_start": current_start,
                    "to_start": contextual[0],
                }
            )

        requirement_span = requirement["source_span"]
        for qualifier_index, qualifier in enumerate(requirement["qualifiers"]):
            current_start = int(qualifier["source_span"]["start"])
            local = _contained_source_starts(
                prompt,
                qualifier["source_text"],
                (requirement_span,),
            )
            if len(local) != 1:
                local = _contained_source_starts(
                    prompt,
                    qualifier["source_text"],
                    argument_spans,
                )
            if len(local) == 1 and local[0] != current_start:
                _set_source_start(prompt, qualifier, local[0])
                actions.append(
                    {
                        "action": "qualifier_source_context_realigned",
                        "requirement_key": requirement["_old_key"],
                        "qualifier_index": qualifier_index,
                        "from_start": current_start,
                        "to_start": local[0],
                    }
                )


def _inherit_local_observable_qualifiers(
    requirements: Sequence[JSON],
    actions: list[JSON],
) -> None:
    """Reuse exact local qualifiers that a sibling claim already extracted."""

    qualifier_required_types = {
        "distribution",
        "composition",
        "style_bundle",
        "environment",
        "boundary",
    }
    for requirement in requirements:
        if (
            requirement["requirement_type"] not in qualifier_required_types
            or requirement["qualifiers"]
        ):
            continue
        start = int(requirement["source_span"]["start"])
        end = int(requirement["source_span"]["end"])
        inherited: list[JSON] = []
        seen: set[tuple[str, int, int]] = set()
        for sibling in requirements:
            if (
                sibling is requirement
                or sibling["polarity"] != requirement["polarity"]
            ):
                continue
            for qualifier in sibling["qualifiers"]:
                qualifier_start = int(qualifier["source_span"]["start"])
                qualifier_end = int(qualifier["source_span"]["end"])
                signature = (
                    str(qualifier["kind"]),
                    qualifier_start,
                    qualifier_end,
                )
                if (
                    start <= qualifier_start
                    and qualifier_end <= end
                    and signature not in seen
                ):
                    seen.add(signature)
                    inherited.append(copy.deepcopy(qualifier))
        if inherited:
            inherited.sort(
                key=lambda value: (
                    int(value["source_span"]["start"]),
                    int(value["source_span"]["end"]),
                    str(value["kind"]),
                )
            )
            requirement["qualifiers"] = inherited
            actions.append(
                {
                    "action": "local_observable_qualifiers_inherited",
                    "requirement_key": requirement["_old_key"],
                    "qualifier_count": len(inherited),
                }
            )


def _unique_old_keys(values: Sequence[JSON], *, path: str) -> None:
    seen: set[str] = set()
    for index, value in enumerate(values):
        key = value["_old_key"]
        if key in seen:
            raise Stage0V2Error("duplicate_key", f"{path}[{index}].key", f"ambiguous duplicate {key!r}")
        seen.add(key)


def _qualifier_semantic_signature(qualifier: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(qualifier["kind"]),
        int(qualifier["source_span"]["start"]),
        int(qualifier["source_span"]["end"]),
        " ".join(
            unicodedata.normalize("NFKC", str(qualifier["name"]))
            .casefold()
            .split()
        ),
    )


def _normalize_declared_logic_requirements(
    requirements: Sequence[JSON],
    actions: list[JSON],
) -> set[str]:
    """Normalize model-declared logic and preserve shared-span operands."""

    by_key = {value["_old_key"]: value for value in requirements}
    shared_span_operands: set[str] = set()
    for requirement in requirements:
        logic = requirement["logic"]
        if logic is None:
            continue
        operand_keys = list(logic["requirement_keys"])
        operands = [by_key[key] for key in operand_keys if key in by_key]
        operator = logic["operator"]

        if requirement["requirement_type"] != "logic":
            if operator in {"any_of", "one_of"}:
                common_arguments = copy.deepcopy(requirement["arguments"])
                common_qualifiers = copy.deepcopy(requirement["qualifiers"])
                for operand in operands:
                    argument_signatures = {
                        (value["entity_key"], value["role"])
                        for value in operand["arguments"]
                    }
                    operand["arguments"].extend(
                        copy.deepcopy(value)
                        for value in common_arguments
                        if (value["entity_key"], value["role"])
                        not in argument_signatures
                    )
                    qualifier_signatures = {
                        _qualifier_semantic_signature(value)
                        for value in operand["qualifiers"]
                    }
                    operand["qualifiers"].extend(
                        copy.deepcopy(value)
                        for value in common_qualifiers
                        if _qualifier_semantic_signature(value)
                        not in qualifier_signatures
                    )
                    if operand["relation"] is None:
                        operand["relation"] = requirement["relation"]
                    if operand["quantity"] is None:
                        operand["quantity"] = copy.deepcopy(requirement["quantity"])
                    if operand["scope"] is None:
                        operand["scope"] = copy.deepcopy(requirement["scope"])

            previous_type = requirement["requirement_type"]
            requirement["requirement_type"] = "logic"
            requirement["arguments"] = []
            requirement["relation"] = None
            requirement["quantity"] = None
            requirement["scope"] = None
            requirement["qualifiers"] = []
            actions.append(
                {
                    "action": "declared_logic_type_normalized",
                    "requirement_key": requirement["_old_key"],
                    "from_type": previous_type,
                    "operator": operator,
                    "operand_count": len(operand_keys),
                }
            )

        if (
            operator not in {"any_of", "one_of"}
            or len(operands) != len(operand_keys)
            or len(operands) < 2
        ):
            continue
        spans = {
            (
                int(operand["source_span"]["start"]),
                int(operand["source_span"]["end"]),
            )
            for operand in operands
        }
        if len(spans) != 1 or "or" not in set(_words(operands[0]["source_text"])):
            continue
        signatures = {_semantic_operand_signature(value) for value in operands}
        if len(signatures) != len(operands):
            continue
        shared_span_operands.update(operand_keys)
        actions.append(
            {
                "action": "shared_disjunction_span_operands_preserved",
                "requirement_key": requirement["_old_key"],
                "operand_count": len(operand_keys),
            }
        )
    return shared_span_operands


def _expand_disjunctions(
    prompt: str,
    requirements: list[JSON],
    entities: Sequence[JSON],
    actions: list[JSON],
) -> None:
    """Turn equivalent ``A or B`` forms into explicit any-of requirement IR."""

    by_key = {value["_old_key"]: value for value in entities}
    shared_span_operands = _normalize_declared_logic_requirements(
        requirements,
        actions,
    )
    generated: list[JSON] = []
    for requirement in requirements:
        if (
            requirement["requirement_type"] == "logic"
            or requirement["_old_key"] in shared_span_operands
        ):
            continue
        words = set(_words(requirement["source_text"]))
        if "or" not in words:
            continue
        quantity = requirement.get("quantity")
        negative_union = requirement["polarity"] == Polarity.NEGATED.value or (
            isinstance(quantity, Mapping)
            and quantity.get("qualitative") in {"none", "almost_none"}
        )
        if negative_union:
            continue
        tagged_collections = [
            (index, argument, by_key.get(argument["entity_key"]))
            for index, argument in enumerate(requirement["arguments"])
            if by_key.get(argument["entity_key"], {}).get("list_mode") == "or"
            and by_key.get(argument["entity_key"], {}).get("member_keys")
        ]
        option_arguments: list[list[JSON]] = []
        option_qualifiers: list[list[JSON]] = []
        if tagged_collections:
            argument_index, original_argument, collection = tagged_collections[0]
            for member_key in collection["member_keys"]:
                arguments = copy.deepcopy(requirement["arguments"])
                arguments[argument_index] = {
                    "entity_key": member_key,
                    "role": original_argument["role"],
                }
                option_arguments.append(arguments)
                option_qualifiers.append(copy.deepcopy(requirement["qualifiers"]))
        elif len(requirement["arguments"]) >= 2:
            arguments = requirement["arguments"]
            if requirement["requirement_type"] == "spatial_relation":
                reference_roles = {"reference", "path", "region"}
                references = [
                    argument
                    for argument in arguments
                    if argument["role"] in reference_roles
                ]
                subjects = [
                    argument
                    for argument in arguments
                    if argument["role"] == "subject"
                ]
                if len(references) >= 2:
                    common = [
                        copy.deepcopy(argument)
                        for argument in arguments
                        if argument["role"] not in reference_roles
                    ]
                    for reference in references:
                        option_arguments.append(
                            [*copy.deepcopy(common), copy.deepcopy(reference)]
                        )
                        option_qualifiers.append(
                            copy.deepcopy(requirement["qualifiers"])
                        )
                elif len(subjects) >= 2:
                    common = [
                        copy.deepcopy(argument)
                        for argument in arguments
                        if argument["role"] != "subject"
                    ]
                    for subject in subjects:
                        option_arguments.append(
                            [copy.deepcopy(subject), *copy.deepcopy(common)]
                        )
                        option_qualifiers.append(
                            copy.deepcopy(requirement["qualifiers"])
                        )
            else:
                for argument in arguments:
                    option_arguments.append([copy.deepcopy(argument)])
                    option_qualifiers.append(copy.deepcopy(requirement["qualifiers"]))
        if not option_arguments:
            match = re.search(
                r"\b([^\W_]+(?:-[^\W_]+)*)\s+or\s+([^\W_]+(?:-[^\W_]+)*)\b",
                requirement["source_text"],
                flags=re.IGNORECASE | re.UNICODE,
            )
            if match is not None and requirement["arguments"]:
                alternatives = (match.group(1), match.group(2))
                alternative_words = {
                    unicodedata.normalize("NFKC", value).casefold()
                    for value in alternatives
                }
                common = [
                    copy.deepcopy(value)
                    for value in requirement["qualifiers"]
                    if not (set(_words(value["source_text"])) & alternative_words)
                ]
                requirement_start = int(requirement["source_span"]["start"])
                for group_index, alternative in enumerate(alternatives, start=1):
                    alternative_key = unicodedata.normalize(
                        "NFKC", alternative
                    ).casefold()
                    matching = [
                        copy.deepcopy(value)
                        for value in requirement["qualifiers"]
                        if alternative_key in set(_words(value["source_text"]))
                        and not (
                            (set(_words(value["source_text"])) & alternative_words)
                            - {alternative_key}
                        )
                    ]
                    if not matching:
                        start = requirement_start + match.start(group_index)
                        end = requirement_start + match.end(group_index)
                        source_text = prompt[start:end]
                        matching = [
                            {
                                "kind": "attribute",
                                "source_text": source_text,
                                "occurrence": _occurrence(prompt, source_text, start),
                                "source_span": {"start": start, "end": end},
                                "name": source_text,
                            }
                        ]
                    option_arguments.append(copy.deepcopy(requirement["arguments"]))
                    option_qualifiers.append([*common, *matching])
        if len(option_arguments) < 2:
            raise Stage0V2Error(
                "unmodeled_disjunction",
                f"requirements[{requirement['_old_key']}]",
                "an explicit 'or' could not be normalized into at least two operands",
            )
        operand_keys: list[str] = []
        for option_index, (arguments, qualifiers) in enumerate(
            zip(option_arguments, option_qualifiers, strict=True), start=1
        ):
            operand = copy.deepcopy(requirement)
            operand_key = f"{requirement['_old_key']}__option_{option_index}"
            operand["_old_key"] = operand_key
            operand["arguments"] = arguments
            operand["qualifiers"] = qualifiers
            operand["logic"] = None
            operand_keys.append(operand_key)
            generated.append(operand)
        requirement["requirement_type"] = "logic"
        requirement["arguments"] = []
        requirement["relation"] = None
        requirement["quantity"] = None
        requirement["scope"] = None
        requirement["qualifiers"] = []
        requirement["logic"] = {
            "operator": "any_of",
            "requirement_keys": operand_keys,
        }
        actions.append(
            {
                "action": "disjunction_expanded",
                "requirement_key": requirement["_old_key"],
                "operand_count": len(operand_keys),
            }
        )
    requirements.extend(generated)


def _normalize_explicit_singleton_one_of(
    requirements: Sequence[JSON],
    actions: list[JSON],
) -> None:
    """Preserve an explicit only as cardinality beside its sole operand."""

    for requirement in requirements:
        logic = requirement["logic"]
        if (
            requirement["requirement_type"] != "logic"
            or logic is None
            or logic["operator"] != "one_of"
            or len(logic["requirement_keys"]) != 1
            or "only" not in set(_words(requirement["source_text"]))
            or not requirement["arguments"]
        ):
            continue
        requirement["requirement_type"] = "quantity"
        requirement["relation"] = None
        requirement["quantity"] = {
            "mode": "exact",
            "value": 1,
            "lower": None,
            "upper": None,
            "qualitative": None,
            "unit": None,
        }
        requirement["scope"] = None
        requirement["logic"] = None
        actions.append(
            {
                "action": "explicit_singleton_one_of_normalized_to_quantity",
                "requirement_key": requirement["_old_key"],
                "value": 1,
            }
        )


def _is_affirmative_requirement(requirement: Mapping[str, Any]) -> bool:
    if requirement["polarity"] != Polarity.AFFIRMATIVE.value:
        return False
    quantity = requirement.get("quantity")
    return not (
        isinstance(quantity, Mapping)
        and quantity.get("qualitative") in {"none", "almost_none"}
    )


def _has_atomic_identity_path(
    entity_key: str,
    requirements: Sequence[JSON],
    entities_by_key: Mapping[str, JSON],
) -> bool:
    for requirement in requirements:
        if requirement["requirement_type"] not in {
            "presence",
            "attribute",
            "material",
        }:
            continue
        argument_keys = [
            argument["entity_key"] for argument in requirement["arguments"]
        ]
        if entity_key not in argument_keys:
            continue
        dependencies = [
            entities_by_key.get(argument_key) for argument_key in argument_keys
        ]
        if all(
            dependency is not None
            and dependency["grounding_mode"]
            in {
                EntityGroundingMode.ACTOR.value,
                EntityGroundingMode.ACTOR_COLLECTION.value,
            }
            for dependency in dependencies
        ):
            return True
    return False


def _add_coordinated_atomic_requirements(
    requirements: list[JSON],
    entities: Sequence[JSON],
    actions: list[JSON],
) -> None:
    """Expose concrete coordinated members as independent atomic claims.

    Presence is already implied by an affirmative coordinated list. The
    compiler makes that implication explicit so identity retrieval and
    targeted evidence do not depend on a global group-level VLM judgment.
    Alternatives in an ``or`` collection are deliberately excluded: their
    atomic predicates remain operands of the logic claim instead of becoming
    independent affirmative obligations.
    """

    entities_by_key = {value["_old_key"]: value for value in entities}
    actor_modes = {
        EntityGroundingMode.ACTOR.value,
        EntityGroundingMode.ACTOR_COLLECTION.value,
    }
    coordinated: set[str] = set()
    referenced: set[str] = set()
    affirmative_requirements = [
        value for value in requirements if _is_affirmative_requirement(value)
    ]
    for requirement in requirements:
        argument_keys = [
            argument["entity_key"] for argument in requirement["arguments"]
        ]
        referenced.update(argument_keys)
        if not _is_affirmative_requirement(requirement):
            continue
        coordinated.update(
            argument["entity_key"]
            for argument in requirement["arguments"]
            if argument["role"] == "member"
        )
        if requirement["requirement_type"] == "set":
            coordinated.update(argument_keys[1:])
        for argument_key in argument_keys:
            parent = entities_by_key.get(argument_key)
            if parent is not None and parent["list_mode"] == "and":
                coordinated.update(parent["member_keys"])

    # An explicit actor-groundable entity that the model mentioned only in a
    # qualifier would otherwise be frozen as unreachable support metadata.
    # Promote it only when its exact source span belongs to an affirmative
    # requirement; ambiguous/negative cases remain fail-closed for LLM repair.
    for entity in entities:
        if (
            entity["_old_key"] in referenced
            or entity["list_mode"] == "or"
            or entity["grounding_mode"] not in actor_modes
        ):
            continue
        entity_start = int(entity["source_span"]["start"])
        entity_end = int(entity["source_span"]["end"])
        if any(
            int(requirement["source_span"]["start"]) <= entity_start
            and entity_end <= int(requirement["source_span"]["end"])
            for requirement in affirmative_requirements
        ):
            coordinated.add(entity["_old_key"])

    used_keys = {value["_old_key"] for value in requirements}
    generated: list[JSON] = []
    for entity_key in sorted(
        (key for key in coordinated if key in entities_by_key),
        key=lambda key: (
            int(entities_by_key[key]["source_span"]["start"]),
            int(entities_by_key[key]["source_span"]["end"]),
            key,
        ),
    ):
        entity = entities_by_key.get(entity_key)
        if (
            entity is None
            or entity["grounding_mode"] not in actor_modes
            or _has_atomic_identity_path(entity_key, requirements, entities_by_key)
        ):
            continue
        base_key = f"{entity_key}__atomic_presence"
        generated_key = base_key
        suffix = 2
        while generated_key in used_keys:
            generated_key = f"{base_key}_{suffix}"
            suffix += 1
        used_keys.add(generated_key)
        generated.append(
            {
                "_old_key": generated_key,
                "source_text": entity["source_text"],
                "occurrence": entity["occurrence"],
                "source_span": copy.deepcopy(entity["source_span"]),
                "name": f"presence of {entity['name']}",
                "requirement_type": "presence",
                "polarity": Polarity.AFFIRMATIVE.value,
                "arguments": [{"entity_key": entity_key, "role": "subject"}],
                "relation": None,
                "quantity": None,
                "scope": None,
                "logic": None,
                "qualifiers": [],
            }
        )
        actions.append(
            {
                "action": "coordinated_atomic_presence_added",
                "entity_key": entity_key,
                "requirement_key": generated_key,
            }
        )
    requirements.extend(generated)


def _visible_qualifier_parts(
    prompt: str,
    qualifier: Mapping[str, Any],
) -> tuple[JSON, ...]:
    """Split coordinated colour/material text into exact prompt spans."""

    source = str(qualifier["source_text"])
    source_start = int(qualifier["source_span"]["start"])
    boundaries = tuple(
        re.finditer(
            r"\s*(?:,|;)\s*|\s+(?:and)\s+",
            source,
            flags=re.IGNORECASE,
        )
    )
    if not boundaries:
        return (copy.deepcopy(dict(qualifier)),)
    parts: list[JSON] = []
    cursor = 0
    for boundary in (*boundaries, None):
        end = boundary.start() if boundary is not None else len(source)
        raw = source[cursor:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            start = source_start + cursor + left
            stop = source_start + cursor + right
            text = prompt[start:stop]
            parts.append(
                {
                    "kind": qualifier["kind"],
                    "source_text": text,
                    "occurrence": _occurrence(prompt, text, start),
                    "source_span": {"start": start, "end": stop},
                    "name": text,
                }
            )
        if boundary is not None:
            cursor = boundary.end()
    return tuple(parts) if len(parts) >= 2 else (copy.deepcopy(dict(qualifier)),)


def _span_gap(first_start: int, first_end: int, second_start: int, second_end: int) -> int:
    if first_end < second_start:
        return second_start - first_end
    if second_end < first_start:
        return first_start - second_end
    return 0


def _local_entity_mention_gap(
    prompt: str,
    requirement: Mapping[str, Any],
    qualifier: Mapping[str, Any],
    entity: Mapping[str, Any],
) -> int | None:
    requirement_start = int(requirement["source_span"]["start"])
    requirement_end = int(requirement["source_span"]["end"])
    qualifier_start = int(qualifier["source_span"]["start"])
    qualifier_end = int(qualifier["source_span"]["end"])
    local_text = prompt[requirement_start:requirement_end]
    terms = {
        str(value).strip()
        for value in (
            entity.get("source_text"),
            entity.get("name"),
            *(entity.get("aliases") or ()),
        )
        if str(value or "").strip()
    }
    gaps: list[int] = []
    for term in terms:
        for match in re.finditer(re.escape(term), local_text, flags=re.IGNORECASE):
            mention_start = requirement_start + match.start()
            mention_end = requirement_start + match.end()
            gaps.append(
                _span_gap(
                    qualifier_start,
                    qualifier_end,
                    mention_start,
                    mention_end,
                )
            )
    return min(gaps) if gaps else None


def _visible_qualifier_subject_argument(
    prompt: str,
    requirement: Mapping[str, Any],
    qualifier: Mapping[str, Any],
    entities_by_key: Mapping[str, JSON],
) -> JSON:
    """Choose the argument whose local mention the qualifier modifies."""

    ranked: list[tuple[int, int, int, Mapping[str, Any]]] = []
    for index, argument in enumerate(requirement["arguments"]):
        entity = entities_by_key.get(argument["entity_key"])
        if entity is None:
            continue
        gap = _local_entity_mention_gap(prompt, requirement, qualifier, entity)
        if gap is None:
            continue
        ranked.append(
            (
                gap,
                0 if argument["role"] == "subject" else 1,
                index,
                argument,
            )
        )
    if ranked:
        selected = min(ranked)[3]
    else:
        selected = next(
            (
                argument
                for argument in requirement["arguments"]
                if argument["role"] == "subject"
            ),
            requirement["arguments"][0],
        )
    result = copy.deepcopy(dict(selected))
    result["role"] = "subject"
    return result


def _add_atomic_visible_qualifier_requirements(
    prompt: str,
    requirements: list[JSON],
    entities: Sequence[JSON],
    actions: list[JSON],
) -> None:
    """Expose every explicit colour/material modifier as a scoreable leaf."""

    used_keys = {value["_old_key"] for value in requirements}
    entities_by_key = {value["_old_key"]: value for value in entities}
    generated: list[JSON] = []
    for requirement in tuple(requirements):
        visible = tuple(
            part
            for qualifier in requirement["qualifiers"]
            if qualifier["kind"] in {"color", "material"}
            for part in _visible_qualifier_parts(prompt, qualifier)
        )
        if not visible:
            continue
        if not requirement["arguments"]:
            raise Stage0V2Error(
                "visible_qualifier_missing_subject",
                f"requirements[{requirement['_old_key']}].qualifiers",
                "colour/material qualifiers require a concrete or scene-global subject",
            )
        for qualifier in visible:
            subject_argument = _visible_qualifier_subject_argument(
                prompt,
                requirement,
                qualifier,
                entities_by_key,
            )
            subject_key = subject_argument["entity_key"]
            requirement_type = (
                "material" if qualifier["kind"] == "material" else "attribute"
            )
            already_atomic = False
            for candidate in (*requirements, *generated):
                candidate_visible = [
                    item
                    for item in candidate["qualifiers"]
                    if item["kind"] in {"color", "material"}
                ]
                if (
                    candidate["requirement_type"] == requirement_type
                    and len(candidate_visible) == 1
                    and candidate_visible[0]["kind"] == qualifier["kind"]
                    and candidate_visible[0]["source_span"] == qualifier["source_span"]
                    and any(
                        argument["entity_key"] == subject_key
                        for argument in candidate["arguments"]
                    )
                ):
                    already_atomic = True
                    break
            if already_atomic:
                continue
            base_key = f"{requirement['_old_key']}__atomic_{qualifier['kind']}"
            generated_key = base_key
            suffix = 2
            while generated_key in used_keys:
                generated_key = f"{base_key}_{suffix}"
                suffix += 1
            used_keys.add(generated_key)
            generated.append(
                {
                    "_old_key": generated_key,
                    "source_text": qualifier["source_text"],
                    "occurrence": qualifier["occurrence"],
                    "source_span": copy.deepcopy(qualifier["source_span"]),
                    "name": f"{qualifier['name']} of {subject_key}",
                    "requirement_type": requirement_type,
                    "polarity": requirement["polarity"],
                    "arguments": [copy.deepcopy(subject_argument)],
                    "relation": None,
                    "quantity": None,
                    "scope": copy.deepcopy(requirement["scope"]),
                    "logic": None,
                    "qualifiers": [copy.deepcopy(qualifier)],
                }
            )
            actions.append(
                {
                    "action": "atomic_visible_qualifier_added",
                    "source_requirement_key": requirement["_old_key"],
                    "requirement_key": generated_key,
                    "qualifier_kind": qualifier["kind"],
                }
            )
    if len(requirements) + len(generated) > _MAX_REQUIREMENTS:
        raise Stage0V2Error(
            "too_many_atomic_requirements",
            "requirements",
            f"normalization exceeds the {_MAX_REQUIREMENTS} requirement limit",
        )
    requirements.extend(generated)


def _resolve_reference(reference: str, values: Sequence[JSON], mapping: Mapping[str, str], *, path: str) -> str:
    if reference in mapping:
        return mapping[reference]
    normalized = unicodedata.normalize("NFKC", reference).casefold()
    matches = [
        mapping[value["_old_key"]]
        for value in values
        if unicodedata.normalize("NFKC", value["name"]).casefold() == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    raise Stage0V2Error("dangling_reference", path, f"cannot resolve {reference!r}")


def _normalize_roles(requirement: JSON, actions: list[JSON]) -> None:
    arguments = requirement["arguments"]
    requirement_type = requirement["requirement_type"]
    if requirement_type == "spatial_relation" and len(arguments) >= 2:
        roles = [value["role"] for value in arguments]
        if "subject" not in roles or "reference" not in roles:
            arguments[0]["role"] = "subject"
            arguments[1]["role"] = "reference"
            actions.append({"action": "spatial_roles_inferred", "requirement_key": requirement["key"]})
    elif requirement_type == "quantity" and arguments:
        arguments[0]["role"] = "collection"
    elif requirement_type == "set" and arguments:
        arguments[0]["role"] = "collection"
        for value in arguments[1:]:
            value["role"] = "member"
    elif requirement_type in {"presence", "attribute", "material", "distribution", "style_bundle"} and arguments:
        if not any(value["role"] == "subject" for value in arguments):
            arguments[0]["role"] = "subject"
    unique: list[JSON] = []
    seen: set[tuple[str, str]] = set()
    for value in arguments:
        signature = (value["entity_key"], value["role"])
        if signature in seen:
            actions.append({"action": "duplicate_argument_removed", "requirement_key": requirement["key"]})
            continue
        seen.add(signature)
        unique.append(value)
    for ordinal, value in enumerate(unique):
        value["ordinal"] = ordinal
    requirement["arguments"] = unique


def _validate_requirement_shape(requirement: Mapping[str, Any]) -> None:
    """Validate semantic structure, not current verifier executability."""

    key = requirement["key"]
    path = f"requirements[{key}]"
    requirement_type = requirement["requirement_type"]
    arguments = requirement["arguments"]
    source_text = requirement["source_text"]
    if len(source_text) > 800 or len(re.findall(r"[.!?](?:\s|$)", source_text)) > 1:
        raise Stage0V2Error(
            "non_atomic_requirement",
            f"{path}.source_text",
            "must describe one local clause rather than collapsing multiple sentences",
        )
    if requirement_type in {
        "presence",
        "quantity",
        "set",
        "spatial_relation",
        "distribution",
        "style_bundle",
        "attribute",
        "material",
    } and not arguments:
        raise Stage0V2Error(
            "missing_argument", f"{path}.arguments", f"{requirement_type} needs an entity argument"
        )
    if requirement_type == "spatial_relation":
        if len(arguments) < 2:
            raise Stage0V2Error(
                "missing_spatial_argument",
                f"{path}.arguments",
                "spatial_relation needs at least subject and reference/path/region",
            )
        if requirement["relation"] is None:
            raise Stage0V2Error(
                "missing_relation", f"{path}.relation", "spatial_relation needs relation"
            )
    if requirement_type in {
        "distribution",
        "composition",
        "style_bundle",
        "environment",
        "boundary",
    } and not requirement["qualifiers"]:
        raise Stage0V2Error(
            "missing_qualifier",
            f"{path}.qualifiers",
            f"{requirement_type} must enumerate its observable semantic cues",
        )
    logic = requirement["logic"]
    if requirement_type != "logic":
        if logic is not None:
            raise Stage0V2Error(
                "unexpected_logic", f"{path}.logic", "only logic requirements own operands"
            )
        return
    operator = logic["operator"]
    count = len(logic["requirement_keys"])
    expected = 1 if operator == "not" else 2
    if count < expected:
        raise Stage0V2Error(
            "missing_logic_operand",
            f"{path}.logic.requirement_keys",
            f"{operator} requires at least {expected} operand(s)",
        )


def _semantic_operand_signature(requirement: Mapping[str, Any]) -> tuple[Any, ...]:
    def normalized(value: Any) -> str:
        return " ".join(
            unicodedata.normalize("NFKC", str(value)).casefold().split()
        )

    return (
        requirement["requirement_type"],
        requirement["polarity"],
        tuple(
            sorted(
                (argument["entity_key"], argument["role"])
                for argument in requirement["arguments"]
            )
        ),
        normalized(requirement.get("relation") or ""),
        json.dumps(requirement.get("quantity"), sort_keys=True),
        json.dumps(requirement.get("scope"), sort_keys=True),
        tuple(
            sorted(
                (
                    qualifier["kind"],
                    normalized(qualifier["name"]),
                )
                for qualifier in requirement["qualifiers"]
            )
        ),
    )


def _validate_logic_operand_distinctness(requirements: Sequence[JSON]) -> None:
    by_key = {value["key"]: value for value in requirements}
    for requirement in requirements:
        logic = requirement["logic"]
        if logic is None or logic["operator"] not in {"any_of", "one_of"}:
            continue
        signatures: dict[tuple[Any, ...], str] = {}
        for operand_key in logic["requirement_keys"]:
            operand = by_key[operand_key]
            signature = _semantic_operand_signature(operand)
            previous = signatures.get(signature)
            if previous is not None:
                raise Stage0V2Error(
                    "duplicate_logic_operand_semantics",
                    f"requirements[{requirement['key']}].logic.requirement_keys",
                    (
                        f"{logic['operator']} operands {previous!r} and "
                        f"{operand_key!r} normalize to the same semantic claim"
                    ),
                )
            signatures[signature] = operand_key


def _validate_scope_coverage(requirements: Sequence[JSON]) -> None:
    """Require every scope cue to have one local explicit semantic owner."""

    by_key = {value["key"]: value for value in requirements}
    for requirement in requirements:
        logic = requirement["logic"]
        scope_words = set(_words(requirement["source_text"])) & {
            "each",
            "every",
            "per",
        }
        if not scope_words or requirement["scope"] is not None:
            continue
        if logic is not None:
            operands = [by_key[key] for key in logic["requirement_keys"]]
            if operands and all(
                operand["scope"] is not None for operand in operands
            ):
                continue
        start = int(requirement["source_span"]["start"])
        end = int(requirement["source_span"]["end"])
        argument_keys = {
            argument["entity_key"] for argument in requirement["arguments"]
        }
        if any(
            sibling is not requirement
            and sibling["scope"] is not None
            and sibling["scope"]["entity_key"] in argument_keys
            and start <= int(sibling["source_span"]["start"])
            and int(sibling["source_span"]["end"]) <= end
            and bool(scope_words & set(_words(sibling["source_text"])))
            for sibling in requirements
        ):
            continue
        raise Stage0V2Error(
            "missing_scope",
            f"requirements[{requirement['key']}].scope",
            (
                f"explicit scope cues {sorted(scope_words)!r} require a scope "
                "object, scoped logic operands, or one contained scoped claim"
            ),
        )


def _is_scene_form_instruction(prompt: str, match: re.Match[str]) -> bool:
    """Return whether ``form`` is a sentence-leading scene authoring verb."""

    word = unicodedata.normalize("NFKC", match.group(0)).casefold()
    if word != "form":
        return False
    prefix = prompt[: match.start()].rstrip()
    if prefix and prefix[-1] not in ".!?":
        return False
    return (
        re.match(
            r"\s+(?:(?:the|a|an|this)\s+)?scene\b",
            prompt[match.end() :],
            flags=re.IGNORECASE | re.UNICODE,
        )
        is not None
    )


def _validate_coverage(
    prompt: str,
    requirements: Sequence[JSON],
    entities: Sequence[JSON],
) -> JSON:
    requirement_spans = [
        SourceSpan(int(value["source_span"]["start"]), int(value["source_span"]["end"]))
        for value in requirements
    ]
    referenced_entity_keys = {
        argument["entity_key"]
        for requirement in requirements
        for argument in requirement["arguments"]
    }
    referenced_entities = [
        value for value in entities if value["key"] in referenced_entity_keys
    ]
    entity_spans = [
        SourceSpan(int(value["source_span"]["start"]), int(value["source_span"]["end"]))
        for value in referenced_entities
    ]
    alias_spans = [
        SourceSpan(match.start(), match.end())
        for value in referenced_entities
        for alias in value["aliases"]
        if alias
        for match in re.finditer(
            rf"(?<![^\W_]){re.escape(alias)}(?![^\W_])",
            prompt,
            flags=re.IGNORECASE | re.UNICODE,
        )
    ]
    spans = [*requirement_spans, *entity_spans, *alias_spans]
    covered_words = {
        unicodedata.normalize("NFKC", match.group(0)).casefold()
        for span in spans
        for match in re.finditer(
            r"[^\W_]+", prompt[span.start : span.end], flags=re.UNICODE
        )
    }
    uncovered: list[str] = []
    content_count = 0
    repeated_mention_count = 0
    for match in re.finditer(r"[^\W_]+", prompt, flags=re.UNICODE):
        word = unicodedata.normalize("NFKC", match.group(0)).casefold()
        if (
            word in _FUNCTION_WORDS
            or (len(word) == 1 and word.isascii())
            or _is_scene_form_instruction(prompt, match)
        ):
            continue
        content_count += 1
        if any(span.start <= match.start() and match.end() <= span.end for span in spans):
            continue
        if word in covered_words:
            repeated_mention_count += 1
            continue
        uncovered.append(match.group(0))
    if uncovered:
        raise Stage0V2Error(
            "incomplete_semantic_coverage",
            "prompt",
            f"uncovered content terms {uncovered[:24]!r}",
        )
    return {
        "content_term_count": content_count,
        "covered_content_term_count": content_count,
        "repeated_mention_count": repeated_mention_count,
        "coverage_ratio": 1.0,
    }


def normalize_semantic_ir(prompt: str, value: Mapping[str, Any]) -> JSON:
    """Normalize one v2 tool draft without consulting Candidate or capabilities."""

    raw = _object(value, path="$")
    if raw.get("draft_version") != DRAFT_VERSION:
        raise Stage0V2Error("invalid_version", "draft_version", f"must be {DRAFT_VERSION!r}")
    actions: list[JSON] = []
    entities = [
        _parse_entity(prompt, item, index, actions)
        for index, item in enumerate(_list(raw.get("entities"), path="entities", maximum=_MAX_ENTITIES))
    ]
    _expand_tagged_lists(prompt, entities, actions)
    requirements = [
        _parse_requirement(prompt, item, index, actions)
        for index, item in enumerate(_list(raw.get("requirements"), path="requirements", maximum=_MAX_REQUIREMENTS))
    ]
    if not requirements:
        raise Stage0V2Error("missing_requirement", "requirements", "must contain at least one semantic requirement")
    _realign_requirement_local_sources(prompt, requirements, entities, actions)
    _inherit_local_observable_qualifiers(requirements, actions)
    _expand_disjunctions(prompt, requirements, entities, actions)
    _normalize_explicit_singleton_one_of(requirements, actions)
    _unique_old_keys(entities, path="entities")
    _unique_old_keys(requirements, path="requirements")

    entities.sort(key=lambda value: (value["source_span"]["start"], value["source_span"]["end"], value["name"].casefold(), value["_old_key"]))
    requirements.sort(key=lambda value: (value["source_span"]["start"], value["source_span"]["end"], value["name"].casefold(), value["_old_key"]))
    entity_keys = {value["_old_key"]: f"e{index}" for index, value in enumerate(entities, start=1)}
    requirement_keys = {value["_old_key"]: f"r{index}" for index, value in enumerate(requirements, start=1)}

    for value in entities:
        value["key"] = entity_keys[value["_old_key"]]
        value["member_keys"] = [
            _resolve_reference(item, entities, entity_keys, path=f"entities[{value['key']}].member_keys")
            for item in value["member_keys"]
        ]
        value.pop("_old_key", None)
    for value in requirements:
        value["key"] = requirement_keys[value["_old_key"]]
        for argument in value["arguments"]:
            argument["entity_key"] = _resolve_reference(
                argument["entity_key"], entities, entity_keys, path=f"requirements[{value['key']}].arguments"
            )
        if value["scope"] is not None and value["scope"]["entity_key"] is not None:
            value["scope"]["entity_key"] = _resolve_reference(
                value["scope"]["entity_key"], entities, entity_keys, path=f"requirements[{value['key']}].scope"
            )
            if not any(argument["entity_key"] == value["scope"]["entity_key"] for argument in value["arguments"]):
                value["arguments"].append({"entity_key": value["scope"]["entity_key"], "role": "scope"})
        if value["logic"] is not None:
            value["logic"]["requirement_keys"] = [
                _resolve_reference(item, requirements, requirement_keys, path=f"requirements[{value['key']}].logic")
                for item in value["logic"]["requirement_keys"]
            ]
        value.pop("_old_key", None)
        _normalize_roles(value, actions)
        _validate_requirement_shape(value)

    _validate_scope_coverage(requirements)
    _validate_logic_operand_distinctness(requirements)
    coverage = _validate_coverage(prompt, requirements, entities)
    return {
        "draft_version": DRAFT_VERSION,
        "entities": entities,
        "requirements": requirements,
        "weight_policy": WEIGHT_POLICY,
        "coverage": coverage,
        "normalization_actions": actions,
    }


def _public_ids(values: Sequence[JSON], prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for index, value in enumerate(values, start=1):
        base = f"{prefix}_{_slug(value['name']) or f'node_{index:02d}'}"
        candidate = base[:64]
        suffix = 2
        while candidate in used:
            text = f"_{suffix}"
            candidate = base[: 64 - len(text)] + text
            suffix += 1
        used.add(candidate)
        result[value["key"]] = candidate
    return result


def lower_semantic_ir(prompt: str, canonical: Mapping[str, Any]) -> RequirementGraph:
    entities = list(canonical["entities"])
    requirements = list(canonical["requirements"])
    entity_ids = _public_ids(entities, "entity")
    predicate_ids = _public_ids(requirements, "predicate")
    entity_nodes = tuple(
        EntityNode(
            id=entity_ids[value["key"]],
            text=value["source_text"],
            name=value["name"],
            source_span=SourceSpan(value["source_span"]["start"], value["source_span"]["end"]),
            entity_type=value["entity_type"],
            referent_kind=value["referent_kind"],
            aliases=tuple(value["aliases"]),
            evaluation_route=EntityEvaluationRoute.STAGE2_VISUAL,
            grounding_mode=value["grounding_mode"],
            member_ids=tuple(entity_ids[item] for item in value["member_keys"]),
        )
        for value in entities
    )
    predicate_nodes = tuple(
        PredicateNode(
            id=predicate_ids[value["key"]],
            text=value["source_text"],
            name=value["name"],
            predicate_type=_TYPE_TO_PREDICATE[value["requirement_type"]],
            source_span=SourceSpan(value["source_span"]["start"], value["source_span"]["end"]),
            polarity=value["polarity"],
            semantic_parameters={
                "stage0_ir_version": DRAFT_VERSION,
                "ir_key": value["key"],
                "requirement_type": value["requirement_type"],
                "relation": value["relation"],
                "quantity": value["quantity"],
                "scope": None if value["scope"] is None else {
                    **value["scope"],
                    "entity_id": None if value["scope"]["entity_key"] is None else entity_ids[value["scope"]["entity_key"]],
                },
                "logic": None if value["logic"] is None else {
                    "operator": value["logic"]["operator"],
                    "requirement_ids": [predicate_ids[item] for item in value["logic"]["requirement_keys"]],
                },
                "qualifiers": value["qualifiers"],
            },
        )
        for value in requirements
    )
    argument_edges = tuple(
        ArgumentEdge(
            predicate_ids[value["key"]],
            entity_ids[argument["entity_key"]],
            argument["role"],
            argument["ordinal"],
        )
        for value in requirements
        for argument in value["arguments"]
    )
    scope_edges = tuple(
        ScopeEdge(predicate_ids[value["key"]], predicate_ids[target])
        for value in requirements
        if value["logic"] is not None
        for target in value["logic"]["requirement_keys"]
    )
    root = RequirementNode("requirement_scene", prompt, SourceSpan(0, len(prompt)))
    referenced = {
        target
        for value in requirements
        if value["logic"] is not None
        for target in value["logic"]["requirement_keys"]
    }
    top_level = [value for value in requirements if value["key"] not in referenced]
    if not top_level:
        raise Stage0V2Error("missing_top_level_requirement", "requirements", "logic references form no top-level claim")
    weight = 1.0 / len(top_level)
    if len(top_level) > MAX_SCORED_REQUIREMENTS:
        raise Stage0V2Error(
            "semantic_budget_exceeded",
            "requirements",
            (
                f"{len(top_level)} top-level claims exceed the "
                f"{MAX_SCORED_REQUIREMENTS}-claim scoring budget"
            ),
        )
    members = tuple(
        RequirementMemberEdge(
            root.id,
            predicate_ids[value["key"]],
            MemberRole.SCORED_FACET if value in top_level else MemberRole.SUPPORT_ONLY,
            weight if value in top_level else None,
        )
        for value in requirements
    ) + tuple(
        RequirementMemberEdge(root.id, node.id, MemberRole.SUPPORT_ONLY)
        for node in entity_nodes
    )
    graph = RequirementGraph(
        prompt=prompt,
        nodes=(*entity_nodes, *predicate_nodes, root),
        edges=(*argument_edges, *scope_edges, *members),
        roots=(RootRequirement(root.id, 1.0),),
        schema_version="2.0",
    )
    return RequirementGraph.from_dict(graph.to_dict())


def compile_semantic_ir(prompt: str, value: Mapping[str, Any]) -> tuple[RequirementGraph, JSON]:
    canonical = normalize_semantic_ir(prompt, value)
    return lower_semantic_ir(prompt, canonical), canonical


def _nullable(schema: JSON) -> JSON:
    return {"anyOf": [schema, {"type": "null"}]}


def tool_schema() -> JSON:
    string_enum = lambda values: {"type": "string", "enum": list(values)}
    source = {
        "source_text": {"type": "string", "minLength": 1},
        "occurrence": {"type": "integer", "minimum": 0, "maximum": 63},
    }
    entity = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "minLength": 1},
            **source,
            "name": {"type": "string", "minLength": 1},
            "aliases": {"type": "array", "maxItems": _MAX_ALIASES, "items": {"type": "string", "minLength": 1}},
            "entity_type": string_enum(item.value for item in EntityType),
            "referent_kind": string_enum(item.value for item in ReferentKind),
            "grounding_mode": string_enum(
                item.value
                for item in EntityGroundingMode
                if item is not EntityGroundingMode.AUTO
            ),
            "list_mode": string_enum(("none", "and", "or")),
            "member_keys": {"type": "array", "maxItems": 24, "items": {"type": "string", "minLength": 1}},
        },
        "required": ["key", "source_text", "occurrence", "name", "aliases", "entity_type", "referent_kind", "grounding_mode", "list_mode", "member_keys"],
        "additionalProperties": False,
    }
    argument = {
        "type": "object",
        "properties": {"entity_key": {"type": "string", "minLength": 1}, "role": {"type": "string", "minLength": 1}},
        "required": ["entity_key", "role"],
        "additionalProperties": False,
    }
    quantity = {
        "type": "object",
        "properties": {
            "mode": string_enum(QUANTITY_MODES),
            "value": _nullable({"type": "number"}),
            "lower": _nullable({"type": "number"}),
            "upper": _nullable({"type": "number"}),
            "qualitative": _nullable(string_enum(QUALITATIVE_QUANTITIES)),
            "unit": _nullable({"type": "string", "minLength": 1}),
        },
        "required": ["mode", "value", "lower", "upper", "qualitative", "unit"],
        "additionalProperties": False,
    }
    scope = {
        "type": "object",
        "properties": {"quantifier": string_enum(SCOPE_QUANTIFIERS), "entity_key": _nullable({"type": "string", "minLength": 1})},
        "required": ["quantifier", "entity_key"],
        "additionalProperties": False,
    }
    logic = {
        "type": "object",
        "properties": {"operator": string_enum(LOGIC_OPERATORS), "requirement_keys": {"type": "array", "maxItems": _MAX_REFERENCES, "items": {"type": "string", "minLength": 1}}},
        "required": ["operator", "requirement_keys"],
        "additionalProperties": False,
    }
    qualifier = {
        "type": "object",
        "properties": {**source, "kind": string_enum(QUALIFIER_KINDS), "name": {"type": "string", "minLength": 1}},
        "required": ["source_text", "occurrence", "kind", "name"],
        "additionalProperties": False,
    }
    requirement = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "minLength": 1},
            **source,
            "name": {"type": "string", "minLength": 1},
            "requirement_type": string_enum(REQUIREMENT_TYPES),
            "polarity": string_enum(item.value for item in Polarity),
            "arguments": {"type": "array", "maxItems": _MAX_ARGUMENTS, "items": argument},
            "relation": _nullable({"type": "string", "minLength": 1}),
            "quantity": _nullable(quantity),
            "scope": _nullable(scope),
            "logic": _nullable(logic),
            "qualifiers": {"type": "array", "maxItems": _MAX_QUALIFIERS, "items": qualifier},
        },
        "required": ["key", "source_text", "occurrence", "name", "requirement_type", "polarity", "arguments", "relation", "quantity", "scope", "logic", "qualifiers"],
        "additionalProperties": False,
    }
    return {
        "name": "return_requirement_graph_draft",
        "description": "Return complete prompt-grounded Stage0 v2 Semantic IR; capability is intentionally unplanned.",
        "parameters": {
            "type": "object",
            "properties": {
                "draft_version": {"type": "string", "enum": [DRAFT_VERSION]},
                "entities": {"type": "array", "maxItems": _MAX_ENTITIES, "items": entity},
                "requirements": {"type": "array", "maxItems": _MAX_REQUIREMENTS, "items": requirement},
            },
            "required": ["draft_version", "entities", "requirements"],
            "additionalProperties": False,
        },
    }


__all__ = [
    "DRAFT_VERSION",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_REVISION",
    "Stage0V2Error",
    "TOOL_SCHEMA_VERSION",
    "compile_semantic_ir",
    "lower_semantic_ir",
    "normalize_semantic_ir",
    "tool_schema",
]
