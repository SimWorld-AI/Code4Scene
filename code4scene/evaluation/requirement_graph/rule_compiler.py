"""Conservative deterministic rule compiler for RequirementGraph Stage 0.

The compiler extracts the narrow prompt subset that can be represented as
deterministic rules.  Clauses outside that subset are routed by ``authoring``
to the semantic LLM draft path.  This module is not a verifier and exposes no
standalone scoring or freeze workflow.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

COMPILER_NAME = "scenebench_requirement_rule_compiler"
COMPILER_VERSION = "1.0.0"

DEFAULT_TAXONOMY: Mapping[str, tuple[str, ...]] = {
    "table": ("table", "tables"),
    "chair": ("chair", "chairs"),
    "lamp": ("lamp", "lamps"),
    "toilet": ("toilet", "toilets"),
    "bookshelf": ("bookshelf", "bookshelves"),
    "sofa": ("sofa", "sofas", "couch", "couches"),
    "bed": ("bed", "beds"),
    "desk": ("desk", "desks"),
    "room": ("room", "rooms"),
    "hall": ("hall", "halls"),
    "building": ("building", "buildings"),
    "road": ("road", "roads"),
}

DEFAULT_ATTRIBUTES: Mapping[str, Mapping[str, Any]] = {
    "red": {"aliases": ("red",), "field": "colour",
            "value": "red", "preferred_evaluator": "vlm"},
    "blue": {"aliases": ("blue",), "field": "colour",
             "value": "blue", "preferred_evaluator": "vlm"},
    "green": {"aliases": ("green",), "field": "colour",
              "value": "green", "preferred_evaluator": "vlm"},
    "wooden": {"aliases": ("wooden", "wood"),
               "field": "material", "value": "wooden",
               "preferred_evaluator": "vlm"},
    "metal": {"aliases": ("metal", "metallic"),
              "field": "material", "value": "metal",
              "preferred_evaluator": "vlm"},
}

_RELATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("around", (r"\baround\b", r"\bsurround(?:ing|s)?\b")),
    ("on_top_of", (r"\bon top of\b",)),
    ("left_of", (r"\b(?:to the )?left of\b",)),
    ("right_of", (r"\b(?:to the )?right of\b",)),
    ("facing", (r"\bfacing\b", r"\bfaces\b")),
    ("inside", (r"\binside(?: of)?\b",)),
    ("above", (r"\babove\b",)),
    ("below", (r"\bbelow\b", r"\bunder\b")),
    ("near", (r"\bnear\b", r"\bnext to\b", r"\bbeside\b")),
    ("far", (r"\bfar from\b",)),
)

_ENGLISH_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "a": 1, "an": 1,
}
_CLAUSES = re.compile(r"[^.!?;\n]+", re.UNICODE)
_VAGUE = re.compile(r"\b(?:several|a few|few|many|some)\b")
_FORBIDDEN = re.compile(
    r"\b(?:without|do not (?:include|add|place)|must not (?:include|add|place)|exclude)\b"
    r"|\bno\s+(?!more\s+than\b)",
    re.IGNORECASE,
)


class RuleCompilationError(ValueError):
    """A deterministic rule draft violates the Stage 0 contract."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "item"


def segment_prompt(prompt: str) -> list[dict[str, Any]]:
    """Split requirements while retaining exact offsets into the prompt."""
    clauses = []
    for match in _CLAUSES.finditer(prompt):
        raw = match.group(0)
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start, end = match.start() + left, match.start() + right
        if end <= start:
            continue
        clauses.append({"id": f"clause_{len(clauses) + 1}",
                        "text": prompt[start:end], "source_span": [start, end],
                        "explicit_requirement": True, "status": "unresolved"})
    return clauses


def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)


def _mentions(text: str, taxonomy: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    candidates = []
    for category, aliases in taxonomy.items():
        for alias in dict.fromkeys((category, *aliases)):
            for match in _alias_pattern(str(alias)).finditer(text):
                candidates.append({"category": category, "alias": alias,
                                   "start": match.start(), "end": match.end(),
                                   "text": match.group(0)})
    candidates.sort(key=lambda item: (item["start"],
                                      -(item["end"] - item["start"])))
    selected = []
    for candidate in candidates:
        if any(candidate["start"] < item["end"]
               and candidate["end"] > item["start"] for item in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item["start"])


def _count(prefix: str) -> tuple[dict[str, int], str] | None:
    found: list[tuple[int, int]] = []
    words = "|".join(_ENGLISH_NUMBERS)
    for match in re.finditer(rf"\b(\d+|{words})\b", prefix, re.IGNORECASE):
        token = match.group(1).lower()
        found.append((match.end(), int(token) if token.isdigit()
                      else _ENGLISH_NUMBERS[token]))
    if not found:
        return None
    end, value = max(found, key=lambda item: item[0])
    before = prefix[max(0, end - 32):end]
    if re.search(r"(?:at least|no fewer than)\s+[^\s]+\s*$", before, re.IGNORECASE):
        return {"min_count": value}, "count.minimum.en.v1"
    if re.search(r"(?:at most|no more than)\s+[^\s]+\s*$", before, re.IGNORECASE):
        return {"max_count": value}, "count.maximum.en.v1"
    return {"exact_count": value}, "count.exact.en.v1"


def _trace(clause: Mapping[str, Any], rule_id: str) -> dict[str, Any]:
    return {"source_clause_ids": [clause["id"]],
            "source_span": list(clause["source_span"]),
            "source_text": clause["text"], "extractor": "rule_based",
            "rule_id": rule_id}


def _diagnostic(clause: Mapping[str, Any], kind: str, reason: str,
                sequence: int) -> dict[str, Any]:
    return {"id": f"{clause['id']}.{kind}.{sequence}",
            "clause_id": clause["id"], "kind": kind, "reason": reason,
            "blocking": True}


def _attribute_mentions(text: str,
                        definitions: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    found = []
    for name, definition in definitions.items():
        for alias in definition.get("aliases") or ():
            for match in _alias_pattern(str(alias)).finditer(text):
                found.append({"name": name, "start": match.start(),
                              "end": match.end(), **definition})
    return sorted(found, key=lambda item: (item["start"], item["end"]))


def _relation_between(text: str) -> str | None:
    for relation, patterns in _RELATIONS:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return relation
    return None


def _unsupported_residual(text: str, taxonomy: Mapping[str, Sequence[str]],
                          attributes: Mapping[str, Mapping[str, Any]]) -> str:
    """Text not explained by the deliberately narrow rule vocabulary."""
    residual = text.lower()
    aliases = [str(alias) for category, values in taxonomy.items()
               for alias in (category, *values)]
    aliases += [str(alias) for definition in attributes.values()
                for alias in definition.get("aliases") or ()]
    relation_phrases = [pattern.replace(r"\b", "").replace("(?:to the )?", "")
                        for _, patterns in _RELATIONS for pattern in patterns
                        if not any(token in pattern for token in ("(?:", "\\s", "["))]
    for alias in sorted((*aliases, *relation_phrases), key=len, reverse=True):
        residual = _alias_pattern(alias).sub(" ", residual)
    residual = re.sub(
        r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
        r"twelve|a|an|several|few|many|some|add|place|put|include|create|build|"
        r"use|have|contain|with|and|or|the|of|in|into|there|is|are|please|scene|"
        r"no|more|fewer|less|than|at|least|most|do|not|must|without|exclude)\b",
        " ", residual, flags=re.IGNORECASE)
    residual = re.sub(r"\d+", " ", residual)
    return re.sub(r"[\s.,!?;:\uff0c\u3002\uff01\uff1f\uff1b\uff1a()\[\]{}'\"-]+", "", residual)


def _normalise_taxonomy(value: Mapping[str, Sequence[str]] | None
                        ) -> Mapping[str, Sequence[str]]:
    return value or DEFAULT_TAXONOMY


def _extract_clause(clause: dict[str, Any], *,
                    taxonomy: Mapping[str, Sequence[str]],
                    attributes: Mapping[str, Mapping[str, Any]],
                    relation_policies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    text = clause["text"]
    mentions = _mentions(text, taxonomy)
    attribute_mentions = _attribute_mentions(text, attributes)
    items: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    closed_world = False

    if _VAGUE.search(text):
        unresolved.append(_diagnostic(
            clause, "vague_quantity",
            "the clause uses a vague quantity that has no frozen numeric bound", 1))

    for index, mention in enumerate(mentions):
        previous_end = mentions[index - 1]["end"] if index else 0
        prefix = text[previous_end:mention["start"]]
        forbidden = bool(_FORBIDDEN.search(prefix))
        if forbidden:
            closed_world = True
            item_type, expected, rule_id = "forbidden_object", {}, "forbidden.v1"
        else:
            count = _count(prefix)
            item_type = "object_count" if count else "object_presence"
            expected = count[0] if count else {"min_count": 1}
            rule_id = count[1] if count else "presence.v1"
        category = str(mention["category"])
        items.append({"id": f"{clause['id']}_{_slug(category)}_"
                            f"{'forbidden' if forbidden else ('count' if count else 'presence')}",
                      "type": item_type,
                      "subject": {"allowed_categories": [category]},
                      "expected": expected, "prompt_span": list(clause["source_span"]),
                      "traceability": _trace(clause, rule_id)})

        # Attribute words bind only when the text between them and the category
        # is punctuation/whitespace.  Ambiguous chains are left for review.
        adjacent = [item for item in attribute_mentions
                    if item["end"] <= mention["start"]
                    and not text[item["end"]:mention["start"]].strip(" -,")]
        for attribute in adjacent:
            items.append({
                "id": f"{clause['id']}_{_slug(category)}_{_slug(attribute['name'])}",
                "type": "attribute_value",
                "subject": {"allowed_categories": [category]},
                "expected": {"field": attribute["field"],
                             "equals": attribute["value"], "all_subjects": True},
                "preferred_evaluator": attribute.get("preferred_evaluator", "snapshot"),
                "prompt_span": list(clause["source_span"]),
                "traceability": _trace(clause, f"attribute.{attribute['name']}.v1"),
            })

    # Only consecutive category mentions are related; a wider inferred binding
    # is precisely the kind of guess this compiler exists to surface.
    for left, right in zip(mentions, mentions[1:], strict=False):
        relation = _relation_between(text[left["end"]:right["start"]])
        if relation is None:
            continue
        key = f"{relation}:{left['category']}:{right['category']}"
        policy = (relation_policies.get(key) or relation_policies.get(relation))
        required = {"near": "maximum_distance_cm", "far": "minimum_distance_cm",
                    "facing": "maximum_angle_deg"}.get(relation)
        if required and (not policy or policy.get(required) is None):
            unresolved.append(_diagnostic(
                clause, "missing_relation_policy",
                f"{relation} needs a reviewed {required} threshold", len(unresolved) + 1))
            continue
        if relation == "around" and policy is None:
            unresolved.append(_diagnostic(
                clause, "missing_relation_policy",
                "around needs reviewed distance/count/coverage thresholds",
                len(unresolved) + 1))
            continue
        expected = {"relation": relation, **dict(policy or {})}
        item_type = "around" if relation == "around" else "object_object_relation"
        items.append({
            "id": f"{clause['id']}_{_slug(left['category'])}_{relation}_"
                  f"{_slug(right['category'])}",
            "type": item_type,
            "subject": {"allowed_categories": [left["category"]]},
            "object": {"allowed_categories": [right["category"]]},
            "expected": expected, "prompt_span": list(clause["source_span"]),
            "traceability": _trace(clause, f"relation.{relation}.v1"),
        })

    if re.search(r"\b(?:red|blue|green|wooden|metal)\s+and\s+", text,
                 re.IGNORECASE):
        unresolved.append(_diagnostic(
            clause, "unresolved_attribute_binding",
            "the attribute conjunction does not bind unambiguously", len(unresolved) + 1))
    if _FORBIDDEN.search(text) and re.search(r"\bor\b|\u6216\u8005|\u6216", text):
        unresolved.append(_diagnostic(
            clause, "ambiguous_negation_scope",
            "the negation spans alternatives and needs review", len(unresolved) + 1))
    residual = _unsupported_residual(text, taxonomy, attributes)
    if mentions and residual:
        unresolved.append(_diagnostic(
            clause, "unsupported_language",
            f"unsupported prompt language remains after extraction: {residual!r}",
            len(unresolved) + 1))
    if not mentions:
        unsupported.append(_diagnostic(
            clause, "unsupported_clause",
            "no configured taxonomy object could be extracted", 1))

    blocked = bool(unresolved or unsupported)
    clause["status"] = ("unsupported" if unsupported and not items else
                        "partially_resolved" if blocked and items else
                        "unresolved" if blocked else "resolved")
    return {"clause": clause, "items": items, "unresolved": unresolved,
            "unsupported": unsupported, "closed_world": closed_world}


def compile_prompt(prompt: str, *, draft_id: str | None = None,
                   taxonomy: Mapping[str, Sequence[str]] | None = None,
                   attributes: Mapping[str, Mapping[str, Any]] | None = None,
                   relation_policies: Mapping[str, Mapping[str, Any]] | None = None,
                   world_assumption: str = "open_world",
                   assumptions: Sequence[str] = ()) -> dict[str, Any]:
    """Compile a reviewable draft; unsupported language blocks freezing."""
    prompt = str(prompt)
    if not prompt.strip():
        raise RuleCompilationError("compile_prompt requires a non-empty prompt")
    clauses = segment_prompt(prompt)
    if not clauses:
        raise RuleCompilationError("the prompt contains no requirement clause")
    results = [_extract_clause(
        clause, taxonomy=_normalise_taxonomy(taxonomy),
        attributes=attributes or DEFAULT_ATTRIBUTES,
        relation_policies=relation_policies or {}) for clause in clauses]
    draft = {
        "schema_version": "0.1.0",
        "draft_id": draft_id or "rule-draft-v1",
        "prompt": prompt,
        "world_assumption": ("closed_world" if any(
            result["closed_world"] for result in results) else world_assumption),
        "compiler": {"name": COMPILER_NAME, "version": COMPILER_VERSION,
                     "mode": "rule_based"},
        "clauses": [result["clause"] for result in results],
        "items": [item for result in results for item in result["items"]],
        "unresolved": [item for result in results for item in result["unresolved"]],
        "unsupported": [item for result in results for item in result["unsupported"]],
        "assumptions": list(assumptions),
    }
    validate_schema(draft)
    return draft


def validate_schema(draft: Mapping[str, Any]) -> None:
    """Validate the provenance boundary rather than accepting loose JSON."""
    prompt = draft.get("prompt")
    if draft.get("schema_version") != "0.1.0" or not isinstance(prompt, str):
        raise RuleCompilationError("invalid draft schema_version or prompt")
    clause_ids = set()
    for clause in draft.get("clauses") or []:
        span = clause.get("source_span") or []
        if (not isinstance(span, list) or len(span) != 2
                or not all(isinstance(value, int) for value in span)
                or span[0] < 0 or span[1] <= span[0]
                or prompt[span[0]:span[1]] != clause.get("text")):
            raise RuleCompilationError("a clause does not match its source_span")
        if clause.get("id") in clause_ids:
            raise RuleCompilationError(f"duplicate clause id {clause.get('id')}")
        clause_ids.add(clause.get("id"))
    item_ids = set()
    for item in draft.get("items") or []:
        trace = item.get("traceability") or {}
        span = trace.get("source_span") or []
        if (not trace.get("source_clause_ids")
                or any(identifier not in clause_ids
                       for identifier in trace["source_clause_ids"])
                or len(span) != 2 or prompt[span[0]:span[1]] != trace.get("source_text")):
            raise RuleCompilationError("an item has invalid source traceability")
        if item.get("id") in item_ids:
            raise RuleCompilationError(f"duplicate item id {item.get('id')}")
        item_ids.add(item.get("id"))


def _count_interval(item: Mapping[str, Any]) -> tuple[float, float] | None:
    if item.get("type") == "forbidden_object":
        return 0.0, 0.0
    if item.get("type") not in ("object_count", "object_presence"):
        return None
    expected = item.get("expected") or {}
    if expected.get("exact_count") is not None:
        value = float(expected["exact_count"])
        return value, value
    return float(expected.get("min_count") or 0), float(
        expected.get("max_count") if expected.get("max_count") is not None
        else "inf")


def validate_draft(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Return every reason a draft cannot be promoted to formal scoring."""
    validate_schema(draft)
    duplicates, conflicts = [], []
    signatures: dict[str, str] = {}
    counts: dict[str, list[Mapping[str, Any]]] = {}
    for item in draft.get("items") or []:
        signature = _canonical({key: value for key, value in item.items()
                                if key not in ("id", "traceability")})
        if signature in signatures:
            duplicates.append({"kind": "duplicate_requirement",
                               "item_ids": [signatures[signature], item["id"]],
                               "blocking": True})
        signatures[signature] = item["id"]
        interval = _count_interval(item)
        if interval is not None:
            counts.setdefault(_canonical(item.get("subject") or {}), []).append(item)
    for items in counts.values():
        intervals = [_count_interval(item) for item in items]
        minimum = max(item[0] for item in intervals if item is not None)
        maximum = min(item[1] for item in intervals if item is not None)
        if minimum > maximum:
            conflicts.append({"kind": "incompatible_count_constraints",
                              "item_ids": [item["id"] for item in items],
                              "observed_intersection": {"minimum": minimum,
                                                        "maximum": maximum},
                              "blocking": True})
    blocked_clauses = {item["clause_id"] for field in ("unresolved", "unsupported")
                       for item in draft.get(field) or [] if item.get("blocking") is not False}
    item_clauses = {identifier for item in draft.get("items") or []
                    for identifier in (item.get("traceability") or {}).get(
                        "source_clause_ids", [])}
    explicit = [item for item in draft.get("clauses") or []
                if item.get("explicit_requirement") is not False]
    covered = [item for item in explicit if item["id"] in item_clauses
               and item["id"] not in blocked_clauses and item["status"] == "resolved"]
    coverage = len(covered) / len(explicit) if explicit else 1.0
    unresolved = [item for item in draft.get("unresolved") or []
                  if item.get("blocking") is not False]
    unsupported = [item for item in draft.get("unsupported") or []
                   if item.get("blocking") is not False]
    blockers = len(unresolved) + len(unsupported) + len(conflicts) + len(duplicates)
    return {"status": "valid" if blockers == 0 and coverage == 1 else "blocked",
            "can_freeze": blockers == 0 and coverage == 1,
            "blocking_diagnostic_count": blockers,
            "coverage": {"explicit_clause_count": len(explicit),
                         "covered_explicit_clause_count": len(covered),
                         "explicit_clause_coverage": coverage,
                         "covered_clause_ids": [item["id"] for item in covered],
                         "uncovered_clause_ids": [item["id"] for item in explicit
                                                  if item not in covered]},
            "unresolved": unresolved, "unsupported": unsupported,
            "conflicts": conflicts, "duplicates": duplicates}


__all__ = ["COMPILER_NAME", "COMPILER_VERSION", "DEFAULT_ATTRIBUTES",
           "DEFAULT_TAXONOMY", "RuleCompilationError", "compile_prompt",
           "segment_prompt", "validate_draft", "validate_schema"]
