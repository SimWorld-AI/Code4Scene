"""One-shot LLM locator over compact, scene-local actor identity groups.

The frozen Stage 0 entity meanings are immutable inputs.  This module may only
suggest actor groups worth photographing; it cannot prove presence or absence
and never emits a semantic verdict.  Opaque group ids are resolved back to live
actor ids only after the native structured tool response passes local checks.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..identity_catalog import SceneIdentityGroup, build_scene_identity_groups
from .actor_inventory import ActorDescriptor
from .contracts import SceneBounds
from .existing_llm import LLMClient, LLMMessage
from .semantic_retrieval import (
    EntityIdentityQuery,
    LexicalRelation,
    LocatorCandidate,
    QueryIdentityRetrieval,
    SceneIdentityRetrievalResult,
    retrieve_scene_identities as retrieve_lexical_scene_identities,
)

_TOOL_NAME = "return_scene_identity_matches"
_SCHEMA_VERSION = "1.0"
_SYSTEM_PROMPT_REVISION = "1.0"
_MAX_REASON_CODEPOINTS = 500
_STATUS_ORDER = {
    "plausible_locator": 0,
    "uncertain": 1,
    "identity_conflict": 2,
    "unrelated": 3,
}
_SYSTEM_PROMPT = """You locate frozen visual requirements in one candidate scene catalog.

The requirements were parsed and frozen before this scene was inspected. Never reinterpret,
broaden, weaken, or drop a modifier from a requirement because the scene lacks a good match.
Return only candidate identity groups that are worth photographing. A candidate is a locator
hint, never evidence that the object exists and never a MATCH/MISMATCH decision.

Use all identity provenance together. Prefer asset, actor class, and Unreal object identity over
weak actor labels. If a weak label claims one category while strong identity says another, use
status identity_conflict and low confidence. Respect compound specificity: a rollup door is not
an ordinary door. Resolve ordinary synonyms and asset naming conventions, but do not replace a
requested category with the nearest category available in the scene. Component locator terms may
help aim a camera but do not override contradictory whole-actor identity.

For each entity return up to the requested number of best groups, including uncertain or
conflicting groups only when they remain useful visual inspection targets. Use unrelated only to
record a considered but unusable group; unrelated groups will not be photographed. Return every
entity_id exactly once and use only the opaque group_ids supplied in the catalog. Return exactly
one call to the provided tool and no prose.
"""


def _tool_schema(*, query_count: int, top_k: int) -> dict[str, Any]:
    candidate = {
        "type": "object",
        "properties": {
            "group_id": {"type": "string", "pattern": r"^g[0-9]{4}$"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "status": {
                "type": "string",
                "enum": list(_STATUS_ORDER),
            },
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": _MAX_REASON_CODEPOINTS,
            },
        },
        "required": ["group_id", "confidence", "status", "reason"],
        "additionalProperties": False,
    }
    match = {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string", "minLength": 1},
            "candidates": {
                "type": "array",
                "maxItems": top_k,
                "items": candidate,
            },
        },
        "required": ["entity_id", "candidates"],
        "additionalProperties": False,
    }
    return {
        "name": _TOOL_NAME,
        "description": "Return locator-only matches between frozen entities and scene identity groups.",
        "parameters": {
            "type": "object",
            "properties": {
                "schema_version": {"type": "string", "enum": [_SCHEMA_VERSION]},
                "matches": {
                    "type": "array",
                    "minItems": query_count,
                    "maxItems": query_count,
                    "items": match,
                },
            },
            "required": ["schema_version", "matches"],
            "additionalProperties": False,
        },
    }


def _strict_object(
    value: Any, *, required: frozenset[str], path: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    keys = set(value)
    if keys != required:
        raise ValueError(
            f"{path} fields differ: missing={sorted(required - keys)!r} "
            f"extra={sorted(keys - required)!r}"
        )
    return value


def _parse_matches(
    response: Any,
    *,
    queries: Sequence[EntityIdentityQuery],
    groups: Sequence[SceneIdentityGroup],
    top_k: int,
) -> dict[str, tuple[dict[str, Any], ...]]:
    calls = tuple(getattr(response, "tool_calls", ()) or ())
    if len(calls) != 1:
        raise ValueError("locator response must contain exactly one tool call")
    call = calls[0]
    if str(getattr(call, "name", "")) != _TOOL_NAME:
        raise ValueError(f"locator response must call {_TOOL_NAME!r}")
    raw = _strict_object(
        getattr(call, "arguments", None),
        required=frozenset({"schema_version", "matches"}),
        path="$",
    )
    if raw["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported locator schema version")
    matches = raw["matches"]
    if not isinstance(matches, list) or len(matches) != len(queries):
        raise ValueError("locator matches must cover every frozen entity exactly once")
    query_ids = {value.query_id for value in queries}
    group_ids = {value.group_id for value in groups}
    parsed: dict[str, tuple[dict[str, Any], ...]] = {}
    for match_index, value in enumerate(matches):
        match = _strict_object(
            value,
            required=frozenset({"entity_id", "candidates"}),
            path=f"matches[{match_index}]",
        )
        entity_id = str(match["entity_id"]).strip()
        if entity_id not in query_ids or entity_id in parsed:
            raise ValueError(f"invalid or duplicate locator entity_id {entity_id!r}")
        candidates = match["candidates"]
        if not isinstance(candidates, list) or len(candidates) > top_k:
            raise ValueError(f"matches[{match_index}].candidates exceeds Top-K")
        selected: list[dict[str, Any]] = []
        seen_groups: set[str] = set()
        for candidate_index, candidate_value in enumerate(candidates):
            candidate = _strict_object(
                candidate_value,
                required=frozenset({"group_id", "confidence", "status", "reason"}),
                path=f"matches[{match_index}].candidates[{candidate_index}]",
            )
            group_id = str(candidate["group_id"]).strip()
            if group_id not in group_ids or group_id in seen_groups:
                raise ValueError(f"invalid or duplicate locator group_id {group_id!r}")
            confidence = candidate["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("locator confidence must be numeric")
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("locator confidence must be finite and in [0, 1]")
            status = str(candidate["status"]).strip()
            if status not in _STATUS_ORDER:
                raise ValueError(f"invalid locator status {status!r}")
            reason = candidate["reason"]
            if (
                not isinstance(reason, str)
                or reason != reason.strip()
                or not reason
                or len(reason) > _MAX_REASON_CODEPOINTS
            ):
                raise ValueError("locator reason must be non-empty canonical text")
            seen_groups.add(group_id)
            selected.append(
                {
                    "group_id": group_id,
                    "confidence": confidence,
                    "status": status,
                    "reason": reason,
                }
            )
        parsed[entity_id] = tuple(selected)
    if set(parsed) != query_ids:
        raise ValueError("locator response omitted a frozen entity")
    return parsed


class LLMIdentityLocatorBackend:
    """Native-tool one-call locator using Code4Scene's configured VLM client."""

    def __init__(self, client: LLMClient, *, max_tokens: int | None = None) -> None:
        if bool(getattr(client, "_text_action_mode", False)):
            raise ValueError("LLM identity locator requires native structured tool calls")
        if hasattr(client, "_strict_tool_calls") and not bool(
            client._strict_tool_calls
        ):
            raise ValueError("LLM identity locator requires strict_tool_calls=True")
        selected_tokens = (
            int(getattr(client, "max_tokens", 4096))
            if max_tokens is None
            else max_tokens
        )
        if (
            isinstance(selected_tokens, bool)
            or not isinstance(selected_tokens, int)
            or selected_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        self.client = client
        self.max_tokens = selected_tokens
        self._lock = threading.Lock()
        self._call_count = 0

    @property
    def name(self) -> str:
        model = str(getattr(self.client, "model", type(self.client).__name__)).strip()
        return f"llm_identity_locator:{model or type(self.client).__name__}"

    def retrieve_scene_identities(
        self,
        queries: Iterable[EntityIdentityQuery],
        actors: Iterable[ActorDescriptor],
        *,
        scene_bounds: SceneBounds,
        top_k: int = 5,
        include_resolved_locators: bool = True,
    ) -> SceneIdentityRetrievalResult:
        """Run lexical retrieval plus at most one structured LLM locator call."""

        query_values = tuple(queries)
        actor_values = tuple(actors)
        base = retrieve_lexical_scene_identities(
            query_values,
            actor_values,
            scene_bounds=scene_bounds,
            top_k=top_k,
            include_resolved_locators=include_resolved_locators,
        )
        groups = build_scene_identity_groups(actor_values, scene_bounds=scene_bounds)
        if not query_values or not groups:
            return SceneIdentityRetrievalResult(
                queries=base.queries,
                unique_identity_count=base.unique_identity_count,
                eligible_actor_count=base.eligible_actor_count,
                semantic_backend=self.name,
                identity_group_count=len(groups),
            )

        payload = {
            "schema_version": _SCHEMA_VERSION,
            "top_k_per_entity": top_k,
            "requirements": [
                {
                    "entity_id": query.query_id,
                    "canonical_name": query.terms[0],
                    "aliases": list(query.terms[1:]),
                }
                for query in query_values
            ],
            "scene_identity_groups": [group.llm_payload() for group in groups],
        }
        schema = _tool_schema(query_count=len(query_values), top_k=top_k)
        with self._lock:
            self._call_count += 1
            request_id = f"s2l_{self._call_count:06d}"
        model = str(getattr(self.client, "model", type(self.client).__name__)).strip()
        manifest = {
            "request_id": request_id,
            "call_kind": "scene_identity_group_locator",
            "schema_version": _SCHEMA_VERSION,
            "system_prompt_revision": _SYSTEM_PROMPT_REVISION,
            "model": model or type(self.client).__name__,
            "query_count": len(query_values),
            "identity_group_count": len(groups),
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "tool_name": _TOOL_NAME,
        }
        try:
            response = self.client.chat(
                [
                    LLMMessage.text("system", _SYSTEM_PROMPT),
                    LLMMessage.text(
                        "user",
                        "Frozen requirements and candidate scene identity catalog:\n"
                        + json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    ),
                ],
                [schema],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            parsed = _parse_matches(
                response,
                queries=query_values,
                groups=groups,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed to lexical retrieval
            error = f"{type(exc).__name__}: {str(exc)[:1200]}"
            return SceneIdentityRetrievalResult(
                queries=base.queries,
                unique_identity_count=base.unique_identity_count,
                eligible_actor_count=base.eligible_actor_count,
                semantic_backend=self.name,
                semantic_backend_error=error,
                identity_group_count=len(groups),
                semantic_request_manifest=(manifest,),
                semantic_raw_records=(
                    {
                        "request_id": request_id,
                        "call_kind": "scene_identity_group_locator",
                        "status": "error",
                        "error": error,
                    },
                ),
            )

        group_by_id = {value.group_id: value for value in groups}
        merged_queries: list[QueryIdentityRetrieval] = []
        for query in query_values:
            base_query = base.for_query(query.query_id)
            if base_query is None:  # pragma: no cover - protected by base contract
                raise RuntimeError("lexical retrieval omitted a query")
            exact_actor_ids = {
                actor_id
                for hit in base_query.exact_matches
                for actor_id in hit.actor_ids
            }
            seen_actor_ids = set(exact_actor_ids)
            candidates: list[LocatorCandidate] = []
            llm_values = sorted(
                parsed[query.query_id],
                key=lambda value: (
                    _STATUS_ORDER[value["status"]],
                    -value["confidence"],
                    value["group_id"],
                ),
            )
            for value in llm_values:
                if value["status"] == "unrelated":
                    continue
                group = group_by_id[value["group_id"]]
                actor_ids = tuple(
                    actor_id
                    for actor_id in group.actor_ids
                    if actor_id not in seen_actor_ids
                )
                if not actor_ids:
                    continue
                seen_actor_ids.update(actor_ids)
                candidates.append(
                    LocatorCandidate(
                        query_id=query.query_id,
                        matched_query_term=query.terms[0],
                        identity_term=group.primary_identity_term,
                        actor_ids=actor_ids,
                        lexical_relation=LexicalRelation.NONE,
                        semantic_score=value["confidence"],
                        rank=len(candidates) + 1,
                        identity_sources=group.identity_sources,
                        raw_values=group.raw_values,
                        locator_group_id=group.group_id,
                        locator_status=value["status"],
                        locator_reason=value["reason"],
                    )
                )
                if len(candidates) >= top_k:
                    break
            for lexical in base_query.locator_candidates:
                if len(candidates) >= top_k:
                    break
                actor_ids = tuple(
                    actor_id
                    for actor_id in lexical.actor_ids
                    if actor_id not in seen_actor_ids
                )
                if not actor_ids:
                    continue
                seen_actor_ids.update(actor_ids)
                replacement = LocatorCandidate(
                    query_id=lexical.query_id,
                    matched_query_term=lexical.matched_query_term,
                    identity_term=lexical.identity_term,
                    actor_ids=actor_ids,
                    lexical_relation=lexical.lexical_relation,
                    semantic_score=lexical.semantic_score,
                    rank=len(candidates) + 1,
                    identity_sources=lexical.identity_sources,
                    raw_values=lexical.raw_values,
                )
                if lexical.can_defer_absence and (
                    replacement.lexical_relation is LexicalRelation.NONE
                ):
                    object.__setattr__(replacement, "can_defer_absence", True)
                candidates.append(replacement)
            merged_queries.append(
                QueryIdentityRetrieval(
                    query_id=query.query_id,
                    normalized_terms=query.terms,
                    exact_matches=base_query.exact_matches,
                    locator_candidates=tuple(candidates),
                    has_calibrated_absence_defer_candidate=(
                        base_query.has_calibrated_absence_defer_candidate
                    ),
                )
            )

        raw_matches = [
            {
                "entity_id": query.query_id,
                "candidates": [dict(value) for value in parsed[query.query_id]],
            }
            for query in query_values
        ]
        usage = getattr(response, "usage", {})
        return SceneIdentityRetrievalResult(
            queries=tuple(merged_queries),
            unique_identity_count=base.unique_identity_count,
            eligible_actor_count=base.eligible_actor_count,
            semantic_backend=self.name,
            identity_group_count=len(groups),
            semantic_request_manifest=(manifest,),
            semantic_raw_records=(
                {
                    "request_id": request_id,
                    "call_kind": "scene_identity_group_locator",
                    "status": "success",
                    "matches": raw_matches,
                    "usage": dict(usage) if isinstance(usage, Mapping) else {},
                },
            ),
        )


__all__ = [
    "LLMIdentityLocatorBackend",
    "SceneIdentityGroup",
    "build_scene_identity_groups",
]
