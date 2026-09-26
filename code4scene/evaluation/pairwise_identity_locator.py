"""Locator-only semantic edges for GT logical-object correspondence.

This is intentionally not a semantic verifier.  It compares two compact
identity catalogs and proposes only candidate edges for the deterministic GT
geometry controller.  Size/spatial blocking and Hungarian assignment remain
outside this module, and no response here is a MATCH/MISMATCH or a score.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .identity_catalog import SceneIdentityGroup, build_scene_identity_groups
from .requirement_graph.actor_inventory import (
    ActorBounds,
    ActorDescriptor,
    IdentitySource,
    IdentityTerm,
    build_actor_descriptor,
)
from .requirement_graph.contracts import SceneBounds
from .requirement_graph.existing_llm import LLMClient, LLMMessage

_TOOL_NAME = "return_pairwise_identity_edges"
_SCHEMA_VERSION = "1.0"
_SYSTEM_PROMPT_REVISION = "1.0"
_MAX_REASON_CODEPOINTS = 500
_STATUSES = (
    "plausible_locator",
    "uncertain",
    "identity_conflict",
    "unrelated",
)

_SYSTEM_PROMPT = """You propose identity-locator edges between canonical and candidate scene
identity catalogs. Both catalogs describe logical scene objects; neither is a natural-language
prompt. Your output is only a shortlist for later geometric blocking and assignment. It is never
evidence that the reconstruction is correct, never a MATCH/MISMATCH decision, and never a score.

Use identity provenance together. Exact asset/category matches are handled deterministically and
normally omitted from this long-tail request. Recognize ordinary synonyms and different asset
implementations of the same object kind, but do not broaden a specific object into a nearby generic
category. Prefer strong asset/class/Unreal identity over weak labels. Mark conflicts explicitly.
Do not use or infer spatial position, size, rotation, count, material, colour, or visual style; the
controller handles geometry and visual_as_judge handles appearance. When a canonical group has a
compatible_candidate_group_ids field, it is a hard controller-computed allowlist: return only ids
from that list and do not infer why other groups were excluded.

For every canonical_group_id return up to the requested number of candidate groups. Only
plausible_locator means the controller may create a semantic candidate edge. uncertain and
identity_conflict remain audit-only; unrelated records a considered but unusable group. Use only
the opaque group ids in the catalogs. Return exactly one tool call and no prose.
"""


@dataclass(frozen=True, slots=True)
class PairwiseIdentityEdge:
    """One group-level locator suggestion with controller-only memberships."""

    candidate_object_ids: tuple[str, ...]
    canonical_object_ids: tuple[str, ...]
    confidence: float
    status: str
    reason: str
    candidate_group_id: str
    canonical_group_id: str

    def __post_init__(self) -> None:
        if (
            not self.candidate_object_ids
            or not self.canonical_object_ids
            or any(not value for value in self.candidate_object_ids)
            or any(not value for value in self.canonical_object_ids)
        ):
            raise ValueError("pairwise identity edge memberships must be non-empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("pairwise identity confidence must be finite in [0, 1]")
        if self.status not in _STATUSES:
            raise ValueError(f"invalid pairwise identity status {self.status!r}")
        if not self.reason or len(self.reason) > _MAX_REASON_CODEPOINTS:
            raise ValueError("pairwise identity reason must be non-empty and bounded")

    @property
    def usable(self) -> bool:
        """Only a plausible locator may open an assignment edge."""

        return self.status == "plausible_locator"

    @property
    def identity_cost(self) -> float:
        """Bounded locator cost; geometry still decides the correspondence."""

        return 0.25 + 0.25 * (1.0 - self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_object_ids": list(self.candidate_object_ids),
            "canonical_object_ids": list(self.canonical_object_ids),
            "confidence": self.confidence,
            "status": self.status,
            "reason": self.reason,
            "candidate_group_id": self.candidate_group_id,
            "canonical_group_id": self.canonical_group_id,
            "usable_for_assignment": self.usable,
        }


@dataclass(frozen=True, slots=True)
class PairwiseIdentityRetrievalResult:
    """Edges plus complete reproducibility/audit metadata."""

    backend: str
    edges: tuple[PairwiseIdentityEdge, ...] = ()
    backend_error: str | None = None
    long_tail_candidate_object_count: int = 0
    long_tail_canonical_object_count: int = 0
    compatible_object_pair_count: int = 0
    compatible_group_pair_count: int = 0
    candidate_group_count: int = 0
    canonical_group_count: int = 0
    request_manifests: tuple[Mapping[str, Any], ...] = ()
    raw_records: tuple[Mapping[str, Any], ...] = ()

    def to_audit(self) -> dict[str, Any]:
        return {
            "role": "locator_only_not_semantic_verdict",
            "backend": self.backend,
            "backend_error": self.backend_error,
            "long_tail_candidate_object_count": (
                self.long_tail_candidate_object_count
            ),
            "long_tail_canonical_object_count": (
                self.long_tail_canonical_object_count
            ),
            "compatible_object_pair_count": self.compatible_object_pair_count,
            "compatible_group_pair_count": self.compatible_group_pair_count,
            "candidate_group_count": self.candidate_group_count,
            "canonical_group_count": self.canonical_group_count,
            "edge_count": len(self.edges),
            "usable_edge_count": sum(edge.usable for edge in self.edges),
            "edges": [edge.to_dict() for edge in self.edges],
            "request_manifests": [dict(value) for value in self.request_manifests],
            "raw_records": [dict(value) for value in self.raw_records],
        }


class PairwiseIdentityLocator(Protocol):
    """Dependency injected into the pure GT geometry comparison."""

    @property
    def name(self) -> str: ...

    def locate(
        self,
        candidate_objects: Sequence[Any],
        canonical_objects: Sequence[Any],
        *,
        compatible_pairs: Mapping[str, Sequence[str]] | None = None,
    ) -> PairwiseIdentityRetrievalResult: ...


def _object_descriptor(value: Any) -> ActorDescriptor:
    """Adapt a GT logical object to the shared identity contract."""

    identity_terms: list[IdentityTerm] = []
    for asset in value.assets:
        identity_terms.extend(
            build_actor_descriptor(
                live_actor_id="identity-adapter",
                asset_path=asset,
            ).identity_terms
        )
    for category in value.categories:
        identity_terms.append(
            IdentityTerm(
                category,
                IdentitySource.STRUCTURED_TAG,
                raw_value=category,
            )
        )
    for actor_class in value.classes:
        identity_terms.extend(
            build_actor_descriptor(
                live_actor_id="identity-adapter",
                actor_class=actor_class,
            ).identity_terms
        )
    for label in value.labels:
        identity_terms.extend(
            build_actor_descriptor(
                live_actor_id="identity-adapter",
                actor_label=label,
            ).identity_terms
        )

    actor = value.as_actor()
    raw_bounds = actor.get("bounds")
    bounds = None
    if isinstance(raw_bounds, Mapping):
        center = raw_bounds.get("origin_cm") or raw_bounds.get("center_cm")
        extent = raw_bounds.get("extent_cm")
        if isinstance(center, Sequence) and isinstance(extent, Sequence):
            bounds = ActorBounds(tuple(center), tuple(extent))
    return ActorDescriptor(
        live_actor_id=value.object_id,
        identity_terms=tuple(identity_terms),
        bounds=bounds,
        active=True,
        renderable=True,
        in_current_level=True,
    )


def _catalog(
    values: Sequence[Any],
) -> tuple[tuple[ActorDescriptor, ...], tuple[SceneIdentityGroup, ...]]:
    descriptors = tuple(_object_descriptor(value) for value in values)
    bounded = [value.bounds for value in descriptors if value.bounds is not None]
    if not bounded:
        return descriptors, ()
    minimum = tuple(min(value.min_cm[axis] for value in bounded) for axis in range(3))
    maximum = tuple(max(value.max_cm[axis] for value in bounded) for axis in range(3))
    bounds = SceneBounds(minimum, maximum)
    return descriptors, build_scene_identity_groups(
        descriptors,
        scene_bounds=bounds,
    )


def select_long_tail_objects(
    values: Sequence[Any],
    other: Sequence[Any],
) -> tuple[Any, ...]:
    """Exclude identities already handled by trusted exact tiers."""

    other_assets = {asset for value in other for asset in value.assets}
    other_categories = {category for value in other for category in value.categories}
    return tuple(
        value
        for value in values
        if not (
            set(value.assets) & other_assets
            or set(value.categories) & other_categories
        )
    )


def _tool_schema(*, canonical_group_count: int, top_k: int) -> dict[str, Any]:
    candidate = {
        "type": "object",
        "properties": {
            "candidate_group_id": {
                "type": "string",
                "pattern": r"^g[0-9]{4}$",
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "status": {"type": "string", "enum": list(_STATUSES)},
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": _MAX_REASON_CODEPOINTS,
            },
        },
        "required": ["candidate_group_id", "confidence", "status", "reason"],
        "additionalProperties": False,
    }
    match = {
        "type": "object",
        "properties": {
            "canonical_group_id": {
                "type": "string",
                "pattern": r"^g[0-9]{4}$",
            },
            "candidates": {
                "type": "array",
                "maxItems": top_k,
                "items": candidate,
            },
        },
        "required": ["canonical_group_id", "candidates"],
        "additionalProperties": False,
    }
    return {
        "name": _TOOL_NAME,
        "description": "Return locator-only edges between canonical and candidate identity groups.",
        "parameters": {
            "type": "object",
            "properties": {
                "schema_version": {"type": "string", "enum": [_SCHEMA_VERSION]},
                "matches": {
                    "type": "array",
                    "minItems": canonical_group_count,
                    "maxItems": canonical_group_count,
                    "items": match,
                },
            },
            "required": ["schema_version", "matches"],
            "additionalProperties": False,
        },
    }


def _strict_object(
    value: Any,
    *,
    required: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    keys = {str(key) for key in value}
    if keys != required:
        raise ValueError(
            f"{path} fields must be exactly {sorted(required)}, got {sorted(keys)}"
        )
    return value


def _parse(
    response: Any,
    *,
    canonical_groups: Sequence[SceneIdentityGroup],
    candidate_groups: Sequence[SceneIdentityGroup],
    top_k: int,
    allowed_candidate_groups: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, tuple[dict[str, Any], ...]]:
    if getattr(response, "text", None):
        raise ValueError("pairwise locator response must not contain prose")
    calls = getattr(response, "tool_calls", None)
    if not isinstance(calls, Sequence) or len(calls) != 1:
        raise ValueError("pairwise locator must return exactly one tool call")
    call = calls[0]
    if getattr(call, "name", None) != _TOOL_NAME:
        raise ValueError(f"pairwise locator returned unexpected tool {getattr(call, 'name', None)!r}")
    arguments = _strict_object(
        getattr(call, "arguments", None),
        required=frozenset({"schema_version", "matches"}),
        path="tool arguments",
    )
    if arguments["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("pairwise locator schema version mismatch")
    raw_matches = arguments["matches"]
    if not isinstance(raw_matches, list):
        raise ValueError("pairwise locator matches must be an array")

    canonical_ids = {value.group_id for value in canonical_groups}
    candidate_ids = {value.group_id for value in candidate_groups}
    parsed: dict[str, tuple[dict[str, Any], ...]] = {}
    for index, raw_match in enumerate(raw_matches):
        match = _strict_object(
            raw_match,
            required=frozenset({"canonical_group_id", "candidates"}),
            path=f"matches[{index}]",
        )
        canonical_id = str(match["canonical_group_id"])
        if canonical_id not in canonical_ids or canonical_id in parsed:
            raise ValueError("pairwise locator returned an unknown or duplicate canonical group")
        candidates = match["candidates"]
        if not isinstance(candidates, list) or len(candidates) > top_k:
            raise ValueError("pairwise locator candidates must be a bounded array")
        seen: set[str] = set()
        selected: list[dict[str, Any]] = []
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate = _strict_object(
                raw_candidate,
                required=frozenset(
                    {"candidate_group_id", "confidence", "status", "reason"}
                ),
                path=f"matches[{index}].candidates[{candidate_index}]",
            )
            candidate_id = str(candidate["candidate_group_id"])
            if candidate_id not in candidate_ids or candidate_id in seen:
                raise ValueError("pairwise locator returned an unknown or duplicate candidate group")
            if (
                allowed_candidate_groups is not None
                and candidate_id
                not in allowed_candidate_groups.get(canonical_id, frozenset())
            ):
                raise ValueError(
                    "pairwise locator returned a candidate outside the controller allowlist"
                )
            confidence = candidate["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("pairwise locator confidence must be numeric")
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("pairwise locator confidence must be finite in [0, 1]")
            status = str(candidate["status"])
            if status not in _STATUSES:
                raise ValueError(f"invalid pairwise locator status {status!r}")
            reason = candidate["reason"]
            if (
                not isinstance(reason, str)
                or reason != reason.strip()
                or not reason
                or len(reason) > _MAX_REASON_CODEPOINTS
            ):
                raise ValueError("pairwise locator reason must be non-empty canonical text")
            seen.add(candidate_id)
            selected.append(
                {
                    "candidate_group_id": candidate_id,
                    "confidence": confidence,
                    "status": status,
                    "reason": reason,
                }
            )
        parsed[canonical_id] = tuple(selected)
    if set(parsed) != canonical_ids:
        raise ValueError("pairwise locator omitted a canonical identity group")
    return parsed


class LLMPairwiseIdentityLocatorBackend:
    """One structured Qwen/VLM call over compact unmatched identity groups."""

    def __init__(
        self,
        client: LLMClient,
        *,
        top_k: int = 5,
        max_tokens: int | None = None,
    ) -> None:
        if bool(getattr(client, "_text_action_mode", False)):
            raise ValueError("pairwise identity locator requires native tool calls")
        if hasattr(client, "_strict_tool_calls") and not bool(client._strict_tool_calls):
            raise ValueError("pairwise identity locator requires strict_tool_calls=True")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
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
        self.top_k = top_k
        self.max_tokens = selected_tokens
        self._lock = threading.Lock()
        self._call_count = 0

    @property
    def name(self) -> str:
        model = str(getattr(self.client, "model", type(self.client).__name__)).strip()
        return f"llm_pairwise_identity_locator:{model or type(self.client).__name__}"

    def locate(
        self,
        candidate_objects: Sequence[Any],
        canonical_objects: Sequence[Any],
        *,
        compatible_pairs: Mapping[str, Sequence[str]] | None = None,
    ) -> PairwiseIdentityRetrievalResult:
        all_long_candidate = select_long_tail_objects(
            candidate_objects, canonical_objects
        )
        all_long_canonical = select_long_tail_objects(
            canonical_objects, candidate_objects
        )
        long_candidate = all_long_candidate
        long_canonical = all_long_canonical
        normalized_pairs: dict[str, frozenset[str]] | None = None
        if compatible_pairs is not None:
            long_candidate_ids = {
                value.object_id for value in all_long_candidate
            }
            long_canonical_ids = {
                value.object_id for value in all_long_canonical
            }
            normalized_pairs = {
                str(canonical_id): frozenset(
                    str(candidate_id)
                    for candidate_id in candidate_ids
                    if str(candidate_id) in long_candidate_ids
                )
                for canonical_id, candidate_ids in compatible_pairs.items()
                if str(canonical_id) in long_canonical_ids
            }
            normalized_pairs = {
                canonical_id: candidate_ids
                for canonical_id, candidate_ids in normalized_pairs.items()
                if candidate_ids
            }
            retained_candidate_ids = {
                candidate_id
                for candidate_ids in normalized_pairs.values()
                for candidate_id in candidate_ids
            }
            long_candidate = tuple(
                value
                for value in all_long_candidate
                if value.object_id in retained_candidate_ids
            )
            long_canonical = tuple(
                value
                for value in all_long_canonical
                if value.object_id in normalized_pairs
            )
        _, candidate_groups = _catalog(long_candidate)
        _, canonical_groups = _catalog(long_canonical)
        compatible_object_pair_count = (
            sum(len(value) for value in normalized_pairs.values())
            if normalized_pairs is not None
            else len(long_candidate) * len(long_canonical)
        )

        allowed_candidate_groups: dict[str, frozenset[str]] | None = None
        if normalized_pairs is not None and candidate_groups and canonical_groups:
            candidate_group_by_object_id = {
                object_id: group.group_id
                for group in candidate_groups
                for object_id in group.actor_ids
            }
            allowed_candidate_groups = {}
            for canonical_group in canonical_groups:
                allowed_candidate_groups[canonical_group.group_id] = frozenset(
                    candidate_group_by_object_id[candidate_object_id]
                    for canonical_object_id in canonical_group.actor_ids
                    for candidate_object_id in normalized_pairs.get(
                        canonical_object_id, frozenset()
                    )
                    if candidate_object_id in candidate_group_by_object_id
                )
        compatible_group_pair_count = (
            sum(len(value) for value in allowed_candidate_groups.values())
            if allowed_candidate_groups is not None
            else len(candidate_groups) * len(canonical_groups)
        )
        if not candidate_groups or not canonical_groups:
            return PairwiseIdentityRetrievalResult(
                backend=self.name,
                long_tail_candidate_object_count=len(all_long_candidate),
                long_tail_canonical_object_count=len(all_long_canonical),
                compatible_object_pair_count=compatible_object_pair_count,
                compatible_group_pair_count=compatible_group_pair_count,
                candidate_group_count=len(candidate_groups),
                canonical_group_count=len(canonical_groups),
            )

        payload = {
            "schema_version": _SCHEMA_VERSION,
            "top_k_per_canonical_group": self.top_k,
            "canonical_identity_groups": [
                {
                    **group.llm_payload(),
                    **(
                        {
                            "compatible_candidate_group_ids": sorted(
                                allowed_candidate_groups[group.group_id]
                            )
                        }
                        if allowed_candidate_groups is not None
                        else {}
                    ),
                }
                for group in canonical_groups
            ],
            "candidate_identity_groups": [
                group.llm_payload() for group in candidate_groups
            ],
        }
        schema = _tool_schema(
            canonical_group_count=len(canonical_groups),
            top_k=self.top_k,
        )
        with self._lock:
            self._call_count += 1
            request_id = f"gtid_{self._call_count:06d}"
        model = str(getattr(self.client, "model", type(self.client).__name__)).strip()
        manifest = {
            "request_id": request_id,
            "call_kind": "gt_pairwise_identity_group_locator",
            "schema_version": _SCHEMA_VERSION,
            "system_prompt_revision": _SYSTEM_PROMPT_REVISION,
            "model": model or type(self.client).__name__,
            "candidate_group_count": len(candidate_groups),
            "canonical_group_count": len(canonical_groups),
            "long_tail_candidate_object_count": len(all_long_candidate),
            "long_tail_canonical_object_count": len(all_long_canonical),
            "compatible_object_pair_count": compatible_object_pair_count,
            "compatible_group_pair_count": compatible_group_pair_count,
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
                        "Canonical and candidate logical-object identity catalogs:\n"
                        + json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    ),
                ],
                [schema],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            parsed = _parse(
                response,
                canonical_groups=canonical_groups,
                candidate_groups=candidate_groups,
                top_k=self.top_k,
                allowed_candidate_groups=allowed_candidate_groups,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed to deterministic tiers
            error = f"{type(exc).__name__}: {str(exc)[:1200]}"
            return PairwiseIdentityRetrievalResult(
                backend=self.name,
                backend_error=error,
                long_tail_candidate_object_count=len(all_long_candidate),
                long_tail_canonical_object_count=len(all_long_canonical),
                compatible_object_pair_count=compatible_object_pair_count,
                compatible_group_pair_count=compatible_group_pair_count,
                candidate_group_count=len(candidate_groups),
                canonical_group_count=len(canonical_groups),
                request_manifests=(manifest,),
                raw_records=(
                    {
                        "request_id": request_id,
                        "call_kind": "gt_pairwise_identity_group_locator",
                        "status": "error",
                        "error": error,
                    },
                ),
            )

        candidate_by_group = {value.group_id: value for value in candidate_groups}
        canonical_by_group = {value.group_id: value for value in canonical_groups}
        edges: list[PairwiseIdentityEdge] = []
        raw_matches: list[dict[str, Any]] = []
        for canonical_group in canonical_groups:
            values = parsed[canonical_group.group_id]
            raw_matches.append(
                {
                    "canonical_group_id": canonical_group.group_id,
                    "candidates": [dict(value) for value in values],
                }
            )
            for value in values:
                candidate_group = candidate_by_group[value["candidate_group_id"]]
                edges.append(
                    PairwiseIdentityEdge(
                        candidate_object_ids=candidate_group.actor_ids,
                        canonical_object_ids=canonical_by_group[
                            canonical_group.group_id
                        ].actor_ids,
                        confidence=value["confidence"],
                        status=value["status"],
                        reason=value["reason"],
                        candidate_group_id=candidate_group.group_id,
                        canonical_group_id=canonical_group.group_id,
                    )
                )
        usage = getattr(response, "usage", {})
        return PairwiseIdentityRetrievalResult(
            backend=self.name,
            edges=tuple(edges),
            long_tail_candidate_object_count=len(all_long_candidate),
            long_tail_canonical_object_count=len(all_long_canonical),
            compatible_object_pair_count=compatible_object_pair_count,
            compatible_group_pair_count=compatible_group_pair_count,
            candidate_group_count=len(candidate_groups),
            canonical_group_count=len(canonical_groups),
            request_manifests=(manifest,),
            raw_records=(
                {
                    "request_id": request_id,
                    "call_kind": "gt_pairwise_identity_group_locator",
                    "status": "success",
                    "matches": raw_matches,
                    "usage": dict(usage) if isinstance(usage, Mapping) else {},
                },
            ),
        )


__all__ = [
    "LLMPairwiseIdentityLocatorBackend",
    "PairwiseIdentityEdge",
    "PairwiseIdentityLocator",
    "PairwiseIdentityRetrievalResult",
    "select_long_tail_objects",
]
