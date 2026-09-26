"""Scene-local open-vocabulary retrieval for Stage 2 acquisition hints.

This module deliberately separates two operations that have different safety
requirements:

* exact lexical hits preserve authoritative acquisition targets; and
* token-substring or semantic Top-K results provide locator hints only.

Stage 1 does not import or invoke this module.  In particular, none of these
retrieval results is an object-existence verdict.

The semantic backend is optional and receives only the de-duplicated identities
that are present in the current live level.  It is called at most once for all
query terms selected for locator retrieval, rather than once per entity/actor
pair.  Backend failures fail
closed: exact lexical results are preserved, token-substring locator hints remain
available, and no semantic result is fabricated.

``LocalEmbeddingServiceBackend`` speaks the same ``POST /embed`` dense+sparse
JSON contract as a standalone embedding service (BGE plus Qdrant/BM25),
but performs ranking against the scene-local identity set in this process.
Nothing in this module loads a model or contacts a service unless a backend is
explicitly supplied by the caller.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .actor_inventory import (
    ActorDescriptor,
    IdentitySource,
    IdentityStrength,
    IdentityTerm,
    normalize_identity_term,
)
from .contracts import JsonSerializable, SceneBounds


# Prompt nouns that are too generic to aim a camera.  The remaining content
# tokens are allowed only in the locator branch; authoritative exact matching
# continues to use the frozen query terms unchanged.
_LOCATOR_QUERY_STOPWORDS = frozenset(
    {
        "area",
        "arrangement",
        "collection",
        "decoration",
        "feature",
        "group",
        "item",
        "object",
        "scene",
        "set",
        "style",
        "structure",
        "unit",
    }
)

# Small deterministic bridges for common asset vocabulary.  Every expansion
# remains locator-only and therefore cannot create an existence verdict.
_LOCATOR_QUERY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "ac unit": ("ac", "air conditioner", "air conditioning"),
    "air conditioner": ("ac", "air conditioning"),
    "air conditioning unit": ("ac", "air conditioner", "air conditioning"),
    "garbage bin": ("bin", "trash", "trash bin", "dumpster"),
    "neon sign": ("neon", "sign"),
    "roll up door": ("door", "shutter", "roller shutter"),
    "rollup door": ("door", "shutter", "roller shutter"),
    "shopfront clutter": ("shopfront", "clutter"),
    "trash bin": ("bin", "trash", "garbage bin", "dumpster"),
}


class LexicalRelation(str, Enum):
    """Directional lexical relation between a query and a live identity."""

    EXACT = "exact"
    QUERY_TOKEN_SUBSTRING = "query_token_substring"
    NONE = "none"

    @property
    def priority(self) -> int:
        return {
            LexicalRelation.EXACT: 2,
            LexicalRelation.QUERY_TOKEN_SUBSTRING: 1,
            LexicalRelation.NONE: 0,
        }[self]


class CandidateUse(str, Enum):
    """What downstream code is permitted to do with a retrieval result."""

    LOCATOR_ONLY = "locator_only"


class SemanticRetrievalBackend(Protocol):
    """Batch scoring interface for an optional scene-local semantic backend.

    Implementations must return one finite, higher-is-better score per
    ``query x identity`` pair.  Scores do not need to be calibrated because
    this layer uses them only for within-query Top-K ordering.
    """

    @property
    def name(self) -> str:
        """Stable backend/model name for diagnostics and caches."""

    def score(
        self,
        queries: Sequence[str],
        identities: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        """Return a matrix with shape ``len(queries) x len(identities)``."""


@dataclass(frozen=True, slots=True)
class CalibratedSemanticScoreBatch:
    """Ranking scores plus an explicit absolute plausibility decision.

    ``scores`` retain their original, possibly relative, ranking semantics.
    ``can_defer_absence`` is a separate boolean matrix produced by a backend
    that has calibrated an absolute semantic-plausibility decision.  A plain
    :class:`SemanticRetrievalBackend` never acquires that authority merely by
    returning a high relative score.

    Shape and value validation is intentionally performed by
    :func:`retrieve_scene_identities`, where the expected query and identity
    counts are known.
    """

    scores: Sequence[Sequence[float]]
    can_defer_absence: Sequence[Sequence[bool]]


class CalibratedSemanticRetrievalBackend(SemanticRetrievalBackend, Protocol):
    """Optional extension for a backend with a calibrated absolute gate.

    Implementations must make this decision independently of scene-local
    rank.  In particular, a threshold over Top-K position or RRF score is not
    an absolute plausibility calibration.
    """

    def score_with_absence_gate(
        self,
        queries: Sequence[str],
        identities: Sequence[str],
    ) -> CalibratedSemanticScoreBatch:
        """Return ranking scores and per-pair calibrated deferral decisions."""


class SemanticBackendError(RuntimeError):
    """The optional semantic backend returned no safe ranking."""


@dataclass(frozen=True, slots=True)
class EntityIdentityQuery(JsonSerializable):
    """One graph entity and its canonical name/aliases for identity retrieval."""

    query_id: str
    terms: tuple[str, ...]

    def __post_init__(self) -> None:
        query_id = str(self.query_id).strip()
        if not query_id:
            raise ValueError("query_id must be non-empty")
        normalized = tuple(
            dict.fromkeys(
                value
                for value in (normalize_identity_term(term) for term in self.terms)
                if value
            )
        )
        if not normalized:
            raise ValueError("terms must contain at least one semantic identity")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "terms", normalized)


@dataclass(frozen=True, slots=True)
class SceneIdentityDocument(JsonSerializable):
    """One normalized identity encoded once for all live actors sharing it."""

    identity_term: str
    actor_ids: tuple[str, ...]
    authoritative_actor_ids: tuple[str, ...]
    identity_sources: tuple[IdentitySource | str, ...]
    raw_values: tuple[str, ...] = ()
    locator_actor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        term = normalize_identity_term(self.identity_term)
        actor_ids = tuple(dict.fromkeys(str(value).strip() for value in self.actor_ids))
        authoritative = tuple(
            dict.fromkeys(str(value).strip() for value in self.authoritative_actor_ids)
        )
        locator_actor_ids = tuple(
            dict.fromkeys(str(value).strip() for value in self.locator_actor_ids)
        )
        sources = tuple(
            dict.fromkeys(
                IdentitySource.coerce(value) for value in self.identity_sources
            )
        )
        raw_values = tuple(
            dict.fromkeys(
                str(value).strip() for value in self.raw_values if str(value).strip()
            )
        )
        if not term:
            raise ValueError("identity_term must contain semantic text")
        if not actor_ids or any(not value for value in actor_ids):
            raise ValueError("actor_ids must contain at least one live actor id")
        if any(value not in actor_ids for value in authoritative):
            raise ValueError("authoritative_actor_ids must be a subset of actor_ids")
        if any(value not in actor_ids for value in locator_actor_ids):
            raise ValueError("locator_actor_ids must be a subset of actor_ids")
        if not sources:
            raise ValueError("identity_sources must be non-empty")
        object.__setattr__(self, "identity_term", term)
        object.__setattr__(self, "actor_ids", actor_ids)
        object.__setattr__(self, "authoritative_actor_ids", authoritative)
        object.__setattr__(self, "locator_actor_ids", locator_actor_ids)
        object.__setattr__(self, "identity_sources", sources)
        object.__setattr__(self, "raw_values", raw_values)


@dataclass(frozen=True, slots=True)
class ExactIdentityHit(JsonSerializable):
    """An exact authoritative identity eligible for the Stage 1 fast path."""

    query_id: str
    matched_query_term: str
    identity_term: str
    actor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        query_term = normalize_identity_term(self.matched_query_term)
        identity_term = normalize_identity_term(self.identity_term)
        actor_ids = tuple(dict.fromkeys(str(value).strip() for value in self.actor_ids))
        if not str(self.query_id).strip():
            raise ValueError("query_id must be non-empty")
        if not query_term or query_term != identity_term:
            raise ValueError("exact identity hits require equal normalized terms")
        if not actor_ids or any(not value for value in actor_ids):
            raise ValueError("actor_ids must be non-empty")
        object.__setattr__(self, "query_id", str(self.query_id).strip())
        object.__setattr__(self, "matched_query_term", query_term)
        object.__setattr__(self, "identity_term", identity_term)
        object.__setattr__(self, "actor_ids", actor_ids)


@dataclass(frozen=True, slots=True)
class LocatorCandidate(JsonSerializable):
    """High-recall actor location hint that can never itself form a verdict."""

    query_id: str
    matched_query_term: str
    identity_term: str
    actor_ids: tuple[str, ...]
    lexical_relation: LexicalRelation | str
    semantic_score: float | None = None
    rank: int = 1
    use: CandidateUse | str = CandidateUse.LOCATOR_ONLY
    identity_sources: tuple[IdentitySource | str, ...] = ()
    raw_values: tuple[str, ...] = ()
    locator_group_id: str | None = None
    locator_status: str | None = None
    locator_reason: str | None = None
    can_defer_absence: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        query_id = str(self.query_id).strip()
        query_term = normalize_identity_term(self.matched_query_term)
        identity_term = normalize_identity_term(self.identity_term)
        actor_ids = tuple(dict.fromkeys(str(value).strip() for value in self.actor_ids))
        relation = (
            self.lexical_relation
            if isinstance(self.lexical_relation, LexicalRelation)
            else LexicalRelation(str(self.lexical_relation).strip().casefold())
        )
        use = (
            self.use
            if isinstance(self.use, CandidateUse)
            else CandidateUse(str(self.use).strip().casefold())
        )
        if not query_id or not query_term or not identity_term:
            raise ValueError("locator candidate identity fields must be non-empty")
        if not actor_ids or any(not value for value in actor_ids):
            raise ValueError("actor_ids must be non-empty")
        if isinstance(self.rank, bool) or int(self.rank) != self.rank or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        semantic_score = self.semantic_score
        if semantic_score is not None:
            semantic_score = float(semantic_score)
            if not math.isfinite(semantic_score):
                raise ValueError("semantic_score must be finite or None")
        if use is not CandidateUse.LOCATOR_ONLY:
            raise ValueError("semantic/substring candidates must remain locator_only")
        # A token-boundary lexical relation is an independently inspectable
        # plausibility signal and always defers a closed-world absence.  A
        # semantic-only candidate defaults to false and can be enabled only by
        # the calibrated backend result consumed inside retrieval below.
        can_defer_absence = relation is not LexicalRelation.NONE
        sources = tuple(
            dict.fromkeys(IdentitySource.coerce(value) for value in self.identity_sources)
        )
        raw_values = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in self.raw_values
                if str(value).strip()
            )
        )
        locator_group_id = (
            str(self.locator_group_id).strip() if self.locator_group_id else None
        )
        locator_status = (
            str(self.locator_status).strip() if self.locator_status else None
        )
        locator_reason = (
            str(self.locator_reason).strip() if self.locator_reason else None
        )
        locator_fields = (locator_group_id, locator_status, locator_reason)
        if any(value is not None for value in locator_fields) and any(
            value is None for value in locator_fields
        ):
            raise ValueError(
                "LLM locator group id, status, and reason must be supplied together"
            )
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "matched_query_term", query_term)
        object.__setattr__(self, "identity_term", identity_term)
        object.__setattr__(self, "actor_ids", actor_ids)
        object.__setattr__(self, "lexical_relation", relation)
        object.__setattr__(self, "semantic_score", semantic_score)
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "use", use)
        object.__setattr__(self, "identity_sources", sources)
        object.__setattr__(self, "raw_values", raw_values)
        object.__setattr__(self, "locator_group_id", locator_group_id)
        object.__setattr__(self, "locator_status", locator_status)
        object.__setattr__(self, "locator_reason", locator_reason)
        object.__setattr__(self, "can_defer_absence", can_defer_absence)


@dataclass(frozen=True, slots=True)
class QueryIdentityRetrieval(JsonSerializable):
    """Fast-path and locator-only results for one graph entity."""

    query_id: str
    normalized_terms: tuple[str, ...]
    exact_matches: tuple[ExactIdentityHit, ...] = ()
    locator_candidates: tuple[LocatorCandidate, ...] = ()
    has_calibrated_absence_defer_candidate: bool = False

    def __post_init__(self) -> None:
        query_id = str(self.query_id).strip()
        terms = tuple(
            dict.fromkeys(
                normalize_identity_term(value) for value in self.normalized_terms
            )
        )
        if not query_id or not terms or any(not value for value in terms):
            raise ValueError("query result requires an id and normalized terms")
        if any(value.query_id != query_id for value in self.exact_matches):
            raise ValueError("exact match query_id does not match its result")
        if any(value.query_id != query_id for value in self.locator_candidates):
            raise ValueError("locator candidate query_id does not match its result")
        if not isinstance(self.has_calibrated_absence_defer_candidate, bool):
            raise TypeError(
                "has_calibrated_absence_defer_candidate must be a bool"
            )
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "normalized_terms", terms)
        object.__setattr__(self, "exact_matches", tuple(self.exact_matches))
        object.__setattr__(self, "locator_candidates", tuple(self.locator_candidates))

    @property
    def resolved_by_exact_identity(self) -> bool:
        return bool(self.exact_matches)

    @property
    def has_absence_defer_candidate(self) -> bool:
        """Whether a locator has enough plausibility to defer closed-world absence."""

        return self.has_calibrated_absence_defer_candidate or any(
            value.can_defer_absence for value in self.locator_candidates
        )


@dataclass(frozen=True, slots=True)
class SceneIdentityRetrievalResult(JsonSerializable):
    """One batch retrieval result over a de-duplicated live identity index."""

    queries: tuple[QueryIdentityRetrieval, ...]
    unique_identity_count: int
    eligible_actor_count: int
    semantic_backend: str | None = None
    semantic_backend_error: str | None = None
    identity_group_count: int = 0
    semantic_request_manifest: tuple[Mapping[str, object], ...] = ()
    semantic_raw_records: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        query_ids = tuple(value.query_id for value in self.queries)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query ids must be unique")
        for name in (
            "unique_identity_count",
            "eligible_actor_count",
            "identity_group_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "queries", tuple(self.queries))
        backend = str(self.semantic_backend).strip() if self.semantic_backend else None
        error = (
            str(self.semantic_backend_error).strip()
            if self.semantic_backend_error
            else None
        )
        object.__setattr__(self, "semantic_backend", backend)
        object.__setattr__(self, "semantic_backend_error", error)
        if any(not isinstance(value, Mapping) for value in self.semantic_request_manifest):
            raise TypeError("semantic_request_manifest must contain objects")
        if any(not isinstance(value, Mapping) for value in self.semantic_raw_records):
            raise TypeError("semantic_raw_records must contain objects")
        object.__setattr__(
            self,
            "semantic_request_manifest",
            tuple(dict(value) for value in self.semantic_request_manifest),
        )
        object.__setattr__(
            self,
            "semantic_raw_records",
            tuple(dict(value) for value in self.semantic_raw_records),
        )

    def for_query(self, query_id: str) -> QueryIdentityRetrieval | None:
        key = str(query_id).strip()
        return next((value for value in self.queries if value.query_id == key), None)


@dataclass(slots=True)
class _IdentityAccumulator:
    actor_ids: set[str]
    authoritative_actor_ids: set[str]
    locator_actor_ids: set[str]
    sources: set[IdentitySource]
    raw_values: set[str]


def _authoritative_terms(actor: ActorDescriptor) -> tuple[IdentityTerm, ...]:
    """Select the same highest-provenance fallback tier used by Stage 1."""

    terms = tuple(actor.identity_terms)
    if not terms:
        return ()
    source_priority = max(term.source_priority for term in terms)
    source_tier = tuple(
        term for term in terms if term.source_priority == source_priority
    )
    strength_priority = max(
        IdentityStrength.coerce(term.strength).priority for term in source_tier
    )
    return tuple(
        term
        for term in source_tier
        if IdentityStrength.coerce(term.strength).priority == strength_priority
    )


def build_scene_identity_documents(
    actors: Iterable[ActorDescriptor],
    *,
    scene_bounds: SceneBounds,
) -> tuple[tuple[SceneIdentityDocument, ...], int]:
    """Create one semantic document per unique eligible live identity.

    Context terms are intentionally excluded.  Lower-provenance identity terms
    and component locator terms remain useful for high-recall retrieval, but
    only actor ids attached to an authoritative whole-actor identity are
    exposed through the exact fast path.
    """

    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")
    accumulators: dict[str, _IdentityAccumulator] = {}
    eligible_actor_ids: set[str] = set()
    for actor in actors:
        if not isinstance(actor, ActorDescriptor):
            raise TypeError("actors must contain ActorDescriptor values")
        if not actor.eligible_for_stage1(scene_bounds):
            continue
        eligible_actor_ids.add(actor.live_actor_id)
        authoritative = {term.term for term in _authoritative_terms(actor)}
        for term in actor.identity_terms:
            accumulator = accumulators.setdefault(
                term.term,
                _IdentityAccumulator(set(), set(), set(), set(), set()),
            )
            accumulator.actor_ids.add(actor.live_actor_id)
            if term.term in authoritative:
                accumulator.authoritative_actor_ids.add(actor.live_actor_id)
            accumulator.sources.add(IdentitySource.coerce(term.source))
            if term.raw_value:
                accumulator.raw_values.add(term.raw_value)
        for term in actor.locator_terms:
            accumulator = accumulators.setdefault(
                term.term,
                _IdentityAccumulator(set(), set(), set(), set(), set()),
            )
            accumulator.actor_ids.add(actor.live_actor_id)
            accumulator.locator_actor_ids.add(actor.live_actor_id)
            accumulator.sources.add(IdentitySource.coerce(term.source))
            if term.raw_value:
                accumulator.raw_values.add(term.raw_value)

    documents = tuple(
        SceneIdentityDocument(
            identity_term=term,
            actor_ids=tuple(sorted(value.actor_ids)),
            authoritative_actor_ids=tuple(sorted(value.authoritative_actor_ids)),
            locator_actor_ids=tuple(sorted(value.locator_actor_ids)),
            identity_sources=tuple(
                sorted(
                    value.sources, key=lambda source: (-source.priority, source.value)
                )
            ),
            raw_values=tuple(sorted(value.raw_values)),
        )
        for term, value in sorted(accumulators.items())
    )
    return documents, len(eligible_actor_ids)


def token_boundary_relation(query_term: str, identity_term: str) -> LexicalRelation:
    """Return whether the normalized query is a contiguous identity token span.

    This is directional.  ``candle`` is a token substring of ``candle holder``;
    ``car`` is not a token substring of ``cart``.  A non-exact substring is only
    a ranking boost and never direct Stage 1 evidence.
    """

    query = tuple(normalize_identity_term(query_term).split())
    identity = tuple(normalize_identity_term(identity_term).split())
    if not query or not identity:
        return LexicalRelation.NONE
    if query == identity:
        return LexicalRelation.EXACT
    if len(query) >= len(identity):
        return LexicalRelation.NONE
    width = len(query)
    if any(
        identity[index : index + width] == query
        for index in range(len(identity) - width + 1)
    ):
        return LexicalRelation.QUERY_TOKEN_SUBSTRING
    return LexicalRelation.NONE


def locator_query_terms(terms: Iterable[str]) -> tuple[str, ...]:
    """Expand entity terms for camera localization, never for exact evidence."""

    selected: dict[str, None] = {}
    for value in terms:
        canonical = normalize_identity_term(value)
        if not canonical:
            continue
        selected.setdefault(canonical, None)
        for alias in _LOCATOR_QUERY_ALIASES.get(canonical, ()):
            normalized = normalize_identity_term(alias)
            if normalized:
                selected.setdefault(normalized, None)
        for token in canonical.split():
            if len(token) >= 2 and token not in _LOCATOR_QUERY_STOPWORDS:
                selected.setdefault(token, None)
    return tuple(selected)


def _validated_score_matrix(
    values: Sequence[Sequence[float]],
    *,
    query_count: int,
    identity_count: int,
) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(float(score) for score in row) for row in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SemanticBackendError("backend scores must be a numeric matrix") from exc
    if len(rows) != query_count or any(len(row) != identity_count for row in rows):
        raise SemanticBackendError(
            "backend score matrix shape does not match queries x identities"
        )
    if any(not math.isfinite(score) for row in rows for score in row):
        raise SemanticBackendError("backend score matrix contains non-finite values")
    return rows


def _validated_absence_gate_matrix(
    values: Sequence[Sequence[bool]],
    *,
    query_count: int,
    identity_count: int,
) -> tuple[tuple[bool, ...], ...]:
    try:
        rows = tuple(tuple(row) for row in values)
    except TypeError as exc:
        raise SemanticBackendError(
            "backend absence gate must be a boolean matrix"
        ) from exc
    if len(rows) != query_count or any(len(row) != identity_count for row in rows):
        raise SemanticBackendError(
            "backend absence gate matrix shape does not match queries x identities"
        )
    if any(not isinstance(value, bool) for row in rows for value in row):
        raise SemanticBackendError(
            "backend absence gate matrix must contain only booleans"
        )
    return rows


def retrieve_scene_identities(
    queries: Iterable[EntityIdentityQuery],
    actors: Iterable[ActorDescriptor],
    *,
    scene_bounds: SceneBounds,
    semantic_backend: SemanticRetrievalBackend | None = None,
    top_k: int = 10,
    include_resolved_locators: bool = False,
    excluded_actor_ids_by_query: Mapping[str, Iterable[str]] | None = None,
) -> SceneIdentityRetrievalResult:
    """Batch exact matching and locator-only retrieval over current live actors.

    By default, exact authoritative matches resolve a query without invoking the
    semantic backend, preserving the original fast-path API.  Callers that need
    additional acquisition candidates around an already-confirmed object can set
    ``include_resolved_locators=True``.  In that mode every query is scored in the
    same single backend call, while exact-hit actors and any caller-supplied
    ``excluded_actor_ids_by_query`` are removed from locator candidates.

    Token-boundary substring candidates sort ahead of ordinary semantic results.
    If the backend is absent or fails, those lexical candidates remain available
    while all semantic-only candidates disappear.  Every returned candidate is
    locator-only regardless of mode.  Lexical candidates may defer a closed-world
    absence; semantic-only candidates may do so only when the backend implements
    ``score_with_absence_gate`` and returns an explicit calibrated decision.
    """

    if isinstance(top_k, bool) or int(top_k) != top_k or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    top_k = int(top_k)
    query_values = tuple(queries)
    if any(not isinstance(value, EntityIdentityQuery) for value in query_values):
        raise TypeError("queries must contain EntityIdentityQuery values")
    query_ids = tuple(value.query_id for value in query_values)
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query ids must be unique")
    if not isinstance(include_resolved_locators, bool):
        raise TypeError("include_resolved_locators must be a bool")
    if excluded_actor_ids_by_query is not None and not isinstance(
        excluded_actor_ids_by_query, Mapping
    ):
        raise TypeError("excluded_actor_ids_by_query must be a mapping")
    supplied_exclusions = excluded_actor_ids_by_query or {}
    unknown_exclusion_queries = sorted(set(supplied_exclusions) - set(query_ids))
    if unknown_exclusion_queries:
        raise ValueError(
            "excluded actor ids contain unknown query ids: "
            f"{unknown_exclusion_queries!r}"
        )
    excluded_by_query = {
        query_id: frozenset(
            actor_id
            for actor_id in (
                str(value).strip()
                for value in supplied_exclusions.get(query_id, ())
            )
            if actor_id
        )
        for query_id in query_ids
    }

    documents, eligible_actor_count = build_scene_identity_documents(
        actors,
        scene_bounds=scene_bounds,
    )
    document_by_term = {value.identity_term: value for value in documents}

    exact_by_query: dict[str, tuple[ExactIdentityHit, ...]] = {}
    unresolved: list[EntityIdentityQuery] = []
    for query in query_values:
        exact_hits: list[ExactIdentityHit] = []
        for query_term in query.terms:
            document = document_by_term.get(query_term)
            if document is None or not document.authoritative_actor_ids:
                continue
            exact_hits.append(
                ExactIdentityHit(
                    query_id=query.query_id,
                    matched_query_term=query_term,
                    identity_term=document.identity_term,
                    actor_ids=document.authoritative_actor_ids,
                )
            )
        exact_by_query[query.query_id] = tuple(exact_hits)
        if not exact_hits:
            unresolved.append(query)

    locator_queries = query_values if include_resolved_locators else tuple(unresolved)
    unique_backend_queries = tuple(
        dict.fromkeys(term for query in locator_queries for term in query.terms)
    )
    backend_scores: dict[tuple[str, str], float] = {}
    backend_absence_gates: dict[tuple[str, str], bool] = {}
    backend_name: str | None = None
    backend_error: str | None = None
    if semantic_backend is not None:
        try:
            backend_name = (
                str(semantic_backend.name).strip() or type(semantic_backend).__name__
            )
        except Exception:  # noqa: BLE001  # pragma: no cover - diagnostic only
            backend_name = type(semantic_backend).__name__
    # A weak ActorLabel remains useful as a lexical camera hint, but it must not
    # enter semantic retrieval when a stronger asset/class/name term says
    # something else.  Restrict the model corpus to identities that are
    # authoritative for at least one live actor; this also substantially cuts
    # the number of texts encoded in large levels.
    semantic_documents = tuple(
        document
        for document in documents
        if document.authoritative_actor_ids or document.locator_actor_ids
    )
    if semantic_backend is not None and unique_backend_queries and semantic_documents:
        try:
            identities = tuple(
                value.identity_term for value in semantic_documents
            )
            calibrated_scorer = getattr(
                semantic_backend, "score_with_absence_gate", None
            )
            if callable(calibrated_scorer):
                scored = calibrated_scorer(unique_backend_queries, identities)
                if not isinstance(scored, CalibratedSemanticScoreBatch):
                    raise SemanticBackendError(
                        "score_with_absence_gate must return "
                        "CalibratedSemanticScoreBatch"
                    )
                raw_scores = scored.scores
                raw_absence_gates = scored.can_defer_absence
            else:
                raw_scores = semantic_backend.score(
                    unique_backend_queries,
                    identities,
                )
                raw_absence_gates = None
            matrix = _validated_score_matrix(
                raw_scores,
                query_count=len(unique_backend_queries),
                identity_count=len(semantic_documents),
            )
            backend_scores = {
                (query_term, document.identity_term): matrix[query_index][
                    document_index
                ]
                for query_index, query_term in enumerate(unique_backend_queries)
                for document_index, document in enumerate(semantic_documents)
            }
            if raw_absence_gates is not None:
                try:
                    gate_matrix = _validated_absence_gate_matrix(
                        raw_absence_gates,
                        query_count=len(unique_backend_queries),
                        identity_count=len(semantic_documents),
                    )
                    backend_absence_gates = {
                        (query_term, document.identity_term): gate_matrix[query_index][
                            document_index
                        ]
                        for query_index, query_term in enumerate(unique_backend_queries)
                        for document_index, document in enumerate(semantic_documents)
                    }
                except Exception as exc:  # noqa: BLE001 - gate degrades independently
                    backend_error = f"{type(exc).__name__}: {exc}"
                    backend_absence_gates = {}
        except Exception as exc:  # noqa: BLE001 - backend failure must fail closed
            backend_error = f"{type(exc).__name__}: {exc}"
            backend_scores = {}
            backend_absence_gates = {}

    results: list[QueryIdentityRetrieval] = []
    for query in query_values:
        exact_hits = exact_by_query[query.query_id]
        locator_candidates: tuple[LocatorCandidate, ...] = ()
        if not exact_hits or include_resolved_locators:
            excluded_actor_ids = set(excluded_by_query[query.query_id])
            if include_resolved_locators:
                for hit in exact_hits:
                    excluded_actor_ids.update(hit.actor_ids)
            ranked: list[
                tuple[
                    int,
                    float,
                    str,
                    str,
                    LexicalRelation,
                    float | None,
                    SceneIdentityDocument,
                    tuple[str, ...],
                    bool,
                ]
            ] = []
            for document in documents:
                options: list[
                    tuple[int, float, str, LexicalRelation, float | None, bool]
                ] = []
                for query_term in locator_query_terms(query.terms):
                    relation = token_boundary_relation(
                        query_term, document.identity_term
                    )
                    semantic_score = backend_scores.get(
                        (query_term, document.identity_term)
                    )
                    semantic_can_defer = backend_absence_gates.get(
                        (query_term, document.identity_term), False
                    )
                    if relation is LexicalRelation.NONE and semantic_score is None:
                        continue
                    options.append(
                        (
                            relation.priority,
                            semantic_score if semantic_score is not None else -math.inf,
                            query_term,
                            relation,
                            semantic_score,
                            semantic_can_defer,
                        )
                    )
                if not options:
                    continue
                _, _, matched_term, relation, semantic_score, _ = max(
                    options,
                    key=lambda value: (value[0], value[1], value[2]),
                )
                # An entity alias is another formulation of the same query.
                # A calibrated acceptance from any alias therefore applies to
                # this document, regardless of which alias supplied the best
                # ranking score or lexical relation.
                semantic_can_defer = any(value[5] for value in options)
                semantic_locator_actor_ids = tuple(
                    dict.fromkeys(
                        (
                            *document.authoritative_actor_ids,
                            *document.locator_actor_ids,
                        )
                    )
                )
                candidate_actor_ids = tuple(
                    actor_id
                    for actor_id in (
                        document.actor_ids
                        if relation is not LexicalRelation.NONE
                        else semantic_locator_actor_ids
                    )
                    if actor_id not in excluded_actor_ids
                )
                if not candidate_actor_ids:
                    continue
                ranked.append(
                    (
                        relation.priority,
                        semantic_score if semantic_score is not None else -math.inf,
                        document.identity_term,
                        matched_term,
                        relation,
                        semantic_score,
                        document,
                        candidate_actor_ids,
                        semantic_can_defer,
                    )
                )
            ranked.sort(key=lambda value: (-value[0], -value[1], value[2], value[3]))
            has_calibrated_absence_defer_candidate = any(
                value[8] for value in ranked
            )
            candidates: list[LocatorCandidate] = []
            for rank, (
                _,
                _,
                _,
                matched_term,
                relation,
                semantic_score,
                document,
                candidate_actor_ids,
                semantic_can_defer,
            ) in enumerate(ranked[:top_k], start=1):
                candidate = LocatorCandidate(
                    query_id=query.query_id,
                    matched_query_term=matched_term,
                    identity_term=document.identity_term,
                    # Semantic-only retrieval may point at authoritative
                    # whole-actor identities or explicit component locator
                    # terms.  It still excludes weak/non-authoritative labels.
                    # A token-boundary lexical hit may additionally retain
                    # those labels as a high-recall camera hint.
                    actor_ids=candidate_actor_ids,
                    lexical_relation=relation,
                    semantic_score=semantic_score,
                    rank=rank,
                    identity_sources=document.identity_sources,
                    raw_values=document.raw_values,
                )
                if relation is LexicalRelation.NONE and semantic_can_defer:
                    # ``can_defer_absence`` is intentionally ``init=False`` so
                    # arbitrary callers cannot promote a relative Top-K result.
                    # This is the sole promotion point, reached only from the
                    # validated structured result above.
                    object.__setattr__(candidate, "can_defer_absence", True)
                candidates.append(candidate)
            locator_candidates = tuple(candidates)
        results.append(
            QueryIdentityRetrieval(
                query_id=query.query_id,
                normalized_terms=query.terms,
                exact_matches=exact_hits,
                locator_candidates=locator_candidates,
                # Top-K is a Stage 2 capture/serialization budget, not an
                # absence-decision budget.  Preserve a calibrated plausible
                # pair even when its locator ranks below the returned slice.
                has_calibrated_absence_defer_candidate=(
                    has_calibrated_absence_defer_candidate
                    if not exact_hits or include_resolved_locators
                    else False
                ),
            )
        )

    return SceneIdentityRetrievalResult(
        queries=tuple(results),
        unique_identity_count=len(documents),
        eligible_actor_count=eligible_actor_count,
        semantic_backend=backend_name,
        semantic_backend_error=backend_error,
    )


def _dense_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise SemanticBackendError("dense embeddings have inconsistent dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _sparse_dot(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def _rank_positions(scores: Sequence[float]) -> tuple[int, ...]:
    """Return competition ranks without inventing order inside score ties.

    Sparse BM25 often produces an all-zero channel for open-vocabulary terms.
    Giving those equal scores index-based ranks would inject document-order bias
    into RRF and can cancel or even reorder the informative dense channel.
    Equal scores therefore share a rank (``1, 1, 3``), contributing the same
    constant to every tied document.
    """

    sorted_scores = sorted(set(scores), reverse=True)
    rank_by_score: dict[float, int] = {}
    position = 1
    for score in sorted_scores:
        rank_by_score[score] = position
        position += sum(value == score for value in scores)
    return tuple(rank_by_score[score] for score in scores)


@dataclass(frozen=True, slots=True)
class LocalEmbeddingServiceBackend:
    """Optional adapter for the repository's loopback BGE/BM25 embed service.

    The service embeds all unique query and live-identity texts in one request.
    Dense cosine and sparse dot-product rankings are fused with reciprocal rank
    fusion.  The endpoint is restricted to loopback hosts so selecting this
    backend cannot silently turn Stage 1 into an external API dependency.
    """

    endpoint: str = "http://127.0.0.1:7777/embed"
    timeout_seconds: float = 30.0
    rrf_constant: int = 60

    def __post_init__(self) -> None:
        endpoint = str(self.endpoint).strip()
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("embedding endpoint must be an HTTP(S) loopback URL")
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite")
        if (
            isinstance(self.rrf_constant, bool)
            or int(self.rrf_constant) != self.rrf_constant
        ):
            raise ValueError("rrf_constant must be a positive integer")
        if self.rrf_constant < 1:
            raise ValueError("rrf_constant must be a positive integer")
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "rrf_constant", int(self.rrf_constant))

    @property
    def name(self) -> str:
        return "local_bge_bm25_rrf"

    def _embed(
        self, texts: Sequence[str]
    ) -> tuple[
        tuple[tuple[float, ...], ...],
        tuple[dict[int, float], ...] | None,
    ]:
        request = Request(
            self.endpoint,
            data=json.dumps({"texts": list(texts)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise SemanticBackendError(
                f"local embed service request failed: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise SemanticBackendError(
                "local embed service returned a non-object payload"
            )

        raw_dense = payload.get("dense")
        if not isinstance(raw_dense, Sequence) or isinstance(raw_dense, (str, bytes)):
            raise SemanticBackendError("local embed service returned no dense vectors")
        try:
            dense = tuple(
                tuple(float(value) for value in row)
                for row in raw_dense
                if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise SemanticBackendError("dense embeddings must be numeric") from exc
        if len(dense) != len(texts) or any(not row for row in dense):
            raise SemanticBackendError(
                "dense embedding count does not match input texts"
            )
        dimension = len(dense[0])
        if any(len(row) != dimension for row in dense):
            raise SemanticBackendError("dense embeddings have inconsistent dimensions")
        if any(not math.isfinite(value) for row in dense for value in row):
            raise SemanticBackendError("dense embeddings contain non-finite values")

        raw_sparse = payload.get("sparse")
        if raw_sparse is None:
            return dense, None
        if not isinstance(raw_sparse, Sequence) or isinstance(raw_sparse, (str, bytes)):
            raise SemanticBackendError("sparse embeddings must be an array")
        sparse: list[dict[int, float]] = []
        for row in raw_sparse:
            if not isinstance(row, Mapping):
                raise SemanticBackendError("sparse embedding rows must be objects")
            indices = row.get("indices")
            values = row.get("values")
            if (
                not isinstance(indices, Sequence)
                or isinstance(indices, (str, bytes))
                or not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
                or len(indices) != len(values)
            ):
                raise SemanticBackendError("invalid sparse indices/values")
            converted: dict[int, float] = {}
            for raw_index, raw_value in zip(indices, values, strict=True):
                if isinstance(raw_index, bool) or int(raw_index) != raw_index:
                    raise SemanticBackendError("sparse indices must be integers")
                index = int(raw_index)
                value = float(raw_value)
                if index < 0 or not math.isfinite(value):
                    raise SemanticBackendError("invalid sparse embedding value")
                converted[index] = value
            sparse.append(converted)
        if len(sparse) != len(texts):
            raise SemanticBackendError(
                "sparse embedding count does not match input texts"
            )
        return dense, tuple(sparse)

    def score(
        self,
        queries: Sequence[str],
        identities: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        if not queries:
            return ()
        if not identities:
            return tuple(() for _ in queries)
        unique_texts = tuple(dict.fromkeys((*queries, *identities)))
        dense, sparse = self._embed(unique_texts)
        text_index = {value: index for index, value in enumerate(unique_texts)}
        result: list[tuple[float, ...]] = []
        for query in queries:
            query_index = text_index[query]
            dense_scores = tuple(
                _dense_cosine(dense[query_index], dense[text_index[identity]])
                for identity in identities
            )
            dense_ranks = _rank_positions(dense_scores)
            channels = [dense_ranks]
            if sparse is not None:
                sparse_scores = tuple(
                    _sparse_dot(sparse[query_index], sparse[text_index[identity]])
                    for identity in identities
                )
                channels.append(_rank_positions(sparse_scores))
            # Normalize each RRF channel so rank 1 contributes 1.0.  Absolute
            # values are diagnostic only; ordering is what the caller uses.
            scores = tuple(
                sum(
                    (self.rrf_constant + 1) / (self.rrf_constant + ranks[index])
                    for ranks in channels
                )
                / len(channels)
                for index in range(len(identities))
            )
            result.append(scores)
        return tuple(result)


__all__ = [
    "CalibratedSemanticRetrievalBackend",
    "CalibratedSemanticScoreBatch",
    "CandidateUse",
    "EntityIdentityQuery",
    "ExactIdentityHit",
    "LexicalRelation",
    "LocalEmbeddingServiceBackend",
    "LocatorCandidate",
    "QueryIdentityRetrieval",
    "SceneIdentityDocument",
    "SceneIdentityRetrievalResult",
    "SemanticBackendError",
    "SemanticRetrievalBackend",
    "build_scene_identity_documents",
    "locator_query_terms",
    "retrieve_scene_identities",
    "token_boundary_relation",
]
