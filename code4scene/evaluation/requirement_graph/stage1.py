"""Stage 1 object grounding and safe existence evaluation from UE metadata.

Stage 1 is deliberately metadata-only.  It compares frozen prompt names and
aliases with the basename of an Actor asset path, its Unreal name, and its
Actor label.  Exact canonical-name matches may decide plain existence.  Alias
or token-boundary matches are retained only as audited *grounding* for targeted
Stage 2 acquisition; they never prove visual modifiers or a complete claim.

The matcher is lexical and deterministic.  It performs no embedding search,
edit-distance match, synonym expansion, or arbitrary substring matching.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum

from .actor_inventory import (
    ActorDescriptor,
    ActorInventorySnapshot,
    AssemblyDescriptor,
    IdentitySource,
    IdentityStrength,
    IdentityTerm,
    InventoryStatus,
    normalize_identity_term,
)
from .contracts import (
    ClaimVerdict,
    EntityEvaluationRoute,
    EntityInventoryRepresentation,
    EntityNode,
    EntityType,
    JsonSerializable,
    Polarity,
    PredicateNode,
    PredicateType,
    RequirementGraph,
    SceneBounds,
)

UE_ASSET_INVENTORY_EVIDENCE = "ue_asset_inventory"
UE_ASSET_INVENTORY_CLOSED_WORLD_EVIDENCE = "ue_asset_inventory_closed_world"
UE_DECLARED_ASSEMBLY_EVIDENCE = "ue_declared_assembly"


def _canonical_category_text(value: object) -> str:
    """Keep a diagnostic category even when UE identity cleanup deems it generic."""

    normalized = normalize_identity_term(value)
    if normalized:
        return normalized
    return " ".join(str(value).strip().casefold().split())


class Stage1AssessmentKind(str, Enum):
    """Why an atomic assessment exists in the Stage 1 result."""

    ENTITY_REQUIREMENT = "entity_requirement"
    ENTITY_DEPENDENCY = "entity_dependency"
    EXISTENCE_PREDICATE = "existence_predicate"
    STAGE2_VISUAL_REQUIREMENT = "stage2_visual_requirement"
    STAGE2_VISUAL_DEPENDENCY = "stage2_visual_dependency"
    STAGE2_VISUAL_EXISTENCE_PREDICATE = "stage2_visual_existence_predicate"

    @classmethod
    def coerce(cls, value: Stage1AssessmentKind | str) -> Stage1AssessmentKind:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().casefold())


class IdentityQuerySource(str, Enum):
    """Frozen graph field that supplied a Stage 1 identity query."""

    CANONICAL_NAME = "canonical_name"
    FROZEN_ALIAS = "frozen_alias"

    @classmethod
    def coerce(cls, value: IdentityQuerySource | str) -> IdentityQuerySource:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().casefold())


class IdentityMatchRule(str, Enum):
    """Auditable lexical rule used to ground one Actor."""

    NORMALIZED_EXACT = "normalized_exact"
    TOKEN_BOUNDARY = "token_boundary"

    @classmethod
    def coerce(cls, value: IdentityMatchRule | str) -> IdentityMatchRule:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().casefold())


_ENTITY_ASSESSMENT_KINDS = frozenset(
    {
        Stage1AssessmentKind.ENTITY_REQUIREMENT,
        Stage1AssessmentKind.ENTITY_DEPENDENCY,
        Stage1AssessmentKind.STAGE2_VISUAL_REQUIREMENT,
        Stage1AssessmentKind.STAGE2_VISUAL_DEPENDENCY,
    }
)
_DEPENDENCY_ASSESSMENT_KINDS = frozenset(
    {
        Stage1AssessmentKind.ENTITY_DEPENDENCY,
        Stage1AssessmentKind.STAGE2_VISUAL_DEPENDENCY,
    }
)
_STAGE2_VISUAL_ASSESSMENT_KINDS = frozenset(
    {
        Stage1AssessmentKind.STAGE2_VISUAL_REQUIREMENT,
        Stage1AssessmentKind.STAGE2_VISUAL_DEPENDENCY,
        Stage1AssessmentKind.STAGE2_VISUAL_EXISTENCE_PREDICATE,
    }
)


@dataclass(frozen=True, slots=True)
class ActorIdentityMatch(JsonSerializable):
    """One deduplicated, audited lexical Actor grounding observation."""

    actor_id: str
    query_term: str
    identity_term: str
    identity_source: IdentitySource | str
    identity_strength: IdentityStrength | str
    raw_value: str = ""
    query_source: IdentityQuerySource | str = IdentityQuerySource.CANONICAL_NAME
    match_rule: IdentityMatchRule | str = IdentityMatchRule.NORMALIZED_EXACT
    raw_query_term: str = ""
    normalized_query_term: str = ""
    normalized_identity_term: str = ""
    decision_eligible: bool = True

    def __post_init__(self) -> None:
        actor_id = str(self.actor_id).strip()
        query_term = normalize_identity_term(self.query_term)
        identity_term = normalize_identity_term(self.identity_term)
        if not actor_id:
            raise ValueError("actor_id must be non-empty")
        if not query_term or not identity_term:
            raise ValueError("identity matches must contain semantic terms")
        object.__setattr__(self, "actor_id", actor_id)
        object.__setattr__(self, "query_term", query_term)
        object.__setattr__(self, "identity_term", identity_term)
        object.__setattr__(
            self, "identity_source", IdentitySource.coerce(self.identity_source)
        )
        object.__setattr__(
            self,
            "identity_strength",
            IdentityStrength.coerce(self.identity_strength),
        )
        object.__setattr__(self, "raw_value", str(self.raw_value).strip())
        object.__setattr__(
            self,
            "query_source",
            IdentityQuerySource.coerce(self.query_source),
        )
        object.__setattr__(
            self,
            "match_rule",
            IdentityMatchRule.coerce(self.match_rule),
        )
        raw_query = str(self.raw_query_term).strip() or query_term
        normalized_query = _matching_identity_term(
            self.normalized_query_term or query_term
        )
        normalized_identity = _matching_identity_term(
            self.normalized_identity_term or identity_term,
            actor_side=True,
        )
        if not normalized_query or not normalized_identity:
            raise ValueError("identity matches require normalized audit terms")
        if not isinstance(self.decision_eligible, bool):
            raise TypeError("decision_eligible must be a bool")
        if self.decision_eligible and (
            self.query_source is not IdentityQuerySource.CANONICAL_NAME
            or self.match_rule is not IdentityMatchRule.NORMALIZED_EXACT
        ):
            raise ValueError(
                "only normalized-exact canonical-name matches may decide Stage 1"
            )
        object.__setattr__(self, "raw_query_term", raw_query)
        object.__setattr__(self, "normalized_query_term", normalized_query)
        object.__setattr__(self, "normalized_identity_term", normalized_identity)

    @property
    def source(self) -> IdentitySource:
        """Short compatibility alias for the identity provenance."""

        return IdentitySource.coerce(self.identity_source)


@dataclass(frozen=True, slots=True)
class AssemblyIdentityMatch(JsonSerializable):
    """The authoritative declaration that matched one logical assembly."""

    assembly_id: str
    query_term: str
    identity_term: str
    identity_source: IdentitySource | str
    identity_strength: IdentityStrength | str
    declaration_source: str
    member_actor_ids: tuple[str, ...]
    raw_value: str = ""

    def __post_init__(self) -> None:
        assembly_id = str(self.assembly_id).strip()
        query_term = normalize_identity_term(self.query_term)
        identity_term = normalize_identity_term(self.identity_term)
        declaration_source = str(self.declaration_source).strip()
        member_ids = tuple(str(value).strip() for value in self.member_actor_ids)
        if not assembly_id:
            raise ValueError("assembly_id must be non-empty")
        if not query_term or not identity_term:
            raise ValueError("assembly matches must contain semantic terms")
        if not declaration_source:
            raise ValueError("assembly matches require declaration_source provenance")
        if not member_ids or any(not value for value in member_ids):
            raise ValueError("assembly matches require member_actor_ids")
        object.__setattr__(self, "assembly_id", assembly_id)
        object.__setattr__(self, "query_term", query_term)
        object.__setattr__(self, "identity_term", identity_term)
        object.__setattr__(
            self, "identity_source", IdentitySource.coerce(self.identity_source)
        )
        object.__setattr__(
            self,
            "identity_strength",
            IdentityStrength.coerce(self.identity_strength),
        )
        object.__setattr__(self, "declaration_source", declaration_source)
        object.__setattr__(self, "member_actor_ids", member_ids)
        object.__setattr__(self, "raw_value", str(self.raw_value).strip())

    @property
    def source(self) -> IdentitySource:
        return IdentitySource.coerce(self.identity_source)


@dataclass(frozen=True, slots=True)
class AtomicEntityAssessment(JsonSerializable):
    """One Stage 1 verdict, separate from RGB/VLM claim assessments."""

    assessment_id: str
    node_id: str
    entity_id: str
    kind: Stage1AssessmentKind | str
    polarity: Polarity | str
    canonical_category: str
    verdict: ClaimVerdict | str
    evidence_type: str | None = None
    matches: tuple[ActorIdentityMatch, ...] = ()
    assembly_matches: tuple[AssemblyIdentityMatch, ...] = ()
    inventory_source: str = ""
    rationale: str = ""
    predicate_id: str | None = None
    dependency_only: bool = False
    # Extension field kept last for positional compatibility with the original
    # Stage 1 assessment contract.
    grounding_matches: tuple[ActorIdentityMatch, ...] = ()

    def __post_init__(self) -> None:
        for name in ("assessment_id", "node_id", "entity_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        category = _canonical_category_text(self.canonical_category)
        if not category:
            raise ValueError("canonical_category must contain semantic text")
        object.__setattr__(self, "canonical_category", category)
        object.__setattr__(self, "kind", Stage1AssessmentKind.coerce(self.kind))
        object.__setattr__(self, "polarity", _coerce_polarity(self.polarity))
        object.__setattr__(self, "verdict", ClaimVerdict.coerce(self.verdict))
        object.__setattr__(self, "matches", tuple(self.matches))
        if any(not isinstance(match, ActorIdentityMatch) for match in self.matches):
            raise TypeError("matches must contain ActorIdentityMatch values")
        grounding_matches = tuple(self.grounding_matches) or tuple(self.matches)
        object.__setattr__(self, "grounding_matches", grounding_matches)
        if any(
            not isinstance(match, ActorIdentityMatch)
            for match in self.grounding_matches
        ):
            raise TypeError(
                "grounding_matches must contain ActorIdentityMatch values"
            )
        grounding_by_actor = {
            match.actor_id.casefold(): match for match in self.grounding_matches
        }
        if len(grounding_by_actor) != len(self.grounding_matches):
            raise ValueError("grounding_matches must contain unique actor ids")
        if any(
            match.actor_id.casefold() not in grounding_by_actor
            for match in self.matches
        ):
            raise ValueError("decision matches must be included in grounding_matches")
        if any(not match.decision_eligible for match in self.matches):
            raise ValueError("matches must contain only decision-eligible evidence")
        object.__setattr__(self, "assembly_matches", tuple(self.assembly_matches))
        if any(
            not isinstance(match, AssemblyIdentityMatch)
            for match in self.assembly_matches
        ):
            raise TypeError(
                "assembly_matches must contain AssemblyIdentityMatch values"
            )
        if self.matches and self.assembly_matches:
            raise ValueError("an assessment cannot mix actor and assembly matches")
        object.__setattr__(
            self,
            "evidence_type",
            str(self.evidence_type).strip() if self.evidence_type else None,
        )
        object.__setattr__(self, "inventory_source", str(self.inventory_source).strip())
        object.__setattr__(self, "rationale", str(self.rationale).strip())
        predicate_id = str(self.predicate_id).strip() if self.predicate_id else None
        object.__setattr__(self, "predicate_id", predicate_id)
        if self.kind in _ENTITY_ASSESSMENT_KINDS:
            if predicate_id is not None:
                raise ValueError("entity assessments cannot name a predicate")
            if self.polarity is not Polarity.AFFIRMATIVE:
                raise ValueError("entity assessments are always affirmative")
            object.__setattr__(
                self,
                "dependency_only",
                self.kind in _DEPENDENCY_ASSESSMENT_KINDS,
            )
        elif predicate_id != self.node_id:
            raise ValueError(
                "existence predicate assessment must identify its predicate node"
            )
        if self.kind in _STAGE2_VISUAL_ASSESSMENT_KINDS:
            # ``STAGE2_VISUAL`` is a fallback route, not a command to ignore
            # exact trusted metadata. An exact asset/class/name or declared
            # assembly identity can close the identity question without RGB.
            # Only the unresolved form must stay evidence-free so Stage 2 can
            # search for a visually equivalent substitute.
            if self.verdict is ClaimVerdict.UNKNOWN and (
                self.evidence_type is not None
                or self.matches
                or self.assembly_matches
            ):
                raise ValueError(
                    "unresolved Stage-2-fallback assessments cannot contain "
                    "Stage 1 evidence"
                )
        if self.matches and self.evidence_type != UE_ASSET_INVENTORY_EVIDENCE:
            raise ValueError("actor matches require ue_asset_inventory evidence")
        if (
            self.assembly_matches
            and self.evidence_type != UE_DECLARED_ASSEMBLY_EVIDENCE
        ):
            raise ValueError("assembly matches require ue_declared_assembly evidence")

    @property
    def matched_actor_ids(self) -> tuple[str, ...]:
        return tuple(match.actor_id for match in self.matches)

    @property
    def grounded_actor_ids(self) -> tuple[str, ...]:
        """All name-grounded Actors, including non-decisive locator evidence."""

        return tuple(match.actor_id for match in self.grounding_matches)

    @property
    def actor_ids(self) -> tuple[str, ...]:
        """Compatibility alias used by evidence consumers."""

        return self.matched_actor_ids

    @property
    def matched_assembly_ids(self) -> tuple[str, ...]:
        return tuple(match.assembly_id for match in self.assembly_matches)

    @property
    def identity_sources(self) -> tuple[IdentitySource, ...]:
        return tuple(match.source for match in self.matches)

    @property
    def stage1_decision_applicable(self) -> bool:
        """Whether inventory metadata is permitted to decide this assessment."""

        return (
            self.kind not in _STAGE2_VISUAL_ASSESSMENT_KINDS
            or self.verdict is not ClaimVerdict.UNKNOWN
        )

    @property
    def routed_to_stage2(self) -> bool:
        return not self.stage1_decision_applicable


@dataclass(frozen=True, slots=True)
class Stage1Result(JsonSerializable):
    """Deterministically ordered metadata-only Stage 1 output."""

    assessments: tuple[AtomicEntityAssessment, ...]
    inventory_status: InventoryStatus | str
    inventory_source: str = ""
    eligible_actor_ids: tuple[str, ...] = ()
    eligible_assembly_ids: tuple[str, ...] = ()
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        assessments = tuple(self.assessments)
        if any(not isinstance(value, AtomicEntityAssessment) for value in assessments):
            raise TypeError("assessments must contain AtomicEntityAssessment values")
        ids = [assessment.assessment_id for assessment in assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("Stage 1 assessment ids must be unique")
        actor_ids = tuple(str(value).strip() for value in self.eligible_actor_ids)
        if any(not value for value in actor_ids):
            raise ValueError("eligible actor ids must be non-empty")
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(
            self, "inventory_status", InventoryStatus.coerce(self.inventory_status)
        )
        object.__setattr__(self, "inventory_source", str(self.inventory_source).strip())
        object.__setattr__(self, "eligible_actor_ids", actor_ids)
        assembly_ids = tuple(str(value).strip() for value in self.eligible_assembly_ids)
        if any(not value for value in assembly_ids):
            raise ValueError("eligible assembly ids must be non-empty")
        object.__setattr__(self, "eligible_assembly_ids", assembly_ids)
        schema_version = str(self.schema_version).strip()
        if schema_version != "1.0":
            raise ValueError("unsupported Stage 1 schema version")
        object.__setattr__(self, "schema_version", schema_version)

    def for_entity(self, entity_id: str) -> AtomicEntityAssessment | None:
        key = str(entity_id).strip()
        return next(
            (
                value
                for value in self.assessments
                if value.kind in _ENTITY_ASSESSMENT_KINDS
                and value.entity_id == key
            ),
            None,
        )

    def for_predicate(self, predicate_id: str) -> AtomicEntityAssessment | None:
        key = str(predicate_id).strip()
        return next(
            (value for value in self.assessments if value.predicate_id == key),
            None,
        )

    def for_node(self, node_id: str) -> AtomicEntityAssessment | None:
        key = str(node_id).strip()
        return next((value for value in self.assessments if value.node_id == key), None)

    def __iter__(self) -> Iterator[AtomicEntityAssessment]:
        return iter(self.assessments)

    def __len__(self) -> int:
        return len(self.assessments)


@dataclass(frozen=True, slots=True)
class _EntityFinding:
    entity: EntityNode
    canonical_category: str
    matches: tuple[ActorIdentityMatch, ...]
    grounding_matches: tuple[ActorIdentityMatch, ...]
    assembly_matches: tuple[AssemblyIdentityMatch, ...]
    uncertain_actor_ids: tuple[str, ...]
    uncertain_assembly_ids: tuple[str, ...]
    category_closed: bool


def _coerce_polarity(value: Polarity | str) -> Polarity:
    if isinstance(value, Polarity):
        return value
    normalized = str(value).strip().casefold()
    aliases = {
        "positive": Polarity.AFFIRMATIVE,
        "affirmative": Polarity.AFFIRMATIVE,
        "negative": Polarity.NEGATED,
        "negated": Polarity.NEGATED,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported polarity {value!r}") from exc


_ACTOR_TECHNICAL_PREFIXES = frozenset(
    {
        "bp",
        "deco",
        "env",
        "mesh",
        "prop",
        "ruin",
        "sk",
        "sm",
    }
)


def _matching_identity_term(value: object, *, actor_side: bool = False) -> str:
    """Normalize one name for deterministic token-boundary comparison.

    Actor-side UE wrapper prefixes and variant suffixes are removed only at
    the identity boundary.  Prompt terms are never modifier-stripped.  The
    resulting tokens remain lexical: no synonym or fuzzy expansion occurs.
    """

    tokens = normalize_identity_term(value).split()
    if actor_side:
        while len(tokens) > 1 and tokens[0] in _ACTOR_TECHNICAL_PREFIXES:
            tokens.pop(0)
        while len(tokens) > 1 and (
            tokens[-1].isdigit()
            or (len(tokens[-1]) == 1 and tokens[-1].isalpha())
        ):
            tokens.pop()
    return " ".join(tokens)


@dataclass(frozen=True, slots=True)
class _IdentityQuery:
    raw_term: str
    normalized_term: str
    source: IdentityQuerySource


def _query_terms(entity: EntityNode) -> tuple[_IdentityQuery, ...]:
    """Return the canonical name plus frozen aliases with provenance."""

    selected: dict[str, _IdentityQuery] = {}
    for raw, source in (
        (entity.name, IdentityQuerySource.CANONICAL_NAME),
        *((value, IdentityQuerySource.FROZEN_ALIAS) for value in entity.aliases),
    ):
        normalized = _matching_identity_term(raw)
        if not normalized:
            continue
        # The canonical name is inserted first and wins when singularization
        # makes an alias lexically identical (for example cliffs -> cliff).
        selected.setdefault(
            normalized,
            _IdentityQuery(str(raw).strip(), normalized, source),
        )
    return tuple(selected.values())


def _term_match_rule(
    query: _IdentityQuery,
    identity_term: IdentityTerm,
) -> tuple[IdentityMatchRule, str] | None:
    identity = _matching_identity_term(identity_term.term, actor_side=True)
    if not identity:
        return None
    if query.normalized_term == identity:
        return IdentityMatchRule.NORMALIZED_EXACT, identity
    query_tokens = query.normalized_term.split()
    identity_tokens = identity.split()
    if len(query_tokens) > len(identity_tokens):
        return None
    for start in range(len(identity_tokens) - len(query_tokens) + 1):
        if identity_tokens[start : start + len(query_tokens)] == query_tokens:
            return IdentityMatchRule.TOKEN_BOUNDARY, identity
    return None


def _actor_name_terms(actor: ActorDescriptor) -> tuple[IdentityTerm, ...]:
    """Expose only the approved Actor identity fields to name grounding.

    ``ACTOR_CLASS`` remains accepted for normalized-exact canonical matches to
    preserve the pre-existing Stage 1 contract.  It cannot create a
    token-boundary or alias grounding match.
    """

    approved = {
        IdentitySource.ASSET_PATH,
        IdentitySource.ACTOR_LABEL,
        IdentitySource.UNREAL_NAME,
        IdentitySource.ACTOR_CLASS,
        IdentitySource.STRUCTURED_TAG,
    }
    values = [
        term
        for term in actor.identity_terms
        if IdentitySource.coerce(term.source) in approved
    ]
    # The scene-evidence adapter historically moved root asset/label names into
    # ``locator_terms`` whenever a catalog category existed.  Compare only the
    # locator observations that exactly correspond to the Actor's own approved
    # fields; component/path-context locators remain excluded.
    approved_raw_values = {
        IdentitySource.ASSET_PATH: actor.asset_path,
        IdentitySource.ACTOR_LABEL: actor.actor_label,
        IdentitySource.UNREAL_NAME: actor.unreal_name,
    }
    for term in actor.locator_terms:
        source = IdentitySource.coerce(term.source)
        raw_value = approved_raw_values.get(source)
        if raw_value and str(term.raw_value).strip() == str(raw_value).strip():
            values.append(term)
    selected: dict[tuple[str, IdentitySource, str], IdentityTerm] = {}
    for term in values:
        key = (
            term.term,
            IdentitySource.coerce(term.source),
            str(term.raw_value).strip(),
        )
        selected.setdefault(key, term)
    return tuple(selected.values())


def _actor_name_match(
    actor: ActorDescriptor,
    query_terms: Iterable[_IdentityQuery],
) -> ActorIdentityMatch | None:
    candidates: list[
        tuple[tuple[int, int, int, int, int, str, str], ActorIdentityMatch]
    ] = []
    for term in _actor_name_terms(actor):
        for query in query_terms:
            matched = _term_match_rule(query, term)
            if matched is None:
                continue
            rule, normalized_identity = matched
            source = IdentitySource.coerce(term.source)
            if source in {
                IdentitySource.ACTOR_CLASS,
                IdentitySource.STRUCTURED_TAG,
            } and (
                query.source is not IdentityQuerySource.CANONICAL_NAME
                or rule is not IdentityMatchRule.NORMALIZED_EXACT
            ):
                continue
            decision_eligible = (
                query.source is IdentityQuerySource.CANONICAL_NAME
                and rule is IdentityMatchRule.NORMALIZED_EXACT
            )
            match = ActorIdentityMatch(
                actor_id=actor.live_actor_id,
                query_term=query.normalized_term,
                identity_term=term.term,
                identity_source=term.source,
                identity_strength=term.strength,
                raw_value=term.raw_value,
                query_source=query.source,
                match_rule=rule,
                raw_query_term=query.raw_term,
                normalized_query_term=query.normalized_term,
                normalized_identity_term=normalized_identity,
                decision_eligible=decision_eligible,
            )
            candidates.append(
                (
                    (
                        0 if decision_eligible else 1,
                        0
                        if query.source is IdentityQuerySource.CANONICAL_NAME
                        else 1,
                        0 if rule is IdentityMatchRule.NORMALIZED_EXACT else 1,
                        -len(query.normalized_term.split()),
                        -term.source_priority,
                        query.normalized_term,
                        normalized_identity,
                    ),
                    match,
                )
            )
    if not candidates:
        return None
    return min(candidates, key=lambda value: value[0])[1]


def _exact_assembly_identity(
    assembly: AssemblyDescriptor,
    query_terms: Iterable[_IdentityQuery],
) -> tuple[str, IdentityTerm] | None:
    """Match only an assembly declaration, before checking live eligibility."""

    terms = tuple(assembly.identity_terms)
    if not terms:
        return None
    priority = max(term.source_priority for term in terms)
    authoritative = tuple(term for term in terms if term.source_priority == priority)
    strength = max(
        IdentityStrength.coerce(term.strength).priority for term in authoritative
    )
    authoritative = tuple(
        term
        for term in authoritative
        if IdentityStrength.coerce(term.strength).priority == strength
    )
    candidates: list[tuple[int, int, str, str, IdentityTerm]] = []
    for term in authoritative:
        for query in query_terms:
            if (
                query.source is IdentityQuerySource.CANONICAL_NAME
                and query.normalized_term
                == _matching_identity_term(term.term, actor_side=True)
            ):
                candidates.append(
                    (
                        0,
                        -len(query.normalized_term.split()),
                        query.normalized_term,
                        term.term,
                        term,
                    )
                )
    if not candidates:
        return None
    _, _, query, _, term = min(candidates, key=lambda value: value[:4])
    return query, term


def _assembly_identity_match(
    assembly: AssemblyDescriptor,
    identity: tuple[str, IdentityTerm],
) -> AssemblyIdentityMatch:
    """Materialize public evidence only for an eligible resolved assembly."""

    query, term = identity
    return AssemblyIdentityMatch(
        assembly_id=assembly.assembly_id,
        query_term=query,
        identity_term=term.term,
        identity_source=term.source,
        identity_strength=term.strength,
        declaration_source=assembly.declaration_source,
        member_actor_ids=assembly.member_actor_ids,
        raw_value=term.raw_value,
    )


def _eligibility_is_uncertain(
    actor: ActorDescriptor, scene_bounds: SceneBounds
) -> bool:
    """Whether this actor could be eligible but lacks an explicit status/bounds."""

    statuses = (actor.in_current_level, actor.active, actor.renderable)
    if any(value is False for value in statuses):
        return False
    if actor.bounds is not None and not actor.bounds.intersects(scene_bounds):
        return False
    return any(value is None for value in statuses) or actor.bounds is None


def _assembly_eligibility_is_uncertain(
    assembly: AssemblyDescriptor,
    scene_bounds: SceneBounds,
    actors: Iterable[ActorDescriptor],
) -> bool:
    """Whether a matching declaration is plausible but not Stage-1 complete."""

    statuses = (
        assembly.in_current_level,
        assembly.active,
        assembly.renderable,
    )
    if any(value is False for value in statuses):
        return False
    if assembly.bounds is not None and not assembly.bounds.intersects(scene_bounds):
        return False
    if not assembly.identity_declared or not assembly.membership_complete:
        return True
    actor_values = tuple(actors)
    resolved = assembly.resolved_members(actor_values)
    if resolved is None:
        return True
    if not any(actor.eligible_for_stage1(scene_bounds) for actor in resolved):
        return any(_eligibility_is_uncertain(actor, scene_bounds) for actor in resolved)
    return any(value is None for value in statuses) or assembly.bounds is None


def _find_entity(
    entity: EntityNode,
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    eligible_actor_ids: frozenset[str] | None = None,
) -> _EntityFinding:
    canonical = _canonical_category_text(entity.name)
    queries = _query_terms(entity)
    matches: list[ActorIdentityMatch] = []
    grounding_matches: list[ActorIdentityMatch] = []
    assembly_matches: list[AssemblyIdentityMatch] = []
    uncertain: list[str] = []
    uncertain_assemblies: list[str] = []
    if queries:
        for actor in inventory.actors:
            if (
                eligible_actor_ids is not None
                and actor.live_actor_id not in eligible_actor_ids
            ):
                continue
            identity_match = _actor_name_match(actor, queries)
            if identity_match is None:
                continue
            if actor.eligible_for_stage1(scene_bounds):
                grounding_matches.append(identity_match)
                if identity_match.decision_eligible:
                    matches.append(identity_match)
            elif _eligibility_is_uncertain(actor, scene_bounds):
                uncertain.append(actor.live_actor_id)
        # Actor identity is the preferred representation. Assemblies are
        # considered only when no eligible whole actor already proves existence
        # and the graph explicitly leaves the assembly representation in scope.
        if (
            not matches
            and entity.inventory_representation
            is EntityInventoryRepresentation.ACTOR_OR_DECLARED_ASSEMBLY
        ):
            for assembly in inventory.assemblies:
                if eligible_actor_ids is not None and not set(
                    assembly.member_actor_ids
                ).issubset(eligible_actor_ids):
                    continue
                identity = _exact_assembly_identity(assembly, queries)
                if identity is None:
                    continue
                if assembly.eligible_for_stage1(scene_bounds, inventory.actors):
                    assembly_matches.append(
                        _assembly_identity_match(assembly, identity)
                    )
                elif _assembly_eligibility_is_uncertain(
                    assembly, scene_bounds, inventory.actors
                ):
                    uncertain_assemblies.append(assembly.assembly_id)
    matches.sort(key=lambda value: (value.actor_id.casefold(), value.actor_id))
    grounding_matches.sort(
        key=lambda value: (value.actor_id.casefold(), value.actor_id)
    )
    assembly_matches.sort(
        key=lambda value: (value.assembly_id.casefold(), value.assembly_id)
    )
    uncertain = sorted(set(uncertain), key=lambda value: (value.casefold(), value))
    uncertain_assemblies = sorted(
        set(uncertain_assemblies), key=lambda value: (value.casefold(), value)
    )
    return _EntityFinding(
        entity=entity,
        canonical_category=canonical,
        matches=tuple(matches),
        grounding_matches=tuple(grounding_matches),
        assembly_matches=tuple(assembly_matches),
        uncertain_actor_ids=tuple(uncertain),
        uncertain_assembly_ids=tuple(uncertain_assemblies),
        # Closure is deliberately checked against the canonical category, not
        # an alias supplied merely to recognize positive actor metadata.
        category_closed=(
            inventory.canonical_identity_category_is_closed(entity.name)
            and inventory.category_is_closed(entity.name)
            and (
                entity.inventory_representation
                is EntityInventoryRepresentation.WHOLE_ACTOR_ONLY
                or inventory.assembly_category_is_closed(entity.name)
            )
        ),
    )


def _verdict_for(
    finding: _EntityFinding,
    polarity: Polarity,
) -> tuple[ClaimVerdict, str | None, str]:
    if finding.matches:
        verdict = (
            ClaimVerdict.MATCH
            if polarity is Polarity.AFFIRMATIVE
            else ClaimVerdict.MISMATCH
        )
        sense = "supports" if polarity is Polarity.AFFIRMATIVE else "contradicts"
        return (
            verdict,
            UE_ASSET_INVENTORY_EVIDENCE,
            f"Exact eligible actor identity {sense} the {polarity.value} existence requirement.",
        )
    if finding.assembly_matches:
        verdict = (
            ClaimVerdict.MATCH
            if polarity is Polarity.AFFIRMATIVE
            else ClaimVerdict.MISMATCH
        )
        sense = "supports" if polarity is Polarity.AFFIRMATIVE else "contradicts"
        return (
            verdict,
            UE_DECLARED_ASSEMBLY_EVIDENCE,
            (
                "Exact eligible declared-assembly identity "
                f"{sense} the {polarity.value} existence requirement."
            ),
        )
    if finding.grounding_matches:
        return (
            ClaimVerdict.UNKNOWN,
            None,
            (
                f"{len(finding.grounding_matches)} eligible Actor(s) were grounded "
                "by a frozen alias or token-boundary name match. This localizes "
                "Stage 2 evidence but does not prove the complete canonical "
                "identity or its visual modifiers."
            ),
        )
    if finding.uncertain_actor_ids or finding.uncertain_assembly_ids:
        return (
            ClaimVerdict.UNKNOWN,
            None,
            (
                "An exact actor or declared-assembly identity match has unknown or "
                "incomplete declaration/membership/eligibility metadata; absence is "
                "not established."
            ),
        )
    if finding.category_closed:
        verdict = (
            ClaimVerdict.MISMATCH
            if polarity is Polarity.AFFIRMATIVE
            else ClaimVerdict.MATCH
        )
        return (
            verdict,
            UE_ASSET_INVENTORY_CLOSED_WORLD_EVIDENCE,
            (
                "The complete inventory explicitly closes this canonical category and "
                "contains no eligible exact identity match."
            ),
        )
    return (
        ClaimVerdict.UNKNOWN,
        None,
        (
            "No eligible exact identity match was found, but this canonical category "
            "is not closed in a complete inventory."
        ),
    )


def _assessment(
    finding: _EntityFinding,
    *,
    kind: Stage1AssessmentKind,
    node_id: str,
    polarity: Polarity,
    inventory_source: str,
    predicate_id: str | None = None,
) -> AtomicEntityAssessment:
    if kind in _STAGE2_VISUAL_ASSESSMENT_KINDS:
        if finding.matches or finding.assembly_matches:
            # Positive exact identity is authoritative even for an open-set
            # entity. The visual route exists only to accept substitutions
            # when that exact identity is absent; it must not force redundant
            # screenshots of an already-known asset/name.
            verdict, evidence_type, rationale = _verdict_for(finding, polarity)
            matches = (
                finding.matches
                if evidence_type == UE_ASSET_INVENTORY_EVIDENCE
                else ()
            )
            assembly_matches = (
                finding.assembly_matches
                if evidence_type == UE_DECLARED_ASSEMBLY_EVIDENCE
                else ()
            )
            rationale = (
                f"{rationale} The Stage 2 visual-substitution fallback was not "
                "needed."
            )
        else:
            verdict = ClaimVerdict.UNKNOWN
            evidence_type = None
            matches = ()
            assembly_matches = ()
            if finding.grounding_matches:
                rationale = (
                    f"{len(finding.grounding_matches)} eligible Actor(s) were "
                    "name-grounded for targeted Stage 2 capture, but only through "
                    "a frozen alias or token-boundary match. Stage 2/3 must verify "
                    "the full canonical identity and remaining visual semantics."
                )
            else:
                rationale = (
                    "No eligible exact asset/class/name or declared-assembly "
                    "identity matched this open-set entity. Stage 2 may retrieve "
                    "and visually confirm a semantically equivalent substitute; "
                    "Stage 1 never uses closed-world absence for this fallback "
                    "route."
                )
    else:
        verdict, evidence_type, rationale = _verdict_for(finding, polarity)
        matches = (
            finding.matches if evidence_type == UE_ASSET_INVENTORY_EVIDENCE else ()
        )
        assembly_matches = (
            finding.assembly_matches
            if evidence_type == UE_DECLARED_ASSEMBLY_EVIDENCE
            else ()
        )
    entity_assessment = kind in _ENTITY_ASSESSMENT_KINDS
    assessment_id = (
        f"entity:{finding.entity.id}:existence"
        if entity_assessment
        else f"predicate:{node_id}"
    )
    return AtomicEntityAssessment(
        assessment_id=assessment_id,
        node_id=node_id,
        entity_id=finding.entity.id,
        predicate_id=predicate_id,
        kind=kind,
        polarity=polarity,
        canonical_category=finding.canonical_category,
        verdict=verdict,
        evidence_type=evidence_type,
        matches=matches,
        grounding_matches=finding.grounding_matches,
        assembly_matches=assembly_matches,
        inventory_source=inventory_source,
        rationale=rationale,
        dependency_only=kind in _DEPENDENCY_ASSESSMENT_KINDS,
    )


def _pure_existence_subject(
    graph: RequirementGraph,
    predicate: PredicateNode,
) -> EntityNode | None:
    """Return the sole object whose metadata can decide pure existence.

    V2 may retain contextual region arguments on an otherwise atomic presence
    leaf.  Those references must not force a redundant screenshot when exact
    Actor identity already proves the requested asset exists.  Conversely,
    qualifiers, scoped quantifiers, relations, quantities, and logic require
    more than identity and therefore remain visual.
    """

    arguments = graph.arguments_for(predicate.id)
    if graph.schema_version == "1.0":
        if len(arguments) != 1:
            return None
        entity = graph.node(arguments[0].target_id)
        return entity if isinstance(entity, EntityNode) else None

    semantic = predicate.semantic_parameters
    if (
        str(semantic.get("requirement_type") or "") != "presence"
        or semantic.get("qualifiers")
        or semantic.get("scope") is not None
        or semantic.get("relation") is not None
        or semantic.get("quantity") is not None
        or semantic.get("logic") is not None
    ):
        return None
    subjects = []
    for argument in arguments:
        if argument.role not in {"subject", "collection"}:
            continue
        entity = graph.node(argument.target_id)
        if isinstance(entity, EntityNode) and entity.entity_type is EntityType.OBJECT:
            subjects.append(entity)
    return subjects[0] if len(subjects) == 1 else None


def evaluate_atomic_entities(
    graph: RequirementGraph,
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    *,
    semantic_backend: object | None = None,
    locator_top_k: int = 10,
    eligible_actor_ids_by_entity: Mapping[str, Iterable[str]] | None = None,
) -> Stage1Result:
    """Evaluate object entities and explicit existence predicates without RGB.

    Every ``EntityType.OBJECT`` receives an affirmative assessment.  An entity
    with positive graph effective weight is an independent entity requirement;
    an unscored entity is a dependency only.  Entities explicitly marked
    ``EntityEvaluationRoute.STAGE2_VISUAL`` receive an explicit routing
    assessment, but no Stage 1 existence/absence decision.  The deprecated
    retrieval keyword arguments are accepted for caller compatibility but are
    deliberately ignored: all candidate retrieval belongs to Stage 2.
    An explicit
    ``PredicateType.EXISTENCE`` receives one additional assessment using its
    own polarity.  Other predicates are not interpreted here; notably an
    ISM/HISM ``instance_count_hint`` never turns into a Stage 1 count verdict.
    """

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    if not isinstance(inventory, ActorInventorySnapshot):
        raise TypeError("inventory must be an ActorInventorySnapshot")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")
    # Compatibility-only parameters.  Do not inspect or call the backend here:
    # Stage 1 must remain deterministic lexical inventory evaluation.
    _ = semantic_backend, locator_top_k
    scoped_actor_ids = {
        str(entity_id): frozenset(str(value) for value in actor_ids)
        for entity_id, actor_ids in (eligible_actor_ids_by_entity or {}).items()
    }

    entities = tuple(
        sorted(
            (
                node
                for node in graph.nodes
                if isinstance(node, EntityNode)
                and node.entity_type is EntityType.OBJECT
            ),
            key=lambda node: (node.id.casefold(), node.id),
        )
    )
    findings = {
        entity.id: _find_entity(
            entity,
            inventory,
            scene_bounds,
            scoped_actor_ids.get(entity.id),
        )
        for entity in entities
    }
    effective_weights = graph.effective_weights()

    assessments: list[AtomicEntityAssessment] = []
    for entity in entities:
        scored = effective_weights.get(entity.id, 0.0) > 0.0
        if entity.evaluation_route is EntityEvaluationRoute.STAGE2_VISUAL:
            kind = (
                Stage1AssessmentKind.STAGE2_VISUAL_REQUIREMENT
                if scored
                else Stage1AssessmentKind.STAGE2_VISUAL_DEPENDENCY
            )
        else:
            kind = (
                Stage1AssessmentKind.ENTITY_REQUIREMENT
                if scored
                else Stage1AssessmentKind.ENTITY_DEPENDENCY
            )
        assessments.append(
            _assessment(
                findings[entity.id],
                kind=kind,
                node_id=entity.id,
                polarity=Polarity.AFFIRMATIVE,
                inventory_source=inventory.source,
            )
        )

    predicates = sorted(
        (
            node
            for node in graph.nodes
            if isinstance(node, PredicateNode)
            and node.predicate_type is PredicateType.EXISTENCE
        ),
        key=lambda node: (node.id.casefold(), node.id),
    )
    for predicate in predicates:
        subject = _pure_existence_subject(graph, predicate)
        if subject is None:
            continue
        finding = findings.get(subject.id)
        if finding is None:
            # Stage 1 intentionally ignores existence predicates over surfaces
            # and regions, even though the graph itself can represent them.
            continue
        entity = finding.entity
        predicate_kind = (
            Stage1AssessmentKind.STAGE2_VISUAL_EXISTENCE_PREDICATE
            if entity.evaluation_route is EntityEvaluationRoute.STAGE2_VISUAL
            else Stage1AssessmentKind.EXISTENCE_PREDICATE
        )
        assessments.append(
            _assessment(
                finding,
                kind=predicate_kind,
                node_id=predicate.id,
                predicate_id=predicate.id,
                polarity=_coerce_polarity(predicate.polarity),
                inventory_source=inventory.source,
            )
        )

    eligible_actor_ids = tuple(
        actor.live_actor_id
        for actor in sorted(
            (
                actor
                for actor in inventory.actors
                if actor.eligible_for_stage1(scene_bounds)
            ),
            key=lambda actor: (actor.live_actor_id.casefold(), actor.live_actor_id),
        )
    )
    eligible_assembly_ids = tuple(
        assembly.assembly_id
        for assembly in sorted(
            (
                assembly
                for assembly in inventory.assemblies
                if assembly.eligible_for_stage1(scene_bounds, inventory.actors)
            ),
            key=lambda assembly: (
                assembly.assembly_id.casefold(),
                assembly.assembly_id,
            ),
        )
    )
    return Stage1Result(
        assessments=tuple(assessments),
        inventory_status=inventory.status,
        inventory_source=inventory.source,
        eligible_actor_ids=eligible_actor_ids,
        eligible_assembly_ids=eligible_assembly_ids,
        schema_version="1.0",
    )


def evaluate_stage1(
    graph: RequirementGraph,
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    *,
    semantic_backend: object | None = None,
    locator_top_k: int = 10,
    eligible_actor_ids_by_entity: Mapping[str, Iterable[str]] | None = None,
) -> Stage1Result:
    """Named pipeline-stage alias for :func:`evaluate_atomic_entities`."""

    return evaluate_atomic_entities(
        graph,
        inventory,
        scene_bounds,
        semantic_backend=semantic_backend,
        locator_top_k=locator_top_k,
        eligible_actor_ids_by_entity=eligible_actor_ids_by_entity,
    )


@dataclass(frozen=True, slots=True)
class Stage1AtomicEntityEvaluator:
    """Small reusable evaluator wrapper for controller integration."""

    scene_bounds: SceneBounds
    semantic_backend: object | None = None
    locator_top_k: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.scene_bounds, SceneBounds):
            raise TypeError("scene_bounds must be a SceneBounds")

    def evaluate(
        self,
        graph: RequirementGraph,
        inventory: ActorInventorySnapshot,
    ) -> Stage1Result:
        return evaluate_atomic_entities(
            graph,
            inventory,
            self.scene_bounds,
            semantic_backend=self.semantic_backend,
            locator_top_k=self.locator_top_k,
        )


__all__ = [
    "UE_ASSET_INVENTORY_CLOSED_WORLD_EVIDENCE",
    "UE_ASSET_INVENTORY_EVIDENCE",
    "UE_DECLARED_ASSEMBLY_EVIDENCE",
    "ActorIdentityMatch",
    "AssemblyIdentityMatch",
    "AtomicEntityAssessment",
    "IdentityMatchRule",
    "IdentityQuerySource",
    "Stage1AssessmentKind",
    "Stage1AtomicEntityEvaluator",
    "Stage1Result",
    "evaluate_atomic_entities",
    "evaluate_stage1",
]
