"""Serializable UE actor inventory contracts for Stage 1 evaluation.

This module deliberately stops at describing actors.  It does not inspect a
live SPEAR wrapper and it does not decide whether a prompt entity exists.  A
runtime adapter may use :func:`descriptor_from_mapping` to remove engine
objects at the boundary; Stage 1 can then operate on deterministic, JSON-safe
metadata only.

Identity metadata and contextual metadata are kept separate.  In particular,
an asset folder such as ``/Game/Gothic/Props`` may help describe context, but
it must not by itself turn a generic mesh actor into a gothic object.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .asset_candidates import asset_path_basename, normalize_asset_text
from .contracts import JsonSerializable, SceneBounds

_MISSING = object()
_NO_DEFAULT = object()


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _singular(token: str) -> str:
    """Apply the same deliberately small lexical singularization as retrieval."""

    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


_GENERIC_IDENTITY_PHRASES = frozenset(
    {
        "actor",
        "brush",
        "camera actor",
        "cine camera actor",
        "default scene root",
        "hierarchical instanced static actor",
        "instanced foliage actor",
        "instanced static actor",
        "level instance",
        "scene component",
        "skeletal actor",
        "static actor",
        "world setting",
    }
)
_GENERIC_IDENTITY_TOKENS = frozenset(
    {
        "actor",
        "asset",
        "base",
        "blueprint",
        "brush",
        "camera",
        "cine",
        "component",
        "default",
        "foliage",
        "geometry",
        "hierarchical",
        "instance",
        "instanced",
        "item",
        "level",
        "mesh",
        "model",
        "object",
        "primitive",
        "prop",
        "root",
        "scene",
        "setting",
        "skeletal",
        "static",
        "world",
    }
)
_PATH_CONTEXT_STOPWORDS = frozenset(
    {
        "asset",
        "assets",
        "blueprint",
        "blueprints",
        "content",
        "environment",
        "environments",
        "game",
        "map",
        "maps",
        "mesh",
        "meshes",
        "prop",
        "props",
        "script",
        "static",
    }
)
_GENERATED_CLASS_SUFFIX_RE = re.compile(r"(?:^|_)C(?:_\d+)?$", re.IGNORECASE)
_ACTOR_CLASS_PREFIX_RE = re.compile(r"^A[A-Z]")
_NON_SEMANTIC_IDENTITY_ERROR = "identity term must contain semantic text"


def normalize_identity_term(value: Any) -> str:
    """Return a canonical lexical identity term, or ``""`` for UE boilerplate.

    The function is intentionally lexical rather than ontological.  It folds
    common Unreal prefixes/separators, removes numeric instance suffixes,
    lightly singularizes words, and rejects names made solely from generic
    engine vocabulary.  It never performs fuzzy or synonym matching.
    """

    raw = _clean_text(value)
    if not raw:
        return ""
    normalized = normalize_asset_text(raw)
    tokens = [_singular(token) for token in normalized.split()]

    # Blueprint-generated class object names conventionally end in ``_C``.
    if tokens and tokens[-1] == "c" and _GENERATED_CLASS_SUFFIX_RE.search(raw):
        tokens.pop()

    # Native actor classes conventionally use an A prefix (ACandleActor).
    if tokens and tokens[0] == "a" and _ACTOR_CLASS_PREFIX_RE.match(raw):
        tokens.pop(0)

    # A semantic class often ends in Actor/Component/Class.  Removing the
    # wrapper leaves the useful noun while a truly generic class is rejected
    # below.  ``object`` is deliberately *not* a wrapper token: it can be a
    # meaningful head noun in an arbitrary prompt (for example
    # ``ceremonial object``).  Treating it as boilerplate used to silently
    # change that category into the adjective-only ``ceremonial``.
    while tokens and tokens[-1] in {"actor", "component", "class"}:
        tokens.pop()

    term = " ".join(tokens)
    if not term or term in _GENERIC_IDENTITY_PHRASES:
        return ""
    if set(tokens).issubset(_GENERIC_IDENTITY_TOKENS):
        return ""
    return term


def _identity_leaf(value: Any) -> str:
    """Extract the object side of an Unreal path before term normalization."""

    raw = _clean_text(value).strip("'\"")
    if not raw:
        return ""
    if "/" in raw or "\\" in raw:
        return asset_path_basename(raw)
    return raw


def _vector3(value: Any, *, name: str) -> tuple[float, float, float]:
    if isinstance(value, Mapping):
        lowered = {str(key).casefold(): item for key, item in value.items()}
        if not all(axis in lowered for axis in ("x", "y", "z")):
            raise ValueError(f"{name} mapping must contain X/Y/Z")
        raw_values: Sequence[Any] = (
            lowered["x"],
            lowered["y"],
            lowered["z"],
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 3:
            raise ValueError(f"{name} must contain exactly three coordinates")
        raw_values = value
    else:
        raise TypeError(f"{name} must be a three-coordinate sequence or mapping")
    try:
        result = tuple(float(component) for component in raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} coordinates must be numeric") from exc
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} coordinates must be finite")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ActorBounds(JsonSerializable):
    """Axis-aligned actor bounds in Unreal centimetres."""

    center_cm: tuple[float, float, float]
    extent_cm: tuple[float, float, float]

    def __post_init__(self) -> None:
        center = _vector3(self.center_cm, name="center_cm")
        extent = _vector3(self.extent_cm, name="extent_cm")
        if any(component < 0.0 for component in extent):
            raise ValueError("extent_cm coordinates must be non-negative")
        object.__setattr__(self, "center_cm", center)
        object.__setattr__(self, "extent_cm", extent)

    @classmethod
    def from_min_max(
        cls,
        min_cm: Sequence[float],
        max_cm: Sequence[float],
    ) -> ActorBounds:
        minimum = _vector3(min_cm, name="min_cm")
        maximum = _vector3(max_cm, name="max_cm")
        if any(high < low for low, high in zip(minimum, maximum, strict=True)):
            raise ValueError("max_cm must be greater than or equal to min_cm")
        return cls(
            tuple(
                (low + high) / 2.0
                for low, high in zip(minimum, maximum, strict=True)
            ),  # type: ignore[arg-type]
            tuple(
                (high - low) / 2.0
                for low, high in zip(minimum, maximum, strict=True)
            ),  # type: ignore[arg-type]
        )

    @property
    def min_cm(self) -> tuple[float, float, float]:
        return tuple(
            center - extent
            for center, extent in zip(self.center_cm, self.extent_cm, strict=True)
        )  # type: ignore[return-value]

    @property
    def max_cm(self) -> tuple[float, float, float]:
        return tuple(
            center + extent
            for center, extent in zip(self.center_cm, self.extent_cm, strict=True)
        )  # type: ignore[return-value]

    def intersects(self, scene_bounds: SceneBounds) -> bool:
        """Return whether these bounds touch or overlap ``scene_bounds``."""

        if not isinstance(scene_bounds, SceneBounds):
            raise TypeError("scene_bounds must be a SceneBounds")
        return all(
            actor_high >= scene_low and actor_low <= scene_high
            for actor_low, actor_high, scene_low, scene_high in zip(
                self.min_cm,
                self.max_cm,
                scene_bounds.min_cm,
                scene_bounds.max_cm,
                strict=True,
            )
        )


class IdentityStrength(str, Enum):
    """Reliability tier attached to one identity metadata observation."""

    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"

    @property
    def priority(self) -> int:
        return {
            IdentityStrength.STRONG: 3,
            IdentityStrength.MEDIUM: 2,
            IdentityStrength.WEAK: 1,
        }[self]

    @classmethod
    def coerce(cls, value: IdentityStrength | str) -> IdentityStrength:
        if isinstance(value, cls):
            return value
        normalized = _clean_text(value).casefold()
        aliases = {
            "high": cls.STRONG,
            "primary": cls.STRONG,
            "moderate": cls.MEDIUM,
            "low": cls.WEAK,
            "fallback": cls.WEAK,
        }
        return aliases.get(normalized, cls(normalized))


class IdentitySource(str, Enum):
    """Provenance for an identity or context term.

    Source priority encodes the handoff policy.  Asset, specific class, and
    specific Unreal object names form the primary tier; structured tags form
    the middle tier; an ActorLabel is the fallback tier.
    """

    ASSET_PATH = "asset_path"
    ACTOR_CLASS = "actor_class"
    UNREAL_NAME = "unreal_name"
    STRUCTURED_TAG = "structured_tag"
    ACTOR_LABEL = "actor_label"

    @property
    def priority(self) -> int:
        return {
            IdentitySource.ASSET_PATH: 500,
            IdentitySource.ACTOR_CLASS: 500,
            IdentitySource.UNREAL_NAME: 500,
            IdentitySource.STRUCTURED_TAG: 300,
            IdentitySource.ACTOR_LABEL: 100,
        }[self]

    @property
    def default_strength(self) -> IdentityStrength:
        if self in {
            IdentitySource.ASSET_PATH,
            IdentitySource.ACTOR_CLASS,
            IdentitySource.UNREAL_NAME,
        }:
            return IdentityStrength.STRONG
        if self is IdentitySource.STRUCTURED_TAG:
            return IdentityStrength.MEDIUM
        return IdentityStrength.WEAK

    @classmethod
    def coerce(cls, value: IdentitySource | str) -> IdentitySource:
        if isinstance(value, cls):
            return value
        normalized = _clean_text(value).casefold().replace("-", "_")
        aliases = {
            "asset": cls.ASSET_PATH,
            "path": cls.ASSET_PATH,
            "class": cls.ACTOR_CLASS,
            "uclass": cls.ACTOR_CLASS,
            "name": cls.UNREAL_NAME,
            "tag": cls.STRUCTURED_TAG,
            "semantic_tag": cls.STRUCTURED_TAG,
            "label": cls.ACTOR_LABEL,
        }
        return aliases.get(normalized, cls(normalized))


@dataclass(frozen=True, slots=True)
class IdentityTerm(JsonSerializable):
    """One normalized term together with its exact metadata provenance."""

    term: str
    source: IdentitySource | str
    strength: IdentityStrength | str | None = None
    raw_value: str = ""

    def __post_init__(self) -> None:
        raw_term = _clean_text(self.term)
        term = normalize_identity_term(raw_term)
        if not term:
            raise ValueError(_NON_SEMANTIC_IDENTITY_ERROR)
        source = IdentitySource.coerce(self.source)
        strength = (
            source.default_strength
            if self.strength is None
            else IdentityStrength.coerce(self.strength)
        )
        object.__setattr__(self, "term", term)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "raw_value", _clean_text(self.raw_value) or raw_term)

    @property
    def value(self) -> str:
        """Descriptive alias for callers that call the normalized term a value."""

        return self.term

    @property
    def source_priority(self) -> int:
        return IdentitySource.coerce(self.source).priority


class MultiplicityKind(str, Enum):
    """How one actor record relates to possible visible instances."""

    SINGLE = "single"
    INSTANCED = "instanced"
    CLUSTER = "cluster"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: MultiplicityKind | str) -> MultiplicityKind:
        if isinstance(value, cls):
            return value
        normalized = _clean_text(value).casefold().replace("-", "_")
        aliases = {
            "actor": cls.SINGLE,
            "single_actor": cls.SINGLE,
            "ism": cls.INSTANCED,
            "hism": cls.INSTANCED,
            "instances": cls.INSTANCED,
            "clustered": cls.CLUSTER,
        }
        return aliases.get(normalized, cls(normalized))


def _coerce_term(
    value: IdentityTerm | Mapping[str, Any] | str,
    *,
    default_source: IdentitySource,
) -> IdentityTerm:
    if isinstance(value, IdentityTerm):
        return value
    if isinstance(value, Mapping):
        term = _mapping_value(value, "term", "value", "text", "name", default="")
        source = _mapping_value(value, "source", default=default_source)
        strength = _mapping_value(value, "strength", default=None)
        raw_value = _mapping_value(value, "raw_value", "raw", default="")
        return IdentityTerm(
            term=_clean_text(term),
            source=source,
            strength=strength,
            raw_value=_clean_text(raw_value),
        )
    return IdentityTerm(term=_clean_text(value), source=default_source)


def _record_skipped_identity_term(
    diagnostics: list[str] | None,
    *,
    source: IdentitySource,
    raw_value: Any,
) -> None:
    """Record a recoverable identity normalization failure when requested."""

    if diagnostics is None:
        return
    raw = _clean_text(raw_value)
    if not raw:
        return
    diagnostics.append(
        f"skipped {source.value} identity term {raw!r}: "
        f"{_NON_SEMANTIC_IDENTITY_ERROR}"
    )


def _raw_identity_value(value: IdentityTerm | Mapping[str, Any] | str) -> Any:
    if isinstance(value, IdentityTerm):
        return value.raw_value or value.term
    if isinstance(value, Mapping):
        return _mapping_value(
            value,
            "raw_value",
            "raw",
            "term",
            "value",
            "text",
            "name",
            default="",
        )
    return value


def _as_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, IdentityTerm, Mapping)):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _normalized_terms(
    values: Iterable[IdentityTerm | Mapping[str, Any] | str],
    *,
    default_source: IdentitySource,
    diagnostics: list[str] | None = None,
) -> tuple[IdentityTerm, ...]:
    selected: dict[tuple[str, IdentitySource], IdentityTerm] = {}
    insertion_order: dict[tuple[str, IdentitySource], int] = {}
    for index, value in enumerate(values):
        try:
            term = _coerce_term(value, default_source=default_source)
        except ValueError as exc:
            if str(exc) != _NON_SEMANTIC_IDENTITY_ERROR:
                raise
            _record_skipped_identity_term(
                diagnostics,
                source=default_source,
                raw_value=_raw_identity_value(value),
            )
            continue
        source = IdentitySource.coerce(term.source)
        key = (term.term, source)
        previous = selected.get(key)
        if previous is None:
            insertion_order[key] = index
            selected[key] = term
        elif (
            IdentityStrength.coerce(term.strength).priority
            > IdentityStrength.coerce(previous.strength).priority
        ):
            selected[key] = term
    return tuple(
        sorted(
            selected.values(),
            key=lambda item: (
                -item.source_priority,
                -IdentityStrength.coerce(item.strength).priority,
                insertion_order[(item.term, IdentitySource.coerce(item.source))],
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class ActorDescriptor(JsonSerializable):
    """JSON-safe description of one authoritative live actor identity.

    ``instance_count_hint`` is diagnostic metadata only.  In particular, one
    ISM/HISM or cluster actor must never be treated as that many visually
    verified objects without RGB evidence.

    ``locator_terms`` are deliberately outside ``identity_terms``.  They may
    describe a visible child/component mesh that is useful for aiming a Stage
    2 camera, but they can never establish the identity of the owning actor in
    Stage 1.

    ``identity_term_diagnostics`` records only malformed identity candidates
    that would otherwise abort inventory construction.  Expected Unreal
    boilerplate filtered by :func:`normalize_identity_term` is not recorded.
    """

    live_actor_id: str
    unreal_name: str | None = None
    actor_class: str | None = None
    asset_path: str | None = None
    actor_label: str | None = None
    identity_terms: tuple[IdentityTerm, ...] = ()
    context_terms: tuple[IdentityTerm, ...] = ()
    bounds: ActorBounds | None = None
    active: bool | None = None
    renderable: bool | None = None
    in_current_level: bool | None = None
    multiplicity: MultiplicityKind | str = MultiplicityKind.SINGLE
    instance_count_hint: int | None = None
    # Keep extension fields at the end so existing positional construction of
    # the original descriptor schema remains backward compatible.
    locator_terms: tuple[IdentityTerm, ...] = ()
    identity_term_diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        live_actor_id = _clean_text(self.live_actor_id)
        if not live_actor_id:
            raise ValueError("live_actor_id must be non-empty")
        object.__setattr__(self, "live_actor_id", live_actor_id)
        for name in ("unreal_name", "actor_class", "asset_path", "actor_label"):
            object.__setattr__(self, name, _clean_text(getattr(self, name)) or None)

        identity_term_diagnostics = [
            text
            for text in (
                _clean_text(value) for value in self.identity_term_diagnostics
            )
            if text
        ]
        object.__setattr__(
            self,
            "identity_terms",
            _normalized_terms(
                self.identity_terms,
                default_source=IdentitySource.STRUCTURED_TAG,
                diagnostics=identity_term_diagnostics,
            ),
        )
        object.__setattr__(
            self,
            "context_terms",
            _normalized_terms(
                self.context_terms,
                default_source=IdentitySource.STRUCTURED_TAG,
                diagnostics=identity_term_diagnostics,
            ),
        )
        object.__setattr__(
            self,
            "locator_terms",
            _normalized_terms(
                self.locator_terms,
                default_source=IdentitySource.ASSET_PATH,
                diagnostics=identity_term_diagnostics,
            ),
        )
        object.__setattr__(
            self,
            "identity_term_diagnostics",
            tuple(dict.fromkeys(identity_term_diagnostics)),
        )
        if self.bounds is not None and not isinstance(self.bounds, ActorBounds):
            raise TypeError("bounds must be ActorBounds or None")
        for name in ("active", "renderable", "in_current_level"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None")
        object.__setattr__(
            self, "multiplicity", MultiplicityKind.coerce(self.multiplicity)
        )
        if self.instance_count_hint is not None:
            if isinstance(self.instance_count_hint, bool):
                raise TypeError("instance_count_hint must be an integer or None")
            try:
                count = int(self.instance_count_hint)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "instance_count_hint must be an integer or None"
                ) from exc
            if count < 0 or count != self.instance_count_hint:
                raise ValueError("instance_count_hint must be a non-negative integer")
            object.__setattr__(self, "instance_count_hint", count)

    @property
    def identity_values(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(term.term for term in self.identity_terms))

    def eligible_for_stage1(self, scene_bounds: SceneBounds | None) -> bool:
        """Fail closed unless level/activity/renderability/bounds are explicit."""

        return (
            self.in_current_level is True
            and self.active is True
            and self.renderable is True
            and scene_bounds is not None
            and self.bounds is not None
            and self.bounds.intersects(scene_bounds)
        )


@dataclass(frozen=True, slots=True)
class AssemblyMemberRole(JsonSerializable):
    """Optional semantic role for one explicitly declared assembly member."""

    actor_id: str
    role: str

    def __post_init__(self) -> None:
        actor_id = _clean_text(self.actor_id)
        role = normalize_identity_term(self.role)
        if not actor_id:
            raise ValueError("assembly member actor_id must be non-empty")
        if not role:
            raise ValueError("assembly member role must contain semantic text")
        object.__setattr__(self, "actor_id", actor_id)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class AssemblyDescriptor(JsonSerializable):
    """Explicit authored identity for a logical object made from actors.

    Assemblies are declarations, never spatial or lexical clusters inferred by
    Stage 1. ``bounds`` is the authored/member-union AABB and is used only for
    scene eligibility. It cannot itself support the assembly's identity.
    """

    assembly_id: str
    identity_terms: tuple[IdentityTerm, ...] = ()
    member_actor_ids: tuple[str, ...] = ()
    member_roles: tuple[AssemblyMemberRole, ...] = ()
    declaration_source: str = ""
    identity_declared: bool = False
    membership_complete: bool = False
    bounds: ActorBounds | None = None
    active: bool | None = None
    renderable: bool | None = None
    in_current_level: bool | None = None

    def __post_init__(self) -> None:
        assembly_id = _clean_text(self.assembly_id)
        if not assembly_id:
            raise ValueError("assembly_id must be non-empty")
        object.__setattr__(self, "assembly_id", assembly_id)
        object.__setattr__(
            self,
            "identity_terms",
            _normalized_terms(
                self.identity_terms,
                default_source=IdentitySource.STRUCTURED_TAG,
            ),
        )

        member_ids = tuple(_clean_text(value) for value in self.member_actor_ids)
        if any(not value for value in member_ids):
            raise ValueError("member_actor_ids must be non-empty")
        folded_members = [value.casefold() for value in member_ids]
        if len(folded_members) != len(set(folded_members)):
            raise ValueError("member_actor_ids must be unique")
        object.__setattr__(self, "member_actor_ids", member_ids)

        roles = tuple(self.member_roles)
        if any(not isinstance(value, AssemblyMemberRole) for value in roles):
            raise TypeError("member_roles must contain AssemblyMemberRole values")
        role_ids = [value.actor_id.casefold() for value in roles]
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("member_roles may name each actor at most once")
        if not set(role_ids).issubset(set(folded_members)):
            raise ValueError("member_roles must refer to declared member_actor_ids")
        object.__setattr__(self, "member_roles", roles)

        declaration_source = _clean_text(self.declaration_source)
        if not isinstance(self.identity_declared, bool):
            raise TypeError("identity_declared must be bool")
        if self.identity_terms and not declaration_source:
            raise ValueError(
                "assembly identity terms require declaration_source provenance"
            )
        object.__setattr__(self, "declaration_source", declaration_source)
        if not isinstance(self.membership_complete, bool):
            raise TypeError("membership_complete must be bool")
        if self.bounds is not None and not isinstance(self.bounds, ActorBounds):
            raise TypeError("bounds must be ActorBounds or None")
        for name in ("active", "renderable", "in_current_level"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None")

    @property
    def identity_values(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(term.term for term in self.identity_terms))

    def members_resolve(self, known_actor_ids: Iterable[str]) -> bool:
        known = {_clean_text(value).casefold() for value in known_actor_ids}
        return bool(self.member_actor_ids) and all(
            actor_id.casefold() in known for actor_id in self.member_actor_ids
        )

    def resolved_members(
        self, actors: Iterable[ActorDescriptor]
    ) -> tuple[ActorDescriptor, ...] | None:
        """Return members in declaration order, or ``None`` if any ref is stale."""

        by_id = {actor.live_actor_id.casefold(): actor for actor in actors}
        if not self.member_actor_ids:
            return None
        resolved: list[ActorDescriptor] = []
        for actor_id in self.member_actor_ids:
            actor = by_id.get(actor_id.casefold())
            if actor is None:
                return None
            resolved.append(actor)
        return tuple(resolved)

    def eligible_for_stage1(
        self,
        scene_bounds: SceneBounds | None,
        actors: Iterable[ActorDescriptor],
    ) -> bool:
        """Fail closed unless declaration, members and eligibility are explicit."""

        resolved = self.resolved_members(actors)
        return (
            self.identity_declared
            and bool(self.identity_terms)
            and self.membership_complete
            and resolved is not None
            # Every declared member must be a known live visual member.  The
            # assembly itself may straddle the configured scene boundary, so
            # require at least one member to intersect the scene rather than
            # incorrectly requiring every member's bounds to be contained.
            and all(
                actor.in_current_level is True
                and actor.active is True
                and actor.renderable is True
                and actor.bounds is not None
                for actor in (resolved or ())
            )
            and any(
                actor.bounds is not None and actor.bounds.intersects(scene_bounds)
                for actor in (resolved or ())
            )
            and self.in_current_level is True
            and self.active is True
            and self.renderable is True
            and scene_bounds is not None
            and self.bounds is not None
            and self.bounds.intersects(scene_bounds)
        )


class InventoryStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    ERROR = "error"

    @classmethod
    def coerce(cls, value: InventoryStatus | str) -> InventoryStatus:
        if isinstance(value, cls):
            return value
        return cls(_clean_text(value).casefold())


@dataclass(frozen=True, slots=True)
class ActorInventorySnapshot(JsonSerializable):
    """One immutable inventory observation and its independent closure contracts.

    ``closed_categories`` says that actor enumeration and required metadata are
    complete for a semantic category.  It does *not* say that arbitrary prompt
    wording can be resolved by exact comparison with the inventory's identity
    terms.  ``canonical_identity_closed_categories`` is the stronger, authored
    assertion that the category's identity vocabulary is canonical and
    exhaustive.  Stage 1 absence therefore needs both declarations; inventory
    completeness alone must not turn an unresolved synonym into a negative.
    """

    actors: tuple[ActorDescriptor, ...] = ()
    status: InventoryStatus | str = InventoryStatus.PARTIAL
    closed_categories: tuple[str, ...] = ()
    source: str = ""
    errors: tuple[str, ...] = ()
    assemblies: tuple[AssemblyDescriptor, ...] = ()
    assembly_closed_categories: tuple[str, ...] = ()
    canonical_identity_closed_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        actors = tuple(self.actors)
        if any(not isinstance(actor, ActorDescriptor) for actor in actors):
            raise TypeError("actors must contain ActorDescriptor values")
        seen: dict[str, str] = {}
        for actor in actors:
            key = actor.live_actor_id.casefold()
            if key in seen:
                raise ValueError(
                    "duplicate live_actor_id in inventory snapshot: "
                    f"{seen[key]!r} and {actor.live_actor_id!r}"
                )
            seen[key] = actor.live_actor_id

        categories: list[str] = []
        for value in self.closed_categories:
            normalized = normalize_identity_term(value)
            if not normalized:
                raise ValueError("closed_categories must contain semantic names")
            if normalized not in categories:
                categories.append(normalized)
        object.__setattr__(self, "actors", actors)
        assemblies = tuple(self.assemblies)
        if any(not isinstance(value, AssemblyDescriptor) for value in assemblies):
            raise TypeError("assemblies must contain AssemblyDescriptor values")
        assembly_ids = [value.assembly_id.casefold() for value in assemblies]
        if len(assembly_ids) != len(set(assembly_ids)):
            raise ValueError("duplicate assembly_id in inventory snapshot")
        object.__setattr__(self, "assemblies", assemblies)
        assembly_categories: list[str] = []
        for value in self.assembly_closed_categories:
            normalized = normalize_identity_term(value)
            if not normalized:
                raise ValueError(
                    "assembly_closed_categories must contain semantic names"
                )
            if normalized not in assembly_categories:
                assembly_categories.append(normalized)
        object.__setattr__(
            self, "assembly_closed_categories", tuple(assembly_categories)
        )
        canonical_identity_categories: list[str] = []
        for value in self.canonical_identity_closed_categories:
            normalized = normalize_identity_term(value)
            if not normalized:
                raise ValueError(
                    "canonical_identity_closed_categories must contain "
                    "semantic names"
                )
            if normalized not in canonical_identity_categories:
                canonical_identity_categories.append(normalized)
        object.__setattr__(
            self,
            "canonical_identity_closed_categories",
            tuple(canonical_identity_categories),
        )
        object.__setattr__(self, "status", InventoryStatus.coerce(self.status))
        object.__setattr__(self, "closed_categories", tuple(categories))
        object.__setattr__(self, "source", _clean_text(self.source))
        object.__setattr__(
            self,
            "errors",
            tuple(
                dict.fromkeys(
                    text
                    for text in (_clean_text(value) for value in self.errors)
                    if text
                )
            ),
        )

    def category_is_closed(self, name: str) -> bool:
        """Return whether absence is meaningful for this semantic category.

        A category declaration is trusted only on a complete snapshot.  A
        partial/error/unavailable inventory therefore cannot manufacture a
        Stage 1 negative verdict even if stale closure metadata was supplied.
        """

        normalized = normalize_identity_term(name)
        return (
            bool(normalized)
            and InventoryStatus.coerce(self.status) is InventoryStatus.COMPLETE
            and normalized in self.closed_categories
        )

    def assembly_category_is_closed(self, name: str) -> bool:
        """Return whether declared-assembly coverage closes this category."""

        normalized = normalize_identity_term(name)
        return (
            bool(normalized)
            and InventoryStatus.coerce(self.status) is InventoryStatus.COMPLETE
            and normalized in self.assembly_closed_categories
        )

    def canonical_identity_category_is_closed(self, name: str) -> bool:
        """Return whether exact identity lookup may prove category absence.

        This is deliberately independent from actor/assembly enumeration
        coverage.  The declaration must come from an authoritative authored
        vocabulary and is never inferred from asset strings or a successful
        loaded-world scan.
        """

        normalized = normalize_identity_term(name)
        return (
            bool(normalized)
            and InventoryStatus.coerce(self.status) is InventoryStatus.COMPLETE
            and normalized in self.canonical_identity_closed_categories
        )


def estimate_camera_content_floor_z(
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
) -> float:
    """Estimate a robust content floor for non-targeted camera placement.

    A single underground, tilted, or unusually thick actor can place the union
    AABB far below the visible scene.  With enough observations, use the
    upper-nearest ten-percentile of eligible actor bottoms; tiny inventories
    retain their conservative minimum.
    """

    if not isinstance(inventory, ActorInventorySnapshot):
        raise TypeError("inventory must be an ActorInventorySnapshot")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")
    bottoms = sorted(
        actor.bounds.min_cm[2]
        for actor in inventory.actors
        if actor.eligible_for_stage1(scene_bounds) and actor.bounds is not None
    )
    if not bottoms:
        return scene_bounds.min_z
    if len(bottoms) < 5:
        return max(scene_bounds.min_z, bottoms[0])
    index = min(len(bottoms) - 1, math.ceil((len(bottoms) - 1) * 0.10))
    return max(scene_bounds.min_z, min(scene_bounds.max_z, bottoms[index]))


def _normalized_mapping_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", _clean_text(value).casefold())


def _mapping_value(
    value: Mapping[str, Any],
    *names: str,
    default: Any = _NO_DEFAULT,
) -> Any:
    normalized = {_normalized_mapping_key(key): item for key, item in value.items()}
    for name in names:
        key = _normalized_mapping_key(name)
        if key in normalized:
            return normalized[key]
    if default is _NO_DEFAULT:
        raise KeyError(names[0] if names else "mapping value")
    return default


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None or value is _MISSING:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = _clean_text(value).casefold()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    if normalized in {"", "none", "null", "unknown"}:
        return None
    raise ValueError(f"{name} must be true, false, or unknown")


def _coerce_bounds(value: Any) -> ActorBounds | None:
    if value is None:
        return None
    if isinstance(value, ActorBounds):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("bounds must be ActorBounds, a mapping, or None")

    center = _mapping_value(
        value,
        "center_cm",
        "bounds_center_cm",
        "center",
        "origin",
        "location_cm",
        "location",
        default=_MISSING,
    )
    extent = _mapping_value(
        value,
        "extent_cm",
        "bounds_extent_cm",
        "extent",
        "box_extent",
        default=_MISSING,
    )
    if center is not _MISSING and extent is not _MISSING:
        return ActorBounds(center_cm=center, extent_cm=extent)

    minimum = _mapping_value(value, "min_cm", "minimum", "min", default=_MISSING)
    maximum = _mapping_value(value, "max_cm", "maximum", "max", default=_MISSING)
    if minimum is not _MISSING and maximum is not _MISSING:
        return ActorBounds.from_min_max(minimum, maximum)
    raise ValueError("bounds mapping must contain center/extent or min/max")


def _path_context_terms(asset_path: str | None) -> tuple[IdentityTerm, ...]:
    raw = _clean_text(asset_path).strip("'\"")
    if not raw or ("/" not in raw and "\\" not in raw):
        return ()
    package = raw.replace("\\", "/").rsplit("/", 1)[0]
    terms: list[IdentityTerm] = []
    for segment in package.split("/"):
        normalized = normalize_identity_term(segment)
        if not normalized or normalized in _PATH_CONTEXT_STOPWORDS:
            continue
        terms.append(
            IdentityTerm(
                term=normalized,
                source=IdentitySource.ASSET_PATH,
                raw_value=segment,
            )
        )
    return tuple(terms)


_IDENTITY_TAG_SCOPES = frozenset(
    {
        "category",
        "identity",
        "identityterm",
        "identityterms",
        "object",
        "objecttype",
        "semantic",
        "type",
    }
)
_CONTEXT_TAG_SCOPES = frozenset(
    {
        "attribute",
        "context",
        "contextterm",
        "contextterms",
        "environment",
        "material",
        "style",
    }
)
_SCOPED_TAG_RE = re.compile(
    r"^(identity|object|category|type|semantic|context|environment|style|"
    r"material|attribute)[\s.:=/]+(.+)$",
    re.IGNORECASE,
)


def _append_tag_value(
    value: Any,
    *,
    context: bool,
    identity_values: list[Any],
    context_values: list[Any],
) -> None:
    target = context_values if context else identity_values
    for item in _as_items(value):
        if isinstance(item, Mapping):
            _collect_structured_tags(item, identity_values, context_values)
            continue
        text = _clean_text(item)
        if not text:
            continue
        match = _SCOPED_TAG_RE.match(text)
        if match:
            scope = _normalized_mapping_key(match.group(1))
            scoped_target = (
                context_values if scope in _CONTEXT_TAG_SCOPES else identity_values
            )
            scoped_target.append(match.group(2))
        else:
            target.append(item)


def _collect_structured_tags(
    tags: Any,
    identity_values: list[Any],
    context_values: list[Any],
) -> None:
    if tags is None:
        return
    if isinstance(tags, Mapping):
        for key, value in tags.items():
            scope = _normalized_mapping_key(key)
            if scope in _IDENTITY_TAG_SCOPES | _CONTEXT_TAG_SCOPES:
                if not isinstance(value, bool):
                    _append_tag_value(
                        value,
                        context=scope in _CONTEXT_TAG_SCOPES,
                        identity_values=identity_values,
                        context_values=context_values,
                    )
                continue
            scoped_key = _SCOPED_TAG_RE.match(_clean_text(key))
            if scoped_key and value is True:
                scoped_scope = _normalized_mapping_key(scoped_key.group(1))
                target = (
                    context_values
                    if scoped_scope in _CONTEXT_TAG_SCOPES
                    else identity_values
                )
                target.append(scoped_key.group(2))
            elif isinstance(value, Mapping):
                _collect_structured_tags(value, identity_values, context_values)
        return
    for item in _as_items(tags):
        if isinstance(item, Mapping):
            _collect_structured_tags(item, identity_values, context_values)
            continue
        match = _SCOPED_TAG_RE.match(_clean_text(item))
        if not match:
            continue
        scope = _normalized_mapping_key(match.group(1))
        target = context_values if scope in _CONTEXT_TAG_SCOPES else identity_values
        target.append(match.group(2))


def _term_from_source(
    value: Any,
    source: IdentitySource,
    *,
    diagnostics: list[str] | None = None,
) -> IdentityTerm | None:
    raw = _clean_text(value)
    candidate = (
        _identity_leaf(raw)
        if source
        in {
            IdentitySource.ASSET_PATH,
            IdentitySource.ACTOR_CLASS,
            IdentitySource.UNREAL_NAME,
        }
        else raw
    )
    normalized = normalize_identity_term(candidate)
    if not normalized:
        # Empty output here is the expected filter for ordinary Unreal
        # boilerplate.  Diagnostics are reserved for the anomalous case where
        # a value first appears semantic but fails IdentityTerm validation.
        return None
    try:
        return IdentityTerm(term=normalized, source=source, raw_value=raw)
    except ValueError as exc:
        if str(exc) != _NON_SEMANTIC_IDENTITY_ERROR:
            raise
        # Normalization is intentionally conservative and is not guaranteed
        # to be idempotent for generated labels such as ``C1_c``.  Such a
        # label must not abort inventory construction: retain the actor's
        # other identity sources and record why this one source was skipped.
        _record_skipped_identity_term(
            diagnostics,
            source=source,
            raw_value=raw,
        )
        return None


def _infer_multiplicity(
    actor_class: str | None,
    unreal_name: str | None,
) -> MultiplicityKind:
    class_key = re.sub(r"[^a-z0-9]+", "", _clean_text(actor_class).casefold())
    name_key = re.sub(r"[^a-z0-9]+", "", _clean_text(unreal_name).casefold())
    if (
        "instancedstaticmesh" in class_key
        or "hierarchicalinstanced" in class_key
        or class_key.startswith(("hism", "ismactor"))
    ):
        return MultiplicityKind.INSTANCED
    if "cluster" in class_key or "clusteractor" in name_key:
        return MultiplicityKind.CLUSTER
    return MultiplicityKind.SINGLE


def build_actor_descriptor(
    *,
    live_actor_id: str,
    unreal_name: str | None = None,
    actor_class: str | None = None,
    asset_path: str | None = None,
    actor_label: str | None = None,
    structured_tags: Any = None,
    identity_tags: Iterable[Any] = (),
    context_tags: Iterable[Any] = (),
    identity_terms: Iterable[IdentityTerm | Mapping[str, Any] | str] = (),
    context_terms: Iterable[IdentityTerm | Mapping[str, Any] | str] = (),
    locator_asset_paths: Iterable[str] = (),
    locator_terms: Iterable[IdentityTerm | Mapping[str, Any] | str] = (),
    identity_term_diagnostics: Iterable[str] = (),
    bounds: ActorBounds | Mapping[str, Any] | None = None,
    active: bool | None = None,
    renderable: bool | None = None,
    in_current_level: bool | None = None,
    multiplicity: MultiplicityKind | str | None = None,
    instance_count_hint: int | None = None,
) -> ActorDescriptor:
    """Build a sanitized descriptor from already-read actor metadata fields."""

    diagnostics = [
        text
        for text in (_clean_text(value) for value in identity_term_diagnostics)
        if text
    ]
    generated_identity: list[IdentityTerm | Mapping[str, Any] | str] = []
    generated_context: list[IdentityTerm | Mapping[str, Any] | str] = []
    for value, source in (
        (asset_path, IdentitySource.ASSET_PATH),
        (actor_class, IdentitySource.ACTOR_CLASS),
        (unreal_name, IdentitySource.UNREAL_NAME),
    ):
        term = _term_from_source(value, source, diagnostics=diagnostics)
        if term is not None:
            generated_identity.append(term)

    tag_identity_values: list[Any] = list(_as_items(identity_tags))
    tag_context_values: list[Any] = list(_as_items(context_tags))
    _collect_structured_tags(
        structured_tags,
        tag_identity_values,
        tag_context_values,
    )
    for value in tag_identity_values:
        term = _term_from_source(
            value,
            IdentitySource.STRUCTURED_TAG,
            diagnostics=diagnostics,
        )
        if term is not None:
            generated_identity.append(term)
    for value in tag_context_values:
        term = _term_from_source(
            value,
            IdentitySource.STRUCTURED_TAG,
            diagnostics=diagnostics,
        )
        if term is not None:
            generated_context.append(term)

    label_term = _term_from_source(
        actor_label,
        IdentitySource.ACTOR_LABEL,
        diagnostics=diagnostics,
    )
    if label_term is not None:
        generated_identity.append(label_term)
    generated_context.extend(_path_context_terms(asset_path))
    generated_identity.extend(_as_items(identity_terms))
    generated_context.extend(_as_items(context_terms))
    generated_locators: list[IdentityTerm | Mapping[str, Any] | str] = []
    for value in _as_items(locator_asset_paths):
        term = _term_from_source(
            value,
            IdentitySource.ASSET_PATH,
            diagnostics=diagnostics,
        )
        if term is not None:
            generated_locators.append(term)
    generated_locators.extend(_as_items(locator_terms))

    return ActorDescriptor(
        live_actor_id=live_actor_id,
        unreal_name=unreal_name,
        actor_class=actor_class,
        asset_path=asset_path,
        actor_label=actor_label,
        identity_terms=tuple(generated_identity),
        context_terms=tuple(generated_context),
        locator_terms=tuple(generated_locators),
        identity_term_diagnostics=tuple(diagnostics),
        bounds=_coerce_bounds(bounds),
        active=_optional_bool(active, name="active"),
        renderable=_optional_bool(renderable, name="renderable"),
        in_current_level=_optional_bool(in_current_level, name="in_current_level"),
        multiplicity=multiplicity
        if multiplicity is not None
        else _infer_multiplicity(actor_class, unreal_name),
        instance_count_hint=instance_count_hint,
    )


def descriptor_from_mapping(
    value: Mapping[str, Any],
    *,
    trust_serialized_terms: bool = False,
) -> ActorDescriptor:
    """Sanitize a runtime/JSON mapping into an :class:`ActorDescriptor`.

    Unknown keys are intentionally ignored.  Consequently reflected values
    commonly stored under ``actor``, ``wrapper``, or ``unreal_actor`` cannot
    enter the descriptor or its serialized artifact.  Derived identity/context
    terms are regenerated from raw fields by default; callers may opt in to
    trusting them only when reading a canonical internal artifact.
    """

    if not isinstance(value, Mapping):
        raise TypeError("actor descriptor input must be a mapping")
    if not isinstance(trust_serialized_terms, bool):
        raise TypeError("trust_serialized_terms must be bool")

    bounds_value = _mapping_value(value, "bounds", "actor_bounds", default=_MISSING)
    if bounds_value is _MISSING:
        center = _mapping_value(
            value,
            "bounds_center_cm",
            "center_cm",
            "location_cm",
            "location",
            default=_MISSING,
        )
        extent = _mapping_value(
            value,
            "extent_cm",
            "bounds_extent_cm",
            "extent",
            "box_extent",
            default=_MISSING,
        )
        bounds_value = (
            {"center_cm": center, "extent_cm": extent}
            if center is not _MISSING and extent is not _MISSING
            else None
        )

    active = _mapping_value(value, "active", "is_active", default=None)
    renderable = _mapping_value(
        value,
        "renderable",
        "is_renderable",
        "visible",
        default=_MISSING,
    )
    if renderable is _MISSING:
        hidden = _mapping_value(
            value,
            "hidden",
            "hidden_in_game",
            "is_hidden",
            default=_MISSING,
        )
        parsed_hidden = _optional_bool(hidden, name="hidden")
        renderable = None if parsed_hidden is None else not parsed_hidden

    return build_actor_descriptor(
        live_actor_id=_clean_text(
            _mapping_value(
                value,
                "live_actor_id",
                "live_actor_key",
                "actor_id",
                "stable_name",
                "id",
                default="",
            )
        ),
        unreal_name=_clean_text(
            _mapping_value(value, "unreal_name", "unrealName", default="")
        )
        or None,
        actor_class=_clean_text(
            _mapping_value(
                value,
                "actor_class",
                "class_name",
                "uclass",
                "class",
                default="",
            )
        )
        or None,
        asset_path=_clean_text(
            _mapping_value(
                value,
                "asset_path",
                "mesh_path",
                "static_mesh_path",
                "path",
                default="",
            )
        )
        or None,
        actor_label=_clean_text(
            _mapping_value(value, "actor_label", "actorLabel", "label", default="")
        )
        or None,
        structured_tags=_mapping_value(
            value,
            "structured_tags",
            "semantic_tags",
            "tags",
            default=None,
        ),
        identity_tags=_as_items(_mapping_value(value, "identity_tags", default=())),
        context_tags=_as_items(_mapping_value(value, "context_tags", default=())),
        identity_terms=(
            _as_items(_mapping_value(value, "identity_terms", default=()))
            if trust_serialized_terms
            else ()
        ),
        context_terms=(
            _as_items(_mapping_value(value, "context_terms", default=()))
            if trust_serialized_terms
            else ()
        ),
        locator_terms=(
            _as_items(_mapping_value(value, "locator_terms", default=()))
            if trust_serialized_terms
            else ()
        ),
        identity_term_diagnostics=(
            _as_items(
                _mapping_value(value, "identity_term_diagnostics", default=())
            )
            if trust_serialized_terms
            else ()
        ),
        bounds=bounds_value,
        active=_optional_bool(active, name="active"),
        renderable=_optional_bool(renderable, name="renderable"),
        in_current_level=_optional_bool(
            _mapping_value(
                value,
                "in_current_level",
                "is_in_current_level",
                "current_level",
                default=None,
            ),
            name="in_current_level",
        ),
        multiplicity=_mapping_value(
            value, "multiplicity", "multiplicity_kind", default=None
        ),
        instance_count_hint=_mapping_value(
            value,
            "instance_count_hint",
            "instance_count",
            "num_instances",
            default=None,
        ),
    )


__all__ = [
    "ActorBounds",
    "ActorDescriptor",
    "ActorInventorySnapshot",
    "AssemblyDescriptor",
    "AssemblyMemberRole",
    "IdentitySource",
    "IdentityStrength",
    "IdentityTerm",
    "InventoryStatus",
    "MultiplicityKind",
    "build_actor_descriptor",
    "descriptor_from_mapping",
    "estimate_camera_content_floor_z",
    "normalize_identity_term",
]
