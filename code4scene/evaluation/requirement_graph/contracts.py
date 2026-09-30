"""Serializable contracts shared by the agentic scene evaluator.

The evaluator writes these values directly to JSON artifacts.  Keeping the
contracts independent of Unreal and of any particular model provider also
makes the scoring and exploration logic usable by offline replay tests.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Recursively convert evaluator values to objects accepted by ``json``."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [to_jsonable(item) for item in value]
    return value


class JsonSerializable:
    """Mixin giving evaluator dataclasses a stable JSON-dictionary form."""

    def to_dict(self) -> dict[str, Any]:
        result = to_jsonable(self)
        if not isinstance(result, dict):  # pragma: no cover - defensive guard
            raise TypeError(f"{type(self).__name__} did not serialize to an object")
        return result


def _vector3(name: str, value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} coordinates must be finite")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class SceneBounds(JsonSerializable):
    """Axis-aligned content bounds in Unreal centimetres."""

    min_cm: tuple[float, float, float]
    max_cm: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = _vector3("min_cm", self.min_cm)
        maximum = _vector3("max_cm", self.max_cm)
        if any(high <= low for low, high in zip(minimum, maximum, strict=True)):
            raise ValueError("max_cm must be strictly greater than min_cm on every axis")
        object.__setattr__(self, "min_cm", minimum)
        object.__setattr__(self, "max_cm", maximum)

    @classmethod
    def from_flat_values(
        cls,
        min_x: float,
        min_y: float,
        min_z: float,
        max_x: float,
        max_y: float,
        max_z: float,
    ) -> SceneBounds:
        return cls((min_x, min_y, min_z), (max_x, max_y, max_z))

    @property
    def min_x(self) -> float:
        return self.min_cm[0]

    @property
    def min_y(self) -> float:
        return self.min_cm[1]

    @property
    def min_z(self) -> float:
        return self.min_cm[2]

    @property
    def max_x(self) -> float:
        return self.max_cm[0]

    @property
    def max_y(self) -> float:
        return self.max_cm[1]

    @property
    def max_z(self) -> float:
        return self.max_cm[2]

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_y - self.min_y

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    @property
    def center_cm(self) -> tuple[float, float, float]:
        return tuple(
            (low + high) / 2.0
            for low, high in zip(self.min_cm, self.max_cm, strict=True)
        )  # type: ignore[return-value]

    @property
    def horizontal_diagonal(self) -> float:
        return math.hypot(self.width, self.depth)

    def expanded(self, fraction: float = 0.1) -> SceneBounds:
        """Expand each side by ``fraction`` of that axis' original span."""

        if not math.isfinite(fraction) or fraction < 0:
            raise ValueError("fraction must be a finite non-negative number")
        margin = (self.width * fraction, self.depth * fraction, self.height * fraction)
        return SceneBounds(
            tuple(
                low - delta
                for low, delta in zip(self.min_cm, margin, strict=True)
            ),  # type: ignore[arg-type]
            tuple(
                high + delta
                for high, delta in zip(self.max_cm, margin, strict=True)
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CameraPose(JsonSerializable):
    """A free-camera transform using Unreal's centimetre/degree convention."""

    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float = 0.0

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.z, self.pitch, self.yaw, self.roll)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("camera pose values must be finite")
        for name in ("x", "y", "z", "pitch", "yaw", "roll"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def location_cm(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def rotation_degrees(self) -> tuple[float, float, float]:
        return (self.pitch, self.yaw, self.roll)

    def to_world_camera_source(self, camera_id: str) -> dict[str, Any]:
        """Return the source object accepted by world-camera capture."""

        if not camera_id or not camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        return {
            "camera_id": camera_id,
            "type": "world",
            "location_cm": list(self.location_cm),
            "rotation_degrees": list(self.rotation_degrees),
        }


class ClaimType(str, Enum):
    SCENE_IDENTITY = "scene_identity"
    SPATIAL_RELATION = "spatial_relation"
    ATMOSPHERE = "atmosphere"
    SURFACE_MATERIAL = "surface_material"
    OBJECT = "object"
    OBJECT_ATTRIBUTE = "object_attribute"


_CLAIM_TYPE_ALIASES = {
    "identity": ClaimType.SCENE_IDENTITY,
    "scene_type": ClaimType.SCENE_IDENTITY,
    "relation": ClaimType.SPATIAL_RELATION,
    "spatial": ClaimType.SPATIAL_RELATION,
    "surface": ClaimType.SURFACE_MATERIAL,
    "material": ClaimType.SURFACE_MATERIAL,
    "attribute": ClaimType.OBJECT_ATTRIBUTE,
}


def coerce_claim_type(value: ClaimType | str) -> ClaimType:
    if isinstance(value, ClaimType):
        return value
    normalized = str(value).strip().lower()
    if normalized in _CLAIM_TYPE_ALIASES:
        return _CLAIM_TYPE_ALIASES[normalized]
    return ClaimType(normalized)


def default_claim_weight(claim_type: ClaimType | str) -> int:
    claim_type = coerce_claim_type(claim_type)
    if claim_type in {ClaimType.SCENE_IDENTITY, ClaimType.SPATIAL_RELATION}:
        return 3
    if claim_type in {ClaimType.ATMOSPHERE, ClaimType.SURFACE_MATERIAL}:
        return 2
    return 1


@dataclass(frozen=True)
class PromptClaim(JsonSerializable):
    id: str
    text: str
    source_span: tuple[int, int]
    type: ClaimType | str
    weight: int

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("claim id must be non-empty")
        if not self.text or not self.text.strip():
            raise ValueError("claim text must be non-empty")
        if len(self.source_span) != 2:
            raise ValueError("source_span must contain start and end offsets")
        start, end = (int(self.source_span[0]), int(self.source_span[1]))
        if start < 0 or end <= start:
            raise ValueError("source_span must satisfy 0 <= start < end")
        claim_type = coerce_claim_type(self.type)
        expected_weight = default_claim_weight(claim_type)
        if int(self.weight) != expected_weight:
            raise ValueError(
                f"{claim_type.value} claims must have weight {expected_weight}, got {self.weight}"
            )
        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "source_span", (start, end))
        object.__setattr__(self, "type", claim_type)
        object.__setattr__(self, "weight", expected_weight)

    @property
    def category(self) -> str:
        return coerce_claim_type(self.type).value


class ClaimVerdict(str, Enum):
    MATCH = "MATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"

    @property
    def score(self) -> float | None:
        return {
            ClaimVerdict.MATCH: 1.0,
            ClaimVerdict.PARTIAL_MATCH: 0.5,
            ClaimVerdict.MISMATCH: 0.0,
            ClaimVerdict.UNKNOWN: None,
        }[self]

    @classmethod
    def coerce(cls, value: ClaimVerdict | str) -> ClaimVerdict:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().upper())


@dataclass(frozen=True)
class ClaimAssessment(JsonSerializable):
    claim_id: str
    verdict: ClaimVerdict | str
    confidence: float = 1.0
    evidence_frame_ids: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.claim_id or not self.claim_id.strip():
            raise ValueError("claim_id must be non-empty")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        evidence = tuple(str(frame_id).strip() for frame_id in self.evidence_frame_ids)
        if any(not frame_id for frame_id in evidence):
            raise ValueError("evidence frame ids must be non-empty")
        object.__setattr__(self, "claim_id", self.claim_id.strip())
        object.__setattr__(self, "verdict", ClaimVerdict.coerce(self.verdict))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_frame_ids", evidence)


@dataclass(frozen=True)
class SceneObservation(JsonSerializable):
    frame_id: str
    description: str
    visible_entities: tuple[str, ...] = ()
    spatial_relations: tuple[str, ...] = ()
    atmosphere: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.frame_id or not self.frame_id.strip():
            raise ValueError("frame_id must be non-empty")
        object.__setattr__(self, "frame_id", self.frame_id.strip())
        object.__setattr__(self, "description", self.description.strip())
        for name in ("visible_entities", "spatial_relations", "atmosphere"):
            object.__setattr__(self, name, tuple(str(value) for value in getattr(self, name)))


@dataclass(frozen=True)
class SceneMemory(JsonSerializable):
    """Prompt-blind, evidence-linked visual memory of a scene."""

    summary: str = ""
    visible_entities: tuple[str, ...] = ()
    spatial_relations: tuple[str, ...] = ()
    atmosphere: tuple[str, ...] = ()
    evidence_frame_ids: tuple[str, ...] = ()
    observations: tuple[SceneObservation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", self.summary.strip())
        for name in (
            "visible_entities",
            "spatial_relations",
            "atmosphere",
            "evidence_frame_ids",
        ):
            object.__setattr__(self, name, tuple(str(value) for value in getattr(self, name)))
        observations = tuple(
            value if isinstance(value, SceneObservation) else SceneObservation(**value)
            for value in self.observations
        )
        object.__setattr__(self, "observations", observations)


@dataclass(frozen=True)
class ScoreSummary(JsonSerializable):
    resolved_score: float | None
    evidence_coverage: float
    scene_score: float | None
    status: str
    category_scores: Mapping[str, float | None] = field(default_factory=dict)
    category_coverage: Mapping[str, float] = field(default_factory=dict)
    total_weight: float = 0.0
    resolved_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("evidence_coverage",):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        for name in ("resolved_score", "scene_score"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            if value is not None:
                object.__setattr__(self, name, float(value))
        object.__setattr__(self, "category_scores", dict(self.category_scores))
        object.__setattr__(self, "category_coverage", dict(self.category_coverage))
        object.__setattr__(self, "total_weight", float(self.total_weight))
        object.__setattr__(self, "resolved_weight", float(self.resolved_weight))


@dataclass(frozen=True)
class SceneEvaluationResult(JsonSerializable):
    prompt: str
    claims: tuple[PromptClaim, ...]
    assessments: tuple[ClaimAssessment, ...]
    score_summary: ScoreSummary
    scene_memory: SceneMemory = field(default_factory=SceneMemory)
    agent_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    status: str = ""
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        claims = tuple(value if isinstance(value, PromptClaim) else PromptClaim(**value) for value in self.claims)
        assessments = tuple(
            value if isinstance(value, ClaimAssessment) else ClaimAssessment(**value)
            for value in self.assessments
        )
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(self, "agent_diagnostics", dict(self.agent_diagnostics))
        object.__setattr__(self, "status", self.status or self.score_summary.status)

    @property
    def scene_score(self) -> float | None:
        return self.score_summary.scene_score


# ---------------------------------------------------------------------------
# Requirement-graph contracts (schema 1.0)
# ---------------------------------------------------------------------------


class GraphValidationError(ValueError):
    """A fail-closed requirement-graph schema or topology error."""

    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = str(code)
        self.path = str(path)
        self.message = str(message)
        super().__init__(f"{self.code} at {self.path}: {self.message}")


class GraphNodeType(str, Enum):
    ENTITY = "entity"
    PREDICATE = "predicate"
    REQUIREMENT = "requirement"


class EntityType(str, Enum):
    OBJECT = "object"
    SURFACE = "surface"
    REGION = "region"


class EntityEvaluationRoute(str, Enum):
    """Stage responsible for deciding an entity requirement.

    ``INVENTORY_EXISTENCE`` preserves the schema-1.0 behaviour: Stage 1 may
    decide object existence from authoritative UE inventory metadata.
    ``STAGE2_VISUAL`` is an explicit open-set/generic route.  Stage 1 emits no
    inventory verdict for it; Stage 2 may independently retrieve actor
    locations for acquisition.

    The graph compiler, rather than phrase-specific code in Stage 1, owns this
    distinction.  This keeps arbitrary prompt vocabulary out of an ever-growing
    matcher blacklist.
    """

    INVENTORY_EXISTENCE = "inventory_existence"
    STAGE2_VISUAL = "stage2_visual"


class EntityGroundingMode(str, Enum):
    """How a frozen semantic entity may be localized at scoring time.

    The mode is authored without Candidate access. It describes the shape of
    the acquisition problem, not a Candidate-specific Actor id or asset path.
    AUTO exists only for backwards-compatible bundle loading; new Stage-0 v2
    authoring always freezes one of the explicit modes.
    """

    AUTO = "auto"
    ACTOR = "actor"
    ACTOR_COLLECTION = "actor_collection"
    DERIVED_REGION = "derived_region"
    SCENE_GLOBAL = "scene_global"


class EntityInventoryRepresentation(str, Enum):
    """Authoritative representations Stage 1 may use for one entity.

    ``ACTOR_OR_DECLARED_ASSEMBLY`` is the backward-compatible, conservative
    default: a positive may come from either a whole actor or an explicitly
    declared assembly, so closed-world absence requires coverage of both
    representations. ``WHOLE_ACTOR_ONLY`` is an explicit graph-compiler
    promise that this requirement denotes a whole UE actor.  In that mode
    assemblies are neither positive evidence nor part of absence coverage.

    This is semantic coverage metadata, not a name heuristic.  If the graph
    compiler cannot make the narrower promise it must retain the default.
    """

    ACTOR_OR_DECLARED_ASSEMBLY = "actor_or_declared_assembly"
    WHOLE_ACTOR_ONLY = "whole_actor_only"


class ReferentKind(str, Enum):
    INDIVIDUAL = "individual"
    COLLECTION = "collection"
    MASS = "mass"


class PredicateType(str, Enum):
    EXISTENCE = "existence"
    ATTRIBUTE = "attribute"
    MATERIAL = "material"
    SPATIAL_RELATION = "spatial_relation"
    COUNT = "count"
    ATMOSPHERE = "atmosphere"
    SCENE_IDENTITY = "scene_identity"
    QUANTITY = "quantity"
    SET = "set"
    DISTRIBUTION = "distribution"
    COMPOSITION = "composition"
    STYLE_BUNDLE = "style_bundle"
    ENVIRONMENT = "environment"
    LOGIC = "logic"
    BOUNDARY = "boundary"


class Polarity(str, Enum):
    AFFIRMATIVE = "affirmative"
    NEGATED = "negated"


class ComparisonOperator(str, Enum):
    EQ = "eq"
    GTE = "gte"
    LTE = "lte"
    BETWEEN = "between"


class MemberRole(str, Enum):
    SCORED_FACET = "scored_facet"
    SUPPORT_ONLY = "support_only"


class GraphEdgeType(str, Enum):
    ARGUMENT = "argument"
    SCOPE = "scope"
    REQUIREMENT_MEMBER = "requirement_member"


def _strict_fields(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    path: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise GraphValidationError("missing_field", path, f"missing {missing!r}")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise GraphValidationError("unknown_field", path, f"unknown {unknown!r}")


def _graph_enum(enum_type: type[Enum], value: Any, *, path: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(str(value))
    except (TypeError, ValueError) as exc:
        raise GraphValidationError(
            "invalid_enum", path, f"unsupported {enum_type.__name__} value {value!r}"
        ) from exc


def _graph_json(value: Any, *, path: str) -> Any:
    """Return a JSON-native semantic payload or fail at the graph boundary."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GraphValidationError(
                "non_finite_number", path, "must contain only finite JSON numbers"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise GraphValidationError(
                    "invalid_json_key", path, "semantic parameter keys must be strings"
                )
            result[key] = _graph_json(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _graph_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise GraphValidationError(
        "invalid_json_value",
        path,
        f"unsupported semantic parameter type {type(value).__name__}",
    )


def _graph_id(value: Any, *, path: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", text):
        raise GraphValidationError(
            "invalid_id",
            path,
            "must match [a-z][a-z0-9_]{0,63}",
        )
    return text


@dataclass(frozen=True)
class SourceSpan(JsonSerializable):
    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
        ):
            raise GraphValidationError(
                "invalid_span", "source_span", "offsets must be integers"
            )
        start = self.start
        end = self.end
        if start < 0 or end <= start:
            raise GraphValidationError(
                "invalid_span", "source_span", "must satisfy 0 <= start < end"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceSpan:
        _strict_fields(
            value,
            allowed={"start", "end"},
            required={"start", "end"},
            path="source_span",
        )
        return cls(start=value["start"], end=value["end"])


def _coerce_source_span(value: SourceSpan | Mapping[str, Any]) -> SourceSpan:
    if isinstance(value, SourceSpan):
        return value
    if isinstance(value, Mapping):
        return SourceSpan.from_dict(value)
    raise GraphValidationError("invalid_span", "source_span", "must be an object")


@dataclass(frozen=True)
class NumericConstraint(JsonSerializable):
    operator: ComparisonOperator | str
    value: int
    upper_value: int | None = None

    def __post_init__(self) -> None:
        operator = _graph_enum(ComparisonOperator, self.operator, path="constraint.operator")
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 0:
            raise GraphValidationError(
                "invalid_count", "constraint.value", "must be a non-negative integer"
            )
        upper = self.upper_value
        if operator is ComparisonOperator.BETWEEN:
            if isinstance(upper, bool) or not isinstance(upper, int) or upper < self.value:
                raise GraphValidationError(
                    "invalid_count",
                    "constraint.upper_value",
                    "BETWEEN requires an integer upper bound >= value",
                )
        elif upper is not None:
            raise GraphValidationError(
                "invalid_count",
                "constraint.upper_value",
                "is valid only for BETWEEN",
            )
        object.__setattr__(self, "operator", operator)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NumericConstraint:
        _strict_fields(
            value,
            allowed={"operator", "value", "upper_value"},
            required={"operator", "value"},
            path="constraint",
        )
        return cls(
            operator=value["operator"],
            value=value["value"],
            upper_value=value.get("upper_value"),
        )


@dataclass(frozen=True)
class EntityNode(JsonSerializable):
    id: str
    text: str
    name: str
    source_span: SourceSpan | Mapping[str, Any]
    entity_type: EntityType | str = EntityType.OBJECT
    referent_kind: ReferentKind | str = ReferentKind.INDIVIDUAL
    aliases: tuple[str, ...] = ()
    evaluation_route: EntityEvaluationRoute | str = (
        EntityEvaluationRoute.INVENTORY_EXISTENCE
    )
    grounding_mode: EntityGroundingMode | str = EntityGroundingMode.AUTO
    inventory_representation: EntityInventoryRepresentation | str = (
        EntityInventoryRepresentation.ACTOR_OR_DECLARED_ASSEMBLY
    )
    member_ids: tuple[str, ...] = ()
    node_type: GraphNodeType = field(default=GraphNodeType.ENTITY, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _graph_id(self.id, path="entity.id"))
        for field_name in ("text", "name"):
            text = str(getattr(self, field_name)).strip()
            if not text:
                raise GraphValidationError(
                    "empty_text", f"{self.id}.{field_name}", "must be non-empty"
                )
            object.__setattr__(self, field_name, text)
        object.__setattr__(self, "source_span", _coerce_source_span(self.source_span))
        object.__setattr__(
            self, "entity_type", _graph_enum(EntityType, self.entity_type, path=f"{self.id}.entity_type")
        )
        object.__setattr__(
            self,
            "referent_kind",
            _graph_enum(ReferentKind, self.referent_kind, path=f"{self.id}.referent_kind"),
        )
        object.__setattr__(
            self,
            "evaluation_route",
            _graph_enum(
                EntityEvaluationRoute,
                self.evaluation_route,
                path=f"{self.id}.evaluation_route",
            ),
        )
        object.__setattr__(
            self,
            "grounding_mode",
            _graph_enum(
                EntityGroundingMode,
                self.grounding_mode,
                path=f"{self.id}.grounding_mode",
            ),
        )
        object.__setattr__(
            self,
            "inventory_representation",
            _graph_enum(
                EntityInventoryRepresentation,
                self.inventory_representation,
                path=f"{self.id}.inventory_representation",
            ),
        )
        aliases = tuple(dict.fromkeys(str(value).strip() for value in self.aliases if str(value).strip()))
        object.__setattr__(self, "aliases", aliases)
        members = tuple(
            dict.fromkeys(
                _graph_id(value, path=f"{self.id}.member_ids")
                for value in self.member_ids
            )
        )
        if self.id in members:
            raise GraphValidationError(
                "self_member", f"{self.id}.member_ids", "an entity cannot contain itself"
            )
        object.__setattr__(self, "member_ids", members)

    @property
    def effective_grounding_mode(self) -> EntityGroundingMode:
        """Return an explicit mode for legacy bundles that predate the field."""

        if self.grounding_mode is not EntityGroundingMode.AUTO:
            return self.grounding_mode
        if self.entity_type is EntityType.REGION:
            return EntityGroundingMode.DERIVED_REGION
        if self.entity_type is EntityType.SURFACE:
            return EntityGroundingMode.ACTOR_COLLECTION
        if self.referent_kind is ReferentKind.COLLECTION:
            return EntityGroundingMode.ACTOR_COLLECTION
        return EntityGroundingMode.ACTOR

    def to_dict(self) -> dict[str, Any]:
        """Keep legacy entity JSON unchanged for the default Stage 1 route."""

        payload = JsonSerializable.to_dict(self)
        if self.evaluation_route is EntityEvaluationRoute.INVENTORY_EXISTENCE:
            payload.pop("evaluation_route", None)
        if self.grounding_mode is EntityGroundingMode.AUTO:
            payload.pop("grounding_mode", None)
        if (
            self.inventory_representation
            is EntityInventoryRepresentation.ACTOR_OR_DECLARED_ASSEMBLY
        ):
            payload.pop("inventory_representation", None)
        if not self.member_ids:
            payload.pop("member_ids", None)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str = "node") -> EntityNode:
        _strict_fields(
            value,
            allowed={
                "node_type", "id", "text", "name", "source_span", "entity_type",
                "referent_kind", "aliases", "evaluation_route",
                "grounding_mode",
                "inventory_representation",
                "member_ids",
            },
            required={"node_type", "id", "text", "name", "source_span"},
            path=path,
        )
        if value["node_type"] != GraphNodeType.ENTITY.value:
            raise GraphValidationError("invalid_tag", f"{path}.node_type", "expected entity")
        aliases = value.get("aliases", ())
        if not isinstance(aliases, list):
            raise GraphValidationError("invalid_field", f"{path}.aliases", "must be a list")
        return cls(
            id=value["id"],
            text=value["text"],
            name=value["name"],
            source_span=value["source_span"],
            entity_type=value.get("entity_type", EntityType.OBJECT.value),
            referent_kind=value.get("referent_kind", ReferentKind.INDIVIDUAL.value),
            aliases=tuple(aliases),
            evaluation_route=value.get(
                "evaluation_route",
                EntityEvaluationRoute.INVENTORY_EXISTENCE.value,
            ),
            grounding_mode=value.get(
                "grounding_mode",
                EntityGroundingMode.AUTO.value,
            ),
            inventory_representation=value.get(
                "inventory_representation",
                EntityInventoryRepresentation.ACTOR_OR_DECLARED_ASSEMBLY.value,
            ),
            member_ids=tuple(value.get("member_ids") or ()),
        )


@dataclass(frozen=True)
class PredicateNode(JsonSerializable):
    id: str
    text: str
    name: str
    predicate_type: PredicateType | str
    source_span: SourceSpan | Mapping[str, Any]
    polarity: Polarity | str = Polarity.AFFIRMATIVE
    constraint: NumericConstraint | Mapping[str, Any] | None = None
    semantic_parameters: Mapping[str, Any] = field(default_factory=dict)
    node_type: GraphNodeType = field(default=GraphNodeType.PREDICATE, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _graph_id(self.id, path="predicate.id"))
        for field_name in ("text", "name"):
            text = str(getattr(self, field_name)).strip()
            if not text:
                raise GraphValidationError(
                    "empty_text", f"{self.id}.{field_name}", "must be non-empty"
                )
            object.__setattr__(self, field_name, text)
        predicate_type = _graph_enum(
            PredicateType, self.predicate_type, path=f"{self.id}.predicate_type"
        )
        constraint = self.constraint
        if isinstance(constraint, Mapping):
            constraint = NumericConstraint.from_dict(constraint)
        elif constraint is not None and not isinstance(constraint, NumericConstraint):
            raise GraphValidationError(
                "invalid_constraint", f"{self.id}.constraint", "must be an object"
            )
        if predicate_type is PredicateType.COUNT and constraint is None:
            raise GraphValidationError(
                "missing_constraint", f"{self.id}.constraint", "COUNT requires a constraint"
            )
        if predicate_type is not PredicateType.COUNT and constraint is not None:
            raise GraphValidationError(
                "unexpected_constraint",
                f"{self.id}.constraint",
                "only COUNT may have a numeric constraint",
            )
        object.__setattr__(self, "source_span", _coerce_source_span(self.source_span))
        object.__setattr__(self, "predicate_type", predicate_type)
        object.__setattr__(
            self, "polarity", _graph_enum(Polarity, self.polarity, path=f"{self.id}.polarity")
        )
        object.__setattr__(self, "constraint", constraint)
        object.__setattr__(
            self,
            "semantic_parameters",
            _graph_json(dict(self.semantic_parameters), path=f"{self.id}.semantic_parameters"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str = "node") -> PredicateNode:
        _strict_fields(
            value,
            allowed={
                "node_type", "id", "text", "name", "predicate_type", "source_span",
                "polarity", "constraint",
                "semantic_parameters",
            },
            required={"node_type", "id", "text", "name", "predicate_type", "source_span"},
            path=path,
        )
        if value["node_type"] != GraphNodeType.PREDICATE.value:
            raise GraphValidationError("invalid_tag", f"{path}.node_type", "expected predicate")
        return cls(
            id=value["id"],
            text=value["text"],
            name=value["name"],
            predicate_type=value["predicate_type"],
            source_span=value["source_span"],
            polarity=value.get("polarity", Polarity.AFFIRMATIVE.value),
            constraint=value.get("constraint"),
            semantic_parameters=value.get("semantic_parameters") or {},
        )


@dataclass(frozen=True)
class RequirementNode(JsonSerializable):
    id: str
    text: str
    source_span: SourceSpan | Mapping[str, Any]
    aggregation: str = "weighted_sum"
    node_type: GraphNodeType = field(default=GraphNodeType.REQUIREMENT, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _graph_id(self.id, path="requirement.id"))
        text = str(self.text).strip()
        if not text:
            raise GraphValidationError("empty_text", f"{self.id}.text", "must be non-empty")
        if self.aggregation != "weighted_sum":
            raise GraphValidationError(
                "invalid_aggregation", f"{self.id}.aggregation", "schema 1.0 supports weighted_sum only"
            )
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "source_span", _coerce_source_span(self.source_span))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str = "node") -> RequirementNode:
        _strict_fields(
            value,
            allowed={"node_type", "id", "text", "source_span", "aggregation"},
            required={"node_type", "id", "text", "source_span"},
            path=path,
        )
        if value["node_type"] != GraphNodeType.REQUIREMENT.value:
            raise GraphValidationError("invalid_tag", f"{path}.node_type", "expected requirement")
        return cls(
            id=value["id"],
            text=value["text"],
            source_span=value["source_span"],
            aggregation=value.get("aggregation", "weighted_sum"),
        )


GraphNode = EntityNode | PredicateNode | RequirementNode


@dataclass(frozen=True)
class ArgumentEdge(JsonSerializable):
    source_id: str
    target_id: str
    role: str
    ordinal: int = 0
    edge_type: GraphEdgeType = field(default=GraphEdgeType.ARGUMENT, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _graph_id(self.source_id, path="argument.source_id"))
        object.__setattr__(self, "target_id", _graph_id(self.target_id, path="argument.target_id"))
        role = str(self.role).strip().casefold()
        if not role:
            raise GraphValidationError("empty_role", "argument.role", "must be non-empty")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise GraphValidationError("invalid_ordinal", "argument.ordinal", "must be non-negative")
        object.__setattr__(self, "role", role)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str = "edge") -> ArgumentEdge:
        _strict_fields(
            value,
            allowed={"edge_type", "source_id", "target_id", "role", "ordinal"},
            required={"edge_type", "source_id", "target_id", "role"},
            path=path,
        )
        return cls(
            source_id=value["source_id"], target_id=value["target_id"],
            role=value["role"], ordinal=value.get("ordinal", 0),
        )


@dataclass(frozen=True)
class ScopeEdge(JsonSerializable):
    source_id: str
    target_id: str
    edge_type: GraphEdgeType = field(default=GraphEdgeType.SCOPE, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _graph_id(self.source_id, path="scope.source_id"))
        object.__setattr__(self, "target_id", _graph_id(self.target_id, path="scope.target_id"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str = "edge") -> ScopeEdge:
        _strict_fields(
            value,
            allowed={"edge_type", "source_id", "target_id"},
            required={"edge_type", "source_id", "target_id"},
            path=path,
        )
        return cls(source_id=value["source_id"], target_id=value["target_id"])


@dataclass(frozen=True)
class RequirementMemberEdge(JsonSerializable):
    source_id: str
    target_id: str
    role: MemberRole | str
    weight_fraction: float | None = None
    edge_type: GraphEdgeType = field(default=GraphEdgeType.REQUIREMENT_MEMBER, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _graph_id(self.source_id, path="member.source_id"))
        object.__setattr__(self, "target_id", _graph_id(self.target_id, path="member.target_id"))
        role = _graph_enum(MemberRole, self.role, path="member.role")
        weight = self.weight_fraction
        if role is MemberRole.SCORED_FACET:
            if isinstance(weight, bool) or weight is None:
                raise GraphValidationError(
                    "missing_weight", "member.weight_fraction", "scored facets require a weight"
                )
            weight = float(weight)
            if not math.isfinite(weight) or weight <= 0.0:
                raise GraphValidationError(
                    "invalid_weight", "member.weight_fraction", "must be finite and positive"
                )
        elif weight is not None:
            raise GraphValidationError(
                "unexpected_weight", "member.weight_fraction", "support-only members cannot carry weight"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "weight_fraction", weight)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, path: str = "edge"
    ) -> RequirementMemberEdge:
        _strict_fields(
            value,
            allowed={"edge_type", "source_id", "target_id", "role", "weight_fraction"},
            required={"edge_type", "source_id", "target_id", "role"},
            path=path,
        )
        return cls(
            source_id=value["source_id"], target_id=value["target_id"],
            role=value["role"], weight_fraction=value.get("weight_fraction"),
        )


GraphEdge = ArgumentEdge | ScopeEdge | RequirementMemberEdge


@dataclass(frozen=True)
class RootRequirement(JsonSerializable):
    requirement_id: str
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requirement_id", _graph_id(self.requirement_id, path="root.requirement_id")
        )
        if isinstance(self.weight, bool):
            raise GraphValidationError("invalid_weight", "root.weight", "must be numeric")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise GraphValidationError("invalid_weight", "root.weight", "must be finite and positive")
        object.__setattr__(self, "weight", weight)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: str = "root") -> RootRequirement:
        _strict_fields(
            value,
            allowed={"requirement_id", "weight"},
            required={"requirement_id", "weight"},
            path=path,
        )
        return cls(requirement_id=value["requirement_id"], weight=value["weight"])


@dataclass(frozen=True)
class RequirementGraph(JsonSerializable):
    """Prompt-grounded graph with explicit dependency and scoring edges."""

    prompt: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    roots: tuple[RootRequirement, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        prompt = str(self.prompt)
        if not prompt.strip():
            raise GraphValidationError("empty_prompt", "prompt", "must be non-empty")
        if self.schema_version not in {"1.0", "2.0"}:
            raise GraphValidationError(
                "unsupported_version", "schema_version", f"unsupported {self.schema_version!r}"
            )
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        roots = tuple(self.roots)
        # A frozen task owns its complete requirement set. Multi-view image
        # authoring may produce many confirmed counts, attributes, and
        # relations; silently truncating them would turn valid observations
        # into unscored audit data. Runtime capture, batching, and timeouts own
        # workload limits, while this contract validates graph structure.
        if not nodes:
            raise GraphValidationError("invalid_size", "nodes", "must not be empty")
        if not roots or len(roots) > 16:
            raise GraphValidationError("invalid_size", "roots", "must contain 1..16 roots")
        if any(not isinstance(value, (EntityNode, PredicateNode, RequirementNode)) for value in nodes):
            raise GraphValidationError("invalid_node", "nodes", "contains an unsupported node")
        if any(
            not isinstance(value, (ArgumentEdge, ScopeEdge, RequirementMemberEdge))
            for value in edges
        ):
            raise GraphValidationError("invalid_edge", "edges", "contains an unsupported edge")
        if any(not isinstance(value, RootRequirement) for value in roots):
            raise GraphValidationError("invalid_root", "roots", "contains an unsupported root")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "roots", roots)
        self._validate()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RequirementGraph:
        if not isinstance(value, Mapping):
            raise GraphValidationError("invalid_graph", "$", "must be an object")
        _strict_fields(
            value,
            allowed={"schema_version", "prompt", "nodes", "edges", "roots"},
            required={"schema_version", "prompt", "nodes", "edges", "roots"},
            path="$",
        )
        raw_nodes = value["nodes"]
        raw_edges = value["edges"]
        raw_roots = value["roots"]
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list) or not isinstance(raw_roots, list):
            raise GraphValidationError("invalid_graph", "$", "nodes, edges, and roots must be lists")
        nodes: list[GraphNode] = []
        for index, item in enumerate(raw_nodes):
            path = f"nodes[{index}]"
            if not isinstance(item, Mapping):
                raise GraphValidationError("invalid_node", path, "must be an object")
            tag = item.get("node_type")
            if tag == GraphNodeType.ENTITY.value:
                nodes.append(EntityNode.from_dict(item, path=path))
            elif tag == GraphNodeType.PREDICATE.value:
                nodes.append(PredicateNode.from_dict(item, path=path))
            elif tag == GraphNodeType.REQUIREMENT.value:
                nodes.append(RequirementNode.from_dict(item, path=path))
            else:
                raise GraphValidationError("invalid_tag", f"{path}.node_type", f"unknown {tag!r}")
        edges: list[GraphEdge] = []
        for index, item in enumerate(raw_edges):
            path = f"edges[{index}]"
            if not isinstance(item, Mapping):
                raise GraphValidationError("invalid_edge", path, "must be an object")
            tag = item.get("edge_type")
            if tag == GraphEdgeType.ARGUMENT.value:
                edges.append(ArgumentEdge.from_dict(item, path=path))
            elif tag == GraphEdgeType.SCOPE.value:
                edges.append(ScopeEdge.from_dict(item, path=path))
            elif tag == GraphEdgeType.REQUIREMENT_MEMBER.value:
                edges.append(RequirementMemberEdge.from_dict(item, path=path))
            else:
                raise GraphValidationError("invalid_tag", f"{path}.edge_type", f"unknown {tag!r}")
        roots = []
        for index, item in enumerate(raw_roots):
            if not isinstance(item, Mapping):
                raise GraphValidationError("invalid_root", f"roots[{index}]", "must be an object")
            roots.append(RootRequirement.from_dict(item, path=f"roots[{index}]"))
        return cls(
            prompt=value["prompt"], nodes=tuple(nodes), edges=tuple(edges),
            roots=tuple(roots), schema_version=value["schema_version"],
        )

    def node(self, node_id: str) -> GraphNode:
        for value in self.nodes:
            if value.id == node_id:
                return value
        raise KeyError(node_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize optional routes without changing legacy default graphs.

        ``to_jsonable`` recursively visits dataclass fields and therefore does
        not call :meth:`EntityNode.to_dict` for nested nodes.  Apply the same
        omission here so schema-1.0 graphs that use the default route retain
        their original wire shape, while explicit Stage-2 routing remains
        visible and round-trippable.
        """

        payload = JsonSerializable.to_dict(self)
        raw_nodes = payload.get("nodes")
        if isinstance(raw_nodes, list):
            for node, raw_node in zip(self.nodes, raw_nodes, strict=True):
                if (
                    isinstance(node, EntityNode)
                    and node.evaluation_route
                    is EntityEvaluationRoute.INVENTORY_EXISTENCE
                    and isinstance(raw_node, dict)
                ):
                    raw_node.pop("evaluation_route", None)
                if (
                    isinstance(node, EntityNode)
                    and node.grounding_mode is EntityGroundingMode.AUTO
                    and isinstance(raw_node, dict)
                ):
                    raw_node.pop("grounding_mode", None)
                if (
                    isinstance(node, EntityNode)
                    and node.inventory_representation
                    is EntityInventoryRepresentation.ACTOR_OR_DECLARED_ASSEMBLY
                    and isinstance(raw_node, dict)
                ):
                    raw_node.pop("inventory_representation", None)
                if (
                    isinstance(node, EntityNode)
                    and not node.member_ids
                    and isinstance(raw_node, dict)
                ):
                    raw_node.pop("member_ids", None)
                if (
                    isinstance(node, PredicateNode)
                    and not node.semantic_parameters
                    and isinstance(raw_node, dict)
                ):
                    raw_node.pop("semantic_parameters", None)
        return payload

    def arguments_for(self, predicate_id: str) -> tuple[ArgumentEdge, ...]:
        return tuple(
            sorted(
                (edge for edge in self.edges if isinstance(edge, ArgumentEdge) and edge.source_id == predicate_id),
                key=lambda edge: (edge.ordinal, edge.role, edge.target_id),
            )
        )

    def members_for(self, requirement_id: str) -> tuple[RequirementMemberEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if isinstance(edge, RequirementMemberEdge) and edge.source_id == requirement_id
        )

    def effective_weights(self, root_id: str | None = None) -> dict[str, float]:
        """Return leaf scoring budgets; support-only dependencies never enter it."""

        roots = tuple(
            root for root in self.roots if root_id is None or root.requirement_id == root_id
        )
        if root_id is not None and not roots:
            raise KeyError(root_id)
        weights: dict[str, float] = {}

        def visit(requirement_id: str, budget: float) -> None:
            for edge in self.members_for(requirement_id):
                if edge.role is MemberRole.SUPPORT_ONLY:
                    continue
                member_budget = budget * float(edge.weight_fraction)
                target = self.node(edge.target_id)
                if isinstance(target, RequirementNode):
                    visit(target.id, member_budget)
                else:
                    weights[target.id] = weights.get(target.id, 0.0) + member_budget

        for root in roots:
            visit(root.requirement_id, root.weight)
        return weights

    def _validate(self) -> None:
        by_id: dict[str, GraphNode] = {}
        for index, node in enumerate(self.nodes):
            if node.id in by_id:
                raise GraphValidationError("duplicate_id", f"nodes[{index}].id", node.id)
            by_id[node.id] = node
            span = node.source_span
            if span.end > len(self.prompt):
                raise GraphValidationError(
                    "span_out_of_bounds", f"{node.id}.source_span", "extends beyond prompt"
                )
            grounded = self.prompt[span.start : span.end]
            if not grounded.strip():
                raise GraphValidationError(
                    "empty_grounding", f"{node.id}.source_span", "selects no prompt text"
                )
            if " ".join(grounded.split()).casefold() != " ".join(node.text.split()).casefold():
                raise GraphValidationError(
                    "text_span_mismatch",
                    f"{node.id}.source_span",
                    f"selects {grounded!r}, not node text {node.text!r}",
                )

        root_ids = [root.requirement_id for root in self.roots]
        if len(root_ids) != len(set(root_ids)):
            raise GraphValidationError("duplicate_root", "roots", "requirement ids must be unique")
        if not math.isclose(sum(root.weight for root in self.roots), 1.0, abs_tol=1e-9):
            raise GraphValidationError("root_weight_sum", "roots", "weights must sum to 1")

        for entity in (node for node in self.nodes if isinstance(node, EntityNode)):
            missing_members = sorted(set(entity.member_ids) - set(by_id))
            if missing_members:
                raise GraphValidationError(
                    "dangling_member",
                    f"{entity.id}.member_ids",
                    f"missing entities {missing_members!r}",
                )
            invalid_members = sorted(
                member_id
                for member_id in entity.member_ids
                if not isinstance(by_id[member_id], EntityNode)
            )
            if invalid_members:
                raise GraphValidationError(
                    "invalid_member",
                    f"{entity.id}.member_ids",
                    f"members are not entities {invalid_members!r}",
                )

        # Stage-0 v2 stores prompt-semantic links inside predicate payloads in
        # addition to the graph's structural edges. They must obey the same
        # referential-integrity rules; otherwise a frozen bundle can validate
        # here and fail much later while Stage 2 builds the VLM claim payload.
        for predicate in (
            node for node in self.nodes if isinstance(node, PredicateNode)
        ):
            semantic = predicate.semantic_parameters
            if str(semantic.get("stage0_ir_version") or "") != "2.0":
                continue
            scope = semantic.get("scope")
            if isinstance(scope, Mapping) and scope.get("entity_id") is not None:
                entity_id = str(scope["entity_id"])
                target = by_id.get(entity_id)
                if target is None:
                    raise GraphValidationError(
                        "dangling_semantic_reference",
                        f"{predicate.id}.semantic_parameters.scope.entity_id",
                        f"missing entity {entity_id!r}",
                    )
                if not isinstance(target, EntityNode):
                    raise GraphValidationError(
                        "invalid_semantic_reference",
                        f"{predicate.id}.semantic_parameters.scope.entity_id",
                        f"target {entity_id!r} is not an entity",
                    )
            logic = semantic.get("logic")
            if isinstance(logic, Mapping):
                for index, requirement_id_value in enumerate(
                    logic.get("requirement_ids") or ()
                ):
                    requirement_id = str(requirement_id_value)
                    target = by_id.get(requirement_id)
                    path = (
                        f"{predicate.id}.semantic_parameters.logic."
                        f"requirement_ids[{index}]"
                    )
                    if target is None:
                        raise GraphValidationError(
                            "dangling_semantic_reference",
                            path,
                            f"missing predicate {requirement_id!r}",
                        )
                    if not isinstance(target, PredicateNode):
                        raise GraphValidationError(
                            "invalid_semantic_reference",
                            path,
                            f"target {requirement_id!r} is not a predicate",
                        )

        argument_edges = tuple(edge for edge in self.edges if isinstance(edge, ArgumentEdge))
        scope_edges = tuple(edge for edge in self.edges if isinstance(edge, ScopeEdge))
        member_edges = tuple(edge for edge in self.edges if isinstance(edge, RequirementMemberEdge))
        edge_keys: set[tuple[Any, ...]] = set()
        for index, edge in enumerate(self.edges):
            key = tuple(to_jsonable(edge).items())
            if key in edge_keys:
                raise GraphValidationError("duplicate_edge", f"edges[{index}]", "duplicate")
            edge_keys.add(key)
            source = by_id.get(edge.source_id)
            target = by_id.get(edge.target_id)
            if source is None or target is None:
                raise GraphValidationError("dangling_edge", f"edges[{index}]", "endpoint is missing")
            if isinstance(edge, ArgumentEdge) and not (
                isinstance(source, PredicateNode) and isinstance(target, EntityNode)
            ):
                raise GraphValidationError(
                    "invalid_endpoint", f"edges[{index}]", "argument must be predicate -> entity"
                )
            if isinstance(edge, ScopeEdge) and not (
                isinstance(source, PredicateNode) and isinstance(target, PredicateNode)
            ):
                raise GraphValidationError(
                    "invalid_endpoint", f"edges[{index}]", "scope must be predicate -> predicate"
                )
            if isinstance(edge, RequirementMemberEdge) and not isinstance(source, RequirementNode):
                raise GraphValidationError(
                    "invalid_endpoint", f"edges[{index}]", "member source must be a requirement"
                )

        incoming_members = {edge.target_id for edge in member_edges}
        for index, root in enumerate(self.roots):
            node = by_id.get(root.requirement_id)
            if not isinstance(node, RequirementNode):
                raise GraphValidationError(
                    "invalid_root", f"roots[{index}]", "must reference a requirement node"
                )
            if root.requirement_id in incoming_members:
                raise GraphValidationError(
                    "nested_root", f"roots[{index}]", "a root cannot also be a member"
                )

        # Requirement-local budgets and dependency classification.
        for requirement in (node for node in self.nodes if isinstance(node, RequirementNode)):
            members = [edge for edge in member_edges if edge.source_id == requirement.id]
            scored = [edge for edge in members if edge.role is MemberRole.SCORED_FACET]
            if not scored:
                raise GraphValidationError(
                    "missing_scored_facet", requirement.id, "requires at least one scored member"
                )
            if not math.isclose(
                sum(float(edge.weight_fraction) for edge in scored), 1.0, abs_tol=1e-9
            ):
                raise GraphValidationError(
                    "facet_weight_sum", requirement.id, "scored weights must sum to 1"
                )
            member_ids: set[str] = set()
            for edge in members:
                if edge.target_id in member_ids:
                    raise GraphValidationError(
                        "duplicate_member", requirement.id, f"member {edge.target_id!r} classified twice"
                    )
                member_ids.add(edge.target_id)
            for edge in members:
                target = by_id[edge.target_id]
                if not isinstance(target, PredicateNode):
                    continue
                dependencies = {
                    item.target_id for item in argument_edges if item.source_id == target.id
                } | {item.target_id for item in scope_edges if item.source_id == target.id}
                missing = sorted(dependencies - member_ids)
                if missing:
                    raise GraphValidationError(
                        "unclassified_dependency",
                        requirement.id,
                        f"predicate {target.id!r} dependencies are not members: {missing!r}",
                    )

        # Predicate arity and scoped count semantics.
        for predicate in (node for node in self.nodes if isinstance(node, PredicateNode)):
            arguments = [edge for edge in argument_edges if edge.source_id == predicate.id]
            roles = [edge.role for edge in arguments]
            scopes = [edge for edge in scope_edges if edge.source_id == predicate.id]
            if predicate.predicate_type is PredicateType.COUNT:
                if roles != ["collection"] and sorted(roles) != ["collection"]:
                    raise GraphValidationError(
                        "invalid_arguments", predicate.id, "COUNT requires exactly one collection argument"
                    )
                if len(scopes) > 1:
                    raise GraphValidationError(
                        "invalid_scope", predicate.id, "COUNT accepts at most one scope predicate"
                    )
                if scopes:
                    collection_id = arguments[0].target_id
                    scoped_dependencies = {
                        edge.target_id
                        for edge in argument_edges
                        if edge.source_id == scopes[0].target_id
                    }
                    if collection_id not in scoped_dependencies:
                        raise GraphValidationError(
                            "invalid_scope",
                            predicate.id,
                            "counted collection must participate in its scope predicate",
                        )
            elif self.schema_version == "1.0" and scopes and predicate.predicate_type in {
                PredicateType.EXISTENCE,
                PredicateType.ATTRIBUTE,
                PredicateType.MATERIAL,
                PredicateType.SPATIAL_RELATION,
                PredicateType.ATMOSPHERE,
                PredicateType.SCENE_IDENTITY,
            }:
                raise GraphValidationError(
                    "invalid_scope", predicate.id, "only COUNT predicates may own a scope edge"
                )
            elif self.schema_version == "1.0" and predicate.predicate_type in {
                PredicateType.EXISTENCE, PredicateType.ATTRIBUTE, PredicateType.MATERIAL,
            }:
                if roles != ["subject"] and sorted(roles) != ["subject"]:
                    raise GraphValidationError(
                        "invalid_arguments", predicate.id, "requires exactly one subject"
                    )
            elif (
                self.schema_version == "1.0"
                and predicate.predicate_type is PredicateType.SPATIAL_RELATION
            ):
                if sorted(roles) != ["reference", "subject"]:
                    raise GraphValidationError(
                        "invalid_arguments", predicate.id, "requires one subject and one reference"
                    )
            elif self.schema_version == "1.0" and predicate.predicate_type in {
                PredicateType.ATMOSPHERE,
                PredicateType.SCENE_IDENTITY,
            } and arguments:
                raise GraphValidationError(
                    "invalid_arguments", predicate.id, "scene/atmosphere predicates take no entity arguments"
                )

        # Directed cycles in predicate scopes or nested requirements.
        def reject_cycles(pairs: Sequence[tuple[str, str]], *, path: str) -> None:
            adjacency: dict[str, list[str]] = {}
            for source, target in pairs:
                adjacency.setdefault(source, []).append(target)
            visiting: set[str] = set()
            visited: set[str] = set()

            def visit(node_id: str) -> None:
                if node_id in visiting:
                    raise GraphValidationError("cycle", path, f"cycle through {node_id!r}")
                if node_id in visited:
                    return
                visiting.add(node_id)
                for target_id in adjacency.get(node_id, ()):
                    visit(target_id)
                visiting.remove(node_id)
                visited.add(node_id)

            for node_id in tuple(adjacency):
                visit(node_id)

        reject_cycles(
            [(edge.source_id, edge.target_id) for edge in scope_edges], path="scope_edges"
        )
        reject_cycles(
            [
                (edge.source_id, edge.target_id)
                for edge in member_edges
                if isinstance(by_id[edge.target_id], RequirementNode)
            ],
            path="requirement_members",
        )

        # Every declared node must be grounded in some root's dependency closure.
        reachable: set[str] = set()

        def mark(node_id: str) -> None:
            if node_id in reachable:
                return
            reachable.add(node_id)
            node = by_id[node_id]
            if isinstance(node, RequirementNode):
                for edge in member_edges:
                    if edge.source_id == node_id:
                        mark(edge.target_id)
            elif isinstance(node, PredicateNode):
                for edge in (*argument_edges, *scope_edges):
                    if edge.source_id == node_id:
                        mark(edge.target_id)

        for root in self.roots:
            mark(root.requirement_id)
        orphans = sorted(set(by_id) - reachable)
        if orphans:
            raise GraphValidationError("orphan_node", "nodes", f"unreachable {orphans!r}")

        # A single explicit root cannot reach one scored leaf through two paths.
        def scored_leaves_for(root_id: str) -> set[str]:
            scored_leaves: set[str] = set()

            def walk_scored(requirement_id: str) -> None:
                for edge in self.members_for(requirement_id):
                    if edge.role is MemberRole.SUPPORT_ONLY:
                        continue
                    target = by_id[edge.target_id]
                    if isinstance(target, RequirementNode):
                        walk_scored(target.id)
                    elif target.id in scored_leaves:
                        raise GraphValidationError(
                            "duplicate_scored_path", root_id, target.id
                        )
                    else:
                        scored_leaves.add(target.id)

            walk_scored(root_id)
            return scored_leaves

        scored_owner: dict[str, str] = {}
        for root in self.roots:
            for leaf_id in scored_leaves_for(root.requirement_id):
                previous_root = scored_owner.get(leaf_id)
                if previous_root is not None:
                    raise GraphValidationError(
                        "duplicate_scored_path",
                        "roots",
                        (
                            f"leaf {leaf_id!r} is scored by both "
                            f"{previous_root!r} and {root.requirement_id!r}"
                        ),
                    )
                scored_owner[leaf_id] = root.requirement_id


__all__ = [
    "ArgumentEdge",
    "CameraPose",
    "ClaimAssessment",
    "ClaimType",
    "ClaimVerdict",
    "ComparisonOperator",
    "EntityEvaluationRoute",
    "EntityGroundingMode",
    "EntityInventoryRepresentation",
    "EntityNode",
    "EntityType",
    "GraphEdge",
    "GraphEdgeType",
    "GraphNode",
    "GraphNodeType",
    "GraphValidationError",
    "JsonSerializable",
    "MemberRole",
    "NumericConstraint",
    "Polarity",
    "PredicateNode",
    "PredicateType",
    "PromptClaim",
    "ReferentKind",
    "RequirementGraph",
    "RequirementMemberEdge",
    "RequirementNode",
    "RootRequirement",
    "SceneBounds",
    "SceneEvaluationResult",
    "SceneMemory",
    "SceneObservation",
    "ScopeEdge",
    "ScoreSummary",
    "SourceSpan",
    "coerce_claim_type",
    "default_claim_weight",
    "to_jsonable",
]
