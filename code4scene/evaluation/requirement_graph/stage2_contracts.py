"""Serializable, provider-agnostic contracts for RequirementGraph Stage 2.

The types in this module deliberately keep acquisition metadata separate from
visual decisions.  RGB frame storage and the metadata-free judge projection
live in :mod:`stage2_frames`; this module owns tasks, budgets, capture plans,
coverage, and tri-state semantic results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .contracts import CameraPose, JsonSerializable, Polarity


class Stage2TaskKind(str, Enum):
    """The visual operation required for one scored graph leaf."""

    OBJECT_EXISTENCE = "object_existence"
    ATTRIBUTE = "attribute"
    MATERIAL = "material"
    SPATIAL_RELATION = "spatial_relation"
    COUNT = "count"
    ATMOSPHERE = "atmosphere"
    SCENE_IDENTITY = "scene_identity"
    VISUAL_GLOBAL = "visual_global_vlm"

    @classmethod
    def coerce(cls, value: Stage2TaskKind | str) -> Stage2TaskKind:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().casefold())


GLOBAL_CONTEXT_TASK_KINDS = frozenset(
    {
        Stage2TaskKind.ATMOSPHERE,
        Stage2TaskKind.SCENE_IDENTITY,
        Stage2TaskKind.VISUAL_GLOBAL,
    }
)

VISUAL_FEATURE_ARGUMENT_ROLES = frozenset(
    {
        "appearance",
        "attribute",
        "color",
        "detail",
        "feature",
        "lighting",
        "material",
        "style",
        "surface",
    }
)


@dataclass(frozen=True, slots=True)
class Stage2TaskArgument(JsonSerializable):
    """One entity needed to localize or frame a Stage 2 task.

    ``source_predicate_id`` identifies dependencies inherited through a COUNT
    scope predicate.  It is controller metadata and must never cross the VLM
    boundary.
    """

    entity_id: str
    role: str
    ordinal: int = 0
    source_predicate_id: str | None = None

    def __post_init__(self) -> None:
        entity_id = str(self.entity_id).strip()
        role = str(self.role).strip().casefold()
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        if not role:
            raise ValueError("role must be non-empty")
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("ordinal must be a non-negative integer")
        source = (
            str(self.source_predicate_id).strip()
            if self.source_predicate_id is not None
            else None
        )
        if source == "":
            raise ValueError("source_predicate_id must be non-empty or None")
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "source_predicate_id", source)


@dataclass(frozen=True, slots=True)
class Stage2Task(JsonSerializable):
    """One active, independently scored RequirementGraph visual leaf."""

    task_id: str
    node_id: str
    kind: Stage2TaskKind | str
    weight: float
    polarity: Polarity | str = Polarity.AFFIRMATIVE
    arguments: tuple[Stage2TaskArgument, ...] = ()

    def __post_init__(self) -> None:
        task_id = str(self.task_id).strip()
        node_id = str(self.node_id).strip()
        if not task_id or not node_id:
            raise ValueError("task_id and node_id must be non-empty")
        try:
            weight = float(self.weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("weight must be numeric") from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("weight must be finite and positive")
        kind = Stage2TaskKind.coerce(self.kind)
        polarity = (
            self.polarity
            if isinstance(self.polarity, Polarity)
            else Polarity(str(self.polarity).strip().casefold())
        )
        arguments = tuple(self.arguments)
        if any(not isinstance(value, Stage2TaskArgument) for value in arguments):
            raise TypeError("arguments must contain Stage2TaskArgument values")
        argument_keys = [
            (
                value.entity_id,
                value.role,
                value.ordinal,
                value.source_predicate_id,
            )
            for value in arguments
        ]
        if len(argument_keys) != len(set(argument_keys)):
            raise ValueError("task arguments must be unique")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "polarity", polarity)
        object.__setattr__(self, "arguments", arguments)

    @property
    def dependency_entity_ids(self) -> tuple[str, ...]:
        """Entity dependencies in deterministic first-use order."""

        return tuple(dict.fromkeys(value.entity_id for value in self.arguments))

    @property
    def visual_feature_entity_ids(self) -> tuple[str, ...]:
        """Feature-only dependencies that may use requirement-owned overviews."""

        return tuple(
            dict.fromkeys(
                value.entity_id
                for value in self.arguments
                if value.role in VISUAL_FEATURE_ARGUMENT_ROLES
            )
        )

    @property
    def allows_overview_fallback(self) -> bool:
        """Whether overview RGB is legal evidence for this visual task."""

        return bool(self.visual_feature_entity_ids) or self.kind in GLOBAL_CONTEXT_TASK_KINDS


@dataclass(frozen=True, slots=True)
class Stage2Budget(JsonSerializable):
    """Acquisition profile and finite per-requirement evidence limits.

    Values may be reduced for tests or constrained runs, but cannot exceed the
    safety caps in the Stage 2 contract.  ``global_capture_limits_enabled``
    preserves the bounded profile for direct/constrained Stage 2 callers.  The
    production RequirementGraph profile disables only the scene-wide aggregate
    cutoff: every entity group and evidence shape remains locally bounded, so
    the graph creates a finite plan without making later requirements compete
    for first-N screenshot slots.
    """

    overview_frames: int = 4
    max_overview_frames: int = 6
    max_targeted_frames: int = 12
    max_recovery_frames: int = 2
    max_valid_frames: int = 18
    max_evidence_frames_per_task: int = 4
    max_individual_representatives: int = 2
    max_collection_representatives: int = 3
    max_count_anchors: int = 8
    max_relation_joint_views: int = 3
    max_targeted_views_per_entity_group: int = 2
    global_capture_limits_enabled: bool = True

    @classmethod
    def graph_default(cls) -> Stage2Budget:
        """Return the complete production RequirementGraph profile.

        Six varied overviews cover whole-scene claims.  Each localizable entity
        group, collection, relation, and unlocalized fallback then receives its
        own finite capture shape.  There is deliberately no aggregate frame cap
        tied to graph order; constrained callers can still use ``Stage2Budget``
        directly or provide an explicit YAML budget.
        """

        return cls(
            overview_frames=6,
            max_overview_frames=6,
            max_evidence_frames_per_task=8,
            max_targeted_views_per_entity_group=2,
            global_capture_limits_enabled=False,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.global_capture_limits_enabled, bool):
            raise TypeError("global_capture_limits_enabled must be a bool")
        hard_caps = {
            "overview_frames": 6,
            "max_overview_frames": 6,
            "max_targeted_frames": 12,
            "max_recovery_frames": 2,
            "max_valid_frames": 18,
            "max_evidence_frames_per_task": 12,
            "max_individual_representatives": 2,
            "max_collection_representatives": 3,
            "max_count_anchors": 8,
            "max_relation_joint_views": 3,
            "max_targeted_views_per_entity_group": 2,
        }
        positive_fields = {
            "max_valid_frames",
            "max_evidence_frames_per_task",
            "max_individual_representatives",
            "max_collection_representatives",
            "max_count_anchors",
            "max_relation_joint_views",
            "max_targeted_views_per_entity_group",
        }
        for name, hard_cap in hard_caps.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            minimum = 1 if name in positive_fields else 0
            if not minimum <= value <= hard_cap:
                raise ValueError(f"{name} must be between {minimum} and {hard_cap}")
        if self.overview_frames > self.max_overview_frames:
            raise ValueError("overview_frames cannot exceed max_overview_frames")
        if self.max_overview_frames > self.max_valid_frames:
            raise ValueError("max_overview_frames cannot exceed max_valid_frames")
        if self.max_evidence_frames_per_task > self.max_valid_frames:
            raise ValueError(
                "max_evidence_frames_per_task cannot exceed max_valid_frames"
            )


class CaptureShotRole(str, Enum):
    """Predicate-aware camera role; geometry only, never semantic evidence."""

    CONTEXT = "context"
    CLOSE = "close"
    DETAIL = "detail"
    OBLIQUE = "oblique"
    JOINT = "joint"
    COLLECTION_WIDE = "collection_wide"
    OVERVIEW = "overview"
    GRID = "grid"
    RECOVERY = "recovery"

    @classmethod
    def coerce(cls, value: CaptureShotRole | str) -> CaptureShotRole:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().casefold())


@dataclass(frozen=True, slots=True)
class CaptureRequest(JsonSerializable):
    """One controller-only world-camera request."""

    request_id: str
    task_ids: tuple[str, ...]
    shot_role: CaptureShotRole | str
    pose: CameraPose
    actor_ids: tuple[str, ...] = ()
    representative_index: int = 0
    recovery_step: int = 0

    def __post_init__(self) -> None:
        request_id = str(self.request_id).strip()
        if not request_id:
            raise ValueError("request_id must be non-empty")
        task_ids = _unique_nonempty_strings(self.task_ids, name="task_ids")
        if not task_ids:
            raise ValueError("task_ids must be non-empty")
        actor_ids = _unique_nonempty_strings(self.actor_ids, name="actor_ids")
        if not isinstance(self.pose, CameraPose):
            raise TypeError("pose must be a CameraPose")
        for name in ("representative_index", "recovery_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "shot_role", CaptureShotRole.coerce(self.shot_role))
        object.__setattr__(self, "actor_ids", actor_ids)


@dataclass(frozen=True, slots=True)
class CaptureProgram(JsonSerializable):
    """A shareable group of related capture requests."""

    program_id: str
    task_ids: tuple[str, ...]
    requests: tuple[CaptureRequest, ...] = ()

    def __post_init__(self) -> None:
        program_id = str(self.program_id).strip()
        if not program_id:
            raise ValueError("program_id must be non-empty")
        task_ids = _unique_nonempty_strings(self.task_ids, name="task_ids")
        if not task_ids:
            raise ValueError("task_ids must be non-empty")
        requests = tuple(self.requests)
        if any(not isinstance(value, CaptureRequest) for value in requests):
            raise TypeError("requests must contain CaptureRequest values")
        request_ids = [value.request_id for value in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("capture request ids must be unique within a program")
        allowed = set(task_ids)
        if any(not set(value.task_ids).issubset(allowed) for value in requests):
            raise ValueError("capture request task_ids must belong to the program")
        object.__setattr__(self, "program_id", program_id)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "requests", requests)


@dataclass(frozen=True, slots=True)
class Stage2CapturePlan(JsonSerializable):
    programs: tuple[CaptureProgram, ...]
    budget: Stage2Budget = field(default_factory=Stage2Budget)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        programs = tuple(self.programs)
        if any(not isinstance(value, CaptureProgram) for value in programs):
            raise TypeError("programs must contain CaptureProgram values")
        if not isinstance(self.budget, Stage2Budget):
            raise TypeError("budget must be a Stage2Budget")
        program_ids = [value.program_id for value in programs]
        if len(program_ids) != len(set(program_ids)):
            raise ValueError("capture program ids must be unique")
        request_ids = [
            request.request_id for program in programs for request in program.requests
        ]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("capture request ids must be globally unique")
        if self.schema_version != "1.0":
            raise ValueError("unsupported Stage 2 capture-plan schema version")
        object.__setattr__(self, "programs", programs)

    @property
    def requests(self) -> tuple[CaptureRequest, ...]:
        return tuple(request for value in self.programs for request in value.requests)


@dataclass(frozen=True, slots=True)
class SearchCoverage(JsonSerializable):
    """Controller-computed capture coverage; never supplied by the VLM."""

    valid_target_views: int = 0
    distinct_target_actors: int = 0
    complementary_overviews: int = 0
    has_joint_view: bool = False
    has_collection_wide_view: bool = False
    has_detail_view: bool = False
    exhaustive_existence_search: bool = False

    def __post_init__(self) -> None:
        for name in (
            "valid_target_views",
            "distinct_target_actors",
            "complementary_overviews",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "has_joint_view",
            "has_collection_wide_view",
            "has_detail_view",
            "exhaustive_existence_search",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


# CaptureCoverage is the capture-controller spelling of the same immutable
# contract.  Keeping one concrete type prevents coverage fields from drifting.
CaptureCoverage = SearchCoverage


@dataclass(frozen=True, slots=True)
class VisualGrounding(JsonSerializable):
    """Untrusted RGB grounding observations returned with a model decision."""

    subject_confirmed: bool = False
    participants_confirmed: bool = False
    relation_scope_covered: bool = False
    collection_complete: bool = False
    instances_countable: bool = False
    visible_instance_count: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "subject_confirmed",
            "participants_confirmed",
            "relation_scope_covered",
            "collection_complete",
            "instances_countable",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.visible_instance_count is not None and (
            isinstance(self.visible_instance_count, bool)
            or not isinstance(self.visible_instance_count, int)
            or self.visible_instance_count < 0
        ):
            raise ValueError("visible_instance_count must be non-negative or None")
        if self.visible_instance_count is not None and not self.instances_countable:
            raise ValueError("visible_instance_count requires instances_countable=True")


class Stage2Verdict(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def coerce(cls, value: Stage2Verdict | str) -> Stage2Verdict:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper()
        # Legacy partial support is not a Stage 2 verdict and is fail-closed.
        if normalized == "PARTIAL_MATCH":
            return cls.UNKNOWN
        return cls(normalized)


class Stage2EvidenceBasis(str, Enum):
    SUPPORT = "support"
    VISIBLE_CONTRADICTION = "visible_contradiction"
    ATTRIBUTE_CONTRADICTION = "attribute_contradiction"
    MATERIAL_CONTRADICTION = "material_contradiction"
    SUFFICIENT_RELATION_VIEW = "sufficient_relation_view"
    SUFFICIENT_COUNT_VIEW = "sufficient_count_view"
    SCENE_CONTRADICTION = "scene_contradiction"
    SEARCHED_ABSENCE = "searched_absence"
    INSUFFICIENT = "insufficient"

    @classmethod
    def coerce(cls, value: Stage2EvidenceBasis | str) -> Stage2EvidenceBasis:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().casefold())


@dataclass(frozen=True, slots=True)
class Stage2Assessment(JsonSerializable):
    """Controller-finalized tri-state result for one active task."""

    task_id: str
    node_id: str
    verdict: Stage2Verdict | str
    confidence: float = 0.0
    evidence_frame_ids: tuple[str, ...] = ()
    rationale: str = ""
    evidence_basis: Stage2EvidenceBasis | str = Stage2EvidenceBasis.INSUFFICIENT
    coverage: SearchCoverage = field(default_factory=SearchCoverage)
    grounding: VisualGrounding = field(default_factory=VisualGrounding)
    unknown_reason: str | None = None
    mismatch_downgraded: bool = False

    def __post_init__(self) -> None:
        task_id = str(self.task_id).strip()
        node_id = str(self.node_id).strip()
        if not task_id or not node_id:
            raise ValueError("task_id and node_id must be non-empty")
        verdict = Stage2Verdict.coerce(self.verdict)
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("confidence must be numeric") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        evidence = _unique_nonempty_strings(
            self.evidence_frame_ids, name="evidence_frame_ids"
        )
        if verdict is not Stage2Verdict.UNKNOWN and not evidence:
            raise ValueError("resolved Stage 2 assessments must cite RGB evidence")
        if not isinstance(self.coverage, SearchCoverage):
            raise TypeError("coverage must be SearchCoverage")
        if not isinstance(self.grounding, VisualGrounding):
            raise TypeError("grounding must be VisualGrounding")
        if not isinstance(self.mismatch_downgraded, bool):
            raise TypeError("mismatch_downgraded must be a bool")
        unknown_reason = (
            str(self.unknown_reason).strip()
            if self.unknown_reason is not None
            else None
        )
        if unknown_reason == "":
            unknown_reason = None
        if verdict is not Stage2Verdict.UNKNOWN and unknown_reason is not None:
            raise ValueError("only UNKNOWN assessments may have unknown_reason")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_frame_ids", evidence)
        object.__setattr__(self, "rationale", str(self.rationale).strip())
        object.__setattr__(
            self, "evidence_basis", Stage2EvidenceBasis.coerce(self.evidence_basis)
        )
        object.__setattr__(self, "unknown_reason", unknown_reason)


@dataclass(frozen=True, slots=True)
class Stage2Result(JsonSerializable):
    assessments: tuple[Stage2Assessment, ...]
    evaluation_error: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        assessments = tuple(self.assessments)
        if any(not isinstance(value, Stage2Assessment) for value in assessments):
            raise TypeError("assessments must contain Stage2Assessment values")
        task_ids = [value.task_id for value in assessments]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Stage 2 assessment task ids must be unique")
        error = (
            str(self.evaluation_error).strip()
            if self.evaluation_error is not None
            else None
        )
        if error == "":
            error = None
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        if self.schema_version != "1.0":
            raise ValueError("unsupported Stage 2 result schema version")
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(self, "evaluation_error", error)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def for_task(self, task_id: str) -> Stage2Assessment | None:
        key = str(task_id).strip()
        return next((value for value in self.assessments if value.task_id == key), None)


def _unique_nonempty_strings(values: Any, *, name: str) -> tuple[str, ...]:
    try:
        normalized = tuple(str(value).strip() for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of strings") from exc
    if any(not value for value in normalized):
        raise ValueError(f"{name} must contain only non-empty values")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


__all__ = [
    "CaptureCoverage",
    "CaptureProgram",
    "CaptureRequest",
    "CaptureShotRole",
    "GLOBAL_CONTEXT_TASK_KINDS",
    "VISUAL_FEATURE_ARGUMENT_ROLES",
    "SearchCoverage",
    "Stage2Assessment",
    "Stage2Budget",
    "Stage2CapturePlan",
    "Stage2EvidenceBasis",
    "Stage2Result",
    "Stage2Task",
    "Stage2TaskArgument",
    "Stage2TaskKind",
    "Stage2Verdict",
    "VisualGrounding",
]
