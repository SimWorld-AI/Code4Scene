"""Strict, provider-independent contracts for RequirementGraph Stage 3.

Stage 3 has two deliberately separate products: controller-side exploration
and RGB-only semantic judgement.  This module keeps both products finite and
JSON serializable while leaving RGB ownership to :class:`FrameStore`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .contracts import CameraPose, JsonSerializable
from .stage2_contracts import CaptureShotRole

_STAGE3_FRAME_ID = re.compile(r"\As3f_[0-9]{6}\Z", re.ASCII)


def _text(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    result = str(value).strip()
    if not result:
        if optional:
            return None
        raise ValueError(f"{name} must be non-empty")
    return result


def _probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return result


def _unique_texts(values: Sequence[Any], name: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if any(not value for value in result):
        raise ValueError(f"{name} must contain only non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


class _CoercibleEnum(str, Enum):
    @classmethod
    def coerce(cls, value: Any):
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().casefold())


@dataclass(frozen=True, slots=True)
class Stage3Budget(JsonSerializable):
    """Finite exploration profile, optionally derived from graph demand."""

    max_seed_frames: int = 10
    max_new_capture_attempts: int = 4
    max_valid_exploration_frames: int = 20
    min_portfolio_frames: int = 4
    max_portfolio_frames: int = 6
    max_recovery_attempts: int = 2

    @classmethod
    def for_requirement_graph(
        cls,
        *,
        seed_frame_count: int,
        focus_actor_count: int,
        grid_task_count: int,
        require_global_context: bool,
    ) -> Stage3Budget:
        """Size exploration from available evidence instead of fixed totals.

        Every Stage 2 frame may be inherited.  Missing actor/grid targets get
        one fresh attempt, global claims reserve a six-view scout shape, and
        one additional slot per focus actor remains available for adaptive
        reframing.  The portfolio is intentionally smaller than the store:
        claim-specific routing adds targeted frames before the shared global
        portfolio, while model requests remain independently batch-bounded.
        """

        counts = {
            "seed_frame_count": seed_frame_count,
            "focus_actor_count": focus_actor_count,
            "grid_task_count": grid_task_count,
        }
        for name, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(require_global_context, bool):
            raise TypeError("require_global_context must be a bool")
        global_attempts = 6 if require_global_context else 0
        new_attempts = focus_actor_count + grid_task_count + global_attempts
        valid_frames = max(
            4,
            seed_frame_count + new_attempts + focus_actor_count,
        )
        portfolio_frames = min(
            valid_frames,
            max(4, min(12, seed_frame_count + new_attempts)),
        )
        return cls(
            max_seed_frames=seed_frame_count,
            max_new_capture_attempts=new_attempts,
            max_valid_exploration_frames=valid_frames,
            min_portfolio_frames=min(4, portfolio_frames),
            max_portfolio_frames=portfolio_frames,
            max_recovery_attempts=2,
        )

    def __post_init__(self) -> None:
        allow_zero = {
            "max_seed_frames",
            "max_new_capture_attempts",
            "max_recovery_attempts",
        }
        for name in (
            "max_seed_frames",
            "max_new_capture_attempts",
            "max_valid_exploration_frames",
            "min_portfolio_frames",
            "max_portfolio_frames",
            "max_recovery_attempts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            minimum = 0 if name in allow_zero else 1
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.min_portfolio_frames > self.max_portfolio_frames:
            raise ValueError("min_portfolio_frames cannot exceed max_portfolio_frames")
        if self.max_portfolio_frames > self.max_valid_exploration_frames:
            raise ValueError(
                "max_portfolio_frames cannot exceed max_valid_exploration_frames"
            )


@dataclass(frozen=True, slots=True)
class Stage3BatchPolicy(JsonSerializable):
    """Versioned batching and execution policy for Stage 3 UNKNOWN resolution.

    Exploration may retain more RGB evidence than one model request should
    carry.  This policy bounds each request and the number of requests without
    conflating either value with the exploration or portfolio budgets.
    ``max_concurrent_claims`` parallelizes independent VLM calls only; Unreal
    capture and all result aggregation remain ordered and single-threaded.
    """

    policy_id: str = "stage3-dynamic-batching-v3"
    max_frames_per_request: int = 6
    max_batches_per_claim: int = 8
    max_concurrent_claims: int = 32
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        if (
            isinstance(self.max_frames_per_request, bool)
            or not isinstance(self.max_frames_per_request, int)
            or not 1 <= self.max_frames_per_request <= 10
        ):
            raise ValueError("max_frames_per_request must be between 1 and 10")
        if (
            isinstance(self.max_batches_per_claim, bool)
            or not isinstance(self.max_batches_per_claim, int)
            or not 1 <= self.max_batches_per_claim <= 12
        ):
            raise ValueError("max_batches_per_claim must be between 1 and 12")
        if (
            isinstance(self.max_concurrent_claims, bool)
            or not isinstance(self.max_concurrent_claims, int)
            or not 1 <= self.max_concurrent_claims <= 32
        ):
            raise ValueError("max_concurrent_claims must be between 1 and 32")
        if self.schema_version != "1.0":
            raise ValueError("unsupported Stage 3 batch-policy schema version")

    @property
    def maximum_frames_per_claim(self) -> int:
        return self.max_frames_per_request * self.max_batches_per_claim


class ExplorationSource(_CoercibleEnum):
    SEED = "seed"
    FOCUS = "focus"
    SCOUT = "scout"
    INTERIOR = "interior"
    RECOVERY = "recovery"
    REFRAME = "reframe"


class ExplorationAttemptStatus(_CoercibleEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    CAPTURE_ERROR = "capture_error"
    EMPTY_CAPTURE = "empty_capture"
    SKIPPED = "skipped"
    CIRCUIT_BREAK = "circuit_break"


class PortfolioRequirement(_CoercibleEnum):
    MINIMUM_SIZE = "minimum_size"
    AERIAL = "aerial"
    GLOBAL = "global"
    INTERIOR = "interior"
    POSE_DIVERSITY = "pose_diversity"


@dataclass(frozen=True, slots=True)
class Stage3ExplorationAttempt(JsonSerializable):
    attempt_id: str
    source: ExplorationSource | str
    pose: CameraPose
    shot_role: CaptureShotRole | str
    status: ExplorationAttemptStatus | str
    frame_id: str | None = None
    reason: str | None = None
    cell: tuple[int, int] | None = None
    recovery_step: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _text(self.attempt_id, "attempt_id"))
        object.__setattr__(self, "source", ExplorationSource.coerce(self.source))
        if not isinstance(self.pose, CameraPose):
            raise TypeError("pose must be a CameraPose")
        object.__setattr__(
            self, "shot_role", CaptureShotRole.coerce(self.shot_role)
        )
        status = ExplorationAttemptStatus.coerce(self.status)
        object.__setattr__(self, "status", status)
        frame_id = _text(self.frame_id, "frame_id", optional=True)
        if frame_id is not None and (
            _STAGE3_FRAME_ID.fullmatch(frame_id) is None
            or int(frame_id.removeprefix("s3f_")) < 1
        ):
            raise ValueError("frame_id must be an opaque s3f_XXXXXX id")
        if status is ExplorationAttemptStatus.ACCEPTED and frame_id is None:
            raise ValueError("accepted attempts require frame_id")
        if status is not ExplorationAttemptStatus.ACCEPTED and frame_id is not None:
            raise ValueError("only accepted attempts may contain frame_id")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "reason", _text(self.reason, "reason", optional=True))
        if self.cell is not None:
            if len(self.cell) != 2:
                raise ValueError("cell must contain exactly two coordinates")
            cell = tuple(self.cell)
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 5
                for value in cell
            ):
                raise ValueError("cell coordinates must be integers in [0, 4]")
            object.__setattr__(self, "cell", cell)
        if (
            isinstance(self.recovery_step, bool)
            or not isinstance(self.recovery_step, int)
            or self.recovery_step < 0
        ):
            raise ValueError("recovery_step must be a non-negative integer")
        if self.source is ExplorationSource.RECOVERY and self.recovery_step < 1:
            raise ValueError("recovery attempts require recovery_step >= 1")
        if self.source is not ExplorationSource.RECOVERY and self.recovery_step != 0:
            raise ValueError("only recovery attempts may have recovery_step")


@dataclass(frozen=True, slots=True)
class Stage3Coverage(JsonSerializable):
    """Coverage derived exclusively from admitted, healthy RGB frames."""

    covered_cells: tuple[tuple[int, int], ...] = ()
    valid_seed_frames: int = 0
    valid_new_frames: int = 0
    focus_frames: int = 0
    focused_task_count: int = 0
    aerial_frames: int = 0
    global_frames: int = 0
    interior_frames: int = 0
    distinct_pose_count: int = 0
    missing_requirements: tuple[PortfolioRequirement | str, ...] = ()

    def __post_init__(self) -> None:
        cells = tuple(tuple(value) for value in self.covered_cells)
        if len(cells) != len(set(cells)):
            raise ValueError("covered_cells must not contain duplicates")
        if any(
            len(cell) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 5
                for value in cell
            )
            for cell in cells
        ):
            raise ValueError("covered cells must use integer coordinates in [0, 4]")
        object.__setattr__(self, "covered_cells", tuple(sorted(cells)))
        for name in (
            "valid_seed_frames",
            "valid_new_frames",
            "focus_frames",
            "focused_task_count",
            "aerial_frames",
            "global_frames",
            "interior_frames",
            "distinct_pose_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        missing = tuple(PortfolioRequirement.coerce(value) for value in self.missing_requirements)
        if len(missing) != len(set(missing)):
            raise ValueError("missing_requirements must not contain duplicates")
        object.__setattr__(self, "missing_requirements", missing)

    @property
    def valid_frame_count(self) -> int:
        return self.valid_seed_frames + self.valid_new_frames

    @property
    def constraints_met(self) -> bool:
        return not self.missing_requirements


@dataclass(frozen=True, slots=True)
class Stage3ExplorationResult:
    """Controller exploration product; the live ``frame_store`` is not JSON."""

    budget: Stage3Budget
    attempts: tuple[Stage3ExplorationAttempt, ...]
    coverage: Stage3Coverage
    portfolio_frame_ids: tuple[str, ...]
    frame_store: Any = field(repr=False, compare=False)
    task_frame_ids: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    frame_actor_ids: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    selector_used: bool = False
    selector_error: str | None = None
    runtime_error: str | None = None
    global_context_required: bool = True
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.budget, Stage3Budget):
            raise TypeError("budget must be a Stage3Budget")
        attempts = tuple(self.attempts)
        if any(not isinstance(value, Stage3ExplorationAttempt) for value in attempts):
            raise TypeError("attempts must contain Stage3ExplorationAttempt values")
        attempt_ids = [value.attempt_id for value in attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("exploration attempt ids must be unique")
        object.__setattr__(self, "attempts", attempts)
        if not isinstance(self.coverage, Stage3Coverage):
            raise TypeError("coverage must be Stage3Coverage")
        ids = _unique_texts(self.portfolio_frame_ids, "portfolio_frame_ids")
        if len(ids) > self.budget.max_portfolio_frames:
            raise ValueError("portfolio exceeds max_portfolio_frames")
        if any(
            _STAGE3_FRAME_ID.fullmatch(value) is None
            or int(value.removeprefix("s3f_")) < 1
            for value in ids
        ):
            raise ValueError("portfolio ids must be opaque s3f_XXXXXX ids")
        getter = getattr(self.frame_store, "get", None)
        if not callable(getter):
            raise TypeError("frame_store must provide get(frame_id)")
        for frame_id in ids:
            getter(frame_id)
        object.__setattr__(self, "portfolio_frame_ids", ids)
        if not isinstance(self.task_frame_ids, Mapping):
            raise TypeError("task_frame_ids must be a mapping")
        routes: dict[str, tuple[str, ...]] = {}
        for raw_task_id, raw_frame_ids in self.task_frame_ids.items():
            task_id = _text(raw_task_id, "task_frame_ids key")
            assert task_id is not None
            if task_id in routes:
                raise ValueError("task_frame_ids contains colliding task ids")
            frame_ids = _unique_texts(
                raw_frame_ids,
                f"task_frame_ids[{task_id!r}]",
            )
            if any(
                _STAGE3_FRAME_ID.fullmatch(value) is None
                or int(value.removeprefix("s3f_")) < 1
                for value in frame_ids
            ):
                raise ValueError(
                    "task_frame_ids values must be opaque s3f_XXXXXX ids"
                )
            for frame_id in frame_ids:
                getter(frame_id)
            routes[task_id] = frame_ids
        object.__setattr__(self, "task_frame_ids", MappingProxyType(routes))
        if not isinstance(self.frame_actor_ids, Mapping):
            raise TypeError("frame_actor_ids must be a mapping")
        actor_routes: dict[str, tuple[str, ...]] = {}
        for raw_frame_id, raw_actor_ids in self.frame_actor_ids.items():
            frame_id = _text(raw_frame_id, "frame_actor_ids key")
            assert frame_id is not None
            if (
                _STAGE3_FRAME_ID.fullmatch(frame_id) is None
                or int(frame_id.removeprefix("s3f_")) < 1
            ):
                raise ValueError(
                    "frame_actor_ids keys must be opaque s3f_XXXXXX ids"
                )
            getter(frame_id)
            actor_routes[frame_id] = _unique_texts(
                raw_actor_ids,
                f"frame_actor_ids[{frame_id!r}]",
            )
        object.__setattr__(
            self,
            "frame_actor_ids",
            MappingProxyType(actor_routes),
        )
        if not isinstance(self.selector_used, bool):
            raise TypeError("selector_used must be a bool")
        if not isinstance(self.global_context_required, bool):
            raise TypeError("global_context_required must be a bool")
        object.__setattr__(
            self, "selector_error", _text(self.selector_error, "selector_error", optional=True)
        )
        object.__setattr__(
            self, "runtime_error", _text(self.runtime_error, "runtime_error", optional=True)
        )
        if self.schema_version != "1.0":
            raise ValueError("unsupported Stage 3 exploration schema version")

    def judge_frames(self) -> tuple[Any, ...]:
        # Local import avoids coupling the provider-independent contracts to a
        # concrete model adapter at module-import time. The Stage 3 judge uses
        # its own exact metadata-free boundary type.
        from .stage3_judge import Stage3JudgeFrame

        return tuple(
            Stage3JudgeFrame(frame_id, self.frame_store.get(frame_id).rgb)
            for frame_id in self.portfolio_frame_ids
        )

    def judge_frames_for_task(self, task_id: str) -> tuple[Any, ...]:
        """Return routed focus RGB first, then the metadata-free portfolio."""

        from .stage3_judge import Stage3JudgeFrame

        selected_task_id = _text(task_id, "task_id")
        assert selected_task_id is not None
        ordered_ids = tuple(
            dict.fromkeys(
                (
                    *self.task_frame_ids.get(selected_task_id, ()),
                    *self.portfolio_frame_ids,
                )
            )
        )
        return tuple(
            Stage3JudgeFrame(frame_id, self.frame_store.get(frame_id).rgb)
            for frame_id in ordered_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "budget": self.budget.to_dict(),
            "attempts": [value.to_dict() for value in self.attempts],
            "coverage": self.coverage.to_dict(),
            "portfolio_frame_ids": list(self.portfolio_frame_ids),
            "selector_used": self.selector_used,
            "selector_error": self.selector_error,
            "runtime_error": self.runtime_error,
            "global_context_required": self.global_context_required,
        }


class Stage3Verdict(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def coerce(cls, value: Any) -> Stage3Verdict:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().upper())


class EvidenceStage(_CoercibleEnum):
    STAGE1 = "stage1"
    STAGE2 = "stage2"
    STAGE3 = "stage3"


class EvidenceKind(_CoercibleEnum):
    METADATA = "metadata"
    RGB = "rgb"
    POLICY = "policy"


@dataclass(frozen=True, slots=True)
class EvidenceRef(JsonSerializable):
    stage: EvidenceStage | str
    evidence_id: str
    kind: EvidenceKind | str = EvidenceKind.RGB

    def __post_init__(self) -> None:
        stage = EvidenceStage.coerce(self.stage)
        kind = EvidenceKind.coerce(self.kind)
        evidence_id = _text(self.evidence_id, "evidence_id")
        if kind is EvidenceKind.RGB:
            expected_prefix = {
                EvidenceStage.STAGE2: "s2f_",
                EvidenceStage.STAGE3: "s3f_",
            }.get(stage)
            if expected_prefix is None:
                raise ValueError("Stage 1 evidence cannot be RGB evidence")
            pattern = rf"\A{expected_prefix}[0-9]{{6}}\Z"
            if re.fullmatch(pattern, evidence_id, re.ASCII) is None or int(
                evidence_id.removeprefix(expected_prefix)
            ) < 1:
                raise ValueError(
                    f"{stage.value} RGB evidence requires {expected_prefix}XXXXXX id"
                )
        if kind is EvidenceKind.METADATA and stage is not EvidenceStage.STAGE1:
            raise ValueError("metadata evidence is restricted to Stage 1")
        if kind is EvidenceKind.POLICY and stage is not EvidenceStage.STAGE3:
            raise ValueError("policy evidence is restricted to Stage 3")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "evidence_id", evidence_id)


@dataclass(frozen=True, slots=True)
class Stage3ClaimResolution(JsonSerializable):
    task_id: str
    node_id: str
    judge_verdict: Stage3Verdict | str | None
    final_verdict: Stage3Verdict | str | None
    confidence: float = 0.0
    evidence_refs: tuple[EvidenceRef, ...] = ()
    rationale: str = ""
    forced_mismatch: bool = False
    forced_reason: str | None = None
    evaluation_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        judge = (
            Stage3Verdict.coerce(self.judge_verdict)
            if self.judge_verdict is not None
            else None
        )
        final = (
            Stage3Verdict.coerce(self.final_verdict)
            if self.final_verdict is not None
            else None
        )
        confidence = _probability(self.confidence, "confidence")
        evidence = tuple(self.evidence_refs)
        if any(not isinstance(value, EvidenceRef) for value in evidence):
            raise TypeError("evidence_refs must contain EvidenceRef values")
        if any(value.stage is not EvidenceStage.STAGE3 for value in evidence):
            raise ValueError("Stage 3 resolutions may cite only Stage 3 evidence")
        error = _text(self.evaluation_error, "evaluation_error", optional=True)
        reason = _text(self.forced_reason, "forced_reason", optional=True)
        if not isinstance(self.forced_mismatch, bool):
            raise TypeError("forced_mismatch must be a bool")
        if error is not None:
            if final is not None or self.forced_mismatch:
                raise ValueError("evaluation errors cannot produce a final verdict")
        else:
            if judge is None or final is None:
                raise ValueError("healthy resolutions require judge and final verdicts")
            if judge is Stage3Verdict.UNKNOWN:
                if self.forced_mismatch:
                    if final is not Stage3Verdict.MISMATCH:
                        raise ValueError(
                            "a forced UNKNOWN fallback must end as MISMATCH"
                        )
                elif final is not Stage3Verdict.UNKNOWN:
                    raise ValueError(
                        "an unresolved judge verdict must remain UNKNOWN unless "
                        "the binary fallback is explicit"
                    )
            elif final is not judge or self.forced_mismatch:
                raise ValueError("resolved judge verdict must be preserved")
            if judge is not Stage3Verdict.UNKNOWN and not evidence:
                raise ValueError("resolved Stage 3 verdicts must cite RGB evidence")
        if self.forced_mismatch != (reason is not None):
            raise ValueError("forced_reason is required exactly for forced mismatch")
        object.__setattr__(self, "judge_verdict", judge)
        object.__setattr__(self, "final_verdict", final)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_refs", evidence)
        object.__setattr__(self, "rationale", str(self.rationale).strip())
        object.__setattr__(self, "forced_reason", reason)
        object.__setattr__(self, "evaluation_error", error)


class HolisticDimensionName(_CoercibleEnum):
    GLOBAL_PROMPT_ALIGNMENT = "global_prompt_alignment"
    COMPOSITION_AND_LAYOUT = "composition_and_layout"
    STYLE_ATMOSPHERE_COHERENCE = "style_atmosphere_coherence"
    COMPLETENESS_AND_POLISH = "completeness_and_polish"


HOLISTIC_DIMENSION_WEIGHTS: Mapping[HolisticDimensionName, float] = {
    HolisticDimensionName.GLOBAL_PROMPT_ALIGNMENT: 0.40,
    HolisticDimensionName.COMPOSITION_AND_LAYOUT: 0.25,
    HolisticDimensionName.STYLE_ATMOSPHERE_COHERENCE: 0.20,
    HolisticDimensionName.COMPLETENESS_AND_POLISH: 0.15,
}


@dataclass(frozen=True, slots=True)
class HolisticDimensionScore(JsonSerializable):
    name: HolisticDimensionName | str
    score: float
    weight: float
    rationale: str = ""
    evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        name = HolisticDimensionName.coerce(self.name)
        score = _probability(self.score, "score")
        weight = _probability(self.weight, "weight")
        expected = HOLISTIC_DIMENSION_WEIGHTS[name]
        if not math.isclose(weight, expected, abs_tol=1e-12):
            raise ValueError(f"{name.value} weight must be {expected}")
        evidence = tuple(self.evidence_refs)
        if any(not isinstance(value, EvidenceRef) for value in evidence):
            raise TypeError("evidence_refs must contain EvidenceRef values")
        if any(
            value.stage is not EvidenceStage.STAGE3
            or value.kind is not EvidenceKind.RGB
            for value in evidence
        ):
            raise ValueError("holistic dimensions may cite only Stage 3 RGB")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "rationale", str(self.rationale).strip())
        object.__setattr__(self, "evidence_refs", evidence)


class HolisticResultStatus(_CoercibleEnum):
    COMPLETE = "complete"
    EVALUATION_ERROR = "evaluation_error"


class TransportStatus(_CoercibleEnum):
    NOT_ATTEMPTED = "not_attempted"
    SUCCESS = "success"
    ERROR = "error"

    @classmethod
    def coerce(cls, value: Any) -> TransportStatus:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold()
        if normalized == "ok":
            normalized = "success"
        return cls(normalized)


class ParseStatus(_CoercibleEnum):
    NOT_ATTEMPTED = "not_attempted"
    SUCCESS = "success"
    ERROR = "error"

    @classmethod
    def coerce(cls, value: Any) -> ParseStatus:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold()
        normalized = {
            "valid": "success",
            "ok": "success",
            "invalid": "error",
        }.get(normalized, normalized)
        return cls(normalized)


@dataclass(frozen=True, slots=True)
class HolisticResult(JsonSerializable):
    dimensions: tuple[HolisticDimensionScore, ...] = ()
    overall_score: float | None = None
    summary: str = ""
    status: HolisticResultStatus | str = HolisticResultStatus.COMPLETE
    transport_status: TransportStatus | str = TransportStatus.SUCCESS
    parse_status: ParseStatus | str = ParseStatus.SUCCESS
    evaluation_error: str | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        dimensions = tuple(self.dimensions)
        if any(not isinstance(value, HolisticDimensionScore) for value in dimensions):
            raise TypeError("dimensions must contain HolisticDimensionScore values")
        names = [value.name for value in dimensions]
        if len(names) != len(set(names)):
            raise ValueError("holistic dimension names must be unique")
        status = HolisticResultStatus.coerce(self.status)
        transport = TransportStatus.coerce(self.transport_status)
        parse = ParseStatus.coerce(self.parse_status)
        error = _text(self.evaluation_error, "evaluation_error", optional=True)
        overall = (
            _probability(self.overall_score, "overall_score")
            if self.overall_score is not None
            else None
        )
        if status is HolisticResultStatus.COMPLETE:
            if transport is not TransportStatus.SUCCESS or parse is not ParseStatus.SUCCESS:
                raise ValueError("complete holistic result requires successful transport and parse")
            if error is not None:
                raise ValueError("complete holistic result cannot have evaluation_error")
            if set(names) != set(HOLISTIC_DIMENSION_WEIGHTS):
                raise ValueError("complete holistic result requires all four dimensions")
            computed = sum(value.score * value.weight for value in dimensions)
            if overall is None or not math.isclose(overall, computed, abs_tol=1e-9):
                raise ValueError("overall_score must equal the fixed weighted sum")
        else:
            if error is None:
                raise ValueError("evaluation-error result requires evaluation_error")
            if overall is not None:
                raise ValueError("evaluation-error result cannot publish overall_score")
            if transport is TransportStatus.SUCCESS and parse is ParseStatus.SUCCESS:
                raise ValueError("evaluation error must identify transport or parse failure")
        if transport is TransportStatus.ERROR and parse is not ParseStatus.NOT_ATTEMPTED:
            raise ValueError("parse must be not_attempted after transport error")
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "overall_score", overall)
        object.__setattr__(self, "summary", str(self.summary).strip())
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "transport_status", transport)
        object.__setattr__(self, "parse_status", parse)
        object.__setattr__(self, "evaluation_error", error)
        if self.schema_version != "1.0":
            raise ValueError("unsupported holistic-result schema version")


class DecisionSource(_CoercibleEnum):
    STAGE1_METADATA = "stage1_metadata"
    STAGE2_VISUAL = "stage2_visual"
    STAGE3_VISUAL = "stage3_visual"
    STAGE3_UNKNOWN_FALLBACK = "stage3_unknown_fallback"
    STAGE3_UNRESOLVED = "stage3_unresolved"


@dataclass(frozen=True, slots=True)
class FinalLeafAssessment(JsonSerializable):
    node_id: str
    weight: float
    verdict: Stage3Verdict | str
    decision_source: DecisionSource | str
    confidence: float = 0.0
    evidence_refs: tuple[EvidenceRef, ...] = ()
    rationale: str = ""
    task_id: str | None = None
    stage3_raw_verdict: Stage3Verdict | str | None = None
    forced_mismatch: bool = False
    forced_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        try:
            weight = float(self.weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("weight must be numeric") from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("weight must be finite and positive")
        verdict = Stage3Verdict.coerce(self.verdict)
        source = DecisionSource.coerce(self.decision_source)
        evidence = tuple(self.evidence_refs)
        if any(not isinstance(value, EvidenceRef) for value in evidence):
            raise TypeError("evidence_refs must contain EvidenceRef values")
        raw = (
            Stage3Verdict.coerce(self.stage3_raw_verdict)
            if self.stage3_raw_verdict is not None
            else None
        )
        task_id = _text(self.task_id, "task_id", optional=True)
        reason = _text(self.forced_reason, "forced_reason", optional=True)
        if not isinstance(self.forced_mismatch, bool):
            raise TypeError("forced_mismatch must be a bool")
        if source is DecisionSource.STAGE3_UNKNOWN_FALLBACK:
            if (
                verdict is not Stage3Verdict.MISMATCH
                or raw is not Stage3Verdict.UNKNOWN
                or not self.forced_mismatch
                or reason is None
            ):
                raise ValueError("Stage 3 fallback must preserve UNKNOWN and force MISMATCH")
        elif source is DecisionSource.STAGE3_UNRESOLVED:
            if (
                verdict is not Stage3Verdict.UNKNOWN
                or raw is not Stage3Verdict.UNKNOWN
                or self.forced_mismatch
                or reason is not None
            ):
                raise ValueError("Stage 3 unresolved leaves must preserve UNKNOWN")
        elif self.forced_mismatch or reason is not None:
            raise ValueError("only Stage 3 UNKNOWN fallback may force mismatch")
        if source in {DecisionSource.STAGE1_METADATA, DecisionSource.STAGE2_VISUAL} and raw is not None:
            raise ValueError("Stage 1/2 decisions cannot contain a Stage 3 raw verdict")
        if source is DecisionSource.STAGE3_VISUAL and raw is not verdict:
            raise ValueError("Stage 3 resolved result must preserve its raw verdict")
        expected_stage = {
            DecisionSource.STAGE1_METADATA: EvidenceStage.STAGE1,
            DecisionSource.STAGE2_VISUAL: EvidenceStage.STAGE2,
            DecisionSource.STAGE3_VISUAL: EvidenceStage.STAGE3,
            DecisionSource.STAGE3_UNKNOWN_FALLBACK: EvidenceStage.STAGE3,
            DecisionSource.STAGE3_UNRESOLVED: EvidenceStage.STAGE3,
        }[source]
        if any(value.stage is not expected_stage for value in evidence):
            raise ValueError("evidence refs must match decision_source")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "decision_source", source)
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        object.__setattr__(self, "evidence_refs", evidence)
        object.__setattr__(self, "rationale", str(self.rationale).strip())
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "stage3_raw_verdict", raw)
        object.__setattr__(self, "forced_reason", reason)


@dataclass(frozen=True, slots=True)
class FinalGraphResult(JsonSerializable):
    assessments: tuple[FinalLeafAssessment, ...]
    requirements_score: float | None
    holistic_result: HolisticResult | None = None
    evaluation_error: str | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        assessments = tuple(self.assessments)
        if any(not isinstance(value, FinalLeafAssessment) for value in assessments):
            raise TypeError("assessments must contain FinalLeafAssessment values")
        ids = [value.node_id for value in assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("final leaf node ids must be unique")
        error = _text(self.evaluation_error, "evaluation_error", optional=True)
        score = (
            _probability(self.requirements_score, "requirements_score")
            if self.requirements_score is not None
            else None
        )
        if error is None:
            if not assessments:
                raise ValueError("healthy final result requires leaves")
            has_unknown = any(
                value.verdict is Stage3Verdict.UNKNOWN for value in assessments
            )
            if has_unknown:
                if score is not None:
                    raise ValueError(
                        "requirements_score must be withheld while a leaf is UNKNOWN"
                    )
            else:
                if score is None:
                    raise ValueError("decided final result requires a score")
                total_weight = sum(value.weight for value in assessments)
                computed = sum(
                    value.weight
                    for value in assessments
                    if value.verdict is Stage3Verdict.MATCH
                ) / total_weight
                if not math.isclose(score, computed, abs_tol=1e-9):
                    raise ValueError("requirements_score must equal weighted binary leaves")
        elif score is not None:
            raise ValueError("evaluation-error result cannot publish requirements_score")
        if self.holistic_result is not None and not isinstance(
            self.holistic_result, HolisticResult
        ):
            raise TypeError("holistic_result must be a HolisticResult or None")
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(self, "requirements_score", score)
        object.__setattr__(self, "evaluation_error", error)
        if self.schema_version != "1.0":
            raise ValueError("unsupported final graph-result schema version")

    def for_node(self, node_id: str) -> FinalLeafAssessment | None:
        key = str(node_id).strip()
        return next((value for value in self.assessments if value.node_id == key), None)


__all__ = [
    "HOLISTIC_DIMENSION_WEIGHTS",
    "DecisionSource",
    "EvidenceKind",
    "EvidenceRef",
    "EvidenceStage",
    "ExplorationAttemptStatus",
    "ExplorationSource",
    "FinalGraphResult",
    "FinalLeafAssessment",
    "HolisticDimensionName",
    "HolisticDimensionScore",
    "HolisticResult",
    "HolisticResultStatus",
    "ParseStatus",
    "PortfolioRequirement",
    "Stage3BatchPolicy",
    "Stage3Budget",
    "Stage3ClaimResolution",
    "Stage3Coverage",
    "Stage3ExplorationAttempt",
    "Stage3ExplorationResult",
    "Stage3Verdict",
    "TransportStatus",
]
