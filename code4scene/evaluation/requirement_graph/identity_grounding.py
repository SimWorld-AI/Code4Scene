"""Cost-aware semantic identity grounding over Stage 1 exact and Stage 2 Top-K.

Identity is deliberately separated from constraint satisfaction. Exact trusted
metadata binds without RGB. Only unresolved locator candidates are inspected,
one at a time; a second angle is captured only after an explicit UNKNOWN.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..scene_diff import actor_identity
from .asset_candidates import plan_bounds_camera_poses
from .bundle import FrozenVerificationBundle
from .contracts import (
    ComparisonOperator,
    EntityNode,
    EntityType,
    PredicateNode,
    PredicateType,
    RequirementGraph,
    SceneBounds,
)
from .runtime import FrameProvider, RgbFrameHealthError
from .stage1 import Stage1Result
from .stage2 import Stage2Evaluation
from .stage2_frames import FrameRecord, validate_frame_quality
from .stage2_routing import Stage2ActorTarget
from .stage3_judge import (
    IdentityDecision,
    IdentityVerdict,
    Stage3JudgeFrame,
)


@dataclass(frozen=True, slots=True)
class IdentityCandidateAssessment:
    """Visual identity result for one ranked Candidate Actor."""

    entity_id: str
    actor_id: str
    rank: int
    verdict: IdentityVerdict
    confidence: float
    source_frame_ids: tuple[str, ...]
    rationale: str
    evaluation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "actor_id": self.actor_id,
            "rank": self.rank,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "source_frame_ids": list(self.source_frame_ids),
            "view_count": len(self.source_frame_ids),
            "rationale": self.rationale,
            "evaluation_error": self.evaluation_error,
        }


@dataclass(frozen=True, slots=True)
class EntityIdentityBinding:
    """Distinct Candidate Actor bindings for one expected graph entity."""

    entity_id: str
    expected_name: str
    required_count: int
    exact_actor_ids: tuple[str, ...] = ()
    visual_actor_ids: tuple[str, ...] = ()
    candidate_assessments: tuple[IdentityCandidateAssessment, ...] = ()
    search_exhausted: bool = False
    evaluation_error: str | None = None

    @property
    def matched_actor_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.exact_actor_ids, *self.visual_actor_ids)))

    @property
    def matched_count(self) -> int:
        return len(self.matched_actor_ids)

    @property
    def score(self) -> float:
        if self.required_count <= 0:
            return 1.0
        return min(1.0, self.matched_count / self.required_count)

    @property
    def complete(self) -> bool:
        return self.matched_count >= self.required_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "expected_name": self.expected_name,
            "required_count": self.required_count,
            "exact_actor_ids": list(self.exact_actor_ids),
            "visual_actor_ids": list(self.visual_actor_ids),
            "matched_actor_ids": list(self.matched_actor_ids),
            "matched_count": self.matched_count,
            "identity_score": self.score,
            "complete": self.complete,
            "search_exhausted": self.search_exhausted,
            "evaluation_error": self.evaluation_error,
            "candidate_assessments": [
                value.to_dict() for value in self.candidate_assessments
            ],
        }


@dataclass(frozen=True, slots=True)
class IdentityGroundingResult:
    bindings: tuple[EntityIdentityBinding, ...]
    capture_count: int = 0
    judge_call_count: int = 0

    def for_entity(self, entity_id: str) -> EntityIdentityBinding | None:
        key = str(entity_id).strip()
        return next((value for value in self.bindings if value.entity_id == key), None)

    @property
    def actor_ids_by_entity(self) -> dict[str, tuple[str, ...]]:
        """Bindings safe to use as deterministic geometry populations."""

        return {
            value.entity_id: value.matched_actor_ids
            for value in self.bindings
            if value.evaluation_error is None
            and (value.complete or value.search_exhausted)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "capture_policy": "exact_then_lazy_topk_one_plus_unknown_reframe",
            "capture_count": self.capture_count,
            "judge_call_count": self.judge_call_count,
            "bindings": [value.to_dict() for value in self.bindings],
        }


def required_identity_count(graph: RequirementGraph, entity_id: str) -> int:
    """Return the minimum distinct instance population needed for one entity."""

    minimum = 1
    for node in graph.nodes:
        if not isinstance(node, PredicateNode) or node.predicate_type is not PredicateType.COUNT:
            continue
        if not any(edge.target_id == entity_id for edge in graph.arguments_for(node.id)):
            continue
        constraint = node.constraint
        if constraint is None:
            continue
        if constraint.operator in {
            ComparisonOperator.EQ,
            ComparisonOperator.GTE,
            ComparisonOperator.BETWEEN,
        }:
            minimum = max(minimum, int(constraint.value))
    return minimum


def _exact_actor_ids(stage1: Stage1Result, entity_id: str) -> tuple[str, ...]:
    assessment = stage1.for_entity(entity_id)
    if assessment is None:
        return ()
    values = [*assessment.matched_actor_ids]
    for assembly in assessment.assembly_matches:
        values.extend(assembly.member_actor_ids)
    return tuple(dict.fromkeys(values))


def _image_expected_asset_paths(
    bundle: FrozenVerificationBundle,
    entity_id: str,
) -> tuple[str, ...]:
    """Return GT-authored exact asset identities for one image repair entity.

    These paths are an identity shortcut only. They are never a constraint
    score and never let a deterministic rule select its own Actor population.
    """

    if str(bundle.provenance.get("case_type") or "") != "image_to_scene":
        return ()
    audits = bundle.provenance.get("repair_target_by_entity")
    audit = audits.get(entity_id) if isinstance(audits, Mapping) else None
    if not isinstance(audit, Mapping):
        return ()
    paths: list[str] = []
    members = audit.get("member_gt_geometry")
    if isinstance(members, list):
        paths.extend(
            str(value.get("asset_path") or "").strip()
            for value in members
            if isinstance(value, Mapping)
        )
    gt_actor = audit.get("gt_actor")
    if isinstance(gt_actor, Mapping):
        paths.append(str(gt_actor.get("asset_path") or "").strip())
    return tuple(dict.fromkeys(value for value in paths if value))


def _image_retained_actor_ids(
    bundle: FrozenVerificationBundle,
    entity_id: str,
) -> tuple[str, ...]:
    """Return exact GT identities for an in-place image repair target.

    A retained target already has a stable Input/GT identity.  Matching every
    Candidate Actor that happens to use the same asset over-binds common
    meshes (for example, every awning in a neighborhood) and corrupts later
    count/geometry measurements.  If the stable target disappeared, the
    replacement must go through semantic retrieval/visual grounding instead
    of being silently accepted as an arbitrary same-asset instance.
    """

    if str(bundle.provenance.get("case_type") or "") != "image_to_scene":
        return ()
    audits = bundle.provenance.get("repair_target_by_entity")
    audit = audits.get(entity_id) if isinstance(audits, Mapping) else None
    if not isinstance(audit, Mapping) or audit.get("operation") != "repair":
        return ()
    values = audit.get("member_gt_actor_identities")
    identities = (
        [str(value).strip() for value in values]
        if isinstance(values, list)
        else []
    )
    fallback = str(audit.get("gt_actor_identity") or "").strip()
    if fallback:
        identities.append(fallback)
    return tuple(dict.fromkeys(value for value in identities if value))


def _eligible_actor_population_for_entity(
    bundle: FrozenVerificationBundle,
    scene: Any,
    entity_id: str,
) -> tuple[frozenset[str], bool]:
    values: list[str] = []
    saw_scope = False
    all_scopes_resolved = True
    for binding in bundle.requirements:
        scope = binding.entity_scopes.get(entity_id)
        if scope is None:
            continue
        saw_scope = True
        resolution = scene.scope(scope)
        if resolution.resolved:
            values.extend(str(value) for value in resolution.actor_ids)
        else:
            all_scopes_resolved = False
    return (
        frozenset(value.casefold() for value in values),
        saw_scope and all_scopes_resolved,
    )


def resolve_exact_identity_bindings(
    bundle: FrozenVerificationBundle,
    stage1: Stage1Result,
    evidence: Any,
    scene: Any,
) -> IdentityGroundingResult:
    """Create the only metadata-only ActorBinding path used by Stage 4.

    Generic Stage 1 exact matches are reused for every task. Image repair can
    additionally bind an exact GT asset path inside the entity's frozen
    population scope. The latter is deliberately centralized here: rule
    evaluators may measure bound Actors but may not independently rediscover
    them from an authoring selector.
    """

    raw_actors = tuple(evidence.candidate_actors())
    bindings: list[EntityIdentityBinding] = []
    for node in bundle.graph.nodes:
        if (
            not isinstance(node, EntityNode)
            or node.entity_type is not EntityType.OBJECT
        ):
            continue
        eligible, population_resolved = _eligible_actor_population_for_entity(
            bundle,
            scene,
            node.id,
        )
        retained_ids = frozenset(
            value.casefold()
            for value in _image_retained_actor_ids(bundle, node.id)
        )
        if retained_ids:
            exact = [
                actor_identity(actor)
                for actor in raw_actors
                if actor_identity(actor).casefold() in retained_ids
                and actor_identity(actor).casefold() in eligible
            ]
        else:
            exact = list(_exact_actor_ids(stage1, node.id))
        expected_paths = frozenset(_image_expected_asset_paths(bundle, node.id))
        if expected_paths and eligible and not retained_ids:
            for actor in raw_actors:
                actor_id = actor_identity(actor)
                if (
                    actor_id.casefold() in eligible
                    and str(actor.get("asset_path") or "").strip()
                    in expected_paths
                ):
                    exact.append(actor_id)
        bindings.append(
            EntityIdentityBinding(
                entity_id=node.id,
                expected_name=node.name,
                required_count=required_identity_count(bundle.graph, node.id),
                exact_actor_ids=tuple(dict.fromkeys(exact)),
                # A successfully measured empty population is conclusive
                # negative evidence. Keep unresolved/missing scopes open for
                # Stage 2/3 instead of turning evidence failure into absence.
                search_exhausted=population_resolved and not eligible,
            )
        )
    return IdentityGroundingResult(bindings=tuple(bindings))


def requires_visual_identity_grounding(
    graph: RequirementGraph,
    stage1: Stage1Result,
    stage2: Stage2Evaluation,
    initial_grounding: IdentityGroundingResult | None = None,
) -> bool:
    """Whether any active object needs a locator candidate inspected by VLM."""

    for task in stage2.tasks:
        for entity_id in task.dependency_entity_ids:
            entity = graph.node(entity_id)
            if (
                not isinstance(entity, EntityNode)
                or entity.entity_type is not EntityType.OBJECT
            ):
                continue
            initial = (
                initial_grounding.for_entity(entity_id)
                if initial_grounding is not None
                else None
            )
            exact_count = len(
                initial.exact_actor_ids
                if initial is not None
                else _exact_actor_ids(stage1, entity_id)
            )
            if exact_count >= required_identity_count(graph, entity_id):
                continue
            route = stage2.routing_plan.for_entity(entity_id)
            if route is not None and route.locator_actor_ids:
                return True
    return False


def _existing_actor_frames(
    stage2: Stage2Evaluation,
    actor_id: str,
) -> tuple[tuple[str, Any], ...]:
    key = actor_id.casefold()
    values: list[tuple[str, Any]] = []
    for record in stage2.frame_store.records:
        if not isinstance(record, FrameRecord):
            continue
        actor_keys = tuple(value.casefold() for value in record.actor_ids)
        # A collection/joint image does not designate one Actor strongly
        # enough for candidate-specific identity adjudication.
        if actor_keys == (key,):
            values.append((record.frame_id, record.rgb))
    return tuple(values[:1])


def _capture_view(
    provider: FrameProvider,
    target: Stage2ActorTarget,
    scene_bounds: SceneBounds,
    *,
    alternate: bool,
    capture_index: int,
) -> tuple[str, Any]:
    context, close, oblique = plan_bounds_camera_poses(
        target.bounds.center_cm,
        target.bounds.extent_cm,
        scene_bounds,
    )
    pose = oblique if alternate else close
    if alternate and pose == close:
        pose = context
    resolver = getattr(provider, "resolve_camera_poses", None)
    if callable(resolver):
        resolved = tuple(
            resolver(
                (pose,),
                actor_ids_by_pose=((target.actor_id,),),
            )
        )
        if len(resolved) != 1 or resolved[0] is None:
            raise RgbFrameHealthError(
                "camera_pose_unresolved",
                f"no unobstructed camera pose for Actor {target.actor_id}",
            )
        pose = resolved[0]
    captured = tuple(
        provider.capture(
            (pose,),
            phase="semantic_identity",
            frame_id_prefix=f"identity_{capture_index:06d}",
        )
    )
    if len(captured) != 1:
        raise RuntimeError(
            f"identity provider returned {len(captured)} frames for one pose"
        )
    frame = captured[0]
    rgb, _ = validate_frame_quality(frame.rgb)
    return str(frame.frame_id), rgb


def _judge_candidate(
    judge: Any,
    entity: EntityNode,
    frames: Sequence[tuple[str, Any]],
    *,
    projection_start: int,
) -> tuple[IdentityDecision, dict[str, str]]:
    projected: list[Stage3JudgeFrame] = []
    source_by_projection: dict[str, str] = {}
    for offset, (source_id, rgb) in enumerate(frames):
        frame_id = f"s3f_{projection_start + offset:06d}"
        projected.append(Stage3JudgeFrame(frame_id, rgb))
        source_by_projection[frame_id] = source_id
    decision = judge.judge_identity(
        {
            "expected_entity_name": entity.name,
            "aliases": list(entity.aliases),
        },
        tuple(projected),
    )
    if not isinstance(decision, IdentityDecision):
        raise TypeError("identity judge must return IdentityDecision")
    return decision, source_by_projection


def resolve_identity_bindings(
    graph: RequirementGraph,
    stage1: Stage1Result,
    stage2: Stage2Evaluation,
    scene_bounds: SceneBounds,
    provider: FrameProvider,
    judge: Any,
    initial_grounding: IdentityGroundingResult | None = None,
) -> IdentityGroundingResult:
    """Resolve expected entities with exact-first, Lazy Top-K visual search."""

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    if not isinstance(stage1, Stage1Result):
        raise TypeError("stage1 must be a Stage1Result")
    if not isinstance(stage2, Stage2Evaluation):
        raise TypeError("stage2 must be a Stage2Evaluation")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")

    initial_entity_ids = tuple(
        value.entity_id
        for value in (
            initial_grounding.bindings if initial_grounding is not None else ()
        )
    )
    routed_entity_ids = tuple(
        entity_id
        for task in stage2.tasks
        for entity_id in task.dependency_entity_ids
    )
    active_entity_ids = tuple(
        dict.fromkeys((*initial_entity_ids, *routed_entity_ids))
    )
    bindings: list[EntityIdentityBinding] = []
    capture_count = 0
    projection_counter = 1
    judge_calls_before = int(getattr(judge, "call_count", 0))

    for entity_id in active_entity_ids:
        entity = graph.node(entity_id)
        if not isinstance(entity, EntityNode) or entity.entity_type is not EntityType.OBJECT:
            continue
        required = required_identity_count(graph, entity_id)
        initial = (
            initial_grounding.for_entity(entity_id)
            if initial_grounding is not None
            else None
        )
        exact = (
            initial.exact_actor_ids
            if initial is not None
            else _exact_actor_ids(stage1, entity_id)
        )
        visual: list[str] = []
        assessments: list[IdentityCandidateAssessment] = []
        errors: list[str] = []
        route = stage2.routing_plan.for_entity(entity_id)
        targets = tuple(route.targets) if route is not None else ()
        exact_keys = {value.casefold() for value in exact}
        candidates = tuple(
            target for target in targets if target.actor_id.casefold() not in exact_keys
        )

        for rank, target in enumerate(candidates, start=1):
            if len(exact) + len(visual) >= required:
                break
            frames = list(_existing_actor_frames(stage2, target.actor_id))
            try:
                if not frames:
                    frames.append(
                        _capture_view(
                            provider,
                            target,
                            scene_bounds,
                            alternate=False,
                            capture_index=capture_count + 1,
                        )
                    )
                    capture_count += 1
                decision, source_map = _judge_candidate(
                    judge,
                    entity,
                    frames,
                    projection_start=projection_counter,
                )
                projection_counter += len(frames)
                if decision.verdict is IdentityVerdict.UNKNOWN and decision.error is None:
                    frames.append(
                        _capture_view(
                            provider,
                            target,
                            scene_bounds,
                            alternate=True,
                            capture_index=capture_count + 1,
                        )
                    )
                    capture_count += 1
                    decision, source_map = _judge_candidate(
                        judge,
                        entity,
                        frames,
                        projection_start=projection_counter,
                    )
                    projection_counter += len(frames)
            except Exception as exc:  # noqa: BLE001 - per-candidate boundary
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                assessments.append(
                    IdentityCandidateAssessment(
                        entity_id=entity_id,
                        actor_id=target.actor_id,
                        rank=rank,
                        verdict=IdentityVerdict.UNKNOWN,
                        confidence=0.0,
                        source_frame_ids=tuple(value[0] for value in frames),
                        rationale="identity capture or judge failed",
                        evaluation_error=error,
                    )
                )
                continue

            source_ids = tuple(
                source_map[value]
                for value in decision.evidence_frame_ids
                if value in source_map
            )
            assessments.append(
                IdentityCandidateAssessment(
                    entity_id=entity_id,
                    actor_id=target.actor_id,
                    rank=rank,
                    verdict=decision.verdict,
                    confidence=decision.confidence,
                    source_frame_ids=(
                        source_ids
                        if source_ids
                        else tuple(value[0] for value in frames)
                    ),
                    rationale=decision.rationale,
                    evaluation_error=decision.error,
                )
            )
            if decision.error is not None:
                errors.append(decision.error)
            elif decision.verdict is IdentityVerdict.MATCH:
                visual.append(target.actor_id)

        matched = len(tuple(dict.fromkeys((*exact, *visual))))
        search_exhausted = matched >= required or (
            bool(candidates) and len(assessments) >= len(candidates)
        )
        binding_error = "; ".join(dict.fromkeys(errors)) if errors and matched < required else None
        bindings.append(
            EntityIdentityBinding(
                entity_id=entity_id,
                expected_name=entity.name,
                required_count=required,
                exact_actor_ids=exact,
                visual_actor_ids=tuple(visual),
                candidate_assessments=tuple(assessments),
                search_exhausted=search_exhausted,
                evaluation_error=binding_error,
            )
        )

    return IdentityGroundingResult(
        bindings=tuple(bindings),
        capture_count=capture_count,
        judge_call_count=max(
            0, int(getattr(judge, "call_count", 0)) - judge_calls_before
        ),
    )


__all__ = [
    "EntityIdentityBinding",
    "IdentityCandidateAssessment",
    "IdentityGroundingResult",
    "required_identity_count",
    "requires_visual_identity_grounding",
    "resolve_exact_identity_bindings",
    "resolve_identity_bindings",
]
