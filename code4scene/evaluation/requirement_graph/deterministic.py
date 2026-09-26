"""Merged deterministic Stage 1 evaluation for rule-compiled graph leaves."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from code4scene.evaluation import contracts as report_contracts
from code4scene.evaluation.assertions import Check
from code4scene.evaluation.scene_diff import (
    actor_identity,
    diff_scenes,
    index_actors,
)
from code4scene.evaluation.semantic_graph import build

from .bundle import (FrozenVerificationBundle, PopulationScope,
                     RequirementBinding, RequirementStatus, UnknownReason)
from .contracts import ClaimVerdict, EntityNode, EntityType
from .deterministic_alignment import align_deterministic_rules
from .evidence_adapter import SceneInventoryEvidence
from .rules import EVALUATORS, FAMILIES, RuleEvaluationContext


@dataclass(frozen=True, slots=True)
class DeterministicAssessment:
    node_id: str
    requirement_id: str
    rule_id: str
    family: str
    verdict: ClaimVerdict | str
    check: Mapping[str, Any] | None = None
    unknown_reason: UnknownReason | str | None = None
    rationale: str = ""
    resolution_stage: str = "stage1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", ClaimVerdict.coerce(self.verdict))
        if self.unknown_reason is not None:
            object.__setattr__(self, "unknown_reason", UnknownReason(self.unknown_reason))
        if self.verdict is ClaimVerdict.UNKNOWN and self.unknown_reason is None:
            raise ValueError("UNKNOWN deterministic assessments need unknown_reason")
        if self.verdict is not ClaimVerdict.UNKNOWN and self.unknown_reason is not None:
            raise ValueError("decided deterministic assessments cannot have unknown_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "requirement_id": self.requirement_id,
            "rule_id": self.rule_id,
            "family": self.family,
            "verdict": self.verdict.value,
            "resolved_by": (
                self.resolution_stage
                if self.verdict is not ClaimVerdict.UNKNOWN
                else None
            ),
            "unknown_reason": (
                self.unknown_reason.value if self.unknown_reason is not None else None
            ),
            "rationale": self.rationale,
            "check": dict(self.check) if self.check is not None else None,
        }


def _scoped_item(
    item: Mapping[str, Any],
    binding: RequirementBinding,
    bundle: FrozenVerificationBundle,
    bound_actor_ids_by_entity: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    value = dict(item)
    arguments = bundle.graph.arguments_for(binding.node_id)
    entity_ids = (
        tuple(argument.target_id for argument in arguments)
        if arguments
        else tuple(binding.entity_scopes)
    )
    if entity_ids and isinstance(item.get("subject"), Mapping):
        subject_is_bound = (
            bound_actor_ids_by_entity is not None
            and entity_ids[0] in bound_actor_ids_by_entity
        )
        subject_scope = (
            f"identity_binding:{entity_ids[0]}"
            if subject_is_bound
            else binding.scope_for_entity(entity_ids[0]).value
        )
        # A confirmed identity binding replaces the authoring-time asset/name
        # selector. Keeping that selector would reject a visually equivalent
        # substitute after the VLM had already confirmed its semantics.
        value["subject"] = (
            {"scope": subject_scope}
            if subject_is_bound
            else {**item["subject"], "scope": subject_scope}
        )
        if subject_is_bound:
            value["visual_substitution_fallback"] = False
    if len(entity_ids) > 1 and isinstance(item.get("object"), Mapping):
        object_is_bound = (
            bound_actor_ids_by_entity is not None
            and entity_ids[1] in bound_actor_ids_by_entity
        )
        object_scope = (
            f"identity_binding:{entity_ids[1]}"
            if object_is_bound
            else binding.scope_for_entity(entity_ids[1]).value
        )
        value["object"] = (
            {"scope": object_scope}
            if object_is_bound
            else {**item["object"], "scope": object_scope}
        )
    return value


def _unknown(
    binding: RequirementBinding,
    rule_id: str,
    family: str,
    reason: UnknownReason,
    rationale: str,
) -> DeterministicAssessment:
    return DeterministicAssessment(
        node_id=binding.node_id,
        requirement_id=binding.requirement_id,
        rule_id=rule_id,
        family=family,
        verdict=ClaimVerdict.UNKNOWN,
        unknown_reason=reason,
        rationale=rationale,
    )


def _bounded_ratio(observed: float, expected: float, *, lower: bool) -> float:
    observed = max(0.0, float(observed))
    expected = max(0.0, float(expected))
    if lower:
        return min(1.0, observed / expected) if expected > 0.0 else 1.0
    return 1.0 if observed <= expected else expected / observed


def _continuous_check_score(family: str, check: Check) -> float | None:
    """Expose measured quality while retaining verdicts for audit routing."""

    expected = check.expected
    observed = check.observed
    if family == "CNT":
        count = float(observed.get("count", 0.0))
        qualities: list[float] = []
        if expected.get("min_count") is not None:
            qualities.append(
                _bounded_ratio(count, float(expected["min_count"]), lower=True)
            )
        if expected.get("max_count") is not None:
            qualities.append(
                _bounded_ratio(count, float(expected["max_count"]), lower=False)
            )
        if expected.get("exact_count") is not None:
            exact = float(expected["exact_count"])
            qualities.append(
                max(0.0, 1.0 - abs(count - exact) / max(count, exact, 1.0))
            )
        return min(qualities) if qualities else None
    if family != "OOR":
        return None
    relation = str(expected.get("relation") or "")
    minimum_subjects = float(expected.get("minimum_subject_count") or 1.0)
    if relation == "around":
        count_quality = _bounded_ratio(
            float(observed.get("within_distance_count", 0.0)),
            minimum_subjects,
            lower=True,
        )
        coverage_quality = _bounded_ratio(
            float(observed.get("angular_coverage", 0.0)),
            float(expected.get("minimum_angular_coverage") or 1.0),
            lower=True,
        )
        return count_quality * coverage_quality
    rows = tuple(value for value in check.evidence if isinstance(value, Mapping))
    if not rows:
        return 0.0

    def row_quality(row: Mapping[str, Any]) -> float:
        if relation in {"facing", "parallel"}:
            error = min(180.0, abs(float(row.get("yaw_error_deg", 180.0))))
            return 1.0 - error / 180.0
        if relation == "near":
            distance = max(0.0, float(row.get("relation_distance_cm", 0.0)))
            scale = max(float(expected.get("maximum_distance_cm") or 1.0), 1.0)
            return 1.0 / (1.0 + (distance / scale) ** 2)
        if relation == "far":
            return _bounded_ratio(
                float(row.get("relation_distance_cm", 0.0)),
                float(expected.get("minimum_distance_cm") or 1.0),
                lower=True,
            )
        if relation == "on_top_of":
            gap = abs(float(row.get("vertical_gap_cm", 1e9)))
            gap_scale = max(float(expected.get("tolerance_cm") or 5.0), 1.0)
            overlap = float(row.get("xy_overlap_fraction", 0.0))
            minimum_overlap = float(
                expected.get("minimum_overlap_fraction") or 0.1
            )
            return math.sqrt(
                (1.0 / (1.0 + (gap / gap_scale) ** 2))
                * _bounded_ratio(overlap, minimum_overlap, lower=True)
            )
        if relation == "against":
            gap = max(
                0.0,
                float(row.get("closest_surface_distance_cm", 1e9)),
            )
            scale = max(float(row.get("maximum_surface_gap_cm") or 10.0), 1.0)
            axes = min(1.0, float(row.get("intersecting_axis_count", 0.0)) / 2.0)
            return math.sqrt(axes / (1.0 + (gap / scale) ** 2))
        if relation == "inside":
            return min(
                1.0,
                float(row.get("overlap_volume_fraction_of_subject", 0.0)),
            )
        if relation in {"occupies", "intersects", "blocking"}:
            return min(
                1.0,
                max(
                    float(row.get("xy_overlap_fraction_of_subject", 0.0)),
                    float(row.get("overlap_volume_fraction_of_subject", 0.0)),
                ),
            )
        return float(bool(row.get("pass")))

    row_scores = tuple(row_quality(value) for value in rows)
    quantifier = str(expected.get("quantifier") or "each_subject")
    if quantifier == "any_pair":
        relation_quality = max(row_scores)
    elif quantifier == "all_pairs":
        relation_quality = sum(row_scores) / len(row_scores)
    else:
        by_subject: dict[str, list[float]] = {}
        for row, score in zip(rows, row_scores, strict=True):
            by_subject.setdefault(str(row.get("subject_key")), []).append(score)
        relation_quality = (
            sum(max(values) for values in by_subject.values()) / len(by_subject)
            if by_subject
            else 0.0
        )
    count_quality = _bounded_ratio(
        float(observed.get("subject_count", 0.0)),
        minimum_subjects,
        lower=True,
    )
    return count_quality * relation_quality


def evaluate_deterministic_rules(
    bundle: FrozenVerificationBundle,
    scene: Any,
    inventory: SceneInventoryEvidence,
    *,
    bound_actor_ids_by_entity: Mapping[str, Sequence[str]] | None = None,
    require_actor_bindings: bool = False,
    resolution_stage: str = "stage1",
) -> tuple[DeterministicAssessment, ...]:
    """Evaluate every rule-compiled leaf with the existing canonical logic."""

    draft = bundle.provenance.get("deterministic_rule_draft")
    items = draft.get("items") if isinstance(draft, Mapping) else None
    by_rule = {
        str(item.get("id")): item
        for item in items or ()
        if isinstance(item, Mapping) and item.get("id")
    }
    runtime_rule_ids = (
        align_deterministic_rules(bundle.graph, draft)
        if isinstance(draft, Mapping)
        else {}
    )
    raw_actors = tuple(scene.candidate_actors())
    raw_by_id = {actor_identity(actor): actor for actor in raw_actors}
    raw_by_folded_id = {key.casefold(): value for key, value in raw_by_id.items()}
    scoped_raw: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for scope, resolution in inventory.scopes.items():
        if resolution.resolved:
            scoped_raw[scope.value] = tuple(
                raw_by_id[value]
                for value in resolution.actor_ids
                if value in raw_by_id
            )
    identity_bindings = bound_actor_ids_by_entity or {}
    for entity_id, actor_ids in identity_bindings.items():
        selected: list[Mapping[str, Any]] = []
        for actor_id in actor_ids:
            actor = raw_by_id.get(str(actor_id)) or raw_by_folded_id.get(
                str(actor_id).casefold()
            )
            if actor is not None and actor not in selected:
                selected.append(actor)
        scoped_raw[f"identity_binding:{entity_id}"] = tuple(selected)
    semantic_graph = build(raw_actors, scopes=scoped_raw)
    rule_context = RuleEvaluationContext(
        graph=semantic_graph,
    )

    results: list[DeterministicAssessment] = []
    for binding in bundle.requirements:
        rule_id = (
            binding.deterministic_rule_id
            or runtime_rule_ids.get(binding.node_id)
        )
        # A WAIVED binding is out of the score by declaration — evaluating its
        # rule anyway would put it back into both the numerator and the
        # denominator. The stage-1 atomic pass and `semantic_nodes()` already
        # skip non-SUPPORTED bindings; this loop must agree with them.
        if (binding.primary_owner != "requirement_graph" or rule_id is None
                or binding.status is not RequirementStatus.SUPPORTED):
            continue
        item = by_rule.get(rule_id)
        family = FAMILIES.get(str((item or {}).get("type")), "")
        if item is None or family not in EVALUATORS:
            results.append(
                _unknown(
                    binding,
                    rule_id,
                    family or "UNSUPPORTED",
                    UnknownReason.UNSUPPORTED,
                    "the frozen deterministic rule is missing or has no evaluator",
                )
            )
            continue
        argument_entity_ids = tuple(
            argument.target_id
            for argument in bundle.graph.arguments_for(binding.node_id)
            if isinstance(bundle.graph.node(argument.target_id), EntityNode)
            and bundle.graph.node(argument.target_id).entity_type
            is EntityType.OBJECT
        )
        missing_bindings = tuple(
            entity_id
            for entity_id in argument_entity_ids
            if entity_id not in identity_bindings
        )
        if require_actor_bindings and missing_bindings:
            results.append(
                _unknown(
                    binding,
                    rule_id,
                    family,
                    UnknownReason.VISUAL_EVIDENCE_INCOMPLETE,
                    "Stage 4 requires ActorBindings for: "
                    + ", ".join(missing_bindings),
                )
            )
            continue
        unresolved = [
            inventory.scope(scope)
            for entity_id, scope in binding.entity_scopes.items()
            if entity_id not in identity_bindings
            and not inventory.scope(scope).resolved
        ]
        if unresolved:
            first = unresolved[0]
            results.append(
                _unknown(
                    binding,
                    rule_id,
                    family,
                    first.unknown_reason or UnknownReason.SCOPE_UNRESOLVED,
                    first.detail or "the required actor population is unresolved",
                )
            )
            continue
        try:
            checks: list[Check] = EVALUATORS[family](
                rule_context,
                _scoped_item(
                    item,
                    binding,
                    bundle,
                    identity_bindings,
                ),
            )
        except Exception as error:  # noqa: BLE001 - result carries the failure
            results.append(
                _unknown(
                    binding,
                    rule_id,
                    family,
                    UnknownReason.EVALUATION_ERROR,
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        if len(checks) != 1:
            results.append(
                _unknown(
                    binding,
                    rule_id,
                    family,
                    UnknownReason.EVALUATION_ERROR,
                    f"deterministic rule produced {len(checks)} checks instead of one",
                )
            )
            continue
        check = checks[0]
        if check.status == report_contracts.PASS:
            verdict, reason = ClaimVerdict.MATCH, None
        elif check.status == report_contracts.FAIL:
            verdict, reason = ClaimVerdict.MISMATCH, None
        else:
            verdict, reason = ClaimVerdict.UNKNOWN, UnknownReason.VISUAL_EVIDENCE_INCOMPLETE
        check_payload = check.to_json_dict()
        continuous_score = _continuous_check_score(family, check)
        if continuous_score is not None:
            check_payload["score"] = max(0.0, min(1.0, continuous_score))
        results.append(
            DeterministicAssessment(
                node_id=binding.node_id,
                requirement_id=binding.requirement_id,
                rule_id=rule_id,
                family=family,
                verdict=verdict,
                check=check_payload,
                unknown_reason=reason,
                rationale=check.failure_reason or "deterministic rule satisfied",
                resolution_stage=resolution_stage,
            )
        )
    return tuple(results)


def evaluate_repair_target_fields(
    bundle: FrozenVerificationBundle,
    scene: Any,
    *,
    bound_actor_ids_by_entity: Mapping[str, Sequence[str]] | None = None,
) -> tuple[DeterministicAssessment, ...]:
    """Measure retained repair fields against GT after identity binding.

    Exact retained Actors have stronger evidence for location, rotation, scale,
    and exported property/material values than an RGB judge. A missing stable
    Actor remains UNKNOWN so Stage 2/3 can accept a visually equivalent
    replacement; add/remove targets likewise keep their existing visual route.
    """

    targets = bundle.provenance.get("repair_target_by_node")
    if not isinstance(targets, Mapping):
        return ()
    candidate_by_id = index_actors({"actors": list(scene.candidate_actors())})
    results: list[DeterministicAssessment] = []
    for binding in bundle.requirements:
        target = targets.get(binding.node_id)
        if (
            not isinstance(target, Mapping)
            or target.get("operation") != "repair"
        ):
            continue
        expected = target.get("structured_expectation")
        expected_fields = tuple(
            str(value) for value in target.get("changed_fields") or ()
        )
        identity = str(target.get("gt_actor_identity") or "").strip()
        target_id = target.get("target_id") or binding.node_id
        rule_id = f"repair_fields:{target_id}"
        if not isinstance(expected, Mapping) or not expected_fields or not identity:
            results.append(
                _unknown(
                    binding,
                    rule_id,
                    "REPAIR_TARGET_FIELDS",
                    UnknownReason.VISUAL_EVIDENCE_INCOMPLETE,
                    "the repair target has no complete structured GT expectation",
                )
            )
            continue
        arguments = bundle.graph.arguments_for(binding.node_id)
        subject_entity_id = arguments[0].target_id if arguments else None
        bound_actor_ids = (
            tuple(bound_actor_ids_by_entity.get(subject_entity_id, ()))
            if bound_actor_ids_by_entity is not None
            and subject_entity_id is not None
            else ()
        )
        candidate_id = (
            identity
            if identity in bound_actor_ids
            else bound_actor_ids[0]
            if len(bound_actor_ids) == 1
            else None
        )
        candidate = candidate_by_id.get(candidate_id) if candidate_id else None
        if candidate is None:
            results.append(
                _unknown(
                    binding,
                    rule_id,
                    "REPAIR_TARGET_FIELDS",
                    UnknownReason.VISUAL_EVIDENCE_INCOMPLETE,
                    (
                        "the retained repair target has no unique ActorBinding"
                    ),
                )
            )
            continue
        difference = diff_scenes(
            {"actors": [dict(expected)]},
            {"actors": [candidate]},
        )
        observed_differences = {
            *(
                f"transform.{field}"
                for change in difference.moved
                for field in change.fields
            ),
            *(
                f"property.{field}"
                for change in difference.modified
                for field in change.fields
            ),
        }
        mismatched = tuple(
            field for field in expected_fields if field in observed_differences
        )
        verdict = ClaimVerdict.MISMATCH if mismatched else ClaimVerdict.MATCH
        results.append(
            DeterministicAssessment(
                node_id=binding.node_id,
                requirement_id=binding.requirement_id,
                rule_id=rule_id,
                family="REPAIR_TARGET_FIELDS",
                verdict=verdict,
                rationale=(
                    "retained Candidate Actor differs from GT on: "
                    + ", ".join(mismatched)
                    if mismatched
                    else "retained Candidate Actor matches every GT repair field"
                ),
                check={
                    "measurement": "retained_actor_changed_fields",
                    "score": 0.0 if mismatched else 1.0,
                    "actor_identity": identity,
                    "expected_fields": list(expected_fields),
                    "mismatched_fields": list(mismatched),
                },
                resolution_stage="stage4_deterministic",
            )
        )
    return tuple(results)


def _location_cm(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 3:
        return None
    try:
        point = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in point):
        return None
    return point  # type: ignore[return-value]


def _row_quality(points: Sequence[tuple[float, float, float]]) -> float:
    """Continuous collinearity × spacing uniformity, without a pass threshold."""

    if len(points) < 3:
        return 0.0
    endpoints = max(
        combinations(points, 2),
        key=lambda pair: math.dist(pair[0], pair[1]),
    )
    span = math.dist(*endpoints)
    if span <= 1e-9:
        return 0.0
    axis = tuple(
        (right - left) / span
        for left, right in zip(*endpoints, strict=True)
    )
    origin = endpoints[0]
    projections: list[float] = []
    normalized_residuals: list[float] = []
    for point in points:
        offset = tuple(
            coordinate - anchor
            for coordinate, anchor in zip(point, origin, strict=True)
        )
        projection = sum(
            component * direction
            for component, direction in zip(offset, axis, strict=True)
        )
        residual = math.sqrt(
            sum(
                (component - projection * direction) ** 2
                for component, direction in zip(offset, axis, strict=True)
            )
        )
        projections.append(projection)
        normalized_residuals.append(residual / span)
    linearity = 1.0 / (1.0 + sum(normalized_residuals) / len(points))
    ordered = sorted(projections)
    gaps = [right - left for left, right in zip(ordered, ordered[1:], strict=False)]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 1e-9:
        return 0.0
    gap_cv = math.sqrt(
        sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
    ) / mean_gap
    uniformity = 1.0 / (1.0 + gap_cv)
    return math.sqrt(linearity * uniformity)


def _geometry_location(value: Mapping[str, Any]) -> tuple[float, float, float] | None:
    transform = value.get("transform")
    raw = (
        transform.get("location_cm")
        if isinstance(transform, Mapping)
        else value.get("location_cm")
    )
    return _location_cm(raw)


def _rotation_vector(value: Mapping[str, Any]) -> tuple[float, float, float] | None:
    transform = value.get("transform")
    raw = (
        transform.get("rotation_deg")
        if isinstance(transform, Mapping)
        else value.get("rotation_deg")
    )
    return _location_cm(raw)


def _extent_vector(value: Mapping[str, Any]) -> tuple[float, float, float] | None:
    bounds = value.get("bounds")
    raw = bounds.get("extent_cm") if isinstance(bounds, Mapping) else None
    return _location_cm(raw)


def _cyclic_rotation_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _crossed_quality(values: Sequence[Mapping[str, Any]]) -> float:
    """Continuous angular separation times crossing-point proximity."""

    if len(values) != 2:
        return 0.0
    locations = tuple(_geometry_location(value) for value in values)
    rotations = tuple(_rotation_vector(value) for value in values)
    extents = tuple(_extent_vector(value) for value in values)
    if any(value is None for value in (*locations, *rotations, *extents)):
        return 0.0
    left_rotation, right_rotation = rotations
    separation = max(
        min(
            _cyclic_rotation_error(left, right),
            180.0 - _cyclic_rotation_error(left, right),
        )
        for left, right in zip(left_rotation, right_rotation, strict=True)
    )
    angle_quality = math.sin(math.radians(separation))
    scale = max(
        2.0 * max(component for extent in extents for component in extent),
        1.0,
    )
    center_distance = math.dist(*locations)
    proximity = 1.0 / (1.0 + (center_distance / scale) ** 2)
    return math.sqrt(max(0.0, angle_quality) * proximity)


def evaluate_repair_collection_geometry(
    bundle: FrozenVerificationBundle,
    scene: Any,
    inventory: SceneInventoryEvidence,
    *,
    bound_actor_ids_by_entity: Mapping[str, Sequence[str]] | None = None,
) -> tuple[DeterministicAssessment, ...]:
    """Score qualitative collection relations over explicitly bound Actors."""

    targets = bundle.provenance.get("repair_target_by_node")
    if not isinstance(targets, Mapping):
        return ()
    additions = inventory.scope(PopulationScope.ADDITIONS)
    if not additions.resolved:
        return ()
    candidate_by_id = index_actors({"actors": list(scene.candidate_actors())})
    results: list[DeterministicAssessment] = []
    for binding in bundle.requirements:
        target = targets.get(binding.node_id)
        facet = str(target.get("facet") or "") if isinstance(target, Mapping) else ""
        arrangement = (
            "row"
            if facet.startswith("row_")
            else "crossed"
            if facet.startswith("crossed_")
            else None
        )
        if (
            not isinstance(target, Mapping)
            or target.get("operation") != "add"
            or arrangement is None
        ):
            continue
        expected_values = target.get("member_gt_geometry")
        if not isinstance(expected_values, Sequence) or isinstance(
            expected_values, (str, bytes)
        ):
            continue
        expected_geometry = tuple(
            value for value in expected_values if isinstance(value, Mapping)
        )
        group_size = int(target.get("group_size") or len(expected_geometry))
        if len(expected_geometry) != group_size:
            continue
        arguments = bundle.graph.arguments_for(binding.node_id)
        subject_entity_id = arguments[0].target_id if arguments else None
        bound_actor_ids = (
            tuple(bound_actor_ids_by_entity.get(subject_entity_id, ()))
            if bound_actor_ids_by_entity is not None and subject_entity_id is not None
            else ()
        )
        rule_id = (
            "repair_collection_geometry:"
            f"{target.get('target_id') or binding.node_id}"
        )
        if not bound_actor_ids:
            results.append(
                _unknown(
                    binding,
                    rule_id,
                    "REPAIR_COLLECTION_GEOMETRY",
                    UnknownReason.VISUAL_EVIDENCE_INCOMPLETE,
                    "Stage 4 collection geometry requires an ActorBinding",
                )
            )
            continue
        candidates: list[tuple[str, Mapping[str, Any]]] = []
        for actor_id in bound_actor_ids:
            actor = candidate_by_id.get(actor_id)
            if not isinstance(actor, Mapping):
                continue
            if _geometry_location(actor) is not None:
                candidates.append((actor_id, actor))
        if len(candidates) < group_size:
            results.append(
                _unknown(
                    binding,
                    rule_id,
                    "REPAIR_COLLECTION_GEOMETRY",
                    UnknownReason.VISUAL_EVIDENCE_INCOMPLETE,
                    "exact structured identity did not cover the full collection; inspect visual replacements",
                )
            )
            continue
        if math.comb(len(candidates), group_size) > 10_000:
            results.append(
                _unknown(
                    binding,
                    rule_id,
                    "REPAIR_COLLECTION_GEOMETRY",
                    UnknownReason.VISUAL_EVIDENCE_INCOMPLETE,
                    "too many exact-asset additions for bounded collection assignment",
                )
            )
            continue
        if arrangement == "row":
            expected_quality = _row_quality(
                tuple(
                    point
                    for value in expected_geometry
                    if (point := _geometry_location(value)) is not None
                )
            )
            scored_subsets = tuple(
                (
                    _row_quality(
                        tuple(
                            point
                            for _, actor in subset
                            if (point := _geometry_location(actor)) is not None
                        )
                    ),
                    tuple(actor_id for actor_id, _ in subset),
                )
                for subset in combinations(candidates, group_size)
            )
            measurement = "relative_row_geometry"
            quality_fields = {
                "candidate_row_quality": None,
                "gt_row_quality_baseline": expected_quality,
                "components": ["collinearity", "spacing_uniformity"],
            }
        else:
            if group_size != 2:
                continue
            expected_quality = _crossed_quality(expected_geometry)
            scored_subsets = tuple(
                (
                    _crossed_quality(tuple(actor for _, actor in subset)),
                    tuple(actor_id for actor_id, _ in subset),
                )
                for subset in combinations(candidates, group_size)
            )
            measurement = "relative_crossed_geometry"
            quality_fields = {
                "candidate_crossed_quality": None,
                "gt_crossed_quality_baseline": expected_quality,
                "components": ["angular_separation", "crossing_point_proximity"],
            }
        candidate_quality, actor_ids = max(scored_subsets, key=lambda value: value[0])
        quality_fields[
            "candidate_row_quality"
            if arrangement == "row"
            else "candidate_crossed_quality"
        ] = candidate_quality
        score = (
            min(1.0, candidate_quality / expected_quality)
            if expected_quality > 1e-9
            else 0.0
        )
        verdict = (
            ClaimVerdict.MATCH
            if math.isclose(score, 1.0, rel_tol=0.0, abs_tol=1e-9)
            else ClaimVerdict.MISMATCH
        )
        results.append(
            DeterministicAssessment(
                node_id=binding.node_id,
                requirement_id=binding.requirement_id,
                rule_id=rule_id,
                family="REPAIR_COLLECTION_GEOMETRY",
                verdict=verdict,
                rationale=(
                    f"structured {arrangement} quality {candidate_quality:.6f} "
                    f"relative to GT qualitative baseline {expected_quality:.6f}"
                ),
                check={
                    "measurement": measurement,
                    "score": score,
                    "actor_ids": list(actor_ids),
                    **quality_fields,
                    "evidence": [
                        {
                            "actor_ids": list(actor_ids),
                            "candidate_quality": candidate_quality,
                            "gt_qualitative_baseline": expected_quality,
                            "components": quality_fields["components"],
                        }
                    ],
                },
                resolution_stage="stage4_deterministic",
            )
        )
    return tuple(results)


__all__ = [
    "DeterministicAssessment",
    "evaluate_deterministic_rules",
    "evaluate_repair_collection_geometry",
    "evaluate_repair_target_fields",
]
