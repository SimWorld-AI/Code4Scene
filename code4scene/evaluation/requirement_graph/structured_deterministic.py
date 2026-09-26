"""Conservative structured-first evaluation for Stage0 v2 requirements.

The frozen v2 graph preserves more semantics than the legacy deterministic
rule compiler can currently lower.  This module closes the safe subset from
the Candidate scene snapshot before RGB acquisition:

* object quantities from a complete trusted category population, or safe
  lower bounds from audited canonical-name Actor grounding;
* intrinsic AABB/Transform relations between uniquely grounded Actors; and
* affirmative surface-colour attributes exposed by resolved UE material
  parameters without texture participation.

Anything that is not provable from those records returns to the existing
Stage 2/3 visual path.  In particular, locator candidates, asset-name guesses,
missing material parameters, textured materials, collection-level ambiguous
identity, and relations that need an undeclared semantic threshold never
produce a structured verdict.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from code4scene.evaluation import contracts as report_contracts
from code4scene.evaluation import spatial_relations
from code4scene.evaluation.scene_diff import actor_identity
from code4scene.evaluation.semantic_graph import build as build_semantic_graph

from .bundle import FrozenVerificationBundle, RequirementStatus
from .contracts import (
    ClaimVerdict,
    EntityNode,
    Polarity,
    PredicateNode,
    ReferentKind,
)
from .deterministic import DeterministicAssessment
from .evidence_adapter import SceneInventoryEvidence
from .stage1 import Stage1Result

STRUCTURED_EVIDENCE_VERSION = "1.0"

_OBJECT_COUNT_UNITS = frozenset(
    {
        "",
        "actor",
        "actors",
        "item",
        "items",
        "object",
        "objects",
        "piece",
        "pieces",
        "structure",
        "structures",
    }
)

# These relations have an intrinsic interpretation in the existing canonical
# geometry evaluator.  Near/far/facing/parallel are deliberately absent: a
# prompt does not specify their distance/angle threshold or an asset's visual
# forward axis, so Transform data alone cannot decide them safely.
_STRUCTURED_RELATIONS = frozenset(
    {
        "above",
        "against",
        "below",
        "blocks",
        "inside",
        "intersects",
        "left_of",
        "occupies",
        "on_top_of",
        "right_of",
    }
)
_RELATION_ALIASES = {
    "blocking": "blocks",
    "butting_against": "against",
    "butting_against_one_another": "against",
    "on_top": "on_top_of",
    "on_topof": "on_top_of",
}

_COLOUR_PROTOTYPES_SRGB: dict[str, tuple[float, float, float]] = {
    "black": (0.03, 0.03, 0.03),
    "blue": (0.08, 0.20, 0.82),
    "brown": (0.45, 0.24, 0.10),
    "beige": (0.76, 0.68, 0.52),
    "cyan": (0.08, 0.78, 0.82),
    "gray": (0.50, 0.50, 0.50),
    "green": (0.10, 0.58, 0.16),
    "olive": (0.45, 0.50, 0.10),
    "orange": (0.95, 0.40, 0.05),
    "pink": (0.94, 0.42, 0.62),
    "purple": (0.48, 0.13, 0.65),
    "red": (0.82, 0.06, 0.05),
    "salmon": (0.98, 0.45, 0.38),
    "white": (0.94, 0.94, 0.94),
    "yellow": (0.92, 0.82, 0.06),
}
_COLOUR_ALIASES = {
    "grey": "gray",
    "olive green": "olive",
    "olive-green": "olive",
    "salmon pink": "salmon",
    "salmon-pink": "salmon",
}


@dataclass(frozen=True, slots=True)
class StructuredEvaluation:
    """Structured verdicts plus explicit reasons for every attempted route."""

    assessments: tuple[DeterministicAssessment, ...] = ()
    attempts: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = STRUCTURED_EVIDENCE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "resolved_count": len(self.assessments),
            "assessments": [value.to_dict() for value in self.assessments],
            "attempts": [dict(value) for value in self.attempts],
        }


def _fold(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _linear_to_srgb(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _canonical_colour(value: Any) -> str | None:
    normalized = _fold(value)
    normalized = _COLOUR_ALIASES.get(normalized, normalized)
    if normalized in _COLOUR_PROTOTYPES_SRGB:
        return normalized
    # Preserve common compound names after punctuation folding.
    for phrase, canonical in _COLOUR_ALIASES.items():
        if _fold(phrase) == normalized:
            return canonical
    return None


def _surface_colour_parameter(name: Any) -> bool:
    normalized = _fold(name).replace(" ", "")
    if any(
        token in normalized
        for token in ("emissive", "specular", "subsurface", "opacity")
    ):
        return False
    return normalized in {
        "albedo",
        "basecolor",
        "color",
        "colour",
        "diffuse",
        "diffusecolor",
        "surfacecolor",
        "tint",
    } or normalized.endswith(("basecolor", "surfacecolor", "tint"))


def _colour_observation(
    linear_rgba: Sequence[Any],
) -> tuple[str, dict[str, Any]] | None:
    if len(linear_rgba) < 3:
        return None
    try:
        linear_rgb = tuple(float(value) for value in linear_rgba[:3])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in linear_rgb):
        return None
    srgb = tuple(_linear_to_srgb(value) for value in linear_rgb)
    distances = sorted(
        (
            (
                math.sqrt(
                    sum(
                        (observed - wanted) ** 2
                        for observed, wanted in zip(
                            srgb,
                            prototype,
                            strict=True,
                        )
                    )
                ),
                name,
                prototype,
            )
            for name, prototype in _COLOUR_PROTOTYPES_SRGB.items()
        )
    )
    distance, observed_colour, prototype = distances[0]
    if distance > 0.30 or distances[1][0] - distance < 0.05:
        return None
    return observed_colour, {
        "observed_colour": observed_colour,
        "linear_rgb": [round(value, 6) for value in linear_rgb],
        "srgb": [round(value, 6) for value in srgb],
        "prototype_srgb": list(prototype),
        "distance": round(distance, 6),
        "maximum_distance": 0.30,
    }


def _raw_actor_index(scene: Any) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for actor in scene.candidate_actors():
        try:
            key = actor_identity(actor)
        except Exception:  # noqa: BLE001 - malformed Actors cannot prove a claim
            continue
        result[key.casefold()] = actor
    return result


def _unique_exact_actor(
    entity: EntityNode,
    stage1: Stage1Result,
    raw_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if entity.referent_kind is not ReferentKind.INDIVIDUAL:
        return None
    assessment = stage1.for_entity(entity.id)
    if (
        assessment is None
        or assessment.assembly_matches
        or not assessment.matches
    ):
        return None
    ids = tuple(dict.fromkeys(value.casefold() for value in assessment.actor_ids))
    if len(ids) != 1:
        return None
    actor = raw_by_id.get(ids[0])
    if actor is None:
        return None
    return actor


def _quantity_bounds(quantity: Mapping[str, Any]) -> dict[str, int | None] | None:
    mode = str(quantity.get("mode") or "").casefold()
    value = quantity.get("value")
    lower = quantity.get("lower")
    upper = quantity.get("upper")
    if mode in {"exact", "approximately", "at_least", "at_most", "more_than", "less_than"}:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
    if mode == "exact":
        return {"min_count": None, "max_count": None, "exact_count": value}
    if mode == "approximately":
        tolerance = max(1, math.ceil(float(value) * 0.15))
        return {
            "min_count": max(0, value - tolerance),
            "max_count": value + tolerance,
            "exact_count": None,
        }
    if mode == "at_least":
        return {"min_count": value, "max_count": None, "exact_count": None}
    if mode == "at_most":
        return {"min_count": None, "max_count": value, "exact_count": None}
    if mode == "more_than":
        return {"min_count": value + 1, "max_count": None, "exact_count": None}
    if mode == "less_than":
        return {
            "min_count": None,
            "max_count": max(-1, value - 1),
            "exact_count": None,
        }
    if mode == "range":
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (lower, upper)):
            return None
        if lower < 0 or upper < lower:
            return None
        return {"min_count": lower, "max_count": upper, "exact_count": None}
    if mode == "qualitative":
        qualitative = str(quantity.get("qualitative") or "").casefold()
        values = {
            "none": {"min_count": None, "max_count": None, "exact_count": 0},
            "almost_none": {"min_count": None, "max_count": 1, "exact_count": None},
            "few": {"min_count": 2, "max_count": 4, "exact_count": None},
            "several": {"min_count": 3, "max_count": 7, "exact_count": None},
            "many": {"min_count": 8, "max_count": None, "exact_count": None},
        }
        return values.get(qualitative)
    return None


def _count_score(observed: int, bounds: Mapping[str, int | None]) -> float:
    if bounds.get("exact_count") is not None:
        expected = int(bounds["exact_count"])
        return max(0.0, 1.0 - abs(observed - expected) / max(observed, expected, 1))
    qualities: list[float] = []
    minimum = bounds.get("min_count")
    maximum = bounds.get("max_count")
    if minimum is not None:
        qualities.append(min(1.0, observed / minimum) if minimum > 0 else 1.0)
    if maximum is not None:
        qualities.append(1.0 if observed <= maximum else maximum / max(observed, 1))
    return min(qualities) if qualities else 0.0


def _quantity_assessment(
    bundle: FrozenVerificationBundle,
    binding: Any,
    node: PredicateNode,
    scene: Any,
    inventory: SceneInventoryEvidence,
    stage1: Stage1Result,
) -> tuple[DeterministicAssessment | None, str]:
    quantity = node.semantic_parameters.get("quantity")
    if not isinstance(quantity, Mapping):
        return None, "quantity_payload_missing"
    unit = _fold(quantity.get("unit"))
    if unit not in _OBJECT_COUNT_UNITS:
        return None, "quantity_unit_is_not_object_count"
    bounds = _quantity_bounds(quantity)
    if bounds is None:
        return None, "quantity_mode_has_no_structured_policy"
    arguments = bundle.graph.arguments_for(node.id)
    if len(arguments) != 1:
        return None, "quantity_requires_one_collection"
    entity = bundle.graph.node(arguments[0].target_id)
    if not isinstance(entity, EntityNode):
        return None, "quantity_subject_is_not_entity"
    scope = inventory.scope(binding.scope_for_entity(entity.id))
    if not scope.resolved:
        return None, "quantity_population_scope_unresolved"
    raw_by_id = _raw_actor_index(scene)
    eligible = {value.casefold() for value in stage1.eligible_actor_ids}
    actors = [
        raw_by_id[actor_id.casefold()]
        for actor_id in scope.actor_ids
        if actor_id.casefold() in raw_by_id
        and actor_id.casefold() in eligible
    ]
    selector = {"allowed_categories": [entity.name, *entity.aliases]}
    selection_evidence: list[dict[str, Any]]
    if scene.asset_catalog_id:
        if any(
            actor.get("asset_category")
            and actor.get("asset_category_source") != "asset_catalog"
            for actor in actors
        ):
            return None, "category_not_resolved_by_asset_catalog"
        selection = build_semantic_graph(actors).select(selector)
        if not selection.objects:
            # A zero match is not proof that an open semantic category is absent;
            # the category vocabulary may simply be narrower than the prompt.
            return None, "no_trusted_category_match"
        observed = len(selection.objects)
        population_complete = selection.complete
        selector_coverage: Mapping[str, Any] = selection.coverage()
        selection_evidence = [
            {
                "evaluator_object_id": value.id,
                "actor_ids": [actor_identity(actor) for actor in value.actors],
            }
            for value in selection.objects
        ]
        measurement = "ue_scene_graph_logical_object_count"
        population_source = "trusted_asset_catalog"
    else:
        assessment = stage1.for_entity(entity.id)
        if assessment is None or not assessment.matches:
            return None, "canonical_name_population_not_grounded"
        scope_ids = {value.casefold() for value in scope.actor_ids}
        actor_ids = tuple(
            dict.fromkeys(
                match.actor_id
                for match in assessment.matches
                if match.actor_id.casefold() in scope_ids
                and match.actor_id.casefold() in raw_by_id
                and match.actor_id.casefold() in eligible
            )
        )
        if not actor_ids:
            return None, "canonical_name_population_not_grounded"
        observed = len(actor_ids)
        # Name matching proves a lower bound.  It cannot prove that an open
        # semantic category contains no differently named Actors, so exact or
        # upper-bound claims remain visual unless a catalog supplied closure.
        population_complete = False
        selector_coverage = {
            "complete": False,
            "matched_actor_count": observed,
            "population_kind": "canonical_name_lower_bound",
        }
        selection_evidence = [
            {
                "actor_ids": [actor_id],
                "name_match": next(
                    match.to_dict()
                    for match in assessment.matches
                    if match.actor_id == actor_id
                ),
            }
            for actor_id in actor_ids
        ]
        measurement = "ue_canonical_name_grounded_actor_lower_bound"
        population_source = "stage1_canonical_name_grounding"
    expected = {
        "selector": selector,
        **bounds,
        "approximation_policy": (
            "relative_15_percent_minimum_one"
            if str(quantity.get("mode")).casefold() == "approximately"
            else None
        ),
    }
    lower_met_despite_incomplete = (
        not population_complete
        and bounds.get("min_count") is not None
        and bounds.get("max_count") is None
        and bounds.get("exact_count") is None
        and observed >= int(bounds["min_count"])
    )
    if not population_complete and not lower_met_despite_incomplete:
        return None, "category_population_incomplete"
    holds = (
        (bounds.get("min_count") is None or observed >= int(bounds["min_count"]))
        and (bounds.get("max_count") is None or observed <= int(bounds["max_count"]))
        and (
            bounds.get("exact_count") is None
            or observed == int(bounds["exact_count"])
        )
    )
    negated = node.polarity is Polarity.NEGATED
    if negated:
        holds = not holds
    verdict = ClaimVerdict.MATCH if holds else ClaimVerdict.MISMATCH
    return DeterministicAssessment(
        node_id=node.id,
        requirement_id=binding.requirement_id,
        rule_id=f"structured_v2_quantity:{node.id}",
        family="STRUCTURED_QUANTITY",
        verdict=verdict,
        rationale=(
            f"the {population_source} contains {observed} distinct logical "
            "object(s)"
        ),
        check={
            "status": report_contracts.PASS if holds else report_contracts.FAIL,
            "measurement": measurement,
            "expected": expected,
            "observed": {
                "count": observed,
                "count_unit": "logical_object",
                "selector_coverage": selector_coverage,
                "population_source": population_source,
            },
            "evidence": selection_evidence,
            "score": (
                1.0 if holds else 0.0
            ) if negated else round(_count_score(observed, bounds), 6),
        },
        resolution_stage="stage1_structured",
    ), "resolved"


def _relation_assessment(
    bundle: FrozenVerificationBundle,
    binding: Any,
    node: PredicateNode,
    scene: Any,
    stage1: Stage1Result,
) -> tuple[DeterministicAssessment | None, str]:
    relation = _fold(node.semantic_parameters.get("relation")).replace(" ", "_")
    relation = _RELATION_ALIASES.get(relation, relation)
    if relation not in _STRUCTURED_RELATIONS:
        return None, "relation_needs_visual_or_undeclared_threshold"
    arguments = bundle.graph.arguments_for(node.id)
    if len(arguments) != 2:
        return None, "relation_requires_two_arguments"
    by_role = {value.role: value for value in arguments}
    subject_edge = by_role.get("subject") or arguments[0]
    object_edge = by_role.get("reference") or by_role.get("object") or arguments[1]
    subject_entity = bundle.graph.node(subject_edge.target_id)
    object_entity = bundle.graph.node(object_edge.target_id)
    if not isinstance(subject_entity, EntityNode) or not isinstance(object_entity, EntityNode):
        return None, "relation_arguments_are_not_entities"
    raw_by_id = _raw_actor_index(scene)
    subject = _unique_exact_actor(subject_entity, stage1, raw_by_id)
    object_actor = _unique_exact_actor(object_entity, stage1, raw_by_id)
    if subject is None or object_actor is None:
        return None, "relation_actors_not_uniquely_grounded"
    rule: dict[str, Any] = {"relation": relation, "quantifier": "all_pairs"}
    if relation == "on_top_of":
        rule.update({"tolerance_cm": 10.0, "minimum_overlap_fraction": 0.1})
    elif relation == "inside":
        rule["tolerance_cm"] = 2.0
    elif relation == "against":
        rule["maximum_surface_gap_cm"] = 10.0
    try:
        result = spatial_relations.evaluate(rule, [subject], [object_actor])
    except spatial_relations.RelationError:
        return None, "canonical_relation_evaluator_declined"
    relation_holds = bool(result["pass"])
    holds = relation_holds
    if node.polarity is Polarity.NEGATED:
        holds = not holds
    verdict = ClaimVerdict.MATCH if holds else ClaimVerdict.MISMATCH
    return DeterministicAssessment(
        node_id=node.id,
        requirement_id=binding.requirement_id,
        rule_id=f"structured_v2_relation:{node.id}",
        family="STRUCTURED_SPATIAL_RELATION",
        verdict=verdict,
        rationale=(
            "the canonical Actor AABB/Transform observation reports "
            f"{relation}={relation_holds}; with {node.polarity.value} polarity "
            f"the requirement is {'satisfied' if holds else 'not satisfied'}"
        ),
        check={
            "status": report_contracts.PASS if holds else report_contracts.FAIL,
            "measurement": "ue_actor_aabb_transform_relation",
            "expected": {
                **result["expected"],
                "polarity": node.polarity.value,
            },
            "observed": result["observed"],
            "evidence": result["rows"],
            "score": 1.0 if holds else 0.0,
        },
        resolution_stage="stage1_structured",
    ), "resolved"


def _colour_assessment(
    bundle: FrozenVerificationBundle,
    binding: Any,
    node: PredicateNode,
    scene: Any,
    stage1: Stage1Result,
) -> tuple[DeterministicAssessment | None, str]:
    qualifiers = node.semantic_parameters.get("qualifiers")
    if not isinstance(qualifiers, Sequence) or isinstance(qualifiers, (str, bytes)):
        return None, "colour_qualifier_missing"
    colour_qualifiers = [
        value
        for value in qualifiers
        if isinstance(value, Mapping) and value.get("kind") == "color"
    ]
    if len(colour_qualifiers) != 1:
        return None, "colour_route_requires_one_atomic_colour"
    if any(
        not isinstance(value, Mapping) or value.get("kind") != "color"
        for value in qualifiers
    ):
        return None, "colour_predicate_has_additional_visual_qualifiers"
    expected = _canonical_colour(
        colour_qualifiers[0].get("name")
        or colour_qualifiers[0].get("source_text")
    )
    if expected is None:
        return None, "colour_name_not_in_structured_vocabulary"
    arguments = bundle.graph.arguments_for(node.id)
    if len(arguments) != 1:
        return None, "colour_route_requires_one_subject"
    entity = bundle.graph.node(arguments[0].target_id)
    if not isinstance(entity, EntityNode):
        return None, "colour_subject_is_not_entity"
    raw_by_id = _raw_actor_index(scene)
    actor = _unique_exact_actor(entity, stage1, raw_by_id)
    if actor is None:
        return None, "colour_subject_not_uniquely_grounded"
    catalog = scene.candidate.get("material_parameter_catalog")
    if not isinstance(catalog, Mapping):
        return None, "material_parameter_catalog_missing"
    material_paths = tuple(
        dict.fromkeys(str(value) for value in actor.get("material_paths") or ())
    )
    if not material_paths:
        return None, "subject_has_no_resolved_materials"
    observations: list[dict[str, Any]] = []
    for path in material_paths:
        entry = catalog.get(str(path))
        if not isinstance(entry, Mapping):
            return None, "material_catalog_entry_missing"
        if (
            entry.get("schema_version") != "1.0"
            or entry.get("probe_status") != "success"
            or entry.get("source")
            != "ue_material_editing_library_resolved_parameters_v1"
        ):
            return None, "material_parameter_probe_not_trusted"
        used_textures = entry.get("used_texture_paths")
        # None means the UE API could not prove whether a texture participates.
        if used_textures is None or used_textures:
            return None, "material_texture_participation_is_not_empty"
        vectors = entry.get("resolved_vector_parameters")
        if not isinstance(vectors, Mapping):
            return None, "resolved_vector_parameters_missing"
        colour_parameters = [
            (parameter_name, rgba)
            for parameter_name, rgba in vectors.items()
            if _surface_colour_parameter(parameter_name)
        ]
        # A material graph with several colour-looking controls is not
        # reducible to one visible surface colour without evaluating the graph.
        if len(colour_parameters) != 1:
            return None, "surface_colour_parameter_is_not_unique"
        parameter_name, rgba = colour_parameters[0]
        if not isinstance(rgba, Sequence) or isinstance(rgba, (str, bytes)):
            return None, "surface_colour_parameter_is_not_vector"
        observation = _colour_observation(rgba)
        if observation is None:
            return None, "surface_colour_parameter_is_ambiguous"
        observations.append(
            {
                "actor_id": actor_identity(actor),
                "material_path": str(path),
                "parameter_name": str(parameter_name),
                **observation[1],
            }
        )
    observed_colours = {
        value["observed_colour"] for value in observations
    }
    if len(observed_colours) != 1:
        return None, "subject_material_colours_disagree"
    observed = next(iter(observed_colours))
    holds = observed == expected
    if node.polarity is Polarity.NEGATED:
        holds = not holds
    verdict = ClaimVerdict.MATCH if holds else ClaimVerdict.MISMATCH
    return DeterministicAssessment(
        node_id=node.id,
        requirement_id=binding.requirement_id,
        rule_id=f"structured_v2_colour:{node.id}",
        family="STRUCTURED_MATERIAL_COLOUR",
        verdict=verdict,
        rationale=(
            "all texture-free, unambiguous UE material colour parameters "
            f"resolve to {observed}; the requirement expects {expected}"
        ),
        check={
            "status": report_contracts.PASS if holds else report_contracts.FAIL,
            "measurement": "ue_resolved_material_vector_parameter",
            "expected": {"colour": expected},
            "observed": {
                "colour": observed,
                "material_count": len(observations),
            },
            "evidence": observations,
            "score": 1.0 if holds else 0.0,
        },
        resolution_stage="stage1_structured",
    ), "resolved"


def evaluate_structured_requirements(
    bundle: FrozenVerificationBundle,
    scene: Any,
    inventory: SceneInventoryEvidence,
    stage1: Stage1Result,
) -> StructuredEvaluation:
    """Resolve the safe v2 metadata subset and leave every other leaf visual."""

    assessments: list[DeterministicAssessment] = []
    attempts: list[dict[str, Any]] = []
    for binding in bundle.requirements:
        if (
            binding.primary_owner != "requirement_graph"
            or binding.status is not RequirementStatus.SUPPORTED
            or binding.deterministic_rule_id is not None
        ):
            continue
        node = bundle.graph.node(binding.node_id)
        if not isinstance(node, PredicateNode):
            continue
        semantic_type = str(
            node.semantic_parameters.get("requirement_type") or ""
        ).casefold()
        if semantic_type not in {
            "attribute",
            "material",
            "quantity",
            "spatial_relation",
        }:
            continue
        if not scene.independent:
            attempts.append(
                {
                    "node_id": node.id,
                    "requirement_id": binding.requirement_id,
                    "semantic_type": semantic_type,
                    "status": "visual_fallback",
                    "reason": "scene_snapshot_not_from_independent_scorer",
                }
            )
            continue
        evaluator = None
        evaluator_args: tuple[Any, ...] = ()
        if semantic_type == "quantity":
            evaluator = _quantity_assessment
            evaluator_args = (
                bundle,
                binding,
                node,
                scene,
                inventory,
                stage1,
            )
        elif semantic_type == "spatial_relation":
            evaluator = _relation_assessment
            evaluator_args = (
                bundle,
                binding,
                node,
                scene,
                stage1,
            )
        elif semantic_type in {"attribute", "material"}:
            evaluator = _colour_assessment
            evaluator_args = (
                bundle,
                binding,
                node,
                scene,
                stage1,
            )
        if evaluator is None:
            continue
        try:
            assessment, reason = evaluator(*evaluator_args)
        except Exception as error:  # noqa: BLE001 - visual fallback is fail-closed
            assessment = None
            reason = f"{type(error).__name__}: {error}"
            status = "error_fallback_to_visual"
        else:
            status = "resolved" if assessment is not None else "visual_fallback"
        attempts.append(
            {
                "node_id": node.id,
                "requirement_id": binding.requirement_id,
                "semantic_type": semantic_type,
                "status": status,
                "reason": reason,
            }
        )
        if assessment is not None:
            assessments.append(assessment)
    return StructuredEvaluation(tuple(assessments), tuple(attempts))


__all__ = [
    "STRUCTURED_EVIDENCE_VERSION",
    "StructuredEvaluation",
    "evaluate_structured_requirements",
]
