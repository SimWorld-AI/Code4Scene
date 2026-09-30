"""Deterministic Stage 0 authoring for GT-backed scene-repair tasks.

Image-to-scene repair tasks already own two stronger sources than pixels for
defining what must change: the frozen Input scene graph and the canonical GT
scene graph.  This module diffs those two documents before Candidate is
inspected and lowers only the changed Actors into a RequirementGraph.

Reference images remain agent-facing task inputs.  They are deliberately not
read, hashed, captioned, or sent to a model by this authoring path.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from ..scene_diff import (
    DEFAULT_TOLERANCES,
    Change,
    _property_differences,
    _transform_differences,
    actor_identity,
    actor_summary,
    diff_scenes,
    index_actors,
)
from .bundle import (
    EvidenceClass,
    FrozenVerificationBundle,
    PopulationScope,
    RequirementBinding,
    TaskMode,
    make_evaluation_binding,
)
from .contracts import (
    ArgumentEdge,
    ComparisonOperator,
    EntityEvaluationRoute,
    EntityInventoryRepresentation,
    EntityNode,
    MemberRole,
    NumericConstraint,
    Polarity,
    PredicateNode,
    PredicateType,
    ReferentKind,
    RequirementGraph,
    RequirementMemberEdge,
    RequirementNode,
    RootRequirement,
    SourceSpan,
)


COMPILER_NAME = "scenebench_gt_input_repair_target_authoring"
COMPILER_VERSION = "2.0.0"

# These fields matter to edit-locality and provenance audits, so the shared
# ``scene_diff`` must continue reporting them.  They are not visible scene
# facts, however, and therefore cannot define an image-to-scene semantic
# repair target.  In particular, independently exported Input/GT maps can
# carry different editor labels for the same stable Actor.
_NON_VISUAL_REPAIR_PROPERTY_FIELDS = frozenset(
    {
        "actor_origin",
        "actor_role",
        "actor_tags",
        "label",
        "logical_object_id",
    }
)

# Evaluation/render cameras are collection infrastructure, not visible scene
# content. Input and GT renders may leave different temporary cameras in their
# maps; lowering those differences into semantic requirements would score the
# capture setup instead of the repair task.
_NON_SEMANTIC_ACTOR_CLASS_NAMES = frozenset(
    {
        "cameraactor",
        "cinecameraactor",
        "scenecapture2d",
        "scenecapturecube",
    }
)
_NON_SEMANTIC_ACTOR_ASSET_NAMES = frozenset({"matineecam_sm"})


def _scene(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a scene-graph object")
    actors = value.get("actors")
    if not isinstance(actors, list) or any(
        not isinstance(actor, Mapping) for actor in actors
    ):
        raise ValueError(f"{name}.actors must be an array of Actor objects")
    metadata = value.get("export_metadata")
    if isinstance(metadata, Mapping) and metadata.get("status") != "success":
        raise ValueError(
            f"{name} export status is {metadata.get('status')!r}: "
            f"{metadata.get('error') or 'no error supplied'}"
        )
    return value


def _asset_object_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    leaf = PurePosixPath(text.split(".", 1)[0]).name
    return leaf


def _class_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.rsplit(".", 1)[-1].rsplit("/", 1)[-1]


def _is_semantic_scene_actor(actor: Mapping[str, Any] | None) -> bool:
    """Whether an Actor may define an image-to-scene semantic target."""

    if actor is None:
        return False
    class_name = _class_name(actor.get("class")).casefold()
    asset_name = _asset_object_name(actor.get("asset_path")).casefold()
    if class_name in _NON_SEMANTIC_ACTOR_CLASS_NAMES:
        return False
    if class_name.endswith("cameraactor") or class_name.startswith("scenecapture"):
        return False
    return asset_name not in _NON_SEMANTIC_ACTOR_ASSET_NAMES


def _human_name(value: str) -> str:
    text = re.sub(r"^(?:SM|BP|SK|M|MI)_", "", str(value), flags=re.IGNORECASE)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"[_\-]+", " ", text)
    return " ".join(text.split()).strip()


def _visible_name(value: str) -> str:
    """Remove UE/library organization words that pixels cannot establish."""

    words = _human_name(value).split()
    while words and words[0].casefold() in {"modular", "environment"}:
        words.pop(0)
    return " ".join(words)


def _identity_name(actor: Mapping[str, Any]) -> str:
    for value in (
        _asset_object_name(actor.get("asset_path")),
        str(actor.get("label") or ""),
        str(actor.get("name") or ""),
        _class_name(actor.get("class")),
    ):
        name = _visible_name(value)
        if name:
            return name
    return actor_identity(actor)


def _identity_aliases(actor: Mapping[str, Any], name: str) -> tuple[str, ...]:
    values: list[str] = []
    for raw in (
        actor.get("label"),
        actor.get("name"),
        _asset_object_name(actor.get("asset_path")),
    ):
        text = str(raw or "").strip()
        if text:
            values.extend((text, _human_name(text)))
    canonical = name.casefold()
    return tuple(
        dict.fromkeys(
            value
            for value in (" ".join(item.split()).strip() for item in values)
            if value and value.casefold() != canonical
        )
    )


def _semantic_identity_name(target: RepairTarget) -> str:
    """Return a prompt-free visible identity derived from GT Actor metadata."""

    # Asset revisions/instance ordinals such as Box01, Vase9, or Chair_2 are
    # useful retrieval aliases but are not visible semantic identities.
    name = re.sub(r"(?:[_\- ]?\d+)+$", "", target.name).strip()
    return name or target.name


def _structured_field_projection(
    actor: Mapping[str, Any] | None,
    changed_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    """Keep only GT fields needed to measure a retained-Actor repair."""

    if actor is None:
        return None
    projected: dict[str, Any] = {}
    for identity_key in (
        "stable_actor_id",
        "actor_guid",
        "label",
        "actor_path",
    ):
        if actor.get(identity_key) is not None:
            projected[identity_key] = copy.deepcopy(actor[identity_key])
    transform_keys = {
        "transform.location": "location_cm",
        "transform.rotation": "rotation_deg",
        "transform.scale": "scale",
    }
    for field in changed_fields:
        if field in transform_keys:
            transform = actor.get("transform")
            key = transform_keys[field]
            if isinstance(transform, Mapping) and key in transform:
                projected.setdefault("transform", {})[key] = copy.deepcopy(
                    transform[key]
                )
        elif field.startswith("property."):
            key = field.removeprefix("property.")
            if key in actor:
                projected[key] = copy.deepcopy(actor[key])
    return projected


@dataclass(frozen=True, slots=True)
class RepairTarget:
    """One Actor-level difference that exists before Candidate evaluation."""

    target_id: str
    operation: str
    desired_actor: Mapping[str, Any] | None
    input_actor: Mapping[str, Any] | None
    changed_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in {"add", "remove", "repair"}:
            raise ValueError(f"unsupported repair operation {self.operation!r}")
        if self.operation == "add" and self.desired_actor is None:
            raise ValueError("add targets require desired_actor")
        if self.operation == "remove" and self.input_actor is None:
            raise ValueError("remove targets require input_actor")
        if self.operation == "repair" and (
            self.desired_actor is None or self.input_actor is None
        ):
            raise ValueError("repair targets require Input and desired Actors")
        object.__setattr__(self, "changed_fields", tuple(self.changed_fields))

    @property
    def actor(self) -> Mapping[str, Any]:
        value = self.desired_actor or self.input_actor
        assert value is not None
        return value

    @property
    def name(self) -> str:
        return _identity_name(self.actor)

    @property
    def aliases(self) -> tuple[str, ...]:
        return _identity_aliases(self.actor, self.name)

    @property
    def population_scope(self) -> PopulationScope:
        return (
            PopulationScope.ADDITIONS
            if self.operation == "add"
            else PopulationScope.CANDIDATE_ALL
        )

    def to_dict(self) -> dict[str, Any]:
        desired = actor_summary(self.desired_actor) if self.desired_actor else None
        source = actor_summary(self.input_actor) if self.input_actor else None
        return {
            "target_id": self.target_id,
            "operation": self.operation,
            "identity_name": self.name,
            "identity_aliases": list(self.aliases),
            "input_actor_identity": (
                actor_identity(self.input_actor) if self.input_actor else None
            ),
            "gt_actor_identity": (
                actor_identity(self.desired_actor) if self.desired_actor else None
            ),
            "input_actor": source,
            "gt_actor": desired,
            "changed_fields": list(self.changed_fields),
            "structured_expectation": (
                _structured_field_projection(
                    self.desired_actor,
                    self.changed_fields,
                )
                if self.operation == "repair"
                else None
            ),
            "candidate_population_scope": self.population_scope.value,
        }


def _changes_by_key(values: list[Change]) -> dict[str, Change]:
    return {value.key: value for value in values}


def _canonical_comparison_scene(
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
) -> dict[str, Any]:
    """Project GT onto evidence fields recorded by both Actor snapshots.

    ``diff_scenes`` intentionally treats the left document as the evidence
    schema: fields absent from it are unknown, not empty.  Image-repair Stage
    0 previously passed the rich Input export on the left and the sparse
    answer-key Actor list on the right.  As a result, every Input-only export
    field looked deleted in GT and hundreds of unchanged Actors became repair
    targets.

    GT is the authoritative desired state, so it owns the left side here.  For
    an Actor present in both scenes, retain only fields and transform members
    observed by both schemas.  Actors absent from Input keep their complete GT
    record so addition identity and authoring metadata are not discarded.
    """

    input_by_identity = index_actors(input_scene)
    projected: list[dict[str, Any]] = []
    for desired in gt_scene.get("actors") or ():
        desired_value = dict(desired)
        source = input_by_identity.get(actor_identity(desired))
        if source is not None:
            desired_value = {
                key: value for key, value in desired_value.items() if key in source
            }
            desired_transform = desired_value.get("transform")
            source_transform = source.get("transform")
            if isinstance(desired_transform, Mapping) and isinstance(
                source_transform, Mapping
            ):
                desired_value["transform"] = {
                    key: value
                    for key, value in desired_transform.items()
                    if key in source_transform
                }
        projected.append(desired_value)
    return {**dict(gt_scene), "actors": projected, "actor_count": len(projected)}


def _equivalent_class_representation(
    desired: Mapping[str, Any], source: Mapping[str, Any]
) -> bool:
    """Return whether two UE class paths name the same visible class."""

    desired_name = _class_name(desired.get("class"))
    source_name = _class_name(source.get("class"))
    return bool(desired_name and desired_name == source_name)


def _same_actor_content(desired: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    """Whether two Actors differ at most in identity and non-visual fields."""

    from ..repair_success import _rotation_error, _vector

    left, right = desired.get("transform") or {}, source.get("transform") or {}
    fields = _transform_differences(left, right, DEFAULT_TOLERANCES)
    a, b = _vector(left.get("rotation_deg")), _vector(right.get("rotation_deg"))
    if a is not None and b is not None:
        fields = [f for f in fields if f != "rotation"]
        if _rotation_error(a, b) > DEFAULT_TOLERANCES["rotation_deg"] + 1e-9:
            return False
    if fields:
        return False
    visible = [
        f for f in _property_differences(desired, source)
        if f not in _NON_VISUAL_REPAIR_PROPERTY_FIELDS
    ]
    if "class" in visible and _equivalent_class_representation(desired, source):
        visible.remove("class")
    if visible:
        return False
    return "properties" not in desired or desired.get("properties") == source.get("properties")


def _identity_only_pairs(
    gt_only: list[Mapping[str, Any]], input_only: list[Mapping[str, Any]]
) -> tuple[set[int], set[int]]:
    """One-to-one pairs of GT-only and Input-only Actors with identical content."""

    paired_gt: set[int] = set()
    paired_input: set[int] = set()
    for desired in gt_only:
        for source in input_only:
            if id(source) not in paired_input and _same_actor_content(desired, source):
                paired_gt.add(id(desired))
                paired_input.add(id(source))
                break
    return paired_gt, paired_input


def derive_repair_targets(
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
) -> tuple[RepairTarget, ...]:
    """Return the complete deterministic Actor repair set for Input -> GT."""

    before = _scene(input_scene, "input_scene")
    after = _scene(gt_scene, "gt_scene")
    canonical = _canonical_comparison_scene(before, after)
    # ``diff_scenes`` is evidence-schema asymmetric.  Put canonical GT on the
    # left, then lower its removed/added directions back into Input -> GT
    # repair operations below.
    difference = diff_scenes(canonical, before)
    pending: list[
        tuple[
            str,
            str,
            Mapping[str, Any] | None,
            Mapping[str, Any] | None,
            tuple[str, ...],
        ]
    ] = []

    gt_only = [actor for actor in difference.removed if _is_semantic_scene_actor(actor)]
    input_only = [actor for actor in difference.added if _is_semantic_scene_actor(actor)]
    paired_gt, paired_input = _identity_only_pairs(gt_only, input_only)
    for actor in gt_only:
        if id(actor) not in paired_gt:
            pending.append((actor_identity(actor), "add", actor, None, ("existence",)))
    for actor in input_only:
        if id(actor) not in paired_input:
            pending.append((actor_identity(actor), "remove", None, actor, ("existence",)))

    moved = _changes_by_key(difference.moved)
    modified = _changes_by_key(difference.modified)
    for key in sorted(set(moved) | set(modified)):
        move = moved.get(key)
        modification = modified.get(key)
        change = move or modification
        assert change is not None
        if not (
            _is_semantic_scene_actor(change.before)
            or _is_semantic_scene_actor(change.after)
        ):
            continue
        fields: list[str] = []
        if move is not None:
            fields.extend(f"transform.{value}" for value in move.fields)
        if modification is not None:
            property_fields = [
                field
                for field in modification.fields
                if field not in _NON_VISUAL_REPAIR_PROPERTY_FIELDS
            ]
            if "class" in property_fields and _equivalent_class_representation(
                modification.before, modification.after
            ):
                property_fields.remove("class")
            fields.extend(f"property.{value}" for value in property_fields)
        if not fields:
            continue
        # With canonical GT on the left, ``before`` is the desired Actor and
        # ``after`` is the Input Actor.
        pending.append(
            (key, "repair", change.before, change.after, tuple(fields))
        )

    pending.sort(key=lambda value: (value[0].casefold(), value[0], value[1]))
    return tuple(
        RepairTarget(
            target_id=f"repair_target_{index:04d}",
            operation=operation,
            desired_actor=desired,
            input_actor=source,
            changed_fields=tuple(fields),
        )
        for index, (_, operation, desired, source, fields) in enumerate(
            pending, start=1
        )
    )


def _semantic_repair_units(
    targets: tuple[RepairTarget, ...],
) -> tuple[tuple[RepairTarget, ...], ...]:
    """Group indistinguishable repeated additions into countable collections.

    Two GT Actors backed by the same asset are not individually identifiable
    after an agent creates visually equivalent replacements.  Giving each one
    an independent existence leaf would allow a single Candidate Actor to
    satisfy both leaves.  A collection/count leaf instead measures the shared
    cardinality and group relation once.  Distinct assets and every retained
    Actor repair remain separate semantic units.
    """

    grouped: dict[tuple[str, ...], list[RepairTarget]] = {}
    for target in targets:
        if target.operation == "add":
            identity = _semantic_identity_name(target).casefold()
            key = ("add_collection", identity)
        else:
            key = (target.operation, target.target_id)
        grouped.setdefault(key, []).append(target)
    return tuple(tuple(values) for values in grouped.values())


def _fallback_instruction(target: RepairTarget) -> str:
    if target.operation == "add":
        return f"visibly restore the missing {target.name}"
    if target.operation == "remove":
        return f"visibly remove the obsolete {target.name}"
    fields = set(target.changed_fields)
    if any(field.startswith("transform.rotation") for field in fields):
        return f"visibly restore the {target.name}'s orientation"
    if any(field.startswith("transform.location") for field in fields):
        return f"visibly restore the {target.name}'s position"
    if any(
        token in field
        for field in fields
        for token in ("material", "color", "texture", "appearance")
    ):
        return f"visibly restore the {target.name}'s material appearance"
    return f"visibly restore the {target.name} repair target"


def _predicate_semantics(
    target: RepairTarget, *, group_size: int = 1
) -> tuple[str, PredicateType]:
    """Type the visual leaf by the fact repaired, not always as existence."""

    if target.operation == "add" and group_size > 1:
        return "count", PredicateType.COUNT
    if target.operation in {"add", "remove"}:
        return "exists", PredicateType.EXISTENCE
    fields = set(target.changed_fields)
    material_tokens = ("material", "color", "texture", "appearance")
    if fields and all(
        any(token in field for token in material_tokens) for field in fields
    ):
        return "material", PredicateType.MATERIAL
    if any(field.startswith("transform.rotation") for field in fields):
        return "orientation", PredicateType.ATTRIBUTE
    if any(field.startswith("transform.location") for field in fields):
        return "position", PredicateType.ATTRIBUTE
    if any(field.startswith("transform.scale") for field in fields):
        return "scale", PredicateType.ATTRIBUTE
    return "attribute", PredicateType.ATTRIBUTE


@dataclass(frozen=True, slots=True)
class _FacetDraft:
    """One atomic repair fact before synthetic prompt spans are assigned."""

    suffix: str
    claim: str
    predicate_name: str
    predicate_type: PredicateType
    constraint: NumericConstraint | None = None
    deterministic_rule: Mapping[str, Any] | None = None


def _asset_paths(unit: tuple[RepairTarget, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(target.desired_actor.get("asset_path")).strip()
            for target in unit
            if target.desired_actor is not None
            and target.desired_actor.get("asset_path")
        )
    )


def _addition_identity_claim(
    target: RepairTarget,
    *,
    group_size: int,
) -> str:
    semantic_name = _semantic_identity_name(target)
    if group_size > 1:
        return (
            f"The repaired scene visibly contains at least {group_size} restored "
            f"{semantic_name} objects"
        )
    return (
        f"The repaired scene visibly contains the designated restored "
        f"{semantic_name}, judging only this target's visible category rather "
        "than sibling repair targets"
    )


def _addition_facets(
    unit: tuple[RepairTarget, ...],
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
) -> tuple[_FacetDraft, ...]:
    del input_scene, gt_scene
    target = unit[0]
    group_size = len(unit)
    assets = _asset_paths(unit)
    primary_type = PredicateType.COUNT if group_size > 1 else PredicateType.EXISTENCE
    primary_rule = (
        {
            "type": "object_count" if group_size > 1 else "object_presence",
            "subject": {
                "scope": PopulationScope.ADDITIONS.value,
                "allowed_asset_paths": list(assets),
            },
            "expected": {"min_count": group_size},
            # Exact trusted asset identity can prove the count.  An undercount
            # cannot disprove it because a visually equivalent substitute may
            # use another asset path; that case must continue to Stage 2/3.
            "visual_substitution_fallback": True,
        }
        if assets
        else None
    )
    facets: list[_FacetDraft] = [
        _FacetDraft(
            suffix="identity",
            claim=_addition_identity_claim(target, group_size=group_size),
            predicate_name="count" if group_size > 1 else "exists",
            predicate_type=primary_type,
            constraint=(
                NumericConstraint(ComparisonOperator.GTE, group_size)
                if group_size > 1
                else None
            ),
            deterministic_rule=primary_rule,
        )
    ]
    return tuple(facets)


def compile_repair_target_bundle(
    input_scene: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
    *,
    task_id: str = "image_to_scene_repair",
    bundle_id: str | None = None,
) -> FrozenVerificationBundle:
    """Compile Input/GT Actor differences without image or model input."""

    targets = derive_repair_targets(input_scene, gt_scene)
    if not targets:
        raise ValueError("Input and GT contain no Actor-level repair targets")

    units = _semantic_repair_units(targets)
    facet_groups: tuple[tuple[_FacetDraft, ...], ...] = tuple(
        (
            _addition_facets(unit, input_scene, gt_scene)
            if unit[0].operation == "add"
            else (
                _FacetDraft(
                    suffix="repair",
                    claim=_fallback_instruction(unit[0]),
                    predicate_name=_predicate_semantics(
                        unit[0], group_size=len(unit)
                    )[0],
                    predicate_type=_predicate_semantics(
                        unit[0], group_size=len(unit)
                    )[1],
                ),
            )
        )
        for unit in units
    )
    graph_prompt = ". ".join(
        facet.claim
        for facets in facet_groups
        for facet in facets
    ) + "."
    span_groups: list[tuple[SourceSpan, ...]] = []
    offset = 0
    for facets in facet_groups:
        spans: list[SourceSpan] = []
        for facet in facets:
            spans.append(SourceSpan(offset, offset + len(facet.claim)))
            offset += len(facet.claim) + 2
        span_groups.append(tuple(spans))

    root = RequirementNode(
        id="repair_targets_root",
        text=graph_prompt,
        source_span=SourceSpan(0, len(graph_prompt)),
    )
    nodes: list[Any] = [root]
    edges: list[Any] = []
    requirements: list[RequirementBinding] = []
    target_by_entity: dict[str, dict[str, Any]] = {}
    target_by_node: dict[str, dict[str, Any]] = {}
    deterministic_items: list[dict[str, Any]] = []
    unit_weight = 1.0 / len(units)

    for index, (unit, facets, spans) in enumerate(
        zip(units, facet_groups, span_groups, strict=True), start=1
    ):
        target = unit[0]
        group_size = len(unit)
        semantic_name = _semantic_identity_name(target)
        entity_id = f"repair_e_{index:04d}"
        unit_span = SourceSpan(spans[0].start, spans[-1].end)
        unit_text = graph_prompt[unit_span.start:unit_span.end]
        entity = EntityNode(
            id=entity_id,
            text=unit_text,
            name=semantic_name,
            aliases=target.aliases,
            source_span=unit_span,
            referent_kind=(
                ReferentKind.COLLECTION
                if group_size > 1
                else ReferentKind.INDIVIDUAL
            ),
            # Reuse the ordinary Stage 1-3 evidence ladder. Stage 1 records
            # that pixels are required; Stage 2 performs scoped exact/semantic
            # top-k retrieval and targeted capture; Stage 3 judges visible
            # substitution. This is the same route used by image-authored
            # objects in the original compiler.  Paired GT/Candidate equivalence remains scene_diff's
            # responsibility.
            evaluation_route=EntityEvaluationRoute.STAGE2_VISUAL,
            inventory_representation=(
                EntityInventoryRepresentation.WHOLE_ACTOR_ONLY
            ),
        )
        nodes.append(entity)
        edges.append(
            RequirementMemberEdge(
                root.id, entity_id, MemberRole.SUPPORT_ONLY
            )
        )
        audit = target.to_dict()
        audit["group_size"] = group_size
        audit["member_target_ids"] = [value.target_id for value in unit]
        audit["member_gt_actor_identities"] = [
            actor_identity(value.desired_actor)
            for value in unit
            if value.desired_actor is not None
        ]
        audit["member_gt_geometry"] = [
            {
                "actor_identity": actor_identity(value.desired_actor),
                "asset_path": value.desired_actor.get("asset_path"),
                "location_cm": copy.deepcopy(
                    value.desired_actor.get("transform", {}).get("location_cm")
                ),
                "rotation_deg": copy.deepcopy(
                    value.desired_actor.get("transform", {}).get("rotation_deg")
                ),
                "bounds": copy.deepcopy(value.desired_actor.get("bounds")),
            }
            for value in unit
            if value.desired_actor is not None
            and isinstance(value.desired_actor.get("transform"), Mapping)
            and value.desired_actor.get("transform", {}).get("location_cm")
            is not None
        ]
        target_by_entity[entity_id] = audit
        facet_weight = unit_weight / len(facets)
        for facet_index, (facet, span) in enumerate(
            zip(facets, spans, strict=True)
        ):
            predicate_id = (
                f"repair_p_{index:04d}"
                if facet_index == 0
                else f"repair_p_{index:04d}_{facet.suffix}"
            )
            predicate = PredicateNode(
                id=predicate_id,
                text=facet.claim,
                name=facet.predicate_name,
                predicate_type=facet.predicate_type,
                source_span=span,
                polarity=(
                    Polarity.NEGATED
                    if target.operation == "remove"
                    else Polarity.AFFIRMATIVE
                ),
                constraint=facet.constraint,
            )
            nodes.append(predicate)
            entity_scopes = {entity_id: target.population_scope}
            edges.append(
                ArgumentEdge(
                    predicate_id,
                    entity_id,
                    (
                        "collection"
                        if facet.predicate_type is PredicateType.COUNT
                        else "subject"
                    ),
                )
            )
            rule_id = None
            if facet.deterministic_rule is not None:
                rule_id = f"repair_rule_{index:04d}_{facet.suffix}"
                deterministic_items.append(
                    {
                        "id": rule_id,
                        **dict(facet.deterministic_rule),
                        "prompt_span": [span.start, span.end],
                        "traceability": {
                            "source_span": [span.start, span.end],
                            "source_text": facet.claim,
                            "extractor": "gt_input_repair_atomic_lowering",
                            "rule_id": rule_id,
                        },
                    }
                )
            edges.append(
                RequirementMemberEdge(
                    root.id,
                    predicate_id,
                    MemberRole.SCORED_FACET,
                    facet_weight,
                )
            )
            node_audit = {
                **audit,
                "facet": facet.suffix,
                "facet_claim": facet.claim,
                "deterministic_rule_id": rule_id,
            }
            target_by_node[predicate_id] = node_audit
            requirements.append(
                RequirementBinding(
                    requirement_id=f"atomic_{predicate_id}",
                    node_id=predicate_id,
                    source_span=(span.start, span.end),
                    source_text=facet.claim,
                    primary_owner="requirement_graph",
                    evidence_class=EvidenceClass.GT,
                    population_scope=target.population_scope,
                    entity_scopes=entity_scopes,
                    deterministic_rule_id=rule_id,
                )
            )

    graph = RequirementGraph(
        prompt=graph_prompt,
        nodes=tuple(nodes),
        edges=tuple(edges),
        roots=(RootRequirement(root.id, 1.0),),
    )
    provenance = {
        "case_type": "image_to_scene",
        "authoring_parser": "deterministic_gt_input_scene_diff",
        "compiler_name": COMPILER_NAME,
        "compiler_version": COMPILER_VERSION,
        "authoring_inputs": ["input_scene_graph", "gt_scene_graph"],
        "reference_images_consumed": False,
        "candidate_consumed_during_authoring": False,
        "repair_target_count": len(targets),
        "semantic_repair_unit_count": len(units),
        "repair_targets": [target.to_dict() for target in targets],
        "repair_target_by_entity": target_by_entity,
        "repair_target_by_node": target_by_node,
        "repair_atomic_fact_count": len(requirements),
        "deterministic_rule_draft": {
            "schema_version": "0.1.0",
            "draft_id": f"{task_id}-repair-rules-v1",
            "prompt": graph_prompt,
            "items": deterministic_items,
        },
        "input_actor_count": len(input_scene.get("actors") or ()),
        "gt_actor_count": len(gt_scene.get("actors") or ()),
    }
    evaluations = tuple(
        make_evaluation_binding(requirement, graph, provenance)
        for requirement in requirements
    )
    return FrozenVerificationBundle(
        bundle_id=bundle_id or f"{task_id}-gt-input-repair-targets-v1",
        task_mode=TaskMode.REPAIR,
        graph=graph,
        requirements=tuple(requirements),
        provenance=provenance,
        evaluations=evaluations,
        review={
            "approved": True,
            "reviewer": "deterministic_gt_input_scene_diff",
            "decision": "automatic_schema_validation",
        },
    )


__all__ = [
    "COMPILER_NAME",
    "COMPILER_VERSION",
    "RepairTarget",
    "compile_repair_target_bundle",
    "derive_repair_targets",
]
