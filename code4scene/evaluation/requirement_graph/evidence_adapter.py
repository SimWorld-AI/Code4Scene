"""Adapt Code4Scene's one-pass scene export to requirement-graph evidence.

This module owns population resolution.  No downstream stage may silently
reinterpret ``additions`` as ``candidate_all``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from code4scene.evaluation.scene_diff import (
    ActorIdentityError,
    actor_identity,
    diff_scenes,
)
from code4scene.evaluation.selection import category_from_actor
from code4scene.evaluation.ue_evidence import SceneEvidence

from .actor_inventory import (
    ActorBounds,
    ActorDescriptor,
    ActorInventorySnapshot,
    AssemblyDescriptor,
    AssemblyMemberRole,
    IdentitySource,
    IdentityStrength,
    IdentityTerm,
    InventoryStatus,
    build_actor_descriptor,
)
from .bundle import PopulationScope, UnknownReason
from .contracts import SceneBounds

ADAPTER_SOURCE = "scenebench_scene_evidence_adapter_v1"


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 3:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _actor_bounds(actor: Mapping[str, Any]) -> ActorBounds | None:
    raw = actor.get("bounds")
    raw = raw if isinstance(raw, Mapping) else {}
    center = _vector3(raw.get("origin_cm") or raw.get("center_cm"))
    if center is None:
        transform = actor.get("transform")
        transform = transform if isinstance(transform, Mapping) else {}
        center = _vector3(transform.get("location_cm"))
    extent = _vector3(raw.get("extent_cm"))
    if center is None or extent is None or any(value < 0.0 for value in extent):
        return None
    return ActorBounds(center, extent)


def _descriptor(actor: Mapping[str, Any]) -> ActorDescriptor:
    bounds = _actor_bounds(actor)
    diagnostics = actor.get("export_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    hidden = diagnostics.get("hidden_in_editor")
    category = _text(category_from_actor(actor))
    component_paths = tuple(
        _text(value)
        for value in actor.get("component_asset_paths") or ()
        if _text(value)
    )
    generated = build_actor_descriptor(
        live_actor_id=actor_identity(actor),
        unreal_name=_text(actor.get("name")) or None,
        actor_class=_text(actor.get("class")) or None,
        asset_path=_text(actor.get("asset_path")) or None,
        actor_label=_text(actor.get("label")) or None,
        structured_tags=actor.get("actor_tags") or (),
        identity_tags=((category,) if category else ()),
        context_tags=tuple(
            value
            for value in (
                _text(actor.get("actor_role")),
                *(_text(value) for value in actor.get("material_paths") or ()),
            )
            if value
        ),
        locator_asset_paths=component_paths,
        bounds=bounds,
        active=True,
        # Successful Code4Scene exports enumerate current-level actors.
        # An explicit hidden flag is authoritative; missing diagnostics in an
        # older configured artifact remain usable for backward compatibility.
        renderable=(not hidden if isinstance(hidden, bool) else True),
        in_current_level=True,
    )
    if not category:
        return generated
    # Code4Scene's semantic category is an explicit authored/exported
    # identity.  Opaque UE asset/class names remain valuable camera locators,
    # but must not outrank that category (for example SM_Chair vs "chair").
    return ActorDescriptor(
        live_actor_id=generated.live_actor_id,
        unreal_name=generated.unreal_name,
        actor_class=generated.actor_class,
        asset_path=generated.asset_path,
        actor_label=generated.actor_label,
        identity_terms=(
            IdentityTerm(
                category,
                IdentitySource.STRUCTURED_TAG,
                IdentityStrength.STRONG,
                raw_value=category,
            ),
        ),
        context_terms=generated.context_terms,
        locator_terms=tuple(
            dict.fromkeys((*generated.locator_terms, *generated.identity_terms))
        ),
        identity_term_diagnostics=generated.identity_term_diagnostics,
        bounds=generated.bounds,
        active=generated.active,
        renderable=generated.renderable,
        in_current_level=generated.in_current_level,
        multiplicity=generated.multiplicity,
        instance_count_hint=generated.instance_count_hint,
    )


def _union_bounds(values: Sequence[ActorBounds]) -> ActorBounds | None:
    if not values:
        return None
    minimum = tuple(min(value.min_cm[i] for value in values) for i in range(3))
    maximum = tuple(max(value.max_cm[i] for value in values) for i in range(3))
    return ActorBounds.from_min_max(minimum, maximum)


def _assembly_descriptors(
    actors: Sequence[Mapping[str, Any]],
    descriptors: Mapping[str, ActorDescriptor],
) -> tuple[AssemblyDescriptor, ...]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for actor in actors:
        logical_id = _text(actor.get("logical_object_id"))
        if logical_id:
            groups[logical_id].append(actor)

    result: list[AssemblyDescriptor] = []
    for logical_id, members in sorted(groups.items()):
        member_ids = tuple(actor_identity(actor) for actor in members)
        categories = tuple(
            dict.fromkeys(
                _text(category_from_actor(actor))
                for actor in members
                if _text(category_from_actor(actor))
            )
        )
        member_bounds = tuple(
            descriptor.bounds
            for actor in members
            if (descriptor := descriptors[actor_identity(actor)]).bounds is not None
        )
        roles = tuple(
            AssemblyMemberRole(actor_identity(actor), _text(actor.get("actor_role")))
            for actor in members
            if _text(actor.get("actor_role"))
        )
        identity_terms = (
            tuple(
                IdentityTerm(value, IdentitySource.STRUCTURED_TAG)
                for value in categories
            )
            if len(categories) == 1
            else ()
        )
        result.append(
            AssemblyDescriptor(
                assembly_id=logical_id,
                identity_terms=identity_terms,
                member_actor_ids=member_ids,
                member_roles=roles,
                declaration_source=(
                    "scene_snapshot.logical_object_id+semantic_category"
                    if identity_terms
                    else "scene_snapshot.logical_object_id"
                ),
                identity_declared=bool(identity_terms),
                membership_complete=len(member_bounds) == len(members),
                bounds=_union_bounds(member_bounds),
                active=True,
                renderable=all(
                    descriptors[value].renderable is True for value in member_ids
                ),
                in_current_level=True,
            )
        )
    return tuple(result)


def _scene_bounds(
    descriptors: Sequence[ActorDescriptor], half_extent_m: float | None
) -> SceneBounds:
    # The CONTENT box, which is what every consumer of this actually wants:
    # `generate_scout_poses` frames it, and `clamp_camera_pose` calls it "the
    # content box" in its own docstring.
    #
    # The world plate is a different fact. It is the constraint — the region an
    # agent may build in, centred on the world origin because that is what the
    # bounds pass enforces — and it says nothing about where the content ended
    # up. Overwriting the measured X/Y with it aimed every scout camera at the
    # centre of the PLATE instead of at the scene.
    #
    # On a level that sits off the origin the two are far apart. Measured on
    # Hangar: content spans X -80..11 m and Y -118..-24 m, centred on
    # (-34, -71); its task declares `size_m: 380`, the smallest origin-centred
    # plate that contains it. The override moved the framing target 79 m away
    # from the scene and pushed the four eye-level cameras out to the plate
    # edge at ±190 m, looking inward at empty ground. Stage 2's judge reported
    # "the provided frames show only sky, clouds, and a horizon line" and
    # refused every visual claim — a refusal whose stated cause (no evidence)
    # was true and whose real cause was the camera.
    #
    # The plate remains the answer in exactly one case: an empty scene, where
    # there is no content to describe and the canvas is all there is to say.
    bounded_all = tuple(
        value.bounds for value in descriptors if value.bounds is not None
    )
    frameable = tuple(
        value.bounds
        for value in descriptors
        if value.bounds is not None and not _is_camera_framing_proxy(value)
    )
    # Sky domes and authored camera proxy meshes are inventory Actors, but
    # their AABBs are not scene content.  A common Engine SkySphere spans
    # roughly 32 km and previously placed four indoor overview cameras at
    # +/-16 km.  Keep proxies in identity/scoping evidence; omit them only
    # from the content box used to aim cameras.  If a level contains nothing
    # else, retain the old all-bounds fallback so the box remains usable.
    bounded = frameable or bounded_all
    extent: float | None = None
    if half_extent_m is not None:
        extent = float(half_extent_m) * 100.0
        if not math.isfinite(extent) or extent <= 0.0:
            raise ValueError("half_extent_m must be finite and positive")
    if bounded:
        minimum = [min(value.min_cm[i] for value in bounded) for i in range(3)]
        maximum = [max(value.max_cm[i] for value in bounded) for i in range(3)]
    elif extent is not None:
        minimum, maximum = [-extent, -extent, -100.0], [extent, extent, 100.0]
    else:
        minimum, maximum = [-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]
    for index in range(3):
        if maximum[index] <= minimum[index]:
            minimum[index] -= 1.0
            maximum[index] += 1.0
    return SceneBounds(tuple(minimum), tuple(maximum))


def _is_camera_framing_proxy(descriptor: ActorDescriptor) -> bool:
    """Return whether an Actor's bounds describe a view helper, not content."""

    values = (
        descriptor.actor_class,
        descriptor.asset_path,
        descriptor.actor_label,
        descriptor.unreal_name,
    )
    normalized = tuple(
        re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
        for value in values
    )
    return any(
        token in value
        for value in normalized
        for token in ("skysphere", "cameraactor", "cinecameraactor")
    )


@dataclass(frozen=True, slots=True)
class ScopeResolution:
    scope: PopulationScope | str
    actor_ids: tuple[str, ...] = ()
    source: str | None = None
    unknown_reason: UnknownReason | str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", PopulationScope(self.scope))
        object.__setattr__(self, "actor_ids", tuple(dict.fromkeys(self.actor_ids)))
        if self.unknown_reason is not None:
            object.__setattr__(self, "unknown_reason", UnknownReason(self.unknown_reason))
        if self.source is None and self.unknown_reason is None:
            raise ValueError("a scope resolution needs a source or unknown_reason")
        if self.source is not None and self.unknown_reason is not None:
            raise ValueError("a scope cannot be both resolved and unknown")

    @property
    def resolved(self) -> bool:
        return self.unknown_reason is None


@dataclass(frozen=True, slots=True)
class SceneInventoryEvidence:
    inventory: ActorInventorySnapshot
    scene_bounds: SceneBounds
    scopes: Mapping[PopulationScope, ScopeResolution]

    def scope(self, value: PopulationScope | str) -> ScopeResolution:
        return self.scopes[PopulationScope(value)]

    def actors_for(self, value: PopulationScope | str) -> tuple[ActorDescriptor, ...]:
        resolution = self.scope(value)
        if not resolution.resolved:
            return ()
        wanted = set(resolution.actor_ids)
        return tuple(
            actor for actor in self.inventory.actors if actor.live_actor_id in wanted
        )


def adapt_scene_snapshot(
    scene: Mapping[str, Any],
    *,
    half_extent_m: float | None = None,
) -> tuple[ActorInventorySnapshot, SceneBounds]:
    """Adapt one rich Code4Scene export for geometry-only camera probes.

    Camera acquisition needs the same sanitized Actor identities and proxy-free
    content bounds as RequirementGraph, but it does not own or reinterpret any
    semantic population scope.  Keeping this as a public adapter prevents the
    paired GT camera path from growing a second snapshot schema.
    """

    if not isinstance(scene, Mapping):
        raise TypeError("scene snapshot must be a mapping")
    raw_actors = scene.get("actors")
    if not isinstance(raw_actors, Sequence) or isinstance(
        raw_actors, (str, bytes, bytearray)
    ):
        raise ValueError("scene snapshot must contain an actors array")
    actors = tuple(value for value in raw_actors if isinstance(value, Mapping))
    if len(actors) != len(raw_actors):
        raise ValueError("scene snapshot actors must all be mappings")
    descriptors = tuple(_descriptor(actor) for actor in actors)
    by_id = {value.live_actor_id: value for value in descriptors}
    errors = tuple(
        f"{descriptor.live_actor_id}: missing valid bounds"
        for descriptor in descriptors
        if descriptor.bounds is None
    )
    inventory = ActorInventorySnapshot(
        actors=descriptors,
        assemblies=_assembly_descriptors(actors, by_id),
        status=InventoryStatus.COMPLETE if not errors else InventoryStatus.PARTIAL,
        source=f"{ADAPTER_SOURCE}:paired_camera",
        errors=errors,
    )
    return inventory, _scene_bounds(descriptors, half_extent_m)


def _provenance_additions(
    actors: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any] | None
) -> ScopeResolution:
    if not isinstance(provenance, Mapping) or provenance.get("trusted") is not True:
        return ScopeResolution(
            PopulationScope.ADDITIONS,
            unknown_reason=UnknownReason.PROVENANCE_UNTRUSTED,
            detail="no trusted operation provenance was supplied",
        )
    requested = {
        _text(value)
        for value in provenance.get("added_actor_ids") or ()
        if _text(value)
    }
    origins = {
        _text(value).casefold()
        for value in provenance.get("added_actor_origins") or ()
        if _text(value)
    }
    selected: list[str] = []
    for actor in actors:
        aliases = {
            actor_identity(actor),
            _text(actor.get("stable_actor_id")),
            _text(actor.get("actor_guid")),
            _text(actor.get("actor_path")),
            _text(actor.get("label")),
        }
        origin = _text(actor.get("actor_origin")).casefold()
        if requested.intersection(aliases) or (origin and origin in origins):
            selected.append(actor_identity(actor))
    if not requested and not origins:
        return ScopeResolution(
            PopulationScope.ADDITIONS,
            unknown_reason=UnknownReason.SCOPE_UNRESOLVED,
            detail="trusted provenance names no added actor ids or actor origins",
        )
    return ScopeResolution(
        PopulationScope.ADDITIONS,
        tuple(selected),
        source="trusted_operation_provenance",
        detail=f"resolved {len(selected)} actor(s)",
    )


def build_scene_inventory(
    evidence: SceneEvidence,
    *,
    half_extent_m: float | None = None,
    operation_provenance: Mapping[str, Any] | None = None,
    closed_categories: Sequence[str] = (),
    canonical_identity_closed_categories: Sequence[str] = (),
) -> SceneInventoryEvidence:
    """Build immutable Stage 1 evidence and resolve supported populations.

    ``input_scene`` has priority for additions.  Provenance is consulted only
    when no comparable input snapshot exists; the Candidate population is
    never used as an additions fallback.
    """

    if not isinstance(evidence, SceneEvidence):
        raise TypeError("evidence must be SceneEvidence")
    actors = tuple(evidence.candidate_actors())
    descriptors = tuple(_descriptor(actor) for actor in actors)
    by_id = {value.live_actor_id: value for value in descriptors}
    errors: list[str] = []
    for actor, descriptor in zip(actors, descriptors, strict=True):
        if descriptor.bounds is None:
            errors.append(f"{descriptor.live_actor_id}: missing valid bounds")
        errors.extend(
            f"{descriptor.live_actor_id}: {value}"
            for value in actor.get("export_errors") or ()
            if _text(value)
        )
    metadata = evidence.candidate.get("export_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    status = (
        InventoryStatus.COMPLETE
        if metadata.get("status") == "success" and not errors
        else InventoryStatus.PARTIAL
    )
    inventory = ActorInventorySnapshot(
        actors=descriptors,
        assemblies=_assembly_descriptors(actors, by_id),
        status=status,
        closed_categories=tuple(closed_categories),
        canonical_identity_closed_categories=tuple(
            canonical_identity_closed_categories
        ),
        assembly_closed_categories=tuple(closed_categories),
        source=ADAPTER_SOURCE,
        errors=tuple(errors),
    )
    candidate_scope = ScopeResolution(
        PopulationScope.CANDIDATE_ALL,
        tuple(value.live_actor_id for value in descriptors),
        source="candidate_scene_snapshot",
    )
    if evidence.input_scene is not None:
        try:
            diff = diff_scenes(evidence.input_scene, evidence.candidate)
            additions_scope = ScopeResolution(
                PopulationScope.ADDITIONS,
                tuple(actor_identity(actor) for actor in diff.added),
                source="validated_before_after_diff",
                detail=f"resolved {len(diff.added)} actor(s)",
            )
        except (ActorIdentityError, TypeError, ValueError) as error:
            additions_scope = ScopeResolution(
                PopulationScope.ADDITIONS,
                unknown_reason=UnknownReason.SCOPE_UNRESOLVED,
                detail=f"scene diff is invalid: {type(error).__name__}: {error}",
            )
    else:
        additions_scope = _provenance_additions(actors, operation_provenance)
    gt_scope = ScopeResolution(
        PopulationScope.GT_TARGETS,
        unknown_reason=UnknownReason.SCOPE_UNRESOLVED,
        detail="GT target ownership belongs to the existing repair verifiers",
    )
    return SceneInventoryEvidence(
        inventory=inventory,
        scene_bounds=_scene_bounds(descriptors, half_extent_m),
        scopes={
            PopulationScope.CANDIDATE_ALL: candidate_scope,
            PopulationScope.ADDITIONS: additions_scope,
            PopulationScope.GT_TARGETS: gt_scope,
        },
    )


__all__ = [
    "ADAPTER_SOURCE",
    "SceneInventoryEvidence",
    "ScopeResolution",
    "adapt_scene_snapshot",
    "build_scene_inventory",
]
