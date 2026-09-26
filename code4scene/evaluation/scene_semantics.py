"""What an Actor IS, read off the names the level gives it.

Two physics questions need this and neither can be answered from geometry
alone. Whether an Actor is submerged needs to know which Actors are water, and
water in a level is a large flat mesh — geometrically indistinguishable from a
plaza. Whether an Actor penetrates something solid needs to know which Actors
are solid, and a particle emitter, a light, a fog volume and a group actor all
have bounds that a box test happily reports as intersected.

So the classification is textual: labels, classes, asset paths, categories,
tags, material paths and component classes. That is a heuristic and it is
named as one — `physics.solid_penetration` never turns a broad-phase box
overlap into a failure on its own, and a water surface must also be large and
flat before anything is called submerged.

Ported from the vendored `environment_consistency.js` and
`solid_penetration.js`, whose token lists are kept verbatim: they were tuned
against real imported levels, and a "tidied" list is a different classifier.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

#: A name that means water. Bounded on both sides so `underwater_cave` and
#: `sealant` do not become oceans.
WATER_TOKEN = re.compile(
    r"(?:^|[^a-z])(water(?:body|surface|plane)?\d*|ocean|sea|lake|river|canal"
    r"|pond|pool)(?:[^a-z]|$)", re.I)

#: Terrain whose AABB says almost nothing about where its surface is. A box
#: test against a mountain is meaningless, so these are never decisive.
IRREGULAR_ENVIRONMENT_TOKEN = re.compile(
    r"(?:^|[^a-z0-9])(?:landscape|terrain|mountain|cliff|cave|island|bedrock"
    r"|rockface|rock_?formation)(?:[^a-z0-9]|$)", re.I)

#: Broad, flat surfaces that other Actors rest on are collision participants,
#: but they are not themselves objects whose distance to *another* ground
#: surface is meaningful.  Word tokenization deliberately avoids classifying
#: labels such as ``GroundedCrate`` as ground infrastructure.
GROUND_SUPPORT_WORDS = frozenset({
    "baseplate", "cobblestone", "floor", "ground", "landscape", "pavement",
    "plaza", "road", "sidewalk", "street", "terrain",
})

#: Words that mean "this Actor has bounds but nothing to bump into".
NON_SOLID_SEMANTIC_WORDS = frozenset({
    "emitter", "fire", "flame", "fxsystem", "niagara", "particle",
    "particles", "vfx",
})
NON_SOLID_TYPE_WORDS = NON_SOLID_SEMANTIC_WORDS | {
    "atmosphere", "camera", "fog", "light", "navmesh", "postprocess",
    "reflection", "sky", "volume"}

#: Below this XY span a flat mesh is a puddle prop, not a body of water.
DEFAULT_MINIMUM_SURFACE_SPAN_CM = 200.0
#: Above this vertical half-size the mesh is a volume; its TOP is the surface.
DEFAULT_MAXIMUM_THIN_SURFACE_EXTENT_CM = 100.0
#: How far above or below the waterline still counts as at it.
DEFAULT_WATERLINE_TOLERANCE_CM = 5.0
#: An AABB larger than this decides nothing on its own.
DEFAULT_MAXIMUM_DECISIVE_SPAN_CM = 3000.0

# Physics-role classification is deterministic and scene-relative.  The
# absolute guard catches kilometre-scale proxy geometry even in very small
# scenes; the median multiplier avoids calling an ordinary large building an
# environment proxy merely because it crosses the AABB-decision threshold.
DEFAULT_ENVIRONMENT_PROXY_MINIMUM_SPAN_CM = 30_000.0
DEFAULT_ENVIRONMENT_PROXY_MEDIAN_SPAN_MULTIPLIER = 10.0
DEFAULT_ENVIRONMENT_PROXY_MINIMUM_CONTAINED_ACTOR_COUNT = 3
DEFAULT_ENVIRONMENT_PROXY_MINIMUM_CONTAINED_ACTOR_FRACTION = 0.05

PHYSICS_ROLE_NON_SOLID = "non_solid"
PHYSICS_ROLE_SUPPORT_SURFACE = "support_surface"
PHYSICS_ROLE_ENVIRONMENT_PROXY = "environment_proxy"
PHYSICS_ROLE_SCORED_SOLID = "scored_solid"
PHYSICS_ROLE_UNRESOLVED = "unresolved"

# These are general environment concepts, not case-specific aliases.  A token
# alone is never enough to classify a proxy: the Actor must also be extremely
# large and contain a material share of independently placed scene Actors.
ENVIRONMENT_PROXY_TOKEN = re.compile(
    r"(?:^|[^a-z0-9])(?:background|bedrock|cliff|hill|landscape|mountain"
    r"|rockface|rock_?formation|terrain)(?:[^a-z0-9]|$)", re.I)
GENERIC_BASIC_SHAPE_TOKEN = re.compile(
    r"(?:^|/)engine/basicshapes/(?:cone|cube|cylinder|plane|sphere)(?:[./]|$)",
    re.I,
)


def semantic_values(actor: Mapping[str, Any]) -> list[str]:
    """Every string the level attached to this Actor, for name matching."""
    diagnostics = actor.get("export_diagnostics")
    components = (diagnostics or {}).get("component_classes") or [] \
        if isinstance(diagnostics, Mapping) else []
    values: list[Any] = [actor.get("label"), actor.get("class"),
                         actor.get("asset_path"), actor.get("asset_category"),
                         actor.get("semantic_category"),
                         *(actor.get("actor_tags") or []),
                         *(actor.get("material_paths") or []),
                         *(actor.get("component_asset_paths") or []),
                         *components]
    return [str(value) for value in values if value is not None]


def _asset_leaf(value: Any) -> str | None:
    """Return the asset/object name without package-directory semantics.

    A pack directory describes where an asset came from, not what every asset
    in it is.  In particular, ``/Game/Lighthouse_Island/Meshes/SM_Barrel``
    must not make the barrel an island.  Keeping the final path component
    preserves useful names such as ``SM_Terrain`` and ``M_Cobblestone``.
    """
    if value is None:
        return None
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _actor_semantic_values(actor: Mapping[str, Any]) -> list[str]:
    """Actor-owned semantics, excluding the names of containing packages."""
    values: list[Any] = [
        actor.get("label"), actor.get("class"), actor.get("asset_category"),
        actor.get("semantic_category"), *(actor.get("actor_tags") or []),
        _asset_leaf(actor.get("asset_path")),
        *(_asset_leaf(value) for value in actor.get("component_asset_paths") or []),
    ]
    return [str(value) for value in values if value is not None]


def water_evidence(actor: Mapping[str, Any]) -> list[str]:
    """The names that made this Actor look like water, for the report.

    Actor-level semantics are decisive.  Material and component paths are
    only hints because the export does not carry component-local bounds: a
    thick house Blueprint may contain a water-tank mesh without the house
    itself being a body of water.  Those secondary hints therefore classify
    only a thin parent Actor, where the parent bounds plausibly describe the
    surface named by the component.
    """
    primary_values: list[Any] = [
        actor.get("label"), actor.get("class"), actor.get("asset_path"),
        actor.get("asset_category"), actor.get("semantic_category"),
        *(actor.get("actor_tags") or []),
    ]
    primary = [str(value) for value in primary_values
               if value is not None and WATER_TOKEN.search(str(value))]

    diagnostics = actor.get("export_diagnostics")
    component_classes = ((diagnostics or {}).get("component_classes") or []
                         if isinstance(diagnostics, Mapping) else [])
    secondary_values = [*(actor.get("material_paths") or []),
                        *(actor.get("component_asset_paths") or []),
                        *component_classes]
    secondary = [str(value) for value in secondary_values
                 if value is not None and WATER_TOKEN.search(str(value))]
    if primary:
        return [*primary, *secondary]
    bounds = bounding_box(actor)
    if (bounds is not None
            and bounds["extent"][2] <= DEFAULT_MAXIMUM_THIN_SURFACE_EXTENT_CM):
        return secondary
    return []


def bounding_box(actor: Mapping[str, Any]) -> dict[str, list[float]] | None:
    """The Actor's AABB, or None when it carries no usable bounds.

    None rather than a point: a physics check over an Actor whose size is
    unknown must say it could not evaluate that Actor, not assume it is a dot.
    """
    bounds = actor.get("bounds")
    if not isinstance(bounds, Mapping):
        return None

    def triplet(value: Any) -> list[float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        return numbers if all(math.isfinite(item) for item in numbers) else None

    origin = triplet(bounds.get("origin_cm"))
    extent = triplet(bounds.get("extent_cm"))
    if origin is None or extent is None:
        return None
    extent = [abs(item) for item in extent]
    return {"origin": origin, "extent": extent,
            "min": [origin[i] - extent[i] for i in range(3)],
            "max": [origin[i] + extent[i] for i in range(3)]}


def water_surface(actor: Mapping[str, Any],
                  options: Mapping[str, Any] | None = None
                  ) -> dict[str, Any] | None:
    """This Actor as a water surface, or None if it is not one.

    Three conditions, all required: it is NAMED like water, it is BIG enough
    in XY to be a body of water, and it has usable bounds. A thin mesh's
    surface is its centre plane; a thick one's is its top face, because a
    water volume is modelled as a box whose top is the waterline.
    """
    options = options or {}
    evidence = water_evidence(actor)
    bounds = bounding_box(actor)
    if not evidence or bounds is None:
        return None
    minimum_span = float(options.get("minimum_surface_span_cm")
                         or DEFAULT_MINIMUM_SURFACE_SPAN_CM)
    if bounds["extent"][0] * 2 < minimum_span or bounds["extent"][1] * 2 < minimum_span:
        return None
    thin = float(options.get("maximum_thin_surface_extent_cm")
                 or DEFAULT_MAXIMUM_THIN_SURFACE_EXTENT_CM)
    surface_z = (bounds["origin"][2] if bounds["extent"][2] <= thin
                 else bounds["max"][2])
    return {"actor": actor, "bounds": bounds, "surface_z_cm": surface_z,
            "semantic_evidence": evidence}


def overlaps_xy(left: Mapping[str, list[float]], right: Mapping[str, list[float]],
                tolerance_cm: float = 0.0) -> bool:
    return (left["min"][0] <= right["max"][0] + tolerance_cm
            and left["max"][0] >= right["min"][0] - tolerance_cm
            and left["min"][1] <= right["max"][1] + tolerance_cm
            and left["max"][1] >= right["min"][1] - tolerance_cm)


def _words(value: Any) -> list[str]:
    """`NiagaraActor` -> ['niagara', 'actor']; `BP_Fog_01` -> ['bp','fog','01']."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    return [word for word in re.split(r"[^a-z0-9]+", text.lower()) if word]


def _semantic_words(actor: Mapping[str, Any]) -> set[str]:
    """Normalized Actor-owned words, including a trailing-number-free form.

    Asset packs routinely name tiles ``SM_Floor01`` or ``Wall_003``.  Treating
    ``floor01`` as unrelated to ``floor`` made otherwise obvious structural
    roots disappear.  Only a numeric suffix is removed; containing directory
    names remain excluded by ``_actor_semantic_values``.
    """
    words: set[str] = set()
    aliases = {
        # Common marketplace spelling in the real Office pack.
        "ceilling": "ceiling",
        "folliage": "foliage",
        # Common compact architecture abbreviation.
        "bld": "building",
    }
    for value in _actor_semantic_values(actor):
        for word in _words(value):
            variants = {word, re.sub(r"\d+$", "", word)}
            # Marketplace assets routinely interleave variant letters and
            # digits: ``Wall05b17`` and ``Ceilling01B177``.  Alphabetic runs
            # recover their semantic head without naming any particular case.
            variants.update(re.findall(r"[a-z]+", word))
            for variant in variants:
                if not variant:
                    continue
                words.add(variant)
                words.add(aliases.get(variant, variant))
    role = actor.get("actor_role")
    if role is not None:
        words.update(_words(role))
    return words


def _explicitly_non_collidable(actor: Mapping[str, Any]) -> bool:
    for field in ("collision_enabled", "has_collision", "collidable"):
        if actor.get(field) is False:
            return True
    collision = actor.get("collision")
    return isinstance(collision, Mapping) and collision.get("collision_enabled") is False


def is_non_solid(actor: Mapping[str, Any]) -> bool:
    """Would bumping into this Actor mean anything.

    Water counts as non-solid here, and that is not a contradiction with
    `environment_consistency`: being IN water is a submersion question, not a
    penetration one, and reporting a boat as penetrating the sea would drown
    the real collisions in noise.
    """
    if _explicitly_non_collidable(actor) or water_evidence(actor):
        return True
    for value in semantic_values(actor):
        compact = re.sub(r"[^a-z0-9]+", "", str(value).lower())
        if "groupactor" in compact:
            return True
        if any(word in NON_SOLID_SEMANTIC_WORDS for word in _words(value)):
            return True
    diagnostics = actor.get("export_diagnostics")
    components = (diagnostics or {}).get("component_classes") or [] \
        if isinstance(diagnostics, Mapping) else []
    return any(word in NON_SOLID_TYPE_WORDS
               for value in [actor.get("class"), *components] if value
               for word in _words(value))


def is_non_solid_ground_gap_target(actor: Mapping[str, Any]) -> bool:
    """Whether local floating should exclude this Actor from its denominator.

    The frozen V3 solid classifier intentionally keeps its historical broad
    water-name rule in :func:`is_non_solid`. Local I2S floating needs a stricter
    applicability decision: marketplace levels contain thick, collidable
    Actors named ``BPP_LI_River_Door`` and ``SM_Pool``. Their names describe an
    area or structure, not a non-solid surface. Only an explicitly
    non-collidable Actor, a non-solid component/type, or a geometrically thin
    water surface is excluded here. This keeps T2S and V3 solid-penetration
    behavior frozen while fixing the new I2S-only population.
    """
    if _explicitly_non_collidable(actor):
        return True
    for value in semantic_values(actor):
        compact = re.sub(r"[^a-z0-9]+", "", str(value).lower())
        if "groupactor" in compact:
            return True
        if any(word in NON_SOLID_SEMANTIC_WORDS for word in _words(value)):
            return True
    diagnostics = actor.get("export_diagnostics")
    components = (diagnostics or {}).get("component_classes") or [] \
        if isinstance(diagnostics, Mapping) else []
    if any(
        word in NON_SOLID_TYPE_WORDS
        for value in [actor.get("class"), *components]
        if value
        for word in _words(value)
    ):
        return True
    bounds = bounding_box(actor)
    return bool(
        bounds is not None
        and bounds["extent"][2] <= DEFAULT_MAXIMUM_THIN_SURFACE_EXTENT_CM
        and water_surface(actor) is not None
    )


def _ground_gap_support_identity_values(actor: Mapping[str, Any]) -> list[str]:
    """Actor identity for the I2S ground-gap applicability classifier.

    Package directories are provenance, not identity.  In particular, a car
    Blueprint stored below ``/Game/Street_NY`` is still a car rather than a
    street.  Labels and semantic categories are already Actor-owned; class,
    asset and component references contribute only their final object name.
    """
    values: list[Any] = [
        actor.get("label"),
        actor.get("asset_category"),
        actor.get("semantic_category"),
        *(actor.get("actor_tags") or []),
        _asset_leaf(actor.get("class")),
        _asset_leaf(actor.get("asset_path")),
        *(
            _asset_leaf(value)
            for value in actor.get("component_asset_paths") or []
        ),
    ]
    return [str(value) for value in values if value is not None]


def _normalized_semantic_words(values: Sequence[Any]) -> set[str]:
    """Normalize marketplace-style identifiers without reading path parents."""
    words: set[str] = set()
    for value in values:
        for word in _words(value):
            words.add(word)
            suffix_free = re.sub(r"\d+$", "", word)
            if suffix_free:
                words.add(suffix_free)
            words.update(re.findall(r"[a-z]+", word))
    return words


def ground_gap_support_surface_evidence(
    actor: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Evidence that an I2S edit is support geometry, not a floating target.

    This classifier is intentionally narrower than
    :func:`is_ground_support_surface`, whose scene-wide V3 callers retain
    their frozen behavior.  I2S local floating excludes an Actor only when:

    * the Actor's own identity (never a containing package directory) names a
      ground/floor/road-like surface; and
    * its exported bounds are broad and thin enough to behave as support.

    A material can supply the semantic name only for a generic Engine basic
    shape.  This prevents a normal prop with a floor-like material from
    disappearing from the denominator.  Thick cliffs, rocks, pools and
    structures remain eligible even when their names contain environment
    words.
    """
    identity_values = _ground_gap_support_identity_values(actor)
    identity_words = _normalized_semantic_words(identity_values)
    matched_words = sorted(identity_words.intersection(GROUND_SUPPORT_WORDS))
    evidence_source = "actor_identity"

    if not matched_words:
        generic_shape_values = [
            actor.get("class"),
            actor.get("asset_path"),
            *(actor.get("component_asset_paths") or []),
        ]
        generic_basic_shape = any(
            value is not None and GENERIC_BASIC_SHAPE_TOKEN.search(str(value))
            for value in generic_shape_values
        )
        material_values = [
            leaf
            for value in actor.get("material_paths") or []
            if (leaf := _asset_leaf(value)) is not None
        ]
        material_words = _normalized_semantic_words(material_values)
        matched_words = sorted(material_words.intersection(GROUND_SUPPORT_WORDS))
        if not generic_basic_shape or not matched_words:
            return None
        evidence_source = "generic_basic_shape_material"

    bounds = bounding_box(actor)
    if bounds is None:
        return None
    horizontal_span = min(bounds["extent"][0], bounds["extent"][1]) * 2.0
    vertical_span = bounds["extent"][2] * 2.0
    if (
        horizontal_span < DEFAULT_MINIMUM_SURFACE_SPAN_CM
        or vertical_span > DEFAULT_MAXIMUM_THIN_SURFACE_EXTENT_CM * 2.0
    ):
        return None
    return {
        "reason": "actor_owned_ground_identity_and_broad_thin_bounds",
        "semantic_evidence_source": evidence_source,
        "matched_support_words": matched_words,
        "horizontal_span_cm": horizontal_span,
        "vertical_span_cm": vertical_span,
    }


def is_ground_support_ground_gap_target(actor: Mapping[str, Any]) -> bool:
    """Whether I2S local floating should exclude this support-surface edit."""
    return ground_gap_support_surface_evidence(actor) is not None


def is_ground_support_surface(actor: Mapping[str, Any]) -> bool:
    """Whether an Actor is scene support rather than a ground-gap target.

    This is a deterministic applicability classifier, not a task-authored
    prompt rule.  It requires both semantic evidence and plausible support
    geometry, except for irregular terrain whose AABB is already known to be
    unsuitable for actor-level physics decisions.
    """
    actor_values = _actor_semantic_values(actor)
    if any(IRREGULAR_ENVIRONMENT_TOKEN.search(value) for value in actor_values):
        return True
    # A surface material can identify an otherwise generic plane, but only its
    # asset name is semantic.  The directory that contains it is not.
    values = [*actor_values, *(
        leaf for value in actor.get("material_paths") or []
        if (leaf := _asset_leaf(value)) is not None
    )]
    words = set(_semantic_words(actor))
    for value in values[len(actor_values):]:
        for word in _words(value):
            words.add(word)
            words.add(re.sub(r"\d+$", "", word))
    if not words.intersection(GROUND_SUPPORT_WORDS):
        return False
    bounds = bounding_box(actor)
    if bounds is None:
        return False
    horizontal_span = min(bounds["extent"][0], bounds["extent"][1]) * 2.0
    vertical_span = bounds["extent"][2] * 2.0
    return (
        horizontal_span >= DEFAULT_MINIMUM_SURFACE_SPAN_CM
        and vertical_span <= DEFAULT_MAXIMUM_THIN_SURFACE_EXTENT_CM * 2.0
    )


def is_irregular_environment(actor: Mapping[str, Any],
                             maximum_span_cm: float | None = None) -> bool:
    """Is this Actor's AABB too coarse to decide anything against."""
    bounds = bounding_box(actor)
    if bounds is None:
        return False
    if any(IRREGULAR_ENVIRONMENT_TOKEN.search(value)
           for value in _actor_semantic_values(actor)):
        return True
    span = float(maximum_span_cm or DEFAULT_MAXIMUM_DECISIVE_SPAN_CM)
    return max(bounds["extent"][axis] * 2 for axis in range(3)) > span


def classify_physics_roles(
    actors: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Assign every scene Actor one deterministic, auditable physics role.

    This is applicability, not semantic quality.  In particular, an
    environment proxy is never silently forgiven: exact contacts with a
    scored Actor become one actor-level ``environment_containment_failure``.
    Requiring size, containment and either environment semantics or a generic
    engine primitive keeps ordinary large buildings in the scored population.
    """
    options = options or {}
    scene = list(actors)
    bounds_by_key = {
        actor_name(actor): bounding_box(actor) for actor in scene
    }
    solid_actors = [actor for actor in scene if not is_non_solid(actor)]
    solid_spans = [
        max(2.0 * extent for extent in bounds["extent"])
        for actor in solid_actors
        if (bounds := bounds_by_key.get(actor_name(actor))) is not None
    ]
    median_span = float(statistics.median(solid_spans)) if solid_spans else 0.0
    decisive_span = float(options.get(
        "maximum_decisive_aabb_span_cm",
        DEFAULT_MAXIMUM_DECISIVE_SPAN_CM,
    ))
    minimum_proxy_span = float(options.get(
        "environment_proxy_minimum_span_cm",
        DEFAULT_ENVIRONMENT_PROXY_MINIMUM_SPAN_CM,
    ))
    median_multiplier = float(options.get(
        "environment_proxy_median_span_multiplier",
        DEFAULT_ENVIRONMENT_PROXY_MEDIAN_SPAN_MULTIPLIER,
    ))
    minimum_contained = int(options.get(
        "environment_proxy_minimum_contained_actor_count",
        DEFAULT_ENVIRONMENT_PROXY_MINIMUM_CONTAINED_ACTOR_COUNT,
    ))
    minimum_fraction = float(options.get(
        "environment_proxy_minimum_contained_actor_fraction",
        DEFAULT_ENVIRONMENT_PROXY_MINIMUM_CONTAINED_ACTOR_FRACTION,
    ))
    other_solid_count = max(0, len(solid_actors) - 1)
    required_contained = max(
        minimum_contained,
        int(math.ceil(minimum_fraction * other_solid_count)),
    )

    roles: dict[str, dict[str, Any]] = {}
    for actor in scene:
        key = actor_name(actor)
        bounds = bounds_by_key[key]
        if is_non_solid(actor):
            roles[key] = {
                "role": PHYSICS_ROLE_NON_SOLID,
                "reasons": ["semantic_or_component_type_is_non_solid"],
            }
            continue
        if bounds is None:
            roles[key] = {
                "role": PHYSICS_ROLE_UNRESOLVED,
                "reasons": ["missing_or_invalid_actor_bounds"],
            }
            continue

        maximum_span = max(2.0 * extent for extent in bounds["extent"])
        vertical_span = 2.0 * bounds["extent"][2]
        values = _actor_semantic_values(actor)
        environment_hint = any(
            ENVIRONMENT_PROXY_TOKEN.search(value) for value in values
        )
        asset_values = [
            actor.get("asset_path"),
            *(actor.get("component_asset_paths") or []),
        ]
        generic_shape = any(
            GENERIC_BASIC_SHAPE_TOKEN.search(str(value).replace("\\", "/"))
            for value in asset_values if value
        )
        native_landscape = any(
            "landscape" in value.lower() for value in values
        )
        ground_support = is_ground_support_surface(actor)
        thin_support = (
            ground_support
            and vertical_span
            <= DEFAULT_MAXIMUM_THIN_SURFACE_EXTENT_CM * 2.0
        )
        if native_landscape or thin_support:
            roles[key] = {
                "role": PHYSICS_ROLE_SUPPORT_SURFACE,
                "reasons": [
                    "native_landscape_surface"
                    if native_landscape else "broad_thin_ground_support_surface"
                ],
                "maximum_span_cm": maximum_span,
            }
            continue

        relatively_oversized = bool(
            median_span > 0.0
            and maximum_span >= median_span * median_multiplier
        )
        absolutely_oversized = maximum_span >= minimum_proxy_span
        oversized = (
            maximum_span > decisive_span
            and (relatively_oversized or absolutely_oversized)
        )
        proxy_candidate = oversized and (environment_hint or generic_shape)
        # Containment is the only scene-wide operation.  Evaluate it only for
        # plausible proxy candidates so ordinary scenes remain O(number of
        # Actors) instead of paying an unnecessary all-pairs cost.
        contained = 0
        if proxy_candidate:
            for candidate in solid_actors:
                if candidate is actor:
                    continue
                candidate_bounds = bounds_by_key.get(actor_name(candidate))
                if candidate_bounds is None:
                    continue
                origin = candidate_bounds["origin"]
                if all(
                    bounds["min"][axis] <= origin[axis] <= bounds["max"][axis]
                    for axis in range(3)
                ):
                    contained += 1
        contained_fraction = (
            contained / other_solid_count if other_solid_count else 0.0
        )
        proxy = bool(
            proxy_candidate
            and contained >= required_contained
        )
        if proxy:
            reasons = []
            if relatively_oversized:
                reasons.append("oversized_relative_to_scene")
            if absolutely_oversized:
                reasons.append("oversized_absolute_span")
            if generic_shape:
                reasons.append("generic_engine_basic_shape")
            if environment_hint:
                reasons.append("environment_semantic_hint")
            reasons.append("contains_many_independent_scene_actors")
            roles[key] = {
                "role": PHYSICS_ROLE_ENVIRONMENT_PROXY,
                "reasons": reasons,
                "maximum_span_cm": maximum_span,
                "scene_median_solid_span_cm": median_span,
                "contained_solid_actor_count": contained,
                "contained_solid_actor_fraction": contained_fraction,
                "required_contained_actor_count": required_contained,
            }
            continue
        if ground_support:
            roles[key] = {
                "role": PHYSICS_ROLE_SUPPORT_SURFACE,
                "reasons": ["semantic_and_geometric_ground_support_surface"],
                "maximum_span_cm": maximum_span,
            }
            continue
        roles[key] = {
            "role": PHYSICS_ROLE_SCORED_SOLID,
            "reasons": ["solid_scene_object"],
            "maximum_span_cm": maximum_span,
        }
    return roles


def actor_name(actor: Mapping[str, Any]) -> str:
    for field in ("stable_actor_id", "actor_path", "label"):
        value = actor.get(field)
        if value:
            return str(value)
    return "unknown_actor"


def overlap_depths(left: Mapping[str, list[float]],
                   right: Mapping[str, list[float]]) -> list[float]:
    return [min(left["max"][axis], right["max"][axis])
            - max(left["min"][axis], right["min"][axis]) for axis in range(3)]


def measurement_for(actor: Mapping[str, Any], measurements: Any
                    ) -> Mapping[str, Any] | None:
    """This Actor's physics record, by label, stable id or path."""
    if not isinstance(measurements, Mapping):
        return None
    records = measurements.get("actors")
    if not isinstance(records, Mapping):
        return None
    for field in ("label", "stable_actor_id", "actor_path"):
        key = actor.get(field)
        if key and isinstance(records.get(str(key)), Mapping):
            return records[str(key)]
    return None


def names(items: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(item.get("actor")) for item in items}


__all__ = ["DEFAULT_ENVIRONMENT_PROXY_MEDIAN_SPAN_MULTIPLIER",
           "DEFAULT_ENVIRONMENT_PROXY_MINIMUM_CONTAINED_ACTOR_COUNT",
           "DEFAULT_ENVIRONMENT_PROXY_MINIMUM_CONTAINED_ACTOR_FRACTION",
           "DEFAULT_ENVIRONMENT_PROXY_MINIMUM_SPAN_CM",
           "DEFAULT_MAXIMUM_DECISIVE_SPAN_CM",
           "DEFAULT_MAXIMUM_THIN_SURFACE_EXTENT_CM",
           "DEFAULT_MINIMUM_SURFACE_SPAN_CM", "DEFAULT_WATERLINE_TOLERANCE_CM",
           "ENVIRONMENT_PROXY_TOKEN", "GENERIC_BASIC_SHAPE_TOKEN",
           "GROUND_SUPPORT_WORDS", "IRREGULAR_ENVIRONMENT_TOKEN",
           "NON_SOLID_SEMANTIC_WORDS",
           "NON_SOLID_TYPE_WORDS", "PHYSICS_ROLE_ENVIRONMENT_PROXY",
           "PHYSICS_ROLE_NON_SOLID", "PHYSICS_ROLE_SCORED_SOLID",
           "PHYSICS_ROLE_SUPPORT_SURFACE", "PHYSICS_ROLE_UNRESOLVED",
           "WATER_TOKEN", "actor_name", "bounding_box",
           "classify_physics_roles", "is_ground_support_surface",
           "ground_gap_support_surface_evidence",
           "is_ground_support_ground_gap_target",
           "is_irregular_environment",
           "is_non_solid", "is_non_solid_ground_gap_target", "measurement_for",
           "names", "overlap_depths", "overlaps_xy", "semantic_values",
           "water_evidence", "water_surface"]
