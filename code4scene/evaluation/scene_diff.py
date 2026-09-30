"""What changed between the scene the agent was handed and the one it left.

This is the comparison behind every question of the form "did you touch what
you were not asked to touch". It reads two scene graphs in the exporter's
shape and sorts every Actor into added, removed, moved, modified or matched.

Three decisions in here are load-bearing and none of them are obvious:

* **Identity is a lookup, not a guess.** Two Actors that resolve to the same
  identity make the whole diff meaningless — one of them would silently stand
  in for the other — so an ambiguous scene raises rather than reporting a
  confident nothing-changed.
* **`input_scene` is not an answer key.** The pre-edit scene is what the agent
  was given; comparing against it asks a question the agent could ask itself.
  That is why the verifiers built on this diff declare `open_ended` even
  though they compare against another scene. The answer key is the CANONICAL
  scene, and it is reached through the answer-key helper in `context`,
  which nothing in this module imports.
* **A property the source snapshot never recorded is not an empty one.**
  Candidate exporters may measure more than the exporter that wrote the input
  did; reading a missing `material_paths` as "no materials" would report every
  such Actor as re-materialised. The field is compared only when the source
  side actually carries it.
* **A level-local material instance belongs to its object, not its package.**
  Saving an Input level under the Candidate package renames a dynamic MID from
  ``/Game/Input.X:PersistentLevel.Actor.Component.MID`` to
  ``/Game/Saved.X:PersistentLevel.Actor.Component.MID`` without changing the
  material.  Only that package prefix is transient: both paths must be
  level-local and their complete suffix after ``:PersistentLevel.`` must agree.
* **An Engine transient dynamic MID has no stable object identity.** Unreal
  allocates names such as ``/Engine/Transient.WaterMID_277`` independently on
  every map load.  When both slots explicitly say ``is_dynamic=true``, that
  numeric object name is capture bookkeeping rather than an authored material
  edit. Component, slot, dynamic/static state and material class still compare.

Ported from the vendored `diff.js` / `matcher.js`, whose tolerances are kept
verbatim: they are what a round-trip through the editor costs, not a quality
threshold, and loosening them here would quietly widen what counts as
untouched.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: What a round-trip through the editor costs. Anything larger is an edit.
DEFAULT_TOLERANCES = {"location_cm": 0.1, "rotation_deg": 0.01, "scale": 0.0001}

#: Actor properties a task can be about. The exporter also writes bookkeeping
#: — capture timestamps, transient ids — and a diff that treated those as
#: scene edits would report every candidate as having modified every Actor.
TASK_RELEVANT_PROPERTY_KEYS = frozenset({
    "intensity", "intensity_units", "light_color", "temperature",
    "use_temperature", "source_angle", "indirect_lighting_intensity",
    "volumetric_scattering_intensity", "fog_density", "fog_height_falloff",
    "start_distance", "sun_disk_scale", "cloud_opacity",
    "exposure_compensation", "min_brightness", "max_brightness",
    "white_temp", "color_saturation", "color_contrast", "color_gamma",
    "color_gain", "color_offset",
})

#: Fields whose change is a property edit rather than a move.
_PROPERTY_FIELDS = ("label", "class", "asset_path", "actor_origin",
                    "logical_object_id", "actor_role")


class ActorIdentityError(Exception):
    """Two Actors in one scene resolve to the same identity."""


def actor_identity(actor: Mapping[str, Any]) -> str:
    """The strongest identity this Actor carries, namespaced by which it is.

    Namespaced because the strengths are not interchangeable: a label that
    happens to equal another Actor's stable id must not match it.
    """
    stable = actor.get("stable_actor_id")
    if isinstance(stable, str) and stable.strip():
        return f"stable:{stable.strip()}"
    guid = actor.get("actor_guid")
    if guid is not None and str(guid).strip():
        return f"guid:{str(guid).strip()}"
    label = actor.get("label")
    if isinstance(label, str) and label:
        return f"label:{label}"
    path = str(actor.get("actor_path") or "")
    marker = ":PersistentLevel."
    index = path.rfind(marker)
    suffix = path[index + len(marker):] if index >= 0 else path
    if suffix:
        return f"path:{suffix}"
    raise ActorIdentityError(
        "an Actor carries no stable id, guid, label or path, so it cannot be "
        "told apart from any other Actor in the scene")


def index_actors(scene: Any) -> dict[str, Mapping[str, Any]]:
    """Identity -> Actor, refusing a scene where two Actors collide."""
    actors = (scene or {}).get("actors") if isinstance(scene, Mapping) else None
    index: dict[str, Mapping[str, Any]] = {}
    for actor in actors or []:
        if not isinstance(actor, Mapping):
            continue
        key = actor_identity(actor)
        if key in index:
            raise ActorIdentityError(
                f"two Actors both resolve to {key}; a diff over an ambiguous "
                f"scene silently substitutes one for the other")
        index[key] = actor
    return index


def _numeric_arrays_equal(left: Any, right: Any, tolerance: float) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if len(left) != len(right):
        return False
    try:
        return all(abs(float(a) - float(b)) <= tolerance
                   for a, b in zip(left, right, strict=True))
    except (TypeError, ValueError):
        return False


def _transform_differences(left: Any, right: Any,
                           tolerances: Mapping[str, float]) -> list[str]:
    left = left if isinstance(left, Mapping) else {}
    right = right if isinstance(right, Mapping) else {}
    fields = []
    for name, key, tolerance in (
        ("location", "location_cm", tolerances["location_cm"]),
        ("rotation", "rotation_deg", tolerances["rotation_deg"]),
        ("scale", "scale", tolerances["scale"]),
    ):
        # A partial/legacy Input snapshot that never recorded this component
        # cannot prove that the value was empty.  Candidate exporters may gain
        # fields over time; comparing the richer side against an implicit
        # empty value would turn an evidence-coverage change into a false edit.
        if key not in left:
            continue
        if not _numeric_arrays_equal(left.get(key), right.get(key), tolerance):
            fields.append(name)
    return fields


def _stable(value: Any) -> str:
    """A canonical rendering, so two equal structures compare equal."""
    return json.dumps(value, sort_keys=True, default=str)


def task_relevant_properties(properties: Any) -> dict[str, Any]:
    if not isinstance(properties, Mapping):
        return {}
    return {str(key).lower(): value for key, value in properties.items()
            if str(key).lower() in TASK_RELEVANT_PROPERTY_KEYS}


def _material_path_identity(value: Any) -> tuple[str, str]:
    """Stable identity for an asset path or a level-local material object.

    Asset paths remain byte-for-byte identities.  For an object owned by the
    level, only the package before ``:PersistentLevel.`` changes when the same
    level is saved as the Candidate; retaining the full object suffix keeps a
    different Actor, component, MID, or subobject detectable.
    """
    path = str(value)
    marker = ":PersistentLevel."
    index = path.rfind(marker)
    if index < 0:
        return "asset", path
    return "level_local", path[index + len(marker):]


def _is_engine_transient_dynamic_slot(value: Any) -> bool:
    """Whether a material slot names a load-local Engine MID instance."""

    return (
        isinstance(value, Mapping)
        and value.get("is_dynamic") is True
        and str(value.get("material_path") or "").casefold().startswith(
            "/engine/transient."
        )
    )


def _is_level_local_dynamic_slot(value: Any) -> bool:
    """Whether a material slot names a dynamic instance owned by the level."""

    return (
        isinstance(value, Mapping)
        and value.get("is_dynamic") is True
        and ":PersistentLevel." in str(value.get("material_path") or "")
    )


def _level_local_dynamic_identity(path: Any) -> tuple[str, str]:
    """A level-local dynamic instance without the counter Unreal renumbers on save."""

    _, local = _material_path_identity(path)
    owner, _, name = local.rpartition(".")
    return "level_local_dynamic", f"{owner}.{re.sub(r'_[0-9]+$', '', name)}"


def _level_local_dynamic_material_paths(actor: Mapping[str, Any]) -> set[str]:
    values = actor.get("component_material_slots") or []
    if not isinstance(values, list):
        return set()
    return {
        str(value["material_path"]).casefold()
        for value in values
        if _is_level_local_dynamic_slot(value)
    }


def _dynamic_transient_material_paths(actor: Mapping[str, Any]) -> set[str]:
    values = actor.get("component_material_slots") or []
    if not isinstance(values, list):
        return set()
    return {
        str(value["material_path"]).casefold()
        for value in values
        if _is_engine_transient_dynamic_slot(value)
    }


def _actor_material_path_identities(
    actor: Mapping[str, Any],
) -> list[tuple[str, str]]:
    transient = _dynamic_transient_material_paths(actor)
    level_dynamic = _level_local_dynamic_material_paths(actor)
    return sorted(
        (
            ("engine_transient_dynamic", "material_instance")
            if str(value).casefold() in transient
            else _level_local_dynamic_identity(value)
            if str(value).casefold() in level_dynamic
            else _material_path_identity(value)
        )
        for value in actor.get("material_paths") or []
    )


def _material_paths_equal(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return _actor_material_path_identities(left) == _actor_material_path_identities(
        right
    )


def _stable_material_slots(actor: Mapping[str, Any]) -> list[Any]:
    """Normalize only the level package portion of material-slot paths.

    This retains Actor, component, slot and MID identity while avoiding a false
    property edit when the same level-local object is saved under the Candidate
    map package.
    """

    values = actor.get("component_material_slots") or []
    if not isinstance(values, list):
        return values
    normalized: list[Any] = []
    for value in values:
        if not isinstance(value, Mapping) or not value.get("material_path"):
            normalized.append(value)
            continue
        item = dict(value)
        item["material_path"] = (
            ("engine_transient_dynamic", "material_instance")
            if _is_engine_transient_dynamic_slot(item)
            else _level_local_dynamic_identity(item["material_path"])
            if _is_level_local_dynamic_slot(item)
            else _material_path_identity(item["material_path"])
        )
        normalized.append(item)
    return normalized


def _property_differences(left: Mapping[str, Any],
                          right: Mapping[str, Any]) -> list[str]:
    fields = [name for name in _PROPERTY_FIELDS
              if name in left
              if (left.get(name) or None) != (right.get(name) or None)]
    if "actor_tags" in left and \
            _stable(sorted(map(str, left.get("actor_tags") or []))) != \
            _stable(sorted(map(str, right.get("actor_tags") or []))):
        fields.append("actor_tags")
    # Slot order and component ownership are semantically meaningful. Compare
    # them only when the source exporter recorded the richer evidence; an old
    # snapshot with only material_paths cannot support a slot-level claim.
    if "component_material_slots" in left and \
            _stable(_stable_material_slots(left)) != \
            _stable(_stable_material_slots(right)):
        fields.append("component_material_slots")
    if "component_asset_paths" in left and \
            _stable(left.get("component_asset_paths") or []) != \
            _stable(right.get("component_asset_paths") or []):
        fields.append("component_asset_paths")
    # Only when the SOURCE side recorded materials at all — see the module
    # docstring. An exporter that did not look is not evidence of an empty set.
    if "material_paths" in left and not _material_paths_equal(left, right):
        fields.append("material_paths")
    if "properties" in left and \
            _stable(task_relevant_properties(left.get("properties"))) != \
            _stable(task_relevant_properties(right.get("properties"))):
        fields.append("properties")
    return fields


@dataclass(frozen=True)
class Change:
    """One Actor that exists on both sides but is not the same any more."""

    key: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    fields: tuple[str, ...]


@dataclass(frozen=True)
class SceneDiff:
    """Two scenes, sorted. ``matched`` is every Actor present on both sides."""

    added: list[Mapping[str, Any]] = field(default_factory=list)
    removed: list[Mapping[str, Any]] = field(default_factory=list)
    moved: list[Change] = field(default_factory=list)
    modified: list[Change] = field(default_factory=list)
    matched: list[Change] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {"added": len(self.added), "removed": len(self.removed),
                "moved": len(self.moved), "modified": len(self.modified),
                "matched": len(self.matched)}


def edited_candidate_actors(diff: SceneDiff) -> list[Mapping[str, Any]]:
    """Return the Candidate-side Actors physically affected by an edit.

    Deletions have no Candidate Actor to probe. Additions do, and a matched
    Actor can be both moved and modified, so the latter two populations are
    de-duplicated by the same strict identity used to construct ``diff``.
    This is the local target population used by image-to-scene repair Physics;
    every selected Actor is still tested against collision geometry from the
    complete Candidate world.
    """

    values: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for actor in diff.added:
        key = actor_identity(actor)
        if key not in seen:
            seen.add(key)
            values.append(actor)
    for change in (*diff.moved, *diff.modified):
        if change.key not in seen:
            seen.add(change.key)
            values.append(change.after)
    return values


def diff_scenes(before: Any, after: Any,
                tolerances: Mapping[str, float] | None = None) -> SceneDiff:
    """Sort every Actor of two scene graphs into what happened to it."""
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    left = index_actors(before)
    right = index_actors(after)
    diff = SceneDiff()
    for key, actor in right.items():
        if key not in left:
            diff.added.append(actor)
    for key, actor in left.items():
        current = right.get(key)
        if current is None:
            diff.removed.append(actor)
            continue
        diff.matched.append(Change(key, actor, current, ()))
        transform_fields = _transform_differences(
            actor.get("transform"), current.get("transform"), tol)
        if transform_fields:
            diff.moved.append(Change(key, actor, current, tuple(transform_fields)))
        property_fields = _property_differences(actor, current)
        if property_fields:
            diff.modified.append(Change(key, actor, current, tuple(property_fields)))
    return diff


def actor_summary(actor: Mapping[str, Any]) -> dict[str, Any]:
    """The identifying fields a report shows for one Actor, and no geometry."""
    return {"stable_actor_id": actor.get("stable_actor_id") or None,
            "label": actor.get("label"),
            "class": actor.get("class"),
            "asset_path": actor.get("asset_path") or None,
            "actor_origin": actor.get("actor_origin") or None,
            "logical_object_id": actor.get("logical_object_id") or None,
            "actor_role": actor.get("actor_role") or None}


def scene_actors(scene: Any) -> list[Mapping[str, Any]]:
    """The Actor list of a scene graph, or an empty one."""
    actors: Sequence[Any] = (scene or {}).get("actors") or [] \
        if isinstance(scene, Mapping) else []
    return [actor for actor in actors if isinstance(actor, Mapping)]


__all__ = ["ActorIdentityError", "Change", "DEFAULT_TOLERANCES", "SceneDiff",
           "TASK_RELEVANT_PROPERTY_KEYS", "actor_identity", "actor_summary",
           "diff_scenes", "edited_candidate_actors", "index_actors", "scene_actors",
           "task_relevant_properties"]
