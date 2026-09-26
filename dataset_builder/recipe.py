"""Edit recipes: the ops that turn a canonical GT level into a task Input level.

A recipe is plain JSON (``recipe.json`` next to each image-to-scene task). The
same op list is executed in two places:

* inside the Unreal Editor by ``ue/c4s_materialize.py`` on the user's locally
  built GT level, and
* here, offline, on an exported scene snapshot (``apply_offline``). The offline
  path predicts the Input snapshot so that the prediction's fingerprint can be
  checked against the expected Input fingerprint without an editor.

Targets are addressed by benchmark stable ID as they appear in the GT level.
Ops are applied in order.

Op vocabulary (``operations[*].op``)
------------------------------------
``remove_actor``      ``target``
``set_transform``     ``target`` plus any of ``location_cm``, ``rotation_deg``
                      (pitch, yaw, roll) and ``scale``; absolute values.
``set_static_mesh``   ``target``, ``component`` (component name), ``static_mesh``;
                      ``expect.materials`` lists the slot materials the editor
                      must report afterwards (the op never overrides materials).
``spawn_actor``       ``stable_id`` of the new actor, ``from`` (``static_mesh`` or
                      ``class``), ``object_name`` (the actor's object name, which
                      the exporter uses for generated IDs), ``label``, absolute
                      transform, optional ``identity_tag`` (write an explicit
                      stable-ID tag) and ``expect`` (component assets, material
                      slots and tags the editor must report for the new actor).
``set_label``         ``target``, ``label``.
``set_component_property`` ``target``, ``component_class``, ``property``, ``value``.

Level-structure ops used by one scene (see its ``scene.json``) are executed by
the editor scripts only; offline they are modelled by ``apply_offline`` through
``sublevel_rename``.

Only the Python standard library is used.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Iterable, Mapping

RECIPE_SCHEMA = "code4scene.recipe.v1"
GENERATED_STABLE_ID_PREFIX = "sca_generated_"
STABLE_ID_TAG_PREFIX = "simcodearena.stable_actor_id="
KNOWN_OPS = (
    "remove_actor",
    "set_transform",
    "set_static_mesh",
    "spawn_actor",
    "set_label",
    "set_component_property",
)


class RecipeError(ValueError):
    pass


# --------------------------------------------------------------------------
# Identity rules. These mirror ue/c4s_export_snapshot.py exactly.
# --------------------------------------------------------------------------

def explicit_stable_id(tags: Iterable[Any]) -> str | None:
    for tag in tags or []:
        text = str(tag)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key.strip().lower() in ("simcodearena.stable_actor_id", "stable_actor_id") and value.strip():
            return value.strip()
    return None


def generated_stable_id(actor_name: str, class_path: str, component_asset_paths: Iterable[str]) -> str:
    signature = {
        "actor_name": str(actor_name),
        "class_path": str(class_path),
        "component_asset_paths": sorted(set(str(p) for p in component_asset_paths)),
    }
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return GENERATED_STABLE_ID_PREFIX + hashlib.sha256(canonical).hexdigest()[:32]


def normalized_actor_scope(actor_path: str, map_package: str) -> str:
    path = str(actor_path or "")
    marker = ":PersistentLevel."
    if marker not in path:
        return path
    owner = path.split(marker, 1)[0]
    if owner.split(".", 1)[0] == str(map_package or ""):
        return "$ROOT_LEVEL"
    return owner


def assign_stable_ids(actors: list[dict[str, Any]], map_package: str) -> None:
    """Assign ``stable_actor_id`` the way the snapshot exporter does."""

    for actor in actors:
        explicit = explicit_stable_id(actor.get("actor_tags"))
        diag = actor.setdefault("export_diagnostics", {})
        if explicit:
            actor["stable_actor_id"] = explicit
            diag["stable_actor_id_provenance"] = "explicit_tag"
        else:
            actor["stable_actor_id"] = generated_stable_id(
                actor.get("name"), actor.get("class"), actor.get("component_asset_paths") or []
            )
            diag["stable_actor_id_provenance"] = "generated_structural_identity"
    groups: dict[str, list[dict[str, Any]]] = {}
    for actor in actors:
        groups.setdefault(actor["stable_actor_id"], []).append(actor)
    for stable_id, members in groups.items():
        if len(members) < 2:
            continue
        if any(m["export_diagnostics"]["stable_actor_id_provenance"] == "explicit_tag" for m in members):
            continue
        seen = set()
        for actor in members:
            scope = normalized_actor_scope(actor.get("actor_path"), map_package)
            canonical = json.dumps(
                {"base_stable_actor_id": stable_id, "actor_scope": scope},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            replacement = GENERATED_STABLE_ID_PREFIX + hashlib.sha256(canonical).hexdigest()[:32]
            if replacement in seen:
                raise RecipeError(f"generated stable ID {stable_id} stays ambiguous in scope {scope}")
            seen.add(replacement)
            actor["stable_actor_id"] = replacement
            actor["export_diagnostics"]["stable_actor_id_provenance"] = (
                "generated_structural_identity_disambiguated_by_actor_scope"
            )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(recipe: Mapping[str, Any]) -> None:
    if recipe.get("schema_version") != RECIPE_SCHEMA:
        raise RecipeError(f"recipe schema_version must be {RECIPE_SCHEMA}")
    for key in ("case_id", "scene_id", "ground_truth_map", "input_map", "operations"):
        if key not in recipe:
            raise RecipeError(f"recipe is missing {key}")
    for index, op in enumerate(recipe["operations"]):
        kind = op.get("op")
        if kind not in KNOWN_OPS:
            raise RecipeError(f"operations[{index}]: unknown op {kind!r}")
        if kind == "spawn_actor":
            if not op.get("stable_id") or not op.get("object_name"):
                raise RecipeError(f"operations[{index}]: spawn_actor needs stable_id and object_name")
            source = op.get("from") or {}
            if not (source.get("static_mesh") or source.get("class")):
                raise RecipeError(f"operations[{index}]: spawn_actor needs from.static_mesh or from.class")
        elif not op.get("target"):
            raise RecipeError(f"operations[{index}]: {kind} needs a target")


# --------------------------------------------------------------------------
# Offline application
# --------------------------------------------------------------------------

def _replace_root(path: str, old_root: str, new_root: str) -> str:
    old_prefix = f"{old_root}.{old_root.rsplit('/', 1)[-1]}"
    new_prefix = f"{new_root}.{new_root.rsplit('/', 1)[-1]}"
    return new_prefix + path[len(old_prefix):] if path.startswith(old_prefix) else path


def apply_offline(
    gt_snapshot: Mapping[str, Any],
    recipe: Mapping[str, Any],
    *,
    input_map: str | None = None,
    sublevel_rename: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Predict the Input snapshot that ``recipe`` produces from ``gt_snapshot``.

    ``sublevel_rename`` maps a GT streaming-sublevel package to the package the
    Input streams instead (see the scene's ``input_level_structure``).
    """

    validate(recipe)
    gt_map = str(gt_snapshot.get("map_path"))
    new_map = input_map or str(recipe["input_map"])
    actors = [copy.deepcopy(dict(a)) for a in gt_snapshot.get("actors") or []]
    renames = dict(sublevel_rename or {})
    gt_state = {a["stable_actor_id"]: copy.deepcopy(dict(a)) for a in gt_snapshot.get("actors") or []}
    for actor in actors:
        path = _replace_root(str(actor.get("actor_path") or ""), gt_map, new_map)
        for old, new in renames.items():
            old_prefix = f"{old}.{old.rsplit('/', 1)[-1]}:"
            if path.startswith(old_prefix):
                path = f"{new}.{new.rsplit('/', 1)[-1]}:" + path[len(old_prefix):]
        actor["actor_path"] = path
        parent = actor.get("attachment_parent_path")
        if parent:
            actor["attachment_parent_path"] = _replace_root(str(parent), gt_map, new_map)
    by_id = {a["stable_actor_id"]: a for a in actors}

    def target(op: Mapping[str, Any]) -> dict[str, Any]:
        actor = by_id.get(op["target"])
        if actor is None:
            raise RecipeError(f"{op['op']}: target {op['target']} not in level")
        return actor

    for op in recipe["operations"]:
        kind = op["op"]
        if kind == "remove_actor":
            actor = target(op)
            actors.remove(actor)
            by_id.pop(op["target"], None)
        elif kind == "set_transform":
            actor = target(op)
            for key in ("location_cm", "rotation_deg", "scale"):
                if key in op:
                    actor["transform"][key] = [float(v) for v in op[key]]
        elif kind == "set_static_mesh":
            actor = target(op)
            mesh = op["static_mesh"]
            paths = [p for p in actor.get("component_asset_paths") or [] if p != op.get("replaces")]
            if op.get("replaces") is None:
                paths = []
            actor["component_asset_paths"] = sorted(set(paths + [mesh]))
            actor["asset_path"] = mesh if len(actor["component_asset_paths"]) == 1 else actor.get("asset_path")
            expect = op.get("expect") or {}
            if "material_slots" in expect:
                keep = [s for s in actor.get("component_material_slots") or []
                        if not str(s.get("component_identity", "")).startswith(op["component"] + "|")]
                actor["component_material_slots"] = keep + [dict(s) for s in expect["material_slots"]]
                actor["material_paths"] = sorted({s["material_path"] for s in actor["component_material_slots"] if s.get("material_path")})
        elif kind == "spawn_actor":
            expect = dict(op.get("expect") or {})
            like = expect.pop("like", None)
            if like:
                template = gt_state.get(like)
                if template is None:
                    raise RecipeError(f"spawn_actor: expect.like {like} is not a GT actor")
                expect.setdefault("component_asset_paths", list(template.get("component_asset_paths") or []))
                expect.setdefault("material_slots", [dict(s) for s in template.get("component_material_slots") or []])
                expect.setdefault("tags", [t for t in template.get("actor_tags") or []
                                           if not str(t).startswith("simcodearena.")])
                expect.setdefault("properties", dict(template.get("properties") or {}))
            source = op.get("from") or {}
            tags = list(expect.get("tags") or [])
            if op.get("identity_tag"):
                tags.append(STABLE_ID_TAG_PREFIX + op["stable_id"])
            class_path = source.get("class") or "/Script/Engine.StaticMeshActor"
            name = op["object_name"]
            new_actor = {
                "stable_actor_id": op["stable_id"],
                "name": name,
                "label": op["label"],
                "class": class_path,
                "actor_path": f"{new_map}.{new_map.rsplit('/', 1)[-1]}:PersistentLevel.{name}",
                "component_asset_paths": sorted(expect.get("component_asset_paths") or ([source["static_mesh"]] if source.get("static_mesh") else [])),
                "component_material_slots": [dict(s) for s in expect.get("material_slots") or []],
                "actor_tags": tags,
                "actor_origin": None,
                "actor_role": None,
                "logical_object_id": None,
                "attachment_parent_path": None,
                "properties": dict(expect.get("properties") or {}),
                "transform": {
                    "location_cm": [float(v) for v in op["location_cm"]],
                    "rotation_deg": [float(v) for v in op["rotation_deg"]],
                    "scale": [float(v) for v in op["scale"]],
                },
                "export_diagnostics": {},
            }
            paths = new_actor["component_asset_paths"]
            new_actor["asset_path"] = paths[0] if len(paths) == 1 else (class_path if class_path.startswith("/Game/") else None)
            new_actor["material_paths"] = sorted({s["material_path"] for s in new_actor["component_material_slots"] if s.get("material_path")})
            actors.append(new_actor)
            by_id[op["stable_id"]] = new_actor
        elif kind == "set_label":
            target(op)["label"] = op["label"]
        elif kind == "set_component_property":
            actor = target(op)
            props = actor.setdefault("properties", {})
            props[op["property"]] = op["value"]
    assign_stable_ids(actors, new_map)
    return {
        "schema_version": "0.3.0",
        "snapshot_type": "ue_editor_scene",
        "units": "cm",
        "map_path": new_map,
        "actor_count": len(actors),
        "actors": actors,
        "export_metadata": {"status": "success", "predicted_by": "dataset_builder.recipe.apply_offline"},
    }


__all__ = [
    "KNOWN_OPS",
    "RECIPE_SCHEMA",
    "RecipeError",
    "apply_offline",
    "assign_stable_ids",
    "explicit_stable_id",
    "generated_stable_id",
    "normalized_actor_scope",
    "validate",
]
