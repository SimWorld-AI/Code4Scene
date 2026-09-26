"""Scene snapshot exporter (runs inside the Unreal Editor's Python).

Produces the same per-actor evidence as the benchmark scorer's exporter
(scene snapshot schema 0.3.0) for every field the content fingerprint uses:
stable ID (explicit tag, else the structural fallback with owning-level
disambiguation), label, class, component mesh assets, component material
slots, tags, benchmark identity tags, attachment parent, transform and the
task-relevant light/fog/post-process properties.

Stock Unreal Engine 5.8 Python API only.
"""

import hashlib
import json
import os

import unreal

SNAPSHOT_SCHEMA_VERSION = "0.3.0"
GENERATED_STABLE_ID_PREFIX = "sca_generated_"

TASK_RELEVANT_PROPERTY_KEYS = (
    "intensity", "intensity_units", "light_color", "temperature", "use_temperature",
    "source_angle", "indirect_lighting_intensity", "volumetric_scattering_intensity",
    "fog_density", "fog_height_falloff", "start_distance", "sun_disk_scale",
    "cloud_opacity", "exposure_compensation", "min_brightness", "max_brightness",
    "white_temp", "color_saturation", "color_contrast", "color_gamma", "color_gain",
    "color_offset",
)


def object_path(value):
    if value is None:
        return None
    try:
        return str(value.get_path_name())
    except Exception:
        return str(value)


def _vector(value):
    return [float(value.x), float(value.y), float(value.z)]


def _rotator(value):
    return [float(value.pitch), float(value.yaw), float(value.roll)]


def current_map_package():
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    if world is None:
        return None
    return str(world.get_path_name()).split(".", 1)[0]


def tag_values(actor):
    try:
        tags = [str(value) for value in actor.tags]
    except Exception:
        tags = []
    parsed = {}
    for tag in tags:
        if "=" in tag:
            key, value = tag.split("=", 1)
            parsed[key.strip().lower()] = value.strip()
    return tags, parsed


def _tag(parsed, *keys):
    for key in keys:
        value = parsed.get(key.lower())
        if value:
            return value
    return None


def generated_stable_actor_id(actor_name, class_path, component_asset_paths):
    signature = {
        "actor_name": str(actor_name),
        "class_path": str(class_path),
        "component_asset_paths": sorted(set(str(p) for p in component_asset_paths)),
    }
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return GENERATED_STABLE_ID_PREFIX + hashlib.sha256(canonical).hexdigest()[:32]


def normalized_actor_scope(actor_path, map_package):
    path = str(actor_path or "")
    marker = ":PersistentLevel."
    if marker not in path:
        return path
    owner = path.split(marker, 1)[0]
    if owner.split(".", 1)[0] == str(map_package or ""):
        return "$ROOT_LEVEL"
    return owner


def disambiguate(actors, map_package):
    groups = {}
    for actor in actors:
        groups.setdefault(actor["stable_actor_id"], []).append(actor)
    for stable_id, members in groups.items():
        if not stable_id or len(members) < 2:
            continue
        if any(m["export_diagnostics"]["stable_actor_id_provenance"] == "explicit_tag" for m in members):
            continue
        seen = set()
        for actor in members:
            scope = normalized_actor_scope(actor.get("actor_path"), map_package)
            canonical = json.dumps({"base_stable_actor_id": stable_id, "actor_scope": scope},
                                   sort_keys=True, separators=(",", ":")).encode("utf-8")
            replacement = GENERATED_STABLE_ID_PREFIX + hashlib.sha256(canonical).hexdigest()[:32]
            if replacement in seen:
                raise RuntimeError("generated stable_actor_id {} stays ambiguous in {}".format(stable_id, scope))
            seen.add(replacement)
            actor["stable_actor_id"] = replacement
            actor["export_diagnostics"]["stable_actor_id_provenance"] = (
                "generated_structural_identity_disambiguated_by_actor_scope")


def _serialized_property(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if all(hasattr(value, key) for key in ("r", "g", "b", "a")):
        return [value.r, value.g, value.b, value.a]
    if all(hasattr(value, key) for key in ("x", "y", "z", "w")):
        return [float(value.x), float(value.y), float(value.z), float(value.w)]
    if all(hasattr(value, key) for key in ("x", "y", "z")):
        return [float(value.x), float(value.y), float(value.z)]
    if hasattr(value, "name") and hasattr(value, "value"):
        return str(value)
    if "enum" in type(value).__name__.lower():
        return str(value)
    return None


def _task_relevant_properties(actor, components):
    result = {}
    owners = [actor] + list(components)
    for name in TASK_RELEVANT_PROPERTY_KEYS:
        for owner in owners:
            try:
                value = owner.get_editor_property(name)
            except Exception:
                continue
            serialized = _serialized_property(value)
            if serialized is not None:
                result[name] = serialized
                break
    return result


def component_data(actor):
    asset_paths, material_paths, slots = [], [], []
    try:
        components = list(actor.get_components_by_class(unreal.ActorComponent))
    except Exception:
        components = []
    for component in components:
        try:
            component_class = str(component.get_class().get_path_name())
        except Exception:
            component_class = type(component).__name__
        try:
            component_name = str(component.get_name())
        except Exception:
            component_name = component_class
        identity = "{}|{}".format(component_name, component_class)
        for prop in ("static_mesh", "skeletal_mesh_asset", "skeletal_mesh"):
            try:
                asset = component.get_editor_property(prop)
            except Exception:
                continue
            path = object_path(asset)
            if path and path not in asset_paths:
                asset_paths.append(path)
        try:
            count = int(component.get_num_materials())
        except Exception:
            count = 0
        for index in range(count):
            try:
                material = component.get_material(index)
            except Exception:
                material = None
            path = object_path(material)
            material_class = None
            try:
                material_class = str(material.get_class().get_path_name())
            except Exception:
                pass
            dynamic = bool(material_class and "MaterialInstanceDynamic" in material_class) or bool(
                path and path.startswith("/Engine/Transient"))
            slots.append({"component_identity": identity, "slot_index": index, "material_path": path,
                          "material_class_path": material_class, "is_dynamic": dynamic})
            if path and path not in material_paths:
                material_paths.append(path)
    slots.sort(key=lambda s: (s["component_identity"], s["slot_index"]))
    return sorted(asset_paths), sorted(material_paths), slots, components


def actor_snapshot(actor):
    tags, parsed = tag_values(actor)
    assets, materials, slots, components = component_data(actor)
    name = str(actor.get_name())
    class_path = str(actor.get_class().get_path_name())
    class_asset = class_path if class_path.startswith("/Game/") else None
    explicit = _tag(parsed, "simcodearena.stable_actor_id", "stable_actor_id")
    if explicit:
        stable_id, provenance = explicit, "explicit_tag"
    else:
        stable_id, provenance = generated_stable_actor_id(name, class_path, assets), "generated_structural_identity"
    try:
        parent = actor.get_attach_parent_actor()
    except Exception:
        parent = None
    try:
        origin, extent = actor.get_actor_bounds(False)
        bounds = {"origin_cm": _vector(origin), "extent_cm": _vector(extent)}
    except Exception:
        bounds = None
    return {
        "stable_actor_id": stable_id,
        "actor_origin": _tag(parsed, "simcodearena.actor_origin", "actor_origin"),
        "logical_object_id": _tag(parsed, "simcodearena.logical_object_id", "logical_object_id"),
        "actor_role": _tag(parsed, "simcodearena.actor_role", "actor_role"),
        "attachment_parent_path": object_path(parent) if parent else None,
        "attachment_parent_label": str(parent.get_actor_label()) if parent else None,
        "label": str(actor.get_actor_label()),
        "name": name,
        "actor_path": str(actor.get_path_name()),
        "class": class_path,
        "asset_path": assets[0] if len(assets) == 1 else class_asset,
        "component_asset_paths": assets,
        "material_paths": materials,
        "component_material_slots": slots,
        "actor_tags": tags,
        "transform": {
            "location_cm": _vector(actor.get_actor_location()),
            "rotation_deg": _rotator(actor.get_actor_rotation()),
            "scale": _vector(actor.get_actor_scale3d()),
        },
        "bounds": bounds,
        "properties": _task_relevant_properties(actor, components),
        "export_diagnostics": {"stable_actor_id_provenance": provenance},
    }


def snapshot_level_actors():
    """Return (records, actor objects) for every actor in the loaded level."""

    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    objects = list(subsystem.get_all_level_actors())
    records = [actor_snapshot(actor) for actor in objects]
    disambiguate(records, current_map_package())
    owners = {}
    for record in records:
        if record["stable_actor_id"] in owners:
            raise RuntimeError("duplicate stable_actor_id {} ({} and {})".format(
                record["stable_actor_id"], owners[record["stable_actor_id"]], record["label"]))
        owners[record["stable_actor_id"]] = record["label"]
    return records, objects


def export_snapshot(output_path):
    records, _ = snapshot_level_actors()
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_type": "ue_editor_scene",
        "units": "cm",
        "map_path": current_map_package(),
        "actor_count": len(records),
        "actors": records,
        "export_metadata": {"status": "success", "exporter": "code4scene.dataset_builder"},
    }
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(temporary, output_path)
    return {"path": output_path, "actor_count": len(records), "map_path": payload["map_path"]}
