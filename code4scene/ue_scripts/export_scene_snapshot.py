"""Export the current or requested UE editor map as a scene snapshot.

This file is executed inside Unreal's Python interpreter. The host adapter
injects SCENE_DISTANCE_MAP and optionally SCENE_DISTANCE_OUTPUT before this
source.  The payload is also published in ``_SB_SCENE_SNAPSHOT_PAYLOAD`` so
case authoring and evaluation use the exact same in-memory evidence contract.
"""

import hashlib
import json
import os
import traceback
import unreal


# Keep this list aligned with the benchmark identity contract. These are scene
# properties whose changes can affect preservation; exporter diagnostics must
# not be mixed into this object.
TASK_RELEVANT_PROPERTY_KEYS = (
    "intensity",
    "intensity_units",
    "light_color",
    "temperature",
    "use_temperature",
    "source_angle",
    "indirect_lighting_intensity",
    "volumetric_scattering_intensity",
    "fog_density",
    "fog_height_falloff",
    "start_distance",
    "sun_disk_scale",
    "cloud_opacity",
    "exposure_compensation",
    "min_brightness",
    "max_brightness",
    "white_temp",
    "color_saturation",
    "color_contrast",
    "color_gamma",
    "color_gain",
    "color_offset",
)

GENERATED_STABLE_ID_PREFIX = "sca_generated_"
_MATERIAL_PARAMETER_CATALOG = {}



DUPLICATE_ID_HINT = (
    "copying a tagged Actor copies its simcodearena.stable_actor_id"
    " tag, so remove or change the tag on the copy and save again"
)

def _object_path(value):
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


def _current_map_package():
    world = unreal.EditorLevelLibrary.get_editor_world()
    if world is None:
        return None
    path = str(world.get_path_name())
    return path.split(".", 1)[0]


def _load_map(package_path):
    if not package_path:
        return
    package_path = str(package_path).replace("\\", "/")
    if package_path.endswith(".umap"):
        package_path = package_path[:-5]
    if not package_path.startswith("/Game/"):
        raise RuntimeError(f"UE map must be a /Game package path, got: {package_path}")
    try:
        registry = unreal.AssetRegistryHelpers.get_asset_registry()
        registry.scan_paths_synchronous([package_path.rsplit("/", 1)[0]])
    except Exception:
        pass
    try:
        loaded = unreal.EditorLoadingAndSavingUtils.load_map(package_path)
    except Exception:
        loaded = None
    if loaded is None:
        # Some UE versions expose the same operation only through the legacy
        # editor level library.
        unreal.EditorLevelLibrary.load_level(package_path)
    current = _current_map_package()
    if not current or current != package_path:
        raise RuntimeError(f"Requested map {package_path} but editor reports {current}")


def _tag_values(actor):
    try:
        tags = [str(value) for value in actor.tags]
    except Exception:
        tags = []
    parsed = {}
    for tag in tags:
        if "=" not in tag:
            continue
        key, value = tag.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return tags, parsed


def _tag(parsed, *keys):
    for key in keys:
        value = parsed.get(key.lower())
        if value:
            return value
    return None


def _structural_actor_signature(actor_name, class_path, component_asset_paths):
    """Map-independent identity for Actors whose tags cannot persist.

    Some construction-script/prefab child Actors are recreated whenever a map
    loads, so an editor-written Actor tag disappears on the next load.  Their
    internal Actor name, class, and component assets remain identical across a
    canonical map and its derived Inputs.  Keep this signature deliberately
    narrow: map packages, GUIDs, transforms, labels, and UObject reprs would
    make equivalent copies disagree.
    """
    return {
        "actor_name": str(actor_name),
        "class_path": str(class_path),
        "component_asset_paths": sorted(
            set(str(path) for path in component_asset_paths)
        ),
    }


def _generated_stable_actor_id(actor_name, class_path, component_asset_paths):
    signature = _structural_actor_signature(
        actor_name, class_path, component_asset_paths
    )
    canonical = json.dumps(
        signature, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return GENERATED_STABLE_ID_PREFIX + hashlib.sha256(canonical).hexdigest()[:32]


def _stable_actor_id(explicit, actor_name, class_path, component_asset_paths):
    if explicit:
        return str(explicit), "explicit_tag"
    return (
        _generated_stable_actor_id(actor_name, class_path, component_asset_paths),
        "generated_structural_identity",
    )


def _normalized_actor_scope(actor_path, map_package):
    """Return a map-copy-stable owner scope for one Actor path.

    Actors in different streamed sublevels may legally have the same UObject
    name, class, and component assets. Treating only those structural fields
    as identity therefore collides on large worlds. The owning sublevel is
    stable across GT/Input/Candidate, while the root level package is renamed
    whenever a map is copied; normalize only that root package.
    """

    path = str(actor_path or "")
    marker = ":PersistentLevel."
    if marker not in path:
        return path
    owner = path.split(marker, 1)[0]
    owner_package = owner.split(".", 1)[0]
    if owner_package == str(map_package or ""):
        return "$ROOT_LEVEL"
    return owner


def _disambiguate_generated_stable_actor_ids(actors, map_package):
    """Disambiguate only colliding fallback IDs by stable owning sublevel.

    Existing non-colliding generated IDs stay byte-for-byte compatible. An
    explicit-ID collision remains an authoring error. A collision that the
    owner scope cannot resolve is also refused instead of silently choosing a
    transform-dependent identity that would break moved-Actor correspondence.
    """

    groups = {}
    for actor in actors:
        groups.setdefault(actor.get("stable_actor_id"), []).append(actor)
    for stable_id, values in groups.items():
        if not stable_id or len(values) < 2:
            continue
        if any(
            actor.get("export_diagnostics", {}).get(
                "stable_actor_id_provenance"
            )
            == "explicit_tag"
            for actor in values
        ):
            continue
        replacements = {}
        for actor in values:
            scope = _normalized_actor_scope(actor.get("actor_path"), map_package)
            canonical = json.dumps(
                {"base_stable_actor_id": stable_id, "actor_scope": scope},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            replacement = GENERATED_STABLE_ID_PREFIX + hashlib.sha256(
                canonical
            ).hexdigest()[:32]
            if replacement in replacements:
                raise RuntimeError(
                    "generated stable_actor_id collision {} remains ambiguous "
                    "inside Actor scope {}".format(stable_id, scope)
                )
            replacements[replacement] = actor
        for replacement, actor in replacements.items():
            actor["stable_actor_id"] = replacement
            actor["export_diagnostics"]["stable_actor_id_provenance"] = (
                "generated_structural_identity_disambiguated_by_actor_scope"
            )
            actor["export_diagnostics"]["stable_actor_id_base"] = stable_id
            actor["export_diagnostics"]["stable_actor_scope"] = (
                _normalized_actor_scope(actor.get("actor_path"), map_package)
            )


def _validate_stable_actor_ids(actors):
    owners = {}
    for actor in actors:
        stable_id = actor.get("stable_actor_id")
        if not stable_id:
            raise RuntimeError(
                "Actor {} has no explicit or generated stable ID".format(
                    actor.get("name") or actor.get("actor_path")
                )
            )
        previous = owners.get(stable_id)
        if previous is not None:
            raise RuntimeError(
                "duplicate stable_actor_id {} on Actors {} and {}: {}".format(
                    stable_id, previous, actor.get("name"), DUPLICATE_ID_HINT
                )
            )
        owners[stable_id] = actor.get("name")


def _actor_guid(actor):
    def _guid_text(value):
        for method_name in ("to_string", "export_text"):
            try:
                method = getattr(value, method_name, None)
                text = str(method() if callable(method) else "").strip()
                if text and text.lower() not in ("none", "invalid"):
                    return text
            except Exception:
                pass
        text = str(value).strip()
        # UE's default Python Struct repr contains a process-local address and
        # is not an Actor identity. Never export it as a GUID fallback.
        if text.startswith("<Struct '"):
            return None
        return text if text and text.lower() not in ("none", "invalid") else None

    try:
        value = actor.get_actor_guid()
        text = _guid_text(value)
        if text:
            return text
    except Exception:
        pass
    try:
        value = actor.get_editor_property("actor_guid")
        text = _guid_text(value)
        if text:
            return text
    except Exception:
        pass
    return None


def _serialized_property(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if all(hasattr(value, key) for key in ("r", "g", "b", "a")):
        return [value.r, value.g, value.b, value.a]
    if all(hasattr(value, key) for key in ("x", "y", "z", "w")):
        return [float(value.x), float(value.y), float(value.z), float(value.w)]
    if all(hasattr(value, key) for key in ("x", "y", "z")):
        return [float(value.x), float(value.y), float(value.z)]
    # UE 5.8 exposes reflected enums (for example LightUnits) as built-in
    # proxy objects rather than Python enum.Enum subclasses. They still carry
    # stable name/value attributes and stringify to the benchmark representation.
    if hasattr(value, "name") and hasattr(value, "value"):
        return str(value)
    # UE enums stringify deterministically. Other opaque UObject/struct values
    # are intentionally omitted rather than leaking diagnostics into the
    # preservation contract.
    value_type = type(value).__name__.lower()
    if "enum" in value_type:
        return str(value)
    return None


def _material_parameter_names(material, kind, errors):
    """Return declared parameter names without assuming one UE minor API."""

    library = unreal.MaterialEditingLibrary
    method = getattr(library, "get_{}_parameter_names".format(kind), None)
    if method is None:
        errors.append("{}_parameter_names_api_unavailable".format(kind))
        return []
    try:
        return sorted(set(str(value) for value in method(material)))
    except Exception as error:
        errors.append("{}_parameter_names: {}".format(kind, error))
        return []


def _unwrap_parameter_value(value):
    # Some reflected UE functions return (success, value) while others return
    # the value directly. Do not treat a reported failure as a zero value.
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], bool):
        return value[1] if value[0] else None
    return value


def _material_parameter_value(material, material_class, kind, name, errors):
    """Resolve the effective instance/default value for one parameter."""

    direct = getattr(material, "get_{}_parameter_value".format(kind), None)
    if direct is not None:
        try:
            value = _unwrap_parameter_value(direct(name))
            if value is not None:
                return value
        except Exception as error:
            errors.append("{}_{} direct: {}".format(kind, name, error))

    library = unreal.MaterialEditingLibrary
    if material_class and "MaterialInstance" in material_class:
        method_name = "get_material_instance_{}_parameter_value".format(kind)
    else:
        method_name = "get_material_default_{}_parameter_value".format(kind)
    method = getattr(library, method_name, None)
    if method is None:
        errors.append("{}_api_unavailable".format(method_name))
        return None
    try:
        return _unwrap_parameter_value(method(material, name))
    except Exception as error:
        errors.append("{}_{}: {}".format(kind, name, error))
        return None


def _material_used_textures(material, errors):
    """Return resolved texture dependencies, or None when UE cannot prove it."""

    available = False
    for method_name in ("get_material_used_textures", "get_used_textures"):
        method = getattr(unreal.MaterialEditingLibrary, method_name, None)
        if method is None:
            continue
        available = True
        try:
            return sorted(
                set(
                    path
                    for path in (
                        _object_path(value) for value in method(material)
                    )
                    if path
                )
            )
        except Exception as error:
            errors.append("{}: {}".format(method_name, error))
    if not available:
        errors.append("material_used_textures_api_unavailable")
    return None


def _material_parameter_evidence(material, material_path, material_class):
    """Cache read-only, resolved material evidence once per material path."""

    if not material_path or material_path in _MATERIAL_PARAMETER_CATALOG:
        return
    errors = []
    scalars = {}
    vectors = {}
    textures = {}
    for name in _material_parameter_names(material, "scalar", errors):
        value = _material_parameter_value(
            material, material_class, "scalar", name, errors
        )
        serialized = _serialized_property(value)
        if isinstance(serialized, (int, float)) and not isinstance(serialized, bool):
            scalars[name] = serialized
    for name in _material_parameter_names(material, "vector", errors):
        value = _material_parameter_value(
            material, material_class, "vector", name, errors
        )
        serialized = _serialized_property(value)
        if isinstance(serialized, list) and len(serialized) >= 3:
            vectors[name] = serialized
    for name in _material_parameter_names(material, "texture", errors):
        value = _material_parameter_value(
            material, material_class, "texture", name, errors
        )
        path = _object_path(value)
        if path:
            textures[name] = path
    used_textures = _material_used_textures(material, errors)
    _MATERIAL_PARAMETER_CATALOG[material_path] = {
        "schema_version": "1.0",
        "material_path": material_path,
        "material_class_path": material_class,
        "resolved_scalar_parameters": scalars,
        "resolved_vector_parameters": vectors,
        "resolved_texture_parameters": textures,
        "used_texture_paths": used_textures,
        "probe_status": "success" if not errors else "partial",
        "probe_errors": errors,
        "source": "ue_material_editing_library_resolved_parameters_v1",
    }


def _task_relevant_properties(actor, components):
    result = {}
    owners = [actor] + list(components)
    for property_name in TASK_RELEVANT_PROPERTY_KEYS:
        for owner in owners:
            try:
                value = owner.get_editor_property(property_name)
            except Exception:
                continue
            serialized = _serialized_property(value)
            if serialized is not None:
                result[property_name] = serialized
                break
    return result


def _component_data(actor):
    asset_paths = []
    material_paths = []
    material_slots = []
    component_classes = []
    try:
        components = actor.get_components_by_class(unreal.ActorComponent)
    except Exception:
        components = []
    for component in components:
        try:
            component_class = str(component.get_class().get_path_name())
            component_classes.append(component_class)
        except Exception:
            component_class = type(component).__name__
        try:
            component_name = str(component.get_name())
        except Exception:
            component_name = component_class
        component_identity = f"{component_name}|{component_class}"
        for property_name in ("static_mesh", "skeletal_mesh_asset", "skeletal_mesh"):
            try:
                asset = component.get_editor_property(property_name)
            except Exception:
                continue
            path = _object_path(asset)
            if path and path not in asset_paths:
                asset_paths.append(path)
        try:
            material_count = int(component.get_num_materials())
        except Exception:
            material_count = 0
        for index in range(material_count):
            try:
                material = component.get_material(index)
            except Exception:
                material = None
            path = _object_path(material)
            material_class = None
            try:
                material_class = str(material.get_class().get_path_name())
            except Exception:
                pass
            dynamic = bool(
                material_class
                and "MaterialInstanceDynamic" in material_class
            ) or bool(path and path.startswith("/Engine/Transient"))
            if material is not None:
                _material_parameter_evidence(
                    material, path, material_class
                )
            material_slots.append({
                "component_identity": component_identity,
                "slot_index": index,
                "material_path": path,
                "material_class_path": material_class,
                "is_dynamic": dynamic,
            })
            if path and path not in material_paths:
                material_paths.append(path)
    return (
        sorted(asset_paths),
        sorted(material_paths),
        sorted(
            material_slots,
            key=lambda value: (
                value["component_identity"],

                value["slot_index"],
            ),
        ),
        sorted(set(component_classes)),
        components,
    )


def _collision_data(components):
    primitive_count = 0
    enabled_count = 0
    enabled_components = []
    for component in components:
        collision_method = getattr(component, "get_collision_enabled", None)
        if collision_method is None:
            continue
        primitive_count += 1
        try:
            state = str(collision_method())
        except Exception:
            continue
        normalized = state.lower().replace(" ", "_")
        if "no_collision" in normalized or normalized.endswith(".none"):
            continue
        enabled_count += 1
        enabled_components.append({
            "name": str(component.get_name()),
            "class": str(component.get_class().get_path_name()),
            "collision_enabled": state,
        })
    return {
        "collision_enabled": enabled_count > 0,
        "primitive_component_count": primitive_count,
        "collision_enabled_component_count": enabled_count,
        "collision_enabled_components": enabled_components,
    }


def _actor_snapshot(actor):
    label = str(actor.get_actor_label())
    location = actor.get_actor_location()
    rotation = actor.get_actor_rotation()
    scale = actor.get_actor_scale3d()
    errors = []
    try:
        origin, extent = actor.get_actor_bounds(False)
    except Exception as error:
        origin = location
        extent = unreal.Vector(0.0, 0.0, 0.0)
        errors.append(f"bounds: {error}")
    tags, parsed_tags = _tag_values(actor)
    component_assets, materials, material_slots, component_classes, components = (
        _component_data(actor)
    )
    collision = _collision_data(components)
    actor_name = str(actor.get_name())
    class_path = str(actor.get_class().get_path_name())
    class_asset = class_path if class_path.startswith("/Game/") else None
    # A single component mesh is a useful primary asset. Multi-mesh actors
    # (notably InstancedFoliageActor) have no canonical first asset, so retain
    # their full component_asset_paths and leave asset_path unset.
    asset_path = component_assets[0] if len(component_assets) == 1 else class_asset
    category = _tag(
        parsed_tags,
        "simcodearena.semantic_category",
        "simcodearena.asset_category",
        "semantic_category",
        "asset_category",
        "category",
    )
    explicit_stable_id = _tag(
        parsed_tags, "simcodearena.stable_actor_id", "stable_actor_id"
    )
    stable_actor_id, stable_actor_id_provenance = _stable_actor_id(
        explicit_stable_id, actor_name, class_path, component_assets
    )
    try:
        folder_path = str(actor.get_folder_path())
    except Exception:
        folder_path = None
    try:
        hidden = bool(actor.is_hidden_ed())
    except Exception:
        hidden = None
    try:
        attachment_parent = actor.get_attach_parent_actor()
    except Exception:
        attachment_parent = None
    try:
        attachment_parent_path = (
            str(attachment_parent.get_path_name()) if attachment_parent else None
        )
    except Exception:
        attachment_parent_path = None
    try:
        attachment_parent_label = (
            str(attachment_parent.get_actor_label()) if attachment_parent else None
        )
    except Exception:
        attachment_parent_label = None
    return {
        "stable_actor_id": stable_actor_id,
        "actor_guid": _actor_guid(actor),
        "actor_origin": _tag(parsed_tags, "simcodearena.actor_origin", "actor_origin"),
        "logical_object_id": _tag(parsed_tags, "simcodearena.logical_object_id", "logical_object_id"),
        "actor_role": _tag(parsed_tags, "simcodearena.actor_role", "actor_role"),
        "attachment_parent_path": attachment_parent_path,
        "attachment_parent_label": attachment_parent_label,
        "label": label,
        "name": actor_name,
        "actor_path": str(actor.get_path_name()),
        "class": class_path,
        "asset_path": asset_path,
        "asset_category": category,
        "component_asset_paths": component_assets,
        "material_paths": materials,
        "component_material_slots": material_slots,
        "actor_tags": tags,
        "collision": collision,
        "transform": {
            "location_cm": _vector(location),
            "rotation_deg": _rotator(rotation),
            "scale": _vector(scale),
        },
        "bounds": {
            "origin_cm": _vector(origin),
            "extent_cm": _vector(extent),
        },
        "properties": _task_relevant_properties(actor, components),
        "export_diagnostics": {
            "folder_path": folder_path,
            "hidden_in_editor": hidden,
            "component_classes": component_classes,
            "stable_actor_id_provenance": stable_actor_id_provenance,
        },
        "export_errors": errors,
    }


def _write_payload(payload, output_path):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, output_path)


def main():
    requested_map = globals().get("SCENE_DISTANCE_MAP", "")
    output_path = globals().get("SCENE_DISTANCE_OUTPUT", "")
    previous_map = _current_map_package()
    try:
        _load_map(requested_map)
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actors = [_actor_snapshot(actor) for actor in subsystem.get_all_level_actors()]
        _disambiguate_generated_stable_actor_ids(
            actors, _current_map_package()
        )
        _validate_stable_actor_ids(actors)
        payload = {
            "schema_version": "0.3.0",
            "snapshot_type": "ue_editor_scene",
            "units": "cm",
            "map_path": _current_map_package(),
            "actor_count": len(actors),
            "actors": actors,
            "material_parameter_catalog": {
                key: _MATERIAL_PARAMETER_CATALOG[key]
                for key in sorted(_MATERIAL_PARAMETER_CATALOG)
            },
            "export_metadata": {
                "status": "success",
                "requested_map": requested_map or None,
                "previous_map_package": previous_map,
                "stable_actor_id_count": sum(1 for actor in actors if actor["stable_actor_id"]),
                "explicit_stable_actor_id_count": sum(
                    1 for actor in actors
                    if actor["export_diagnostics"]["stable_actor_id_provenance"]
                    == "explicit_tag"
                ),
                "generated_stable_actor_id_count": sum(
                    1 for actor in actors
                    if actor["export_diagnostics"][
                        "stable_actor_id_provenance"
                    ].startswith("generated_structural_identity")
                ),
                "disambiguated_generated_stable_actor_id_count": sum(
                    1 for actor in actors
                    if actor["export_diagnostics"][
                        "stable_actor_id_provenance"
                    ]
                    == (
                        "generated_structural_identity_"
                        "disambiguated_by_actor_scope"
                    )
                ),
                "generated_stable_actor_id_algorithm": {
                    "prefix": GENERATED_STABLE_ID_PREFIX,
                    "hash": "sha256-first-128-bits",
                    "canonical_json": "sort_keys=True,separators=(',', ':')",
                    "identity_fields": [
                        "actor_name",
                        "class_path",
                        "sorted_unique_component_asset_paths",
                    ],
                    "collision_disambiguator": (
                        "normalized_owning_level_scope_only_when_needed"
                    ),
                },
                "actor_guid_count": sum(1 for actor in actors if actor["actor_guid"]),
                "asset_path_count": sum(1 for actor in actors if actor["asset_path"]),
                "material_actor_count": sum(1 for actor in actors if actor["material_paths"]),
                "material_parameter_catalog_count": len(
                    _MATERIAL_PARAMETER_CATALOG
                ),
                "actor_error_count": sum(1 for actor in actors if actor["export_errors"]),
            },
        }
    except Exception as error:
        payload = {
            "schema_version": "0.3.0",
            "snapshot_type": "ue_editor_scene",
            "units": "cm",
            "actor_count": 0,
            "actors": [],
            "export_metadata": {
                "status": "error",
                "requested_map": requested_map or None,
                "previous_map_package": previous_map,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    globals()["_SB_SCENE_SNAPSHOT_PAYLOAD"] = payload
    if output_path:
        _write_payload(payload, output_path)
    print(
        "SCENE_DISTANCE_SNAPSHOT status={} path={}".format(
            payload["export_metadata"]["status"], output_path or "<memory>"
        )
    )


main()
