"""Measure per-Actor physics evidence in the currently loaded UE editor map.

The host adapter injects SCENE_PHYSICS_OUTPUT plus JSON-encoded target and
option values before executing this file in Unreal Python.
"""

import json
import os
import re
import time
import traceback
import unreal


NON_SOLID_SEMANTIC_TOKEN = (
    "water", "ocean", "sea", "lake", "river", "canal", "pond", "pool",
    "sky", "atmosphere", "postprocess", "reflection", "navmesh",
    "particle", "particles", "niagara", "emitter", "vfx", "fxsystem",
)
NON_SOLID_CLASS_SUFFIX = (
    "light", "camera", "fog", "atmosphere", "volume", "reflectioncapture",
    "groupactor", "emitter", "niagaraactor",
)

# Operational broad-phase defaults.  They change only how many exact UE
# collision queries are attempted, never which AABB intersections qualify.
SOLID_GRID_CELL_SIZE_CM = 1000.0
SOLID_GRID_MAX_CELLS_PER_ACTOR = 4096

# I2S local-support v2 is deliberately bounded.  A 5x5 grid on each side of
# one panel component produces at most 50 extra traces for an edited Actor.
LATERAL_SUPPORT_MODEL = "ground_or_lateral_v1"
LATERAL_GRID_DIMENSION = 5
LATERAL_PANEL_THINNESS_RATIO = 4.0
LATERAL_PANEL_MAX_NORMAL_Z = 0.5
LATERAL_PANEL_MIN_VERTICAL_AXIS_Z = 0.75
LATERAL_MIN_COLLIDER_FACE_AREA_RATIO = 4.0


def _current_map_package():
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    if world is None:
        return None
    return str(world.get_path_name()).split(".", 1)[0]


def _actor_path(actor):
    try:
        return str(actor.get_path_name())
    except Exception:
        return None


def _actor_label(actor):
    try:
        return str(actor.get_actor_label())
    except Exception:
        try:
            return str(actor.get_name())
        except Exception:
            return "unknown_actor"


def _actor_tags(actor):
    try:
        return [str(value) for value in actor.tags]
    except Exception:
        return []


def _actor_class_path(actor):
    try:
        return str(actor.get_class().get_path_name())
    except Exception:
        return None


def _stable_actor_id(actor):
    for tag in _actor_tags(actor):
        if "=" not in tag:
            continue
        key, value = tag.split("=", 1)
        if key.strip().lower() in ("simcodearena.stable_actor_id", "stable_actor_id"):
            return value.strip() or None
    return None


def _semantic_text(actor):
    # The actor's own names only: level and folder names say nothing about it.
    values = [_actor_label(actor)] + _actor_tags(actor)
    for name_of in (lambda: actor.get_name(), lambda: actor.get_class().get_name()):
        try:
            values.append(str(name_of()))
        except Exception:
            pass
    try:
        for component in actor.get_components_by_class(unreal.ActorComponent):
            for property_name in ("static_mesh", "skeletal_mesh_asset", "skeletal_mesh"):
                try:
                    asset = component.get_editor_property(property_name)
                    if asset:
                        values.append(str(asset.get_name()))
                except Exception:
                    pass
    except Exception:
        pass
    return " ".join(str(value).lower() for value in values if value)


def _is_non_solid(actor):
    text = _semantic_text(actor)
    # Asset package names are not semantic substrings: a folder containing the
    # word "Lighthouse" must not match the Actor concept "Light".  A raw
    # substring check classified the entire island as a non-solid light and
    # discarded valid terrain hits.
    tokens = set(re.findall(r"[a-z0-9]+", text.replace("_", " ")))
    if any(token in tokens for token in NON_SOLID_SEMANTIC_TOKEN):
        return True
    try:
        class_name = str(actor.get_class().get_name()).lower()
    except Exception:
        class_name = ""
    return class_name.endswith(NON_SOLID_CLASS_SUFFIX)


def _collision_enabled(component):
    """Return whether a PrimitiveComponent participates in collision queries."""
    try:
        value = component.get_collision_enabled()
    except Exception:
        return False
    text = str(value).lower().replace(" ", "_")
    return "no_collision" not in text and not text.endswith(".none")


def _collidable_components(actor):
    try:
        components = list(actor.get_components_by_class(unreal.PrimitiveComponent))
    except Exception:
        return []
    return [component for component in components if _collision_enabled(component)]


def _aabb_overlap_depths(left_origin, left_extent, right_origin, right_extent):
    return [
        min(
            float(left_origin.x + left_extent.x),
            float(right_origin.x + right_extent.x),
        ) - max(
            float(left_origin.x - left_extent.x),
            float(right_origin.x - right_extent.x),
        ),
        min(
            float(left_origin.y + left_extent.y),
            float(right_origin.y + right_extent.y),
        ) - max(
            float(left_origin.y - left_extent.y),
            float(right_origin.y - right_extent.y),
        ),
        min(
            float(left_origin.z + left_extent.z),
            float(right_origin.z + right_extent.z),
        ) - max(
            float(left_origin.z - left_extent.z),
            float(right_origin.z - right_extent.z),
        ),
    ]


def _object_type_queries():
    values = []
    object_type_query = getattr(unreal, "ObjectTypeQuery", None)
    if object_type_query is None:
        return values
    # UE 5.8 gives the built-in object queries descriptive names and keeps the
    # numbered entries only as deprecated aliases. Prefer the descriptive
    # entries while retaining numbered compatibility for older UE versions and
    # project-defined object channels.
    for name in (
        "ECC_WORLD_STATIC", "ECC_WORLD_DYNAMIC", "ECC_PAWN",
        "ECC_PHYSICS_BODY", "ECC_VEHICLE", "ECC_DESTRUCTIBLE",
    ):
        if hasattr(object_type_query, name):
            values.append(getattr(object_type_query, name))
    first_legacy_index = 7 if values else 1
    for index in range(first_legacy_index, 33):
        name = f"OBJECT_TYPE_QUERY{index}"
        if hasattr(object_type_query, name):
            values.append(getattr(object_type_query, name))
    return values


def _overlap_components(result):
    """Normalize UE's direct Array or (success, Array) overlap return layouts."""
    if result is None:
        return []
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
        result = result[1]
    try:
        values = list(result or [])
    except TypeError:
        values = [result]
    return [value for value in values if hasattr(value, "get_owner")]


def _grid_cell_range(origin, extent, cell_size, maximum_cells):
    """Return every spatial-hash cell touched by an AABB, or None if huge.

    Huge terrain/building bounds belong in the cache's overflow list.  Querying
    that short list for every target is cheaper than materializing millions of
    cells and remains exact because overflow entries are never discarded.
    """
    minimum = (
        int(float(origin.x - extent.x) // cell_size),
        int(float(origin.y - extent.y) // cell_size),
        int(float(origin.z - extent.z) // cell_size),
    )
    maximum = (
        int(float(origin.x + extent.x) // cell_size),
        int(float(origin.y + extent.y) // cell_size),
        int(float(origin.z + extent.z) // cell_size),
    )
    counts = [maximum[axis] - minimum[axis] + 1 for axis in range(3)]
    if any(count <= 0 for count in counts):
        return None
    if counts[0] * counts[1] * counts[2] > maximum_cells:
        return None
    return [
        (x, y, z)
        for x in range(minimum[0], maximum[0] + 1)
        for y in range(minimum[1], maximum[1] + 1)
        for z in range(minimum[2], maximum[2] + 1)
    ]


def _build_solid_actor_cache(all_actors, options=None):
    """Resolve cheap collider semantics/bounds and build an exact spatial index.

    Collision-component discovery crosses the UE Python boundary and can load
    expensive body setup data.  Resolving it eagerly for every Actor made the
    first chunk of a 2k-Actor scene take longer than its entire deadline even
    though most Actors were spatially irrelevant.  Entries therefore resolve
    components lazily, after the exact AABB gate, and retain that answer for
    later targets/chunks in the same loaded world.
    """
    options = options or {}
    cell_size = max(
        1.0, float(options.get("solid_broad_phase_grid_cell_cm",
                               SOLID_GRID_CELL_SIZE_CM))
    )
    maximum_cells = max(
        1, int(options.get("solid_broad_phase_max_cells_per_actor",
                           SOLID_GRID_MAX_CELLS_PER_ACTOR))
    )
    entries = []
    grid = {}
    overflow_entries = []
    filtered = {"semantic_non_solid": 0}
    for collider in all_actors:
        if _is_non_solid(collider):
            filtered["semantic_non_solid"] += 1
            continue
        try:
            origin, extent = collider.get_actor_bounds(False)
        except Exception:
            continue
        entry = {
            "actor": collider,
            "actor_path": _actor_path(collider),
            "actor_label": _actor_label(collider),
            "components": None,
            "collision_components_resolved": False,
            "origin": origin,
            "extent": extent,
        }
        entries.append(entry)
        cells = _grid_cell_range(origin, extent, cell_size, maximum_cells)
        if cells is None:
            overflow_entries.append(entry)
            continue
        for cell in cells:
            grid.setdefault(cell, []).append(entry)
    return {
        "entries": entries,
        "by_actor_path": {
            entry["actor_path"]: entry
            for entry in entries
            if entry.get("actor_path")
        },
        "filtered": filtered,
        "grid": grid,
        "overflow_entries": overflow_entries,
        "grid_cell_size_cm": cell_size,
        "grid_max_cells_per_actor": maximum_cells,
        "component_resolution": {
            "checked_actor_count": 0,
            "collision_disabled_actor_count": 0,
        },
    }


def _cached_collidable_components(entry, cache):
    """Resolve one spatial candidate's collision components at most once."""
    if entry.get("collision_components_resolved") is not True:
        components = _collidable_components(entry["actor"])
        entry["components"] = components
        entry["collision_components_resolved"] = True
        resolution = cache["component_resolution"]
        resolution["checked_actor_count"] += 1
        if not components:
            resolution["collision_disabled_actor_count"] += 1
    return entry.get("components") or []


def _spatial_broad_phase_entries(cache, origin, extent):
    cells = _grid_cell_range(
        origin,
        extent,
        cache["grid_cell_size_cm"],
        cache["grid_max_cells_per_actor"],
    )
    if cells is None:
        return list(cache["entries"]), "all_entries_for_large_target"
    unique = {}
    for entry in cache["overflow_entries"]:
        unique[id(entry["actor"])] = entry
    for cell in cells:
        for entry in cache["grid"].get(cell, ()):
            unique[id(entry["actor"])] = entry
    return list(unique.values()), "uniform_grid_exact"


def _solid_broad_phase_candidates(actor, all_actors, tolerance,
                                  solid_actor_cache=None):
    origin, extent = actor.get_actor_bounds(False)
    candidates = []
    cache = solid_actor_cache or _build_solid_actor_cache(all_actors)
    filtered = {
        "semantic_non_solid": cache["filtered"]["semantic_non_solid"],
        "collision_disabled": 0,
    }
    actor_path = _actor_path(actor)
    broad_phase_entries, query_method = _spatial_broad_phase_entries(
        cache, origin, extent
    )
    for cached in broad_phase_entries:
        collider = cached["actor"]
        collider_path = cached["actor_path"]
        if collider is actor or collider_path == actor_path:
            continue
        # Another measurement target is still a physical collider.  Excluding
        # every selected Actor here made a candidate_all contract blind to all
        # StaticMeshActor-vs-StaticMeshActor penetrations.  The current Actor
        # itself was already excluded above; semantic and collision-enabled
        # filters below decide whether every other Actor is a solid candidate.
        collider_origin, collider_extent = cached["origin"], cached["extent"]
        depths = _aabb_overlap_depths(origin, extent, collider_origin, collider_extent)
        if not all(depth > tolerance for depth in depths):
            continue
        components = _cached_collidable_components(cached, cache)
        if not components:
            filtered["collision_disabled"] += 1
            continue
        candidates.append({
            "actor": collider,
            "actor_path": collider_path,
            "actor_label": _actor_label(collider),
            "components": components,
            "aabb_overlap_cm": [float(depth) for depth in depths],
            "estimated_minimum_penetration_cm": float(min(depths) - tolerance),
        })
    candidates.sort(key=lambda item: str(item.get("actor_path") or ""))
    return candidates, filtered, {
        "method": query_method,
        "total_semantic_solid_actor_count": len(cache["entries"]),
        "considered_actor_count": len(broad_phase_entries),
        "collision_component_checked_actor_count": (
            cache["component_resolution"]["checked_actor_count"]
        ),
        "grid_cell_size_cm": cache["grid_cell_size_cm"],
        "overflow_actor_count": len(cache["overflow_entries"]),
    }


_PENETRATION_DEPTH_PATTERN = re.compile(
    r"(?:^|[, (])PenetrationDepth=(-?[0-9]+(?:\.[0-9]+)?)"
)


def _component_path(component):
    try:
        return str(component.get_path_name())
    except Exception:
        return None


def _collision_response_token(value):
    """Normalize UE's reflected ECollisionResponse value."""
    if value is None:
        return None
    tokens = set(re.findall(r"[a-z]+", str(value).lower()))
    for response in ("ignore", "overlap", "block"):
        if response in tokens:
            return response
    raw_value = getattr(value, "value", value)
    if not isinstance(raw_value, bool):
        try:
            numeric = int(raw_value)
        except (TypeError, ValueError):
            numeric = None
        if numeric in (0, 1, 2):
            return ("ignore", "overlap", "block")[numeric]
    return None


def _component_pair_collision_response(target_component, collider_component):
    """Return the effective response for one component pair.

    UE resolves a pair to the least blocking response declared by either side.
    An object-query overlap can still enumerate Ignore/Overlap pairs, while a
    swept movement produces an initial-overlap MTD only for a blocking pair.
    Record both declarations so a touch-only query is never misrepresented as
    unresolved solid penetration.
    """
    errors = []
    try:
        target_object_type = target_component.get_collision_object_type()
    except Exception as error:
        target_object_type = None
        errors.append("target object type unavailable: " + str(error))
    try:
        collider_object_type = collider_component.get_collision_object_type()
    except Exception as error:
        collider_object_type = None
        errors.append("collider object type unavailable: " + str(error))
    try:
        target_response_value = (
            target_component.get_collision_response_to_channel(
                collider_object_type
            )
            if collider_object_type is not None else None
        )
    except Exception as error:
        target_response_value = None
        errors.append("target response unavailable: " + str(error))
    try:
        collider_response_value = (
            collider_component.get_collision_response_to_channel(
                target_object_type
            )
            if target_object_type is not None else None
        )
    except Exception as error:
        collider_response_value = None
        errors.append("collider response unavailable: " + str(error))

    target_response = _collision_response_token(target_response_value)
    collider_response = _collision_response_token(collider_response_value)
    declared = (target_response, collider_response)
    if "ignore" in declared:
        effective = "ignore"
    elif "overlap" in declared:
        # Even if the other response could not be reflected, Ignore/Overlap
        # versus any possible response can never resolve to Block.
        effective = "overlap"
    elif declared == ("block", "block"):
        effective = "block"
    else:
        effective = None
    return {
        "target_object_type": (
            str(target_object_type) if target_object_type is not None else None
        ),
        "collider_object_type": (
            str(collider_object_type)
            if collider_object_type is not None else None
        ),
        "target_response": target_response,
        "collider_response": collider_response,
        "effective_response": effective,
        "status": "measured" if effective is not None else "unavailable",
        "errors": errors,
    }


def _unit_probe_directions(target_location, collider_location):
    """Return deterministic directions for reversible initial-overlap probes."""
    delta = (
        float(collider_location.x - target_location.x),
        float(collider_location.y - target_location.y),
        float(collider_location.z - target_location.z),
    )
    length = sum(value * value for value in delta) ** 0.5
    toward = (
        tuple(value / length for value in delta)
        if length > 1e-6 else (1.0, 0.0, 0.0)
    )
    raw = [
        toward,
        tuple(-value for value in toward),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    unique = []
    seen = set()
    for direction in raw:
        key = tuple(round(value, 6) for value in direction)
        if key in seen:
            continue
        seen.add(key)
        unique.append(direction)
    return unique


def _hit_result_object(value):
    """Normalize UE bindings that return FHitResult directly or in a tuple."""
    if hasattr(value, "to_dict"):
        return value
    if isinstance(value, tuple):
        for item in value:
            if hasattr(item, "to_dict"):
                return item
    return None


def _native_mtd_from_hit(hit, collider_owner, collider_path):
    if hit is None:
        return {"status": "unavailable",
                "error": "swept component move returned no FHitResult"}
    hit_values = hit.to_dict()
    hit_actor = hit_values.get("hit_actor")
    hit_component = hit_values.get("hit_component")
    hit_actor_path = _actor_path(hit_actor) if hit_actor is not None else None
    hit_component_path = (
        _component_path(hit_component) if hit_component is not None else None
    )
    expected_actor_path = _actor_path(collider_owner)
    if not bool(hit_values.get("initial_overlap")):
        return {"status": "unavailable",
                "error": "sweep did not report a starting overlap"}
    if hit_actor_path != expected_actor_path:
        return {"status": "unavailable",
                "error": "sweep resolved a different overlapping Actor",
                "hit_actor_path": hit_actor_path}
    if hit_component_path != collider_path:
        return {"status": "unavailable",
                "error": "sweep resolved a different overlapping component",
                "hit_actor_path": hit_actor_path,
                "hit_component_path": hit_component_path}
    match = _PENETRATION_DEPTH_PATTERN.search(str(hit.export_text()))
    if match is None:
        return {"status": "unavailable",
                "error": "FHitResult omitted PenetrationDepth"}
    depth = float(match.group(1))
    if depth < 0.0:
        return {"status": "unavailable",
                "error": "FHitResult returned a negative PenetrationDepth"}
    normal = hit_values.get("normal")
    return {
        "status": "measured",
        "penetration_depth_cm": depth,
        "penetration_normal": (
            [float(normal.x), float(normal.y), float(normal.z)]
            if normal is not None else None
        ),
        "hit_component": (
            str(hit_component.get_name()) if hit_component is not None else None
        ),
        "hit_component_path": hit_component_path,
        "depth_method": "ue_fhitresult_initial_overlap_mtd",
    }


def _native_sweep_penetration_prepared(target_component, collider_component,
                                       original_location):
    """Return UE's native MTD after the caller isolated one collider.

    ``ComponentOverlapComponents`` enumerates the real collision-body contacts
    but does not return ``FHitResult``.  A 0.1 cm swept move from the current
    transform does.  UE marks a starting overlap with ``bStartPenetrating`` and
    stores the minimum-translation distance in ``PenetrationDepth``.

    UE 5.8's Python wrapper deliberately omits that field from ``to_dict`` and
    ``to_tuple`` even though the underlying FHitResult owns it.  ``export_text``
    preserves the native value, so parse only that named field.  The target is
    restored in ``finally``; the probe never saves the level.  Move-ignore
    isolation is managed by ``_native_sweep_penetrations`` once for the whole
    contact set so a target with k contacts performs O(k), not O(k squared),
    Python-to-UE ignore operations.
    """
    collider_path = _component_path(collider_component)
    probe_distance_cm = 0.1
    probe_errors = []
    try:
        target_owner = target_component.get_owner()
        collider_owner = collider_component.get_owner()
        target_location = target_owner.get_actor_location()
        collider_location = collider_owner.get_actor_location()
        directions = _unit_probe_directions(target_location, collider_location)
        for probe_index, direction in enumerate(directions):
            destination = unreal.Vector(
                float(original_location.x) + probe_distance_cm * direction[0],
                float(original_location.y) + probe_distance_cm * direction[1],
                float(original_location.z) + probe_distance_cm * direction[2],
            )
            try:
                hit = _hit_result_object(
                    target_component.set_world_location(
                        destination, True, False
                    )
                )
                measured = _native_mtd_from_hit(
                    hit, collider_owner, collider_path
                )
            except Exception as error:
                measured = {"status": "unavailable", "error": str(error)}
            finally:
                try:
                    target_component.set_world_location(
                        original_location, False, True
                    )
                except Exception:
                    pass
            if measured.get("status") == "measured":
                return {
                    **measured,
                    "probe_distance_cm": probe_distance_cm,
                    "probe_direction": [float(value) for value in direction],
                    "probe_attempt_count": probe_index + 1,
                    "probe_orientation": "target_to_collider",
                }
            probe_errors.append({
                "probe_direction": [float(value) for value in direction],
                "error": measured.get("error") or "native MTD unavailable",
            })
        return {
            "status": "unavailable",
            "error": "all reversible component sweep probes failed",
            "probe_distance_cm": probe_distance_cm,
            "probe_attempt_count": len(directions),
            "probe_errors": probe_errors,
        }
    except Exception as error:
        return {"status": "unavailable", "error": str(error),
                "probe_errors": probe_errors}
    finally:
        try:
            target_component.set_world_location(original_location, False, True)
        except Exception:
            pass


def _native_sweep_penetrations(target_component, overlapping_components,
                               measured_components=None):
    """Measure all native MTD contacts with one reversible isolation setup.

    The old implementation rebuilt a move-ignore list of k components for
    every one of k contacts.  On a building mesh overlapping thousands of room
    props, those O(k squared) reflected calls dominated the entire evaluation.
    This routine ignores the unique contact set once, temporarily enables one
    collider for the same native swept move, then restores the exact original
    state once.  The collision query and returned FHitResult are unchanged.
    """
    unique_components = {}
    for component in overlapping_components:
        component_path = _component_path(component)
        component_key = component_path or f"python-id:{id(component)}"
        unique_components.setdefault(component_key, component)
    measured_unique_components = {}
    for component in (
        overlapping_components
        if measured_components is None
        else measured_components
    ):
        component_path = _component_path(component)
        component_key = component_path or f"python-id:{id(component)}"
        measured_unique_components.setdefault(component_key, component)
    try:
        original_location = target_component.get_world_location()
        original_ignored = list(
            target_component.copy_array_of_move_ignore_components() or []
        )
    except Exception as error:
        unavailable = {"status": "unavailable", "error": str(error)}
        return {key: unavailable for key in measured_unique_components}

    original_ignored_keys = {
        _component_path(component) or f"python-id:{id(component)}"
        for component in original_ignored
    }
    results = {}
    try:
        # A starting component can overlap many colliders, while a swept move
        # returns only one blocking FHitResult.  Isolate the whole set once.
        for component in unique_components.values():
            target_component.ignore_component_when_moving(component, True)
        for component_key, collider_component in measured_unique_components.items():
            if component_key in original_ignored_keys:
                results[component_key] = {
                    "status": "unavailable",
                    "error": "collider was already ignored for movement",
                }
                continue
            try:
                target_component.ignore_component_when_moving(
                    collider_component, False
                )
                results[component_key] = _native_sweep_penetration_prepared(
                    target_component, collider_component, original_location
                )
            except Exception as error:
                results[component_key] = {
                    "status": "unavailable", "error": str(error)
                }
            finally:
                try:
                    target_component.ignore_component_when_moving(
                        collider_component, True
                    )
                except Exception:
                    pass
    except Exception as error:
        unavailable = {"status": "unavailable", "error": str(error)}
        for component_key in measured_unique_components:
            results.setdefault(component_key, unavailable)
    finally:
        try:
            target_component.set_world_location(original_location, False, True)
        except Exception:
            pass
        try:
            target_component.clear_move_ignore_components()
            for component in original_ignored:
                target_component.ignore_component_when_moving(component, True)
        except Exception:
            pass
    return results


def _native_sweep_penetration(target_component, collider_component,
                              overlapping_components):
    """Compatibility wrapper for callers measuring a single contact."""
    component_path = _component_path(collider_component)
    component_key = component_path or f"python-id:{id(collider_component)}"
    return _native_sweep_penetrations(
        target_component, overlapping_components, [collider_component]
    ).get(component_key, {
        "status": "unavailable",
        "error": "collider was absent from the confirmed overlap set",
    })


def _reciprocal_native_sweep_penetration(
        target_component, collider_component, overlap_method, object_types):
    """Retry a blocking pair by sweeping the collider against the target.

    Some component hierarchies do not emit an FHitResult when the queried
    target component moves, while the reciprocal component does.  The returned
    depth is still UE's native starting-overlap MTD.  Invert the normal so all
    contact records retain the target-out-of-collider convention.
    """
    try:
        collider_owner = collider_component.get_owner()
        reciprocal_result = overlap_method(
            collider_component,
            collider_component.get_world_transform(),
            object_types,
            unreal.PrimitiveComponent,
            [collider_owner],
        )
        reciprocal_overlaps = _overlap_components(reciprocal_result)
        target_path = _component_path(target_component)
        target_key = target_path or f"python-id:{id(target_component)}"
        overlap_keys = {
            _component_path(component) or f"python-id:{id(component)}"
            for component in reciprocal_overlaps
        }
        if target_key not in overlap_keys:
            return {
                "status": "unavailable",
                "error": "reciprocal query did not return the target component",
            }
        measured = _native_sweep_penetrations(
            collider_component, reciprocal_overlaps, [target_component]
        ).get(target_key, {
            "status": "unavailable",
            "error": "target was absent from reciprocal MTD results",
        })
        if measured.get("status") != "measured":
            return measured
        normal = measured.get("penetration_normal")
        return {
            **measured,
            "penetration_normal": (
                [-float(value) for value in normal]
                if isinstance(normal, list) and len(normal) == 3 else normal
            ),
            "probe_orientation": "collider_to_target",
        }
    except Exception as error:
        return {"status": "unavailable", "error": str(error)}


def _measure_solid_overlaps(actor, all_actors, options,
                            solid_actor_cache=None):
    """Use AABBs only to select candidates, then confirm with UE collision bodies."""
    # The host derives this gate from the smallest solid-penetration threshold
    # in the frozen case contract. AABB may eliminate contacts whose maximum
    # possible MTD is already within that threshold, but it never invents a
    # contact depth or violation. The gate is stored with every record so a
    # later, tighter re-score can refuse instead of reusing insufficient data.
    tolerance = max(
        0.0, float(options.get("solid_penetration_tolerance_cm", 0.0))
    )
    target_components = _collidable_components(actor)
    if not target_components:
        return {
            "solid_penetration_evaluated": True,
            "solid_penetration_method": "ue_collision_filter_no_collidable_target",
            "solid_penetrations": [],
            "solid_penetration_errors": [],
            "solid_broad_phase_candidates": [],
            "solid_collision_component_count": 0,
            "solid_broad_phase_candidate_count": 0,
            "solid_broad_phase_tolerance_cm": tolerance,
            "solid_filtered_actor_counts": {
                "semantic_non_solid": 0,
                "collision_disabled": 0,
            },
        }
    candidates, filtered, broad_phase_index = _solid_broad_phase_candidates(
        actor, all_actors, tolerance, solid_actor_cache
    )
    base = {
        "solid_collision_component_count": len(target_components),
        "solid_broad_phase_candidate_count": len(candidates),
        "solid_broad_phase_tolerance_cm": tolerance,
        "solid_broad_phase_candidates": [{
            "collider": item["actor_label"],
            "collider_path": item["actor_path"],
            "aabb_overlap_cm": item["aabb_overlap_cm"],
            "estimated_minimum_penetration_cm": item["estimated_minimum_penetration_cm"],
        } for item in candidates],
        "solid_filtered_actor_counts": filtered,
        "solid_broad_phase_index": broad_phase_index,
    }
    if not candidates:
        base.update({
            "solid_penetration_evaluated": True,
            "solid_penetration_method": "ue_aabb_broad_phase_no_candidates",
            "solid_penetrations": [],
            "solid_penetration_errors": [],
        })
        return base
    overlap_method = getattr(unreal.SystemLibrary, "component_overlap_components", None)
    object_types = _object_type_queries()
    if overlap_method is None or not object_types:
        base.update({
            "solid_penetration_evaluated": False,
            "solid_penetration_method": "ue_component_overlap_unavailable",
            "solid_penetrations": [],
            "solid_penetration_errors": ["component_overlap_components or ObjectTypeQuery unavailable"],
        })
        return base
    candidate_by_path = {
        item["actor_path"]: item for item in candidates if item["actor_path"]
    }
    confirmed = {}
    filtered_non_blocking = {}
    errors = []
    for component in target_components:
        try:
            result = overlap_method(
                component,
                component.get_world_transform(),
                object_types,
                unreal.PrimitiveComponent,
                [actor],
            )
            overlapping_components = _overlap_components(result)
            candidate_overlapping_components = []
            pair_responses = {}
            for overlapping_component in overlapping_components:
                try:
                    overlapping_owner = overlapping_component.get_owner()
                except Exception:
                    overlapping_owner = None
                overlapping_owner_path = (
                    _actor_path(overlapping_owner)
                    if overlapping_owner is not None else None
                )
                if overlapping_owner_path in candidate_by_path:
                    response = _component_pair_collision_response(
                        component, overlapping_component
                    )
                    component_path = _component_path(overlapping_component)
                    component_key = (
                        component_path
                        or f"python-id:{id(overlapping_component)}"
                    )
                    pair_responses[component_key] = response
                    if response.get("effective_response") in (
                        "ignore", "overlap"
                    ):
                        contact_key = (
                            overlapping_owner_path,
                            _component_path(component),
                            component_path,
                        )
                        candidate = candidate_by_path[overlapping_owner_path]
                        filtered_non_blocking[contact_key] = {
                            "collider": candidate["actor_label"],
                            "collider_path": overlapping_owner_path,
                            "target_component": str(component.get_name()),
                            "target_component_path": _component_path(component),
                            "collider_component": str(
                                overlapping_component.get_name()
                            ),
                            "collider_component_path": component_path,
                            "collision_response": response,
                            "filter_reason": "effective_response_is_not_block",
                        }
                    else:
                        # An unavailable response remains eligible for an
                        # authoritative FHitResult probe. A successful starting
                        # overlap proves that the effective response blocks.
                        candidate_overlapping_components.append(
                            overlapping_component
                        )
            native_depths = _native_sweep_penetrations(
                component,
                overlapping_components,
                candidate_overlapping_components,
            )
            for overlapping_component in overlapping_components:
                try:
                    owner = overlapping_component.get_owner()
                except Exception:
                    owner = None
                owner_path = _actor_path(owner) if owner is not None else None
                candidate = candidate_by_path.get(owner_path)
                if candidate is None:
                    continue
                overlapping_component_path = _component_path(
                    overlapping_component
                )
                overlapping_component_key = (
                    overlapping_component_path
                    or f"python-id:{id(overlapping_component)}"
                )
                response = pair_responses.get(overlapping_component_key)
                if response is None or response.get(
                    "effective_response"
                ) in ("ignore", "overlap"):
                    continue
                contact_key = (
                    owner_path,
                    _component_path(component),
                    overlapping_component_path,
                )
                native_depth = native_depths.get(overlapping_component_key, {
                    "status": "unavailable",
                    "error": "confirmed overlap was absent from batched MTD results",
                })
                forward_depth = native_depth
                if native_depth.get("status") != "measured":
                    reciprocal_depth = _reciprocal_native_sweep_penetration(
                        component,
                        overlapping_component,
                        overlap_method,
                        object_types,
                    )
                    if reciprocal_depth.get("status") == "measured":
                        native_depth = reciprocal_depth
                    else:
                        native_depth = {
                            **native_depth,
                            "forward_error": forward_depth.get("error"),
                            "forward_probe_errors": forward_depth.get(
                                "probe_errors"
                            ),
                            "reciprocal_error": reciprocal_depth.get("error"),
                            "reciprocal_probe_errors": reciprocal_depth.get(
                                "probe_errors"
                            ),
                        }
                confirmed[contact_key] = {
                    "collider": candidate["actor_label"],
                    "collider_path": owner_path,
                    "confirmed_overlap": True,
                    "penetration_depth_cm": native_depth.get("penetration_depth_cm"),
                    "penetration_normal": native_depth.get("penetration_normal"),
                    "detection_method": (
                        "ue_component_overlap_components+"
                        "ue_fhitresult_initial_overlap_mtd"
                        if native_depth.get("status") == "measured"
                        else "ue_component_overlap_components"
                    ),
                    "depth_method": native_depth.get("depth_method"),
                    "depth_status": native_depth.get("status"),
                    "depth_error": native_depth.get("error"),
                    "depth_probe_distance_cm": native_depth.get("probe_distance_cm"),
                    "depth_probe_direction": native_depth.get("probe_direction"),
                    "depth_probe_attempt_count": native_depth.get(
                        "probe_attempt_count"
                    ),
                    "depth_probe_orientation": native_depth.get(
                        "probe_orientation"
                    ),
                    "depth_probe_errors": native_depth.get("probe_errors"),
                    "depth_forward_error": native_depth.get("forward_error"),
                    "depth_reciprocal_error": native_depth.get(
                        "reciprocal_error"
                    ),
                    "collision_response": response,
                    "target_component": str(component.get_name()),
                    "target_component_path": _component_path(component),
                    "collider_component": str(overlapping_component.get_name()),
                    "collider_component_path": overlapping_component_path,
                    "aabb_overlap_cm": candidate["aabb_overlap_cm"],
                    "estimated_minimum_penetration_cm": candidate["estimated_minimum_penetration_cm"],
                }
        except Exception as error:
            errors.append({"component": str(component.get_name()), "error": str(error)})
    base.update({
        "solid_penetration_evaluated": not errors,
        "solid_penetration_method": (
            "ue_component_overlap_components+pairwise_collision_response+"
            "ue_fhitresult_initial_overlap_mtd"
        ),
        "solid_penetrations": list(confirmed.values()),
        "solid_non_blocking_overlaps": list(filtered_non_blocking.values()),
        "solid_penetration_errors": errors,
        "solid_penetration_coverage": {
            "confirmed_contact_count": len(confirmed),
            "filtered_non_blocking_contact_count": len(filtered_non_blocking),
            "collision_response_unknown_contact_count": sum(
                (item.get("collision_response") or {}).get(
                    "effective_response"
                ) is None
                for item in confirmed.values()
            ),
            "depth_measured_contact_count": sum(
                item.get("depth_status") == "measured"
                for item in confirmed.values()
            ),
            "depth_unresolved_contact_count": sum(
                item.get("depth_status") != "measured"
                for item in confirmed.values()
            ),
        },
    })
    return base


def _attach_solid_overlap_measurement(measurement, actor, all_actors, options,
                                      solid_actor_cache=None):
    # Keep the continuous mesh/surface depth separate from the exact collision
    # body overlap. ComponentOverlapComponents confirms overlap but does not
    # provide a trustworthy minimum-translation depth.
    measurement["mesh_surface_penetration_cm"] = measurement.get("solid_penetration_cm")
    solid = _measure_solid_overlaps(
        actor, all_actors, options, solid_actor_cache
    )
    measurement.update(solid)
    overlaps = solid.get("solid_penetrations") or []
    measurement["solid_overlap_count"] = len(overlaps)
    measurement["solid_overlap_detected"] = bool(overlaps)
    measurement["solid_overlap_depth_available"] = bool(overlaps) and all(
        isinstance(item.get("penetration_depth_cm"), (int, float))
        for item in overlaps
    )
    if solid.get("solid_collision_component_count") == 0:
        # A collision-disabled target is not a physical solid. Preserve the
        # separate ground-contact values, but do not call visual mesh overlap
        # a solid-penetration failure.
        measurement["solid_penetration_cm"] = 0.0
    return measurement


def _measure_actor_solid_only(actor, options, all_actors,
                              solid_actor_cache=None):
    """Collect only fields consumed by the solid-penetration verifier.

    Physical Safety v2 obtains floating from the independent scene-rate probe.
    Repeating mesh-vertex and vertical ground traces here made collision
    scoring pay for the same dimension twice, especially in foliage-heavy
    scenes.  Non-solid and collision-disabled Actors remain explicit trusted
    records; they are not silently removed from coverage.
    """
    origin, extent = actor.get_actor_bounds(False)
    measurement = {
        "actor_path": _actor_path(actor),
        "actor_label": _actor_label(actor),
        "actor_class": _actor_class_path(actor),
        "actor_tags": _actor_tags(actor),
        "stable_actor_id": _stable_actor_id(actor),
        "measurement_method": "ue_solid_penetration_only",
        "bounds_min_cm": [
            float(origin.x - extent.x),
            float(origin.y - extent.y),
            float(origin.z - extent.z),
        ],
        "bounds_max_cm": [
            float(origin.x + extent.x),
            float(origin.y + extent.y),
            float(origin.z + extent.z),
        ],
        "solid_penetration_cm": None,
    }
    if _is_non_solid(actor):
        measurement.update({
            "solid_penetration_evaluated": True,
            "solid_penetration_method": "ue_semantic_non_solid_filtered",
            "solid_penetrations": [],
            "solid_penetration_errors": [],
            "solid_collision_component_count": 0,
            "solid_broad_phase_candidate_count": 0,
            "solid_broad_phase_candidates": [],
            "solid_penetration_cm": 0.0,
        })
        return measurement
    return _attach_solid_overlap_measurement(
        measurement, actor, all_actors, options, solid_actor_cache
    )


def _hit_values(hit):
    try:
        return hit.to_tuple()
    except Exception:
        return ()


def _hit_location(values):
    # UE HitResult exposes location and impact_point next to each other. Prefer
    # impact_point when available, while retaining the tuple layout used by the
    # repository's existing physics_metrics implementation.
    for index in (5, 4):
        try:
            value = values[index]
            if hasattr(value, "z"):
                return value
        except Exception:
            pass
    return None


def _hit_normal(hit):
    try:
        values = hit.to_dict()
    except Exception:
        values = {}
    normal = values.get("impact_normal") or values.get("normal")
    if normal is not None and all(hasattr(normal, axis) for axis in "xyz"):
        return (float(normal.x), float(normal.y), float(normal.z))
    return None


def _hit_actor(values):
    try:
        return values[9]
    except Exception:
        return None


def _initial_overlap(values):
    try:
        return bool(values[1])
    except Exception:
        return False


def _trace_hits(world, actor, start, end):
    trace_type_query = getattr(unreal, "TraceTypeQuery", None)
    trace_channel = (
        getattr(trace_type_query, "ECC_VISIBILITY", None)
        if trace_type_query is not None else None
    )
    if trace_channel is None:
        trace_channel = unreal.TraceTypeQuery.TRACE_TYPE_QUERY1
    arguments = (
        world,
        start,
        end,
        trace_channel,
        True,
        [actor],
        unreal.DrawDebugTrace.NONE,
        True,
    )
    try:
        result = unreal.SystemLibrary.line_trace_multi(*arguments)
        # UE 5.8 returns ``(blocking_hit, Array[HitResult])`` where the Array is
        # an Unreal proxy, not a native Python list/tuple.  Requiring a native
        # sequence silently discarded every real hit by treating the outer
        # tuple as the hit collection.
        if isinstance(result, tuple) and len(result) == 2:
            try:
                return list(result[1] or [])
            except TypeError:
                pass
        values = list(result or [])
        hits = [value for value in values if hasattr(value, "to_tuple")]
        if hits:
            return hits
        return []
    except Exception:
        try:
            result = unreal.SystemLibrary.line_trace_single(*arguments)
            return [result] if result else []
        except Exception:
            return []


def _first_solid_hit(world, actor, start, end, actor_top_z,
                     excluded_actor_paths=None):
    excluded_actor_paths = set(excluded_actor_paths or ())
    for hit in _trace_hits(world, actor, start, end):
        values = _hit_values(hit)
        location = _hit_location(values)
        collider = _hit_actor(values)
        if location is None:
            continue
        if collider is actor or (collider is not None and _is_non_solid(collider)):
            continue
        if collider is not None and _actor_path(collider) in excluded_actor_paths:
            continue
        # Starting just above the target prevents roofs and other geometry
        # entirely above the Actor from becoming false ground contacts.
        if float(location.z) > actor_top_z + 5.0 and not _initial_overlap(values):
            continue
        return {
            "z_cm": float(location.z),
            "initial_overlap": _initial_overlap(values),
            "collider_label": _actor_label(collider) if collider is not None else None,
            "collider_path": _actor_path(collider) if collider is not None else None,
        }
    return None


def _sample_offsets():
    return (
        (0.0, 0.0),
        (-0.65, -0.65), (-0.65, 0.0), (-0.65, 0.65),
        (0.0, -0.65), (0.0, 0.65),
        (0.65, -0.65), (0.65, 0.0), (0.65, 0.65),
    )


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * float(fraction)))
    return float(ordered[max(0, min(len(ordered) - 1, index))])


def _convex_hull_area(points):
    """Return the XY support-polygon area for topology-independent contact."""
    unique = sorted(set((round(float(x), 4), round(float(y), 4)) for x, y in points))
    if len(unique) < 3:
        return 0.0

    def cross(origin, left, right):
        return ((left[0] - origin[0]) * (right[1] - origin[1])
                - (left[1] - origin[1]) * (right[0] - origin[0]))

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return abs(sum(
        hull[index][0] * hull[(index + 1) % len(hull)][1]
        - hull[(index + 1) % len(hull)][0] * hull[index][1]
        for index in range(len(hull))
    )) * 0.5


def _mesh_world_vertices(actor, maximum_samples):
    """Return deterministic LOD0 world-space vertex samples for an Actor."""
    vertices = []
    diagnostics = []
    try:
        components = list(actor.get_components_by_class(unreal.StaticMeshComponent))
    except Exception:
        components = []
    for component in components:
        entry = {"component": str(component.get_name())}
        try:
            mesh = component.get_editor_property("static_mesh")
            if mesh is None:
                entry["status"] = "missing_static_mesh"
                diagnostics.append(entry)
                continue
            entry["mesh_path"] = str(mesh.get_path_name())
            description_method = getattr(mesh, "get_static_mesh_description", None)
            if description_method is None:
                description_method = getattr(mesh, "get_mesh_description", None)
            if description_method is None:
                entry["status"] = "mesh_description_unavailable"
                diagnostics.append(entry)
                continue
            description = description_method(0)
            if description is None or not hasattr(description, "get_vertex_count"):
                entry["status"] = "lod0_description_unavailable"
                diagnostics.append(entry)
                continue
            vertex_count = int(description.get_vertex_count())
            entry["lod0_vertex_count"] = vertex_count
            if vertex_count <= 0:
                entry["status"] = "empty_lod0"
                diagnostics.append(entry)
                continue
            remaining = max(1, int(maximum_samples) - len(vertices))
            stride = max(1, int((vertex_count + remaining - 1) // remaining))
            transform = component.get_world_transform()
            sampled = 0
            for index in range(0, vertex_count, stride):
                vertex_id = unreal.VertexID(index)
                if hasattr(description, "is_vertex_valid") and not description.is_vertex_valid(vertex_id):
                    continue
                local = description.get_vertex_position(vertex_id)
                world_position = transform.transform_location(local)
                vertices.append(world_position)
                sampled += 1
                if len(vertices) >= int(maximum_samples):
                    break
            entry.update({"status": "success", "sample_stride": stride, "sampled_vertex_count": sampled})
        except Exception as error:
            entry.update({"status": "error", "error": str(error)})
        diagnostics.append(entry)
        if len(vertices) >= int(maximum_samples):
            break
    return vertices, diagnostics


def _vector_tuple(value):
    return (float(value.x), float(value.y), float(value.z))


def _vector_length(value):
    return sum(float(component) * float(component) for component in value) ** 0.5


def _normalized(value):
    length = _vector_length(value)
    if length <= 1.0e-6:
        return None
    return tuple(float(component) / length for component in value)


def _dot(left, right):
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _scaled(value, scale):
    return tuple(float(component) * float(scale) for component in value)


def _added(left, right):
    return tuple(float(left[index]) + float(right[index]) for index in range(3))


def _panel_geometry(half_extents, world_axes):
    """Return a rotation-safe panel frame, or a reason it is ineligible."""
    lengths = [_vector_length(axis) for axis in world_axes]
    directions = [_normalized(axis) for axis in world_axes]
    if any(direction is None for direction in directions):
        return None, "degenerate_component_transform"
    physical = [abs(float(half_extents[index])) * lengths[index]
                for index in range(3)]
    thin_axis = min(range(3), key=lambda index: physical[index])
    plane_axes = [index for index in range(3) if index != thin_axis]
    thin = physical[thin_axis]
    if thin <= 1.0e-4:
        return None, "degenerate_panel_thickness"
    if any(
        physical[index] < thin * LATERAL_PANEL_THINNESS_RATIO
        for index in plane_axes
    ):
        return None, "not_a_thin_panel"
    normal = directions[thin_axis]
    if abs(normal[2]) > LATERAL_PANEL_MAX_NORMAL_Z:
        return None, "panel_is_not_vertical"
    vertical_axis = max(plane_axes, key=lambda index: abs(directions[index][2]))
    if abs(directions[vertical_axis][2]) < LATERAL_PANEL_MIN_VERTICAL_AXIS_Z:
        return None, "panel_has_no_vertical_in_plane_axis"
    horizontal_axis = next(index for index in plane_axes if index != vertical_axis)
    face_area = 4.0 * physical[vertical_axis] * physical[horizontal_axis]
    if face_area <= 1.0e-4:
        return None, "degenerate_panel_face"
    return {
        "thin_axis": thin_axis,
        "vertical_axis": vertical_axis,
        "horizontal_axis": horizontal_axis,
        "normal": normal,
        "vertical_direction": directions[vertical_axis],
        "horizontal_direction": directions[horizontal_axis],
        "physical_half_extents_cm": physical,
        "face_area_cm2": face_area,
    }, None


def _component_panel_frame(component):
    try:
        mesh = component.get_editor_property("static_mesh")
        bounds = mesh.get_bounds() if mesh is not None else None
        local_origin = bounds.origin
        local_extent = bounds.box_extent
        transform = component.get_world_transform()
        world_axes = []
        world_zero = transform.transform_location(unreal.Vector(0.0, 0.0, 0.0))
        world_zero_tuple = _vector_tuple(world_zero)
        for axis in (
            unreal.Vector(1.0, 0.0, 0.0),
            unreal.Vector(0.0, 1.0, 0.0),
            unreal.Vector(0.0, 0.0, 1.0),
        ):
            # UE 5.8's Python Transform binding exposes transform_location but
            # not transform_vector.  The difference between a transformed unit
            # endpoint and transformed origin is the same scaled/rotated basis
            # vector, with translation cancelled out.
            world_endpoint = _vector_tuple(transform.transform_location(axis))
            world_axes.append(tuple(
                world_endpoint[index] - world_zero_tuple[index]
                for index in range(3)
            ))
        frame, reason = _panel_geometry(
            _vector_tuple(local_extent), world_axes
        )
        if frame is None:
            return None, reason
        frame.update({
            "component": component,
            "component_name": str(component.get_name()),
            "component_path": _component_path(component),
            "local_origin": local_origin,
            "local_extent": local_extent,
            "transform": transform,
        })
        return frame, None
    except Exception as error:
        return None, "component_bounds_unavailable: " + str(error)


def _best_panel_frame(actor):
    try:
        components = list(actor.get_components_by_class(unreal.StaticMeshComponent))
    except Exception:
        components = []
    eligible = []
    diagnostics = []
    for component in components:
        frame, reason = _component_panel_frame(component)
        diagnostics.append({
            "component": str(component.get_name()),
            "eligible": frame is not None,
            "reason": reason,
        })
        if frame is not None:
            eligible.append(frame)
    if not eligible:
        return None, diagnostics
    return max(eligible, key=lambda value: value["face_area_cm2"]), diagnostics


def _projected_aabb_span(extent, direction):
    return 2.0 * sum(
        abs(float(direction[index])) * float(component)
        for index, component in enumerate((extent.x, extent.y, extent.z))
    )


def _lateral_collider_entry(collider, solid_actor_cache):
    if solid_actor_cache is None:
        return None
    return solid_actor_cache.get("by_actor_path", {}).get(_actor_path(collider))


def _first_lateral_support_hit(world, actor, start, end, face_point,
                               direction, target_paths, frame,
                               maximum_gap, solid_actor_cache):
    for hit in _trace_hits(world, actor, start, end):
        values = _hit_values(hit)
        location = _hit_location(values)
        collider = _hit_actor(values)
        if location is None or collider is None or collider is actor:
            continue
        collider_path = _actor_path(collider)
        if collider_path in target_paths or _is_non_solid(collider):
            continue
        entry = _lateral_collider_entry(collider, solid_actor_cache)
        components = (
            _cached_collidable_components(entry, solid_actor_cache)
            if entry is not None else _collidable_components(collider)
        )
        if not components:
            continue
        normal = _hit_normal(hit)
        if normal is None or abs(_dot(normal, direction)) < 0.8:
            continue
        delta = (
            float(location.x) - face_point[0],
            float(location.y) - face_point[1],
            float(location.z) - face_point[2],
        )
        distance = max(0.0, _dot(delta, direction))
        if distance > maximum_gap:
            continue
        try:
            _collider_origin, collider_extent = collider.get_actor_bounds(False)
        except Exception:
            continue
        collider_face_area = (
            _projected_aabb_span(
                collider_extent, frame["vertical_direction"]
            )
            * _projected_aabb_span(
                collider_extent, frame["horizontal_direction"]
            )
        )
        area_ratio = collider_face_area / frame["face_area_cm2"]
        if area_ratio < LATERAL_MIN_COLLIDER_FACE_AREA_RATIO:
            continue
        return {
            "distance_cm": distance,
            "initial_overlap": _initial_overlap(values),
            "impact_normal": list(normal),
            "collider_label": _actor_label(collider),
            "collider_path": collider_path,
            "collider_face_area_ratio": area_ratio,
        }
    return None


def _measure_lateral_support(actor, world, options, target_paths,
                             solid_actor_cache):
    frame, component_diagnostics = _best_panel_frame(actor)
    base = {
        "lateral_support_detected": False,
        "lateral_support_distance_cm": None,
        "lateral_support_fraction": 0.0,
        "lateral_trace_count": 0,
        "lateral_supporting_colliders": [],
        "lateral_panel_component": None,
        "lateral_panel_diagnostics": component_diagnostics,
    }
    if frame is None:
        base["lateral_support_skip_reason"] = "no_eligible_vertical_panel"
        return base
    maximum_gap = max(
        0.0, float(options.get("lateral_support_tolerance_cm", 5.0))
    )
    minimum_fraction = max(
        0.0, min(1.0, float(options.get(
            "lateral_support_minimum_fraction", 0.05
        )))
    )
    fractions = tuple(
        -0.8 + 1.6 * float(index) / float(LATERAL_GRID_DIMENSION - 1)
        for index in range(LATERAL_GRID_DIMENSION)
    )
    local_origin = frame["local_origin"]
    local_extent = frame["local_extent"]
    center = [float(local_origin.x), float(local_origin.y), float(local_origin.z)]
    extents = [float(local_extent.x), float(local_extent.y), float(local_extent.z)]
    best_side = None
    trace_count = 0
    for sign in (-1.0, 1.0):
        direction = _scaled(frame["normal"], sign)
        hits = []
        for first_fraction in fractions:
            for second_fraction in fractions:
                local = list(center)
                local[frame["thin_axis"]] += sign * extents[frame["thin_axis"]]
                local[frame["vertical_axis"]] += (
                    first_fraction * extents[frame["vertical_axis"]]
                )
                local[frame["horizontal_axis"]] += (
                    second_fraction * extents[frame["horizontal_axis"]]
                )
                face = frame["transform"].transform_location(unreal.Vector(*local))
                face_point = _vector_tuple(face)
                start_point = _added(face_point, _scaled(direction, -0.5))
                end_point = _added(face_point, _scaled(direction, maximum_gap + 0.5))
                trace_count += 1
                hit = _first_lateral_support_hit(
                    world,
                    actor,
                    unreal.Vector(*start_point),
                    unreal.Vector(*end_point),
                    face_point,
                    direction,
                    target_paths,
                    frame,
                    maximum_gap,
                    solid_actor_cache,
                )
                if hit is not None:
                    hits.append(hit)
        fraction = float(len(hits)) / float(LATERAL_GRID_DIMENSION ** 2)
        side = {"sign": sign, "support_fraction": fraction, "hits": hits}
        if best_side is None or fraction > best_side["support_fraction"]:
            best_side = side
    hits = best_side["hits"] if best_side is not None else []
    collider_counts = {}
    for hit in hits:
        key = (hit.get("collider_path"), hit.get("collider_label"))
        collider_counts.setdefault(key, {"count": 0, "distances": [],
                                         "area_ratios": []})
        collider_counts[key]["count"] += 1
        collider_counts[key]["distances"].append(hit["distance_cm"])
        collider_counts[key]["area_ratios"].append(
            hit["collider_face_area_ratio"]
        )
    support_fraction = best_side["support_fraction"] if best_side else 0.0
    detected = support_fraction >= minimum_fraction
    distances = [hit["distance_cm"] for hit in hits]
    base.update({
        "lateral_support_detected": detected,
        "lateral_support_distance_cm": min(distances) if distances else None,
        "lateral_support_fraction": support_fraction,
        "lateral_trace_count": trace_count,
        "lateral_supporting_colliders": [
            {
                "collider_path": path,
                "collider_label": label,
                "supporting_sample_count": values["count"],
                "supporting_sample_fraction": (
                    float(values["count"]) / float(LATERAL_GRID_DIMENSION ** 2)
                ),
                "minimum_distance_cm": min(values["distances"]),
                "minimum_collider_face_area_ratio": min(values["area_ratios"]),
            }
            for (path, label), values in sorted(
                collider_counts.items(), key=lambda item: str(item[0])
            )
        ],
        "lateral_panel_component": {
            "component_name": frame["component_name"],
            "component_path": frame["component_path"],
            "thin_axis": frame["thin_axis"],
            "face_area_cm2": frame["face_area_cm2"],
            "physical_half_extents_cm": frame["physical_half_extents_cm"],
            "normal": list(frame["normal"]),
        },
        "lateral_support_policy": {
            "policy_id": "bounded-panel-lateral-support-v1",
            "grid_dimension": LATERAL_GRID_DIMENSION,
            "maximum_trace_count": 2 * LATERAL_GRID_DIMENSION ** 2,
            "maximum_gap_cm": maximum_gap,
            "minimum_support_fraction": minimum_fraction,
            "minimum_collider_face_area_ratio": (
                LATERAL_MIN_COLLIDER_FACE_AREA_RATIO
            ),
        },
    })
    return base


def _attach_support_measurement(measurement, actor, world, options,
                                target_paths, solid_actor_cache):
    if options.get("support_model") != LATERAL_SUPPORT_MODEL:
        return measurement
    grounded = measurement.get("grounded") is True
    if grounded:
        measurement.update({
            "supported": True,
            "support_mode": "ground",
            "lateral_support_detected": False,
            "lateral_support_distance_cm": None,
            "lateral_support_fraction": 0.0,
            "lateral_trace_count": 0,
            "lateral_supporting_colliders": [],
            "lateral_support_skip_reason": "ground_contact_already_detected",
        })
        return measurement
    lateral = _measure_lateral_support(
        actor, world, options, set(target_paths or ()), solid_actor_cache
    )
    measurement.update(lateral)
    measurement["supported"] = lateral["lateral_support_detected"] is True
    measurement["support_mode"] = (
        "lateral" if measurement["supported"] else "none"
    )
    return measurement


def _measure_actor_mesh(actor, world, options, origin, extent,
                        excluded_support_paths=None):
    maximum_samples = max(9, int(options.get("maximum_mesh_vertex_samples", 4096)))
    world_vertices, mesh_diagnostics = _mesh_world_vertices(actor, maximum_samples)
    if not world_vertices:
        return None, mesh_diagnostics
    bottom_z = float(origin.z - extent.z)
    top_z = float(origin.z + extent.z)
    probe_below = float(options.get("probe_below_cm", 5000.0))
    support_tolerance = float(options.get("support_tolerance_cm", 5.0))
    surface_cache = {}
    gaps = []
    missing_surface_count = 0
    collider_counts = {}
    supporting_collider_counts = {}
    examples = []
    support_points = []
    for vertex in world_vertices:
        # Meshes commonly duplicate vertex positions for normals/UV seams. A
        # 0.1 cm cache preserves the support weighting while avoiding repeated
        # collision traces for an identical world-space XY sample.
        cache_key = (round(float(vertex.x), 1), round(float(vertex.y), 1))
        if cache_key not in surface_cache:
            start = unreal.Vector(float(vertex.x), float(vertex.y), top_z + 2.0)
            end = unreal.Vector(float(vertex.x), float(vertex.y), bottom_z - probe_below)
            surface_cache[cache_key] = _first_solid_hit(
                world,
                actor,
                start,
                end,
                top_z,
                excluded_support_paths,
            )
        hit = surface_cache[cache_key]
        if hit is None:
            missing_surface_count += 1
            continue
        gap = float(vertex.z) - float(hit["z_cm"])
        gaps.append(gap)
        if abs(gap) <= support_tolerance:
            support_points.append((float(vertex.x), float(vertex.y)))
            support_key = (hit.get("collider_path"), hit.get("collider_label"))
            supporting_collider_counts[support_key] = (
                supporting_collider_counts.get(support_key, 0) + 1
            )
        collider = hit.get("collider_label") or "unknown"
        collider_counts[collider] = collider_counts.get(collider, 0) + 1
        if len(examples) < 12 or gap < min(item["surface_gap_cm"] for item in examples):
            examples.append({
                "vertex_world_cm": [float(vertex.x), float(vertex.y), float(vertex.z)],
                "surface_z_cm": float(hit["z_cm"]),
                "surface_gap_cm": gap,
                "collider_label": hit.get("collider_label"),
                "collider_path": hit.get("collider_path"),
            })
            examples = sorted(examples, key=lambda item: item["surface_gap_cm"])[:12]
    if not gaps:
        return {
            "measurement_method": "ue_lod0_vertex_terrain_trace",
            "grounded": False,
            "surface_detected": False,
            "ground_contact_detected": False,
            "ground_gap_cm": None,
            "penetration_cm": None,
            "solid_penetration_cm": None,
            "support_fraction": 0.0,
            "sample_count": len(world_vertices),
            "surface_comparison_count": 0,
            "missing_surface_count": missing_surface_count,
            "mesh_components": mesh_diagnostics,
            "samples": [],
        }, mesh_diagnostics
    minimum_gap = min(gaps)
    support_count = sum(1 for gap in gaps if abs(gap) <= support_tolerance)
    support_hull_area = _convex_hull_area(support_points)
    mesh_footprint_area = _convex_hull_area([
        (float(vertex.x), float(vertex.y)) for vertex in world_vertices
    ])
    aabb_footprint_area = max(0.0, float(extent.x * 2.0) * float(extent.y * 2.0))
    footprint_area = mesh_footprint_area if mesh_footprint_area > 0.0 else aabb_footprint_area
    support_fraction = min(1.0, support_hull_area / footprint_area) if footprint_area > 0.0 else 0.0
    ground_contact_detected = support_fraction > 0.0
    return {
        "measurement_method": "ue_lod0_vertex_terrain_trace",
        "grounded": ground_contact_detected,
        "surface_detected": True,
        "ground_contact_detected": ground_contact_detected,
        "ground_gap_cm": max(0.0, minimum_gap),
        "penetration_cm": max(0.0, -minimum_gap),
        "solid_penetration_cm": max(0.0, -minimum_gap),
        "support_fraction": support_fraction,
        "support_fraction_method": "contact_xy_convex_hull_over_lod0_xy_convex_hull",
        "raw_support_sample_fraction": float(support_count) / float(len(gaps)),
        "support_hull_area_cm2": support_hull_area,
        "actor_footprint_area_cm2": footprint_area,
        "actor_aabb_footprint_area_cm2": aabb_footprint_area,
        "sample_count": len(world_vertices),
        "surface_comparison_count": len(gaps),
        "supporting_sample_count": support_count,
        "missing_surface_count": missing_surface_count,
        "surface_gap_statistics_cm": {
            "minimum": float(minimum_gap),
            "p50": _percentile(gaps, 0.50),
            "p95": _percentile(gaps, 0.95),
            "maximum": float(max(gaps)),
        },
        "collider_counts": collider_counts,
        "supporting_colliders": [
            {
                "collider_path": collider_path,
                "collider_label": collider_label,
                "supporting_sample_count": count,
                "supporting_sample_fraction": float(count) / float(len(gaps)),
            }
            for (collider_path, collider_label), count
            in supporting_collider_counts.items()
        ],
        "mesh_components": mesh_diagnostics,
        "samples": examples,
    }, mesh_diagnostics


def _outside_bounds(bounds_min, bounds_max, allowed):
    if not isinstance(allowed, dict):
        return None
    allowed_min = allowed.get("min_cm")
    allowed_max = allowed.get("max_cm")
    if not isinstance(allowed_min, list) or not isinstance(allowed_max, list):
        return None
    if len(allowed_min) != 3 or len(allowed_max) != 3:
        return None
    return any(
        bounds_min[axis] < float(allowed_min[axis])
        or bounds_max[axis] > float(allowed_max[axis])
        for axis in range(3)
    )


def _measure_actor(actor, world, options, all_actors=None,
                   solid_actor_cache=None, measurement_target_paths=None):
    all_actors = list(all_actors or [])
    measurement_target_paths = set(measurement_target_paths or ())
    excluded_support_paths = (
        measurement_target_paths
        if options.get("support_model") == LATERAL_SUPPORT_MODEL
        else set()
    )
    origin, extent = actor.get_actor_bounds(False)
    bottom_z = float(origin.z - extent.z)
    top_z = float(origin.z + extent.z)
    mesh_measurement, mesh_diagnostics = _measure_actor_mesh(
        actor,
        world,
        options,
        origin,
        extent,
        excluded_support_paths,
    )
    if mesh_measurement is not None:
        bounds_min = [float(origin.x - extent.x), float(origin.y - extent.y), bottom_z]
        bounds_max = [float(origin.x + extent.x), float(origin.y + extent.y), top_z]
        mesh_measurement.update({
            "actor_path": _actor_path(actor),
            "actor_label": _actor_label(actor),
            "actor_class": _actor_class_path(actor),
            "actor_tags": _actor_tags(actor),
            "stable_actor_id": _stable_actor_id(actor),
            "out_of_bounds": _outside_bounds(bounds_min, bounds_max, options.get("allowed_bounds_cm")),
            "bounds_min_cm": bounds_min,
            "bounds_max_cm": bounds_max,
        })
        mesh_measurement = _attach_support_measurement(
            mesh_measurement,
            actor,
            world,
            options,
            measurement_target_paths,
            solid_actor_cache,
        )
        return _attach_solid_overlap_measurement(
            mesh_measurement, actor, all_actors, options, solid_actor_cache
        )
    probe_below = float(options.get("probe_below_cm", 5000.0))
    support_tolerance = float(options.get("support_tolerance_cm", 5.0))
    samples = []
    signed_gaps = []
    initial_overlap_count = 0
    for x_fraction, y_fraction in _sample_offsets():
        x = float(origin.x + extent.x * x_fraction)
        y = float(origin.y + extent.y * y_fraction)
        start = unreal.Vector(x, y, top_z + 2.0)
        end = unreal.Vector(x, y, bottom_z - probe_below)
        hit = _first_solid_hit(
            world, actor, start, end, top_z, excluded_support_paths
        )
        if hit is None:
            samples.append({"xy_cm": [x, y], "status": "no_solid_hit"})
            continue
        if hit["initial_overlap"]:
            initial_overlap_count += 1
            signed_gap = -max(top_z - bottom_z, support_tolerance + 1.0)
        else:
            signed_gap = bottom_z - hit["z_cm"]
        signed_gaps.append(signed_gap)
        samples.append({
            "xy_cm": [x, y],
            "status": "solid_hit",
            "signed_ground_gap_cm": signed_gap,
            "surface_z_cm": hit["z_cm"],
            "initial_overlap": hit["initial_overlap"],
            "collider_label": hit["collider_label"],
            "collider_path": hit["collider_path"],
        })

    support_count = sum(1 for value in signed_gaps if abs(value) <= support_tolerance)
    support_fraction = float(support_count) / float(len(_sample_offsets()))
    if signed_gaps:
        nearest_gap = min(signed_gaps, key=lambda value: abs(value))
        ground_gap = max(0.0, nearest_gap)
        penetration = max(0.0, -min(signed_gaps))
    else:
        ground_gap = None
        penetration = None
    bounds_min = [float(origin.x - extent.x), float(origin.y - extent.y), bottom_z]
    bounds_max = [float(origin.x + extent.x), float(origin.y + extent.y), top_z]
    measurement = {
        "actor_path": _actor_path(actor),
        "actor_label": _actor_label(actor),
        "actor_class": _actor_class_path(actor),
        "actor_tags": _actor_tags(actor),
        "stable_actor_id": _stable_actor_id(actor),
        "measurement_method": "ue_vertical_collision_trace_3x3",
        "measurement_fallback_reason": "lod0_mesh_vertices_unavailable",
        "mesh_components": mesh_diagnostics,
        "grounded": support_count > 0,
        "surface_detected": bool(signed_gaps),
        "ground_contact_detected": support_count > 0,
        "ground_gap_cm": ground_gap,
        "penetration_cm": penetration,
        "solid_penetration_cm": penetration,
        "support_fraction": support_fraction,
        "raw_support_sample_fraction": (
            float(support_count) / float(len(signed_gaps)) if signed_gaps else 0.0
        ),
        "supporting_colliders": [
            {
                "collider_path": sample.get("collider_path"),
                "collider_label": sample.get("collider_label"),
                "supporting_sample_count": sum(
                    1 for other in samples
                    if other.get("collider_path") == sample.get("collider_path")
                    and other.get("collider_label") == sample.get("collider_label")
                    and isinstance(other.get("signed_ground_gap_cm"), (int, float))
                    and abs(float(other["signed_ground_gap_cm"])) <= support_tolerance
                ),
            }
            for index, sample in enumerate(samples)
            if isinstance(sample.get("signed_ground_gap_cm"), (int, float))
            and abs(float(sample["signed_ground_gap_cm"])) <= support_tolerance
            and not any(
                earlier.get("collider_path") == sample.get("collider_path")
                and earlier.get("collider_label") == sample.get("collider_label")
                for earlier in samples[:index]
            )
        ],
        "out_of_bounds": _outside_bounds(bounds_min, bounds_max, options.get("allowed_bounds_cm")),
        "sample_count": len(_sample_offsets()),
        "solid_hit_count": len(signed_gaps),
        "initial_overlap_count": initial_overlap_count,
        "bounds_min_cm": bounds_min,
        "bounds_max_cm": bounds_max,
        "samples": samples,
    }
    measurement = _attach_support_measurement(
        measurement,
        actor,
        world,
        options,
        measurement_target_paths,
        solid_actor_cache,
    )
    return _attach_solid_overlap_measurement(
        measurement, actor, all_actors, options, solid_actor_cache
    )


def _resolve_targets(all_actors, target_references):
    by_path = {}
    by_stable_id = {}
    by_label = {}
    for actor in all_actors:
        path = _actor_path(actor)
        stable_id = _stable_actor_id(actor)
        label = _actor_label(actor)
        if path:
            by_path[path] = actor
        if stable_id:
            by_stable_id.setdefault(stable_id, []).append(actor)
        by_label.setdefault(label, []).append(actor)
    resolved = []
    unresolved = []
    for reference in target_references:
        actor = None
        actor_path = reference.get("actor_path")
        stable_id = reference.get("stable_actor_id")
        label = reference.get("label")
        if actor_path:
            actor = by_path.get(actor_path)
        if actor is None and stable_id and len(by_stable_id.get(stable_id, [])) == 1:
            actor = by_stable_id[stable_id][0]
        if actor is None and label and len(by_label.get(label, [])) == 1:
            actor = by_label[label][0]
        if actor is None:
            unresolved.append(reference)
        else:
            resolved.append((reference, actor))
    return resolved, unresolved


def _write_payload(payload, output_path):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, output_path)


def _injected_json(name, legacy_name, default):
    encoded = globals().get(name)
    if isinstance(encoded, str):
        return json.loads(encoded)
    return globals().get(legacy_name, default)


def _minimum_surface_gap(measurement):
    statistics = measurement.get("surface_gap_statistics_cm") or {}
    value = statistics.get("minimum")
    if isinstance(value, (int, float)):
        return float(value)
    gap = measurement.get("ground_gap_cm")
    penetration = measurement.get("penetration_cm")
    if isinstance(gap, (int, float)) and isinstance(penetration, (int, float)):
        return float(gap) - float(penetration)
    return None


def _run_calibration_sweeps(all_actors, world, options,
                            solid_actor_cache=None):
    """Temporarily offset Actors in Z, measure them, and restore every transform."""
    sweep_specs = options.get("calibration_sweeps", [])
    if not isinstance(sweep_specs, list) or not sweep_specs:
        return []
    results = []
    for specification in sweep_specs:
        if not isinstance(specification, dict):
            results.append({"status": "invalid_spec", "specification": specification})
            continue
        reference = {
            "label": specification.get("label"),
            "stable_actor_id": specification.get("stable_actor_id"),
            "actor_path": specification.get("actor_path"),
        }
        resolved, unresolved = _resolve_targets(all_actors, [reference])
        if unresolved or not resolved:
            results.append({"status": "unresolved", "reference": reference})
            continue
        actor = resolved[0][1]
        offsets = specification.get("z_offsets_cm", [])
        if not isinstance(offsets, list):
            results.append({"status": "invalid_offsets", "reference": reference})
            continue
        original = actor.get_actor_location()
        original_xyz = (float(original.x), float(original.y), float(original.z))
        samples = []
        try:
            for raw_offset in offsets:
                offset = float(raw_offset)
                location = unreal.Vector(original_xyz[0], original_xyz[1], original_xyz[2] + offset)
                actor.set_actor_location(location, False, False)
                measurement = _measure_actor(
                    actor, world, options, all_actors, solid_actor_cache
                )
                samples.append({
                    "requested_z_offset_cm": offset,
                    "observed_actor_z_cm": float(actor.get_actor_location().z),
                    "minimum_surface_gap_cm": _minimum_surface_gap(measurement),
                    "ground_gap_cm": measurement.get("ground_gap_cm"),
                    "penetration_cm": measurement.get("penetration_cm"),
                    "support_fraction": measurement.get("support_fraction"),
                    "measurement_method": measurement.get("measurement_method"),
                })
        except Exception as error:
            results.append({
                "status": "error",
                "reference": reference,
                "error": str(error),
                "samples": samples,
            })
            continue
        finally:
            actor.set_actor_location(unreal.Vector(*original_xyz), False, False)
        zero_sample = next(
            (sample for sample in samples if abs(sample["requested_z_offset_cm"]) <= 1.0e-6),
            None,
        )
        zero_gap = zero_sample.get("minimum_surface_gap_cm") if zero_sample else None
        if isinstance(zero_gap, (int, float)):
            for sample in samples:
                observed = sample.get("minimum_surface_gap_cm")
                if isinstance(observed, (int, float)):
                    sample["observed_delta_from_zero_cm"] = float(observed) - float(zero_gap)
                    sample["delta_error_cm"] = (
                        sample["observed_delta_from_zero_cm"] - sample["requested_z_offset_cm"]
                    )
        results.append({
            "status": "success",
            "reference": reference,
            "original_location_cm": list(original_xyz),
            "restored_location_cm": [
                float(actor.get_actor_location().x),
                float(actor.get_actor_location().y),
                float(actor.get_actor_location().z),
            ],
            "samples": samples,
        })
    return results


def main():
    measurement_started_at = time.perf_counter()
    output_path = globals().get("SCENE_PHYSICS_OUTPUT", "")
    target_references = _injected_json(
        "SCENE_PHYSICS_TARGETS_JSON", "SCENE_PHYSICS_TARGETS", []
    )
    options = _injected_json("SCENE_PHYSICS_OPTIONS_JSON", "SCENE_PHYSICS_OPTIONS", {})
    if not output_path:
        raise RuntimeError("SCENE_PHYSICS_OUTPUT is required")
    try:
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        all_actors = list(subsystem.get_all_level_actors())
        resolved, unresolved = _resolve_targets(all_actors, target_references)
        measurement_target_paths = {
            _actor_path(actor)
            for _reference, actor in resolved
            if _actor_path(actor)
        }
        map_path = _current_map_package()
        cache_key = (
            map_path,
            len(all_actors),
            float(options.get("solid_broad_phase_grid_cell_cm",
                              SOLID_GRID_CELL_SIZE_CM)),
            int(options.get("solid_broad_phase_max_cells_per_actor",
                            SOLID_GRID_MAX_CELLS_PER_ACTOR)),
        )
        cached = globals().get("_SB_SCENE_PHYSICS_SOLID_CACHE")
        cache_reused = bool(
            isinstance(cached, dict)
            and cached.get("key") == cache_key
            and cached.get("world") == world
            and isinstance(cached.get("value"), dict)
        )
        if cache_reused:
            solid_actor_cache = cached["value"]
        else:
            solid_actor_cache = _build_solid_actor_cache(all_actors, options)
            globals()["_SB_SCENE_PHYSICS_SOLID_CACHE"] = {
                "key": cache_key,
                "world": world,
                "value": solid_actor_cache,
            }
        measurements = {}
        errors = []
        for reference, actor in resolved:
            # Key by an identity that is UNIQUE. A label is a DISPLAY name and
            # UE lets any number of Actors carry the same one, so keying this
            # dict by it kept only the last of each. A scene whose ground was
            # tiled by eight Actors all called `Gnd_00` lost 56 of its 250
            # measurements that way — and `status` still said success, because
            # nothing was unresolved and nothing raised. The Actor's own path
            # is unique within a level, is what the record already carries,
            # and is what the matcher looks for before it ever tries a label.
            key = (_actor_path(actor)
                   or reference.get("stable_actor_id")
                   or reference.get("actor_path")
                   or reference.get("label"))
            label = reference.get("label") or key
            if not key:
                errors.append({"actor": None, "error":
                               "target has no identity to key a measurement by"})
                continue
            if key in measurements:
                # Two targets naming one Actor. Reported rather than silently
                # overwritten: a measurement that disappears without a trace is
                # exactly what the count envelope exists to make impossible.
                errors.append({"actor": label, "error":
                               "duplicate target identity " + str(key)})
                continue
            try:
                if options.get("solid_penetration_only") is True:
                    measurements[key] = _measure_actor_solid_only(
                        actor, options, all_actors, solid_actor_cache
                    )
                else:
                    measurements[key] = _measure_actor(
                        actor,
                        world,
                        options,
                        all_actors,
                        solid_actor_cache,
                        measurement_target_paths,
                    )
            except Exception as error:
                errors.append({"actor": label, "error": str(error)})
        calibration_sweeps = (
            [] if options.get("solid_penetration_only") is True
            else _run_calibration_sweeps(
                all_actors, world, options, solid_actor_cache
            )
        )
        payload = {
            "schema_version": "0.5.0",
            "measurement_type": "ue_editor_actor_physics",
            "map_path": map_path,
            "capabilities": {
                "support": {
                    "support_model": "ground_or_lateral_v1",
                    "ground_method": "ue_vertical_collision_trace",
                    "lateral_method": "bounded_panel_lateral_support_v1",
                    "maximum_lateral_trace_count_per_actor": (
                        2 * LATERAL_GRID_DIMENSION ** 2
                    ),
                    "target_to_target_support": "excluded",
                },
                "solid_penetration": {
                    "broad_phase": "actor_world_aabb",
                    "broad_phase_acceleration": "uniform_grid_exact",
                    "overlap_method": "ue_component_overlap_components",
                    "depth_method": "ue_fhitresult_initial_overlap_mtd",
                    "collision_response_filter": (
                        "pairwise_effective_collision_response_v1"
                    ),
                    "depth_probe_strategy": (
                        "reversible_multivector_bidirectional_component_sweep_v1"
                    ),
                    "aabb_role": "broad_phase_only",
                },
            },
            "runtime_provenance": {
                "project_file_path": os.path.realpath(unreal.Paths.convert_relative_path_to_full(
                    str(unreal.Paths.get_project_file_path())
                )),
                "engine_version": str(unreal.SystemLibrary.get_engine_version()),
                "map_path": map_path,
            },
            "actors": measurements,
            "diagnostics": {
                "status": "success" if not unresolved and not errors else "partial",
                "requested_actor_count": len(target_references),
                "measured_actor_count": len(measurements),
                "unresolved_targets": unresolved,
                "measurement_errors": errors,
                "calibration_sweeps": calibration_sweeps,
                "options": options,
                "solid_actor_cache": {
                    "semantic_solid_actor_count": len(solid_actor_cache["entries"]),
                    "semantic_non_solid_count": solid_actor_cache["filtered"]["semantic_non_solid"],
                    "collision_component_checked_actor_count": solid_actor_cache["component_resolution"]["checked_actor_count"],
                    "collision_disabled_count": solid_actor_cache["component_resolution"]["collision_disabled_actor_count"],
                    "grid_cell_size_cm": solid_actor_cache["grid_cell_size_cm"],
                    "grid_cell_count": len(solid_actor_cache["grid"]),
                    "overflow_actor_count": len(solid_actor_cache["overflow_entries"]),
                    "reused_from_previous_chunk": cache_reused,
                },
            },
        }
    except Exception as error:
        map_path = _current_map_package()
        payload = {
            "schema_version": "0.4.1",
            "measurement_type": "ue_editor_actor_physics",
            "map_path": map_path,
            "capabilities": {
                "solid_penetration": {
                    "broad_phase": "actor_world_aabb",
                    "broad_phase_acceleration": "uniform_grid_exact",
                    "overlap_method": "ue_component_overlap_components",
                    "depth_method": "ue_fhitresult_initial_overlap_mtd",
                    "collision_response_filter": (
                        "pairwise_effective_collision_response_v1"
                    ),
                    "depth_probe_strategy": (
                        "reversible_multivector_bidirectional_component_sweep_v1"
                    ),
                    "aabb_role": "broad_phase_only",
                },
            },
            "runtime_provenance": {
                "project_file_path": os.path.realpath(unreal.Paths.convert_relative_path_to_full(
                    str(unreal.Paths.get_project_file_path())
                )),
                "engine_version": str(unreal.SystemLibrary.get_engine_version()),
                "map_path": map_path,
            },
            "actors": {},
            "diagnostics": {
                "status": "error",
                "requested_actor_count": len(target_references),
                "measured_actor_count": 0,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    payload["diagnostics"]["measurement_elapsed_s"] = round(
        time.perf_counter() - measurement_started_at, 6
    )
    _write_payload(payload, output_path)
    print("SCENE_PHYSICS_MEASUREMENTS status={} path={}".format(payload["diagnostics"]["status"], output_path))


main()
