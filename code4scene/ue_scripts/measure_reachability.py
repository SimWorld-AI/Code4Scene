"""Measure which placed Actors an embodied agent can walk up to, in the editor.

The host adapter injects SCENE_REACHABILITY_OUTPUT plus a JSON-encoded option
value before executing this file in Unreal Python. Like the physics probe, this
file produces EVIDENCE and no verdict: the score lives in
``evaluation/verifiers/reachability.py``, which never opens an editor.

The order matters and is the whole method:

1. set the agent radius on the RecastNavMesh actor, so "reachable" means
   reachable by something the size of a person rather than by a point;
2. rebuild, and then *prove the rebuild settled*. Two paths, and the payload
   records which one ran:

   * ``witnessed_drain`` (the default): read the remaining build-task counter
     AFTER asking for the rebuild and BEFORE waiting at all. That reading must
     be above zero — it is the proof a rebuild actually started, and without it
     there is nothing to watch drain. Then wait for the counter to reach zero
     and read zero twice, because zero is readable in the moment a tile is
     dirtied but before its task is queued. Only then sample twice and require
     the two samples to agree: the equality gate stays, because a drained queue
     is not the same claim as a finished build — the counter can read zero
     while the navigation system is still building;
   * ``conservative``: the resample-equality test on its own, on a wall clock.
     This is what runs when the witness never fires (a synchronous rebuild, or
     a counter this engine build will not answer), and it is the path to ask
     for by hand when a component structure is going to be quoted;

   a settle that does not converge returns ``settled: false``, which the scorer
   turns into a withheld score. A too-short wait therefore costs a withheld
   cell, never a wrong number, and that asymmetry is what makes the fast path
   safe to default;
3. sample K seeded points, and group them into path-connected components with
   navigation queries rather than with geometry — two floors of one building
   overlap in XY and are not one component. The query budget is denominated in
   SAMPLE BATCHES — one per sample — and never in pairs; see ``_components``;
4. project each collidable Actor's footprint onto the navmesh and ask whether a
   path exists from there to the largest component.

Actors with no collision are not walked up to and never were: a
BoxReflectionCapture sits in the middle of a room, projects nothing, and scored
the room unreachable until it was excluded here rather than in the scorer.

The fast settle is certified for the score and NOT for component structure: on
the certification scenes the fast passes ran 2.75x the live spread of the
largest-component share while the approachability share held to a stated
<= 0.01 downward bias. ``component_spread_passes`` re-settles and re-groups N
times so a component structure quoted off a fast pass carries a spread instead
of being a bare point estimate; the scorer withholds that structure when
neither that nor the conservative settle is present.
"""

import json
import os
import random
import re
import time
import traceback
import unreal


#: 0.1.1 adds navmesh.settle.path, components.cost and components.spread. The
#: scorer reads 0.1.x, so an older payload still scores — it just cannot say
#: which settle produced it, and is read as the conservative one it was.
SCHEMA_VERSION = "0.1.1"

#: Classes whose Actors are not obstacles and not destinations. Same lesson,
#: same list shape as measure_scene_physics.py's non-solid filter.
NO_COLLISION_CLASS_SUFFIX = (
    "light", "camera", "fog", "atmosphere", "volume", "reflectioncapture",
    "groupactor", "emitter", "niagaraactor", "postprocessvolume",
    "navmeshboundsvolume", "recastnavmesh", "worldsettings", "brush",
    "lightmassimportancevolume", "skyatmosphere", "playerstart",
)
NO_COLLISION_SEMANTIC_TOKEN = (
    "navmesh", "reflection", "postprocess", "atmosphere", "sky", "particles",
    "niagara", "emitter", "vfx",
)


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


def _class_name(actor):
    try:
        return str(actor.get_class().get_name())
    except Exception:
        return ""


def _collision_enabled(component):
    """Whether a PrimitiveComponent participates in collision queries."""
    try:
        value = component.get_collision_enabled()
    except Exception:
        return False
    text = str(value).lower().replace(" ", "_")
    return "no_collision" not in text and not text.endswith(".none")


def _collidable_component_count(actor):
    try:
        components = list(actor.get_components_by_class(unreal.PrimitiveComponent))
    except Exception:
        return 0
    return sum(1 for component in components if _collision_enabled(component))


def _no_collision_class(actor):
    class_name = _class_name(actor).lower()
    if class_name.endswith(NO_COLLISION_CLASS_SUFFIX):
        return True
    text = " ".join(str(value).lower() for value in
                    (_actor_label(actor), _actor_path(actor), class_name) if value)
    tokens = set(re.findall(r"[a-z0-9]+", text.replace("_", " ")))
    return any(token in tokens for token in NO_COLLISION_SEMANTIC_TOKEN)


def _eligibility(actor):
    """Why this Actor is not a destination, or None when it is one."""
    if _no_collision_class(actor):
        return "no_collision_class"
    if _collidable_component_count(actor) == 0:
        return "collision_disabled"
    return None


# ── the navmesh actor: set the radius, rebuild, read the result back ──────

RECAST_PARAMETER_NAMES = (
    "agent_radius", "agent_height", "agent_max_slope", "agent_max_step_height",
    "cell_size", "cell_height", "tile_size_uu", "runtime_generation",
)


def _navmesh_actor(all_actors):
    recast = getattr(unreal, "RecastNavMesh", None)
    for actor in all_actors:
        if recast is not None:
            try:
                if isinstance(actor, recast):
                    return actor
            except Exception:
                pass
        if "recastnavmesh" in _class_name(actor).lower():
            return actor
    return None


def _set_agent_radius(actor, radius_cm):
    """Set AgentRadius wherever this engine build keeps it; say which worked."""
    applied = []
    errors = []
    try:
        actor.set_editor_property("agent_radius", float(radius_cm))
        applied.append("agent_radius")
    except Exception as error:
        errors.append("agent_radius: %s" % error)
    try:
        config = actor.get_editor_property("nav_data_config")
        config.set_editor_property("agent_radius", float(radius_cm))
        actor.set_editor_property("nav_data_config", config)
        applied.append("nav_data_config.agent_radius")
    except Exception as error:
        errors.append("nav_data_config.agent_radius: %s" % error)
    return applied, errors


def _navmesh_parameters(actor):
    """What the actor says about itself AFTER the rebuild — proof of build."""
    parameters = {}
    for name in RECAST_PARAMETER_NAMES:
        try:
            value = actor.get_editor_property(name)
        except Exception:
            continue
        parameters[name] = value if isinstance(value, (int, float, bool)) else str(value)
    try:
        config = actor.get_editor_property("nav_data_config")
        parameters["nav_data_config.agent_radius"] = float(
            config.get_editor_property("agent_radius"))
        parameters["nav_data_config.agent_height"] = float(
            config.get_editor_property("agent_height"))
    except Exception:
        pass
    return parameters


def _navigation_system(world):
    for call in (
        lambda: unreal.NavigationSystemV1.get_navigation_system(world),
        lambda: unreal.NavigationSystemV1.get_navigation_system_v1(world),
    ):
        try:
            system = call()
            if system is not None:
                return system
        except Exception:
            continue
    return None


def _rebuild_navigation(nav_system):
    attempts = []
    for name, call in (
        ("NavigationSystemV1.build_navigation",
         lambda: unreal.NavigationSystemV1.build_navigation()),
        ("nav_system.build", lambda: nav_system.build() if nav_system else None),
    ):
        try:
            call()
            attempts.append({"method": name, "status": "called"})
            return attempts
        except Exception as error:
            attempts.append({"method": name, "status": "unavailable", "error": str(error)})
    return attempts


def _remaining_build_tasks(nav_system):
    """Recorded, never trusted: it reads zero between dirty-area passes."""
    try:
        return int(nav_system.get_num_remaining_build_tasks())
    except Exception:
        return None


# ── sampling and connectivity ─────────────────────────────────────────────

def _vector(point):
    return [float(point.x), float(point.y), float(point.z)]


def _project(world, point, extent_cm):
    extent = unreal.Vector(float(extent_cm[0]), float(extent_cm[1]), float(extent_cm[2]))
    try:
        projected = unreal.NavigationSystemV1.project_point_to_navigation(
            world, point, None, None, extent)
    except Exception:
        return None
    if projected is None:
        return None
    # A failed projection comes back as the zero vector on some builds; a real
    # hit at the world origin is indistinguishable, so require it to be within
    # the query extent of what was asked for.
    for axis, allowed in zip(("x", "y", "z"), extent_cm):
        if abs(float(getattr(projected, axis)) - float(getattr(point, axis))) > float(allowed) + 1.0:
            return None
    return projected


def _sample_points(world, seed, count, bounds_min, bounds_max, extent_cm):
    """Seeded uniform XY over the navmesh bounds, projected down onto it."""
    generator = random.Random(seed)
    points = []
    attempts = 0
    maximum_attempts = int(count) * 25
    z = (float(bounds_min[2]) + float(bounds_max[2])) * 0.5
    while len(points) < int(count) and attempts < maximum_attempts:
        attempts += 1
        candidate = unreal.Vector(
            generator.uniform(float(bounds_min[0]), float(bounds_max[0])),
            generator.uniform(float(bounds_min[1]), float(bounds_max[1])),
            z,
        )
        projected = _project(world, candidate, extent_cm)
        if projected is not None:
            points.append(projected)
    return points, attempts


def _fingerprint(points):
    return sorted(
        (round(float(p.x), 0), round(float(p.y), 0), round(float(p.z), 0))
        for p in points
    )


#: Consecutive zero readings of the build-task counter that count as a drain.
#: One is not enough: zero is readable in the moment a tile is dirtied but
#: before its task is queued.
SETTLE_ZERO_READS = 2


def _settle_test(world, options, bounds_min, bounds_max, extent_cm, remaining_tasks):
    """Sample twice with a wait; equal samples mean the build stopped moving.

    ponytail: the wait is wall-clock, because editor Python owns the game
    thread and there is no tick to await. Raise settle_wait_s on a machine
    where a large level rebuilds slowly; the equality gate is what makes a
    too-short wait a withheld score rather than a wrong one.
    """
    probe_count = int(options.get("settle_probe_count", 32))
    wait_s = float(options.get("settle_wait_s", 5.0))
    attempts = int(options.get("settle_attempts", 3))
    seed = int(options.get("settle_seed", 20260817))
    passes = []
    previous = None
    for attempt in range(max(1, attempts)):
        points, _ = _sample_points(
            world, seed, probe_count, bounds_min, bounds_max, extent_cm)
        current = _fingerprint(points)
        passes.append({"attempt": attempt, "sample_count": len(points),
                       "remaining_build_tasks": remaining_tasks})
        if previous is not None and current == previous and current:
            return {"settled": True, "method": "resample_equality",
                    "probe_count": probe_count, "wait_s": wait_s,
                    "passes": passes,
                    "remaining_build_tasks_reported": remaining_tasks}
        previous = current
        time.sleep(wait_s)
    return {"settled": False, "method": "resample_equality",
            "probe_count": probe_count, "wait_s": wait_s, "passes": passes,
            "remaining_build_tasks_reported": remaining_tasks,
            "reason": "two consecutive navmesh samples never agreed"}


def _drain(nav_system, options, witnessed_tasks):
    """Watch the witnessed build queue empty. Returns (drained, record)."""
    poll_s = float(options.get("settle_poll_s", 0.5))
    timeout_s = float(options.get("fast_settle_timeout_s", 120.0))
    started = time.time()
    readings = [witnessed_tasks]
    zeros = 0
    while zeros < SETTLE_ZERO_READS:
        if time.time() - started > timeout_s:
            break
        time.sleep(poll_s)
        tasks = _remaining_build_tasks(nav_system)
        readings.append(tasks)
        zeros = zeros + 1 if tasks == 0 else 0
    return zeros >= SETTLE_ZERO_READS, {
        "witnessed_build_tasks": witnessed_tasks,
        "witness_note": "read after the rebuild call and before any wait",
        "zero_reads_required": SETTLE_ZERO_READS,
        "poll_s": poll_s,
        "timeout_s": timeout_s,
        "build_task_readings": readings,
        "drain_s": round(time.time() - started, 2),
    }


def _settle(world, nav_system, options, bounds_min, bounds_max, extent_cm,
            witnessed_tasks):
    """Settle the navmesh, on evidence where there is any, on a clock where not.

    Default ON because it keeps the conservative path's equality gate and adds
    the drain to it: strictly more evidence, not less. A missing or zero
    witness means no rebuild was seen to start, and this refuses to guess from
    that — it falls back.
    """
    conservative = not options.get("fast_settle", True)
    if conservative or not witnessed_tasks:
        settle = _settle_test(
            world, options, bounds_min, bounds_max, extent_cm, witnessed_tasks)
        settle["path"] = "conservative"
        settle["fallback_reason"] = None if conservative else (
            "no rebuild witness: the remaining build-task counter read "
            "{} before any wait, so there was no drain to watch".format(
                witnessed_tasks))
        return settle
    drained, witness = _drain(nav_system, options, witnessed_tasks)
    if not drained:
        return {"settled": False, "method": "witnessed_drain_then_resample_equality",
                "path": "witnessed_drain", "witness": witness,
                "remaining_build_tasks_reported": witness["build_task_readings"][-1],
                "reason": "the build queue never drained to zero within the timeout"}
    settle = _settle_test(world, options, bounds_min, bounds_max, extent_cm,
                          witness["build_task_readings"][-1])
    settle["method"] = "witnessed_drain_then_resample_equality"
    settle["path"] = "witnessed_drain"
    settle["witness"] = witness
    return settle


def _path_connected(world, start, end, tolerance_cm):
    """A path query, refusing partial paths — they are how a wall reads as a door."""
    try:
        path = unreal.NavigationSystemV1.find_path_to_location_synchronously(
            world, start, end)
    except Exception:
        return False
    if path is None:
        return False
    try:
        if path.is_partial():
            return False
    except Exception:
        pass
    try:
        if path.is_valid() is False:
            return False
    except Exception:
        pass
    try:
        points = list(path.path_points or [])
    except Exception:
        points = []
    if len(points) < 2:
        return False
    last = points[-1]
    distance = (
        (float(last.x) - float(end.x)) ** 2
        + (float(last.y) - float(end.y)) ** 2
        + (float(last.z) - float(end.z)) ** 2
    ) ** 0.5
    return distance <= float(tolerance_cm)


def _components(world, points, tolerance_cm, budget):
    """Path-connected components, one BATCH of queries per point.

    A batch is this point against every component known so far. It costs as
    many path queries as there are components, and it is the unit the budget is
    denominated in — never the pair, and this is the whole point:

    the batch has no early exit, so a point is tested against every component
    even once one has answered yes, and the pair count runs several times what
    a first-hit-wins loop would spend. Charge those pairs against a
    pair-denominated cap and the loop stops partway through the samples, which
    does not produce a slower answer but a DIFFERENT one — it stops testing
    later points and manufactures components out of the ones it never tested.
    That is measured, not theoretical: the certification harness spent 5576
    pairs where its sequential ancestor spent 2277, hit a 6000-PAIR cap, and
    reported 56 components at a 0.475 largest share against a recorded 49-52 at
    0.495-0.535. Budgeting batches instead restores what the cost guard was
    for, and since the loop spends at most one batch per point, the default
    budget (one per sample) cannot truncate at all. Truncation is structurally
    impossible there, which is the reason to write it this way rather than to
    pick a bigger number.

    ponytail: O(points x components), not O(points^2) — a scene with one
    walkable floor costs 200 queries. A scene that really is 200 islands costs
    the quadratic, and that scene withholds its score anyway.
    """
    groups = []          # each: {"representative": Vector, "members": [index]}
    batches = 0
    queries = 0
    truncated = False
    for index, point in enumerate(points):
        if groups and batches >= budget:
            truncated = True
            groups.append({"representative": point, "members": [index]})
            continue
        joined = []
        for group_index, group in enumerate(groups):
            queries += 1
            if _path_connected(world, point, group["representative"], tolerance_cm):
                joined.append(group_index)
        if groups:
            batches += 1
        if not joined:
            groups.append({"representative": point, "members": [index]})
            continue
        first = groups[joined[0]]
        first["members"].append(index)
        for group_index in reversed(joined[1:]):
            first["members"].extend(groups[group_index]["members"])
            groups.pop(group_index)
    return groups, {"query_batches": batches, "path_queries": queries,
                    "query_budget": budget, "query_budget_unit": "sample_batches",
                    "query_budget_truncated": truncated}


def _actor_footprint_points(actor, offsets):
    origin, extent = actor.get_actor_bounds(False)
    bottom_z = float(origin.z - extent.z)
    return [
        unreal.Vector(float(origin.x + extent.x * x), float(origin.y + extent.y * y),
                      bottom_z)
        for x, y in offsets
    ]


FOOTPRINT_OFFSETS = (
    (0.0, 0.0),
    (-0.9, -0.9), (-0.9, 0.0), (-0.9, 0.9),
    (0.0, -0.9), (0.0, 0.9),
    (0.9, -0.9), (0.9, 0.0), (0.9, 0.9),
)


def _measure_actor(actor, world, options, largest_representative):
    projection_extent = float(options.get("projection_extent_cm", 100.0))
    tolerance = float(options.get("path_endpoint_tolerance_cm", 100.0))
    record = {
        "actor_path": _actor_path(actor),
        "label": _actor_label(actor),
        "class": _class_name(actor),
        "eligible": True,
        "ineligible_reason": None,
        "projected_point_cm": None,
        "projection_offset_index": None,
        "connected": None,
    }
    reason = _eligibility(actor)
    if reason is not None:
        record.update({"eligible": False, "ineligible_reason": reason})
        return record
    for index, point in enumerate(_actor_footprint_points(actor, FOOTPRINT_OFFSETS)):
        projected = _project(
            world, point, (projection_extent, projection_extent, projection_extent))
        if projected is None:
            continue
        record["projected_point_cm"] = _vector(projected)
        record["projection_offset_index"] = index
        record["connected"] = (
            _path_connected(world, projected, largest_representative, tolerance)
            if largest_representative is not None else None
        )
        return record
    return record


def _navmesh_bounds(navmesh_actor, all_actors, options):
    configured = options.get("navmesh_bounds_cm")
    if isinstance(configured, dict):
        minimum = configured.get("min_cm")
        maximum = configured.get("max_cm")
        if isinstance(minimum, list) and isinstance(maximum, list):
            return [float(v) for v in minimum], [float(v) for v in maximum], "options"
    volumes = [actor for actor in all_actors
               if "navmeshboundsvolume" in _class_name(actor).lower()]
    sources = volumes or ([navmesh_actor] if navmesh_actor is not None else [])
    minimum = None
    maximum = None
    for actor in sources:
        try:
            origin, extent = actor.get_actor_bounds(False)
        except Exception:
            continue
        low = [float(origin.x - extent.x), float(origin.y - extent.y),
               float(origin.z - extent.z)]
        high = [float(origin.x + extent.x), float(origin.y + extent.y),
                float(origin.z + extent.z)]
        minimum = low if minimum is None else [min(a, b) for a, b in zip(minimum, low)]
        maximum = high if maximum is None else [max(a, b) for a, b in zip(maximum, high)]
    source = "nav_bounds_volumes" if volumes else "navmesh_actor_bounds"
    return minimum, maximum, (source if minimum is not None else "unavailable")


def _write_payload(payload, output_path):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, output_path)


def _injected_json(name, default):
    encoded = globals().get(name)
    if isinstance(encoded, str):
        return json.loads(encoded)
    return default


def _runtime_provenance(map_path):
    return {
        "project_file_path": os.path.realpath(unreal.Paths.convert_relative_path_to_full(
            str(unreal.Paths.get_project_file_path())
        )),
        "engine_version": str(unreal.SystemLibrary.get_engine_version()),
        "map_path": map_path,
    }


def _envelope(map_path, **rest):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "measurement_type": "ue_editor_reachability",
        "map_path": map_path,
        "runtime_provenance": _runtime_provenance(map_path),
    }
    payload.update(rest)
    return payload


def main():
    output_path = globals().get("SCENE_REACHABILITY_OUTPUT", "")
    options = _injected_json("SCENE_REACHABILITY_OPTIONS_JSON", {})
    if not output_path:
        raise RuntimeError("SCENE_REACHABILITY_OUTPUT is required")
    map_path = None
    try:
        map_path = _current_map_package()
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        all_actors = list(subsystem.get_all_level_actors())
        agent_radius = float(options.get("agent_radius_cm", 50.0))
        sample_count = int(options.get("sample_count", 200))
        sample_seed = int(options.get("sample_seed", 20260817))
        sample_extent = float(options.get("sample_projection_extent_cm", 500.0))
        tolerance = float(options.get("path_endpoint_tolerance_cm", 100.0))

        navmesh_actor = _navmesh_actor(all_actors)
        unresolved = []
        applied = []
        radius_errors = []
        if navmesh_actor is None:
            unresolved.append({"what": "RecastNavMesh actor", "reason": "not_in_level"})
        else:
            applied, radius_errors = _set_agent_radius(navmesh_actor, agent_radius)
        nav_system = _navigation_system(world)
        if nav_system is None:
            unresolved.append({"what": "UNavigationSystemV1", "reason": "unavailable"})
        # One batch per sample, so the default cannot truncate. See _components.
        query_budget = int(options.get("component_query_budget", sample_count))
        spread_passes = max(1, int(options.get("component_spread_passes", 1)))

        rebuild = _rebuild_navigation(nav_system)
        remaining = _remaining_build_tasks(nav_system)

        bounds_min, bounds_max, bounds_source = _navmesh_bounds(
            navmesh_actor, all_actors, options)
        extent = (sample_extent, sample_extent, sample_extent)
        spread = []
        if bounds_min is None:
            settle = {"settled": False, "method": "resample_equality",
                      "path": "conservative",
                      "reason": "no navmesh bounds to sample"}
            samples = []
            attempts = 0
            groups = []
            cost = {}
        else:
            # Each pass rebuilds, settles and regroups from scratch, on the same
            # sample seed: a spread over passes is then the rebuild's own
            # variance rather than the draw's. The last pass is the one reported
            # and the one the Actors are measured against.
            for index in range(spread_passes):
                if index:
                    rebuild = _rebuild_navigation(nav_system)
                    remaining = _remaining_build_tasks(nav_system)
                settle = _settle(world, nav_system, options, bounds_min, bounds_max,
                                 extent, remaining)
                samples, attempts = _sample_points(
                    world, sample_seed, sample_count, bounds_min, bounds_max, extent)
                groups, cost = (_components(world, samples, tolerance, query_budget)
                                if samples else ([], {}))
                groups = sorted(groups, key=lambda group: len(group["members"]),
                                reverse=True)
                spread.append({
                    "settled": settle.get("settled") is True,
                    "settle_path": settle.get("path"),
                    "count": len(groups),
                    "largest_size": len(groups[0]["members"]) if groups else 0,
                })
        largest = groups[0]["representative"] if groups else None

        actors = {}
        errors = []
        for actor in all_actors:
            key = _actor_path(actor) or _actor_label(actor)
            try:
                actors[key] = _measure_actor(actor, world, options, largest)
            except Exception as error:
                errors.append({"actor": key, "error": str(error)})
        eligible = [record for record in actors.values() if record["eligible"]]
        projected = [r for r in eligible if r["projected_point_cm"] is not None]
        connected = [r for r in projected if r["connected"] is True]

        payload = _envelope(
            map_path,
            navmesh={
                "actor_path": _actor_path(navmesh_actor) if navmesh_actor else None,
                "class": _class_name(navmesh_actor) if navmesh_actor else None,
                "agent_radius_requested_cm": agent_radius,
                "agent_radius_properties_set": applied,
                "agent_radius_errors": radius_errors,
                "rebuild": rebuild,
                # Read back off the actor AFTER the rebuild: what the navmesh
                # says about itself, not what we asked it for.
                "parameters_after_rebuild": (
                    _navmesh_parameters(navmesh_actor) if navmesh_actor else {}),
                "bounds_min_cm": bounds_min,
                "bounds_max_cm": bounds_max,
                "bounds_source": bounds_source,
                "settle": settle,
            },
            samples={
                "seed": sample_seed,
                "requested_count": sample_count,
                "accepted_count": len(samples),
                "projection_attempts": attempts,
                "projection_extent_cm": sample_extent,
                "points_cm": [_vector(point) for point in samples],
            },
            components={
                "count": len(groups),
                "sizes": [len(group["members"]) for group in groups],
                "largest_size": len(groups[0]["members"]) if groups else 0,
                "largest_representative_cm": _vector(largest) if largest else None,
                "path_endpoint_tolerance_cm": tolerance,
                "cost": cost,
                # One entry per settle-and-regroup pass. More than one settled
                # entry is what lets a component structure be quoted off the
                # fast settle at all — see the module docstring.
                "spread": spread,
            },
            actors=actors,
            diagnostics={
                "status": ("success" if navmesh_actor is not None and not unresolved
                           and not errors and settle.get("settled")
                           and not cost.get("query_budget_truncated") else "partial"),
                "level_actor_count": len(all_actors),
                "eligible_actor_count": len(eligible),
                "projected_actor_count": len(projected),
                "connected_actor_count": len(connected),
                "unresolved": unresolved,
                "measurement_errors": errors,
                "options": options,
            },
        )
    except Exception as error:
        payload = _envelope(
            map_path,
            navmesh={},
            samples={},
            components={},
            actors={},
            diagnostics={
                "status": "error",
                "error": str(error),
                "traceback": traceback.format_exc(),
                "options": options,
            },
        )
    _write_payload(payload, output_path)
    print("SCENE_REACHABILITY status={} path={}".format(
        payload["diagnostics"]["status"], output_path))


main()
