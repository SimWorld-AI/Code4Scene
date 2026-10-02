"""Level building for the Code4Scene dataset (runs inside the Unreal Editor).

Implements the two build stages on a stock UE 5.8 editor:

* ``canonicalize(scene)``: load the pack's demo map, write benchmark identity
  tags, apply the scene patches and save the result as the GT level.
* ``materialize(recipe, scene)``: load the locally built GT level, apply the
  recipe's level-structure ops and edit ops, and save the result as the task
  Input level.

The op semantics mirror ``dataset_builder/recipe.py`` (offline simulation).
Nothing here saves a package that belongs to a Fab pack.
"""

import hashlib
import uuid

import unreal

import c4s_snapshot as snap

STABLE_TAG_KEY = "simcodearena.stable_actor_id"
IDENTITY_TAG_KEYS = ("simcodearena.stable_actor_id", "simcodearena.logical_object_id",
                     "simcodearena.actor_role", "simcodearena.actor_origin")


class BuildError(RuntimeError):
    pass


def actors_subsystem():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def editor_world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def load_map(path):
    if not unreal.EditorAssetLibrary.does_asset_exist(path):
        raise BuildError("map does not exist: {}".format(path))
    world = unreal.EditorLoadingAndSavingUtils.load_map(path)
    if world is None or snap.current_map_package() != path:
        raise BuildError("failed to load {} (editor reports {})".format(path, snap.current_map_package()))
    return world


def save_map_as(world, path):
    if not unreal.EditorLoadingAndSavingUtils.save_map(world, path):
        # UE may report False after writing the package; trust the asset registry.
        if not unreal.EditorAssetLibrary.does_asset_exist(path):
            raise BuildError("failed to save {}".format(path))
    return path


def delete_if_exists(path):
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        if not unreal.EditorAssetLibrary.delete_asset(path):
            raise BuildError("could not delete existing {}".format(path))


def load_object(path):
    asset = unreal.load_asset(path)
    if asset is None:
        raise BuildError("asset not found: {}".format(path))
    return asset


def load_class(path):
    if path.startswith("/Script/"):
        cls = unreal.load_class(None, path)
    else:
        cls = unreal.load_class(None, path) or unreal.EditorAssetLibrary.load_blueprint_class(path.rsplit("_C", 1)[0])
    if cls is None:
        raise BuildError("class not found: {}".format(path))
    return cls


def set_tags(actor, tags):
    actor.set_editor_property("tags", [unreal.Name(str(t)) for t in tags])


def without_identity_tags(tags):
    return [str(t) for t in tags if not any(str(t).startswith(k + "=") for k in IDENTITY_TAG_KEYS)]


def rotator(values):
    pitch, yaw, roll = (float(v) for v in values)
    return unreal.Rotator(roll=roll, pitch=pitch, yaw=yaw)


def vector(values):
    return unreal.Vector(float(values[0]), float(values[1]), float(values[2]))


def package_of(actor):
    return str(actor.get_path_name()).split(":", 1)[0].split(".", 1)[0]


# ---------------------------------------------------------------------------
# Identity (canonicalization)
# ---------------------------------------------------------------------------

def _uuid5_stable_id(namespace_url, gt_id, source_map, actor_path, class_path):
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, namespace_url)
    seed = "v1|source|{}|{}|{}|{}".format(gt_id, source_map, actor_path, class_path)
    return "sca_source_" + uuid.uuid5(namespace, seed).hex


def _is_child_actor(actor):
    if "_GEN_VARIABLE_" in str(actor.get_path_name()):
        return True
    try:
        return actor.get_parent_actor() is not None
    except Exception:
        return False


def write_identity(scene):
    identity = scene["identity"]
    scheme = identity["scheme"]
    source_map = scene["source_map"]
    tagged = 0
    seen = set()
    if scheme == "untagged":
        return {"scheme": scheme, "tagged": 0}
    for actor in list(actors_subsystem().get_all_level_actors()):
        path = str(actor.get_path_name())
        class_path = str(actor.get_class().get_path_name())
        tags = [str(t) for t in actor.tags]
        if scheme == "uuid5-source-v1":
            if _is_child_actor(actor) or package_of(actor) != source_map:
                continue
            stable = _uuid5_stable_id(identity["namespace_url"], identity["gt_id"], source_map, path, class_path)
            previous = {t.split("=", 1)[0]: t.split("=", 1)[1] for t in tags if "=" in t}
            merged = without_identity_tags(tags)
            merged.append("{}={}".format(STABLE_TAG_KEY, stable))
            for key in ("simcodearena.logical_object_id", "simcodearena.actor_role"):
                if previous.get(key):
                    merged.append("{}={}".format(key, previous[key]))
            merged.append("simcodearena.actor_origin={}".format(identity.get("tag_actor_origin", "source")))
        elif scheme == "sha256-source-v1":
            seed = source_map + "|" + str(actor.get_name()) + "|" + class_path
            stable = identity["prefix"] + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]
            prefix = STABLE_TAG_KEY + "="
            merged = [t for t in tags if not t.lower().startswith(prefix.lower())]
            merged.append(prefix + stable)
        else:
            raise BuildError("unknown identity scheme {}".format(scheme))
        if stable in seen:
            raise BuildError("duplicate generated stable ID {}".format(stable))
        seen.add(stable)
        try:
            set_tags(actor, merged)
            tagged += 1
        except Exception as error:
            raise BuildError("could not tag {}: {}".format(path, error))
    return {"scheme": scheme, "tagged": tagged}


# ---------------------------------------------------------------------------
# Streaming levels
# ---------------------------------------------------------------------------

def _level_package(level):
    try:
        return str(level.get_outermost().get_name())
    except Exception:
        return str(level.get_path_name()).split(".", 1)[0]


def _levels(world):
    return list(unreal.EditorLevelUtils.get_levels(world))


def make_level_current(level):
    try:
        unreal.EditorLevelUtils.make_level_current(level)
        return
    except Exception:
        pass
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).set_current_level_by_name(
        _level_package(level).rsplit("/", 1)[-1])


def make_persistent_current(world):
    root = snap.current_map_package()
    for level in _levels(world):
        if _level_package(level) == root:
            make_level_current(level)
            return
    raise BuildError("persistent level not found")


def _streaming_class(world, package):
    try:
        streaming = unreal.GameplayStatics.get_streaming_level(world, package)
        if streaming is not None:
            return streaming.get_class()
    except Exception:
        pass
    return unreal.LevelStreamingDynamic


def remove_streaming_level(world, package):
    cls = _streaming_class(world, package)
    for level in _levels(world):
        if _level_package(level) == package:
            if not unreal.EditorLevelUtils.remove_level_from_world(level):
                raise BuildError("could not remove streaming level {}".format(package))
            return cls
    raise BuildError("streaming level {} is not loaded in {}".format(package, snap.current_map_package()))


def add_streaming_level(world, package, streaming_class, create_empty):
    if create_empty:
        delete_if_exists(package)
        streaming = unreal.EditorLevelUtils.create_new_streaming_level(streaming_class, package, False)
    else:
        streaming = unreal.EditorLevelUtils.add_level_to_world(world, package, streaming_class)
    if streaming is None:
        raise BuildError("could not add streaming level {}".format(package))
    return streaming


def save_streaming_level(world, streaming):
    level = streaming.get_loaded_level()
    if level is None:
        raise BuildError("streaming level is not loaded")
    make_level_current(level)
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level():
        raise BuildError("could not save streaming level {}".format(_level_package(level)))
    make_persistent_current(world)


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------

def index_by_stable_id():
    records, objects = snap.snapshot_level_actors()
    return {r["stable_actor_id"]: (r, o) for r, o in zip(records, objects)}


def _target(index, op):
    entry = index.get(op["target"])
    if entry is None:
        raise BuildError("{}: target {} is not in the level (pack version mismatch?)".format(op["op"], op["target"]))
    return entry[1]


def _mesh_component(actor, name):
    components = list(actor.get_components_by_class(unreal.StaticMeshComponent))
    for component in components:
        if str(component.get_name()) == name:
            return component
    if len(components) == 1:
        return components[0]
    raise BuildError("component {} not found on {}".format(name, actor.get_actor_label()))


def _slot_paths(component):
    result = []
    for index in range(int(component.get_num_materials())):
        material = component.get_material(index)
        result.append(snap.object_path(material))
    return result


def _set_first_property(actor, component_class, prop, value):
    classes = [unreal.ActorComponent]
    try:
        cls = unreal.load_class(None, component_class)
        if cls is not None:
            classes.insert(0, cls)
    except Exception:
        pass
    for cls in classes:
        for component in actor.get_components_by_class(cls):
            try:
                component.get_editor_property(prop)
            except Exception:
                continue
            component.modify()
            component.set_editor_property(prop, value)
            return True
    return False


def _rename(actor, name):
    if str(actor.get_name()) == name:
        return True
    try:
        actor.rename(name)
    except Exception:
        return False
    return str(actor.get_name()) == name


def apply_op(op, index, templates, notes):
    kind = op["op"]
    subsystem = actors_subsystem()
    if kind == "remove_actor":
        actor = _target(index, op)
        actor.modify()
        if not subsystem.destroy_actor(actor):
            raise BuildError("could not remove {}".format(op["target"]))
    elif kind == "set_transform":
        actor = _target(index, op)
        actor.modify()
        if "location_cm" in op:
            actor.set_actor_location(vector(op["location_cm"]), False, True)
        if "rotation_deg" in op:
            actor.set_actor_rotation(rotator(op["rotation_deg"]), True)
        if "scale" in op:
            actor.set_actor_scale3d(vector(op["scale"]))
    elif kind == "set_static_mesh":
        actor = _target(index, op)
        component = _mesh_component(actor, op["component"])
        actor.modify()
        component.modify()
        if not component.set_static_mesh(load_object(op["static_mesh"])):
            raise BuildError("could not assign {}".format(op["static_mesh"]))
        expected = [s.get("material_path") for s in sorted((op.get("expect") or {}).get("material_slots") or [],
                                                             key=lambda s: s["slot_index"])]
        observed = _slot_paths(component)
        if expected and observed != expected:
            notes.append({"op": kind, "target": op["target"], "warning": "materials differ after mesh swap",
                          "expected": expected, "observed": observed})
    elif kind == "set_label":
        actor = _target(index, op)
        actor.modify()
        actor.set_actor_label(op["label"], True)
    elif kind == "set_component_property":
        actor = _target(index, op)
        if not _set_first_property(actor, op.get("component_class", ""), op["property"], op["value"]):
            raise BuildError("no component of {} has property {}".format(op["target"], op["property"]))
    elif kind == "spawn_actor":
        source = op.get("from") or {}
        location, rotation = vector(op["location_cm"]), rotator(op["rotation_deg"])
        if source.get("static_mesh"):
            actor = subsystem.spawn_actor_from_class(unreal.StaticMeshActor, location, rotation)
            if actor is None:
                raise BuildError("could not spawn StaticMeshActor")
            component = actor.get_editor_property("static_mesh_component")
            component.modify()
            if not component.set_static_mesh(load_object(source["static_mesh"])):
                raise BuildError("could not assign {}".format(source["static_mesh"]))
        else:
            actor = subsystem.spawn_actor_from_class(load_class(source["class"]), location, rotation)
            if actor is None:
                raise BuildError("could not spawn {}".format(source["class"]))
        actor.set_actor_scale3d(vector(op["scale"]))
        actor.set_actor_rotation(rotation, True)
        renamed = _rename(actor, op["object_name"])
        actor.set_actor_label(op["label"], True)
        expect = op.get("expect") or {}
        tags = op.get("tags")
        if tags is None:
            if expect.get("like"):
                tags = templates.get(expect["like"], {}).get("tags")
            else:
                tags = expect.get("tags")
        tags = list(tags if tags is not None else without_identity_tags([str(t) for t in actor.tags]))
        if op.get("identity_tag") or not renamed:
            # An explicit identity tag reproduces the expected ID even if the
            # engine refused the object name the ID was derived from.
            tags.append("{}={}".format(STABLE_TAG_KEY, op["stable_id"]))
            if not renamed:
                notes.append({"op": kind, "stable_id": op["stable_id"],
                              "note": "object rename refused; wrote explicit identity tag instead"})
        set_tags(actor, tags)
    else:
        raise BuildError("unknown op {}".format(kind))


def templates_for(ops, index):
    wanted = {((op.get("expect") or {}).get("like")) for op in ops if op.get("op") == "spawn_actor"}
    result = {}
    for stable_id in wanted:
        if stable_id and stable_id in index:
            record = index[stable_id][0]
            result[stable_id] = {"tags": without_identity_tags(record["actor_tags"]),
                                 "record": record}
    return result


def ensure_supplements(ops):
    """Generate content a scene needs that no pack ships (idempotent).

    ``duplicate_asset`` copies an installed asset to the package path a pack
    references, e.g. Starter Content's macro texture for the
    ``/Game/Cabin_Pack/...`` texture that OldBuilding's materials expect. No
    map or material is resaved.
    """
    library = unreal.EditorAssetLibrary
    done = []
    for op in ops:
        if op.get("op") != "duplicate_asset":
            raise BuildError("unknown supplement op {}".format(op.get("op")))
        source, destination = op["source"], op["destination"]
        if library.does_asset_exist(destination):
            asset = library.load_asset(destination)
            created = False
        else:
            if library.load_asset(source) is None:
                raise BuildError("supplement source {} is not installed (install the pack that "
                                 "ships it, e.g. Starter Content)".format(source))
            asset = library.duplicate_asset(source, destination)
            if asset is None or not library.save_asset(destination, only_if_is_dirty=False):
                raise BuildError("could not create {} from {}".format(destination, source))
            created = True
        expected = op.get("asset_class")
        if expected and asset.get_class().get_name() != expected:
            raise BuildError("{} is a {}, expected {}".format(destination, asset.get_class().get_name(), expected))
        done.append({"destination": destination, "source": source, "created": created})
    return {"supplements": done}


def canonicalize(scene, force=False):
    target = scene["ground_truth_map"]
    if unreal.EditorAssetLibrary.does_asset_exist(target) and not force:
        return {"scene_id": scene["scene_id"], "status": "exists", "map": target}
    world = load_map(scene["source_map"])
    identity = write_identity(scene)
    notes = []
    patches = scene.get("patches") or []
    index = index_by_stable_id()
    created_levels = {}
    for op in patches:
        kind = op["op"]
        if kind == "remove_streaming_level":
            created_levels.setdefault("_classes", {})[op["level"]] = remove_streaming_level(world, op["level"])
        elif kind == "add_streaming_level":
            cls = created_levels.get("_classes", {}).get(op.get("streaming_class_like"), unreal.LevelStreamingDynamic)
            created_levels[op["level"]] = add_streaming_level(world, op["level"], cls, op.get("create_empty", False))
        elif kind == "spawn_actor" and op.get("level"):
            streaming = created_levels.get(op["level"])
            if streaming is None:
                raise BuildError("spawn into unknown level {}".format(op["level"]))
            make_level_current(streaming.get_loaded_level())
            spawn = dict(op)
            spawn.setdefault("expect", {})
            apply_op(spawn, index, {}, notes)
            make_persistent_current(world)
        else:
            apply_op(op, index, {}, notes)
    delete_if_exists(target)
    for package, streaming in created_levels.items():
        if package != "_classes":
            save_streaming_level(world, streaming)
    save_map_as(world, target)
    return {"scene_id": scene["scene_id"], "status": "built", "map": target, "identity": identity, "notes": notes}


def materialize(recipe, force=False):
    target = recipe["input_map"]
    if unreal.EditorAssetLibrary.does_asset_exist(target) and not force:
        return {"case_id": recipe["case_id"], "status": "exists", "map": target}
    world = load_map(recipe["ground_truth_map"])
    notes = []
    for op in recipe.get("level_structure") or []:
        if op["op"] != "retarget_streaming_level":
            raise BuildError("unknown level-structure op {}".format(op["op"]))
        copied = False
        if op.get("copy_package") and not unreal.EditorAssetLibrary.does_asset_exist(op["to"]):
            if unreal.EditorAssetLibrary.duplicate_asset(op["from"], op["to"]) is None:
                raise BuildError("could not copy {} to {}".format(op["from"], op["to"]))
            copied = True
        cls = remove_streaming_level(world, op["from"])
        streaming = add_streaming_level(world, op["to"], cls, False)
        if copied:
            save_streaming_level(world, streaming)
    index = index_by_stable_id()
    templates = templates_for(recipe["operations"], index)
    for op in recipe["operations"]:
        apply_op(op, index, templates, notes)
    delete_if_exists(target)
    save_map_as(world, target)
    return {"case_id": recipe["case_id"], "status": "built", "map": target, "notes": notes}


def create_blank_stage(path, force=False):
    if unreal.EditorAssetLibrary.does_asset_exist(path) and not force:
        return {"map": path, "status": "exists"}
    delete_if_exists(path)
    world = unreal.EditorLoadingAndSavingUtils.new_blank_map(False)
    save_map_as(world, path)
    return {"map": path, "status": "built"}
