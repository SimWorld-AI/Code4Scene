"""What is in the level, as the editor sees it.

Shared by the two halves of an ablation task: the generator records an
inventory before it removes anything, and the restoration verifier takes one
of the finished scene to compare against. Kept here rather than in
``code4scene.tasks.taskgen`` so that scoring a task does not import the machinery
that makes one — an installed harness that never generates a task still has to
be able to score one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from .bridge import Bridge

#: The editor global an inventory is stashed in. Read back by name rather than
#: scraped from the log, which is a drifting time window and not this call's.
INVENTORY_KEY = "_SB_INVENTORY"

#: Actors that are the level rather than its contents. An ablation never
#: removes these and a restoration never counts them against an agent: the
#: landscape and the lighting were there before the episode and will be there
#: after it.
KEEP_CLASSES = ("WorldSettings", "Landscape", "LandscapeStreamingProxy",
                "DirectionalLight", "SkyLight", "SkyAtmosphere",
                "ExponentialHeightFog", "VolumetricCloud", "PostProcessVolume",
                "PlayerStart", "LevelBounds", "Brush", "AbstractNavData",
                "SphereReflectionCapture", "WorldPartitionMiniMap")


#: Label prefixes for the scenery that frames a level rather than sits in it
#: (raw case, ``startswith``). The sibling of ``KEEP_CLASSES`` above, matched
#: on the LABEL because the bounds pass sees labels and not classes.
#:
#: Here in `core` because two layers must agree on it and neither may import
#: the other. `evaluation.bounds` skips these actors when it enforces the
#: world plate; `tasks.taskgen` skips them when it MEASURES that plate. They
#: disagreed exactly once, and once was enough: Hangar's sky sphere sits
#: 16,384 m out, so a plate measured over every actor came to 34 km — a
#: boundary nothing can be outside of, which is the failure a plate exists to
#: prevent.
SCENERY_PREFIXES = ("Directional", "Sky", "Fog", "PostProcess", "Landscape",
                    "WorldSettings", "Brush", "Atmospher", "Volumetric")


def script() -> str:
    """Editor-python that records what is in the level, with enough to group it."""
    return "\n".join([
        "import unreal",
        "_subs = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
        f"_keep = {list(KEEP_CLASSES)!r}",
        "_items = []",
        "for _a in _subs.get_all_level_actors():",
        "    try:",
        "        _cls = _a.get_class().get_name()",
        "        _o, _e = _a.get_actor_bounds(False)",
        "        _items.append({'label': _a.get_actor_label(), 'cls': _cls,",
        "                       'keep': any(_k in _cls for _k in _keep),",
        "                       'loc': [_o.x, _o.y, _o.z],",
        "                       'extent': [_e.x, _e.y, _e.z]})",
        "    except Exception:",
        "        pass",
        f"globals()[{INVENTORY_KEY!r}] = {{'actors': _items}}",
    ])


def read(bridge: Bridge, timeout: float = 600.0) -> list[dict[str, Any]]:
    """Take an inventory of the level the editor is on."""
    bridge.exec_python(f"globals().pop({INVENTORY_KEY!r}, None)", timeout=60.0)
    return bridge.exec_python_result(script(), INVENTORY_KEY,
                                     timeout=timeout)["actors"]


#: The editor global a scene graph is stashed in.
GRAPH_KEY = "_SB_SCENEGRAPH"
SNAPSHOT_PAYLOAD_KEY = "_SB_SCENE_SNAPSHOT_PAYLOAD"
SNAPSHOT_SCHEMA_VERSION = "0.3.0"
ATTRIBUTE_EVIDENCE_FIELDS = (
    "properties",
    "material_paths",
    "component_material_slots",
)
SNAPSHOT_EXPORTER = (
    Path(__file__).resolve().parents[1] / "ue_scripts" / "export_scene_snapshot.py"
)
GENERATED_STABLE_ID_PREFIX = "sca_generated_"


def _generated_stable_actor_id(
    actor_name: str,
    class_path: str,
    component_asset_paths: list[str],
) -> str:
    """Mirror the pinned snapshot exporter's fallback Actor identity."""
    signature = {
        "actor_name": str(actor_name),
        "class_path": str(class_path),
        "component_asset_paths": sorted(
            set(str(path) for path in component_asset_paths)
        ),
    }
    canonical = json.dumps(
        signature, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return GENERATED_STABLE_ID_PREFIX + hashlib.sha256(canonical).hexdigest()[:32]


def scene_graph_script() -> str:
    """Export the verifier's full scene-snapshot contract in memory.

    Case authoring previously maintained a smaller second exporter here. It
    omitted properties, material sets, and component/slot identities, so a
    freshly authored GT label could never drive ``scene_diff.attribute_diff``.
    Execute the canonical UE exporter instead and return exactly the same
    payload through the bridge without requiring a shared output file.
    """

    source = SNAPSHOT_EXPORTER.read_text()
    return "\n".join(
        [
            "SCENE_DISTANCE_MAP = ''",
            "SCENE_DISTANCE_OUTPUT = ''",
            source,
            f"globals()[{GRAPH_KEY!r}] = globals().get(",
            f"    {SNAPSHOT_PAYLOAD_KEY!r}, {{}})",
        ]
    )


def compact_scene_graph_script() -> str:
    """Export the compact repair identity shape used by older authoring tools.

    Richer than :func:`script` because the scene-repair policy asks about
    identity, not just position: which asset, which class, and the stable id
    the pinned exporter assigns. Explicit ids travel in the UMAP as Actor tags;
    construction-script Actors that cannot retain tags use the same structural
    fallback here and in the pinned exporter.
    """
    return "\n".join([
        "import unreal",
        "import hashlib",
        "import json",
        "_subs = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
        "_items = []",
        "for _a in _subs.get_all_level_actors():",
        "    try:",
        "        _t = _a.get_actor_transform()",
        "        _loc, _rot, _sc = _t.translation, _t.rotation.rotator(), _t.scale3d",
        # Asset and identity must be spelled the way the ANSWER KEY spells them.
        # `scene_repair` matches (asset_path, class) between this candidate and
        # the release's own exporter output, so a divergence here is not a
        # cosmetic difference: it makes an honest restoration unmatchable, and
        # the actor is then counted BOTH missing and unexpected. The rules
        # below mirror ue/export_scene_snapshot.py verbatim —
        #   class: the real class PATH (a Blueprint is /Game/../BP_X.BP_X_C,
        #          never /Script/Engine.BP_X_C);
        #   asset: every component's static_mesh / skeletal_mesh_asset /
        #          skeletal_mesh, and a single one is the primary asset; a
        #          multi-mesh actor has no canonical first asset, so it falls
        #          back to the class path when that is content, else unset.
        "        _assets = []",
        "        try:",
        "            for _comp in (_a.get_components_by_class(unreal.ActorComponent) or []):",
        "                for _prop in ('static_mesh', 'skeletal_mesh_asset', 'skeletal_mesh'):",
        "                    try:",
        "                        _obj = _comp.get_editor_property(_prop)",
        "                    except Exception:",
        "                        continue",
        "                    if _obj:",
        "                        _p = _obj.get_path_name()",
        "                        if _p and _p not in _assets:",
        "                            _assets.append(_p)",
        "        except Exception:",
        "            pass",
        "        _assets = sorted(set(str(_p) for _p in _assets))",
        "        _actor_name = str(_a.get_name())",
        "        _cls = _a.get_class().get_path_name()",
        "        _asset = _assets[0] if len(_assets) == 1 else (",
        "            _cls if _cls.startswith('/Game/') else None)",
        "        _tags = [str(_x) for _x in _a.tags]",
        "        _parsed_tags = {}",
        "        for _tag in _tags:",
        "            if '=' in _tag:",
        "                _key, _value = _tag.split('=', 1)",
        "                _parsed_tags[_key.strip().lower()] = _value.strip()",
        "        _stable_id = (",
        "            _parsed_tags.get('simcodearena.stable_actor_id') or",
        "            _parsed_tags.get('stable_actor_id'))",
        "        if not _stable_id:",
        "            _signature = {",
        "                'actor_name': _actor_name,",
        "                'class_path': str(_cls),",
        "                'component_asset_paths': sorted(set(",
        "                    str(_path) for _path in _assets)),",
        "            }",
        "            _canonical = json.dumps(",
        "                _signature, sort_keys=True, separators=(',', ':')",
        "            ).encode('utf-8')",
        f"            _stable_id = {GENERATED_STABLE_ID_PREFIX!r} + hashlib.sha256(",
        "                _canonical).hexdigest()[:32]",
        "        _items.append({",
        "            'label': _a.get_actor_label(),",
        "            'stable_actor_id': str(_stable_id),",
        "            'actor_path': _a.get_path_name(),",
        "            'class': _cls,",
        "            'asset_path': _asset,",
        "            'actor_tags': _tags,",
        "            'transform': {'location_cm': [_loc.x, _loc.y, _loc.z],",
        "                          'rotation_deg': [_rot.pitch, _rot.yaw, _rot.roll],",
        "                          'scale': [_sc.x, _sc.y, _sc.z]}})",
        "    except Exception:",
        "        pass",
        "_stable_id_owners = {}",
        "for _item in _items:",
        "    _stable_id = _item.get('stable_actor_id')",
        "    if not _stable_id:",
        "        raise RuntimeError('candidate Actor has no stable identity')",
        "    if _stable_id in _stable_id_owners:",
        "        raise RuntimeError(",
        "            'duplicate stable_actor_id {} on Actors {} and {}: copying a tagged Actor copies its simcodearena.stable_actor_id tag, so remove or change the tag on the copy and save again'.format(",
        "                _stable_id, _stable_id_owners[_stable_id],",
        "                _item.get('label')))",
        "    _stable_id_owners[_stable_id] = _item.get('label')",
        f"globals()[{GRAPH_KEY!r}] = {{'actors': _items}}",
    ])


def attribute_evidence_summary(
    actors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe whether every Actor carries the attribute scoring contract."""

    actor_count = len(actors)
    missing_by_field = {
        field: sum(field not in actor for actor in actors)
        for field in ATTRIBUTE_EVIDENCE_FIELDS
    }
    complete = sum(
        all(field in actor for field in ATTRIBUTE_EVIDENCE_FIELDS)
        for actor in actors
    )
    return {
        "required_fields": list(ATTRIBUTE_EVIDENCE_FIELDS),
        "actor_count": actor_count,
        "complete_actor_count": complete,
        "missing_actor_count": actor_count - complete,
        "coverage": round(complete / actor_count, 6) if actor_count else None,
        "missing_actor_count_by_field": missing_by_field,
    }


def read_scene_snapshot(bridge: Bridge, timeout: float = 900.0) -> dict[str, Any]:
    """Export and validate the canonical full scene-snapshot payload."""

    bridge.exec_python(f"globals().pop({GRAPH_KEY!r}, None)", timeout=60.0)
    payload = bridge.exec_python_result(
        scene_graph_script(), GRAPH_KEY, timeout=timeout
    )
    if not isinstance(payload, Mapping):
        raise ValueError("scene snapshot exporter returned no object")
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "scene snapshot schema mismatch: expected "
            f"{SNAPSHOT_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    metadata = payload.get("export_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("status") != "success":
        reason = metadata.get("error") if isinstance(metadata, Mapping) else None
        raise ValueError(
            f"scene snapshot export failed: {reason or 'unknown error'}"
        )
    actors = payload.get("actors")
    if not isinstance(actors, list) or any(
        not isinstance(actor, Mapping) for actor in actors
    ):
        raise ValueError("scene snapshot actors must be an array of objects")
    if payload.get("actor_count") != len(actors):
        raise ValueError("scene snapshot actor_count does not match actors")
    return dict(payload)


def read_scene_graph(
    bridge: Bridge, timeout: float = 900.0
) -> list[dict[str, Any]]:
    """Export the current level in the scene-repair exporter's shape."""

    return list(read_scene_snapshot(bridge, timeout=timeout)["actors"])


__all__ = [
    "ATTRIBUTE_EVIDENCE_FIELDS",
    "GENERATED_STABLE_ID_PREFIX",
    "GRAPH_KEY",
    "INVENTORY_KEY",
    "KEEP_CLASSES",
    "SNAPSHOT_PAYLOAD_KEY",
    "SNAPSHOT_SCHEMA_VERSION",
    "attribute_evidence_summary",
    "compact_scene_graph_script",
    "read",
    "read_scene_graph",
    "read_scene_snapshot",
    "scene_graph_script",
    "script",
]
