"""Content fingerprints for Code4Scene levels.

A fingerprint summarises one saved Unreal level, as exported by
``ue/export_snapshot.py`` (scene snapshot schema 0.3.0), without revealing its
arrangement. It is used to check that a level a user built from the Fab packs
is the same level the benchmark was scored on.

Design
------
* One *leaf* per actor, keyed by the actor's benchmark stable ID
  (``stable_actor_id``). The leaf is a SHA-256 over four canonical *facets*:

  ``identity``    class path, actor label, component mesh assets, actor tags
                  (without the stable-ID tag, which is the key itself), the
                  benchmark origin/role/logical-object tags and the attachment
                  parent (as a stable ID when the parent is in the level).
  ``materials``   every (component, slot, material path) triple. Dynamic
                  material instances created at load time (``/Engine/Transient``
                  objects and level sub-objects) are reduced to their class,
                  because their object paths change with the map name and
                  between editor sessions.
  ``transform``   location, rotation and scale, quantised: location to 0.5 cm,
                  rotation as the 3x3 rotation matrix to 5e-4 (about 0.03 degrees),
                  scale to 5e-4. This is far below the scorer's tolerances
                  (5 cm, 5 degrees, 5 %) and far above float noise.
                  The rotation matrix is used instead of Euler angles so that
                  equivalent rotators (yaw 180 vs -180, gimbal aliases) agree.
  ``properties``  the task-relevant light/fog/post-process properties recorded
                  by the exporter, floats rounded to 6 significant digits.

* The *root* is a SHA-256 over the sorted ``"<stable_id> <leaf>"`` lines, so it
  does not depend on actor order.
* The expected fingerprints shipped with the benchmark contain the root, the
  actor count and, per stable ID, a 16-hex-digit leaf plus 16-hex-digit facet
  digests. No transform, label or asset path is shipped.

Float tolerance
---------------
Quantisation alone would make a value that sits within float noise of a
rounding boundary hash differently on two machines. ``match_leaf`` therefore
re-hashes a locally computed actor with the neighbouring bucket for every
quantised coordinate that lies within ``PROBE_FRACTION`` of a quantum of a
boundary (at most 2**k combinations for the k nearest such coordinates, k <=
``MAX_PROBE_COORDINATES``). A local actor matches when any combination
reproduces the expected leaf. Hence differences below 0.1 cm, 1e-4 per rotation
matrix entry (about 0.006 degrees) and 1e-4 in scale always match, and
differences above one quantum never do.

Only the Python standard library is used, so this module runs both on the
host and, if needed, inside the Unreal Editor's Python.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any, Iterable, Mapping, Sequence

FINGERPRINT_SCHEMA = "code4scene.content_fingerprint.v1"
ALGORITHM = "c4s-fp-1"

LOCATION_QUANTUM_CM = 0.5
ROTATION_QUANTUM = 5e-4
SCALE_QUANTUM = 5e-4
PROBE_FRACTION = 0.2
MAX_PROBE_COORDINATES = 12

STABLE_ID_TAG_PREFIX = "simcodearena.stable_actor_id="
TRANSIENT_PREFIX = "/Engine/Transient"
LEAF_DIGITS = 16


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalize_material_path(path: Any, material_class: Any = None, is_dynamic: Any = False) -> Any:
    """Reduce load-time material instances to their class.

    Construction scripts create dynamic material instances every time a level
    loads. Their object path contains either ``/Engine/Transient`` or the
    owning level package (``<map>.<map>:PersistentLevel.<actor>.<MID>``), so it
    changes with the map name and between sessions. Only their class is stable.
    """

    if not isinstance(path, str):
        return path
    if bool(is_dynamic) or path.startswith(TRANSIENT_PREFIX) or ":PersistentLevel." in path:
        cls = str(material_class or "MaterialInstanceDynamic").rsplit(".", 1)[-1]
        return f"<dynamic:{cls}>"
    return path


def _round_sig(value: float, digits: int = 6) -> Any:
    if not isinstance(value, float) or not math.isfinite(value):
        return value
    if value == 0.0:
        return 0.0
    return float(f"{value:.{digits}g}")


def _normalize_property(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _round_sig(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_property(item) for item in value]
    return str(value)


def rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> list[float]:
    """Unreal ``FRotationMatrix`` of a rotator, row-major (X, Y, Z axes)."""

    p, y, r = (math.radians(float(v)) for v in (pitch_deg, yaw_deg, roll_deg))
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    sr, cr = math.sin(r), math.cos(r)
    return [
        cp * cy, cp * sy, sp,
        sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp,
        -(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp,
    ]


def _transform_values(actor: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Return (value, quantum) pairs in a fixed order."""

    transform = actor.get("transform") or {}
    location = [float(v) for v in transform.get("location_cm") or (0.0, 0.0, 0.0)]
    rotation = [float(v) for v in transform.get("rotation_deg") or (0.0, 0.0, 0.0)]
    scale = [float(v) for v in transform.get("scale") or (1.0, 1.0, 1.0)]
    values = [(v, LOCATION_QUANTUM_CM) for v in location]
    values += [(v, ROTATION_QUANTUM) for v in rotation_matrix(*rotation)]
    values += [(v, SCALE_QUANTUM) for v in scale]
    return values


def _bucket(value: float, quantum: float) -> int:
    return int(math.floor(value / quantum + 0.5))


def _near_boundary(value: float, quantum: float) -> int:
    """Return -1/+1 when ``value`` is within the probe band of a boundary."""

    scaled = value / quantum + 0.5
    frac = scaled - math.floor(scaled)
    if frac < PROBE_FRACTION:
        return -1
    if frac > 1.0 - PROBE_FRACTION:
        return 1
    return 0


def _boundary_distance(value: float, quantum: float) -> float:
    scaled = value / quantum + 0.5
    frac = scaled - math.floor(scaled)
    return min(frac, 1.0 - frac)


class SnapshotIndex:
    """Lookups that facets need across actors of the same snapshot."""

    def __init__(self, snapshot: Mapping[str, Any]):
        self.map_path = str(snapshot.get("map_path") or "")
        self.by_path: dict[str, str] = {}
        for actor in snapshot.get("actors") or []:
            path = actor.get("actor_path")
            if path:
                self.by_path[str(path)] = str(actor.get("stable_actor_id"))

    def parent_key(self, actor: Mapping[str, Any]) -> Any:
        path = actor.get("attachment_parent_path")
        if not path:
            return None
        stable = self.by_path.get(str(path))
        if stable:
            return {"stable_actor_id": stable}
        return {"label": actor.get("attachment_parent_label")}


def facets(actor: Mapping[str, Any], index: SnapshotIndex) -> dict[str, Any]:
    tags = sorted(
        str(tag) for tag in (actor.get("actor_tags") or [])
        if not str(tag).startswith(STABLE_ID_TAG_PREFIX)
    )
    identity = {
        "class": actor.get("class"),
        "label": actor.get("label"),
        "component_asset_paths": sorted(set(actor.get("component_asset_paths") or [])),
        "tags": tags,
        "actor_origin": actor.get("actor_origin"),
        "actor_role": actor.get("actor_role"),
        "logical_object_id": actor.get("logical_object_id"),
        "attachment_parent": index.parent_key(actor),
    }
    materials = sorted(
        [
            str(slot.get("component_identity")),
            int(slot.get("slot_index") or 0),
            normalize_material_path(
                slot.get("material_path"),
                slot.get("material_class_path"),
                slot.get("is_dynamic"),
            ),
        ]
        for slot in (actor.get("component_material_slots") or [])
    )
    properties = {
        str(key): _normalize_property(value)
        for key, value in sorted((actor.get("properties") or {}).items())
    }
    return {"identity": identity, "materials": materials, "properties": properties}


def _digest_facets(stable_id: str, parts: Mapping[str, Any], buckets: Sequence[int]) -> tuple[str, dict[str, str]]:
    facet_digests = {
        "identity": _sha256(_canonical(parts["identity"])),
        "materials": _sha256(_canonical(parts["materials"])),
        "transform": _sha256(_canonical(list(buckets))),
        "properties": _sha256(_canonical(parts["properties"])),
    }
    leaf = _sha256(_canonical({"id": stable_id, **facet_digests}))
    return leaf[:LEAF_DIGITS], {k: v[:LEAF_DIGITS] for k, v in facet_digests.items()}


def actor_leaf(actor: Mapping[str, Any], index: SnapshotIndex) -> dict[str, Any]:
    stable_id = str(actor.get("stable_actor_id"))
    parts = facets(actor, index)
    values = _transform_values(actor)
    buckets = [_bucket(v, q) for v, q in values]
    leaf, facet_digests = _digest_facets(stable_id, parts, buckets)
    return {"leaf": leaf, "facets": facet_digests}


def match_leaf(actor: Mapping[str, Any], index: SnapshotIndex, expected_leaf: str) -> tuple[bool, dict[str, Any]]:
    """Match one local actor against an expected leaf with boundary probing."""

    stable_id = str(actor.get("stable_actor_id"))
    parts = facets(actor, index)
    values = _transform_values(actor)
    buckets = [_bucket(v, q) for v, q in values]
    leaf, facet_digests = _digest_facets(stable_id, parts, buckets)
    if leaf == expected_leaf:
        return True, {"leaf": leaf, "facets": facet_digests, "probed": False}
    near = [(i, _near_boundary(v, q), _boundary_distance(v, q)) for i, (v, q) in enumerate(values)]
    near = sorted((item for item in near if item[1]), key=lambda item: item[2])[:MAX_PROBE_COORDINATES]
    near = [(i, d) for i, d, _ in near]
    for choice in itertools.product((0, 1), repeat=len(near)):
        if not any(choice):
            continue
        probe = list(buckets)
        for (i, direction), use in zip(near, choice):
            if use:
                probe[i] += direction
        probe_leaf, probe_facets = _digest_facets(stable_id, parts, probe)
        if probe_leaf == expected_leaf:
            return True, {"leaf": probe_leaf, "facets": probe_facets, "probed": True}
    return False, {"leaf": leaf, "facets": facet_digests, "probed": bool(near)}


def root_of(leaves: Mapping[str, str]) -> str:
    lines = "\n".join(f"{key} {leaves[key]}" for key in sorted(leaves))
    return _sha256(f"{ALGORITHM}\n{lines}\n")


def fingerprint_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the full fingerprint (root plus per-actor leaves) of a snapshot."""

    index = SnapshotIndex(snapshot)
    actors: dict[str, dict[str, Any]] = {}
    for actor in snapshot.get("actors") or []:
        stable_id = str(actor.get("stable_actor_id"))
        if stable_id in actors:
            raise ValueError(f"duplicate stable_actor_id in snapshot: {stable_id}")
        actors[stable_id] = actor_leaf(actor, index)
    leaves = {key: value["leaf"] for key, value in actors.items()}
    return {
        "schema_version": FINGERPRINT_SCHEMA,
        "algorithm": ALGORITHM,
        "actor_count": len(actors),
        "root": root_of(leaves),
        "actors": actors,
    }


def delta_fingerprint(base: Mapping[str, Any], derived: Mapping[str, Any]) -> dict[str, Any]:
    """Express ``derived`` as a delta over ``base`` (both full fingerprints).

    Used for Input levels: an Input is its GT plus a few edits, so only the
    changed, added and removed stable IDs are stored next to the Input's root.
    """

    base_actors = base["actors"]
    derived_actors = derived["actors"]
    changed = {
        key: value for key, value in derived_actors.items()
        if key not in base_actors or base_actors[key]["leaf"] != value["leaf"]
    }
    removed = sorted(key for key in base_actors if key not in derived_actors)
    return {
        "schema_version": FINGERPRINT_SCHEMA,
        "algorithm": ALGORITHM,
        "base_root": base["root"],
        "actor_count": derived["actor_count"],
        "root": derived["root"],
        "changed_or_added": dict(sorted(changed.items())),
        "removed": removed,
    }


def expand_delta(base: Mapping[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a full expected fingerprint from a base and a delta."""

    actors = {key: dict(value) for key, value in base["actors"].items()}
    for key in delta.get("removed") or []:
        actors.pop(key, None)
    for key, value in (delta.get("changed_or_added") or {}).items():
        actors[key] = dict(value)
    leaves = {key: value["leaf"] for key, value in actors.items()}
    root = root_of(leaves)
    if root != delta["root"]:
        raise ValueError("expanded delta does not reproduce its declared root")
    return {
        "schema_version": FINGERPRINT_SCHEMA,
        "algorithm": ALGORITHM,
        "actor_count": len(actors),
        "root": root,
        "actors": actors,
    }


def compare(snapshot: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Compare a local snapshot against an expected full fingerprint."""

    index = SnapshotIndex(snapshot)
    expected_actors = expected["actors"]
    local_ids = []
    matched: dict[str, str] = {}
    mismatched = []
    extra = []
    probed = 0
    for actor in snapshot.get("actors") or []:
        stable_id = str(actor.get("stable_actor_id"))
        local_ids.append(stable_id)
        want = expected_actors.get(stable_id)
        if want is None:
            extra.append({"stable_actor_id": stable_id, "label": actor.get("label"),
                          "class": actor.get("class")})
            continue
        ok, info = match_leaf(actor, index, want["leaf"])
        if ok:
            matched[stable_id] = want["leaf"]
            probed += int(info["probed"])
            continue
        differing = sorted(
            name for name, digest in info["facets"].items()
            if digest != want["facets"].get(name)
        )
        mismatched.append({
            "stable_actor_id": stable_id,
            "label": actor.get("label"),
            "class": actor.get("class"),
            "differing_facets": differing,
        })
    missing = sorted(set(expected_actors) - set(local_ids))
    exact = not (mismatched or extra or missing)
    local_root = root_of({**matched, **{m["stable_actor_id"]: "mismatch" for m in mismatched}})
    return {
        "match": exact,
        "expected_root": expected["root"],
        "resolved_root": expected["root"] if exact else local_root,
        "expected_actor_count": expected["actor_count"],
        "local_actor_count": len(local_ids),
        "matched_actor_count": len(matched),
        "matched_after_float_probe": probed,
        "mismatched": mismatched,
        "missing": missing,
        "extra": extra,
    }


FACET_ORDER = ("leaf", "identity", "materials", "transform", "properties")


def compact_fingerprint(full: Mapping[str, Any]) -> dict[str, Any]:
    """Serialise a full fingerprint in the compact on-disk form."""

    return {
        "schema_version": FINGERPRINT_SCHEMA,
        "algorithm": ALGORITHM,
        "actor_count": full["actor_count"],
        "root": full["root"],
        "facet_order": list(FACET_ORDER),
        "actors": {
            key: [value["leaf"]] + [value["facets"][name] for name in FACET_ORDER[1:]]
            for key, value in sorted(full["actors"].items())
        },
    }


def load_expected(document: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either the compact on-disk form or a full fingerprint."""

    if document.get("algorithm") != ALGORITHM:
        raise ValueError(f"unsupported fingerprint algorithm {document.get('algorithm')!r}")
    actors = {}
    for key, value in (document.get("actors") or {}).items():
        if isinstance(value, list):
            order = document.get("facet_order") or list(FACET_ORDER)
            parts = dict(zip(order, value))
            actors[key] = {"leaf": parts["leaf"],
                           "facets": {name: parts[name] for name in FACET_ORDER[1:]}}
        else:
            actors[key] = {"leaf": value["leaf"], "facets": dict(value["facets"])}
    root = root_of({key: value["leaf"] for key, value in actors.items()})
    if root != document.get("root"):
        raise ValueError("expected fingerprint is internally inconsistent (root does not match leaves)")
    return {
        "schema_version": FINGERPRINT_SCHEMA,
        "algorithm": ALGORITHM,
        "actor_count": len(actors),
        "root": root,
        "actors": actors,
    }


def summarize_facets(mismatched: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in mismatched:
        for name in item.get("differing_facets") or []:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "ALGORITHM",
    "FACET_ORDER",
    "FINGERPRINT_SCHEMA",
    "compact_fingerprint",
    "compare",
    "load_expected",
    "delta_fingerprint",
    "expand_delta",
    "fingerprint_snapshot",
    "normalize_material_path",
    "root_of",
    "rotation_matrix",
    "summarize_facets",
]
