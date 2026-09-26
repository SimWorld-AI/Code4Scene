"""What the editor can be asked about a finished scene — evidence, not verdicts.

This is the Python half of what used to be the vendored scene-editing runtime:
choosing the scoring editor, exporting the loaded Candidate, and collecting
per-Actor physics measurements. The JavaScript that turned those numbers into
eighteen checks is gone — scoring is Python now, one metric per verifier — but
the collection was never JavaScript and is the only way the harness can see
inside a level.

It produces EVIDENCE. Turning evidence into a report belongs to the verifier
that asked for it, the same division of labour as `measure.py` and the
`measure_scene` verifier.

The two scripts under `code4scene/ue_scripts/` run INSIDE the editor: their source is
sent over the bridge with a prelude that names the output path, and the result
is read back from that file.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code4scene.core import inventory as core_inventory
from code4scene.core import scene as core_scene
from code4scene.core.sharedfs import mkdir_shared

from . import asset_catalog as asset_catalog_module
from .scene_diff import diff_scenes, edited_candidate_actors
from .selection import select_candidate_actors

UE_ROOT = Path(__file__).resolve().parents[1] / "ue_scripts"
SNAPSHOT_SCRIPT = UE_ROOT / "export_scene_snapshot.py"
PHYSICS_SCRIPT = UE_ROOT / "measure_scene_physics.py"
DEPENDENCIES_SCRIPT = UE_ROOT / "export_scene_dependencies.py"
REACHABILITY_SCRIPT = UE_ROOT / "measure_reachability.py"

#: What produced the numbers, recorded on every report that carries them. A
#: reader comparing two runs needs to know which collector answered.
EVIDENCE_SOURCE = "scenebench ue probes v1"

# Operational only: chunking changes neither the selected Actors nor any
# physical threshold.  It bounds one serial UE game-thread job so progress is
# materialized between chunks and a single pathological Actor group cannot
# monopolize the editor for the whole scene.
# Live 2k-Actor calibration showed that a chunk of 16 can still exceed ten
# minutes when several building-scale meshes happen to be adjacent in target
# order.  Four bounds that worst-case concentration while still amortizing
# bridge overhead and reusing the same-map spatial/collision-component cache.
DEFAULT_PHYSICS_CHUNK_SIZE = 4
LARGE_SCENE_PHYSICS_TARGET_COUNT = 1000
LARGE_SCENE_PHYSICS_CHUNK_SIZE = 1
MAX_PHYSICS_CHUNK_SIZE = 512
DEFAULT_PHYSICS_CHUNK_RETRIES = 1


class EvidenceError(Exception):
    """The evidence a verifier needs could not be collected."""


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "scene"))
    return text.strip("-.") or "scene"


def _task_path(context: Any, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() and getattr(context.task, "path", None):
        path = Path(context.task.path).parent / path
    return path.resolve()


def _hashed_document(context: Any, descriptor: dict[str, Any], name: str) -> Any:
    raw_path = descriptor.get("path")
    expected = descriptor.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise EvidenceError(f"{name} descriptor path must be non-empty")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected)
    ):
        raise EvidenceError(f"{name} descriptor sha256 must be 64 hex characters")
    path = _task_path(context, raw_path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EvidenceError(f"cannot read {name} from {path}: {error}") from error
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected.casefold():
        raise EvidenceError(
            f"{name} sha256 mismatch for {path}: expected {expected}, got {observed}"
        )
    try:
        return json.loads(payload)
    except ValueError as error:
        raise EvidenceError(f"cannot parse {name} from {path}: {error}") from error


def _derived_input_document(
    context: Any, descriptor: dict[str, Any], name: str
) -> dict[str, Any]:
    base = _hashed_document(context, descriptor, name)
    if not isinstance(base, dict) or not isinstance(base.get("actors"), list):
        raise EvidenceError(f"{name} canonical base must contain an actors array")
    excluded = descriptor.get("exclude_stable_actor_ids")
    if not isinstance(excluded, list) or not excluded:
        raise EvidenceError(
            f"{name} descriptor needs non-empty exclude_stable_actor_ids"
        )
    excluded_ids = {str(value).strip() for value in excluded}
    if "" in excluded_ids or len(excluded_ids) != len(excluded):
        raise EvidenceError(
            f"{name} descriptor stable Actor IDs must be non-empty and unique"
        )
    actors = base["actors"]
    observed = {
        str(actor.get("stable_actor_id") or "").strip()
        for actor in actors
        if isinstance(actor, dict)
        and str(actor.get("stable_actor_id") or "").strip() in excluded_ids
    }
    missing = sorted(excluded_ids - observed)
    if missing:
        raise EvidenceError(
            f"{name} canonical base does not contain excluded stable Actor IDs: "
            f"{missing}"
        )
    result = copy.deepcopy(base)
    result["actors"] = [
        actor
        for actor in result["actors"]
        if str(actor.get("stable_actor_id") or "").strip() not in excluded_ids
    ]
    result["actor_count"] = len(result["actors"])
    result["map_path"] = str(descriptor.get("map_path") or "").strip()
    result["snapshot_type"] = "frozen_input_scene_graph"
    metadata = dict(result.get("export_metadata") or {})
    metadata.update(
        {
            "status": "success",
            "derived_from": str(descriptor["path"]),
            "derived_from_sha256": str(descriptor["sha256"]).casefold(),
            "corruption_operation": "delete_by_stable_actor_id",
        }
    )
    result["export_metadata"] = metadata
    return result


def _document(context: Any, value: Any, name: str) -> Any:
    descriptor = value if isinstance(value, dict) else None
    if descriptor is not None and set(descriptor) == {"path", "sha256"}:
        return _hashed_document(context, descriptor, name)
    if descriptor is not None and set(descriptor) == {
        "path",
        "sha256",
        "exclude_stable_actor_ids",
        "map_path",
    }:
        return _derived_input_document(context, descriptor, name)
    if not isinstance(value, (str, Path)):
        return value
    path = _task_path(context, value)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise EvidenceError(f"cannot read {name} from {path}: {error}") from error


def document(context: Any, value: Any, name: str) -> Any:
    """A JSON document a verifier spec names, read relative to the task file.

    The same resolution the candidate scene and the case specification get: a
    task may name a path or inline the object, and a path is relative to the
    task rather than to whatever directory the harness was started in.
    """
    return _document(context, value, name)


def write_json(path: Path, value: Any) -> None:
    # The evidence root is shared with the editor's own exports: whichever
    # side creates it first, the other must still be able to write into it,
    # and on a shared volume the editor is a different uid.
    mkdir_shared(path.parent)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def artifact_root(context: Any) -> Path:
    scoring_path = getattr(context.scoring, "measure_path", None)
    base = (Path(scoring_path).parent if scoring_path else
            Path(context.artifacts_dir) if context.artifacts_dir else
            Path(context.out_dir) if context.out_dir else None)
    if base is None:
        raise EvidenceError(
            "scene evidence needs out_dir, artifacts_dir, or the independent "
            "scorer's shared measurement path to write into")
    return base / "scene_evidence" / (
        f"{_slug(context.ids.get('task_bundle_id'))}-"
        f"{_slug(context.ids.get('episode_id'))}")


def _bridge(context: Any) -> tuple[Any, str, bool]:
    if context.scoring is not None:
        return context.scoring.bridge, "independent_scoring_editor", True
    if context.spec.get("require_independent", False):
        raise EvidenceError(
            "this verifier requires an independently loaded scoring editor, "
            "but independent scoring did not succeed")
    if context.bridge is None:
        raise EvidenceError("a live editor is needed to export the Candidate")
    return context.bridge, "agent_editor", False


def _run_editor_export(bridge: Any, source_path: Path, prelude: str,
                       output_path: Path, result_key: str,
                       timeout: float) -> dict[str, Any]:
    # The EDITOR writes the output file; on a shared volume it is a different
    # uid, so the directory this process creates must be writable by it.
    mkdir_shared(output_path.parent)
    output_path.unlink(missing_ok=True)
    source = source_path.read_text()
    script = (
        prelude + "\n" + source + "\n"
        + f"globals()[{result_key!r}] = {{'finished': True}}\n"
    )
    result = bridge.exec_python_result(script, result_key, timeout=timeout)
    if not isinstance(result, dict) or result.get("finished") is not True:
        raise EvidenceError(f"UE did not finish {source_path.name}")
    payload = None
    for _ in range(25):
        try:
            payload = json.loads(output_path.read_text())
            break
        except (OSError, ValueError):
            time.sleep(0.25)
    if not isinstance(payload, dict):
        raise EvidenceError(f"UE produced no readable evidence at {output_path}")
    return payload


def _export_candidate(context: Any, root: Path) -> tuple[dict[str, Any], str, bool]:
    configured = context.spec.get("candidate_scene")
    if configured is None:
        configured = context.spec.get("candidate_scene_graph")
    output = root / "candidate.scene.json"
    if configured is not None:
        if context.spec.get("require_independent", False):
            raise EvidenceError(
                "require_independent cannot use a configured candidate_scene; "
                "the Candidate must be exported from the scoring editor")
        candidate = _document(context, configured, "candidate scene graph")
        if not isinstance(candidate, dict):
            raise EvidenceError("candidate scene graph must be a JSON object")
        write_json(output, candidate)
        return candidate, "configured_artifact", False

    bridge, source_name, independent = _bridge(context)
    timeout = float(context.spec.get("ue_timeout_s") or 900)
    prelude = (
        "SCENE_DISTANCE_MAP = ''\n"
        f"SCENE_DISTANCE_OUTPUT = {str(output)!r}"
    )
    candidate = _run_editor_export(
        bridge, SNAPSHOT_SCRIPT, prelude, output,
        "_SB_SCENE_SNAPSHOT", timeout,
    )
    metadata = candidate.get("export_metadata") or {}
    if metadata.get("status") != "success":
        raise EvidenceError(
            f"UE Candidate export failed: {metadata.get('error') or 'unknown error'}")

    return candidate, source_name, independent
def _candidate_cache_key(context: Any) -> str:
    spec = context.spec or {}
    configured = spec.get("candidate_scene")
    if configured is None:
        configured = spec.get("candidate_scene_graph")
    return json.dumps(
        {
            "configured": configured,
            "require_independent": bool(spec.get("require_independent", False)),
            "ue_timeout_s": spec.get("ue_timeout_s"),
            "source": "scoring" if context.scoring is not None else "agent",
        },
        sort_keys=True,
        default=str,
    )


def _cached_export_candidate(
    context: Any, root: Path
) -> tuple[dict[str, Any], str, bool]:
    cache = getattr(context, "cache", None)
    key = ("candidate_scene_export", _candidate_cache_key(context))
    if isinstance(cache, dict) and key in cache:
        candidate, source, independent = cache[key]
        return copy.deepcopy(candidate), source, independent
    candidate, source, independent = _export_candidate(context, root)
    if isinstance(cache, dict):
        cache[key] = (copy.deepcopy(candidate), source, independent)
    return copy.deepcopy(candidate), source, independent


def _runtime_candidate_map(
    context: Any, candidate_scene: dict[str, Any] | None = None
) -> str:
    candidates = [
        (candidate_scene or {}).get("map_path"),
        ((getattr(context, "record", {}) or {}).get("official") or {}).get(
            "level"
        ),
        (getattr(context, "record", {}) or {}).get("scene_map"),
    ]
    for value in candidates:
        package = str(value or "").strip().split(".", 1)[0]
        if package.startswith("/Game/"):
            return package
    raise EvidenceError(
        "runtime repair capture cannot identify this episode's Candidate map"
    )


def _load_runtime_package(context: Any, package: str, timeout: float) -> None:
    bridge, _source_name, independent = _bridge(context)
    if not independent:
        raise EvidenceError(
            "runtime repair scene capture requires the independent scoring editor"
        )
    bridge.exec_python("globals().pop('_SB_LOADED', None)", timeout=60.0)
    result = bridge.exec_python_result(
        core_scene.load_script(package), "_SB_LOADED", timeout=timeout
    )
    if not isinstance(result, dict) or result.get("loaded") is not True:
        raise EvidenceError(f"independent scorer could not load {package}: {result!r}")


def capture_runtime_map_scene(
    context: Any,
    package: str,
    *,
    role: str,
    candidate_scene: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Export one task-owned map live, then restore the run-bound Candidate.

    Task YAMLs carry UE package provenance, never a path to one run's JSON
    artifact.  Input and GT are opened only in the independent scoring editor;
    the Candidate package staged for this episode is restored in ``finally``
    so later verifiers cannot accidentally score the answer key.
    """

    normalized = str(package or "").strip().split(".", 1)[0]
    if not normalized.startswith("/Game/"):
        raise EvidenceError(f"runtime {role} map must be a /Game package")
    candidate_map = _runtime_candidate_map(context, candidate_scene)
    timeout = float(context.spec.get("ue_timeout_s") or 900)
    root = artifact_root(context)
    output = root / f"{_slug(role)}.scene.json"
    key = ("runtime_map_scene", role, normalized, candidate_map)
    cache = getattr(context, "cache", None)
    if isinstance(cache, dict) and key in cache:
        return copy.deepcopy(cache[key]), output

    bridge, _source_name, independent = _bridge(context)
    if not independent:
        raise EvidenceError(
            "runtime repair scene capture requires the independent scoring editor"
        )
    try:
        if normalized != candidate_map:
            _load_runtime_package(context, normalized, timeout)
        scene = core_inventory.read_scene_snapshot(bridge, timeout=timeout)
        observed = str(scene.get("map_path") or "").strip().split(".", 1)[0]
        if observed != normalized:
            raise EvidenceError(
                f"runtime {role} export read {observed!r}, expected {normalized!r}"
            )
        write_json(output, scene)
    finally:
        if normalized != candidate_map:
            _load_runtime_package(context, candidate_map, timeout)
    if isinstance(cache, dict):
        cache[key] = copy.deepcopy(scene)
    return scene, output


def capture_task_ground_truth(
    context: Any,
    *,
    candidate_scene: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    package = getattr(context.task, "ground_truth_map", None)
    if not isinstance(package, str) or not package.strip():
        raise EvidenceError("runtime repair capture needs task.ground_truth_map")
    return capture_runtime_map_scene(
        context,
        package,
        role="ground_truth",
        candidate_scene=candidate_scene,
    )


def _actor_reference(actor: dict[str, Any]) -> dict[str, str]:
    reference = {
        key: str(actor[key]) for key in ("actor_path", "stable_actor_id", "label")
        if actor.get(key)
    }
    if not reference:
        raise EvidenceError("a selected measurement Actor has no stable id, path, or label")
    return reference


def _case_measurement_targets(
    case_spec: Any,
    candidate: dict[str, Any],
    primitives: set[str] | None = None,
    input_scene: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if not isinstance(case_spec, dict):
        return []
    selected: list[dict[str, Any]] = []
    for assertion in case_spec.get("assertions") or []:
        if not isinstance(assertion, dict):
            continue
        primitive = assertion.get("primitive")
        if primitive not in ("physics", "solid_penetration", "physics_regression"):
            continue
        if primitives is not None and primitive not in primitives:
            continue
        raw_selector = assertion.get("target_selector")
        selector = raw_selector if isinstance(raw_selector, dict) else {}
        scope = selector.get("scope") or assertion.get("scope") or "primary_additions"
        if scope == "candidate_all":
            population = candidate.get("actors") or []
        elif scope == "edited_actors":
            if input_scene is None:
                raise EvidenceError(
                    "edited_actors physics scope needs an Input scene snapshot"
                )
            population = edited_candidate_actors(
                diff_scenes(input_scene, candidate)
            )
        else:
            continue
        selected.extend(select_candidate_actors(population, selector))
    unique: dict[str, dict[str, str]] = {}
    for actor in selected:
        reference = _actor_reference(actor)
        key = (reference.get("actor_path") or reference.get("stable_actor_id")
               or reference["label"])
        unique[key] = reference
    return list(unique.values())


def _solid_broad_phase_tolerance(case_spec: Any) -> float:
    """Return the smallest frozen penetration threshold, or zero if absent."""

    if not isinstance(case_spec, dict):
        return 0.0
    thresholds: list[float] = []
    for assertion in case_spec.get("assertions") or []:
        if (
            not isinstance(assertion, dict)
            or assertion.get("primitive") != "solid_penetration"
        ):
            continue
        value = assertion.get("maximum_penetration_cm")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0.0
        ):
            thresholds.append(float(value))
    return min(thresholds, default=0.0)


def _measurement_targets(
    context: Any,
    candidate: dict[str, Any],
    case_spec: Any = None,
    input_scene: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    raw = context.spec.get("measurement_targets")
    if raw is None:
        primitives = (
            {"solid_penetration"}
            if context.spec.get("physics_measurement_mode") == "solid_penetration"
            else None
        )
        return _case_measurement_targets(
            case_spec,
            candidate,
            primitives,
            input_scene,
        )
    raw = _document(context, raw, "measurement target selector")
    if isinstance(raw, list):
        targets = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise EvidenceError(f"measurement_targets[{index}] must be an object")
            reference = {
                key: str(item[key]) for key in ("actor_path", "stable_actor_id", "label")
                if item.get(key)
            }
            if not reference:
                raise EvidenceError(
                    f"measurement_targets[{index}] needs actor_path, stable_actor_id, or label")
            targets.append(reference)
        return targets
    if not isinstance(raw, dict):
        raise EvidenceError("measurement_targets must be a list or selector object")

    selector = {
        "allowed_asset_paths": raw.get("asset_paths") or raw.get("allowed_asset_paths"),
        "allowed_categories": raw.get("categories") or raw.get("allowed_categories"),
        "allowed_classes": raw.get("classes") or raw.get("allowed_classes"),
        "labels": raw.get("labels"),
        "stable_actor_ids": raw.get("stable_actor_ids"),
        "logical_object_ids": raw.get("logical_object_ids"),
        "actor_roles": raw.get("actor_roles"),
        "actor_origins": raw.get("actor_origins"),
        "required_tags": raw.get("required_tags"),
    }
    selector = {key: value for key, value in selector.items() if value}
    actor_paths = {str(value) for value in raw.get("actor_paths") or []}
    if not selector and not actor_paths:
        raise EvidenceError(
            "measurement target selector needs labels, actor_paths, stable_actor_ids, "
            "asset_paths, categories, classes, logical_object_ids, actor_roles, "
            "actor_origins, or required_tags")
    selected = select_candidate_actors(candidate.get("actors") or [], selector)
    if actor_paths:
        selected = [actor for actor in selected if str(actor.get("actor_path")) in actor_paths]
    if not selected:
        raise EvidenceError("measurement target selector matched no Candidate Actors")
    return [_actor_reference(actor) for actor in selected]


def _read_physics_chunk(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _physics_chunk_size(spec: dict[str, Any], target_count: int) -> int:
    """Choose an operational checkpoint size without changing coverage.

    Very large scenes are more likely to contain building-scale targets whose
    exact UE overlap/MTD work dominates a chunk.  A single-Actor checkpoint
    puts the timeout boundary around that irreducible unit while the persistent
    same-map cache still amortizes world scanning and component discovery.
    Explicit task configuration remains authoritative.
    """
    configured = spec.get("physics_chunk_size")
    if configured is not None:
        return int(configured)
    if target_count >= LARGE_SCENE_PHYSICS_TARGET_COUNT:
        return LARGE_SCENE_PHYSICS_CHUNK_SIZE
    return DEFAULT_PHYSICS_CHUNK_SIZE


def _physics_chunk_fingerprint(
    context: Any,
    root: Path,
    targets: list[dict[str, str]],
    options: dict[str, Any],
    index: int,
    count: int,
) -> str:
    candidate_path = root / "candidate.scene.json"
    try:
        candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    except OSError as error:
        raise EvidenceError(
            f"cannot fingerprint Physics Candidate snapshot {candidate_path}: {error}"
        ) from error
    document = {
        "policy_id": "physics-chunk-input-v1",
        "candidate_scene_sha256": candidate_sha256,
        "physics_script_sha256": hashlib.sha256(
            PHYSICS_SCRIPT.read_bytes()
        ).hexdigest(),
        "targets": targets,
        "options": options,
        "chunk_index": index,
        "chunk_count": count,
        "task_bundle_id": context.ids.get("task_bundle_id"),
        "episode_id": context.ids.get("episode_id"),
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_physics_chunk(
    payload: dict[str, Any], expected_targets: list[dict[str, str]]
) -> None:
    diagnostics = payload.get("diagnostics") or {}
    actors = payload.get("actors")
    expected_count = len(expected_targets)
    if diagnostics.get("status") != "success":
        raise EvidenceError(
            "UE physics chunk was incomplete: "
            f"{diagnostics.get('error') or diagnostics}"
        )
    if not isinstance(actors, dict):
        raise EvidenceError("UE physics chunk has no actors object")
    requested = diagnostics.get("requested_actor_count")
    measured = diagnostics.get("measured_actor_count")
    if requested != expected_count or measured != expected_count:
        raise EvidenceError(
            "UE physics chunk coverage mismatch: expected "
            f"{expected_count}, requested {requested}, measured {measured}"
        )
    if len(actors) != expected_count:
        raise EvidenceError(
            "UE physics chunk actor map lost or duplicated identities: "
            f"expected {expected_count}, found {len(actors)}"
        )
    # Candidate snapshots normally provide Actor paths, which are the same
    # strong identities used as measurement keys inside UE. Count equality
    # alone is insufficient: a resolver bug could otherwise replace one
    # requested Actor with another and still produce a superficially complete
    # chunk. Fall back to the count envelope only for legacy label-only input.
    expected_paths = {
        str(target["actor_path"])
        for target in expected_targets
        if target.get("actor_path")
    }
    if len(expected_paths) == expected_count and set(actors) != expected_paths:
        missing = sorted(expected_paths - set(actors))
        unexpected = sorted(set(actors) - expected_paths)
        raise EvidenceError(
            "UE physics chunk Actor identities do not match its target partition: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def _wait_for_editor_recovery(bridge: Any, timeout: float) -> bool:
    """Wait for the supervisor to replace an explicitly abandoned UE job."""
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        try:
            status = bridge.busy(timeout=10.0)
        except Exception:  # noqa: BLE001 - bridge may be between processes
            status = {}
        if bool(status.get("busy")):
            time.sleep(15)
            continue
        try:
            bridge.ping(timeout=20.0)
            return True
        except Exception:  # noqa: BLE001 - cold boot is expected here
            time.sleep(15)
    return False


def _merge_physics_chunks(
    payloads: list[dict[str, Any]],
    *,
    requested_actor_count: int,
    options: dict[str, Any],
    chunk_size: int,
    chunk_timeout: float,
    chunk_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not payloads:
        raise EvidenceError("cannot merge an empty Physics measurement")
    first = payloads[0]
    actors: dict[str, Any] = {}
    unresolved: list[Any] = []
    errors: list[Any] = []
    calibration_sweeps: list[Any] = []
    cache_records: list[Any] = []
    for index, payload in enumerate(payloads):
        for field in ("map_path", "capabilities", "runtime_provenance"):
            if payload.get(field) != first.get(field):
                raise EvidenceError(
                    f"Physics chunk {index} disagrees on {field}; refusing to "
                    "merge measurements from different editor states"
                )
        for key, measurement in (payload.get("actors") or {}).items():
            if key in actors:
                raise EvidenceError(
                    f"Physics chunks contain duplicate Actor identity {key!r}"
                )
            actors[key] = measurement
        diagnostics = payload.get("diagnostics") or {}
        unresolved.extend(diagnostics.get("unresolved_targets") or [])
        errors.extend(diagnostics.get("measurement_errors") or [])
        calibration_sweeps.extend(diagnostics.get("calibration_sweeps") or [])
        if diagnostics.get("solid_actor_cache") is not None:
            cache_records.append(diagnostics["solid_actor_cache"])
    status = (
        "success"
        if not unresolved and not errors and len(actors) == requested_actor_count
        else "partial"
    )
    return {
        "schema_version": first.get("schema_version"),
        "measurement_type": first.get("measurement_type"),
        "map_path": first.get("map_path"),
        "capabilities": first.get("capabilities"),
        "runtime_provenance": first.get("runtime_provenance"),
        "actors": actors,
        "diagnostics": {
            "status": status,
            "requested_actor_count": requested_actor_count,
            "measured_actor_count": len(actors),
            "unresolved_targets": unresolved,
            "measurement_errors": errors,
            "calibration_sweeps": calibration_sweeps,
            "options": options,
            "solid_actor_cache_by_chunk": cache_records,
            "chunking": {
                "policy_id": "complete-target-partition-v1",
                "chunk_size": chunk_size,
                "chunk_timeout_s": chunk_timeout,
                "chunk_count": len(payloads),
                "completed_chunk_count": len(payloads),
                "complete_coverage_required": True,
                "chunks": chunk_records,
            },
        },
    }


def _measure_candidate(
    context: Any,
    root: Path,
    targets: list[dict[str, str]],
    case_spec: Any,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    configured = context.spec.get("candidate_measurements")
    output = root / "candidate.physics.json"
    if configured is not None:
        if context.spec.get("require_independent", False):
            raise EvidenceError(
                "require_independent cannot use configured candidate_measurements; "
                "physics evidence must come from the scoring editor that exported "
                "the Candidate"
            )
        measurements = _document(context, configured, "Candidate physics measurements")
        if not isinstance(measurements, dict):
            raise EvidenceError("Candidate physics measurements must be a JSON object")
        write_json(output, measurements)
        return measurements, "configured_artifact", False
    if not targets:
        return None, None, False

    bridge, source_name, independent = _bridge(context)
    configured_options = context.spec.get("physics_options") or {}
    if not isinstance(configured_options, dict):
        raise EvidenceError("physics_options must be an object")
    options = dict(configured_options)
    derived_tolerance = _solid_broad_phase_tolerance(case_spec)
    configured_tolerance = options.get("solid_penetration_tolerance_cm")
    if (
        configured_tolerance is not None
        and float(configured_tolerance) != derived_tolerance
    ):
        raise EvidenceError(
            "physics_options.solid_penetration_tolerance_cm is derived from "
            "the frozen solid_penetration assertion and cannot override it"
        )
    options["solid_penetration_tolerance_cm"] = derived_tolerance
    if context.spec.get("physics_measurement_mode") == "solid_penetration":
        options["solid_penetration_only"] = True

    timeout = float(context.spec.get("ue_timeout_s") or 900)
    chunk_timeout = float(context.spec.get("physics_chunk_timeout_s") or timeout)
    chunk_size = _physics_chunk_size(context.spec, len(targets))
    if chunk_size < 1 or chunk_size > MAX_PHYSICS_CHUNK_SIZE:
        raise EvidenceError(
            f"physics_chunk_size must be in [1, {MAX_PHYSICS_CHUNK_SIZE}]"
        )
    if chunk_timeout <= 0.0:
        raise EvidenceError("physics_chunk_timeout_s must be positive")
    retries = int(context.spec.get(
        "physics_chunk_retries", DEFAULT_PHYSICS_CHUNK_RETRIES
    ))
    if retries < 0 or retries > 3:
        raise EvidenceError("physics_chunk_retries must be in [0, 3]")
    recovery_timeout = float(
        context.spec.get("physics_recovery_timeout_s") or 1800.0
    )
    if recovery_timeout <= 0.0:
        raise EvidenceError("physics_recovery_timeout_s must be positive")
    chunks = [targets[index:index + chunk_size]
              for index in range(0, len(targets), chunk_size)]
    configured_progress_root = os.environ.get("CODE4SCENE_PHYSICS_PROGRESS_DIR")
    chunk_root = (
        Path(configured_progress_root).expanduser().resolve()
        if configured_progress_root
        else root / "candidate.physics.chunks"
    )
    chunk_payloads: list[dict[str, Any]] = []
    chunk_records: list[dict[str, Any]] = []
    for index, chunk_targets in enumerate(chunks):
        chunk_output = chunk_root / f"chunk-{index:05d}.json"
        chunk_options = dict(options)
        # Calibration moves are global diagnostics, not per-target evidence.
        # Run them once rather than once per operational chunk.
        if index:
            chunk_options["calibration_sweeps"] = []
        chunk_options.update({
            "measurement_chunk_index": index,
            "measurement_chunk_count": len(chunks),
        })
        chunk_fingerprint = _physics_chunk_fingerprint(
            context,
            root,
            chunk_targets,
            chunk_options,
            index,
            len(chunks),
        )
        prelude = (
            f"SCENE_PHYSICS_OUTPUT = {str(chunk_output)!r}\n"
            f"SCENE_PHYSICS_TARGETS_JSON = {json.dumps(chunk_targets)!r}\n"
            f"SCENE_PHYSICS_OPTIONS_JSON = {json.dumps(chunk_options)!r}"
        )
        started = time.monotonic()
        last_error: Exception | None = None
        payload: dict[str, Any] | None = None
        attempts = 0
        reused_progress = False
        existing_payload = _read_physics_chunk(chunk_output)
        if (
            existing_payload is not None
            and (existing_payload.get("chunk_provenance") or {}).get("fingerprint")
            == chunk_fingerprint
        ):
            try:
                _validate_physics_chunk(existing_payload, chunk_targets)
            except EvidenceError:
                pass
            else:
                payload = existing_payload
                reused_progress = True
        for attempt in range(retries + 1):
            if payload is not None:
                break
            attempts = attempt + 1
            try:
                candidate_payload = _run_editor_export(
                    bridge,
                    PHYSICS_SCRIPT,
                    prelude,
                    chunk_output,
                    f"_SB_SCENE_PHYSICS_{index}",
                    chunk_timeout,
                )
                _validate_physics_chunk(candidate_payload, chunk_targets)
                candidate_payload["chunk_provenance"] = {
                    "policy_id": "physics-chunk-input-v1",
                    "fingerprint": chunk_fingerprint,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                }
                write_json(chunk_output, candidate_payload)
                payload = candidate_payload
                break
            except Exception as error:  # noqa: BLE001 - retry only after recovery
                last_error = error
                if attempt >= retries:
                    break
                if not _wait_for_editor_recovery(bridge, recovery_timeout):
                    break
                # A payload can finish writing its evidence after the caller's
                # socket deadline but before UE restarts.  Reuse that exact
                # chunk rather than executing collision probes twice.
                late_payload = _read_physics_chunk(chunk_output)
                if late_payload is not None:
                    try:
                        _validate_physics_chunk(late_payload, chunk_targets)
                    except EvidenceError:
                        pass
                    else:
                        late_payload["chunk_provenance"] = {
                            "policy_id": "physics-chunk-input-v1",
                            "fingerprint": chunk_fingerprint,
                            "chunk_index": index,
                            "chunk_count": len(chunks),
                        }
                        write_json(chunk_output, late_payload)
                        payload = late_payload
                        break
        if payload is None:
            raise EvidenceError(
                f"UE physics chunk {index + 1}/{len(chunks)} failed after "
                f"{attempts} attempt(s): {last_error}"
            )
        chunk_payloads.append(payload)
        chunk_records.append({
            "index": index,
            "target_count": len(chunk_targets),
            "measured_actor_count": len(payload.get("actors") or {}),
            "attempts": attempts,
            "reused_progress": reused_progress,
            "fingerprint": chunk_fingerprint,
            "elapsed_s": round(time.monotonic() - started, 3),
            "artifact": str(chunk_output),
        })
    measurements = _merge_physics_chunks(
        chunk_payloads,
        requested_actor_count=len(targets),
        options=options,
        chunk_size=chunk_size,
        chunk_timeout=chunk_timeout,
        chunk_records=chunk_records,
    )
    write_json(output, measurements)
    diagnostics = measurements.get("diagnostics") or {}
    if diagnostics.get("status") != "success":
        raise EvidenceError(
            "UE Candidate physics measurement was incomplete: "
            f"{diagnostics.get('error') or diagnostics}")
    return measurements, source_name, independent


def _input_scene(
    context: Any,
    root: Path,
    candidate_scene: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """The scene the agent was HANDED, when the task supplies one.

    Not an answer key: this is the pre-edit level, the same content the agent
    drove all episode. Comparing against it asks "did you break what you were
    not asked to touch", which the agent could ask itself — so the verifiers
    built on it are open-ended. The canonical scene is a different document,
    reached through the answer-key helper in `context`, and nothing here
    can see it.

    It arrives as a configured artifact rather than a second editor load: the
    input level is what the harness staged, so it can be exported once when
    the episode is provisioned instead of re-opened per verifier.
    """
    configured = context.spec.get("input_scene")
    if configured is None:
        configured = context.spec.get("input_scene_graph")
    if configured is None:
        return None, None
    if configured == {"runtime_task_input_map": True}:
        package = str(getattr(context.task, "init_map", "") or "").strip()
        scene, _path = capture_runtime_map_scene(
            context,
            package,
            role="input",
            candidate_scene=candidate_scene,
        )
        return scene, "independent_scoring_editor_runtime_task_input_map"
    scene = _document(context, configured, "input scene graph")
    if not isinstance(scene, dict):
        raise EvidenceError("input scene graph must be a JSON object")
    # The candidate export is refused unless it says it succeeded; the input
    # export was not, and a failed one is an empty Actor list — against which
    # every preservation question answers "nothing was disturbed".
    metadata = scene.get("export_metadata")
    if isinstance(metadata, dict) and metadata.get("status") != "success":
        raise EvidenceError(
            f"the input scene export reports status "
            f"{metadata.get('status')!r} ({metadata.get('error') or 'no error given'}); "
            f"comparing against a scene that failed to export reads as a scene "
            f"nothing was done to")
    write_json(root / "input.scene.json", scene)
    return scene, "configured_artifact"


def _input_measurements(context: Any, root: Path) -> dict[str, Any] | None:
    """Per-Actor physics for the input scene, when the task supplies them.

    `physics_regression` needs both sides measured the same way; measuring
    only the candidate and calling the difference a regression would report
    every scene that was already penetrating as newly broken.
    """
    configured = context.spec.get("input_measurements")
    if configured is None:
        return None
    measurements = _document(context, configured, "input physics measurements")
    if not isinstance(measurements, dict):
        raise EvidenceError("input physics measurements must be a JSON object")
    write_json(root / "input.physics.json", measurements)
    return measurements


def run_probe(context: Any, script: Path, prelude_values: dict[str, Any],
              output: Path, result_key: str) -> tuple[dict[str, Any], str, bool]:
    """Run one `code4scene/ue_scripts/` script in the editor and read its payload back.

    The editor choice, the timeout and the read-back retry are the same for
    every probe; only the script and the values it is handed differ.
    """
    bridge, source_name, independent = _bridge(context)
    prelude = "\n".join(f"{name} = {value!r}" for name, value in prelude_values.items())
    timeout = float(context.spec.get("ue_timeout_s") or 900)
    payload = _run_editor_export(bridge, script, prelude, output, result_key, timeout)
    return payload, source_name, independent


def semantic_contract(context: Any) -> Any:
    """The frozen case specification a task declares, read from disk."""
    explicit = context.spec.get("semantic_contract")
    legacy = context.spec.get("case_spec")
    if explicit is not None and legacy is not None:
        raise EvidenceError("use semantic_contract or case_spec, not both")
    value = explicit if explicit is not None else legacy
    return _document(context, value, "semantic contract") if value is not None else None


def case_id(context: Any) -> str:
    return str(context.spec.get("case_id") or context.task.id)


@dataclass(frozen=True)
class SceneEvidence:
    """One collection pass, with where every piece of it came from."""

    candidate: dict[str, Any]
    candidate_path: Path
    case_spec: Any
    case_id: str
    root: Path
    measurements: dict[str, Any] | None = None
    measurements_path: Path | None = None
    #: Why physics evidence is absent, when it is. A verifier that needs it
    #: reports this rather than scoring a scene it could not measure.
    measurement_error: str | None = None
    exported_from: str = "agent_editor"
    independent: bool = False
    measurements_from: str | None = None
    measurements_independent: bool = False
    target_count: int = 0
    #: The pre-edit scene, when the task supplies one. Absent is the normal
    #: case: most tasks generate rather than edit, and a verifier that needs
    #: it says so rather than inventing an empty baseline.
    input_scene: dict[str, Any] | None = None
    input_scene_from: str | None = None
    input_measurements: dict[str, Any] | None = None
    #: Which catalog answered "what kind of thing is this", and how much of
    #: the Candidate it could name. None means no catalog was configured, and
    #: any category a selector sees is metadata the Candidate carried — which
    #: on a generated scene is metadata the agent wrote. A reader comparing
    #: two semantic scores needs to know which of those they are.
    asset_catalog_id: str | None = None
    asset_catalog_size: int = 0
    categories_resolved: int = 0
    categories_unknown: int = 0

    def provenance(self) -> dict[str, Any]:
        """Where each piece came from, for the atom's evidence envelope."""
        return {
            "candidate_exported_from": self.exported_from,
            "independent_scoring_editor": self.independent,
            "measurements_collected_from": self.measurements_from,
            "measurements_from_independent_editor": self.measurements_independent,
        }

    def evidence(self) -> dict[str, Any]:
        """The evidence block a VerifierReport carries."""
        return {
            **self.provenance(),
            "measurement_collection_error": self.measurement_error,
            "measurement_target_count": self.target_count,
            "semantic_contract_present": self.case_spec is not None,
            "input_scene_present": self.input_scene is not None,
            "input_scene_from": self.input_scene_from,
            "asset_catalog_id": self.asset_catalog_id,
            "asset_catalog_size": self.asset_catalog_size,
            "categories_resolved": self.categories_resolved,
            "categories_unknown": self.categories_unknown,
            "evidence_source": EVIDENCE_SOURCE,
        }

    def candidate_actors(self) -> list[dict[str, Any]]:
        return [actor for actor in self.candidate.get("actors") or []
                if isinstance(actor, dict)]

    def input_actors(self) -> list[dict[str, Any]]:
        return [actor for actor in (self.input_scene or {}).get("actors") or []
                if isinstance(actor, dict)]

    def probes_used(self) -> list[str]:
        """Name the mechanisms actually used, including configured artifacts."""
        probes = ["configured_scene_artifact"
                  if self.exported_from == "configured_artifact"
                  else "ue_scene_snapshot"]
        if self.asset_catalog_id:
            probes.append("asset_catalog")
        if self.measurements_path:
            probes.append("configured_physics_artifact"
                          if self.measurements_from == "configured_artifact"
                          else "ue_actor_physics")
        return probes

    def artifacts(self) -> dict[str, str]:
        return {
            "candidate_scene_graph": str(self.candidate_path),
            **({"candidate_measurements": str(self.measurements_path)}
               if self.measurements_path else {}),
        }


#: The verifier-spec keys `collect` reads. Two verifiers whose specs agree on
#: all of them get the same evidence, so they share one collection pass; a
#: verifier that asked for different targets gets its own.
_COLLECTION_KEYS = ("candidate_scene", "candidate_scene_graph", "input_scene",
                    "input_scene_graph", "candidate_measurements",
                    "input_measurements", "measurement_targets",
                    "physics_options", "semantic_contract", "case_spec",
                    "case_id", "asset_catalog", "require_independent",
                    "ue_timeout_s", "physics_measurement_mode",
                    "physics_chunk_size", "physics_chunk_timeout_s",
                    "physics_chunk_retries", "physics_recovery_timeout_s")


def _cache_key(context: Any) -> str:
    spec = context.spec or {}
    return json.dumps({key: spec.get(key) for key in _COLLECTION_KEYS},
                      sort_keys=True, default=str)


def collect(context: Any) -> SceneEvidence:
    """Export the Candidate and measure what the task asked to have measured.

    A failure to MEASURE is carried, not raised: the population is still known
    and a verifier can report exactly which Actors it could not see. A failure
    to EXPORT is raised, because without the Candidate there is no population
    and nothing downstream can mean anything.

    The result is cached per episode. Sixteen verifiers now read this, and an
    export runs arbitrary Python inside the editor over every Actor in the
    level: doing that sixteen times would make the scoring pass cost more than
    the episode, and — worse — two verifiers could then be describing two
    different exports of the same scene.
    """
    key = _cache_key(context)
    cache = getattr(context, "cache", None)
    if isinstance(cache, dict):
        cached = cache.get(("scene_evidence", key))
        if cached is not None:
            return cached
    evidence = _collect(context)
    if isinstance(cache, dict):
        cache[("scene_evidence", key)] = evidence
    return evidence


def _asset_catalog(context: Any) -> Any:
    """The catalog a task names, if it names one."""
    configured = context.spec.get("asset_catalog")
    if configured is None:
        return None
    return asset_catalog_module.load(
        _document(context, configured, "asset catalog"),
        identifier=configured if isinstance(configured, str) else None)


def _collect(context: Any) -> SceneEvidence:
    root = artifact_root(context)
    candidate, exported_from, independent = _cached_export_candidate(context, root)
    catalog = _asset_catalog(context)
    catalog_counts: dict[str, int] = {}
    if catalog is not None:
        # The Candidate only. `input_scene` is an authored level the agent
        # never wrote, so the categories in it are curated rather than
        # claimed, and re-deriving them would discard better information.
        catalog_counts = catalog.apply(
            [actor for actor in candidate.get("actors") or []
             if isinstance(actor, dict)])
        write_json(root / "candidate.scene.json", candidate)
    case_spec = semantic_contract(context)
    # A local repair-physics population is defined by Input -> Candidate.
    # Capture the Input before choosing UE measurement targets; Candidate-only
    # policies retain exactly the same target set and evidence semantics.
    input_scene, input_scene_from = _input_scene(context, root, candidate)
    measurement_error = None
    try:
        targets = _measurement_targets(
            context,
            candidate,
            case_spec,
            input_scene,
        )
    except Exception as error:      # noqa: BLE001 — carried as evidence
        targets = []
        measurement_error = f"{type(error).__name__}: {error}"
    measurements = measurements_from = None
    measurements_independent = False
    if measurement_error is None:
        try:
            measurements, measurements_from, measurements_independent = _measure_candidate(
                context, root, targets, case_spec
            )
        except Exception as error:  # noqa: BLE001 — carried as evidence
            measurement_error = f"{type(error).__name__}: {error}"
    return SceneEvidence(
        candidate=candidate,
        candidate_path=root / "candidate.scene.json",
        case_spec=case_spec,
        case_id=case_id(context),
        root=root,
        measurements=measurements,
        measurements_path=(root / "candidate.physics.json"
                           if measurements is not None else None),
        measurement_error=measurement_error,
        exported_from=exported_from,
        independent=independent,
        measurements_from=measurements_from,
        measurements_independent=measurements_independent,
        target_count=len(targets),
        input_scene=input_scene,
        input_scene_from=input_scene_from,
        input_measurements=_input_measurements(context, root),
        asset_catalog_id=getattr(catalog, "id", None),
        asset_catalog_size=len(catalog) if catalog is not None else 0,
        categories_resolved=catalog_counts.get("categories_resolved", 0),
        categories_unknown=catalog_counts.get("categories_unknown", 0),
    )


__all__ = ["EVIDENCE_SOURCE", "REACHABILITY_SCRIPT", "EvidenceError", "SceneEvidence",
           "artifact_root", "capture_runtime_map_scene",
           "capture_task_ground_truth", "case_id", "collect", "document", "run_probe",
           "semantic_contract", "write_json"]


def scene_dependencies(bridge: Any, out_dir: Any, level: str = "",
                       timeout: float = 900.0) -> dict[str, Any]:
    """Every package the open level references, resolved or not.

    A separate probe rather than a field on the scene snapshot: the snapshot
    describes actors, this describes packages, and the two answer different
    questions about the same scene. It is also the only evidence that survives
    the environment — an actor list says what was placed, a dependency
    manifest says whether the next editor can open it.
    """
    if bridge is None:
        raise EvidenceError("a live editor is needed to list scene dependencies")
    output = Path(out_dir) / "scene_dependencies.json"
    prelude = (f"SCENE_DEPENDENCIES_OUTPUT = {str(output)!r}\n"
               f"SCENE_DEPENDENCIES_MAP = {level!r}")
    payload = _run_editor_export(bridge, DEPENDENCIES_SCRIPT, prelude, output,
                                 "_SB_SCENE_DEPENDENCIES", timeout)
    if payload.get("status") != "success":
        raise EvidenceError(
            f"listing scene dependencies failed: {payload.get('error')}")
    return payload
