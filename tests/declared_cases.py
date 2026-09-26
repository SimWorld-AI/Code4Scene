"""Builders for the declared-assertion verifiers' tests.

These verifiers read a scene graph and a frozen case specification. Both
arrive as CONFIGURED ARTIFACTS — a supported path in `ue_evidence`, the same
one a task uses when the scene was exported ahead of time — so the whole
scoring half is exercised without an editor. What is NOT exercised this way is
the export itself; that needs a real editor and is not what these tests claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code4scene.evaluation.context import Context

IDS = {"task_bundle_id": "bundle-test", "episode_id": "episode-test"}


class StubTask:
    """The two fields evidence collection reads off a task."""

    def __init__(self, path: Path, task_id: str = "declared-case-test",
                 prompt: str = "build the declared scene") -> None:
        self.path = path
        self.id = task_id
        self.prompt = prompt


def actor(label: str, *, location=(0.0, 0.0, 0.0), extent=(25.0, 25.0, 50.0),
          origin: tuple[float, float, float] | None = None,
          asset_path: str | None = None, actor_class: str | None = None,
          rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
          **extra: Any) -> dict[str, Any]:
    """One Actor in the exporter's shape. ``origin`` defaults to the pivot."""
    return {
        "label": label,
        "actor_path": f"/Game/Test.Test:PersistentLevel.{label}",
        "class": actor_class or "/Script/Engine.StaticMeshActor",
        "asset_path": asset_path,
        "actor_tags": [],
        "transform": {"location_cm": list(location),
                      "rotation_deg": list(rotation), "scale": list(scale)},
        "bounds": {"origin_cm": list(origin if origin is not None else location),
                   "extent_cm": list(extent)},
        "properties": {},
        **extra,
    }


def scene(actors: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"actor_count": len(actors), "actors": actors,
            "map_path": "/Game/Test", **extra}


def case(assertions: list[dict[str, Any]],
         case_id: str = "declared-case-test") -> dict[str, Any]:
    return {"schema_version": "0.2.0", "case_id": case_id,
            "assertions": assertions}


def context(tmp_path: Path, *, candidate: dict[str, Any],
            case_spec: dict[str, Any] | None = None,
            input_scene: dict[str, Any] | None = None,
            spec: dict[str, Any] | None = None) -> Context:
    """A Context whose evidence comes from files, not from an editor."""
    task_path = tmp_path / "task.yaml"
    task_path.write_text("id: declared-case-test\n")
    entry: dict[str, Any] = dict(spec or {})

    def write(name: str, document: Any) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(document))
        return str(path)

    entry["candidate_scene"] = write("candidate.json", candidate)
    if case_spec is not None:
        entry["semantic_contract"] = write("case.json", case_spec)
        entry.setdefault("case_id", case_spec.get("case_id"))
    if input_scene is not None:
        entry["input_scene"] = write("input.json", input_scene)
    return Context(record={"metrics": {}}, task=StubTask(task_path), ids=dict(IDS),
                   out_dir=tmp_path / "out", spec=entry)


def check_by_id(report: dict[str, Any], check_id: str) -> dict[str, Any]:
    """One check out of a report, by its id. Fails loudly when absent."""
    for item in report["metrics"]["checks"]:
        if item["id"] == check_id:
            return item
    raise AssertionError(
        f"{check_id!r} not among {[i['id'] for i in report['metrics']['checks']]}")
