"""Evidence bundles: everything needed to score one case offline.

A bundle is a directory holding a ``bundle.json`` manifest plus the evidence
files it references. All paths are relative to the bundle directory; absolute
paths and ``..`` components are rejected. Every referenced file is listed with
its SHA-256 in ``files`` and verified on load. JSON evidence may be stored
gzip-compressed (``*.json.gz``).

Manifest (``schema_version: code4scene.bundle.v1``)::

    {
      "schema_version": "code4scene.bundle.v1",
      "task": {"id": "<case id>", "setting": "text-to-scene"
               | "image-to-scene/indoor" | "image-to-scene/outdoor",
               "sha256": "<sha256 of the task.yaml scored against, optional>"},
      "run": {"model": "<label>", "submission": "submitted" | "missing",
              "candidate_sha256": "<saved-scene hash, optional>",
              "batch_status": "complete", "authoritative": true},
      "candidate_integrity": {"status": "valid" | "invalid",
                              "leaves": {"candidate_snapshot_integrity": "valid",
                                         "content_and_dependency_parity": "valid",
                                         "asset_library_manifest_parity": "valid"},
                              "failure_reason": null},
      "scenes": {"candidate": "scenes/candidate.scene.json",
                 "input": "scenes/input.scene.json",            # image-to-scene
                 "ground_truth": "scenes/ground_truth.scene.json"},  # image-to-scene
      "physics": {"report": "physics/physical_safety.json"},
      "semantic": {"decisions": "semantic/decisions.json",      # text-to-scene:
                                              # {"report_status", "decisions": [...]}
                   "stage3_plan": "semantic/stage3_plan.json"}, # optional, VLM replay
      "overview": {"judgement": "overview/judgement.json",      # optional, recorded
                   "views": ["renders/overview/view_1.png", ...]},  # four RGB views
      "renders": [{"role": "overview", "view": "view_1", "channel": "rgb",
                   "path": "renders/overview/view_1.png"}],
      "files": {"<relative path>": "<sha256>"}
    }

What each part is for:

* ``scenes`` -- independently exported scene snapshots (actor class, asset,
  transform, bounds, materials, recorded properties). Repair F1 and the
  text-to-scene support check are computed from them.
* ``candidate_integrity`` -- the recorded verdicts of the three integrity
  checks. Dependency and asset-library parity need the editor and the
  content packs; offline scoring re-checks only the snapshot schema and the
  task's minimum actor count and otherwise uses the recorded verdicts.
* ``physics.report`` -- the recorded ``physical_safety`` report. Solid
  penetration (and image-to-scene support) are engine measurements
  (collision bodies and native minimum-translation depth) and cannot be
  recomputed from a snapshot.
* ``semantic.decisions`` -- the frozen per-requirement decisions of Detailed
  Alignment with the stage that resolved each (structured Stage 1/2 or visual
  Stage 3). ``semantic.stage3_plan`` optionally carries the recorded Stage 3
  request schedule so visual decisions can be re-judged by a VLM.
* ``overview`` -- the four overview renders and, optionally, the recorded
  judge outputs (dimension scores, structural-integrity score, severe-defect
  eligibility).
* ``run`` -- the subset of the harness run record that scoring reads.

A bundle holds third-party scene content (actor lists of the scored level)
when built from real runs. Such bundles are evaluation artifacts and are not
part of this repository.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .protocol.constants import SETTINGS

SCHEMA_VERSION = "code4scene.bundle.v1"
MANIFEST = "bundle.json"
SCENE_ROLES = ("candidate", "input", "ground_truth")
INTEGRITY_LEAVES = (
    "candidate_snapshot_integrity",
    "content_and_dependency_parity",
    "asset_library_manifest_parity",
)
_TOP_KEYS = {"schema_version", "task", "run", "candidate_integrity", "scenes", "physics",
             "semantic", "overview", "renders", "files", "notes"}
_OBJECT_KEYS = ("run", "candidate_integrity", "scenes", "physics", "semantic", "overview")
#: A scene snapshot is tens of megabytes; a compressed file that expands past
#: this is not one.
MAX_JSON_BYTES = 1 << 30


class BundleError(ValueError):
    """The bundle is malformed, incomplete, or its content changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BundleError(f"{field} must be a non-empty relative path")
    text = value.replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text.startswith("~"):
        raise BundleError(f"{field} must stay inside the bundle: {value!r}")
    return str(path)


def read_json(path: Path) -> Any:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise BundleError(f"{path.name} expands past {MAX_JSON_BYTES} bytes")
    else:
        raw = path.read_bytes()
    try:
        return json.loads(raw)
    except RecursionError as exc:
        raise BundleError(f"{path.name} is nested too deeply to read") from exc


def _object(value: Any, what: str) -> Any:
    """A JSON document that must be an object, or None when it is absent."""

    if value is not None and not isinstance(value, Mapping):
        raise BundleError(f"{what} must be a JSON object")
    return value


@dataclass(frozen=True)
class Bundle:
    root: Path
    manifest: dict[str, Any]

    # -- identity -----------------------------------------------------------
    @property
    def task_id(self) -> str:
        return str(self.manifest["task"]["id"])

    @property
    def setting(self) -> str:
        return str(self.manifest["task"]["setting"])

    @property
    def is_t2s(self) -> bool:
        return self.setting == "text-to-scene"

    @property
    def run(self) -> dict[str, Any]:
        return dict(self.manifest.get("run") or {})

    # -- files --------------------------------------------------------------
    def path(self, relative: str) -> Path:
        return self.root / _relative(relative, "path")

    def json(self, relative: str | None) -> Any:
        return None if not relative else read_json(self.path(relative))

    def scene(self, role: str) -> dict[str, Any] | None:
        return self.json((self.manifest.get("scenes") or {}).get(role))

    def physics_report(self) -> dict[str, Any] | None:
        return _object(self.json((self.manifest.get("physics") or {}).get("report")),
                       "physics.report")

    def decisions_document(self) -> dict[str, Any] | None:
        value = self.json((self.manifest.get("semantic") or {}).get("decisions"))
        if isinstance(value, list):
            return {"report_status": "measured", "decisions": value}
        return _object(value, "semantic.decisions")

    def decisions(self) -> list[dict[str, Any]] | None:
        document = self.decisions_document()
        return None if document is None else list(document.get("decisions") or [])

    def stage3_plan(self) -> dict[str, Any] | None:
        return _object(self.json((self.manifest.get("semantic") or {}).get("stage3_plan")),
                       "semantic.stage3_plan")

    def overview_judgement(self) -> dict[str, Any] | None:
        return _object(self.json((self.manifest.get("overview") or {}).get("judgement")),
                       "overview.judgement")

    def overview_views(self) -> list[Path]:
        return [self.path(p) for p in (self.manifest.get("overview") or {}).get("views") or ()]

    @property
    def integrity(self) -> dict[str, Any]:
        return dict(self.manifest.get("candidate_integrity") or {})


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Check the manifest shape; return it unchanged."""

    if not isinstance(manifest, Mapping):
        raise BundleError("bundle.json must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(
            f"unsupported bundle schema {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}")
    unknown = sorted(set(manifest) - _TOP_KEYS)
    if unknown:
        raise BundleError(f"unknown bundle.json keys: {unknown}")
    for key in _OBJECT_KEYS:
        if manifest.get(key) and not isinstance(manifest[key], Mapping):
            raise BundleError(f"{key} must be a JSON object")
    renders = manifest.get("renders")
    if renders and (not isinstance(renders, list)
                    or not all(isinstance(r, Mapping) for r in renders)):
        raise BundleError("renders must be a list of objects")
    views = (manifest.get("overview") or {}).get("views")
    if views and not isinstance(views, list):
        raise BundleError("overview.views must be a list of paths")
    task = manifest.get("task")
    if not isinstance(task, Mapping) or not task.get("id"):
        raise BundleError("task.id is required")
    if task.get("setting") not in SETTINGS:
        raise BundleError(f"task.setting must be one of {SETTINGS}")
    run = manifest.get("run") or {}
    submission = run.get("submission", "submitted")
    if submission not in {"submitted", "missing"}:
        raise BundleError("run.submission must be 'submitted' or 'missing'")
    integrity = manifest.get("candidate_integrity") or {}
    if submission == "submitted":
        if integrity.get("status") not in {"valid", "invalid"}:
            raise BundleError("candidate_integrity.status must be 'valid' or 'invalid'")
        scenes = manifest.get("scenes") or {}
        if integrity.get("status") == "valid":
            # Repair F1 needs all three snapshots. A text-to-scene bundle
            # without the candidate snapshot falls back to the recorded
            # support measurement (and the recorded integrity verdict).
            needed = [] if task["setting"] == "text-to-scene" else [
                "candidate", "input", "ground_truth"]
            missing = [r for r in needed if not scenes.get(r)]
            if missing:
                raise BundleError(f"a valid {task['setting']} case needs scenes: {missing}")
            if not (manifest.get("physics") or {}).get("report"):
                raise BundleError("a valid case needs physics.report")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise BundleError("files must map every referenced path to its sha256")
    for key, digest in files.items():
        _relative(key, "files key")
        if not isinstance(digest, str) or len(digest) != 64:
            raise BundleError(f"files[{key!r}] must be a sha256 hex digest")
    for rel in referenced_paths(manifest):
        if rel not in files:
            raise BundleError(f"{rel} is referenced but not listed in files")
    return dict(manifest)


def referenced_paths(manifest: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for role in SCENE_ROLES:
        value = (manifest.get("scenes") or {}).get(role)
        if value:
            paths.append(_relative(value, f"scenes.{role}"))
    for section, keys in (("physics", ("report",)), ("semantic", ("decisions", "stage3_plan")),
                          ("overview", ("judgement",))):
        for key in keys:
            value = (manifest.get(section) or {}).get(key)
            if value:
                paths.append(_relative(value, f"{section}.{key}"))
    for i, value in enumerate((manifest.get("overview") or {}).get("views") or ()):
        paths.append(_relative(value, f"overview.views[{i}]"))
    for i, render in enumerate(manifest.get("renders") or ()):
        paths.append(_relative((render or {}).get("path"), f"renders[{i}].path"))
    return list(dict.fromkeys(paths))


def load(root: str | Path, *, verify: bool = True) -> Bundle:
    """Load and validate a bundle directory (hashes are checked by default)."""

    root = Path(root)
    manifest_path = root / MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text())
    except OSError as exc:
        raise BundleError(f"cannot read {manifest_path}: {exc}") from exc
    except ValueError as exc:
        raise BundleError(f"{manifest_path} is not valid JSON: {exc}") from exc
    manifest = validate_manifest(manifest)
    for rel, digest in manifest["files"].items():
        path = root / rel
        if not path.is_file():
            raise BundleError(f"missing bundle file: {rel}")
        if verify and sha256_file(path) != digest:
            raise BundleError(f"bundle file changed since it was packed: {rel}")
    for rel in _stage3_frame_paths(root, manifest):
        if rel not in manifest["files"]:
            raise BundleError(f"{rel} is used by the Stage 3 plan but not listed in files")
    return Bundle(root=root, manifest=manifest)


def _stage3_frame_paths(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    """The frames a recorded Stage 3 plan replays, which the manifest must hash too."""

    relative = (manifest.get("semantic") or {}).get("stage3_plan")
    if not relative:
        return []
    plan = _object(read_json(root / _relative(relative, "semantic.stage3_plan")),
                   "semantic.stage3_plan")
    frames = _object((plan or {}).get("frames"), "stage3_plan.frames") or {}
    return [_relative((_object(spec, f"stage3_plan.frames[{key!r}]") or {}).get("path"),
                      f"stage3_plan.frames[{key!r}].path")
            for key, spec in frames.items()]


# ---------------------------------------------------------------------------
# Writing bundles
# ---------------------------------------------------------------------------


class BundleWriter:
    """Assemble a bundle directory file by file, then write the manifest."""

    def __init__(self, root: str | Path, *, link: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.link = link
        self.files: dict[str, str] = {}

    def add_json(self, relative: str, value: Any, *, compress: bool = False) -> str:
        relative = _relative(relative + (".gz" if compress else ""), "path")
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, ensure_ascii=False, indent=None if compress else 2).encode()
        target.write_bytes(gzip.compress(raw, mtime=0) if compress else raw)
        self.files[relative] = sha256_file(target)
        return relative

    def add_file(self, relative: str, source: str | Path) -> str:
        relative = _relative(relative, "path")
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if self.link:
            os.symlink(Path(source).resolve(), target)
        else:
            shutil.copyfile(source, target)
        self.files[relative] = sha256_file(target)
        return relative

    def finish(self, manifest: Mapping[str, Any]) -> Bundle:
        document = {"schema_version": SCHEMA_VERSION, **manifest, "files": dict(self.files)}
        validate_manifest(document)
        (self.root / MANIFEST).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        return load(self.root)


# ---------------------------------------------------------------------------
# From a saved verifier result
# ---------------------------------------------------------------------------


def _setting_from_result(result: Mapping[str, Any]) -> str:
    track = str(result.get("benchmark_track") or "")
    if track == "text_to_scene":
        return "text-to-scene"
    if track.startswith("image_to_scene"):
        env = str(result.get("scene_environment") or track.rsplit("_", 1)[-1])
        return f"image-to-scene/{env}"
    # Older results carry no track field: infer it from the verifier reports.
    reports = _reports(result)
    if "gt_repair" in reports:
        env = str(result.get("scene_environment") or "")
        if env not in {"indoor", "outdoor"}:
            task_file = str(result.get("task_file") or "").replace("\\", "/")
            env = next((e for e in ("indoor", "outdoor") if f"/{e}/" in task_file), "")
        if env:
            return f"image-to-scene/{env}"
    elif "semantic_requirements" in reports or "overview_prompt_alignment" in reports:
        return "text-to-scene"
    raise BundleError("cannot infer the setting of this result; pass --setting")


def _reports(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r.get("report_id")): r for r in result.get("reports") or ()
            if isinstance(r, Mapping)}


def integrity_from_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    leaves = {}
    for leaf in (report.get("metrics") or {}).get("leaf_results") or ():
        if isinstance(leaf, Mapping):
            leaves[str(leaf.get("leaf_id"))] = str(leaf.get("status"))
    return {"status": "valid" if report.get("status") == "valid" else "invalid",
            "recorded_status": report.get("status"), "leaves": leaves,
            "failure_reason": report.get("failure_reason")}


def decisions_from_report(report: Mapping[str, Any] | None) -> list[dict[str, Any]] | None:
    """Frozen per-requirement decisions from a saved ``semantic_requirements`` report."""

    from .evaluation import scalar_score

    metrics = (report or {}).get("metrics") or {}
    rows = metrics.get("semantic_requirement_aggregation")
    if not rows:
        return None
    resolved_by = {}
    for check in metrics.get("checks") or ():
        observed = check.get("observed") or {}
        node = str(check.get("id") or "").removeprefix("atomic_")
        resolved_by[node] = observed.get("resolved_by")
    out = []
    for row in scalar_score.without_intervals(rows):
        out.append({**row, "resolved_by": row.get("resolved_by") or resolved_by.get(
            str(row.get("node_id")))})
    return out


def overview_judgement_from_report(report: Mapping[str, Any] | None) -> dict[str, Any] | None:
    report = report or {}
    metrics = report.get("metrics") or {}
    if report.get("status") != "measured" or not metrics.get("dimensions"):
        return None
    return {
        "status": "measured",
        "dimension_scores": {k: (v or {}).get("score")
                             for k, v in metrics["dimensions"].items()},
        "structural_integrity_score": metrics.get("structural_integrity_score"),
        "severe_structural_cap_eligible": bool(metrics.get("severe_structural_cap_eligible")),
        "recorded_score": report.get("score"),
    }


def from_result(
    result_path: str | Path,
    out_dir: str | Path,
    *,
    scenes: Mapping[str, str | Path] | None = None,
    semantic_report: Mapping[str, Any] | None = None,
    model: str | None = None,
    setting: str | None = None,
    link: bool = False,
    compress: bool = False,
) -> Bundle:
    """Build a bundle from a saved verifier ``result.json`` and its scene snapshots.

    ``scenes`` maps roles to snapshot files; when omitted, the unique
    ``scene_evidence/*/`` directory next to the result is used.
    ``semantic_report`` replaces the result's Semantic report (for results
    whose Semantic verifier was re-run separately).
    """

    result_path = Path(result_path)
    result = read_json(result_path)
    reports = _reports(result)
    if setting is not None and setting not in SETTINGS:
        raise BundleError(f"setting must be one of {SETTINGS}")
    setting = setting or _setting_from_result(result)
    if scenes is None:
        found = sorted((result_path.parent / "scene_evidence").glob("*/candidate.scene.json"))
        scenes = {}
        if len(found) == 1:
            for role in SCENE_ROLES:
                candidate = found[0].with_name(f"{role}.scene.json")
                if candidate.is_file():
                    scenes[role] = candidate
    writer = BundleWriter(out_dir, link=link and not compress)
    manifest: dict[str, Any] = {
        "task": {"id": str(result.get("task_id")), "setting": setting},
        "run": {"model": model, "submission": "submitted",
                "candidate_sha256": result.get("candidate_sha256"),
                "batch_status": result.get("batch_status"),
                "authoritative": result.get("authoritative")},
        "candidate_integrity": integrity_from_report(reports.get("candidate_integrity")),
    }
    scene_entries = {}
    for role, source in (scenes or {}).items():
        if compress:
            scene_entries[role] = writer.add_json(
                f"scenes/{role}.scene.json", read_json(Path(source)), compress=True)
        else:
            suffix = ".gz" if str(source).endswith(".gz") else ""
            scene_entries[role] = writer.add_file(f"scenes/{role}.scene.json{suffix}", source)
    if scene_entries:
        manifest["scenes"] = scene_entries
    if reports.get("physical_safety"):
        manifest["physics"] = {
            "report": writer.add_json("physics/physical_safety.json", reports["physical_safety"])}
    if setting == "text-to-scene":
        semantic = semantic_report or reports.get("semantic_requirements")
        decisions = decisions_from_report(semantic)
        if decisions is not None:
            manifest["semantic"] = {"decisions": writer.add_json(
                "semantic/decisions.json",
                {"report_status": (semantic or {}).get("status"), "decisions": decisions})}
        judgement = overview_judgement_from_report(reports.get("overview_prompt_alignment"))
        overview: dict[str, Any] = {}
        if judgement is not None:
            overview["judgement"] = writer.add_json("overview/judgement.json", judgement)
        views = _overview_views(result)
        if views:
            overview["views"] = [writer.add_file(f"renders/overview/view_{i + 1}.png", path)
                                 for i, path in enumerate(views)]
            manifest["renders"] = [{"role": "overview", "view": f"view_{i + 1}",
                                    "channel": "rgb", "path": rel}
                                   for i, rel in enumerate(overview["views"])]
        if overview:
            manifest["overview"] = overview
    return writer.finish(manifest)


def _overview_views(result: Mapping[str, Any]) -> list[Path]:
    """The four overview frames a measured overview report froze, if on disk."""

    from .evaluation import offline_overview_prompt_alignment as overview

    try:
        frames = overview._formal_overview_frames(result)
    except overview.OverviewAlignmentError:
        return []
    return [Path(str(frame["path"])) for frame in frames or ()]


__all__ = [
    "Bundle", "BundleError", "BundleWriter", "MANIFEST", "SCHEMA_VERSION",
    "decisions_from_report", "from_result", "integrity_from_report", "load",
    "overview_judgement_from_report", "read_json", "referenced_paths", "validate_manifest",
]
