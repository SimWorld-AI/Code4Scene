"""Read the public task definitions shipped under ``benchmark/public``."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "benchmark"
PUBLIC = BENCHMARK / "public"
SETTINGS = (
    ("text-to-scene", "t2s"),
    ("image-to-scene/indoor", "indoor"),
    ("image-to-scene/outdoor", "outdoor"),
)
BLANK_STAGE = "/Game/SceneBench/BlankStage"
BUILDER_ROOTS = ("Code4SceneInputs", "Code4SceneGT")
GAME_PATH = re.compile(r"/Game/[A-Za-z0-9_./-]+")


@dataclass
class Case:
    case_id: str
    setting: str
    directory: Path
    task_text: str
    recipe: dict | None = None
    cameras: dict | None = None
    packs: list = field(default_factory=list)

    @property
    def is_i2s(self) -> bool:
        return self.setting != "text-to-scene"

    @property
    def scene_id(self) -> str | None:
        return self.recipe.get("scene_id") if self.recipe else None


def _packs_from_task(text: str) -> list[str]:
    """Extract ``assets.packs`` without requiring PyYAML."""

    packs, inside, indent = [], False, None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("packs:"):
            inline = stripped[len("packs:"):].strip()
            if inline.startswith("["):
                return [p.strip().strip("'\"") for p in inline.strip("[]").split(",") if p.strip()]
            inside, indent = True, len(line) - len(line.lstrip())
            continue
        if inside:
            if stripped.startswith("- "):
                packs.append(stripped[2:].strip().strip("'\""))
            elif stripped and (len(line) - len(line.lstrip())) <= indent:
                break
    return packs


def case_ids(setting_short: str) -> list[str]:
    path = BENCHMARK / f"public-{setting_short}-cases.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_case(setting: str, case_id: str) -> Case:
    directory = PUBLIC / setting / case_id
    task = directory / "task.yaml"
    if not task.exists():
        raise FileNotFoundError(f"no task definition for {case_id} ({task})")
    text = task.read_text()
    case = Case(case_id=case_id, setting=setting, directory=directory, task_text=text,
                packs=_packs_from_task(text))
    if (directory / "recipe.json").exists():
        case.recipe = json.loads((directory / "recipe.json").read_text())
        case.cameras = json.loads((directory / "cameras.json").read_text())
    return case


def load_cases(selected: list[str] | None = None, settings: list[str] | None = None) -> list[Case]:
    cases = []
    for setting, short in SETTINGS:
        if settings and short not in settings and setting not in settings:
            continue
        for case_id in case_ids(short):
            if selected and case_id not in selected:
                continue
            directory = PUBLIC / setting / case_id
            if not (directory / "task.yaml").exists():
                # Listed but not (yet) generated; reported by the pack check.
                cases.append(Case(case_id=case_id, setting=setting, directory=directory, task_text=""))
                continue
            cases.append(load_case(setting, case_id))
    return cases


def load_scene(scene_id: str) -> dict[str, Any]:
    return json.loads((PUBLIC / "scenes" / scene_id / "scene.json").read_text())


def load_expected_gt(scene_id: str) -> dict[str, Any]:
    return json.loads((PUBLIC / "scenes" / scene_id / "gt.fingerprint.json").read_text())


def load_input_delta(case: Case) -> dict[str, Any]:
    return json.loads((case.directory / "input.fingerprint.json").read_text())


def palette_assets(case: Case) -> list[str]:
    """/Game asset paths listed in a text-to-scene prompt's asset palette."""

    text = case.task_text
    try:
        import yaml  # type: ignore
        text = str((yaml.safe_load(text) or {}).get("inputs", {}).get("prompt", ""))
    except Exception:
        pass
    seen = []
    for match in GAME_PATH.findall(text):
        path = match.rstrip(".,;")
        if path not in seen and path != BLANK_STAGE:
            seen.append(path)
    return seen


def pack_listing() -> dict[str, dict]:
    """Optional: map content roots to Fab listings from ``benchmark/packs.yaml``."""

    path = BENCHMARK / "packs.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
    listing = {}
    entries = data.get("packs") if isinstance(data, dict) else data
    if isinstance(entries, dict):
        entries = [dict(v, key=k) for k, v in entries.items() if isinstance(v, dict)]
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        roots = entry.get("folder") or entry.get("content_roots") or entry.get("roots") or []
        if isinstance(roots, str):
            roots = [roots]
        for root in roots:
            listing[str(root).strip("/").split("/")[-1]] = entry
    return listing
