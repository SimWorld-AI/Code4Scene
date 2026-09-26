"""Synthetic task files for tests (no benchmark case content).

The release tests never read the benchmark's own task files: those belong to
the dataset and can change. These writers produce the smallest task YAML the
loader and the verifiers' evidence planners accept for each setting.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def image_to_scene(tmp_path: Path, environment: str, *, task_id: str | None = None) -> Path:
    """An image-to-scene repair task for ``environment`` ('indoor' or 'outdoor')."""

    task_id = task_id or f"synthetic-{environment}-repair"
    data = {
        "id": task_id,
        "kind": "scene_repair",
        "case_type": "image_to_scene",
        "scene_environment": environment,
        "inputs": {
            "prompt": "Repair the current scene so it matches the provided reference "
                      "images. Make only the minimum changes needed.",
            "init_map": f"/Game/_SceneRepairInputs/synthetic/{task_id}/input_v1",
            "budget": {"tool_calls": 80, "wall_minutes": 20},
        },
        "verifiers": [{
            "name": "gt_repair",
            "ground_truth": f"/Game/_SceneRepairGT/synthetic/{task_id}/gt_v1",
            "visual_semantic_diff": {"caption_diff": True, "paired_visual": True},
        }],
        "source": {"pack": "Synthetic Pack"},
        "assets": {"packs": ["SyntheticPack", "_SceneRepairInputs"]},
        "status": "ready",
    }
    path = tmp_path / environment / f"{task_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def text_to_scene(tmp_path: Path, *, task_id: str = "synthetic-t2s") -> Path:
    data = {
        "id": task_id,
        "kind": "scene_generation",
        "inputs": {"prompt": "A small harbor with three boats and a lighthouse.",
                   "init_map": "/Game/SceneBench/BlankStage",
                   "budget": {"tool_calls": 80, "wall_minutes": 20}},
        "verifiers": [{"name": "overview_prompt_alignment"}],
        "assets": {"packs": ["SyntheticPack"]},
    }
    path = tmp_path / f"{task_id}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
