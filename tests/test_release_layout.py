"""The shipped public set loads, and the built dataset keeps answers away from agents.

The dataset builder writes two trees: ``agent/`` (task prompt, case facts, reference
views) is the only part an agent under test may see; ``scorer/`` holds the task
files, labels, recipes, cameras and fingerprints. These tests pin that split and
the absence of answer summaries in the files that ship with the repository.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # dataset_builder is run from the repository root, not installed
    sys.path.insert(0, str(REPO))

from code4scene.tasks import task as task_mod  # noqa: E402
from dataset_builder import build, catalog  # noqa: E402

PUBLIC = REPO / "benchmark" / "public"
I2S_CASES = sorted(PUBLIC.glob("image-to-scene/*/*/task.yaml"))
AGENT_FILES = {"prompt.txt", "case.json"}
SCORER_ONLY = {"task.yaml", "task.label.json", "recipe.json", "cameras.json", "input.fingerprint.json"}


def test_public_case_lists_match_the_task_directories():
    for setting, short in catalog.SETTINGS:
        listed = catalog.case_ids(short)
        assert listed, short
        for case_id in listed:
            assert (PUBLIC / setting / case_id / "task.yaml").is_file(), case_id


def test_shipped_files_carry_no_corruption_summary():
    for path in PUBLIC.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "corruption_contract" not in text and "authored_corruption" not in text, path


def test_outdoor_tasks_do_not_name_the_corruption():
    for path in PUBLIC.glob("image-to-scene/outdoor/*/task.yaml"):
        source = yaml.safe_load(path.read_text(encoding="utf-8"))["source"]
        for key in ("category", "mechanics", "intrinsic_tier_prior"):
            assert key not in source, (path, key)


@pytest.mark.parametrize("path", I2S_CASES, ids=lambda p: p.parent.name)
def test_image_to_scene_tasks_load_and_agree_on_the_ground_truth(path):
    task = task_mod.load(path)
    label = json.loads(path.with_suffix(".label.json").read_text(encoding="utf-8"))
    assert set(label) == {"canonical_map", "source_level", "gt_id", "schema_version"}
    assert task.ground_truth_map == task_mod.normalize_ground_truth_map(label["canonical_map"])
    assert task_mod.resolve_ground_truth_map(path, task.verifiers) == task.ground_truth_map


@pytest.mark.parametrize("path", sorted(PUBLIC.glob("text-to-scene/*/task.yaml")), ids=lambda p: p.parent.name)
def test_text_to_scene_tasks_load(path):
    assert task_mod.load(path).case_type == "prompt_to_scene"


def _package(tmp_path, case_ids):
    cases = catalog.load_cases(selected=case_ids)
    assert {c.case_id for c in cases} == set(case_ids)
    build.step_package(Namespace(dataset=tmp_path), cases)
    return cases


def test_package_writes_agent_and_scorer_trees(tmp_path):
    ids = ["a20-s01-extra-chair-removal", "comp-02-new-york-bench-reset-v2", "bazaar"]
    cases = _package(tmp_path, ids)
    for case in cases:
        agent = build.agent_dir(tmp_path, case)
        scorer = build.scorer_dir(tmp_path, case)
        assert {p.name for p in agent.iterdir()} == AGENT_FILES, case.case_id
        task = yaml.safe_load((case.directory / "task.yaml").read_text(encoding="utf-8"))
        assert (agent / "prompt.txt").read_text(encoding="utf-8") == task["inputs"]["prompt"]
        facts = json.loads((agent / "case.json").read_text(encoding="utf-8"))
        assert facts["id"] == case.case_id and facts["init_map"] == task["inputs"]["init_map"]
        assert "Code4SceneGT" not in json.dumps(facts)
        shipped = {p.name for p in case.directory.iterdir()}
        assert shipped <= {p.name for p in scorer.iterdir()}, case.case_id
        if case.is_i2s:
            assert SCORER_ONLY <= shipped
            assert facts["reference_views"] == task["source"]["reference_views"]
        else:
            assert (scorer / "requirements").is_dir() and facts["size_m"] == task["inputs"]["size_m"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["layout"] == {"agent": "agent/<setting>/<case>", "scorer": "scorer/<setting>/<case>"}
    assert not any(p.name in SCORER_ONLY for p in (tmp_path / "agent").rglob("*"))


def test_render_publishes_reference_views_into_the_agent_tree(tmp_path, monkeypatch):
    from PIL import Image

    cases = catalog.load_cases(selected=["a20-s01-extra-chair-removal"])

    def fake_run_job(args, name, group, render=False):
        for task in group:
            out = Path(task["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            for view in task["cameras"]["views"]:
                Image.new("RGB", (64, 36), (120, 120, 120)).save(out / f"{view['name']}.png")

    monkeypatch.setattr(build, "run_job", fake_run_job)
    build.step_render(Namespace(dataset=tmp_path, dry_run=False), cases, {})
    case = cases[0]
    for view in case.cameras["views"]:
        assert (build.agent_dir(tmp_path, case) / view["publish"]["file"]).is_file()
        assert (build.raw_render_dir(tmp_path, case) / f"{view['name']}.png").is_file()
    assert not (tmp_path / "scorer").exists()
