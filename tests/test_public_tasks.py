"""Every shipped task loads and plans its scoring without an editor."""

from __future__ import annotations

from pathlib import Path

import pytest

from code4scene.evaluation import verifiers
from code4scene.evaluation.evaluation_policy import load_evaluation_policy
from code4scene.tasks import task as task_mod

PUBLIC = Path(__file__).resolve().parents[1] / "benchmark" / "public"
TASKS = sorted(PUBLIC.rglob("task.yaml"))


def test_the_public_set_has_tasks():
    assert TASKS


@pytest.mark.parametrize("path", TASKS, ids=lambda p: str(p.parent.relative_to(PUBLIC)))
def test_every_shipped_task_loads_and_plans(path):
    task = task_mod.load(path)
    names = [spec.get("name") for spec in verifiers._canonical_specs(task)]
    assert names[0] == "candidate_integrity"
    assert set(names) <= set(verifiers.kinds())
    verifiers.render_evidence_requests(task)
    policy = load_evaluation_policy(task)
    assert isinstance(policy.evidence(), dict)
    if task.case_type == "image_to_scene":
        assert policy.source_snapshot is not None
