"""One editor export per episode, not one per verifier.

Invisible offline and expensive on real hardware: `verifiers.run` built a
fresh Context — and therefore a fresh cache — for every verifier it ran, so
nothing could ever hit the cache. On a live 243-Actor level each verifier that
wanted the scene re-exported it, at 24 seconds a time, nineteen times over.

It is also a correctness property. The new physical and semantic composites
project several leaves from one Candidate export; per-context caching would
quietly let those leaves describe different scene states.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from code4scene.evaluation import contracts, verifiers
from code4scene.tasks import task as task_mod

IDS = {"task_bundle_id": "b", "episode_id": "e"}


def write_task(tmp_path: Path, kinds: list[str]) -> Path:
    path = tmp_path / "task.yaml"
    specs = []
    for kind in kinds:
        spec = {"name": kind}
        if kind == "scene_diff":
            spec["ground_truth"] = "/Game/GT/Canonical"
        specs.append(spec)
    path.write_text(yaml.safe_dump({
        "id": "cache-test", "kind": "scene_generation",
        "inputs": {"prompt": "p", "init_map": "/Game/Maps/empty"}, "assets": {"packs": ["Maps"]},
        "verifiers": specs}))
    return path


def test_every_verifier_shares_one_cache(tmp_path):
    seen: list[int] = []

    def spy(context):
        seen.append(id(context.cache))
        context.cache["scene"] = context.cache.get("scene", 0) + 1
        kind = context.spec["name"]
        status = contracts.VALID if kind == "candidate_integrity" else contracts.MEASURED
        return {"report_id": kind, "task_bundle_id": "b", "episode_id": "e",
                "status": status,
                "score": None if status == contracts.VALID else 1.0,
                "metrics": {},
                "evidence": {"exports": context.cache["scene"]},
                "artifacts": {}, "probes_used": ()}

    kinds = [
        "candidate_integrity",
        "physical_safety",
        "scene_diff",
    ]
    original = {kind: verifiers.REGISTRY[kind] for kind in kinds}
    verifiers.REGISTRY.update({kind: spy for kind in kinds})
    try:
        reports = verifiers.run(task_mod.load(write_task(
            tmp_path, ["scene_diff"]
        )),
                                {"metrics": {"actors": 1}}, IDS)
    finally:
        verifiers.REGISTRY.update(original)

    assert [value["report_id"] for value in reports] == kinds
    assert len(seen) == 3
    assert len(set(seen)) == 1, (
        "each verifier got its own cache, so an episode re-exports the level "
        "once per verifier and two verifiers reading 'one' comparison read two")
    assert [r["evidence"]["exports"] for r in reports] == [1, 2, 3]
