"""Acceptance tests for the canonical verifier responsibility boundaries."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from code4scene.evaluation import atomic_registry, composite, contracts, verifiers
from code4scene.evaluation.context import Context
from code4scene.evaluation.evaluation_policy import (
    FrozenEvaluationPolicy,
    default_policy,
)
from code4scene.evaluation.gt_geometry_compare import compare_detailed
from code4scene.evaluation.requirement_graph.bundle import (
    BindingSourceKind,
    FrozenVerificationBundle,
    RequirementEvaluationStatus,
)
from code4scene.evaluation.requirement_graph.legacy_migration import (
    migrate_legacy_contract,
)
from code4scene.evaluation.selection import select_candidate_actors
from code4scene.evaluation.scene_diff import diff_scenes
from code4scene.evaluation.verifiers import (
    physical_safety,
    scene_diff as scene_diff_verifier,
    semantic_requirements,
    source_preservation,
    visual_as_judge as visual_as_judge_verifier,
)
from code4scene.tasks.task import TaskError, load as load_task
from code4scene.tasks.verifier_schema import (
    AUTOMATIC_POLICY_KINDS,
    CANONICAL_VERIFIER_KINDS,
)


IDS = {"task_bundle_id": "bundle", "episode_id": "episode"}


def test_canonical_quality_modules_do_not_emit_pass_fail_statuses():
    root = Path(verifiers.__file__).resolve().parent
    patterns = (
        re.compile(r"""["']status["']\s*:\s*contracts\.(?:PASS|FAIL)"""),
        re.compile(r"""["']status["']\s*:\s*["'](?:pass|fail)["']"""),
        re.compile(r"""\bstatus\s*=\s*contracts\.(?:PASS|FAIL)"""),
    )
    offenders = []
    for path in [root / f"{kind}.py" for kind in verifiers.kinds()]:
        source = path.read_text()
        if any(pattern.search(source) for pattern in patterns):
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == [], (
        "canonical verifier modules must publish measured numeric quality "
        f"results, not PASS/FAIL statuses: {offenders}"
    )


def _task(
    tmp_path: Path,
    *,
    verifiers_: list[dict] | None = None,
    prompt: str = "Keep two props separate.",
    kind: str = "scene_generation",
    data: dict | None = None,
) -> SimpleNamespace:
    path = tmp_path / "task.yaml"
    path.write_text("id: reconstruction-test\n")
    payload = dict(data or {})
    payload.setdefault("verifiers", verifiers_ or [])
    return SimpleNamespace(
        id="reconstruction-test",
        path=path,
        prompt=prompt,
        kind=kind,
        case_type="prompt_to_scene",
        data=payload,
        verifiers=payload["verifiers"],
    )


def _actor(
    label: str,
    x: float,
    *,
    category: str | None = None,
    actor_class: str = "/Script/Engine.StaticMeshActor",
    material_path: str | None = None,
    dynamic_material: bool = False,
) -> dict:
    value = {
        "label": label,
        "actor_path": f"/Game/Test.Test:PersistentLevel.{label}",
        "class": actor_class,
        "asset_path": f"/Game/Test/{label}.{label}",
        "actor_tags": [],
        "transform": {
            "location_cm": [x, 0.0, 0.0],
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "bounds": {
            "origin_cm": [x, 0.0, 0.0],
            "extent_cm": [10.0, 10.0, 10.0],
        },
        "properties": {},
    }
    if category is not None:
        value["asset_category"] = category
    if material_path is not None:
        value["component_material_slots"] = [
            {
                "component_identity": "Mesh|StaticMeshComponent",
                "slot_index": 0,
                "material_path": material_path,
                "is_dynamic": dynamic_material,
            }
        ]
    return value


def _scene(*actors: dict) -> dict:
    return {
        "actor_count": len(actors),
        "actors": list(actors),
        "map_path": "/Game/Test",
    }


def _report(report_id: str, status: str, score: float | None = None) -> dict:
    value = {
        **contracts.base(report_id, IDS),
        "status": status,
        "score": score,
        "metrics": {},
        "evidence": {},
        "artifacts": {},
        "probes_used": (),
    }
    if status not in {contracts.PASS, contracts.MEASURED, contracts.VALID}:
        value["failure_reason"] = f"{report_id} did not pass"
    return value


def test_registry_exposes_exactly_the_current_public_surface():
    assert set(verifiers.kinds()) == set(CANONICAL_VERIFIER_KINDS)
    assert set(verifiers.REGISTRY) == set(CANONICAL_VERIFIER_KINDS)
    assert set(verifiers.CLASSES) == set(CANONICAL_VERIFIER_KINDS)


def test_composite_owns_canonical_leaf_addresses_and_artifact_keys(tmp_path):
    context = Context(record={}, task=object(), ids=IDS)
    leaf = {
        **_report("gt_caption_similarity", contracts.PASS),
        "leaf_id": "independent_caption_embedding_distance",
        "artifacts": {"comparison": str(tmp_path / "comparison.json")},
    }

    report = composite.composite_report("caption_similarity", context, [leaf])
    child = report["metrics"]["leaf_results"][0]

    assert child["leaf_id"] == "independent_caption_embedding_distance"
    assert child["report_id"] == (
        "caption_similarity.independent_caption_embedding_distance"
    )
    assert set(report["artifacts"]) == {
        "independent_caption_embedding_distance.comparison"
    }
    assert "gt_caption_similarity" not in str(report)


def test_composite_scores_available_children_and_reports_missing_coverage():
    context = Context(record={}, task=object(), ids=IDS)
    measured = {
        **_report("measured_leaf", contracts.MEASURED),
        "leaf_id": "measured_leaf",
        "score": 0.75,
    }
    unavailable = {
        **_report("missing_leaf", "not_evaluated"),
        "leaf_id": "missing_leaf",
        "score": None,
    }

    report = composite.composite_report(
        "partial_vector", context, [measured, unavailable]
    )

    assert report["status"] == contracts.MEASURED
    assert report["score"] == 0.75
    assert report["metrics"]["score_vector"] == {"measured_leaf": 0.75}
    assert report["metrics"]["score_coverage"] == 0.5
    assert report["metrics"]["incomplete_leaf_ids"] == ["missing_leaf"]


def test_composite_keeps_direct_lower_is_better_result_and_orients_parent():
    context = Context(record={}, task=object(), ids=IDS)
    direct_rate = {
        **_report("floating", contracts.MEASURED),
        "leaf_id": "floating",
        "score": 0.4,
        "metadata": {
            "score_direction": "lower_is_better",
            "result_semantics": "direct_measured_rate",
        },
    }

    report = composite.composite_report("physical_safety", context, [direct_rate])

    assert report["metrics"]["score_vector"] == {"floating": 0.4}
    assert report["metrics"]["aggregate_score_vector"] == {"floating": 0.4}
    assert report["metrics"]["normalized_aggregate_score_vector"] == {
        "floating": 0.6
    }
    assert report["metrics"]["leaf_score_directions"] == {
        "floating": "lower_is_better"
    }
    assert report["score"] == 0.6


def test_composite_can_require_a_complete_score_vector():
    context = Context(record={}, task=object(), ids=IDS)
    measured = {
        **_report("floating", contracts.MEASURED),
        "leaf_id": "floating",
        "score": 0.8,
    }
    unavailable = {
        **_report("solid_penetration", "not_evaluated"),
        "leaf_id": "solid_penetration",
        "score": None,
    }

    report = composite.composite_report(
        "physical_safety",
        context,
        [measured, unavailable],
        require_all_scored=True,
    )

    assert report["score"] is None
    assert report["metrics"]["requires_all_scored_leaves"] is True
    assert report["metrics"]["all_required_scores_present"] is False
    assert report["metrics"]["incomplete_leaf_ids"] == ["solid_penetration"]


def test_complete_vector_can_exclude_only_not_applicable_leaves():
    context = Context(record={}, task=object(), ids=IDS)
    measured = {
        **_report("floating", contracts.MEASURED),
        "leaf_id": "floating",
        "score": 0.2,
        "metadata": {"score_direction": "lower_is_better"},
    }
    not_applicable = {
        **_report("solid_penetration", "not_applicable"),
        "leaf_id": "solid_penetration",
        "score": None,
    }

    strict = composite.composite_report(
        "physical_safety",
        context,
        [measured, not_applicable],
        require_all_scored=True,
    )
    local = composite.composite_report(
        "physical_safety",
        context,
        [measured, not_applicable],
        require_all_scored=True,
        exclude_not_applicable_from_required_scores=True,
    )

    assert strict["score"] is None
    assert local["score"] == pytest.approx(0.8)
    assert local["metrics"]["all_required_scores_present"] is True
    assert local["metrics"]["incomplete_leaf_ids"] == []
    assert local["metrics"]["required_score_leaf_ids"] == ["floating"]
    assert local["metrics"]["excluded_not_applicable_leaf_ids"] == [
        "solid_penetration"
    ]

    unavailable = {
        **_report("solid_penetration", "not_evaluated"),
        "leaf_id": "solid_penetration",
        "score": None,
    }
    incomplete = composite.composite_report(
        "physical_safety",
        context,
        [measured, unavailable],
        require_all_scored=True,
        exclude_not_applicable_from_required_scores=True,
    )
    assert incomplete["score"] is None
    assert incomplete["metrics"]["incomplete_leaf_ids"] == [
        "solid_penetration"
    ]


def test_composite_retains_report_only_leaf_without_scoring_or_blocking_it():
    context = Context(record={}, task=object(), ids=IDS)
    measured = {
        **_report("measured_leaf", contracts.MEASURED),
        "leaf_id": "measured_leaf",
        "score": 0.8,
    }
    diagnostic = {
        **_report("diagnostic_leaf", contracts.ERROR),
        "leaf_id": "diagnostic_leaf",
        "score": 0.1,
        "contributes_to_aggregate": False,
        "score_role": "report_only",
    }

    report = composite.composite_report(
        "report_only_vector", context, [measured, diagnostic]
    )

    assert report["status"] == contracts.MEASURED
    assert report["score"] == 0.8
    assert report["metrics"]["score_vector"] == {
        "measured_leaf": 0.8,
        "diagnostic_leaf": 0.1,
    }
    assert report["metrics"]["aggregate_score_vector"] == {
        "measured_leaf": 0.8,
    }
    assert report["metrics"]["report_only_leaf_ids"] == ["diagnostic_leaf"]


def test_caption_atomic_is_native_continuous_metric():
    evaluator = atomic_registry.get("independent_caption_embedding_distance")
    descriptor = evaluator.descriptor

    assert descriptor.evaluator_id == (
        "caption.independent_caption_embedding_distance"
    )
    assert descriptor.unit == "cosine_distance"
    result = contracts.MetricResult(
        id=descriptor.evaluator_id,
        instance_id="caption:episode",
        metric_version=descriptor.evaluator_version,
        dimension=descriptor.dimension,
        applicability_policy=descriptor.applicability_policy,
        required_evidence=descriptor.required_evidence,
        applicable=True,
        status="measured",
        coverage=1.0,
        raw={"cosine_distance": 0.1},
        score=0.9,
        normalization_policy="clip_cosine_similarity_to_unit_interval",
        normalization_parameters={},
        calibration_status="direct_metric_not_human_calibrated",
        contributes_to_aggregate=True,
    )
    assert result.to_json_dict()["score"] == 0.9


def test_integrity_gate_runs_first_and_blocks_every_scene_dependent_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[str] = []

    def integrity(context: Context) -> dict:
        calls.append(context.spec["name"])
        return _report("candidate_integrity", contracts.ERROR)

    monkeypatch.setitem(
        verifiers.REGISTRY, "candidate_integrity", integrity
    )
    task = _task(tmp_path, verifiers_=[{"name": "semantic_requirements"}])
    reports = verifiers.run(task, {"metrics": {}}, IDS)

    expected = [*AUTOMATIC_POLICY_KINDS, "semantic_requirements"]
    assert [value["report_id"] for value in reports] == expected
    assert calls == ["candidate_integrity"]
    assert all(value["score"] is None for value in reports)
    assert all(
        value["evidence"].get("blocked_by") == "candidate_integrity"
        for value in reports[1:]
    )


def test_integrity_gate_allows_physics_when_only_visual_evidence_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[str] = []

    def integrity(context: Context) -> dict:
        report = _report("candidate_integrity", contracts.ERROR)
        report["evidence"] = {
            "gate_mode": "per_verifier_eligibility.v2",
            "verifier_eligibility": {
                "physical_safety": {"eligible": True},
                "semantic_requirements": {
                    "eligible": False,
                    "reason_code": "dependency_source_missing",
                    "source_leaf": "content_and_dependency_parity",
                    "reason": "a material resolved during generation is unavailable",
                },
            },
        }
        return report

    def physics(context: Context) -> dict:
        calls.append("physical_safety")
        return _report("physical_safety", contracts.MEASURED, 0.75)

    monkeypatch.setitem(verifiers.REGISTRY, "candidate_integrity", integrity)
    monkeypatch.setitem(verifiers.REGISTRY, "physical_safety", physics)
    task = _task(tmp_path, verifiers_=[{"name": "semantic_requirements"}])
    reports = verifiers.run(task, {"metrics": {}}, IDS)
    by_id = {value["report_id"]: value for value in reports}

    assert calls == ["physical_safety"]
    assert by_id["physical_safety"]["status"] == contracts.MEASURED
    semantic = by_id["semantic_requirements"]
    assert semantic["status"] == "not_evaluated"
    assert semantic["evidence"]["reason_code"] == "dependency_source_missing"


def test_frozen_physics_exemptions_are_real_selector_controls():
    profile = dict(default_policy(None).physics_profile)
    profile.update(
        excluded_classes=["CameraActor"],
        support_surface_classes=["FloorActor"],
        allow_floating_categories=["hanging_lamp"],
        allow_submerged_categories=["submarine"],
    )
    policy = FrozenEvaluationPolicy(
        policy_id="physics-exemptions",
        physics_profile=profile,
    )
    assertions = {
        value["primitive"]: value
        for value in policy.physics_contract("case")["assertions"]
    }
    ground = assertions["physics"]["target_selector"]
    solid = assertions["solid_penetration"]["target_selector"]
    assert set(assertions) == {"physics", "solid_penetration"}
    assert assertions["solid_penetration"][
        "adaptive_penetration_tolerance"
    ] is True
    assert assertions["solid_penetration"][
        "relative_penetration_tolerance_fraction"
    ] == 0.05
    assert assertions["solid_penetration"][
        "maximum_adaptive_penetration_tolerance_cm"
    ] == 50.0
    assert policy.leaf_spec(
        SimpleNamespace(id="case"), "solid_penetration"
    )["physics_measurement_mode"] == "solid_penetration"

    actors = [
        _actor("camera", 0.0, actor_class="CameraActor"),
        _actor("floor", 100.0, actor_class="FloorActor"),
        _actor("lamp", 200.0, category="hanging_lamp"),
        _actor("sub", 300.0, category="submarine"),
        _actor("chair", 400.0, category="chair"),
    ]
    assert [value["label"] for value in select_candidate_actors(actors, ground)] == [
        "sub",
        "chair",
    ]
    assert [value["label"] for value in select_candidate_actors(actors, solid)] == [
        "floor",
        "lamp",
        "sub",
        "chair",
    ]

    invalid = dict(profile)
    invalid["maximum_ground_gaap_cm"] = 5.0
    with pytest.raises(ValueError, match="unsupported fields"):
        FrozenEvaluationPolicy(policy_id="typo", physics_profile=invalid)

    invalid_cap = dict(profile)
    invalid_cap["maximum_adaptive_penetration_tolerance_cm"] = 4.0
    with pytest.raises(ValueError, match="must be >= maximum_penetration_cm"):
        FrozenEvaluationPolicy(policy_id="invalid-cap", physics_profile=invalid_cap)


def test_image_repair_policy_can_freeze_local_physics_without_changing_text_v3():
    text_policy = default_policy(None)
    local_profile = {
        **text_policy.physics_profile,
        "profile_id": "image-to-scene-edited-actors-physical-safety-v1",
        "selector": {"scope": "edited_actors"},
    }
    image_policy = FrozenEvaluationPolicy(
        policy_id="image-local",
        source_snapshot={"actors": []},
        physics_profile=local_profile,
    )

    text_contract = text_policy.physics_contract("text-case")
    image_contract = image_policy.physics_contract("image-case")

    assert {
        item["target_selector"]["scope"]
        for item in text_contract["assertions"]
    } == {"candidate_all"}
    assert {
        item["target_selector"]["scope"]
        for item in image_contract["assertions"]
    } == {"edited_actors"}
    assert text_policy.leaf_spec(
        SimpleNamespace(id="text-case"), "solid_penetration"
    )["physics_measurement_mode"] == "solid_penetration"
    assert "physics_measurement_mode" not in image_policy.leaf_spec(
        SimpleNamespace(id="image-case"), "solid_penetration"
    )

    lateral_profile = {
        **local_profile,
        "profile_id": "image-to-scene-edited-actors-physical-safety-v2",
        "support_model": "ground_or_lateral_v1",
    }
    lateral_policy = FrozenEvaluationPolicy(
        policy_id="image-local-v3",
        source_snapshot={"actors": []},
        physics_profile=lateral_profile,
    )
    lateral_ground = next(
        item for item in lateral_policy.physics_contract("image-case")["assertions"]
        if item["primitive"] == "physics"
    )
    assert lateral_ground["support_model"] == "ground_or_lateral_v1"
    assert all(
        "support_model" not in item
        for item in text_contract["assertions"]
    )


def test_image_tasks_use_code_owned_local_physics_policy_only(tmp_path):
    from synthetic_tasks import image_to_scene

    written = image_to_scene(tmp_path, "outdoor")
    with_policy = written.with_name("with-policy.yaml")
    with_policy.write_text(written.read_text() + "evaluation_policy: {}\n")
    with pytest.raises(TaskError, match="evaluation_policy"):
        load_task(with_policy)

    policy = default_policy(
        SimpleNamespace(case_type="image_to_scene", data={})
    )
    assert policy.policy_id == "image-to-scene-runtime-input-local-physics"
    assert policy.physics_profile["selector"] == {"scope": "edited_actors"}
    assert policy.physics_profile["profile_id"] == (
        "image-to-scene-edited-actors-physical-safety"
    )
    assert policy.physics_profile["support_model"] == "ground_or_lateral_v1"


def test_edit_physical_safety_invokes_only_floating_and_collision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    source = _scene(_actor("source", 0.0))
    policy = FrozenEvaluationPolicy(
        policy_id="edit-physics",
        source_snapshot=source,
        edit_scope={},
        physics_profile=default_policy(None).physics_profile,
    )
    task = _task(
        tmp_path,
        kind="scene_edit",
        data={
            "evaluation_policy": policy.to_dict(),
        },
    )
    calls: list[str] = []

    class FakeAtomic:
        def __init__(self, leaf_id: str):
            self.leaf_id = leaf_id

        def evaluate_report(self, context: Context, _profile: Mapping) -> dict:
            calls.append(self.leaf_id)
            status = (
                contracts.MEASURED
                if self.leaf_id == "floating"
                else contracts.PASS
            )
            result = _report(
                self.leaf_id,
                status,
                0.8 if status == contracts.MEASURED else 1.0,
            )
            if status == contracts.MEASURED:
                result.pop("failure_reason", None)
            return result

    monkeypatch.setattr(
        atomic_registry,
        "get",
        lambda leaf_id: FakeAtomic(leaf_id),
    )
    report = physical_safety.verify(
        Context(
            record={"metrics": {}},
            task=task,
            ids=IDS,
            out_dir=tmp_path / "out",
            spec={},
        )
    )

    assert calls == ["floating", "solid_penetration"]
    assert physical_safety._LEAVES == ("floating", "solid_penetration")
    assert set(physical_safety._LEAVES).issubset(
        atomic_registry.kinds("physics")
    )
    assert "ground_gap" not in calls
    assert "out_of_bounds" not in calls
    assert "bounds_discipline" not in calls
    assert "environment_consistency" not in calls
    assert "physics_regression" not in calls
    assert "physics_regression" not in {
        assertion["primitive"]
        for assertion in policy.physics_contract(task.id)["assertions"]
    }
    assert report["status"] == contracts.MEASURED
    assert report["score"] == pytest.approx(0.9)
    assert report["metrics"]["leaf_count"] == 2
    assert report["metrics"]["scored_leaf_count"] == 2
    assert report["metrics"]["requires_all_scored_leaves"] is True
    assert report["metrics"]["all_required_scores_present"] is True
    assert report["metrics"]["report_only_leaf_ids"] == []
    assert report["metrics"]["status_counts"][contracts.MEASURED] == 1
    assert report["metrics"]["status_counts"][contracts.PASS] == 1
    floating_leaf = next(
        value
        for value in report["metrics"]["leaf_results"]
        if value["leaf_id"] == "floating"
    )
    assert floating_leaf["status"] == contracts.MEASURED
    assert floating_leaf["score"] == 0.8


def test_edit_physical_safety_scores_applicable_leaves_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    local_profile = {
        **default_policy(None).physics_profile,
        "profile_id": "image-to-scene-edited-actors-physical-safety-v1",
        "selector": {"scope": "edited_actors"},
    }
    policy = FrozenEvaluationPolicy(
        policy_id="edit-physics-applicability",
        source_snapshot=_scene(_actor("source", 0.0)),
        edit_scope={},
        physics_profile=local_profile,
    )
    task = _task(
        tmp_path,
        kind="scene_edit",
        data={"evaluation_policy": policy.to_dict()},
    )

    class FakeAtomic:
        def __init__(self, leaf_id: str):
            self.leaf_id = leaf_id

        def evaluate_report(self, context: Context, _profile: Mapping) -> dict:
            if self.leaf_id == "floating":
                result = _report(
                    self.leaf_id,
                    contracts.MEASURED,
                    0.2,
                )
                result["metadata"] = {
                    "score_direction": "lower_is_better"
                }
                result.pop("failure_reason", None)
                return result
            return _report(self.leaf_id, "not_applicable", None)

    monkeypatch.setattr(
        atomic_registry,
        "get",
        lambda leaf_id: FakeAtomic(leaf_id),
    )

    report = physical_safety.verify(
        Context(
            record={"metrics": {}},
            task=task,
            ids=IDS,
            out_dir=tmp_path / "out",
            spec={},
        )
    )

    assert report["status"] == contracts.MEASURED
    assert report["score"] == pytest.approx(0.8)
    assert report["metrics"]["requires_all_scored_leaves"] is True
    assert report["metrics"][
        "exclude_not_applicable_from_required_scores"
    ] is True
    assert report["metrics"]["all_required_scores_present"] is True
    assert report["metrics"]["required_score_leaf_ids"] == ["floating"]
    assert report["metrics"]["excluded_not_applicable_leaf_ids"] == [
        "solid_penetration"
    ]
    assert report["evidence"]["score_requirement_policy"] == (
        "all_applicable_physics_leaves"
    )


def test_contract_migration_is_deterministic_and_runtime_reads_only_frozen_artifacts(
    tmp_path: Path,
):
    contract = {
        "schema_version": "0.2.0",
        "case_id": "reconstruction-test",
        "assertions": [
            {
                "id": "layout",
                "primitive": "no_overlap",
                "target_selector": {"scope": "candidate_all"},
            }
        ],
    }
    prompt = "Keep two props separate."
    first = migrate_legacy_contract(
        contract,
        prompt=prompt,
        task_mode="generation",
        bundle_id="reconstruction-v1",
    )
    second = migrate_legacy_contract(
        copy.deepcopy(contract),
        prompt=prompt,
        task_mode="generation",
        bundle_id="reconstruction-v1",
    )
    assert first.verification_bundle is not None
    assert second.verification_bundle is not None
    assert first.verification_bundle.to_dict() == second.verification_bundle.to_dict()
    restored = FrozenVerificationBundle.from_dict(
        first.verification_bundle.to_dict()
    )
    assert restored.schema_version == "1.0"
    assert all(
        value.source_provenance["kind"] == BindingSourceKind.LEGACY_CONTRACT.value
        for value in restored.evaluations
    )
    strict_payload = restored.to_dict()
    strict_payload["unexpected_runtime_fallback"] = True
    with pytest.raises(ValueError, match="unsupported fields"):
        FrozenVerificationBundle.from_dict(strict_payload)

    assert len(restored.evaluations) == len(restored.requirements)
    wrong_version = restored.to_dict()
    wrong_version["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="unsupported frozen bundle"):
        FrozenVerificationBundle.from_dict(wrong_version)
    assert first.migration_manifest["runtime_legacy_contract_read_allowed"] is False
    assert {
        "structure_count",
        "structure_concepts",
        "structure_additions",
        "spatial_overlap",
        "spatial_clearance",
        "spatial_cluster",
        "spatial_relations",
    } == set(atomic_registry.kinds("semantic"))

    # Mutating the authoring input after freeze cannot change runtime scoring.
    contract["assertions"][0]["touch_tolerance_cm"] = 100000.0
    policy = first.evaluation_policy
    task = _task(
        tmp_path,
        prompt=prompt,
        data={
            "evaluation_policy": policy.to_dict(),
        },
    )
    report = semantic_requirements.verify(
        Context(
            record={"metrics": {}},
            task=task,
            ids=IDS,
            out_dir=tmp_path / "semantic-out",
            spec={
                "candidate_scene": _scene(
                    _actor("left", 0.0), _actor("right", 100.0)
                ),
                "verification_bundle": restored.to_dict(),
            },
        )
    )
    assert report["status"] == contracts.MEASURED
    assert report["metrics"]["unknown_count"] == 0
    assert {
        value["observed"]["evaluation_status"]
        for value in report["metrics"]["checks"]
    } == {RequirementEvaluationStatus.MATCH.value}


def test_edit_migration_requires_frozen_source_and_refuses_to_guess_scope():
    contract = {
        "schema_version": "0.2.0",
        "case_id": "edit",
        "assertions": [
            {
                "id": "local",
                "primitive": "edit_locality",
                "maximum_unauthorized_actor_count": 1,
            }
        ],
    }
    with pytest.raises(ValueError, match="source_snapshot"):
        migrate_legacy_contract(
            contract,
            prompt="Move one chair.",
            task_mode="edit",
            bundle_id="edit-v1",
        )
    with pytest.raises(ValueError, match="explicit_edit_scope"):
        migrate_legacy_contract(
            contract,
            prompt="Move one chair.",
            task_mode="edit",
            bundle_id="edit-v1",
            source_snapshot={"actors": []},
        )


def test_preservation_does_not_treat_unrecorded_source_fields_as_empty():
    source_actor = _actor("chair", 0.0)
    source_actor["transform"].pop("rotation_deg")
    for field in (
        "actor_origin",
        "logical_object_id",
        "actor_role",
        "component_asset_paths",
        "properties",
    ):
        source_actor.pop(field, None)
    candidate_actor = copy.deepcopy(source_actor)
    candidate_actor.update(
        {
            "actor_origin": "candidate_export",
            "logical_object_id": "chair-1",
            "actor_role": "seat",
            "component_asset_paths": [candidate_actor["asset_path"]],
            "properties": {"intensity": 10.0},
        }
    )
    candidate_actor["transform"]["rotation_deg"] = [0.0, 0.0, 90.0]

    diff = diff_scenes(_scene(source_actor), _scene(candidate_actor))

    assert len(diff.matched) == 1
    assert diff.moved == []
    assert diff.modified == [], (
        "a richer Candidate export cannot turn fields the Input exporter did "
        "not measure into off-target edits"
    )


def test_preservation_ignores_runtime_dynamic_material_package_paths():
    source_actor = _actor(
        "sky",
        0.0,
        material_path=(
            "/Game/Input.Input:PersistentLevel.Sky.Mesh.MID_Sky_0"
        ),
        dynamic_material=True,
    )
    source_actor["material_paths"] = [
        source_actor["component_material_slots"][0]["material_path"]
    ]
    candidate_actor = copy.deepcopy(source_actor)
    candidate_path = (
        "/Game/SavedScenes/Candidate.Candidate:PersistentLevel."
        "Sky.Mesh.MID_Sky_0"
    )
    candidate_actor["component_material_slots"][0]["material_path"] = (
        candidate_path
    )
    candidate_actor["material_paths"] = [candidate_path]

    diff = diff_scenes(_scene(source_actor), _scene(candidate_actor))

    assert diff.modified == [], (
        "a runtime-owned MID path changes with the level package and is not "
        "evidence of an authored material edit"
    )


def test_preservation_ignores_engine_transient_dynamic_mid_instance_ids():
    source_actor = _actor(
        "water",
        0.0,
        material_path="/Engine/Transient.WaterMID_277",
        dynamic_material=True,
    )
    source_actor["material_paths"] = [
        source_actor["component_material_slots"][0]["material_path"]
    ]
    candidate_actor = copy.deepcopy(source_actor)
    candidate_actor["component_material_slots"][0]["material_path"] = (
        "/Engine/Transient.WaterMID_293"
    )
    candidate_actor["material_paths"] = ["/Engine/Transient.WaterMID_293"]

    diff = diff_scenes(_scene(source_actor), _scene(candidate_actor))

    assert diff.modified == [], (
        "Engine transient dynamic MID ordinals are allocated per load and "
        "cannot define an authored material change"
    )


def test_preservation_still_detects_dynamic_to_static_material_change():
    source_actor = _actor(
        "sky",
        0.0,
        material_path="/Game/Input.Input:PersistentLevel.Sky.Mesh.MID_Sky_0",
        dynamic_material=True,
    )
    source_actor["material_paths"] = [
        source_actor["component_material_slots"][0]["material_path"]
    ]
    candidate_actor = copy.deepcopy(source_actor)
    candidate_actor["component_material_slots"][0].update(
        {
            "material_path": "/Game/Materials/M_Sky.M_Sky",
            "is_dynamic": False,
        }
    )
    candidate_actor["material_paths"] = ["/Game/Materials/M_Sky.M_Sky"]

    diff = diff_scenes(_scene(source_actor), _scene(candidate_actor))

    assert len(diff.modified) == 1
    assert set(diff.modified[0].fields) == {
        "component_material_slots",
        "material_paths",
    }


def test_image_repair_case_executes_all_five_source_preservation_leaves(
    tmp_path: Path,
):
    wall = _actor("wall", 0.0)
    door = _actor("garage_door", 100.0)
    source = _scene(wall)
    candidate = _scene(wall, door)
    policy = FrozenEvaluationPolicy(
        policy_id="image-repair-preservation-case-v1",
        source_snapshot=source,
        edit_scope={
            "targets": [
                {
                    "id": "restore-garage-door",
                    "selector": {
                        "allowed_asset_paths": [door["asset_path"]],
                        "allowed_classes": [door["class"]],
                    },
                    "allow": {"add": True},
                }
            ],
            "tolerances": {
                "location_cm": 0.1,
                "rotation_deg": 0.01,
                "scale": 0.0001,
            },
            "limits": {
                "maximum_added_actor_count": 1,
                "maximum_removed_actor_count": 0,
                "maximum_moved_actor_count": 0,
                "maximum_modified_actor_count": 0,
                "maximum_off_target_change_count": 0,
            },
        },
        physics_profile=default_policy(None).physics_profile,
    )
    task = _task(
        tmp_path,
        kind="scene_repair",
        data={"evaluation_policy": policy.to_dict()},
    )

    report = source_preservation.verify(
        Context(
            record={"metrics": {}},
            task=task,
            ids=IDS,
            out_dir=tmp_path / "preservation-out",
            spec={"candidate_scene": candidate},
        )
    )

    leaves = {
        value["leaf_id"]: value
        for value in report["metrics"]["leaf_results"]
    }
    assert set(leaves) == {
        "actor_set_preserved",
        "transform_scope_preserved",
        "property_scope_preserved",
        "edit_locality",
        "off_target_change",
    }
    assert report["status"] == contracts.MEASURED
    assert all(value["status"] == contracts.MEASURED for value in leaves.values())
    assert leaves["actor_set_preserved"]["metrics"]["added_actor_count"] == 1
    assert leaves["actor_set_preserved"]["metrics"][
        "unauthorized_added_actor_count"
    ] == 0
    assert report["probes_used"] == ("input_candidate_scene_diff",)


@pytest.mark.parametrize(
    ("candidate_slots", "expected_status", "expected_mismatches"),
    [
        (("/Game/M/Red.Red", False), "measured", 0),
        (("/Game/M/Blue.Blue", False), "measured", 1),
        (("/Game/M/Red.Red", True), "not_evaluated", 0),
        (None, "not_evaluated", 0),
    ],
)
def test_gt_material_slot_metric_refuses_unreliable_evidence(
    candidate_slots,
    expected_status,
    expected_mismatches,
):
    canonical = _actor(
        "chair", 0.0, material_path="/Game/M/Red.Red"
    )
    candidate = _actor(
        "chair",
        0.0,
        material_path=(candidate_slots[0] if candidate_slots else None),
        dynamic_material=(candidate_slots[1] if candidate_slots else False),
    )
    metrics = compare_detailed([candidate], [canonical])["metrics"]

    assert metrics["material_slot_status"] == expected_status
    assert metrics["material_slot_mismatch_count"] == expected_mismatches
    if expected_status == "not_evaluated":
        assert metrics["material_slot_match_rate"] is None


def _full_attribute_actor(
    label: str = "chair",
    *,
    material_path: str = "/Game/M/Red.Red",
    intensity: float = 10.0,
) -> dict:
    actor = _actor(label, 0.0, material_path=material_path)
    actor["material_paths"] = [material_path]
    actor["properties"] = {"intensity": intensity}
    return actor


@pytest.mark.parametrize(
    ("mutation", "expected_score", "expected_channels"),
    [
        (
            "property",
            0.666667,
            {
                "task_relevant_properties": 0.0,
                "legacy_material_set": 1.0,
                "component_material_slots": 1.0,
            },
        ),
        (
            "material_slot",
            0.333333,
            {
                "task_relevant_properties": 1.0,
                "legacy_material_set": 0.0,
                "component_material_slots": 0.0,
            },
        ),
    ],
)
def test_attribute_diff_score_drops_for_measured_mismatches(
    mutation: str,
    expected_score: float,
    expected_channels: dict[str, float],
):
    canonical = _full_attribute_actor()
    candidate = copy.deepcopy(canonical)
    if mutation == "property":
        candidate["properties"]["intensity"] = 25.0
    else:
        blue = "/Game/M/Blue.Blue"
        candidate["material_paths"] = [blue]
        candidate["component_material_slots"][0]["material_path"] = blue

    metrics = compare_detailed([candidate], [canonical])["metrics"]
    leaf = scene_diff_verifier._attribute_leaf(
        SimpleNamespace(ids=IDS), metrics, {}, {}
    )

    assert leaf["status"] == contracts.MEASURED
    assert leaf["score"] == expected_score
    assert leaf["evidence"]["normalized_channels"] == expected_channels


def test_missing_material_slot_evidence_is_not_scored_as_match_or_mismatch():
    canonical = _full_attribute_actor()
    candidate = copy.deepcopy(canonical)
    canonical.pop("component_material_slots")

    metrics = compare_detailed([candidate], [canonical])["metrics"]
    leaf = scene_diff_verifier._attribute_leaf(
        SimpleNamespace(ids=IDS), metrics, {}, {}
    )

    assert metrics["material_slot_status"] == "not_evaluated"
    assert metrics["material_slot_match_rate"] is None
    assert metrics["candidate_attribute_schema_coverage"] == 1.0
    assert metrics["canonical_attribute_schema_coverage"] == 0.0
    assert leaf["status"] == contracts.MEASURED
    assert leaf["score"] == 1.0
    assert leaf["metrics"]["component_material_slots"]["status"] == (
        "not_evaluated"
    )
    assert leaf["evidence"]["normalized_channels"][
        "component_material_slots"
    ] is None


def test_old_gt_schema_is_attributed_to_canonical_and_withholds_attribute_leaf():
    canonical = _full_attribute_actor()
    candidate = copy.deepcopy(canonical)
    for field in ("properties", "material_paths", "component_material_slots"):
        canonical.pop(field)

    metrics = compare_detailed([candidate], [canonical])["metrics"]
    leaf = scene_diff_verifier._attribute_leaf(
        SimpleNamespace(ids=IDS), metrics, {}, {}
    )

    assert metrics["candidate_attribute_schema_missing_pair_count"] == 0
    assert metrics["canonical_attribute_schema_missing_pair_count"] == 1
    assert leaf["status"] == "not_evaluated"
    assert leaf["score"] is None
    assert "candidate=0 canonical=1" in leaf["failure_reason"]


def test_scene_diff_reuses_one_correspondence_for_all_five_leaves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    canonical = [_actor("chair", 0.0, material_path="/Game/M/Red.Red")]
    candidate = [
        _actor("chair", 25.0, material_path="/Game/M/Blue.Blue"),
        _actor("table", 200.0),
    ]
    detailed = compare_detailed(candidate, canonical)
    calls = 0

    def compare_once(_context: Context) -> dict:
        nonlocal calls
        calls += 1
        return {
            **contracts.base("gt_geometry", IDS),
            "status": contracts.PASS,
            "score": None,
            "metrics": {
                **detailed["metrics"],
                "metric_version": "gt-geometry-v4",
            },
            "evidence": {
                "gt_id": "canonical",
                "candidate_exported_from": "independent_scoring_editor",
                "logical_object_policy": "shared_test_correspondence",
            },
            "artifacts": {"correspondence": str(tmp_path / "audit.json")},
            "probes_used": ("gt_scene_correspondence",),
        }

    monkeypatch.setattr(scene_diff_verifier.gt_geometry, "verify", compare_once)
    report = scene_diff_verifier.verify(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={"ground_truth": "/Game/GT/Canonical"},
        )
    )

    top_leaves = {
        value["report_id"].rsplit(".", 1)[-1]: value
        for value in report["metrics"]["leaf_results"]
    }
    structured = top_leaves["structured_scene_diff"]
    leaves = {
        value["report_id"].rsplit(".", 1)[-1]: value
        for value in structured["metrics"]["leaf_results"]
    }
    assert calls == 1
    assert report["score"] is not None
    assert set(top_leaves) == {
        "structured_scene_diff",
        "visual_semantic_diff",
    }
    assert set(leaves) == {
        "actor_correspondence",
        "identity_diff",
        "transform_diff",
        "geometry_diff",
        "attribute_diff",
    }
    assert leaves["identity_diff"]["metrics"]["extra_logical_object_count"] == 1
    assert leaves["transform_diff"]["metrics"]["absolute_location_error_cm"] == (
        detailed["metrics"]["absolute_location_error_cm"]
    )
    assert leaves["attribute_diff"]["metrics"]["component_material_slots"][
        "material_slot_mismatch_count"
    ] == 1
    assert structured["evidence"]["correspondence_reused_by"] == [
        "identity_diff",
        "transform_diff",
        "geometry_diff",
        "attribute_diff",
    ]
    visual = top_leaves["visual_semantic_diff"]
    assert {
        value["leaf_id"] for value in visual["metrics"]["leaf_results"]
    } == {
        "caption_diff",
        "paired_render_diff",
        "visual_equivalence",
        "calibrated_visual_score",
    }
    assert all(
        value["status"] == "not_applicable"
        for value in visual["metrics"]["leaf_results"]
    )


def test_scene_diff_splits_one_paired_visual_run_into_stable_public_leaves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def structured(_context: Context) -> dict:
        return {
            **contracts.base("gt_geometry", IDS),
            "status": contracts.PASS,
            "score": None,
            "metrics": {
                **compare_detailed([_actor("chair", 0.0)], [_actor("chair", 0.0)])[
                    "metrics"
                ],
                "metric_version": "gt-geometry-v4",
            },
            "evidence": {"logical_object_policy": "one_shared_match"},
            "artifacts": {},
            "probes_used": ("gt_scene_correspondence",),
        }

    calls = 0

    def paired_visual(_context: Context) -> dict:
        nonlocal calls
        calls += 1
        return {
            **contracts.base("visual_as_judge", IDS),
            "status": contracts.PASS,
            "score": 0.875,
            "metrics": {
                "paired_render_diff": {
                    "status": "measured",
                    "metrics": {"aggregate": {"rgb_perceptual_similarity": 0.9}},
                },
                "visual_equivalence": {
                    "status": "measured",
                    "dimensions": {"composition_layout": 0.8},
                },
                "calibrated_visual_score": {
                    "score": 0.875,
                    "dimensions": {"composition_layout": 0.84},
                    "aggregation": "frozen_deterministic_vlm_blend",
                },
            },
            "evidence": {
                "judge_policy": "visual-gt-paired-legacy-formal/v15",
                "calibration_status": "frozen",
            },
            "artifacts": {"paired_view_0": str(tmp_path / "paired.png")},
            "probes_used": ("vlm_as_judge",),
        }

    monkeypatch.setattr(scene_diff_verifier.gt_geometry, "verify", structured)
    monkeypatch.setattr(
        scene_diff_verifier.visual_as_judge, "verify", paired_visual
    )
    report = scene_diff_verifier.verify(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={
                "name": "scene_diff",
                "ground_truth": "/Game/GT/Canonical",
                "visual_semantic_diff": {
                    "caption_diff": False,
                    "paired_visual": True,
                    "judge_policy": str(tmp_path / "policy.yaml"),
                },
            },
        )
    )

    assert calls == 1
    assert report["report_id"] == "scene_diff"
    assert report["score"] is not None
    visual = next(
        value
        for value in report["metrics"]["leaf_results"]
        if value["leaf_id"] == "visual_semantic_diff"
    )
    leaves = {
        value["leaf_id"]: value for value in visual["metrics"]["leaf_results"]
    }
    assert leaves["caption_diff"]["status"] == "not_applicable"
    assert leaves["paired_render_diff"]["status"] == contracts.MEASURED
    assert leaves["paired_render_diff"]["score"] == 0.9
    assert leaves["visual_equivalence"]["status"] == contracts.MEASURED
    assert leaves["visual_equivalence"]["score"] == 0.8
    assert leaves["visual_equivalence"]["metrics"]["dimensions"] == {
        "composition_layout": 0.8
    }
    calibrated = leaves["calibrated_visual_score"]
    assert calibrated["report_id"] == (
        "scene_diff.visual_semantic_diff.calibrated_visual_score"
    )
    assert calibrated["status"] == contracts.MEASURED
    assert calibrated["score"] == 0.875
    assert calibrated["metrics"]["dimensions"] == {
        "composition_layout": 0.84
    }
    assert calibrated["metrics"]["contributes_to_gt_aggregate"] is True
    assert visual["score"] == pytest.approx(0.875)
    assert visual["metrics"]["report_only_leaf_ids"] == [
        "paired_render_diff",
        "visual_equivalence",
    ]
    assert visual["metrics"]["aggregate_score_vector"] == {
        "calibrated_visual_score": 0.875
    }


def test_visual_wrapper_keeps_raw_inputs_separate_until_calibration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        visual_as_judge_verifier,
        "evidence_requests",
        lambda _task, _spec: (SimpleNamespace(protocol="paired-test"),),
    )
    monkeypatch.setattr(
        visual_as_judge_verifier.vlm_as_judge,
        "verify",
        lambda _context: {
            **contracts.base("vlm_as_judge", IDS),
            "status": contracts.PASS,
            "score": 0.75,
            "metrics": {
                "deterministic": {
                    "aggregate": {"base_color_similarity": 0.7},
                    "per_view": {"view_0": {"base_color_similarity": 0.7}},
                },
                "vlm": {
                    "dimensions": {"material_fidelity": 0.8},
                    "per_view": {"view_0": {"material_fidelity": 0.8}},
                },
                "calibrated": {
                    "dimensions": {"material_fidelity": 0.75},
                    "overall_visual_similarity": 0.75,
                },
                "model_call_count": 1,
                "total_time_s": 0.1,
            },
            "evidence": {"observations": [{"dimension": "material_fidelity"}]},
            "artifacts": {},
            "probes_used": ("vlm_as_judge",),
        },
    )

    report = visual_as_judge_verifier.verify(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={"mode": "gt_paired"},
            render_evidence={"paired-test": object()},
        )
    )

    paired = report["metrics"]["paired_render_diff"]
    semantic = report["metrics"]["visual_equivalence"]
    calibrated = report["metrics"]["calibrated_visual_score"]
    assert paired["metrics"]["aggregate"] == {"base_color_similarity": 0.7}
    assert "dimensions" not in paired
    assert semantic["dimensions"] == {"material_fidelity": 0.8}
    assert calibrated["dimensions"] == {"material_fidelity": 0.75}
    assert calibrated["score"] == 0.75
    assert "deterministic" not in semantic


def test_scene_diff_caption_leaf_has_one_canonical_public_address(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        scene_diff_verifier.gt_geometry,
        "verify",
        lambda _context: _report("gt_geometry", contracts.PASS),
    )

    class FakeCaptionAtomic:
        def evaluate_report(self, context: Context, _policy: Mapping) -> dict:
            return {
                **contracts.base("gt_caption_similarity", context.ids),
                "status": contracts.PASS,
                "score": None,
                "metrics": {"cosine_distance": 0.125},
                "evidence": {"caption_model": "frozen-test-model"},
                "artifacts": {},
                "probes_used": ("caption_model",),
            }

    monkeypatch.setattr(
        scene_diff_verifier.atomic_registry,
        "get",
        lambda evaluator_id: FakeCaptionAtomic()
        if evaluator_id == "independent_caption_embedding_distance"
        else atomic_registry.get(evaluator_id),
    )
    report = scene_diff_verifier.verify(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={
                "name": "scene_diff",
                "ground_truth": "/Game/GT/Canonical",
                "visual_semantic_diff": {
                    "caption_diff": True,
                    "paired_visual": False,
                },
            },
        )
    )

    visual = next(
        value
        for value in report["metrics"]["leaf_results"]
        if value["leaf_id"] == "visual_semantic_diff"
    )
    caption = next(
        value
        for value in visual["metrics"]["leaf_results"]
        if value["leaf_id"] == "caption_diff"
    )
    assert caption["report_id"] == (
        "scene_diff.visual_semantic_diff.caption_diff"
    )
    assert caption["score"] is None
    assert caption["metrics"]["cosine_distance"] == 0.125
    assert "gt_caption_similarity" not in caption["report_id"]


def test_scene_diff_rejects_a_boolean_calibrated_visual_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        scene_diff_verifier.visual_as_judge,
        "verify",
        lambda _context: {
            **contracts.base("visual_as_judge", IDS),
            "status": contracts.PASS,
            "score": True,
            "metrics": {
                "paired_render_diff": {"status": "measured"},
                "visual_equivalence": {"status": "measured"},
                "calibrated_visual_score": {"score": True},
            },
            "evidence": {},
            "artifacts": {},
            "probes_used": (),
        },
    )
    leaves = scene_diff_verifier._paired_visual_leaves(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={
                "ground_truth": "/Game/GT/Canonical",
                "visual_semantic_diff": {},
            },
        )
    )

    calibrated = next(
        value
        for value in leaves
        if value["leaf_id"] == "calibrated_visual_score"
    )
    assert calibrated["status"] == contracts.ERROR
    assert calibrated["score"] is None
    assert "finite calibrated score" in calibrated["failure_reason"]
