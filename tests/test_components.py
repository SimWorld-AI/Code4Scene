"""The canonical verifier responsibilities stay in step."""

from __future__ import annotations

from pathlib import Path

import pytest

from code4scene.evaluation import components, verifiers
from code4scene.tasks.task import VERIFIER_KINDS


def test_every_component_names_verifiers_that_exist():
    unknown = {component.id: sorted(set(component.verifiers) - set(VERIFIER_KINDS))
               for component in components.COMPONENTS
               if set(component.verifiers) - set(VERIFIER_KINDS)}
    assert not unknown, (
        f"these components name verifiers nothing implements: {unknown}. A "
        f"component map that lists a verifier which does not exist is a "
        f"coverage claim nobody can check.")


def test_every_verifier_answers_a_declared_component():
    unplaced = sorted(set(VERIFIER_KINDS) - components.covered_verifiers())
    assert not unplaced, (
        f"these verifiers are under no component: {unplaced}. Either it answers "
        f"a declared component question and belongs in `components.py`, or it "
        f"is a new question and that is a decision worth writing down.")


def test_there_are_nine_canonical_responsibilities():
    assert len(components.COMPONENTS) == 9
    assert len(components.BY_ID) == 9, "two components share an id"


def test_no_component_score_exists():
    """Combining is what the port removed; this is where it would come back."""
    source = (Path(components.__file__)).read_text()
    for forbidden in ("def score(", "weighted", "def combine", "def overall"):
        assert forbidden not in source, (
            f"`components.py` grew {forbidden!r}. Components are a map from "
            f"questions to verifiers, not a scoring layer — an atomic verifier "
            f"whose number is averaged into a component score is an atomic "
            f"verifier in name only.")


def test_continuous_formal_verifiers_publish_scores_without_gates():
    assert components.is_scored("semantic_requirements")
    assert not components.is_scored("overview_prompt_alignment")
    assert not components.is_scored("candidate_integrity")
    assert components.is_scored("source_preservation")
    assert components.is_scored("physical_safety")
    assert components.is_scored("gt_repair.repair_target_diff")
    assert components.is_scored("scene_diff.structured_scene_diff")
    assert components.is_scored("scene_diff.visual_semantic_diff.caption_diff")
    assert components.is_scored(
        "scene_diff.visual_semantic_diff.calibrated_visual_score"
    )


def test_overview_prompt_alignment_stays_report_only_until_calibrated():
    component = components.BY_ID["visual.overview_prompt_alignment"]
    assert component.policy == "not_scored"
    assert component.verifiers == ("overview_prompt_alignment",)


def test_visual_prompt_alignment_is_a_formal_score():
    component = components.BY_ID["gt.visual_semantic_diff"]
    assert component.policy == "formal"
    assert component.verifiers == ("scene_diff", "gt_repair")
    assert component.report_paths == (
        "scene_diff.visual_semantic_diff.calibrated_visual_score",
    )


def test_the_frozen_visual_judge_has_no_report_only_component_branch():
    served = [
        component
        for component in components.COMPONENTS
        if component.policy == "formal" and "scene_diff" in component.verifiers
    ]

    assert served
    assert {component.policy for component in served} == {"formal"}
    assert {rubric for component in served for rubric in component.rubrics} == {
        "gt-paired/v1"
    }


def test_an_unknown_report_is_never_published_as_a_formal_score():
    assert not components.is_scored("a_verifier_nobody_placed")


@pytest.mark.parametrize("component", components.COMPONENTS,
                         ids=lambda item: item.id)
def test_each_component_declares_an_axis_a_policy_and_a_question(component):
    assert component.axis in components.AXES
    assert component.policy in components.POLICIES
    assert component.question and component.question[0].islower()


def test_grouping_a_run_shows_what_it_did_not_answer():
    """Partial coverage has to be visible, or a passing subset reads as a pass."""
    reports = [
        {"report_id": "physical_safety", "status": "measured", "score": 0.8}
    ]
    grouped = components.group(reports)
    physics = grouped["physics.default_safety"]
    assert len(physics["reports"]) == 1
    assert physics["unanswered"] == []
    assert grouped["semantic.requirement_graph"]["reports"] == []


def test_the_map_and_the_registry_name_the_same_verifiers():
    assert components.covered_verifiers() == set(verifiers.REGISTRY)
