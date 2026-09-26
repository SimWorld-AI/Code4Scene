from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from code4scene.evaluation.offline_overview_prompt_alignment import (
    OverviewAlignmentError,
    _validate_alignment,
    _validate_structural_integrity,
    evaluate_case,
    run_sweep,
    select_overview_frames,
)
from code4scene.evaluation.requirement_graph.existing_llm import (
    LLMResponse,
    ToolCall,
)


class FakeClient:
    model = "fake-overview-model"

    def __init__(
        self,
        *,
        structural_integrity_score: float = 0.9,
        severe_intrinsic_corruption: bool = False,
        structural_confidence: float = 0.95,
        structural_evidence_view_indices: list[int] | None = None,
        allowed_nonstandard_geometry: list[str] | None = None,
    ) -> None:
        self.calls = []
        self.structural_integrity_score = structural_integrity_score
        self.severe_intrinsic_corruption = severe_intrinsic_corruption
        self.structural_confidence = structural_confidence
        self.structural_evidence_view_indices = (
            structural_evidence_view_indices or [1, 2, 3, 4]
        )
        self.allowed_nonstandard_geometry = allowed_nonstandard_geometry or []

    def chat(self, messages, tools, *, max_tokens=1024, temperature=0.0):
        self.calls.append((messages, tools, max_tokens, temperature))
        name = tools[0]["name"]
        if "structural_geometry_integrity" in name:
            score = self.structural_integrity_score
            arguments = {
                "score": score,
                "status": (
                    "valid"
                    if score >= 0.75
                    else "invalid" if score <= 0.2 else "degraded"
                ),
                "issues": (
                    []
                    if score >= 0.75
                    else ["Several major structures visibly lean across views."]
                ),
                "issue_categories": (
                    [] if score >= 0.75 else ["mesh_deformation"]
                ),
                "evidence_view_indices": self.structural_evidence_view_indices,
                "severe_intrinsic_corruption": self.severe_intrinsic_corruption,
                "confidence": self.structural_confidence,
                "rationale": (
                    "Major structures are visually upright and coherent."
                    if score >= 0.75
                    else "Major structures show repeated geometric deformation."
                ),
            }
            return LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="structural", name=name, arguments=arguments)
                ],
                usage={"total_tokens": 5},
                raw={"request_id": "raw-structural-response"},
            )
        arguments = {
            "visible_scene_summary": (
                "A coherent square desert town with low masonry blocks."
            ),
            "matched_prompt_elements": ["square desert town"],
            "missing_or_unsupported_prompt_elements": ["fine detail"],
            "allowed_nonstandard_geometry": (
                self.allowed_nonstandard_geometry
            ),
            "global_prompt_alignment_score": 0.7,
            "global_prompt_alignment_rationale": "The setting matches.",
            "composition_and_layout_score": 0.6,
            "composition_and_layout_rationale": "The broad layout matches.",
            "style_atmosphere_coherence_score": 0.5,
            "style_atmosphere_coherence_rationale": (
                "Some atmosphere is visible."
            ),
            "completeness_and_polish_score": 0.4,
            "completeness_and_polish_rationale": (
                "Several requested details are absent."
            ),
            "summary": "The overview partially matches the prompt.",
        }
        return LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="call", name=name, arguments=arguments)],
            usage={"total_tokens": 10},
        )


def _case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "results" / "case"
    stage3 = (
        case_dir
        / "scene_evidence"
        / "episode"
        / "requirement_graph"
        / "stage3"
    )
    frames_dir = stage3 / "stage3_frames"
    frames_dir.mkdir(parents=True)
    requirements = stage3.parent / "requirements.json"
    requirements.write_text("[]\n", encoding="utf-8")
    frames = []
    roles = ["close", "overview", "collection_wide", "overview", "overview"]
    for index, role in enumerate(roles, start=1):
        value = 128 + index
        path = frames_dir / f"s3f_{index:06d}.png"
        Image.new("RGB", (64, 36), (value, value, value)).save(path)
        frames.append(
            {
                "frame_id": f"s3f_{index:06d}",
                "image_hash": f"hash-{index}",
                "shot_role": role,
                "phase": "test",
                "quality": {},
            }
        )
    (stage3 / "stage3_frame_index.json").write_text(
        json.dumps({"frames": frames}) + "\n",
        encoding="utf-8",
    )
    task = tmp_path / "task.frozen.yaml"
    task.write_text(
        yaml.safe_dump({"inputs": {"prompt": "Build a coherent square desert town."}}),
        encoding="utf-8",
    )
    (case_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "case",
                "task_file": str(task),
                "batch_status": "complete",
                "reports": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return case_dir


def test_select_overview_frames_prioritizes_wide_roles(tmp_path: Path) -> None:
    selected = select_overview_frames(_case(tmp_path))
    assert [item["frame_id"] for item in selected] == [
        "s3f_000002",
        "s3f_000004",
        "s3f_000005",
        "s3f_000003",
    ]


def test_select_overview_frames_prefers_frozen_formal_gallery(tmp_path: Path) -> None:
    case_dir = _case(tmp_path)
    gallery = case_dir / "render_evidence" / "formal"
    gallery.mkdir(parents=True)
    formal = []
    for index in range(4):
        path = gallery / f"view_{index}.png"
        Image.new("RGB", (64, 36), (30 + index, 60, 90)).save(path)
        formal.append(
            {
                "frame_id": f"formal_view_{index}",
                "path": str(path),
                "shot_role": "overview",
            }
        )
    result = json.loads((case_dir / "result.json").read_text())
    result["reports"] = [
        {
            "report_id": "overview_prompt_alignment",
            "status": "measured",
            "evidence": {"frames": formal},
        }
    ]

    selected = select_overview_frames(case_dir, result)

    assert [item["frame_id"] for item in selected] == [
        "formal_view_0",
        "formal_view_1",
        "formal_view_2",
        "formal_view_3",
    ]
    assert all("render_evidence/formal" in item["path"] for item in selected)


def test_failed_formal_overview_does_not_fall_back_to_stage3(tmp_path: Path) -> None:
    case_dir = _case(tmp_path)
    result = json.loads((case_dir / "result.json").read_text())
    result["reports"] = [
        {
            "report_id": "overview_prompt_alignment",
            "status": "error",
            "failure_reason": "formal render unavailable",
        }
    ]

    with pytest.raises(
        OverviewAlignmentError,
        match="source overview report is not measured",
    ):
        select_overview_frames(case_dir, result)


def test_direct_metric_sends_prompt_and_images_in_one_request(tmp_path: Path) -> None:
    client = FakeClient()
    result = evaluate_case(
        _case(tmp_path),
        model_label="test-model",
        client_factory=lambda: client,
    )

    assert result["overview_alignment_score"] == 0.59
    assert result["structural_integrity_score"] == 0.9
    assert result["structural_adjustment_multiplier"] == 0.975
    assert result["score"] == 0.5752
    assert result["structural_adjustment_policy"] == {
        "soft_floor": 0.75,
        "structural_score_weight": 0.25,
        "severe_score_cap": 0.4,
        "severe_minimum_confidence": 0.8,
        "severe_minimum_evidence_views": 2,
        "severe_requires_issue_text_and_category": True,
    }
    assert result["ue_recapture_performed"] is False
    assert result["calls"]["count"] == 2
    calls_by_name = {call[1][0]["name"]: call for call in client.calls}
    direct_call = next(
        call for name, call in calls_by_name.items() if "direct_overview" in name
    )
    structural_call = next(
        call
        for name, call in calls_by_name.items()
        if "structural_geometry_integrity" in name
    )
    direct_text = json.dumps(
        [block for message in direct_call[0] for block in message.content],
        default=str,
    )
    structural_text = json.dumps(
        [block for message in structural_call[0] for block in message.content],
        default=str,
    )
    assert "Build a coherent square desert town" in direct_text
    assert "Build a coherent square desert town" not in structural_text
    assert "prompt-blind" in structural_text.casefold()
    assert "do not penalize floating islands" in structural_text.casefold()
    assert len(client.calls) == 2
    assert result["visible_scene_summary"].startswith("A coherent square")
    assert result["matched_prompt_elements"] == ["square desert town"]
    assert result["allowed_nonstandard_geometry"] == []
    assert result["structural_geometry_integrity"]["score"] == 0.9
    assert result["structural_geometry_integrity"][
        "evidence_view_indices"
    ] == [0, 1, 2, 3]
    structural_audit = result["calls"][
        "prompt_blind_structural_geometry_integrity"
    ]
    assert structural_audit["raw_tool_arguments"][
        "evidence_view_indices"
    ] == [1, 2, 3, 4]
    assert structural_audit["validated_tool_arguments"][
        "evidence_view_indices"
    ] == [0, 1, 2, 3]
    assert structural_audit["raw_response"] == {
        "request_id": "raw-structural-response"
    }
    assert structural_audit["response_normalizations"] == [
        {
            "field": "evidence_view_indices",
            "operation": "one_based_to_zero_based",
            "raw": [1, 2, 3, 4],
            "normalized": [0, 1, 2, 3],
        }
    ]


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ([0, 1, 2, 3], [0, 1, 2, 3]),
        ([1, 2, 3, 4], [0, 1, 2, 3]),
        ([1, 2, 3], [1, 2, 3]),
    ],
)
def test_structural_indices_accept_zero_based_and_unambiguous_one_based(
    indices,
    expected,
) -> None:
    result = _validate_structural_integrity(
        {
            "score": 0.9,
            "status": "valid",
            "issues": [],
            "issue_categories": [],
            "evidence_view_indices": indices,
            "severe_intrinsic_corruption": False,
            "confidence": 0.95,
            "rationale": "The structures are coherent.",
        }
    )

    assert result["evidence_view_indices"] == expected


def test_structural_integrity_synthesizes_missing_audit_rationale() -> None:
    result = _validate_structural_integrity(
        {
            "score": 0.75,
            "status": "degraded",
            "issues": ["A few panels appear detached."],
            "issue_categories": ["structural_fragmentation"],
            "evidence_view_indices": [0, 1, 2, 3],
            "severe_intrinsic_corruption": False,
            "confidence": 0.62,
        }
    )

    assert result["score"] == 0.75
    assert result["rationale"] == "A few panels appear detached."


@pytest.mark.parametrize(
    "indices",
    [
        [0, 1, 2, 4],
        [1, 2, 3, 5],
        [1, 2, 4, 4],
    ],
)
def test_structural_indices_do_not_broadly_accept_invalid_values(indices) -> None:
    with pytest.raises(
        OverviewAlignmentError,
        match="structural geometry integrity contains invalid values",
    ):
        _validate_structural_integrity(
            {
                "score": 0.9,
                "status": "valid",
                "issues": [],
                "issue_categories": [],
                "evidence_view_indices": indices,
                "severe_intrinsic_corruption": False,
                "confidence": 0.95,
                "rationale": "The structures are coherent.",
            }
        )


def test_structural_geometry_integrity_softly_adjusts_overview_score(
    tmp_path: Path,
) -> None:
    result = evaluate_case(
        _case(tmp_path),
        model_label="test-model",
        client_factory=lambda: FakeClient(structural_integrity_score=0.3),
    )

    assert result["overview_alignment_score"] == 0.59
    assert result["structural_integrity_score"] == 0.3
    assert result["structural_integrity_status"] == "degraded"
    assert result["structural_adjustment_multiplier"] == 0.825
    assert result["score"] == pytest.approx(0.4868, abs=0.0001)
    assert result["severe_structural_cap_eligible"] is False
    assert result["severe_structural_cap_applied"] is False


def test_corroborated_high_confidence_severe_intrinsic_corruption_caps_score(
    tmp_path: Path,
) -> None:
    result = evaluate_case(
        _case(tmp_path),
        model_label="test-model",
        client_factory=lambda: FakeClient(
            structural_integrity_score=0.3,
            severe_intrinsic_corruption=True,
            structural_confidence=0.95,
        ),
    )

    assert result["overview_score_before_severe_cap"] == pytest.approx(
        0.4868,
        abs=0.0001,
    )
    assert result["severe_structural_cap_eligible"] is True
    assert result["severe_structural_cap_applied"] is True
    assert result["score"] == 0.4


def test_uncorroborated_severe_claim_does_not_activate_cap(tmp_path: Path) -> None:
    result = evaluate_case(
        _case(tmp_path),
        model_label="test-model",
        client_factory=lambda: FakeClient(
            structural_integrity_score=0.3,
            severe_intrinsic_corruption=True,
            structural_confidence=0.95,
            structural_evidence_view_indices=[1],
        ),
    )

    assert result["severe_structural_cap_eligible"] is False
    assert result["severe_structural_cap_applied"] is False
    assert result["score"] == pytest.approx(0.4868, abs=0.0001)


def test_unclassified_severe_claim_is_audited_but_cannot_cap(
    tmp_path: Path,
) -> None:
    result = evaluate_case(
        _case(tmp_path),
        model_label="test-model",
        client_factory=lambda: FakeClient(
            structural_integrity_score=0.9,
            severe_intrinsic_corruption=True,
            structural_confidence=0.95,
        ),
    )

    assert result["structural_geometry_integrity"][
        "severe_intrinsic_corruption"
    ] is True
    assert result["structural_geometry_integrity"]["issue_categories"] == []
    assert result["severe_structural_cap_eligible"] is False
    assert result["severe_structural_cap_applied"] is False
    assert result["score"] == 0.5752


def test_prompt_authorized_nonstandard_geometry_is_preserved_for_audit(
    tmp_path: Path,
) -> None:
    result = evaluate_case(
        _case(tmp_path),
        model_label="test-model",
        client_factory=lambda: FakeClient(
            structural_integrity_score=1.0,
            allowed_nonstandard_geometry=["floating_or_suspended_elements"],
        ),
    )

    assert result["allowed_nonstandard_geometry"] == [
        "floating_or_suspended_elements"
    ]
    assert result["score"] == 0.59


def test_alignment_accepts_strict_json_string_wrapper() -> None:
    dimensions = {
        name: {"score": 0.5, "rationale": f"{name} rationale"}
        for name in (
            "global_prompt_alignment",
            "composition_and_layout",
            "style_atmosphere_coherence",
            "completeness_and_polish",
        )
    }
    wrapped = {
        "dimensions": json.dumps(
            {"dimensions": dimensions, "summary": "wrapped summary"}
        )
    }
    result = _validate_alignment(wrapped)
    assert result["summary"] == "wrapped summary"
    assert result["dimensions"]["global_prompt_alignment"]["score"] == 0.5


def test_sweep_isolates_case_missing_frozen_stage3_evidence(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    invalid = root / "results" / "invalid"
    invalid.mkdir(parents=True)
    (invalid / "result.json").write_text(
        json.dumps(
            {
                "task_id": "invalid",
                "task_file": str(tmp_path / "missing-task.yaml"),
                "batch_status": "complete",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_sweep(
        {"model": root},
        tmp_path / "out",
        workers=1,
        client_factory=lambda: FakeClient(),
    )

    assert summary["job_count"] == 0
    assert summary["error_count"] == 1
    assert summary["records"][0]["case"] == "invalid"
    sidecar = json.loads(
        (tmp_path / "out" / "model" / "invalid.json").read_text()
    )
    assert sidecar["status"] == "error"
