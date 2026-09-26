"""Public verifier vocabulary, independent from scoring implementations.

The task layer owns the names a frozen task may contain. Evaluation imports
this module; task loading never imports the evaluator. Canonical verifier IDs
describe evidence and policy responsibilities. The schema accepts only the
current public surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


AUTOMATIC_POLICY_KINDS: tuple[str, ...] = (
    "candidate_integrity",
    "physical_safety",
)

CONDITIONAL_POLICY_KINDS: tuple[str, ...] = (
    "source_preservation",
)

RUNNER_POLICY_KINDS: tuple[str, ...] = (
    "candidate_integrity",
    "source_preservation",
    "physical_safety",
)

TASK_VERIFIER_KINDS: tuple[str, ...] = (
    "semantic_requirements",
    "overview_prompt_alignment",
    "scene_diff",
    "gt_repair",
)

CANONICAL_VERIFIER_KINDS: frozenset[str] = frozenset(
    (*RUNNER_POLICY_KINDS, *TASK_VERIFIER_KINDS)
)

REQUIRED_CANONICAL_FIELDS: dict[str, tuple[str, ...]] = {
    "semantic_requirements": ("verification_bundle",),
    "scene_diff": ("ground_truth",),
    "gt_repair": ("ground_truth",),
}

_VISUAL_SEMANTIC_FIELDS = frozenset(
    {
        "caption_diff",
        "paired_visual",
    }
)


def _validate_scene_diff(spec: Mapping[str, Any]) -> None:
    """Validate the one public GT interface and its frozen leaf selection."""

    configured = spec.get("visual_semantic_diff")
    if configured is None:
        # Structured-only scene_diff is valid. New authoring may explicitly
        # enable caption or paired-visual branches below.
        return
    if not isinstance(configured, Mapping):
        raise ValueError("scene_diff.visual_semantic_diff must be a mapping")
    unknown = sorted(set(configured) - _VISUAL_SEMANTIC_FIELDS)
    if unknown:
        raise ValueError(
            "scene_diff.visual_semantic_diff has unsupported field(s): "
            f"{unknown}"
        )
    for field in ("caption_diff", "paired_visual"):
        if not isinstance(configured.get(field), bool):
            raise ValueError(
                f"scene_diff.visual_semantic_diff.{field} must be boolean"
            )


def validate_verifier_spec(
    spec: Mapping[str, Any],
    *,
    case_type: str = "prompt_to_scene",
) -> None:
    """Validate one canonical verifier's frozen runtime inputs."""

    name = str(spec.get("name") or "")
    required_fields = REQUIRED_CANONICAL_FIELDS.get(name, ())
    if name == "semantic_requirements" and case_type == "image_to_scene":
        required_fields = ()
    missing = [
        field
        for field in required_fields
        if not isinstance(spec.get(field), str) or not str(spec[field]).strip()
    ]
    if missing:
        raise ValueError(f"{name} needs frozen authoring input(s): {missing}")
    if name in {"scene_diff", "gt_repair"}:
        _validate_scene_diff(spec)


def validate_verifier_set(
    specs: Sequence[Mapping[str, Any]],
    *,
    case_type: str = "prompt_to_scene",
) -> None:
    """Reject unknown or duplicate IDs and validate every entry."""

    names = [str(spec.get("name") or "") for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"verifier(s) declared twice: {duplicates}")
    unknown = sorted(set(names) - CANONICAL_VERIFIER_KINDS)
    if unknown:
        raise ValueError(f"unknown verifier(s): {unknown}")
    for spec in specs:
        validate_verifier_spec(spec, case_type=case_type)


__all__ = [
    "AUTOMATIC_POLICY_KINDS",
    "CANONICAL_VERIFIER_KINDS",
    "CONDITIONAL_POLICY_KINDS",
    "REQUIRED_CANONICAL_FIELDS",
    "RUNNER_POLICY_KINDS",
    "TASK_VERIFIER_KINDS",
    "validate_verifier_set",
    "validate_verifier_spec",
]
