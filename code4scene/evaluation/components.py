"""Canonical verifier responsibilities and publication policy.

Components document which stable top-level interface answers each benchmark
question. Atomic leaf metrics stay inside their owning composite; aggregation
is performed only by the report layer with the complete score vector retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from code4scene.evaluation.composite import iter_report_tree


POLICIES = ("formal", "gate", "not_scored")
AXES = (
    "execution",
    "semantic_fidelity",
    "spatial_logic",
    "physics",
    "visual_quality",
    "functionality",
)


@dataclass(frozen=True)
class Component:
    """One public evaluation responsibility."""

    id: str
    axis: str
    policy: str
    question: str
    verifiers: tuple[str, ...]
    rubrics: tuple[str, ...] = ()
    report_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"{self.id}: unknown axis {self.axis!r}")
        if self.policy not in POLICIES:
            raise ValueError(f"{self.id}: unknown policy {self.policy!r}")
        if not self.verifiers and self.policy != "not_scored":
            raise ValueError(f"{self.id}: a component needs a verifier")


COMPONENTS: tuple[Component, ...] = (
    Component(
        "execution.candidate_integrity",
        "execution",
        "gate",
        "is the Candidate publishable, reloadable, structurally complete and on the frozen asset library",
        ("candidate_integrity",),
    ),
    Component(
        "preservation.source",
        "semantic_fidelity",
        "formal",
        "how much of an image-guided edit remains unchanged outside its frozen edit scope",
        ("source_preservation",),
    ),
    Component(
        "physics.default_safety",
        "physics",
        "formal",
        "what fraction of the Candidate satisfies each physical measurement",
        ("physical_safety",),
    ),
    Component(
        "semantic.requirement_graph",
        "semantic_fidelity",
        "formal",
        "do the frozen text- or image-authored task requirements hold",
        ("semantic_requirements",),
    ),
    Component(
        "visual.overview_prompt_alignment",
        "visual_quality",
        "not_scored",
        "does direct four-view prompt alignment hold under prompt-blind structural integrity",
        ("overview_prompt_alignment",),
        rubrics=(
            "direct-multiview-overview-alignment",
            "prompt-blind-four-view-intrinsic-structural-geometry-integrity",
        ),
    ),
    Component(
        "gt.repair_target_diff",
        "semantic_fidelity",
        "formal",
        "did the Candidate recover the frozen GT-minus-Input repair targets",
        ("gt_repair",),
        report_paths=("gt_repair.repair_target_diff",),
    ),
    Component(
        "gt.structured_scene_diff",
        "spatial_logic",
        "formal",
        "how does the Candidate differ structurally from the canonical scene",
        ("scene_diff", "gt_repair"),
        report_paths=("scene_diff.structured_scene_diff",),
    ),
    Component(
        "gt.visual_semantic_diff",
        "visual_quality",
        "formal",
        "are aligned Candidate and GT renders visually equivalent under the frozen policy",
        ("scene_diff", "gt_repair"),
        rubrics=("gt-paired/v1",),
        report_paths=(
            "scene_diff.visual_semantic_diff.calibrated_visual_score",
        ),
    ),
    Component(
        "gt.caption_diff",
        "semantic_fidelity",
        "formal",
        "how close are independently captioned Candidate and GT renders in text space",
        ("scene_diff", "gt_repair"),
        report_paths=("scene_diff.visual_semantic_diff.caption_diff",),
    ),
)

BY_ID = {component.id: component for component in COMPONENTS}


def components_for(kind: str) -> tuple[str, ...]:
    """Return responsibilities served by one canonical verifier."""

    return tuple(
        component.id
        for component in COMPONENTS
        if kind in component.verifiers
    )


def covered_verifiers() -> frozenset[str]:
    """Canonical verifier IDs represented by the public component map."""

    return frozenset(
        kind for component in COMPONENTS for kind in component.verifiers
    )


def scored_kinds() -> frozenset[str]:
    """Public report paths that publish continuous numeric quality scores."""

    return frozenset(
        kind
        for component in COMPONENTS
        if component.policy == "formal"
        for kind in (component.report_paths or component.verifiers)
    )


def is_scored(kind: str) -> bool:
    """Whether this exact canonical report path publishes a formal score."""

    return kind in scored_kinds()


def group(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group reports by responsibility without combining their metrics."""

    by_address: dict[str, list[dict[str, Any]]] = {}
    for root in reports:
        for item in iter_report_tree(root):
            report_id = str(item.get("report_id"))
            by_address.setdefault(report_id, []).append(item)
    grouped = {}
    for component in COMPONENTS:
        addresses = component.report_paths or component.verifiers
        present = [
            item
            for address in addresses
            for item in by_address.get(address, ())
        ]
        grouped[component.id] = {
            "axis": component.axis,
            "policy": component.policy,
            "question": component.question,
            "verifiers": list(component.verifiers),
            "unanswered": [
                address for address in addresses if address not in by_address
            ],
            "reports": present,
            "statuses": sorted(
                {str(item.get("status")) for item in present}
            ),
        }
    return grouped


__all__ = [
    "AXES",
    "BY_ID",
    "COMPONENTS",
    "POLICIES",
    "Component",
    "components_for",
    "covered_verifiers",
    "group",
    "is_scored",
    "scored_kinds",
]
