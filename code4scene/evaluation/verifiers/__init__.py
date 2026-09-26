"""Registry and runner for the canonical verifier interfaces."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from code4scene.tasks.verifier_schema import CANONICAL_VERIFIER_KINDS

from .. import contracts
from ..composite import blocked_report
from ..context import Context, LABEL_SUFFIX, VerifierError, error, read_label
from ..evaluation_policy import task_mode
from ..render_evidence import EvidenceRequest, merge_requests


CLASSES_ALLOWED = ("open_ended", "gt")
_CANONICAL_CLASSES = {
    "candidate_integrity": "open_ended",
    "source_preservation": "open_ended",
    "physical_safety": "open_ended",
    "semantic_requirements": "open_ended",
    "overview_prompt_alignment": "open_ended",
    "scene_diff": "gt",
    "gt_repair": "gt",
}


def _module_verifier(kind: str) -> tuple[Callable[[Context], dict[str, Any]], str, dict[str, str]]:
    module = importlib.import_module(f"{__name__}.{kind}")
    verifier = getattr(module, "verify", None)
    declared = getattr(module, "CLASS", None)
    modes = getattr(module, "CLASS_BY_MODE", {})
    if not callable(verifier):
        raise VerifierError(f"{kind} exports no verify(context)")
    if declared not in CLASSES_ALLOWED:
        raise VerifierError(
            f"{kind} declares CLASS={declared!r}; expected one of {CLASSES_ALLOWED}"
        )
    if not isinstance(modes, dict) or any(
        value not in CLASSES_ALLOWED for value in modes.values()
    ):
        raise VerifierError(f"{kind} has invalid CLASS_BY_MODE")
    return verifier, declared, dict(modes)


CANONICAL_REGISTRY: dict[str, Callable[[Context], dict[str, Any]]] = {}
CANONICAL_CLASSES: dict[str, str] = {}
for _kind in sorted(CANONICAL_VERIFIER_KINDS):
    _verify, _declared, _modes = _module_verifier(_kind)
    expected = _CANONICAL_CLASSES[_kind]
    if _declared != expected:
        raise VerifierError(
            f"{_kind} declares {_declared!r}, canonical schema requires {expected!r}"
        )
    CANONICAL_REGISTRY[_kind] = _verify
    CANONICAL_CLASSES[_kind] = _declared

REGISTRY = CANONICAL_REGISTRY
CLASSES = CANONICAL_CLASSES


def kinds() -> list[str]:
    """Canonical public verifier IDs."""

    return sorted(CANONICAL_REGISTRY)


def evidence_requests_for_spec(
    task: Any,
    spec: Mapping[str, Any],
) -> tuple[EvidenceRequest, ...]:
    """Ask the verifier module for evidence without interpreting its ID."""

    kind = str(spec.get("name"))
    if kind not in REGISTRY:
        return ()
    module = importlib.import_module(f"{__name__}.{kind}")
    planner = getattr(module, "evidence_requests", None)
    if planner is None:
        return ()
    if not callable(planner):
        raise VerifierError(f"{kind}.evidence_requests is not callable")
    values = planner(task, dict(spec))
    if not isinstance(values, tuple) or any(
        not isinstance(value, EvidenceRequest) for value in values
    ):
        raise VerifierError(
            f"{kind}.evidence_requests must return tuple[EvidenceRequest, ...]"
        )
    return values


def _selected_specs(
    task: Any,
    only_verifiers: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    """Return the canonical policy surface or one explicit recovery slice.

    A selective run is an evaluation-recovery tool, not a way to bypass the
    Candidate gate.  ``candidate_integrity`` therefore remains first even when
    the caller asks to recompute one expensive downstream verifier.
    """

    specs = _canonical_specs(task)
    if only_verifiers is None:
        return specs
    requested = tuple(dict.fromkeys(str(value).strip() for value in only_verifiers))
    if not requested or any(not value for value in requested):
        raise VerifierError("only_verifiers must contain non-empty verifier IDs")
    available = {str(value.get("name")) for value in specs}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise VerifierError(
            "selected verifiers are not available for this task: "
            + ", ".join(unknown)
        )
    selected = {"candidate_integrity", *requested}
    return [value for value in specs if str(value.get("name")) in selected]


def render_evidence_requests(
    task: Any,
    *,
    only_verifiers: tuple[str, ...] | None = None,
) -> tuple[EvidenceRequest, ...]:
    values = [
        request
        for spec in _selected_specs(task, only_verifiers)
        for request in evidence_requests_for_spec(task, spec)
    ]
    return merge_requests(values)


def classify(kind: str, report: dict[str, Any] | None = None) -> str:
    """Return the code-owned open-ended/GT class for a report."""

    if (
        kind == "scene_diff"
        or kind.startswith("scene_diff.")
        or kind == "gt_repair"
        or kind.startswith("gt_repair.")
    ):
        return "gt"
    return CLASSES.get(kind, "open_ended")


def _context(
    *,
    task: Any,
    record: dict[str, Any],
    ids: dict[str, str],
    spec: Mapping[str, Any],
    bridge: Any,
    images: list[str] | None,
    reference_images: list[Any] | None,
    out_dir: Any,
    scoring: Any,
    renders: Any,
    visual_renders: Any,
    render_evidence: Mapping[str, Any] | None,
    artifacts_dir: Any,
    judge_verdict: dict[str, Any] | None,
    cache: dict[Any, Any],
) -> Context:
    kind = str(spec.get("name") or "")
    formal = dict(render_evidence or {})
    if kind in CANONICAL_VERIFIER_KINDS:
        # Canonical consumers address `render_evidence` by protocol. Keep the
        # unkeyed convenience slots empty so one leaf cannot accidentally
        # consume another leaf's screenshots.
        renders = None
        visual_renders = None
    return Context(
        record=record,
        task=task,
        ids=ids,
        bridge=bridge,
        scoring=scoring,
        images=list(images or []),
        reference_images=list(reference_images or []),
        renders=renders,
        visual_renders=visual_renders,
        render_evidence=formal,
        artifacts_dir=artifacts_dir,
        out_dir=out_dir,
        judge_verdict=judge_verdict,
        spec=dict(spec),
        cache=cache,
    )


def _has_image_guidance(task: Any) -> bool:
    if str(getattr(task, "case_type", "")) == "image_to_scene":
        return True
    data = getattr(task, "data", {})
    source = data.get("source") if isinstance(data, Mapping) else None
    references = source.get("reference_views") if isinstance(source, Mapping) else None
    return isinstance(references, list) and bool(references)


def _runner_policy_kinds(task: Any) -> tuple[str, ...]:
    """Route Input preservation only to image-guided edit/repair tasks."""

    routed = ["candidate_integrity"]
    declared = {
        str(value.get("name"))
        for value in (getattr(task, "verifiers", None) or [])
        if isinstance(value, Mapping)
    }
    if (
        "gt_repair" not in declared
        and task_mode(getattr(task, "kind", "")) != "generation"
        and _has_image_guidance(task)
    ):
        routed.append("source_preservation")
    routed.append("physical_safety")
    return tuple(routed)


def _canonical_specs(task: Any) -> list[dict[str, Any]]:
    declared = [dict(value) for value in (task.verifiers or [])]
    by_name = {str(value.get("name")): value for value in declared}
    routed_kinds = _runner_policy_kinds(task)
    policies = [
        by_name.get(kind, {"name": kind}) for kind in routed_kinds
    ]
    task_owned = [
        value
        for value in declared
        if str(value.get("name")) not in routed_kinds
    ]
    return [*policies, *task_owned]


def _integrity_block_for(
    kind: str,
    gate: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return an explicit block for one verifier, retaining legacy fail-closed."""

    if gate is None:
        return None
    evidence = gate.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    eligibility = evidence.get("verifier_eligibility")
    eligibility = eligibility if isinstance(eligibility, Mapping) else {}
    entry = eligibility.get(kind)
    if isinstance(entry, Mapping) and isinstance(entry.get("eligible"), bool):
        return None if entry["eligible"] else entry
    if gate.get("status") == contracts.VALID:
        return None
    return {
        "eligible": False,
        "reason_code": "candidate_integrity_not_valid",
        "source_leaf": None,
        "reason": gate.get("failure_reason"),
    }


def run(
    task: Any,
    record: dict[str, Any],
    ids: dict[str, str],
    bridge: Any = None,
    images: list[str] | None = None,
    reference_images: list[Any] | None = None,
    out_dir: Any = None,
    scoring: Any = None,
    renders: Any = None,
    visual_renders: Any = None,
    render_evidence: Mapping[str, Any] | None = None,
    artifacts_dir: Any = None,
    judge_verdict: dict[str, Any] | None = None,
    only_verifiers: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Run a task through the canonical policy order."""

    specs = _selected_specs(task, only_verifiers)
    reports: list[dict[str, Any]] = []
    shared: dict[Any, Any] = {}
    integrity_gate: dict[str, Any] | None = None

    for spec in specs:
        kind = str(spec.get("name"))
        integrity_block = _integrity_block_for(kind, integrity_gate)
        context = _context(
            task=task,
            record=record,
            ids=ids,
            spec=spec,
            bridge=bridge,
            images=images,
            reference_images=reference_images,
            out_dir=out_dir,
            scoring=scoring,
            renders=renders,
            visual_renders=visual_renders,
            render_evidence=render_evidence,
            artifacts_dir=artifacts_dir,
            judge_verdict=judge_verdict,
            cache=shared,
        )
        verifier = REGISTRY.get(kind)
        if verifier is None:
            report = error(
                kind,
                context,
                f"no verifier named {kind!r}; available={kinds()}",
            )
        elif integrity_block is not None:
            report = blocked_report(
                kind,
                context,
                integrity_gate or {},
                eligibility=integrity_block,
            )
        else:
            try:
                report = verifier(context)
            except Exception as exc:  # noqa: BLE001 - a report must exist
                report = error(kind, context, f"{type(exc).__name__}: {exc}")

        reports.append(report)
        if kind == "candidate_integrity":
            integrity_gate = report

    for report in reports:
        report["probes_used"] = list(report.get("probes_used") or ())
    return reports


__all__ = [
    "CANONICAL_CLASSES",
    "CANONICAL_REGISTRY",
    "CLASSES",
    "Context",
    "LABEL_SUFFIX",
    "REGISTRY",
    "VerifierError",
    "classify",
    "error",
    "evidence_requests_for_spec",
    "kinds",
    "read_label",
    "render_evidence_requests",
    "run",
]
