"""Conditional Input-to-Candidate preservation policy."""

from __future__ import annotations

from typing import Any

from ..composite import composite_report, not_applicable_report
from ..context import Context, error
from ..evaluation_policy import load_evaluation_policy, task_mode
from ..policy_preservation import evaluate


CLASS = "open_ended"


def verify(context: Context) -> dict[str, Any]:
    try:
        policy = load_evaluation_policy(context.task)
    except Exception as exc:  # noqa: BLE001 - frozen policy refusal
        return error("source_preservation", context, f"{type(exc).__name__}: {exc}")
    mode = task_mode(context.task.kind)
    if mode == "generation":
        return not_applicable_report(
            "source_preservation",
            context,
            "source preservation is not applicable to from-scratch generation",
            evidence={**policy.evidence(), "task_mode": mode},
        )
    if policy.source_snapshot is None:
        return {
            **error(
                "source_preservation",
                context,
                "image-guided edit/repair policy has no frozen source_snapshot; "
                "preservation cannot be inferred from reference images, the "
                "prompt, or the agent trajectory",
            ),
            "evidence": {**policy.evidence(), "task_mode": mode},
        }

    measured = evaluate(context, policy)
    if isinstance(measured, dict):
        return measured
    return composite_report(
        "source_preservation",
        context,
        measured,
        evidence={**policy.evidence(), "task_mode": mode},
    )


__all__ = ["verify"]
