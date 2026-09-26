"""Deterministic RequirementGraph rule evaluators.

These are internal Stage 1 implementations, not independently selectable
verifiers.  ``semantic_requirements`` is the only public semantic verifier.
"""

from . import attribute, count, object_architecture, object_object
from .contracts import FAMILIES, RuleEvaluationContext

EVALUATORS = {
    "CNT": count.evaluate_item,
    "ATR": attribute.evaluate_item,
    "OAR": object_architecture.evaluate_item,
    "OOR": object_object.evaluate_item,
}

__all__ = ["EVALUATORS", "FAMILIES", "RuleEvaluationContext"]
