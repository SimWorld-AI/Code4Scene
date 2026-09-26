"""Canonical RequirementGraph engine merged into Code4Scene.

This is shared evaluation machinery, not an independently registered verifier.
The flat verifier entry points under :mod:`code4scene.evaluation.verifiers`
adapt its atomic results to Code4Scene reports.
"""

from .contracts import GraphValidationError, RequirementGraph, SceneBounds
from .evidence_adapter import (
    SceneInventoryEvidence,
    ScopeResolution,
    build_scene_inventory,
)
from .runtime import CapturedFrame, FrameProvider, RgbFrameHealthError
from .stage0 import Stage0Compiler, compile_semantic_draft
from .stage1 import Stage1Result, evaluate_stage1
from .stage2 import Stage2Evaluation, evaluate_stage2_detailed
from .stage3 import Stage3Evaluation, evaluate_stage3_detailed

__all__ = [
    "CapturedFrame",
    "FrameProvider",
    "GraphValidationError",
    "RequirementGraph",
    "RgbFrameHealthError",
    "SceneBounds",
    "SceneInventoryEvidence",
    "ScopeResolution",
    "Stage0Compiler",
    "Stage1Result",
    "Stage2Evaluation",
    "Stage3Evaluation",
    "compile_semantic_draft",
    "build_scene_inventory",
    "evaluate_stage1",
    "evaluate_stage2_detailed",
    "evaluate_stage3_detailed",
]
