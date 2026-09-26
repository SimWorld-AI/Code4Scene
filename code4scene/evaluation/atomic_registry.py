"""Atomic evaluator registry.

Atomic evaluators own one measurement and one normalization policy. Composite
verifiers may orchestrate them, but cannot silently average their results or
collapse unavailable evidence into a failed scene.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import contracts
from .context import Context
from .render_evidence import CAPTION_ENVIRONMENT_RENDER_PROTOCOL


@dataclass(frozen=True, slots=True)
class AtomicDescriptor:
    evaluator_id: str
    evaluator_version: str
    dimension: str
    unit: str
    required_evidence: tuple[str, ...]
    verifier_module: str
    applicability_policy: str


class AtomicReportEvaluator:
    """Compatibility shell while an atom's native implementation is migrated.

    It calls the existing exact algorithm once, then validates its complete
    report through the canonical raw/normalized metric contracts. This is an
    adapter around measurement logic, not a second implementation of it.
    """

    def __init__(self, descriptor: AtomicDescriptor):
        self.descriptor = descriptor

    def _verifier(self) -> Any:
        module = importlib.import_module(
            f"code4scene.evaluation.verifiers.{self.descriptor.verifier_module}"
        )
        verifier = getattr(module, "verify", None)
        if not callable(verifier):
            raise TypeError(
                f"{self.descriptor.verifier_module} exports no verify(context)"
            )
        return verifier

    def measure(self, evidence: Context) -> contracts.RawMeasurement[dict[str, Any]]:
        report = self._verifier()(evidence)
        status = str(report.get("status"))
        instance_id = (
            f"{self.descriptor.evaluator_id}:"
            f"{evidence.ids.get('episode_id', 'episode')}"
        )
        envelope = {"atomic_report": copy.deepcopy(report)}
        if status in {contracts.MEASURED, contracts.PASS, contracts.FAIL}:
            return contracts.RawMeasurement(
                id=self.descriptor.evaluator_id,
                instance_id=instance_id,
                metric_version=self.descriptor.evaluator_version,
                applicable=True,
                status="measured",
                coverage=1.0,
                raw=envelope,
            )
        if status == "not_applicable":
            return contracts.RawMeasurement(
                id=self.descriptor.evaluator_id,
                instance_id=instance_id,
                metric_version=self.descriptor.evaluator_version,
                applicable=False,
                status="not_applicable",
                coverage=None,
                raw=None,
                evidence=(envelope,),
            )
        if status not in {contracts.ERROR, "not_evaluated"}:
            raise ValueError(
                f"{self.descriptor.evaluator_id} returned unsupported status {status!r}"
            )
        return contracts.RawMeasurement(
            id=self.descriptor.evaluator_id,
            instance_id=instance_id,
            metric_version=self.descriptor.evaluator_version,
            applicable=True,
            status="error" if status == contracts.ERROR else "not_evaluated",
            coverage=None,
            raw=None,
            evidence=(envelope,),
            failure_reason=str(
                report.get("failure_reason") or "atomic evaluator did not complete"
            ),
        )

    def normalize(
        self,
        measurement: contracts.RawMeasurement[dict[str, Any]],
        policy: Mapping[str, Any],
    ) -> contracts.MetricResult[dict[str, Any]]:
        if (
            measurement.id != self.descriptor.evaluator_id
            or measurement.metric_version != self.descriptor.evaluator_version
        ):
            raise ValueError("atomic normalizer received another evaluator's result")
        report = (
            (measurement.raw or {}).get("atomic_report")
            or (measurement.evidence[0].get("atomic_report") if measurement.evidence else {})
        )
        common = {
            "id": self.descriptor.evaluator_id,
            "instance_id": measurement.instance_id,
            "metric_version": self.descriptor.evaluator_version,
            "dimension": self.descriptor.dimension,
            "applicability_policy": self.descriptor.applicability_policy,
            "required_evidence": self.descriptor.required_evidence,
            "normalization_parameters": dict(policy),
            "calibration_status": "continuous_score_without_pass_fail_threshold",
            "contributes_to_aggregate": False,
            "evidence": measurement.evidence,
        }
        if measurement.status == "not_applicable":
            return contracts.MetricResult(
                **common,
                applicable=False,
                status="not_applicable",
                coverage=None,
                raw=None,
                score=None,
                normalization_policy=None,
            )
        if measurement.status in {"not_evaluated", "error"}:
            return contracts.MetricResult(
                **common,
                applicable=True,
                status=measurement.status,
                coverage=measurement.coverage,
                raw=None,
                score=None,
                normalization_policy=None,
                failure_reason=measurement.failure_reason,
            )
        score = report.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(
                f"{self.descriptor.evaluator_id}: measured report needs numeric score"
            )
        return contracts.MetricResult(
            **common,
            applicable=True,
            status=contracts.MEASURED,
            coverage=1.0,
            raw={
                "unit": self.descriptor.unit,
                "metrics": report.get("metrics") or {},
            },
            score=float(score),
            normalization_policy="existing_continuous_atomic_score",
            failure_reason=None,
        )

    def evaluate_report(
        self, context: Context, policy: Mapping[str, Any]
    ) -> dict[str, Any]:
        measurement = self.measure(context)
        result = self.normalize(measurement, policy)
        report = (
            (measurement.raw or {}).get("atomic_report")
            or (measurement.evidence[0].get("atomic_report") if measurement.evidence else {})
        )
        output = copy.deepcopy(report)
        metrics = dict(output.get("metrics") or {})
        metrics["atomic_result"] = result.to_json_dict()
        output["metrics"] = metrics
        output["status"] = result.status
        output["score"] = result.score
        if result.status in {"error", "not_evaluated", "not_applicable"}:
            output["score"] = None
        if result.status == contracts.MEASURED:
            output.pop("failure_reason", None)
        elif result.failure_reason:
            output["failure_reason"] = result.failure_reason
        return output


class CaptionAtomicEvaluator:
    """Native image -> captions -> embeddings -> cosine atomic evaluator."""

    def __init__(self, descriptor: AtomicDescriptor):
        self.descriptor = descriptor

    @staticmethod
    def _module() -> Any:
        return importlib.import_module(
            "code4scene.evaluation.verifiers.gt_caption_similarity"
        )

    def measure(self, evidence: Context) -> contracts.RawMeasurement[dict[str, Any]]:
        return self._module().measure(evidence)

    def normalize(
        self,
        measurement: contracts.RawMeasurement[dict[str, Any]],
        policy: Mapping[str, Any],
    ) -> contracts.MetricResult[dict[str, Any]]:
        return self._module().normalize(measurement, policy)

    def evaluate_report(
        self, context: Context, policy: Mapping[str, Any]
    ) -> dict[str, Any]:
        measurement = self.measure(context)
        result = self.normalize(measurement, policy)
        return self._module().report_from_atomic(context, measurement, result)


_PHYSICS_POLICY = "frozen_evaluation_policy.physics_profile"
_BINDING_POLICY = "frozen_requirement_graph.evaluation_binding"

_CAPTION_DESCRIPTOR = AtomicDescriptor(
    "caption.independent_caption_embedding_distance",
    "gt-caption-cosine-v1",
    "independent caption embedding distance",
    "cosine_distance",
    (
        f"paired_render:{CAPTION_ENVIRONMENT_RENDER_PROTOCOL}",
        "caption_model",
        "text_embedding_model",
    ),
    "gt_caption_similarity",
    "gt_paired_caption_continuous",
)

_DESCRIPTORS = (
    AtomicDescriptor("physics.floating", "floating-v1", "unsupported scene actors", "ratio", ("measure_scene",), "floating", _PHYSICS_POLICY),
    AtomicDescriptor("physics.ground_gap", "ground-gap-v1", "actor-to-ground separation", "cm", ("candidate_scene_graph", "ue_physics_measurement"), "ground_gap", _PHYSICS_POLICY),
    AtomicDescriptor("physics.out_of_bounds", "out-of-bounds-v1", "actors outside frozen bounds", "ratio", ("candidate_measurement", "world_bounds"), "out_of_bounds", _PHYSICS_POLICY),
    AtomicDescriptor("physics.bounds_discipline", "bounds-discipline-v1", "harness bounds corrections", "actor_count", ("episode_bounds_passes",), "bounds_discipline", _PHYSICS_POLICY),
    AtomicDescriptor("physics.solid_penetration", "solid-penetration-v3", "eligible solid Actor collision-free rate", "ratio", ("candidate_scene_graph", "ue_physics_measurement"), "solid_penetration", _PHYSICS_POLICY),
    AtomicDescriptor("physics.environment_consistency", "environment-consistency-v1", "environment/category consistency", "actor_count", ("candidate_scene_graph", "ue_physics_measurement"), "environment_consistency", _PHYSICS_POLICY),
    AtomicDescriptor("semantic.structure_count", "structure-count-v1", "required Actor count", "actor_count", ("candidate_scene_graph", "source_snapshot", "evaluation_binding"), "structure_count", _BINDING_POLICY),
    AtomicDescriptor("semantic.structure_concepts", "structure-concepts-v1", "required concept count", "ratio", ("candidate_scene_graph", "source_snapshot", "evaluation_binding"), "structure_concepts", _BINDING_POLICY),
    AtomicDescriptor("semantic.structure_additions", "structure-additions-v1", "required and forbidden additions", "actor_count", ("candidate_scene_graph", "source_snapshot", "evaluation_binding"), "structure_additions", _BINDING_POLICY),
    AtomicDescriptor("semantic.spatial_overlap", "spatial-overlap-v1", "task-specific overlap", "ratio", ("candidate_scene_graph", "evaluation_binding"), "spatial_overlap", _BINDING_POLICY),
    AtomicDescriptor("semantic.spatial_clearance", "spatial-clearance-v1", "task-specific clearance", "cm", ("candidate_scene_graph", "evaluation_binding"), "spatial_clearance", _BINDING_POLICY),
    AtomicDescriptor("semantic.spatial_cluster", "spatial-cluster-v1", "task-specific compactness", "cm", ("candidate_scene_graph", "evaluation_binding"), "spatial_cluster", _BINDING_POLICY),
    AtomicDescriptor("semantic.spatial_relations", "spatial-relations-v1", "task-specific spatial relation", "ratio", ("candidate_scene_graph", "evaluation_binding"), "spatial_relations", _BINDING_POLICY),
)

REGISTRY = {
    value.evaluator_id.rsplit(".", 1)[-1]: AtomicReportEvaluator(value)
    for value in _DESCRIPTORS
}
REGISTRY["independent_caption_embedding_distance"] = CaptionAtomicEvaluator(
    _CAPTION_DESCRIPTOR
)


def kinds(namespace: str | None = None) -> tuple[str, ...]:
    return tuple(
        sorted(
            key
            for key, evaluator in REGISTRY.items()
            if namespace is None
            or evaluator.descriptor.evaluator_id.startswith(f"{namespace}.")
        )
    )


def get(evaluator_id: str) -> AtomicReportEvaluator | CaptionAtomicEvaluator:
    try:
        return REGISTRY[evaluator_id]
    except KeyError as exc:
        raise KeyError(f"unknown atomic evaluator {evaluator_id!r}") from exc


__all__ = [
    "AtomicDescriptor",
    "AtomicReportEvaluator",
    "CaptionAtomicEvaluator",
    "REGISTRY",
    "get",
    "kinds",
]
