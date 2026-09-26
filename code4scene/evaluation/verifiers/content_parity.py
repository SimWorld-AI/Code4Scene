"""``content_parity`` — could the scoring editor open what the agent saved.

A `.umap` stores REFERENCES. The material graph, the mesh, the texture are
separate packages and the map records only their paths, so a scene that
renders perfectly in the editor that built it can open to nothing in the next
one — while the run record, holding that map and a screenshot taken before it
moved, says the scene was fine.

Three different causes produce that same empty scene, and a score cannot tell
them apart. All three were observed on one grid:

* **the agent's failure.** It created sixteen materials, including the one
  under the whole square, and never saved them. They lived in the editor's
  memory for the rest of the episode; the screenshot was correct; the packages
  never reached disk. Only a second editor can see this — measure in the
  agent's own and the scene looks finished.
* **ours.** The path resolved when the scene was built and does not resolve
  when it is scored, because the two environments mounted different content.
  Nothing in the record distinguished that from an agent inventing a path, so
  it was charged to the model.
* **the content store's.** A material instance shipped without the texture it
  samples — present in both environments, missing in both.

This verifier separates them, and refuses rather than scoring: a scene nobody
could open is not a badly built scene. What it reports is which packages did
not resolve, and whether the two environments agree on the content release
they were provisioned from.

It is deliberately NOT a rate. "Nine of four hundred references are missing"
is not nine per cent of a scene — the nine may be the ground and the sky.

Serves component `execution.candidate_integrity`.
"""

from __future__ import annotations

from typing import Any

from .. import contracts
from ..context import Context, error


DOWNSTREAM_VERIFIER_KINDS = (
    "source_preservation",
    "physical_safety",
    "semantic_requirements",
    "overview_prompt_alignment",
    "scene_diff",
    "gt_repair",
)
VISUAL_VERIFIER_KINDS = tuple(
    value for value in DOWNSTREAM_VERIFIER_KINDS if value != "physical_safety"
)


def _package_key(value: Any) -> str:
    package = str(value or "").strip().strip("'\"")
    if "'" in package:
        package = package.split("'", 1)[-1].rstrip("'")
    tail = package.rsplit("/", 1)[-1]
    if "." in tail:
        package = package[: -(len(tail) - tail.index("."))]
    return package.casefold()


def _package_root(value: str) -> str | None:
    prefix = "/Game/"
    if not value.startswith(prefix):
        return None
    return value[len(prefix) :].split("/", 1)[0].casefold()


def _visual_only_dependency(value: str) -> bool:
    """Conservatively identify dependencies that cannot affect geometry."""

    lowered = value.casefold()
    leaf = lowered.rsplit("/", 1)[-1]
    return (
        "/materials/" in lowered
        or "/master_mat/" in lowered
        or "/textures/" in lowered
        or "/materialfunctions/" in lowered
        or leaf.startswith(("m_", "mi_", "mat_", "t_", "mf_"))
    )


def _blocked_verifiers(packages: list[str], *, block_all: bool = False) -> list[str]:
    if not packages and not block_all:
        return []
    if block_all or any(not _visual_only_dependency(value) for value in packages):
        return list(DOWNSTREAM_VERIFIER_KINDS)
    return list(VISUAL_VERIFIER_KINDS)


def _scoring_release(context: Context) -> str | None:
    """The release the SCORING environment was provisioned from."""
    scoring = context.scoring
    saved = getattr(scoring, "saved_scenes_dir", None) if scoring else None
    if saved is None:
        return None
    import json
    from pathlib import Path

    marker = Path(saved).parent.parent / ".scenebench-pristine.json"
    try:
        return str(json.loads(marker.read_text()).get("release") or "") or None
    except (OSError, ValueError):
        return None


def verify(context: Context) -> dict[str, Any]:
    manifest = context.record.get("scene_dependencies")
    if not isinstance(manifest, dict):
        blocked = list(DOWNSTREAM_VERIFIER_KINDS)
        return {
            **error(
                "content_parity",
                context,
                "this run recorded no dependency manifest, so whether the saved "
                "level can be opened anywhere else is unknown; the episode "
                "collects one from the editor that saved the scene",
            ),
            "evidence": {
                "dependency_classification": "dependency_evidence_unavailable",
                "environment_readiness": "unknown",
                "blocked_verifiers": blocked,
                "eligible_verifiers": [],
            },
            "probes_used": ("scene_dependencies",),
            "artifacts": contracts.artifacts(context.record),
        }

    unresolved = sorted({str(value) for value in (manifest.get("unresolved") or [])})
    integrity_allowance = context.record.get("dependency_integrity_allowance")
    integrity_allowance = (
        integrity_allowance if isinstance(integrity_allowance, dict) else {}
    )
    calibration = context.record.get("dependency_calibration_override")
    calibration = calibration if isinstance(calibration, dict) else {}
    allowance = (
        integrity_allowance
        if integrity_allowance.get("mode")
        == "shared_baseline_missing_dependencies.v2"
        else calibration
    )
    allowed = (
        {str(value) for value in (allowance.get("allowed_unresolved") or [])}
        if allowance.get("mode")
        in {
            "shared_baseline_missing_dependencies.v1",
            "shared_baseline_missing_dependencies.v2",
        }
        else set()
    )
    ignored = sorted(set(unresolved) & allowed)
    effective_unresolved = sorted(set(unresolved) - allowed)
    source_manifest = context.record.get("candidate_dependency_manifest")
    source_manifest = source_manifest if isinstance(source_manifest, dict) else {}
    resolved_at_generation = {
        _package_key(value)
        for value in (source_manifest.get("resolved_packages") or [])
        if _package_key(value)
    }
    unmounted_roots = {
        str(value).casefold()
        for value in (source_manifest.get("unmounted_resolved_packs") or [])
    }
    environment_unresolved = sorted(
        value
        for value in effective_unresolved
        if (
            _package_key(value) in resolved_at_generation
            or _package_root(value) in unmounted_roots
        )
    )
    intrinsic_unresolved = sorted(
        set(effective_unresolved) - set(environment_unresolved)
    )
    built = manifest.get("release")
    scoring = _scoring_release(context)
    drifted = bool(built and scoring and built != scoring)
    blocked = _blocked_verifiers(effective_unresolved, block_all=drifted)
    eligible = [
        value for value in DOWNSTREAM_VERIFIER_KINDS if value not in blocked
    ]
    if drifted:
        classification = "content_release_drift"
        readiness = "error"
        reason_code = "content_release_drift"
    elif environment_unresolved and intrinsic_unresolved:
        classification = "mixed_dependency_missing"
        readiness = "degraded"
        reason_code = "mixed_candidate_and_environment_dependency_missing"
    elif environment_unresolved:
        classification = "dependency_source_missing"
        readiness = "degraded"
        reason_code = "dependency_source_missing"
    elif intrinsic_unresolved:
        classification = "candidate_dependency_missing"
        readiness = "ready"
        reason_code = "candidate_dependency_missing"
    else:
        classification = "complete"
        readiness = "ready"
        reason_code = None

    evidence = {
        "package_count": manifest.get("package_count"),
        "unresolved_count": len(unresolved),
        # Capped in the report, complete in the manifest beside the run: a
        # scene that lost a whole pack would otherwise push everything else
        # out of the record.
        "unresolved": sorted(unresolved)[:40],
        "built_against_release": built,
        "scored_against_release": scoring,
        "bytes_referenced": manifest.get("bytes"),
        "dependency_classification": classification,
        "environment_readiness": readiness,
        "reason_code": reason_code,
        "candidate_unresolved": intrinsic_unresolved,
        "environment_unresolved": environment_unresolved,
        "resolved_at_generation_count": len(resolved_at_generation),
        "generation_dependency_manifest": {
            "manifest": source_manifest.get("manifest"),
            "present": source_manifest.get("present"),
            "read_error": source_manifest.get("read_error"),
            "content_root": source_manifest.get("content_root"),
            "unmounted_resolved_packs": sorted(unmounted_roots),
        },
        "blocked_verifiers": blocked,
        "eligible_verifiers": eligible,
        "dependency_impact": (
            "scene_graph_or_geometry"
            if blocked == list(DOWNSTREAM_VERIFIER_KINDS) and effective_unresolved
            else "visual_appearance_only"
            if effective_unresolved
            else "none"
        ),
    }
    if allowed:
        calibration_mode = (
            allowance.get("mode")
            == "shared_baseline_missing_dependencies.v1"
        )
        evidence.update({
            "calibration_only": bool(
                allowance.get("calibration_only", calibration_mode)
            ),
            "authoritative": bool(
                allowance.get("authoritative", not calibration_mode)
            ),
            "dependency_allowance_mode": allowance.get("mode"),
            "allowed_unresolved": sorted(allowed),
            "ignored_unresolved": ignored,
            "effective_unresolved": effective_unresolved,
        })
        if allowance is calibration:
            evidence.update({
                "calibration_override_mode": calibration.get("mode"),
                "calibration_allowed_unresolved": sorted(allowed),
                "calibration_ignored_unresolved": ignored,
            })
    metrics = {"unresolved_count": len(unresolved)}
    if allowed:
        metrics.update({
            "ignored_unresolved_count": len(ignored),
            "effective_unresolved_count": len(effective_unresolved),
        })
        if allowance is calibration:
            metrics["calibration_ignored_unresolved_count"] = len(ignored)
    report = {**contracts.base("content_parity", context.ids),
              "metrics": metrics,
              "evidence": evidence,
              "artifacts": contracts.artifacts(context.record),
              "probes_used": ("scene_dependencies",)}

    # Environment drift first: it explains missing packages, and reporting the
    # symptom over the cause is what charged the store's mistakes to the model.
    # ERROR must carry no score: `contracts.score_for_aggregate` raises on a
    # withheld status with a number attached, so a 0.0 here crashed the
    # aggregate exactly when this verifier caught the drift it exists for.
    if drifted:
        return {**report, "status": contracts.ERROR, "score": None,
                "failure_reason": (
                    f"the scene was built against content release {built!r} and "
                    f"scored against {scoring!r}; a reference that resolves in "
                    f"one and not the other is a difference between the two "
                    f"environments, not something the agent did")}
    if environment_unresolved and not intrinsic_unresolved:
        return {
            **report,
            "status": contracts.ERROR,
            "score": None,
            "failure_reason": (
                f"{len(environment_unresolved)} package(s) resolved during generation "
                "but are unavailable in the scorer environment: "
                f"{', '.join(environment_unresolved[:5])}"
                f"{' …' if len(environment_unresolved) > 5 else ''}. This is an "
                "evaluation-environment/data-retention failure, not a Candidate defect"
            ),
        }
    if intrinsic_unresolved:
        environment_suffix = (
            f"; {len(environment_unresolved)} additional package(s) were lost from "
            "the evaluation environment"
            if environment_unresolved
            else ""
        )
        return {
            **report,
            "status": contracts.INVALID,
            "score": None,
            "failure_reason": (
                f"{len(intrinsic_unresolved)} Candidate package reference(s) were "
                "not resolved by the generation record and are absent in the scorer: "
                f"{', '.join(intrinsic_unresolved[:5])}"
                f"{' …' if len(intrinsic_unresolved) > 5 else ''}"
                f"{environment_suffix}. A Candidate-created asset may have existed "
                "in editor memory and never saved it to disk"
            ),
        }
    return {**report, "status": contracts.VALID, "score": None}


__all__ = ["verify"]
