"""Score one evidence bundle offline under the paper protocol.

Judge modes (``judge``):

``live``
    VLM-dependent leaves are re-judged: Overview Alignment from the bundle's
    four overview renders, and visual Detailed Alignment requirements by
    replaying the bundle's recorded Stage 3 request schedule. Needs the judge
    endpoint (``--vlm-base-url`` / ``CODE4SCENE_VLM_BASE_URL``). A VLM-dependent
    leaf whose evidence is absent from the bundle is reported as not evaluated.
``recorded``
    The recorded judge outputs stored in the bundle are re-aggregated; no
    model is called. This is how the published numbers are reproduced.
``none`` (``--no-vlm``)
    Only structured leaves are scored: Candidate Integrity, Physical Safety,
    structurally resolved Detailed Alignment requirements and, for
    image-to-scene, Repair F1. VLM-dependent leaves are marked not evaluated
    and contribute zero, so a text-to-scene result is flagged ``partial``
    and is not comparable to the paper. Image-to-scene case scores need no
    VLM at all and remain complete.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from . import bundle as bundle_mod
from .evaluation.case_spec import validate_candidate_contract
from .protocol import constants, i2s, physics, t2s

SCHEMA_VERSION = "code4scene.case_score.v1"
JUDGE_MODES = ("live", "recorded", "none")
#: Detailed Alignment decisions resolved without a model (engine state).
STRUCTURED_STAGES = ("stage1", "stage1_structured", "stage2_structured", "structured")


class ScoringError(ValueError):
    """The bundle cannot be scored under the requested mode."""


def _structured(decision: Mapping[str, Any]) -> bool:
    return str(decision.get("resolved_by") or "").startswith(STRUCTURED_STAGES)


def _vlm_dependent(decision: Mapping[str, Any]) -> bool:
    return str(decision.get("resolved_by") or "").startswith("stage3")


def _minimum_actor_count(task: Any) -> int:
    for spec in getattr(task, "verifiers", None) or ():
        if isinstance(spec, Mapping) and spec.get("name") == "candidate_integrity":
            value = spec.get("minimum_actor_count")
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return 1


def check_integrity(bundle: bundle_mod.Bundle, task: Any = None) -> dict[str, Any]:
    """Candidate Integrity: recorded verdicts plus an offline snapshot re-check."""

    recorded = bundle.integrity
    if bundle.run.get("submission") == "missing":
        return {"status": "missing_submission", "valid": False, "recorded": recorded,
                "offline_checks": {}}
    offline: dict[str, Any] = {}
    reasons = []
    if recorded.get("status") == "valid" and not (bundle.manifest.get("scenes") or {}).get(
            "candidate"):
        offline = {"note": "no candidate snapshot in the bundle; recorded verdicts used"}
    elif recorded.get("status") == "valid":
        scene = bundle.scene("candidate")
        errors = validate_candidate_contract(scene)
        actors = len((scene or {}).get("actors") or [])
        minimum = _minimum_actor_count(task)
        offline = {"schema_errors": errors[:10], "actor_count": actors,
                   "minimum_actor_count": minimum}
        if errors:
            reasons.append("candidate snapshot is not a well-formed scene graph")
        if actors < minimum:
            reasons.append(f"candidate holds {actors} actor(s), under the minimum {minimum}")
        for leaf in bundle_mod.INTEGRITY_LEAVES:
            status = (recorded.get("leaves") or {}).get(leaf)
            if status is not None and status != "valid":
                reasons.append(f"recorded {leaf} verdict is {status!r}")
    else:
        reasons.append(recorded.get("failure_reason") or "recorded integrity verdict is invalid")
    valid = recorded.get("status") == "valid" and not reasons
    return {"status": "valid" if valid else "invalid", "valid": valid,
            "reasons": reasons, "recorded": recorded, "offline_checks": offline}


# ---------------------------------------------------------------------------
# Text-to-scene
# ---------------------------------------------------------------------------


def _detailed(bundle: bundle_mod.Bundle, judge: str,
              client_factory: Callable[[], Any] | None) -> dict[str, Any]:
    document = bundle.decisions_document() or {}
    decisions = list(document.get("decisions") or [])
    if not decisions:
        return {"status": "not_evaluated", "score": None,
                "reason": "the bundle holds no Detailed Alignment decisions"}
    if not t2s.usable({"status": document.get("report_status")}):
        return {"status": "not_evaluated", "score": None,
                "reason": f"the Semantic report was {document.get('report_status')!r}; "
                          "an unavailable component scores zero at its fixed weight"}
    visual = [d for d in decisions if _vlm_dependent(d)]
    note = None
    rows = decisions
    if judge == "live" and visual:
        plan = bundle.stage3_plan()
        if plan is None:
            rows, note = _drop_visual(decisions), (
                "no Stage 3 request schedule in the bundle; visual requirements "
                "were not re-judged")
        else:
            rows, note = _replay_stage3(bundle, plan, client_factory), "re-judged by the VLM"
    elif judge == "none" and visual:
        rows, note = _drop_visual(decisions), "visual requirements need the VLM judge"
    result = t2s.detailed_from_decisions(rows)
    result["visual_requirement_count"] = len(visual)
    result["structured_requirement_count"] = sum(_structured(d) for d in decisions)
    if note:
        result["note"] = note
    if judge != "recorded" and visual and note != "re-judged by the VLM":
        # Skipped by the scoring mode, not missing from the evidence.
        result["status"], result["skipped"] = "partial", True
    return result


def _drop_visual(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in copy.deepcopy(decisions):
        if _vlm_dependent(row):
            row.update(evaluation_status="NOT_EVALUATED", effective_score=0.0,
                       score_known=False, not_evaluated_reason="needs_vlm_judge")
        out.append(row)
    return out


def _replay_stage3(bundle: bundle_mod.Bundle, plan: Mapping[str, Any],
                   client_factory: Callable[[], Any] | None) -> list[dict[str, Any]]:
    """Re-issue the recorded Stage 3 request schedule against the live judge."""

    from .evaluation import semantic_repeatability as replay
    from .evaluation.requirement_graph.vlm_client import tool_client_from_env

    plan = copy.deepcopy(dict(plan))
    for spec in plan["frames"].values():
        spec["path"] = str(bundle.path(spec["path"]))
    frames = {}
    for key, spec in plan["frames"].items():
        spec.setdefault("png_sha256", bundle_mod.sha256_file(bundle.path(spec["path"])))
        frames[key] = replay.load_frame(spec)
    factory = client_factory or tool_client_from_env
    outcomes = [replay.replay_claim(plan, claim, frames, factory) for claim in plan["claims"]]
    scored = replay.score_replayed(plan, outcomes)
    return list(scored["requirements"])


def _overview(bundle: bundle_mod.Bundle, task: Any, judge: str,
              client_factory: Callable[[], Any] | None) -> dict[str, Any]:
    if judge == "none":
        return {"status": "not_evaluated", "score": None, "skipped": True,
                "reason": "Overview Alignment needs the VLM judge"}
    if judge == "recorded":
        recorded = bundle.overview_judgement()
        if not recorded:
            return {"status": "not_evaluated", "score": None,
                    "reason": "the bundle holds no recorded overview judgement"}
        return t2s.overview_from_judgement(
            recorded["dimension_scores"], recorded["structural_integrity_score"],
            severe_cap_eligible=bool(recorded.get("severe_structural_cap_eligible")))
    views = bundle.overview_views()
    if len(views) != 4:
        return {"status": "not_evaluated", "score": None, "skipped": True,
                "reason": f"live Overview Alignment needs four overview views, bundle has "
                          f"{len(views)}"}
    from .evaluation import offline_overview_prompt_alignment as overview
    from .evaluation.requirement_graph.vlm_client import tool_client_from_env

    prompt = getattr(task, "prompt", None)
    if not prompt:
        raise ScoringError("live Overview Alignment needs the task prompt (--task)")
    measured = overview.evaluate_overview_frames(
        prompt, [{"path": str(p), "frame_id": f"overview_{i + 1}"} for i, p in enumerate(views)],
        model_label=str(bundle.run.get("model") or "candidate"),
        client_factory=client_factory or tool_client_from_env)
    judged = t2s.overview_from_judgement(
        {k: v["score"] for k, v in measured["dimensions"].items()},
        measured["structural_integrity_score"],
        severe_cap_eligible=bool(measured["severe_structural_cap_eligible"]))
    return {**judged, "judge_record": measured}


def _score_t2s(bundle, task, integrity, judge, client_factory) -> dict[str, Any]:
    if not integrity["valid"]:
        case = t2s.case_score(valid=False, detailed=None, overview=None, physics=None)
        return {"case": case, "components": {}}
    phys = physics.t2s_score(bundle.scene("candidate"), bundle.physics_report())
    detailed = _detailed(bundle, judge, client_factory)
    overview = _overview(bundle, task, judge, client_factory)
    case = t2s.case_score(valid=True, detailed=detailed.get("score"),
                          overview=overview.get("score"), physics=phys["score"])
    return {"case": case,
            "components": {"detailed_alignment": detailed, "overview_alignment": overview,
                           "physical_safety": phys}}


# ---------------------------------------------------------------------------
# Image-to-scene
# ---------------------------------------------------------------------------


def _score_i2s(bundle, integrity) -> dict[str, Any]:
    scored = i2s.score_case(
        valid=integrity["valid"], input_scene=bundle.scene("input") if integrity["valid"] else None,
        ground_truth_scene=bundle.scene("ground_truth") if integrity["valid"] else None,
        candidate_scene=bundle.scene("candidate") if integrity["valid"] else None,
        physical_safety_report=bundle.physics_report())
    case = {k: scored[k] for k in ("policy", "status", "score", "repair_f1", "physics")}
    components = {"repair_f1": scored.get("actor_f1"),
                  "physical_safety": scored.get("physics_detail")}
    return {"case": case, "components": components,
            "audit": {"actor_f1": scored.get("actor_f1_audit")}}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def score_bundle(
    bundle: bundle_mod.Bundle,
    task: Any = None,
    *,
    judge: str = "live",
    client_factory: Callable[[], Any] | None = None,
    include_audit: bool = False,
) -> dict[str, Any]:
    """Score one bundle; returns a ``code4scene.case_score.v1`` record."""

    if judge not in JUDGE_MODES:
        raise ScoringError(f"judge must be one of {JUDGE_MODES}")
    task_check = None
    if task is not None:
        task_check = {"task_id_matches": getattr(task, "id", None) == bundle.task_id,
                      "task_sha256": getattr(task, "sha256", None),
                      "bundle_task_sha256": (bundle.manifest.get("task") or {}).get("sha256")}
        if not task_check["task_id_matches"]:
            raise ScoringError(
                f"task id {getattr(task, 'id', None)!r} does not match bundle "
                f"{bundle.task_id!r}")
    integrity = check_integrity(bundle, task)
    if bundle.is_t2s:
        scored = _score_t2s(bundle, task, integrity, judge, client_factory)
    else:
        scored = _score_i2s(bundle, integrity)
    case = scored["case"]
    status = case["status"]
    if integrity["status"] == "missing_submission":
        status = "zero_missing_submission"
    # A component the chosen mode skipped makes the case partial (not a paper
    # score). Evidence that is genuinely unavailable scores zero at its fixed
    # weight, as in the paper, and keeps the case comparable.
    partial = [name for name, comp in scored["components"].items()
               if isinstance(comp, Mapping) and comp.get("skipped")]
    if partial and status.startswith("measured"):
        status = "partial"
    record = {
        "schema_version": SCHEMA_VERSION,
        "task_id": bundle.task_id,
        "setting": bundle.setting,
        "model": bundle.run.get("model"),
        "status": status,
        "comparable_to_paper": not partial,
        "score": case["score"],
        "case": case,
        "not_evaluated_components": partial,
        "judge_mode": judge,
        "integrity": integrity,
        "components": scored["components"],
        "policies": {
            "t2s_case": constants.T2S_CASE_POLICY, "i2s_case": constants.I2S_CASE_POLICY,
            "actor_f1": constants.ACTOR_F1_POLICY, "physics": constants.PHYSICS_POLICY,
            "t2s_floating": constants.T2S_FLOATING_METRIC,
        },
        "task_check": task_check,
    }
    if judge == "live":
        from .evaluation import vlm_model_config

        record["judge"] = {"model": vlm_model_config.model(),
                           "base_url_configured": bool(vlm_model_config.base_url())}
    if include_audit and scored.get("audit"):
        record["audit"] = scored["audit"]
    return record


# ---------------------------------------------------------------------------
# The verifier layer's view of the same case
# ---------------------------------------------------------------------------


def _replace_report(reports: list, report_id: str, report: Mapping[str, Any]) -> None:
    for index, value in enumerate(reports):
        if isinstance(value, Mapping) and value.get("report_id") == report_id:
            reports[index] = dict(report)
            return
    reports.append(dict(report))


def rebuild_result(
    result: Mapping[str, Any],
    bundle: bundle_mod.Bundle,
    *,
    semantic_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild a saved result.json through the verifier layer.

    The measurements the paper protocol takes from scene snapshots are
    re-measured from the bundle with the verifiers' own functions (Actor Repair
    F1 into the gt_repair report; the text-to-scene support leaf into the
    physical_safety report), then the result's primary score is assembled
    exactly as a live evaluation assembles it. Its ``overall_score`` must equal
    the protocol case score of :func:`score_bundle`.
    """

    from .evaluation import primary_score, score_policy
    from .evaluation.verifiers import floating, gt_repair
    from .resources import config_file

    rebuilt = copy.deepcopy(dict(result))
    reports = [dict(r) for r in rebuilt.get("reports") or () if isinstance(r, Mapping)]
    rebuilt["reports"] = reports
    by_id = {r.get("report_id"): r for r in reports}
    if bundle.is_t2s:
        if semantic_report is not None:
            _replace_report(reports, "semantic_requirements", semantic_report)
        physical = by_id.get("physical_safety")
        scene = bundle.scene("candidate") if (bundle.manifest.get("scenes") or {}).get(
            "candidate") else None
        if isinstance(physical, dict) and scene is not None:
            physical = copy.deepcopy(physical)
            leaves = (physical.get("metrics") or {}).get("leaf_results") or []
            for index, leaf in enumerate(leaves):
                if leaf.get("leaf_id") != "floating":
                    continue
                ids = {"task_bundle_id": leaf.get("task_bundle_id") or "offline",
                       "episode_id": leaf.get("episode_id") or "offline"}
                fresh = floating.snapshot_leaf(
                    scene, ids, live_rate=(leaf.get("metrics") or {}).get("floating_rate"),
                    artifacts=leaf.get("artifacts"))
                if fresh is not None:
                    leaves[index] = {**leaf, **fresh, "report_id": leaf.get("report_id"),
                                     "leaf_id": "floating"}
            physical["score"] = physics.score_from_report(physical)["score"]
            _replace_report(reports, "physical_safety", physical)
        policy = score_policy.load(config_file(
            "score-policies", "text-to-scene-human-aligned.yaml"))
        return score_policy.apply_to_result(rebuilt, policy)
    repair = by_id.get("gt_repair")
    if isinstance(repair, dict) and check_integrity(bundle)["valid"]:
        repair = copy.deepcopy(repair)
        repair.setdefault("metrics", {})["actor_repair_f1"] = gt_repair.measure_actor_repair_f1(
            bundle.scene("input"), bundle.scene("ground_truth"), bundle.scene("candidate"))
        _replace_report(reports, "gt_repair", repair)
    environment = bundle.setting.rsplit("/", 1)[-1]
    return primary_score.apply_to_result(rebuilt, scene_environment=environment)


def verifier_layer_summary(rebuilt: Mapping[str, Any], record: Mapping[str, Any]) -> dict:
    """Compare the rebuilt result's single overall number with the case score."""

    primary = rebuilt.get("primary_score") or {}
    overall = rebuilt.get("overall_score")
    score = record.get("score")
    return {
        "overall_score": overall, "primary_status": primary.get("status"),
        "primary_source_id": primary.get("source_id"),
        "reason_codes": list(primary.get("reason_codes") or ()),
        "consistent": overall is not None and score is not None
        and abs(float(overall) - float(score)) == 0.0,
    }


def case_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a case-score record into the row shape ``aggregate`` reads."""

    case = record.get("case") or {}
    comp = record.get("components") or {}
    f1 = comp.get("repair_f1") or {}
    return {"model": record.get("model"), "setting": record.get("setting"),
            "case_id": record.get("task_id"), "score": record.get("score"),
            "status": "measured" if str(record.get("status")).startswith("measured")
            else record.get("status"),
            "repair_f1": case.get("repair_f1"), "physics": case.get("physics"),
            "precision": f1.get("precision"), "recall": f1.get("recall")}


__all__ = ["JUDGE_MODES", "SCHEMA_VERSION", "ScoringError", "case_row", "check_integrity",
           "rebuild_result", "score_bundle"]
