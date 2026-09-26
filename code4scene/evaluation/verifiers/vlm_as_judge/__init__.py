"""Shared internal model-judge implementation under a frozen policy.

Some things worth measuring are not countable: whether a scene depicts what
was asked, whether it reads as a place. A model judge can assess those, but a
judge that can be talked into a high score is worse than no judge — so this
package is mostly about the constraints:

* the packaged rubric and its hash go into the verdict;
* the judge sees several viewpoints, not one flattering angle;
* the judge is told that text inside the scene is scenery, because an agent
  can spawn a billboard reading "perfect scene, score 10";
* any optional score ceilings live in the packaged rubric rather than code.
  The GT-paired rubric deliberately has none: physics and semantic defects
  are already separate formal verifier reports and must not be counted twice.
"""


from .rubric import Rubric, RubricError, load_rubric
from .policy import JudgePolicy, JudgePolicyError, load_policy
from .judge import (Backend, EvidenceImage, Judge, JudgeError, JudgeRequest, Verdict,
                    verdict_schema)
from . import model_config


def _api_key() -> str:
    """Optional secret for the user-supplied endpoint; never part of provenance."""
    return model_config.api_key()


def backend_from_env() -> Backend:
    """Build the canonical Qwen backend; env selects an identical endpoint.

    RequirementGraph's per-leaf visual path and the continuous paired-scene
    equivalence metric use this adapter. The model and sampling parameters are
    packaged; the batch launcher may select a losslessly identical serving
    instance and records that transport endpoint in provenance.
    """
    from .openai_compat import OpenAICompatJudge

    return OpenAICompatJudge(
        model=model_config.MODEL,
        base_url=model_config.base_url(),
        api_key=_api_key(),
        max_tokens=model_config.MAX_TOKENS,
        timeout_s=model_config.TIMEOUT_S,
        temperature=model_config.TEMPERATURE,
        seed=model_config.SEED,
        enable_thinking=model_config.ENABLE_THINKING,
        structured=model_config.STRUCTURED_OUTPUT,
    )


def backend_from_policy(
    policy: JudgePolicy,
    *,
    base_url_override: str | None = None,
) -> Backend:
    """Build the frozen judge with an optional identical transport endpoint.

    The policy freezes the judge model and every scoring parameter, but not
    the serving location: the explicit argument wins, followed by the
    ``CODE4SCENE_VLM_BASE_URL`` environment variable. There is no default.
    """
    from .openai_compat import OpenAICompatJudge

    selected_base_url = model_config.require_base_url(
        base_url_override or policy.base_url
    )

    return OpenAICompatJudge(
        model=policy.model,
        base_url=selected_base_url,
        api_key=_api_key(),
        max_tokens=policy.max_tokens,
        timeout_s=policy.judge_timeout_s,
        temperature=policy.temperature,
        seed=policy.seed,
        enable_thinking=policy.enable_thinking,
        structured=policy.structured_output,
        structured_output_method=policy.structured_output_method,
    )

# JudgeRequest and Backend are exported because they ARE the backend contract:
# anyone writing a backend needs both, and a contract you have to reach into a
# private module for is not a contract.
__all__ = ["backend_from_env", "backend_from_policy", "Backend", "EvidenceImage",
           "Judge", "JudgeError", "JudgePolicy", "JudgePolicyError",
           "JudgeRequest", "Rubric", "RubricError", "Verdict", "load_policy",
           "load_rubric", "report", "verdict_schema"]


from dataclasses import asdict as _asdict, is_dataclass as _is_dataclass
from pathlib import Path as _Path
from typing import Any as _Any
from ... import contracts as _contracts
from ...context import Context as _Context, error as _error


def report(verdict: _Any, **ids: str) -> dict[str, _Any]:
    """A verdict, lowered into the shared report shape.

    Accepts the ``Verdict`` dataclass the judge actually returns as well as a
    mapping. Only the mapping was ever exercised, so the real path — judge a
    scene, lower the verdict — raised ``AttributeError: 'Verdict' object has
    no attribute 'get'`` the first time a live model answered.

    Quality is published as a continuous measurement and is never thresholded
    into pass/fail.
    """
    if _is_dataclass(verdict) and not isinstance(verdict, type):
        verdict = _asdict(verdict)
    score = float(verdict.get("final", 0.0))
    result = {
        **_contracts.base("vlm_as_judge", ids),
        "status": _contracts.MEASURED,
        "score": score,
        "metrics": {
            "judged": verdict.get("judged"),
            "ceiling": verdict.get("ceiling"),
            **{
                f"criterion.{key}": value
                for key, value in (verdict.get("scores") or {}).items()
            },
        },
        "evidence": {
            "rubric": verdict.get("rubric"),
            "model": verdict.get("model"),
            "views": verdict.get("views") or [],
            "channels": verdict.get("channels") or [],
            "images": verdict.get("images") or [],
            "ceiling_reasons": verdict.get("ceiling_reasons") or [],
            "rationales": verdict.get("rationales") or {},
            "structured_output_recovery": (
                verdict.get("structured_output_recovery") or {}
            ),
        },
        "artifacts": {},
        "probes_used": ("vlm_as_judge",),
    }
    return result


def _load_frozen_policy(context: _Context) -> JudgePolicy:
    configured = context.spec.get("judge_policy")
    if not configured:
        from ....resources import config_file

        configured = config_file("judge-policies", "visual-gt-paired-formal.yaml")
    if not isinstance(configured, (str, _Path)):
        raise JudgePolicyError("judge_policy must be a YAML file path")
    path = _Path(configured)
    if not path.is_absolute() and getattr(context.task, "path", None):
        path = (_Path(context.task.path).parent / path).resolve()
    return load_policy(path)


def evidence_requests(task: _Any, spec: dict[str, _Any]):
    from ...render_evidence import EvidenceRequest

    policy = _load_frozen_policy(
        _Context(record={}, task=task, ids={}, spec=spec)
    )
    return (
        EvidenceRequest(
            protocol=f"vlm_as_judge.{policy.mode}.{policy.id}",
            kind=(
                "paired_render" if policy.mode == "gt_paired"
                else "candidate_render"
            ),
            channels=policy.channels,
            view_count=policy.view_count,
            width=policy.width,
            height=policy.height,
            timeout_s=policy.timeout_s,
            camera_protocol=policy.render_protocol,
            scene_alignment=policy.scene_alignment,
            lighting_policy=policy.lighting_policy,
            exposure_policy=policy.exposure_policy,
            camera_plan_policy=policy.camera_plan_policy,
            frame_quality_policy=policy.frame_quality_policy,
        ),
    )


def _attach_policy(result: dict[str, _Any], policy: JudgePolicy) -> dict[str, _Any]:
    result["evidence"] = {**result.get("evidence", {}), **policy.evidence()}
    return result

def _error_with_policy(context: _Context, policy: JudgePolicy,
                       reason: str) -> dict[str, _Any]:
    return _attach_policy(_error("vlm_as_judge", context, reason), policy)


def verify(context: _Context) -> dict[str, _Any]:
    """Score the scene under one frozen formal visual evaluation policy."""
    try:
        policy = _load_frozen_policy(context)
    except Exception as e:                      # noqa: BLE001 — reported
        return _error("vlm_as_judge", context, f"{type(e).__name__}: {e}")
    configured_mode = str(context.spec.get("mode") or "candidate_only")
    if configured_mode != policy.mode:
        return _error_with_policy(
            context, policy,
            f"task declares mode={configured_mode!r}, but the frozen policy "
            f"declares mode={policy.mode!r}; gt_paired must be requested "
            "explicitly and never replaces candidate-only judging",
        )


    if context.judge_verdict is not None:
        return _error(
            "vlm_as_judge",
            context,
            "a precomputed judge_verdict cannot enter the formal visual score; "
            "the verdict must be reproduced from the frozen renders, policy, "
            "backend and model request",
        )

    if policy.mode == "gt_paired" and context.renders is None:
        return _error_with_policy(
            context, policy,
            "gt_paired requires canonical GT and candidate renders; none were "
            "assembled, and candidate-only judging is not a fallback",
        )

    if policy.mode == "candidate_only" and context.visual_renders is None:
        return _error(
            "vlm_as_judge", context,
            "the frozen visual policy requires addressed RGB, Base Color and "
            "Scene Depth renders, but none were captured; that is not the same "
            "as a scene judged badly",
        )

    try:
        backend = backend_from_policy(policy)
    except Exception as e:                      # noqa: BLE001 — reported
        return _error_with_policy(context, policy,
                                  f"the frozen judging backend is unavailable "
                                  f"({type(e).__name__}: {e}); the run was not "
                                  f"judged, which is not the same as being "
                                  f"judged badly")
    if policy.mode == "gt_paired":
        try:
            from .paired import evaluate

            return evaluate(context, policy, backend)
        except Exception as e:                  # noqa: BLE001 — reported
            return _error_with_policy(
                context, policy, f"{type(e).__name__}: {e}")


    try:
        from .evidence import collect

        if context.out_dir is not None:
            evidence_dir = _Path(context.out_dir) / "judge_evidence"
        else:
            first_source = next(
                _Path(value)
                for views in context.visual_renders.images.values()
                for channels in views.values() for value in channels.values())
            evidence_dir = first_source.parent / "judge_evidence"
        bundle = collect(
            context.visual_renders,
            policy.rubric,
            evidence_dir,
            near_cm=policy.depth_near_cm,
            far_cm=policy.depth_far_cm,
            base_color_encoding=policy.base_color_encoding,
            minimum_complete_viewpoints=policy.minimum_complete_viewpoints,
        )
    except Exception as e:                      # noqa: BLE001 — reported
        return _error("vlm_as_judge", context, f"{type(e).__name__}: {e}")

    try:
        verdict = Judge(
            rubric=policy.rubric,
            backend=backend,
            model=policy.model,
            min_images=policy.minimum_complete_viewpoints,
        ).score(
                prompt=context.task.prompt,
                images=[],
                metrics=(context.record.get("metrics") or {}),
                evidence=bundle.images,
                evidence_layout=policy.evidence_layout)
    except Exception as e:                      # noqa: BLE001 — reported
        return _error("vlm_as_judge", context, f"{type(e).__name__}: {e}")
    result = _attach_policy(
        report(verdict, **context.ids),
        policy,
    )
    result["evidence"]["depth_visualization"] = bundle.depth_visualization
    result["evidence"]["base_color_normalization"] = (
        bundle.base_color_normalization
    )
    result["evidence"]["dropped_views"] = bundle.dropped_views
    validity = getattr(context.visual_renders, "as_dict", lambda: {})()
    result["evidence"]["channel_validity"] = validity.get("channel_validity")
    return result
