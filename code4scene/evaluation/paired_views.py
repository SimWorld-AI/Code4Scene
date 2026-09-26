"""Pairing renders of two scenes by camera, and refusing when they cannot be.

Every pixel metric in this layer rests on one condition: the two images were
taken from the SAME viewpoint in two scenes. Break it and PSNR is measuring
how differently the two shots were framed, which is a large, stable,
meaningless number.

So the pairing is a gate, not a lookup. It names which scenes are present,
which cameras they share, and why any camera was dropped, and it refuses when
the reference side is missing entirely — a comparison against nothing is not a
score of zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import render, scene_graph_capture

#: What the reference side of the pair may be called, best first. The
#: canonical scene is the real reference; the pre-edit scene is a fallback
#: that answers a different question and is labelled as such in the report.
REFERENCE_SCENES = ("gt", "canonical", "reference", "input")
CANDIDATE_SCENE = "candidate"

class PairingError(Exception):
    """The renders cannot be paired, so no pixel metric may be computed."""


@dataclass(frozen=True)
class Pairing:
    """The cameras two scenes share, and the images at each."""

    reference_scene: str
    views: tuple[str, ...]
    pairs: tuple[tuple[str, str, str], ...]      # (view, reference, candidate)
    dropped: tuple[dict[str, Any], ...] = ()

    def evidence(self) -> dict[str, Any]:
        return {"reference_scene": self.reference_scene,
                "paired_view_count": len(self.pairs),
                "dropped_views": list(self.dropped),
                "camera_protocol": "identical viewpoint in both scenes"}


def pair(renders: Any, channel: str = "rgb") -> Pairing:
    """The shared cameras of the candidate and the best available reference."""
    # Three different absences, and they used to share one sentence. "No
    # renders were captured" was reported on a run that had photographed the
    # candidate four times — the episode simply never assembled a RenderSet to
    # hand over — and the wrong half of the pipeline was searched for an hour
    # because of it. A refusal is only useful if it names the thing that is
    # actually missing.
    if renders is None:
        raise PairingError(
            "this run assembled no render set at all, so the comparison never "
            "reached the images: whether any scene was photographed is not "
            "what this says. The episode passes one to the verifiers only when "
            "it has both a candidate and a reference to put in it")
    if not getattr(renders, "images", None):
        raise PairingError(
            "the render set is empty, so there is nothing to compare; a scene "
            "nobody photographed is not a scene that photographed badly")
    images = renders.images
    if CANDIDATE_SCENE not in images:
        raise PairingError(
            f"the render set carries no {CANDIDATE_SCENE!r} scene; it has "
            f"{sorted(images)}")
    reference = next((name for name in REFERENCE_SCENES if name in images), None)
    if reference is None:
        raise PairingError(
            "these metrics compare the candidate against a reference render of "
            "the same cameras, and the render set carries none: "
            f"{sorted(images)}. Rendering the canonical scene needs the scoring "
            f"environment, which is the only place its content is mounted")

    pairs, dropped = [], []
    for view in sorted(set(images[reference]) | set(images[CANDIDATE_SCENE])):
        left = (images[reference].get(view) or {}).get(channel)
        right = (images[CANDIDATE_SCENE].get(view) or {}).get(channel)
        if left and right:
            pairs.append((view, left, right))
        else:
            dropped.append({"view": view, "channel": channel,
                            "reason": ("candidate render missing" if left
                                       else "reference render missing")})
    if not pairs:
        raise PairingError(
            f"the candidate and {reference} renders share no camera on the "
            f"{channel!r} channel, so no pair can be compared")
    return Pairing(reference_scene=reference,
                   views=tuple(view for view, _, _ in pairs),
                   pairs=tuple(pairs), dropped=tuple(dropped))


@dataclass(frozen=True)
class StrictPairing:
    """A complete multi-channel pair whose relative cameras were proven equal."""

    reference_scene: str
    views: tuple[str, ...]
    channels: tuple[str, ...]
    pairs: dict[str, tuple[tuple[str, str, str], ...]]
    camera_evidence: dict[str, Any]

    def evidence(self) -> dict[str, Any]:
        return {
            "reference_scene": self.reference_scene,
            "paired_view_count": len(self.views),
            "views": list(self.views),
            "channels": list(self.channels),
            "camera_alignment": self.camera_evidence,
        }


def _require_camera_plan(
    renders: Any,
    scenes: Sequence[str],
    *,
    planner_id: str,
    pairing_policy: str,
) -> dict[str, Any]:
    audits = getattr(renders, "camera_plan_audit", None) or {}
    selected: dict[str, Any] = {}
    for scene in scenes:
        value = audits.get(scene)
        if not isinstance(value, dict):
            raise PairingError(
                f"scene {scene!r} lacks camera-plan audit metadata"
            )
        if value.get("planner_id") != planner_id:
            raise PairingError(
                f"scene {scene!r} used camera planner {value.get('planner_id')!r}, "
                f"expected {planner_id!r}"
            )
        if value.get("pairing_policy") != pairing_policy:
            raise PairingError(
                f"scene {scene!r} used pairing policy "
                f"{value.get('pairing_policy')!r}, expected {pairing_policy!r}"
            )
        selected[scene] = value
    return selected


def _require_geometry_locked_per_view_independent_photometry(
    renders: Any,
    scenes: Sequence[str],
    *,
    lighting_policy: str,
    frame_quality_policy: str,
) -> dict[str, Any]:
    """Validate per-view photometry without weakening camera equality."""

    overrides = getattr(renders, "capture_overrides", None) or []
    groups = [
        value for value in overrides
        if value.get("policy") == frame_quality_policy
        and value.get("scene") is None
        and value.get("scope")
        == "candidate-gt-geometry-locked-per-view-independent-photometry"
    ]
    if len(groups) != 1 or groups[0].get("accepted") is not True:
        raise PairingError(
            "geometry-locked per-view independent-photometry group audit is "
            "missing or rejected"
        )
    group = groups[0]
    camera_lock = group.get("camera_lock")
    if (
        not isinstance(camera_lock, dict)
        or camera_lock.get("status") != "locked"
        or camera_lock.get("camera_adjustment_scope") != "candidate_gt_pair"
        or camera_lock.get("single_scene_reframing_allowed") is not False
    ):
        raise PairingError(
            "per-view independent photometry lacks the required joint camera lock"
        )

    selected_all = group.get("per_scene_view_selected")
    lighting_all = group.get("per_scene_view_final_lighting")
    exposure_all = group.get("per_scene_view_final_exposure")
    quality_all = group.get("per_scene_view_final_quality")
    if not all(
        isinstance(value, dict)
        for value in (selected_all, lighting_all, exposure_all, quality_all)
    ):
        raise PairingError(
            "per-view independent photometry lacks structured final evidence"
        )

    selected: dict[str, dict[str, Any]] = {}
    final_lighting: dict[str, dict[str, Any]] = {}
    final_exposure: dict[str, dict[str, Any]] = {}
    quality: dict[str, dict[str, Any]] = {}
    expected_view_ids: set[str] | None = None
    for scene in scenes:
        camera_specs = getattr(renders, "camera_specs", {}).get(scene)
        if not isinstance(camera_specs, dict) or not camera_specs:
            raise PairingError(f"scene {scene!r} lacks per-view camera specs")
        view_ids = set(camera_specs)
        if expected_view_ids is None:
            expected_view_ids = view_ids
        elif view_ids != expected_view_ids:
            raise PairingError(
                "Candidate/GT per-view photometry has different view ids"
            )

        scene_selected = selected_all.get(scene)
        scene_lighting = lighting_all.get(scene)
        scene_exposure = exposure_all.get(scene)
        scene_quality = quality_all.get(scene)
        if not all(
            isinstance(value, dict)
            for value in (
                scene_selected,
                scene_lighting,
                scene_exposure,
                scene_quality,
            )
        ):
            raise PairingError(
                f"scene {scene!r} lacks per-view photometry evidence"
            )
        if not all(set(value) == view_ids for value in (
            scene_selected,
            scene_lighting,
            scene_exposure,
            scene_quality,
        )):
            raise PairingError(
                f"scene {scene!r} per-view photometry coverage is incomplete"
            )

        selected[scene] = {}
        final_lighting[scene] = {}
        final_exposure[scene] = {}
        quality[scene] = {}
        for view_id in sorted(view_ids):
            setting = scene_selected[view_id]
            light = scene_lighting[view_id]
            exposure = scene_exposure[view_id]
            final = scene_quality[view_id]
            if not all(
                isinstance(value, dict)
                for value in (setting, light, exposure, final)
            ):
                raise PairingError(
                    f"{scene}/{view_id} has malformed photometry evidence"
                )
            if (
                final.get("accepted") is not True
                or set(final.get("image_quality") or {}) != {view_id}
                or setting.get("lighting_policy") is None
                or setting.get("exposure_policy") is None
            ):
                raise PairingError(
                    f"{scene}/{view_id} has rejected final frame quality"
                )

            selected_policy = setting["lighting_policy"]
            selected_setting = setting.get("lighting_setting")
            selected_hash = setting.get("lighting_rig_sha256")
            if selected_setting == "native":
                if (
                    selected_policy != lighting_policy
                    or selected_hash is not None
                    or light.get("policy") != lighting_policy
                    or light.get("rig_sha256") is not None
                ):
                    raise PairingError(
                        f"{scene}/{view_id} has invalid native lighting audit"
                    )
            else:
                level = render.PHOTOMETRY_FILL_BY_POLICY.get(
                    str(selected_policy)
                )
                expected_count = len(level["rig_spec"]) if level else 0
                if (
                    level is None
                    or light.get("policy") != selected_policy
                    or light.get("rig_sha256") != selected_hash
                    or light.get("spawned") != expected_count
                    or light.get("configured") != expected_count
                    or light.get("removed_after_capture") != expected_count
                    or light.get("errors")
                ):
                    raise PairingError(
                        f"{scene}/{view_id} has invalid fill audit"
                    )

            light_matches = [
                value for value in overrides
                if value.get("policy") == selected_policy
                and value.get("scene") == scene
                and value.get("view_id") == view_id
                and value.get("phase") == "final_capture"
            ]
            if light_matches != [light]:
                raise PairingError(
                    f"{scene}/{view_id} needs exactly one final lighting audit"
                )

            selected_exposure_policy = setting["exposure_policy"]
            exposure_matches = [
                value for value in overrides
                if value.get("policy") == selected_exposure_policy
                and value.get("scene") == scene
                and value.get("view_id") == view_id
                and value.get("phase") == "final_capture"
            ]
            if exposure_matches != [exposure]:
                raise PairingError(
                    f"{scene}/{view_id} needs exactly one final exposure audit"
                )
            if (
                exposure.get("paired_bias_search") is not True
                or exposure.get("bias_ev") != setting.get("bias_ev")
                or exposure.get("camera_override_mode")
                != render.PHOTOMETRY_CAMERA_EXPOSURE_MODE
                or exposure.get("camera_override_applied") is not True
                or exposure.get("camera_override_applied_view_count") != 1
            ):
                raise PairingError(
                    f"{scene}/{view_id} has invalid exposure audit"
                )

            selected[scene][view_id] = dict(setting)
            final_lighting[scene][view_id] = light
            final_exposure[scene][view_id] = exposure
            quality[scene][view_id] = final

    return {
        "status": "geometry_locked_per_view_independently_visible",
        "lighting_policy": lighting_policy,
        "frame_quality_policy": frame_quality_policy,
        "photometry_scope": "per_scene_per_view",
        "per_scene_view_selected": selected,
        "group_audit": group,
        "per_scene_view_final_lighting": final_lighting,
        "per_scene_view_final_exposure": final_exposure,
        "per_scene_view_quality": quality,
    }

def _require_paired_lighting(
    renders: Any,
    scenes: Sequence[str],
    *,
    lighting_policy: str,
    frame_quality_policy: str,
) -> dict[str, Any]:
    overrides = getattr(renders, "capture_overrides", None) or []
    if frame_quality_policy == scene_graph_capture.GT_VISUAL_FRAME_QUALITY:
        return _require_geometry_locked_per_view_independent_photometry(
            renders,
            scenes,
            lighting_policy=lighting_policy,
            frame_quality_policy=frame_quality_policy,
        )
    if frame_quality_policy != scene_graph_capture.CAPTION_FRAME_QUALITY:
        raise PairingError(
            f"no current paired quality verifier for {frame_quality_policy!r}"
        )
    quality: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        scene_quality = [
            value
            for value in overrides
            if value.get("policy") == frame_quality_policy
            and value.get("scene") == scene
        ]
        if len(scene_quality) != 1:
            raise PairingError(
                f"scene {scene!r} needs exactly one caption visibility audit; "
                f"got {len(scene_quality)}"
            )
        check = scene_quality[0]
        if check.get("accepted") is not True:
            raise PairingError(
                f"scene {scene!r} did not pass the caption visibility "
                f"guard: {check}"
            )
        quality[scene] = check
    return {
        "status": "matched_and_visible",
        "lighting_policy": lighting_policy,
        "frame_quality_policy": frame_quality_policy,
        "per_scene_quality": quality,
    }


def strict_pair(
    renders: Any,
    channels: Sequence[str],
    *,
    expected_view_count: int,
    camera_protocol: str,
    camera_plan_policy: str | None = None,
    lighting_policy: str | None = None,
    frame_quality_policy: str | None = None,
) -> StrictPairing:
    """Require a complete pair and exact scene-relative camera metadata."""
    if expected_view_count < 2:
        raise PairingError("strict paired judging needs at least two viewpoints")
    channel_pairs = {channel: pair(renders, channel) for channel in channels}
    references = {value.reference_scene for value in channel_pairs.values()}
    if len(references) != 1:
        raise PairingError(
            f"render channels disagree on the reference scene: {sorted(references)}")
    reference = next(iter(references))
    view_sets = {channel: set(value.views)
                 for channel, value in channel_pairs.items()}
    union = set().union(*view_sets.values()) if view_sets else set()
    if len(union) != expected_view_count:
        raise PairingError(
            f"strict paired judging requires exactly {expected_view_count} "
            f"complete viewpoints, found {len(union)}: {sorted(union)}")
    incomplete = {
        channel: sorted(union - values)
        for channel, values in view_sets.items() if values != union
    }
    if incomplete:
        raise PairingError(
            f"paired channels do not cover the same viewpoints: {incomplete}")
    dropped = {
        channel: list(value.dropped)
        for channel, value in channel_pairs.items() if value.dropped
    }
    if dropped:
        raise PairingError(
            f"strict paired judging does not permit dropped renders: {dropped}")

    camera_specs = getattr(renders, "camera_specs", None) or {}
    settings = getattr(renders, "capture_settings", None) or {}
    anchors = getattr(renders, "scene_anchors_cm", None) or {}
    for scene in (reference, CANDIDATE_SCENE):
        if scene not in camera_specs or scene not in settings or scene not in anchors:
            raise PairingError(
                f"scene {scene!r} lacks strict camera/settings/anchor metadata")
    if settings[reference] != settings[CANDIDATE_SCENE]:
        raise PairingError(
            "GT and candidate capture settings differ; visual similarity is "
            f"not evaluated: {settings[reference]} vs {settings[CANDIDATE_SCENE]}")
    if settings[reference].get("camera_protocol") != camera_protocol:
        raise PairingError(
            f"paired render protocol is {settings[reference].get('camera_protocol')!r}, "
            f"expected {camera_protocol!r}")
    if (
        lighting_policy is not None
        and settings[reference].get("lighting_policy") != lighting_policy
    ):
        raise PairingError(
            f"paired lighting policy is "
            f"{settings[reference].get('lighting_policy')!r}, expected "
            f"{lighting_policy!r}"
        )
    mismatches = {}
    for view in sorted(union):
        gt_spec = camera_specs[reference].get(view)
        candidate_spec = camera_specs[CANDIDATE_SCENE].get(view)
        if gt_spec is None or candidate_spec is None:
            mismatches[view] = {
                "reference": gt_spec,
                "candidate": candidate_spec,
            }
        elif gt_spec != candidate_spec:
            mismatches[view] = {
                "reference": gt_spec,
                "candidate": candidate_spec,
            }
    if mismatches:
        raise PairingError(
            "GT and candidate scene-relative cameras differ; visual similarity "
            f"is not evaluated: {mismatches}")

    plan_audit = None
    joint_camera_adjustment = None
    comparison_space = "scene_relative"
    if camera_protocol in {
        *render.GT_PAIRED_SCENE_GRAPH_PROTOCOLS,
        render.GT_CAPTION_ENVIRONMENT_SCENE_GRAPH_PROTOCOL,
    }:
        planner_id = camera_plan_policy or scene_graph_capture.GT_VISUAL_CAMERA_PLAN
        planner_contracts = {
            scene_graph_capture.GT_VISUAL_CAMERA_PLAN: (
                "environment-routed-frozen-absolute-cameras",
                None,
                "environment_routed_absolute_frozen",
            ),
            scene_graph_capture.CAPTION_CAMERA_PLAN: (
                "environment-routed-frozen-absolute-cameras",
                None,
                "environment_routed_absolute_frozen",
            ),
            scene_graph_capture.GT_REPAIR_TARGET_OUTDOOR_CAMERA_PLAN: (
                "gt-repair-target-visibility-frozen-absolute-cameras",
                "gt_minus_input_target",
                "gt_repair_target_visibility_planned_absolute_frozen",
            ),
            scene_graph_capture.GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN: (
                "gt-repair-target-visibility-frozen-absolute-cameras",
                "gt_minus_input_target",
                "gt_repair_target_indoor_room_side_absolute_frozen",
            ),
            scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN: (
                "gt-repair-target-visibility-frozen-absolute-cameras",
                None,
                "gt_repair_target_case_camera_priority_absolute_frozen",
            ),
        }
        if planner_id not in planner_contracts:
            raise PairingError(
                f"unsupported GT paired camera planner {planner_id!r}"
            )
        pairing_policy, planning_source, comparison_space = (
            planner_contracts[planner_id]
        )
        plan_audit = _require_camera_plan(
            renders,
            (reference, CANDIDATE_SCENE),
            planner_id=planner_id,
            pairing_policy=pairing_policy,
        )
        sources = {
            value.get("planning_source") for value in plan_audit.values()
        }
        environments = {
            value.get("scene_environment") for value in plan_audit.values()
        }
        allowed_planning_sources = (
            {planning_source} if planning_source is not None else set()
        )
        if planner_id == scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN:
            allowed_planning_sources = {
                "case_camera_manifest_then_gt_minus_input_target_fill",
                "gt_minus_input_target",
            }
        elif planning_source is None:
            if environments == {"indoor"}:
                allowed_planning_sources = {
                    "input_gt_shared_actor_geometry",
                    "input_gt_shared_room_geometry",
                    "input_gt_unedited_structure",
                }
            elif environments == {"outdoor"}:
                allowed_planning_sources = {"gt"}
            else:
                raise PairingError(
                    "environment-routed camera plan lacks one explicit "
                    f"indoor/outdoor provenance value: {environments}"
                )
        hashes = {
            value.get("camera_specs_sha256") for value in plan_audit.values()
        }
        source_is_valid = (
            len(sources) == 1
            and None not in sources
            and sources <= allowed_planning_sources
        )
        if not source_is_valid or len(hashes) != 1 or None in hashes:
            source_label = (
                "GT"
                if allowed_planning_sources == {"gt"}
                else "Input/GT unedited structure"
                if "input_gt_unedited_structure" in allowed_planning_sources
                else "the GT-minus-Input repair target"
            )
            raise PairingError(
                f"GT visual camera plan was not frozen once from {source_label}: "
                f"sources={sorted(str(value) for value in sources)}, "
                f"camera_hashes={sorted(str(value) for value in hashes)}"
            )
        if anchors[reference] != anchors[CANDIDATE_SCENE]:
            raise PairingError(
                "GT and candidate absolute scene anchors differ; single-scene "
                "camera adjustment is forbidden"
            )
    lighting_audit = None
    if frame_quality_policy is not None:
        if lighting_policy is None:
            raise PairingError(
                "paired frame-quality policy cannot be verified without a "
                "lighting policy"
            )
        lighting_audit = _require_paired_lighting(
            renders,
            (reference, CANDIDATE_SCENE),
            lighting_policy=lighting_policy,
            frame_quality_policy=frame_quality_policy,
        )

    return StrictPairing(
        reference_scene=reference,
        views=tuple(sorted(union)),
        channels=tuple(channels),
        pairs={channel: value.pairs for channel, value in channel_pairs.items()},
        camera_evidence={
            "status": "matched",
            "protocol": camera_protocol,
            "comparison_space": comparison_space,
            "settings": settings[reference],
            "scene_anchors_cm": {
                reference: anchors[reference],
                CANDIDATE_SCENE: anchors[CANDIDATE_SCENE],
            },
            "per_view": {
                view: camera_specs[reference][view] for view in sorted(union)
            },
            "camera_plan_audit": plan_audit,
            "joint_camera_adjustment": joint_camera_adjustment,
            "lighting_alignment": lighting_audit,
        },
    )


__all__ = ["CANDIDATE_SCENE", "REFERENCE_SCENES", "Pairing", "PairingError",
           "StrictPairing", "pair", "strict_pair"]
