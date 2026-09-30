"""Actor-level Repair F1 for image-to-scene cases (paper protocol, Appendix C.5).

This is the Repair F1 used by every image-to-scene number in the paper
(policy ``unified-actor-repair-success-f1.v2``). It compares three
independently exported scene snapshots:

* the corrupted **input** scene the agent received,
* the withheld **ground-truth** scene, and
* the **candidate** scene the agent saved.

Repair targets are derived from input vs. ground truth only, before the
candidate is inspected. Each affected actor is one target: ``n+`` targets must
be present after repair (additions and state restorations), ``n-`` are
required removals.

Counting (paper Eq. 4 and 5)::

    TP = m + d
    FP = |C| - m + o
    FN = (n+ - m) + (n- - d)
    P = TP / (TP + FP)   (0 when TP + FP = 0)
    R = TP / (TP + FN)
    F1 = 2 TP / (2 TP + FP + FN)

where ``m`` is the number of one-to-one matched present-targets that pass every
applicable nominal acceptance test (5 cm, 5 degrees, 5 % scale; exact asset,
class and recorded attributes), ``d`` the number of completed required
removals, ``C`` the candidate actors that are newly added, semantically
changed, or surviving present-targets, and ``o`` the unintended background
losses (including background actors repurposed to satisfy a target).

Input/candidate correspondence never uses actor names, labels, GUIDs or
paths. It (1) maximizes unchanged one-to-one matches within structural and
spatial blocks, preserving background multiplicity before target identities,
(2) assigns remaining actors with the same structural descriptor at minimum
cost ``d/(1+d) + 0.2*phi/180 + 0.05*k`` and (3) lets an unchanged pose anchor
an asset replacement of a same-class actor. Correspondence only identifies an
edit; repair success still requires the ground-truth acceptance tests.

Ported from the paper's reference implementation of the metric. The
per-actor repair-success tests, target authoring and assignment solver are the
unchanged verifier functions in :mod:`code4scene.evaluation`.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

from ..evaluation import repair_success
from ..evaluation.assignment import match
from ..evaluation.repair_target_scope import _descriptor_key
from ..evaluation.requirement_graph.repair_target_authoring import (
    _is_semantic_scene_actor,
    derive_repair_targets,
)
from ..evaluation.scene_diff import (
    DEFAULT_TOLERANCES,
    _property_differences,
    _transform_differences,
    actor_identity,
    index_actors,
)
from .constants import ACTOR_F1_POLICY, CORRESPONDENCE_POLICY

POLICY = ACTOR_F1_POLICY
COUNTS = ("true_positive", "false_positive", "false_negative")
#: Editor provenance fields that are not visible edits.
NON_VISUAL = {"label", "actor_origin", "logical_object_id", "actor_role", "actor_tags"}


class ActorF1Error(ValueError):
    """The three scene snapshots cannot define a measurable repair case."""


# ---------------------------------------------------------------------------
# Rates and scene filtering
# ---------------------------------------------------------------------------


def rates(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """Precision, recall and F1 from counts (paper Eq. 5)."""

    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def semantic_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    """Drop camera, scene-capture and other non-semantic helper actors."""

    return {**scene, "actors": [a for a in scene["actors"] if _is_semantic_scene_actor(a)]}


# ---------------------------------------------------------------------------
# Input/candidate correspondence (ID-independent)
# ---------------------------------------------------------------------------


def changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[str, ...]:
    """Visible differences, using strict editor-roundtrip (not repair) limits."""

    left, right = before.get("transform") or {}, after.get("transform") or {}
    fields = _transform_differences(left, right, DEFAULT_TOLERANCES)
    # Equivalent Euler representations are not an authored rotation edit.
    a = repair_success._vector(left.get("rotation_deg"))
    b = repair_success._vector(right.get("rotation_deg"))
    if a is not None and b is not None:
        fields = [f for f in fields if f != "rotation"]
        if repair_success._rotation_error(a, b) > DEFAULT_TOLERANCES["rotation_deg"] + 1e-9:
            fields.append("rotation")
    fields = ["transform." + f for f in fields]
    fields += [
        "property." + f for f in _property_differences(before, after) if f not in NON_VISUAL
    ]
    # Repair success requires all recorded properties, not only the scene-diff
    # light/post-process allowlist, so a real edit is never hidden here.
    if "properties" in before and before["properties"] != after.get("properties"):
        fields.append("property.properties")
    # Dynamic bounds-only changes of background actors are not independent
    # edits; required-target bounds are still checked by repair success.
    return tuple(sorted(set(fields)))


def order_key(actor: Mapping[str, Any]) -> str:
    """Deterministic order without names or IDs, so renaming cannot change the optimum."""

    fields = ("class", "asset_path", "component_asset_paths", "transform", "bounds",
              "properties", "material_paths", "component_material_slots")
    return json.dumps({k: actor[k] for k in fields if k in actor}, sort_keys=True)


def position(actor: Mapping[str, Any]) -> tuple[float, ...] | None:
    return repair_success._vector((actor.get("transform") or {}).get("location_cm"))


def cell(point: Sequence[float]) -> tuple[int, ...]:
    return tuple(math.floor(v / DEFAULT_TOLERANCES["location_cm"]) for v in point)


def maximum_matching(
    rows: Sequence[int], edges: Mapping[int, Sequence[int]], available: set[int],
) -> list[tuple[int, int]]:
    """Sparse maximum-cardinality bipartite matching with augmenting paths."""

    left_to_right: dict[int, int] = {}
    right_to_left: dict[int, int] = {}
    for root in rows:
        queue, seen_left, parent_right = deque([root]), {root}, {}
        end = None
        while queue and end is None:
            left = queue.popleft()
            for right in edges[left]:
                if right not in available or right in parent_right:
                    continue
                parent_right[right] = left
                owner = right_to_left.get(right)
                if owner is None:
                    end = right
                    break
                if owner not in seen_left:
                    seen_left.add(owner)
                    queue.append(owner)
        while end is not None:
            left = parent_right[end]
            previous = left_to_right.get(left)
            left_to_right[left], right_to_left[end] = end, left
            end = previous
    return sorted(left_to_right.items())


def changed_cost(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    """``d/(1+d) + 0.2 * phi/180 + 0.05 * k`` (paper Appendix C.5)."""

    a, b = position(before), position(after)
    distance = math.dist(a, b) if a is not None and b is not None else math.inf
    distance_cost = distance / (1 + distance) if math.isfinite(distance) else 1.0
    left, right = before.get("transform") or {}, after.get("transform") or {}
    a = repair_success._vector(left.get("rotation_deg"))
    b = repair_success._vector(right.get("rotation_deg"))
    rotation = (
        repair_success._rotation_error(a, b) / 180 if a is not None and b is not None else 1.0
    )
    return distance_cost + 0.2 * rotation + 0.05 * len(changes(before, after))


def _has_pose(actor: Mapping[str, Any]) -> bool:
    transform = actor.get("transform") or {}
    return all(
        repair_success._vector(transform.get(field)) is not None
        for field in ("location_cm", "rotation_deg", "scale")
    )


def correspond(
    initial: Mapping[str, Any],
    candidate: Mapping[str, Any],
    frozen_input_ids: set[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """One-to-one input/candidate matching that preserves unchanged background first.

    Phase 1 finds maximum-cardinality unchanged pairs within each structural
    and spatial block (background first, then frozen target identities).
    Phase 2 matches remaining same-structure actors at minimum cost; such a
    pair remains an edit and is never declared correct. Phase 3 pairs
    remaining same-class, same-pose actors whose asset changed; they are
    wrong edits recorded once, not a fabricated deletion plus addition.
    """

    before, after = index_actors(initial), index_actors(candidate)
    left = sorted(before.values(), key=order_key)
    right = sorted(after.values(), key=order_key)
    grid: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for j, actor in enumerate(right):
        point = position(actor)
        if point is not None:
            grid[_descriptor_key(actor), cell(point)].append(j)
    edges: dict[int, list[int]] = {}
    for i, actor in enumerate(left):
        point = position(actor)
        options: list[int] = []
        if point is not None:
            base = cell(point)
            for delta in product((-1, 0, 1), repeat=3):
                key = tuple(x + y for x, y in zip(base, delta, strict=True))
                options.extend(grid.get((_descriptor_key(actor), key), ()))
        # Missing pose evidence cannot establish an unchanged correspondence.
        edges[i] = sorted(
            j for j in options
            if _has_pose(actor) and _has_pose(right[j]) and not changes(actor, right[j])
        )

    available = set(range(len(right)))
    pairs: list[tuple[int, int, str]] = []
    # Identical duplicates preserve required background multiplicity before
    # target identities; one remaining object cannot fill two roles.
    for target_phase in (False, True):
        rows = [i for i, a in enumerate(left)
                if (actor_identity(a) in frozen_input_ids) == target_phase]
        current = maximum_matching(rows, edges, available)
        pairs.extend((i, j, "unchanged_semantic") for i, j in current)
        available.difference_update(j for _, j in current)

    assigned_left = {i for i, _, _ in pairs}
    left_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    right_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, a in enumerate(left):
        if i not in assigned_left:
            left_groups[_descriptor_key(a)].append(i)
    for j in sorted(available):
        right_groups[_descriptor_key(right[j])].append(j)
    for key, rows in left_groups.items():
        columns = right_groups.get(key, [])
        if not columns:
            continue
        costs = [[changed_cost(left[i], right[j]) for j in columns] for i in rows]
        pairs.extend(
            (rows[i], columns[j], "changed_structural_assignment") for i, j in match(costs)
        )

    assigned_left = {i for i, _, _ in pairs}
    available.difference_update(j for _, j, _ in pairs)
    # An exact pose anchors a replaced or cleared asset. IDs never enable this
    # fallback and an asset-changing pair is never evidence of correctness.
    pose_grid: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for j in sorted(available):
        point = position(right[j])
        if point is not None:
            pose_grid[_descriptor_key(right[j])[0], cell(point)].append(j)
    pose_edges: dict[int, list[int]] = {}
    for i, a in enumerate(left):
        if i in assigned_left:
            continue
        options = []
        point = position(a)
        if point is not None:
            base = cell(point)
            for delta in product((-1, 0, 1), repeat=3):
                options.extend(pose_grid.get(
                    (_descriptor_key(a)[0], tuple(x + y for x, y in zip(base, delta, strict=True))), ()))
        pose_edges[i] = sorted(
            j for j in options
            if _has_pose(a) and _has_pose(right[j])
            and not any(f.startswith("transform.") for f in changes(a, right[j]))
        )
    for target_phase in (False, True):
        rows = [i for i in pose_edges
                if (actor_identity(left[i]) in frozen_input_ids) == target_phase]
        current = maximum_matching(rows, pose_edges, available)
        pairs.extend((i, j, "pose_anchored_asset_change") for i, j in current)
        available.difference_update(j for _, j in current)

    mapping: dict[str, str] = {}
    audit = []
    for i, j, method in pairs:
        old_id, new_id = actor_identity(left[i]), actor_identity(right[j])
        mapping[old_id] = new_id
        audit.append({
            "input": old_id, "candidate": new_id, "method": method,
            "structural_compatible": _descriptor_key(left[i]) == _descriptor_key(right[j]),
            "identity_changed": old_id != new_id,
            "changed_fields": list(changes(left[i], right[j])),
        })
    if len(mapping) != len(set(mapping.values())):
        raise RuntimeError("correspondence is not one-to-one")
    removed, added = set(before) - set(mapping), set(after) - set(mapping.values())
    return mapping, {
        "policy": CORRESPONDENCE_POLICY, "ids_used_for_matching": False,
        "roundtrip_tolerances": dict(DEFAULT_TOLERANCES),
        "background_bounds_only_edits_scored": False,
        "matches": audit, "unmatched_input_ids": sorted(removed),
        "unmatched_candidate_ids": sorted(added),
    }


# ---------------------------------------------------------------------------
# Repair F1
# ---------------------------------------------------------------------------


def _check_scene(name: str, scene: Any) -> Mapping[str, Any]:
    if not isinstance(scene, Mapping) or not isinstance(scene.get("actors"), list):
        raise ActorF1Error(f"{name} scene snapshot must contain an 'actors' list")
    meta = scene.get("export_metadata") or {}
    if meta and meta.get("status") != "success":
        raise ActorF1Error(f"{name} scene export did not succeed: {meta.get('status')!r}")
    return scene


def measure(
    initial: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(counts_and_rates, audit)`` for one image-to-scene case."""

    initial = _check_scene("input", initial)
    ground_truth = _check_scene("ground_truth", ground_truth)
    candidate = _check_scene("candidate", candidate)
    # The requested edits are never redefined from the model's output.
    targets = derive_repair_targets(initial, ground_truth)
    if not targets:
        raise ActorF1Error("input and ground truth define no repair target")
    desired = [t.desired_actor for t in targets if t.desired_actor is not None]
    removal_targets = {actor_identity(t.input_actor) for t in targets if t.operation == "remove"}
    frozen_ids = {actor_identity(t.input_actor) for t in targets if t.input_actor is not None}
    semantic_input, semantic_candidate = semantic_scene(initial), semantic_scene(candidate)
    before, after = index_actors(semantic_input), index_actors(semantic_candidate)
    mapping, correspondence = correspond(semantic_input, semantic_candidate, frozen_ids)
    inverse = {v: k for k, v in mapping.items()}
    reasons: dict[str, list[str]] = {
        k: ["semantic_added"] for k in correspondence["unmatched_candidate_ids"]
    }
    for pair in correspondence["matches"]:
        why = []
        if pair["changed_fields"]:
            why.append("semantic_changed")
        if pair["input"] in frozen_ids - removal_targets:
            why.append("surviving_repair_target")
        if why:
            reasons[pair["candidate"]] = why
    candidates = sorted((after[k] for k in reasons), key=order_key)
    nominal = repair_success.PROFILES["nominal"]
    failures = [
        [repair_success._failures(repair_success.pair_errors(c, g), nominal) for c in candidates]
        for g in desired
    ]
    assigned = (
        match([[float(bool(f)) for f in row] for row in failures])
        if desired and candidates else []
    )
    passed = [(g, c) for g, c in assigned if not failures[g][c]]
    matched_gt, matched_candidate = {g for g, _ in passed}, {c for _, c in passed}
    # A candidate that restores a repair target stands for that target's own
    # input actor when correspondence tied it to a removal target instead.
    restored_by = {
        id(t.desired_actor): actor_identity(t.input_actor)
        for t in targets
        if t.operation == "repair" and t.input_actor is not None and t.desired_actor is not None
    }
    rebound = []
    for g, c in passed:
        own = restored_by.get(id(desired[g]))
        candidate_id = actor_identity(candidates[c])
        held_by = inverse.get(candidate_id)
        if own and held_by and held_by != own and held_by in removal_targets and own not in mapping:
            del mapping[held_by]
            mapping[own] = candidate_id
            inverse[candidate_id] = own
            rebound.append({"candidate": candidate_id, "from": held_by, "to": own})
    removed = set(before) - set(mapping)
    # A pose-anchored replacement no longer contains the old asset. Its new
    # presence is evaluated separately; this is not an ID-based deletion.
    replaced = {p["input"] for p in correspondence["matches"] if not p["structural_compatible"]}
    correct_removals = removal_targets & (removed | replaced)
    missing_removals = removal_targets - (removed | replaced)
    off_target_removals = removed - frozen_ids
    # A background actor cannot both count as preserved background and satisfy
    # a repair target after being moved: charge its missing source role.
    repurposed_background = {
        inverse[actor_identity(candidates[c])]
        for _, c in passed
        if actor_identity(candidates[c]) in inverse
        and inverse[actor_identity(candidates[c])] not in frozen_ids
    }
    off_target_removals |= repurposed_background
    tp = len(passed) + len(correct_removals)
    fp = len(candidates) - len(passed) + len(off_target_removals)
    fn = len(desired) - len(passed) + len(missing_removals)
    if tp + fn != len(targets):
        raise RuntimeError("every target must contribute exactly one TP or FN")
    audit = {
        "presence_matches": [
            {"gt": actor_identity(desired[g]), "candidate": actor_identity(candidates[c])}
            for g, c in passed
        ],
        "presence_unmatched_gt": [
            actor_identity(a) for i, a in enumerate(desired) if i not in matched_gt
        ],
        "presence_unmatched_candidates": [
            actor_identity(a) for i, a in enumerate(candidates) if i not in matched_candidate
        ],
        "removal_target_ids": sorted(removal_targets),
        "correct_removal_ids": sorted(correct_removals),
        "missing_removal_ids": sorted(missing_removals),
        "off_target_removal_ids": sorted(off_target_removals),
        "repurposed_background_ids": sorted(repurposed_background),
        "structurally_replaced_input_ids": sorted(replaced),
        "target_rebinds": rebound,
        "removed_non_deletion_target_ids": sorted((removed & frozen_ids) - removal_targets),
        "candidate_reasons": reasons,
        "actor_correspondence": correspondence,
    }
    values = {
        "policy": POLICY,
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        **rates(tp, fp, fn),
        "desired_count": len(targets), "candidate_count": tp + fp,
        "presence_true_positive": len(passed),
        "presence_false_positive": len(candidates) - len(passed),
        "presence_false_negative": len(desired) - len(passed),
        "removal_true_positive": len(correct_removals),
        "removal_false_positive": len(off_target_removals),
        "removal_false_negative": len(missing_removals),
    }
    return values, audit


def zero_for_invalid_case() -> dict[str, Any]:
    """Reporting zeros for an invalid/missing candidate (not measured counts)."""

    return {"policy": POLICY, "status": "zero_invalid_candidate",
            "true_positive": 0, "false_positive": 0, "false_negative": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0}


__all__ = [
    "ActorF1Error", "COUNTS", "POLICY", "changed_cost", "changes", "correspond",
    "maximum_matching", "measure", "order_key", "rates", "semantic_scene",
    "zero_for_invalid_case",
]
