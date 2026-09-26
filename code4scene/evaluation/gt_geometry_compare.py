"""Logical-object geometry, rigid alignment, and auditable correspondence.

Raw Actor topology is evidence, but it is not object topology.  A floor made
from twenty-five tiles must pay an over-fragmentation penalty; those tiles
must then become one floor before position, footprint, and layout are
measured, or one modelling mistake is charged repeatedly.

Likewise, a scene reconstructed seventy metres from the canonical origin must
pay for that global placement error.  Local layout is measured only after the
best identity-anchored XY rigid transform is applied, so the same translation
does not inflate every downstream metric.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from code4scene.core.inventory import ATTRIBUTE_EVIDENCE_FIELDS

from .assignment import match
from .pairwise_identity_locator import (
    PairwiseIdentityLocator,
    PairwiseIdentityRetrievalResult,
    select_long_tail_objects,
)
from .scene_geometry import (aabb, center_cm, cyclic_degrees, extent_cm,
                             has_bounds, rotation_deg, rounded)
from .selection import (category_from_actor, normalize_asset_path,
                        normalize_category, normalize_class)
from .values import as_text


AUTO_GROUP_GAP_CM = 75.0
AUTO_GROUP_GAP_CAP_CM = 1500.0
AUTO_GROUP_GAP_RATIO = 0.5
IDENTITY_GATE = 0.5
MIN_SPATIAL_BLOCK_CM = 2000.0
REJECTED_MATCH_COST = 1_000_000.0
SIZE_BLOCK_LOG_THRESHOLD = 1.5
SPATIAL_BLOCK_SCALE = 20.0
MATCH_WEIGHTS = {
    "identity": 0.7,
    "aligned_position": 0.2,
    "bounds_size": 0.1,
}

_INFRASTRUCTURE_TERMS = (
    "worldsettings",
    "directionallight",
    "skylight",
    "skyatmosphere",
    "skysphere",
    "exponentialheightfog",
    "volumetriccloud",
    "postprocessvolume",
    "playerstart",
    "levelbounds",
    "blockingvolume",
    "abstractnavdata",
    "reflectioncapture",
)
_LABEL_STOP_WORDS = {
    "actor", "bp", "combined", "mesh", "sk", "sm", "static",
}


def _triplet(value: Any, default: float = 0.0) -> list[float]:
    values = list(value) if isinstance(value, (list, tuple)) else []
    result: list[float] = []
    for index in range(3):
        try:
            number = float(values[index]) if index < len(values) else default
        except (TypeError, ValueError):
            number = default
        result.append(number if math.isfinite(number) else default)
    return result


def _size_cm(actor: Mapping[str, Any]) -> list[float]:
    return [abs(value) * 2.0 for value in extent_cm(actor)]


def _log_rmse(left: Sequence[float], right: Sequence[float]) -> float | None:
    ratios = [
        math.log(max(a, 1e-6) / max(b, 1e-6))
        for a, b in zip(left, right, strict=False)
        if a > 0 and b > 0
    ]
    if not ratios:
        return None
    return math.sqrt(sum(value * value for value in ratios) / len(ratios))


def _scale_of(actor: Mapping[str, Any]) -> list[float]:
    transform = actor.get("transform")
    return _triplet(
        transform.get("scale") if isinstance(transform, Mapping) else None,

        default=1.0,
    )
def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    rank = max(1, math.ceil(0.95 * len(finite)))
    return {
        "count": len(finite),
        "mean": rounded(sum(finite) / len(finite)),
        "p95": rounded(finite[rank - 1]),
        "max": rounded(finite[-1]),
    }


def _scale_cm(actors: Sequence[Mapping[str, Any]]) -> float:
    sizes = sorted(
        value
        for actor in actors
        for value in _size_cm(actor)
        if value > 0
    )
    if not sizes:
        return 1.0
    return max(1.0, sizes[len(sizes) // 2])


def _label_tokens(value: Any) -> frozenset[str]:
    text = as_text(value) or ""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"[a-z]+", text.lower())
    return frozenset(token for token in tokens if token not in _LABEL_STOP_WORDS)


def _content_actor(actor: Mapping[str, Any]) -> bool:
    if actor.get("keep") is True:
        return False
    haystack = " ".join(
        str(actor.get(key) or "").lower()
        for key in ("class", "label", "asset_path")
    ).replace("_", "")
    return not any(term in haystack for term in _INFRASTRUCTURE_TERMS)


def _actor_sort_key(actor: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        normalize_asset_path(actor.get("asset_path")) or "",
        str(actor.get("label") or "").lower(),
        *[round(value, 6) for value in center_cm(actor)],
        str(actor.get("actor_path") or ""),
    )


def _identity_values(
    actors: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    assets = sorted({
        value for actor in actors
        if (value := normalize_asset_path(actor.get("asset_path")))
    })
    categories = sorted({
        value for actor in actors
        if (value := normalize_category(category_from_actor(actor)))
    })
    classes = sorted({
        value for actor in actors
        if (value := normalize_class(actor.get("class")))
    })
    labels = sorted({
        " ".join(sorted(tokens))
        for actor in actors
        if (tokens := _label_tokens(actor.get("label")))
    })
    return tuple(assets), tuple(categories), tuple(classes), tuple(labels)


@dataclass(frozen=True)
class GeometryObject:
    """One comparison object and the raw Actors that constitute it."""

    object_id: str
    actors: tuple[Mapping[str, Any], ...]
    grouping_source: str
    stable_actor_ids: tuple[str, ...]
    assets: tuple[str, ...]
    categories: tuple[str, ...]
    classes: tuple[str, ...]
    labels: tuple[str, ...]

    @property
    def raw_actor_count(self) -> int:
        return len(self.actors)

    @property
    def bounds_available(self) -> bool:
        return bool(self.actors) and all(has_bounds(actor) for actor in self.actors)

    @property
    def rotation_available(self) -> bool:
        return len(self.actors) == 1

    @property
    def scale_available(self) -> bool:
        return len(self.actors) == 1

    def as_actor(self) -> dict[str, Any]:
        """Adapt the union object back into the shared Actor geometry shape."""

        first = self.actors[0] if self.actors else {}
        if self.bounds_available:
            boxes = [aabb(actor) for actor in self.actors]
            minimum = [
                min(box[0][axis] for box in boxes)
                for axis in range(3)
            ]
            maximum = [
                max(box[1][axis] for box in boxes)
                for axis in range(3)
            ]
            center = [
                (minimum[axis] + maximum[axis]) / 2.0
                for axis in range(3)
            ]
            bounds: dict[str, Any] | None = {
                "origin_cm": center,
                "extent_cm": [
                    (maximum[axis] - minimum[axis]) / 2.0
                    for axis in range(3)
                ],
            }
        else:
            centers = [center_cm(actor) for actor in self.actors]
            center = [
                sum(value[axis] for value in centers) / len(centers)
                for axis in range(3)
            ] if centers else [0.0, 0.0, 0.0]
            bounds = None
        source_transform = first.get("transform")
        source_transform = (
            source_transform if isinstance(source_transform, Mapping) else {}
        )
        result = {
            **first,
            "label": first.get("label"),
            "transform": {
                "location_cm": center,
                "rotation_deg": (
                    _triplet(source_transform.get("rotation_deg"))
                    if self.rotation_available
                    else [0.0, 0.0, 0.0]
                ),
                "scale": (
                    _triplet(source_transform.get("scale"), default=1.0)
                    if self.scale_available
                    else [1.0, 1.0, 1.0]
                ),
            },
            "evaluator_object_id": self.object_id,
            "is_logical_object": True,
            "raw_actor_count": self.raw_actor_count,
            "grouping_source": self.grouping_source,
            "rotation_measured": self.rotation_available,
            "scale_measured": self.scale_available,
        }
        if bounds is not None:
            result["bounds"] = bounds
        else:
            result.pop("bounds", None)
        return result


def _make_object(
    object_id: str,
    actors: Sequence[Mapping[str, Any]],
    source: str,
) -> GeometryObject:
    ordered = tuple(sorted(actors, key=_actor_sort_key))
    assets, categories, classes, labels = _identity_values(ordered)
    return GeometryObject(
        object_id=object_id,
        actors=ordered,
        grouping_source=source,
        stable_actor_ids=tuple(sorted({
            value
            for actor in ordered
            if (value := as_text(actor.get("stable_actor_id")))
        })),
        assets=assets,
        categories=categories,
        classes=classes,
        labels=labels,
    )


def _modular_part(actor: Mapping[str, Any]) -> bool:
    if not has_bounds(actor):
        return False
    size = _size_cm(actor)
    longest = max(size)
    shortest = min(size)
    horizontal = max(size[:2])
    vertical = size[2]
    if longest <= 0 or vertical > horizontal:
        return False
    return shortest <= max(25.0, longest * 0.12) or horizontal >= 500.0


def _near(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_min, left_max = aabb(left)
    right_min, right_max = aabb(right)
    gaps = [
        max(0.0, left_min[axis] - right_max[axis],
            right_min[axis] - left_max[axis])
        for axis in range(3)
    ]
    left_horizontal = max(_size_cm(left)[:2])
    right_horizontal = max(_size_cm(right)[:2])
    adaptive_gap = min(
        AUTO_GROUP_GAP_CAP_CM,
        AUTO_GROUP_GAP_RATIO * left_horizontal,
        AUTO_GROUP_GAP_RATIO * right_horizontal,
    )
    return math.hypot(*gaps) <= max(AUTO_GROUP_GAP_CM, adaptive_gap)


def _components(
    actors: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    """Connected components for one exact asset identity."""

    ordered = list(sorted(actors, key=_actor_sort_key))
    neighbours: list[list[int]] = [[] for _ in ordered]
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            if _near(ordered[left], ordered[right]):
                neighbours[left].append(right)
                neighbours[right].append(left)
    result: list[list[Mapping[str, Any]]] = []
    seen: set[int] = set()
    for start in range(len(ordered)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        indexes: list[int] = []
        while stack:
            current = stack.pop()
            indexes.append(current)
            for neighbour in neighbours[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        result.append([ordered[index] for index in sorted(indexes)])
    return result


def build_logical_objects(
    actors: Sequence[Mapping[str, Any]],
    *,
    side: str,
    trust_declared_groups: bool,
) -> tuple[list[GeometryObject], dict[str, Any]]:
    """Filter infrastructure and aggregate modular pieces deterministically."""

    exported = [actor for actor in actors if isinstance(actor, Mapping)]
    content = [actor for actor in exported if _content_actor(actor)]
    declared: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    remaining: list[Mapping[str, Any]] = []
    ignored_declared_actor_count = 0
    for actor in content:
        logical_id = as_text(actor.get("logical_object_id"))
        if logical_id and trust_declared_groups:
            declared[logical_id].append(actor)
        else:
            if logical_id:
                ignored_declared_actor_count += 1
            remaining.append(actor)

    provisional: list[tuple[list[Mapping[str, Any]], str]] = [
        (members, "declared_logical_object_id")
        for _, members in sorted(declared.items())
    ]
    by_asset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    loose: list[Mapping[str, Any]] = []
    for actor in remaining:
        asset = normalize_asset_path(actor.get("asset_path"))
        if asset:
            by_asset[asset].append(actor)
        else:
            loose.append(actor)

    for _, members in sorted(by_asset.items()):
        modular = [actor for actor in members if _modular_part(actor)]
        independent = [actor for actor in members if not _modular_part(actor)]
        if len(modular) > 1:
            for component in _components(modular):
                provisional.append((
                    component,
                    "auto_connected_asset" if len(component) > 1 else "single_actor",
                ))
        else:
            provisional.extend(([actor], "single_actor") for actor in modular)
        provisional.extend(
            ([actor], "single_actor") for actor in independent
        )
    provisional.extend(([actor], "single_actor") for actor in loose)
    provisional.sort(key=lambda item: _actor_sort_key(item[0][0]))
    objects = [
        _make_object(f"{side}:object:{index:04d}", members, source)
        for index, (members, source) in enumerate(provisional)
    ]
    summary = {
        "exported_actor_count": len(exported),
        "content_actor_count": len(content),
        "excluded_infrastructure_actor_count": len(exported) - len(content),
        "logical_object_count": len(objects),
        "declared_group_count": len(declared),
        "ignored_untrusted_declared_actor_count": ignored_declared_actor_count,
        "auto_group_count": sum(
            item.grouping_source == "auto_connected_asset" for item in objects
        ),
    }
    return objects, summary


def _fragmentation_by_asset(
    candidate: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int, list[dict[str, Any]]]:
    """Compare raw modular-part counts before connected-component grouping.

    One canonical fence may become several disconnected candidate components.
    Measuring fragmentation only after one-to-one object matching would call
    the unmatched components "extra objects" and can even reverse 178 versus
    98 parts into under-segmentation. Exact-asset totals retain the actual raw
    representation error; connectivity remains a separate topology fact.
    """

    def populations(
        values: Sequence[Mapping[str, Any]],
    ) -> dict[str, list[Mapping[str, Any]]]:
        result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for actor in values:
            if not _content_actor(actor):
                continue
            identity = normalize_asset_path(actor.get("asset_path"))
            if identity:
                result[identity].append(actor)
        return result

    def connected_modular(values: Sequence[Mapping[str, Any]]) -> bool:
        modular = [actor for actor in values if _modular_part(actor)]
        if len(modular) < 2:
            return False
        return any(
            _near(modular[left], modular[right])
            for left in range(len(modular))
            for right in range(left + 1, len(modular))
        )

    candidate_populations = populations(candidate)
    canonical_populations = populations(canonical)
    details: list[dict[str, Any]] = []
    over, under, denominator = 0, 0, 0
    common_assets = candidate_populations.keys() & canonical_populations.keys()
    for asset in sorted(common_assets):
        candidate_members = candidate_populations[asset]
        canonical_members = canonical_populations[asset]
        if not (
            connected_modular(candidate_members)
            or connected_modular(canonical_members)
        ):
            continue
        candidate_count = len(candidate_members)
        canonical_count = len(canonical_members)
        excess = max(0, candidate_count - canonical_count)
        deficit = max(0, canonical_count - candidate_count)
        over += excess
        under += deficit
        denominator += canonical_count
        details.append({
            "asset_path": asset,
            "candidate_raw_part_count": candidate_count,
            "canonical_raw_part_count": canonical_count,
            "over_fragmented_actor_count": excess,
            "under_segmented_actor_count": deficit,
        })
    return over, under, denominator, details


@dataclass(frozen=True)
class Alignment:
    source_center_cm: tuple[float, float, float]
    target_center_cm: tuple[float, float, float]
    yaw_deg: float
    anchor_pairs: tuple[tuple[GeometryObject, GeometryObject], ...]
    method: str

    @property
    def center_translation_cm(self) -> list[float]:
        return [
            self.target_center_cm[index] - self.source_center_cm[index]
            for index in range(3)
        ]

    @property
    def origin_translation_cm(self) -> list[float]:
        angle = math.radians(self.yaw_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        x, y, z = self.source_center_cm
        rotated = [
            cosine * x - sine * y,
            sine * x + cosine * y,
            z,
        ]
        return [
            self.target_center_cm[index] - rotated[index]
            for index in range(3)
        ]


def _mean_center(objects: Sequence[GeometryObject]) -> tuple[float, float, float]:
    centers = [center_cm(item.as_actor()) for item in objects]
    if not centers:
        return (0.0, 0.0, 0.0)
    return tuple(
        sum(value[axis] for value in centers) / len(centers)
        for axis in range(3)
    )


def _alignment_anchors(
    candidate: Sequence[GeometryObject],
    canonical: Sequence[GeometryObject],
) -> list[tuple[GeometryObject, GeometryObject]]:
    candidate_stable: dict[str, list[GeometryObject]] = defaultdict(list)
    canonical_stable: dict[str, list[GeometryObject]] = defaultdict(list)
    for item in candidate:
        for stable_id in item.stable_actor_ids:
            candidate_stable[stable_id].append(item)
    for item in canonical:
        for stable_id in item.stable_actor_ids:
            canonical_stable[stable_id].append(item)
    stable_pairs: list[tuple[GeometryObject, GeometryObject]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for stable_id in sorted(candidate_stable.keys() & canonical_stable.keys()):
        left = candidate_stable[stable_id]
        right = canonical_stable[stable_id]
        if len(left) != 1 or len(right) != 1:
            continue
        key = (left[0].object_id, right[0].object_id)
        if key not in seen_pairs:
            seen_pairs.add(key)
            stable_pairs.append((left[0], right[0]))
    if stable_pairs:
        return stable_pairs

    candidate_assets: dict[str, list[GeometryObject]] = defaultdict(list)
    canonical_assets: dict[str, list[GeometryObject]] = defaultdict(list)
    for item in candidate:
        if len(item.assets) == 1:
            candidate_assets[item.assets[0]].append(item)
    for item in canonical:
        if len(item.assets) == 1:
            canonical_assets[item.assets[0]].append(item)
    return [
        (candidate_assets[key][0], canonical_assets[key][0])
        for key in sorted(candidate_assets.keys() & canonical_assets.keys())
        if len(candidate_assets[key]) == 1 and len(canonical_assets[key]) == 1
    ]


def estimate_alignment(
    candidate: Sequence[GeometryObject],
    canonical: Sequence[GeometryObject],
) -> Alignment:
    anchors = _alignment_anchors(candidate, canonical)
    if anchors:
        candidate_center = _mean_center([pair[0] for pair in anchors])
        canonical_center = _mean_center([pair[1] for pair in anchors])
        method = (
            "exact_stable_actor_id_least_squares"
            if all(
                set(left.stable_actor_ids) & set(right.stable_actor_ids)
                for left, right in anchors
            )
            else "unique_exact_asset_least_squares"
        )
    else:
        candidate_center = _mean_center(candidate)
        canonical_center = _mean_center(canonical)
        method = "logical_scene_centroid_translation_only"

    yaw = 0.0
    if len(anchors) >= 2:
        cross, dot = 0.0, 0.0
        for candidate_item, canonical_item in anchors:
            left = center_cm(candidate_item.as_actor())
            right = center_cm(canonical_item.as_actor())
            px = left[0] - candidate_center[0]
            py = left[1] - candidate_center[1]
            qx = right[0] - canonical_center[0]
            qy = right[1] - canonical_center[1]
            cross += px * qy - py * qx
            dot += px * qx + py * qy
        if math.hypot(cross, dot) > 1e-9:
            yaw = math.degrees(math.atan2(cross, dot))
    return Alignment(
        source_center_cm=candidate_center,
        target_center_cm=canonical_center,
        yaw_deg=yaw,
        anchor_pairs=tuple(anchors),
        method=method,
    )


def _transform_point(value: Sequence[float], alignment: Alignment) -> list[float]:
    point = _triplet(value)
    x = point[0] - alignment.source_center_cm[0]
    y = point[1] - alignment.source_center_cm[1]
    angle = math.radians(alignment.yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        cosine * x - sine * y + alignment.target_center_cm[0],
        sine * x + cosine * y + alignment.target_center_cm[1],
        point[2] - alignment.source_center_cm[2] + alignment.target_center_cm[2],
    ]


def _aligned_actor(actor: Mapping[str, Any], alignment: Alignment) -> dict[str, Any]:
    result = dict(actor)
    transform = actor.get("transform")
    transform = dict(transform) if isinstance(transform, Mapping) else {}
    transform["location_cm"] = _transform_point(
        transform.get("location_cm") or center_cm(actor),
        alignment,
    )
    observed_rotation = _triplet(transform.get("rotation_deg"))
    observed_rotation[1] = (
        observed_rotation[1] + alignment.yaw_deg + 180.0
    ) % 360.0 - 180.0
    transform["rotation_deg"] = observed_rotation
    result["transform"] = transform
    if has_bounds(actor):
        bounds = actor.get("bounds")
        assert isinstance(bounds, Mapping)
        extent = extent_cm(actor)
        angle = math.radians(alignment.yaw_deg)
        cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
        result["bounds"] = {
            "origin_cm": _transform_point(bounds.get("origin_cm"), alignment),
            "extent_cm": [
                cosine * extent[0] + sine * extent[1],
                sine * extent[0] + cosine * extent[1],
                extent[2],
            ],
        }
    return result


def _identity_distance(
    left: GeometryObject,
    right: GeometryObject,
) -> tuple[float, str]:
    if set(left.stable_actor_ids) & set(right.stable_actor_ids):
        return 0.0, "exact_stable_actor_id"
    if set(left.assets) & set(right.assets):
        return 0.0, "exact_asset_path"
    if set(left.categories) & set(right.categories):
        return 0.1, "exact_trusted_category"
    left_labels = set(left.labels)
    right_labels = set(right.labels)
    best = 0.0
    for left_label in left_labels:
        left_tokens = set(left_label.split())
        for right_label in right_labels:
            right_tokens = set(right_label.split())
            union = left_tokens | right_tokens
            if union:
                best = max(best, len(left_tokens & right_tokens) / len(union))
    if best >= 0.5:
        return 0.2 + 0.3 * (1.0 - best), "label_token_overlap"
    if set(left.classes) & set(right.classes):
        return 0.9, "class_only_rejected"
    return 1.0, "no_identity_evidence"


def _structured_identity_match(
    left: GeometryObject,
    right: GeometryObject,
) -> bool | None:
    """Compare the strongest structured identity carried by both objects."""

    if left.assets or right.assets:
        return left.assets == right.assets
    if left.categories or right.categories:
        return left.categories == right.categories
    if left.classes or right.classes:
        return left.classes == right.classes
    return None


def _bounded(value: float) -> float:
    return value / (1.0 + value)


def _match_cost(
    candidate_object: GeometryObject,
    candidate_actor: Mapping[str, Any],
    canonical_object: GeometryObject,
    canonical_actor: Mapping[str, Any],
    scale: float,
) -> float:
    identity, _ = _identity_distance(candidate_object, canonical_object)
    radial = math.dist(center_cm(candidate_actor), center_cm(canonical_actor)) / scale
    size = _log_rmse(_size_cm(candidate_actor), _size_cm(canonical_actor))
    parts = {
        "identity": identity,
        "aligned_position": _bounded(radial),
        "bounds_size": _bounded(size) if size is not None else None,
    }
    weighted = [
        (MATCH_WEIGHTS[name], value)
        for name, value in parts.items()
        if value is not None
    ]
    total = sum(weight for weight, _ in weighted)
    return (
        sum(weight * value for weight, value in weighted) / total
        if total
        else 1.0
    )


def _footprint_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_min, left_max = aabb(left)
    right_min, right_max = aabb(right)
    intersection_size = [
        max(0.0, min(left_max[axis], right_max[axis])
            - max(left_min[axis], right_min[axis]))
        for axis in range(2)
    ]
    intersection = math.prod(intersection_size)
    left_area = math.prod(_size_cm(left)[:2])
    right_area = math.prod(_size_cm(right)[:2])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _rotation_error(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return math.hypot(*[
        cyclic_degrees(a, b)
        for a, b in zip(rotation_deg(left), rotation_deg(right), strict=True)
    ])


def actor_pair_metrics(
    candidate_actor: Mapping[str, Any],
    canonical_actor: Mapping[str, Any],
    *,
    alignment_audit: Mapping[str, Any] | None = None,
) -> dict[str, float | None]:
    """Return the canonical aligned metrics for one raw Actor pair.

    Local repair-target correspondence sometimes needs to rematch exchangeable
    instances after the full-scene assignment has established the frozen global
    alignment.  Keeping this calculation here prevents that local projection
    from drifting from the metrics used by the full-scene audit.
    """

    aligned_candidate: Mapping[str, Any] = candidate_actor
    if alignment_audit is not None:
        alignment = Alignment(
            source_center_cm=tuple(
                _triplet(alignment_audit.get("source_center_cm"))
            ),
            target_center_cm=tuple(
                _triplet(alignment_audit.get("target_center_cm"))
            ),
            yaw_deg=_triplet(
                [alignment_audit.get("yaw_deg"), 0.0, 0.0]
            )[0],
            anchor_pairs=(),
            method=str(alignment_audit.get("method") or "audit_replay"),
        )
        aligned_candidate = _aligned_actor(candidate_actor, alignment)

    bounds_pair = has_bounds(aligned_candidate) and has_bounds(canonical_actor)
    return {
        "absolute_center_distance_cm": rounded(
            math.dist(center_cm(candidate_actor), center_cm(canonical_actor))
        ),
        "aligned_center_distance_cm": rounded(
            math.dist(center_cm(aligned_candidate), center_cm(canonical_actor))
        ),
        "aligned_rotation_error_deg": rounded(
            _rotation_error(aligned_candidate, canonical_actor)
        ),
        "scale_log_error": rounded(
            _log_rmse(
                [abs(item) for item in _scale_of(aligned_candidate)],
                [abs(item) for item in _scale_of(canonical_actor)],
            )
        ),
        "bounds_size_log_rmse": (
            rounded(
                _log_rmse(
                    _size_cm(aligned_candidate),
                    _size_cm(canonical_actor),
                )
            )
            if bounds_pair
            else None
        ),
        "footprint_iou": (
            rounded(_footprint_iou(aligned_candidate, canonical_actor))
            if bounds_pair
            else None
        ),
    }


def _pairwise_layout(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    scale: float,
    limit: int = 20000,
) -> tuple[float | None, int]:
    if len(pairs) < 2:
        return None, 0
    errors: list[float] = []
    for left in range(len(pairs)):
        for right in range(left + 1, len(pairs)):
            if len(errors) >= limit:
                break
            candidate_distance = math.dist(
                center_cm(pairs[left][0]), center_cm(pairs[right][0])
            )
            canonical_distance = math.dist(
                center_cm(pairs[left][1]), center_cm(pairs[right][1])
            )
            errors.append(abs(candidate_distance - canonical_distance) / scale)
        if len(errors) >= limit:
            break
    return (sum(errors) / len(errors) if errors else None), len(errors)


def _object_summary(item: GeometryObject) -> dict[str, Any]:
    actor = item.as_actor()
    return {
        "object_id": item.object_id,
        "label": actor.get("label"),
        "grouping_source": item.grouping_source,
        "stable_actor_ids": list(item.stable_actor_ids),
        "raw_actor_count": item.raw_actor_count,
        "asset_paths": list(item.assets),
        "semantic_categories": list(item.categories),
        "center_cm": [rounded(value, 3) for value in center_cm(actor)],
        "size_cm": [rounded(value, 3) for value in _size_cm(actor)],
        "actor_labels_sample": [
            str(member.get("label") or "")
            for member in item.actors[:8]
        ],
    }
def _normalized_ue_object_path(value: Any) -> str | None:
    text = as_text(value)
    if text is None:
        return None
    text = text.strip()
    if "'" in text and text.endswith("'"):
        text = text.split("'", 1)[1][:-1]
    return text.casefold() if text.startswith("/") else None


def _material_set(actor: Mapping[str, Any]) -> frozenset[str] | None:
    if "material_paths" not in actor:
        return None
    normalized = {
        value
        for raw in actor.get("material_paths") or ()
        if (value := _normalized_ue_object_path(raw)) is not None
    }
    return frozenset(normalized)


def _material_slots(
    actor: Mapping[str, Any],
) -> tuple[str, dict[tuple[str, int], str]]:
    if "component_material_slots" not in actor:
        return "missing", {}
    values = actor.get("component_material_slots")
    if not isinstance(values, list):
        return "invalid", {}
    result: dict[tuple[str, int], str] = {}
    for value in values:
        if not isinstance(value, Mapping):
            return "invalid", {}
        if value.get("is_dynamic") is True:
            return "dynamic", {}
        component = as_text(value.get("component_identity"))
        index = value.get("slot_index")
        path = _normalized_ue_object_path(value.get("material_path"))
        if (
            component is None
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or path is None
            or (component, index) in result
        ):
            return "invalid", {}
        result[(component, index)] = path
    return "measured", result


def actor_pair_attribute_metrics(
    candidate_actor: Mapping[str, Any],
    canonical_actor: Mapping[str, Any],
) -> dict[str, bool | None]:
    """Compare auditable attributes for one already-corresponded Actor pair.

    Local repair correspondence is intentionally independent of the full-scene
    assignment.  Exposing the same normalisation used by the global attribute
    audit lets the local projection score its own assignment without silently
    reusing attribute rows from a different pairing.
    """

    properties_match = (
        candidate_actor.get("properties") == canonical_actor.get("properties")
        if "properties" in candidate_actor and "properties" in canonical_actor
        else None
    )
    candidate_materials = _material_set(candidate_actor)
    canonical_materials = _material_set(canonical_actor)
    material_set_match = (
        candidate_materials == canonical_materials
        if candidate_materials is not None and canonical_materials is not None
        else None
    )
    candidate_slot_status, candidate_slots = _material_slots(candidate_actor)
    canonical_slot_status, canonical_slots = _material_slots(canonical_actor)
    material_slots_match = (
        candidate_slots == canonical_slots
        if candidate_slot_status == canonical_slot_status == "measured"
        else None
    )
    return {
        "task_relevant_properties_match": properties_match,
        "material_set_match": material_set_match,
        "component_material_slots_match": material_slots_match,
    }


def _attribute_comparison(
    candidate_objects: Sequence[GeometryObject],
    canonical_objects: Sequence[GeometryObject],
    matched: Sequence[tuple[int, int, float, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    property_pairs = 0
    property_mismatches = 0
    material_set_pairs = 0
    material_set_mismatches = 0
    slot_pairs = 0
    slot_mismatches = 0
    slot_missing = 0
    slot_dynamic = 0
    slot_invalid = 0
    single_actor_pairs = 0
    candidate_schema_complete = 0
    canonical_schema_complete = 0
    paired_schema_complete = 0
    candidate_missing_by_field = {
        field: 0 for field in ATTRIBUTE_EVIDENCE_FIELDS
    }
    canonical_missing_by_field = {
        field: 0 for field in ATTRIBUTE_EVIDENCE_FIELDS
    }
    rows: list[dict[str, Any]] = []

    for row, column, _, _ in matched:
        candidate_object = candidate_objects[row]
        canonical_object = canonical_objects[column]
        if (
            candidate_object.raw_actor_count != 1
            or canonical_object.raw_actor_count != 1
        ):
            continue
        single_actor_pairs += 1
        candidate = candidate_object.actors[0]
        canonical = canonical_object.actors[0]
        candidate_missing = [
            field for field in ATTRIBUTE_EVIDENCE_FIELDS
            if field not in candidate
        ]
        canonical_missing = [
            field for field in ATTRIBUTE_EVIDENCE_FIELDS
            if field not in canonical
        ]
        candidate_schema_complete += not candidate_missing
        canonical_schema_complete += not canonical_missing
        paired_schema_complete += not candidate_missing and not canonical_missing
        for field in candidate_missing:
            candidate_missing_by_field[field] += 1
        for field in canonical_missing:
            canonical_missing_by_field[field] += 1
        row_audit: dict[str, Any] = {
            "candidate_object_id": candidate_object.object_id,
            "canonical_object_id": canonical_object.object_id,
            "candidate_attribute_schema_missing_fields": candidate_missing,
            "canonical_attribute_schema_missing_fields": canonical_missing,
        }

        if "properties" in candidate and "properties" in canonical:
            property_pairs += 1
            left_properties = candidate.get("properties")
            right_properties = canonical.get("properties")
            property_match = left_properties == right_properties
            property_mismatches += not property_match
            row_audit["task_relevant_properties_match"] = property_match
        else:
            row_audit["task_relevant_properties_match"] = None

        left_set = _material_set(candidate)
        right_set = _material_set(canonical)
        if left_set is not None and right_set is not None:
            material_set_pairs += 1
            set_match = left_set == right_set
            material_set_mismatches += not set_match
            row_audit["material_set_match"] = set_match
        else:
            row_audit["material_set_match"] = None

        left_status, left_slots = _material_slots(candidate)
        right_status, right_slots = _material_slots(canonical)
        slot_statuses = {left_status, right_status}
        if slot_statuses == {"measured"}:
            slot_pairs += 1
            slots_match = left_slots == right_slots
            slot_mismatches += not slots_match
            row_audit["component_material_slots_match"] = slots_match
            if not slots_match:
                row_audit["candidate_component_material_slots"] = [
                    {
                        "component_identity": component,
                        "slot_index": index,
                        "material_path": path,
                    }
                    for (component, index), path in sorted(left_slots.items())
                ]
                row_audit["canonical_component_material_slots"] = [
                    {
                        "component_identity": component,
                        "slot_index": index,
                        "material_path": path,
                    }
                    for (component, index), path in sorted(right_slots.items())
                ]
        else:
            row_audit["component_material_slots_match"] = None
            if "dynamic" in slot_statuses:
                slot_dynamic += 1
                row_audit["component_material_slots_status"] = "dynamic_refused"
            elif "invalid" in slot_statuses:
                slot_invalid += 1
                row_audit["component_material_slots_status"] = "invalid_evidence"
            else:
                slot_missing += 1
                row_audit["component_material_slots_status"] = "missing_evidence"
        rows.append(row_audit)

    def status(pair_count: int, unavailable: int = 0) -> str:
        return (
            "measured"
            if pair_count > 0 and unavailable == 0
            else "not_evaluated"
        )

    slot_unavailable = slot_missing + slot_dynamic + slot_invalid
    metrics = {
        "attribute_single_actor_pair_count": single_actor_pairs,
        "attribute_schema_required_fields": list(ATTRIBUTE_EVIDENCE_FIELDS),
        "candidate_attribute_schema_complete_pair_count": (
            candidate_schema_complete
        ),
        "candidate_attribute_schema_missing_pair_count": (
            single_actor_pairs - candidate_schema_complete
        ),
        "candidate_attribute_schema_coverage": rounded(
            candidate_schema_complete / single_actor_pairs
        ) if single_actor_pairs else None,
        "candidate_attribute_schema_missing_by_field": (
            candidate_missing_by_field
        ),
        "canonical_attribute_schema_complete_pair_count": (
            canonical_schema_complete
        ),
        "canonical_attribute_schema_missing_pair_count": (
            single_actor_pairs - canonical_schema_complete
        ),
        "canonical_attribute_schema_coverage": rounded(
            canonical_schema_complete / single_actor_pairs
        ) if single_actor_pairs else None,
        "canonical_attribute_schema_missing_by_field": (
            canonical_missing_by_field
        ),
        "paired_attribute_schema_complete_pair_count": paired_schema_complete,
        "paired_attribute_schema_coverage": rounded(
            paired_schema_complete / single_actor_pairs
        ) if single_actor_pairs else None,
        "task_property_status": status(property_pairs),
        "task_property_pair_count": property_pairs,
        "task_property_mismatch_count": property_mismatches,
        "task_property_match_rate": rounded(
            1.0 - property_mismatches / property_pairs
        ) if property_pairs else None,
        "material_set_status": status(material_set_pairs),
        "material_set_pair_count": material_set_pairs,
        "material_set_mismatch_count": material_set_mismatches,
        "material_set_match_rate": rounded(
            1.0 - material_set_mismatches / material_set_pairs
        ) if material_set_pairs else None,
        "material_slot_status": status(slot_pairs, slot_unavailable),
        "material_slot_pair_count": slot_pairs,
        "material_slot_mismatch_count": slot_mismatches,
        "material_slot_match_rate": rounded(
            1.0 - slot_mismatches / slot_pairs
        ) if slot_pairs else None,
        "material_slot_missing_pair_count": slot_missing,
        "material_slot_dynamic_refused_pair_count": slot_dynamic,
        "material_slot_invalid_pair_count": slot_invalid,
        "material_slot_coverage": rounded(
            slot_pairs / single_actor_pairs
        ) if single_actor_pairs else 0.0,
    }
    return metrics, {"matches": rows}


def _spatial_block_key(
    actor: Mapping[str, Any],
    span_cm: float,
) -> tuple[int, int, int]:
    return tuple(
        math.floor(value / span_cm) for value in center_cm(actor)
    )


def _size_block_key(actor: Mapping[str, Any], scale: float) -> int:
    sizes = [value for value in _size_cm(actor) if value > 0]
    if not sizes:
        return 0
    return math.floor(math.log2(max(sizes) / max(scale, 1.0)))


def _blocking_compatible(
    candidate: Mapping[str, Any],
    canonical: Mapping[str, Any],
    span_cm: float,
) -> bool:
    candidate_cell = _spatial_block_key(candidate, span_cm)
    canonical_cell = _spatial_block_key(canonical, span_cm)
    if any(
        abs(left - right) > 1
        for left, right in zip(candidate_cell, canonical_cell, strict=True)
    ):
        return False
    if has_bounds(candidate) and has_bounds(canonical):
        # World-axis AABB dimensions permute as an Actor rotates. Blocking is
        # only allowed to reject genuinely incompatible scale, not the exact
        # same asset at a different pose; pose is scored after correspondence.
        size_error = _log_rmse(
            sorted(_size_cm(candidate)),
            sorted(_size_cm(canonical)),
        )
        if size_error is not None and size_error > SIZE_BLOCK_LOG_THRESHOLD:
            return False
    return True


def _locator_compatible_pairs(
    candidate_objects: Sequence[GeometryObject],
    candidate_actors: Sequence[Mapping[str, Any]],
    canonical_objects: Sequence[GeometryObject],
    canonical_actors: Sequence[Mapping[str, Any]],
    *,
    scale: float,
) -> dict[str, tuple[str, ...]]:
    """Pre-block long-tail identity pairs before any LLM request.

    This uses exactly the same aligned spatial cells and size gate as final
    assignment, so it cannot remove a pair that the controller could later
    accept.  The grid index avoids rebuilding a dense long-tail matrix.
    """

    long_candidate_ids = {
        value.object_id
        for value in select_long_tail_objects(
            candidate_objects, canonical_objects
        )
    }
    long_canonical_ids = {
        value.object_id
        for value in select_long_tail_objects(
            canonical_objects, candidate_objects
        )
    }
    if not long_candidate_ids or not long_canonical_ids:
        return {}

    block_span_cm = max(MIN_SPATIAL_BLOCK_CM, scale * SPATIAL_BLOCK_SCALE)
    canonical_by_cell: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for column, (item, actor) in enumerate(
        zip(canonical_objects, canonical_actors, strict=True)
    ):
        if item.object_id in long_canonical_ids:
            canonical_by_cell[_spatial_block_key(actor, block_span_cm)].append(
                column
            )

    compatible: dict[str, set[str]] = defaultdict(set)
    for item, actor in zip(candidate_objects, candidate_actors, strict=True):
        if item.object_id not in long_candidate_ids:
            continue
        cell = _spatial_block_key(actor, block_span_cm)
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    neighbor = (
                        cell[0] + delta_x,
                        cell[1] + delta_y,
                        cell[2] + delta_z,
                    )
                    for column in canonical_by_cell.get(neighbor, ()):
                        if _blocking_compatible(
                            actor, canonical_actors[column], block_span_cm
                        ):
                            compatible[
                                canonical_objects[column].object_id
                            ].add(item.object_id)
    return {
        canonical_id: tuple(sorted(candidate_ids))
        for canonical_id, candidate_ids in sorted(compatible.items())
    }


def _assign_identity_aware(
    candidate_objects: Sequence[GeometryObject],
    candidate_actors: Sequence[Mapping[str, Any]],
    canonical_objects: Sequence[GeometryObject],
    canonical_actors: Sequence[Mapping[str, Any]],
    scale: float,
    locator_result: PairwiseIdentityRetrievalResult,
) -> tuple[list[tuple[int, int, float, str]], dict[str, Any]]:
    """Solve only within plausible identity blocks.

    A dense candidate-by-canonical matrix is both wasteful and unsafe on
    authored maps with ten thousand Actors: pairs with no shared identity are
    rejected after assignment anyway. Frozen stable Actor IDs are anchored
    before any spatial blocking so repeated assets cannot permute and hide a
    misplaced repair. Exact asset and category tiers are then blocked by size
    and aligned spatial cell. Remaining compatible edges may reach adjacent
    cells, then split into connected bipartite components before assignment.
    """

    remaining_candidate = set(range(len(candidate_objects)))
    remaining_canonical = set(range(len(canonical_objects)))
    block_span_cm = max(MIN_SPATIAL_BLOCK_CM, scale * SPATIAL_BLOCK_SCALE)
    matched: list[tuple[int, int, float, str]] = []
    block_count = 0
    cost_cell_count = 0
    largest_block = (0, 0)

    def solve_block(
        rows: Sequence[int],
        columns: Sequence[int],
        allowed: Mapping[int, Mapping[int, tuple[float, str]]] | None = None,
    ) -> None:
        nonlocal block_count, cost_cell_count, largest_block
        ordered_rows = sorted(rows)
        ordered_columns = sorted(columns)
        if not ordered_rows or not ordered_columns:
            return
        block_count += 1
        cells = len(ordered_rows) * len(ordered_columns)
        cost_cell_count += cells
        if cells > largest_block[0] * largest_block[1]:
            largest_block = (len(ordered_rows), len(ordered_columns))
        costs = [
            [
                (
                    _match_cost(
                        candidate_objects[row],
                        candidate_actors[row],
                        canonical_objects[column],
                        canonical_actors[column],
                        scale,
                    )
                    if allowed is None or column in allowed.get(row, {})
                    else REJECTED_MATCH_COST
                )
                for column in ordered_columns
            ]
            for row in ordered_rows
        ]
        for local_row, local_column in match(costs):
            row = ordered_rows[local_row]
            column = ordered_columns[local_column]
            if allowed is not None and column not in allowed.get(row, {}):
                continue
            identity, source = (
                allowed[row][column]
                if allowed is not None
                else _identity_distance(
                    candidate_objects[row], canonical_objects[column]
                )
            )
            if identity > IDENTITY_GATE:
                continue
            matched.append((row, column, identity, source))
            remaining_candidate.discard(row)
            remaining_canonical.discard(column)

    def singleton_buckets(
        objects: Sequence[GeometryObject],
        actors: Sequence[Mapping[str, Any]],
        indexes: set[int],
        attribute: str,
    ) -> dict[tuple[str, int, tuple[int, int, int]], list[int]]:
        buckets: dict[
            tuple[str, int, tuple[int, int, int]], list[int]
        ] = defaultdict(list)
        for index in sorted(indexes):
            values = getattr(objects[index], attribute)
            if len(values) == 1:
                key = (
                    values[0],
                    _size_block_key(actors[index], scale),
                    _spatial_block_key(actors[index], block_span_cm),
                )
                buckets[key].append(index)
        return buckets

    tier_block_counts: dict[str, int] = {}
    candidate_by_stable_id: dict[str, set[int]] = defaultdict(set)
    canonical_by_stable_id: dict[str, set[int]] = defaultdict(set)
    for index, item in enumerate(candidate_objects):
        for stable_id in item.stable_actor_ids:
            candidate_by_stable_id[stable_id].add(index)
    for index, item in enumerate(canonical_objects):
        for stable_id in item.stable_actor_ids:
            canonical_by_stable_id[stable_id].add(index)
    stable_block_count = 0
    for stable_id in sorted(
        candidate_by_stable_id.keys() & canonical_by_stable_id.keys()
    ):
        rows = candidate_by_stable_id[stable_id] & remaining_candidate
        columns = canonical_by_stable_id[stable_id] & remaining_canonical
        if len(rows) == 1 and len(columns) == 1:
            solve_block(rows, columns)
            stable_block_count += 1
    tier_block_counts["exact_stable_actor_id"] = stable_block_count
    for tier, attribute in (
        ("exact_asset_path", "assets"),
        ("exact_trusted_category", "categories"),
    ):
        candidate_buckets = singleton_buckets(
            candidate_objects,
            candidate_actors,
            remaining_candidate,
            attribute,
        )
        canonical_buckets = singleton_buckets(
            canonical_objects,
            canonical_actors,
            remaining_canonical,
            attribute,
        )
        common = sorted(candidate_buckets.keys() & canonical_buckets.keys())
        tier_block_counts[tier] = len(common)
        for key in common:
            solve_block(candidate_buckets[key], canonical_buckets[key])

    by_asset: dict[str, set[int]] = defaultdict(set)
    by_category: dict[str, set[int]] = defaultdict(set)
    by_label_token: dict[str, set[int]] = defaultdict(set)
    for column in remaining_canonical:
        item = canonical_objects[column]
        for value in item.assets:
            by_asset[value].add(column)
        for value in item.categories:
            by_category[value].add(column)
        for label in item.labels:
            for token in label.split():
                by_label_token[token].add(column)

    candidate_index = {
        value.object_id: index for index, value in enumerate(candidate_objects)
    }
    canonical_index = {
        value.object_id: index for index, value in enumerate(canonical_objects)
    }
    locator_edges: dict[int, dict[int, tuple[float, str]]] = defaultdict(dict)
    for edge in locator_result.edges:
        if not edge.usable:
            continue
        rows = [
            candidate_index[value]
            for value in edge.candidate_object_ids
            if value in candidate_index
        ]
        columns = [
            canonical_index[value]
            for value in edge.canonical_object_ids
            if value in canonical_index
        ]
        for row in rows:
            for column in columns:
                if not _blocking_compatible(
                    candidate_actors[row],
                    canonical_actors[column],
                    block_span_cm,
                ):
                    continue
                previous = locator_edges[row].get(column)
                proposed = (edge.identity_cost, "llm_pairwise_locator")
                if previous is None or proposed[0] < previous[0]:
                    locator_edges[row][column] = proposed

    edges: dict[int, dict[int, tuple[float, str]]] = {}
    reverse_edges: dict[int, set[int]] = defaultdict(set)
    for row in sorted(remaining_candidate):
        item = candidate_objects[row]
        candidates: set[int] = set()
        for value in item.assets:
            candidates.update(by_asset[value])
        for value in item.categories:
            candidates.update(by_category[value])
        for label in item.labels:
            for token in label.split():
                candidates.update(by_label_token[token])
        candidates.update(locator_edges.get(row, {}))
        plausible: dict[int, tuple[float, str]] = {}
        for column in sorted(candidates):
            identity = _identity_distance(item, canonical_objects[column])
            locator_identity = locator_edges.get(row, {}).get(column)
            if locator_identity is not None and locator_identity[0] < identity[0]:
                identity = locator_identity
            if (
                identity[0] <= IDENTITY_GATE
                and _blocking_compatible(
                    candidate_actors[row], canonical_actors[column], block_span_cm
                )
            ):
                plausible[column] = identity
                reverse_edges[column].add(row)
        if plausible:
            edges[row] = plausible

    seen_rows: set[int] = set()
    seen_columns: set[int] = set()
    sparse_component_count = 0
    for start in sorted(edges):
        if start in seen_rows:
            continue
        component_rows: set[int] = set()
        component_columns: set[int] = set()
        stack: list[tuple[str, int]] = [("row", start)]
        while stack:
            side, index = stack.pop()
            if side == "row":
                if index in seen_rows:
                    continue
                seen_rows.add(index)
                component_rows.add(index)
                stack.extend(("column", value) for value in edges[index])
            else:
                if index in seen_columns:
                    continue
                seen_columns.add(index)
                component_columns.add(index)
                stack.extend(("row", value) for value in reverse_edges[index])
        sparse_component_count += 1
        solve_block(component_rows, component_columns, edges)

    dense_cells = len(candidate_objects) * len(canonical_objects)
    source_counts: dict[str, int] = defaultdict(int)
    for _, _, _, source in matched:
        source_counts[source] += 1
    stats = {
        "strategy": "identity_size_spatial_blocks_then_bipartite_components",
        "blocking_policy": {
            "identity": [
                "exact_stable_actor_id",
                "exact_asset_path",
                "exact_trusted_category",
                "llm_pairwise_locator",
                "label_token_overlap",
            ],
            "size_log_threshold": SIZE_BLOCK_LOG_THRESHOLD,
            "spatial_cell_cm": rounded(block_span_cm),
            "fallback_cell_radius": 1,
            "long_tail_locator": (
                "shared_pairwise_identity_locator_then_label_token_fallback"
                if locator_result.backend != "disabled"
                else "label_token_fallback"
            ),
            "locator_backend": locator_result.backend,
            "locator_backend_error": locator_result.backend_error,
            "locator_usable_edge_count": sum(
                edge.usable for edge in locator_result.edges
            ),
            "locator_expanded_compatible_edge_count": sum(
                len(value) for value in locator_edges.values()
            ),
        },
        "dense_pair_count_avoided": dense_cells - cost_cell_count,
        "cost_matrix_cell_count": cost_cell_count,
        "cost_matrix_reduction_ratio": rounded(
            1.0 - cost_cell_count / dense_cells if dense_cells else 0.0
        ),
        "assignment_block_count": block_count,
        "tier_block_counts": tier_block_counts,
        "sparse_component_count": sparse_component_count,
        "largest_assignment_block": {
            "candidate_count": largest_block[0],
            "canonical_count": largest_block[1],
        },
        "plausible_edge_count": sum(len(value) for value in edges.values()),
        "match_source_counts": dict(sorted(source_counts.items())),
    }
    return sorted(matched), stats


def compare_detailed(
    candidate: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
    *,
    identity_locator: PairwiseIdentityLocator | None = None,
) -> dict[str, Any]:
    """Compare raw scenes and return metrics plus an auditable correspondence."""

    candidate_objects, candidate_population = build_logical_objects(
        candidate, side="candidate", trust_declared_groups=False
    )
    canonical_objects, canonical_population = build_logical_objects(
        canonical, side="canonical", trust_declared_groups=True
    )
    alignment = estimate_alignment(candidate_objects, canonical_objects)
    candidate_actors = [
        _aligned_actor(item.as_actor(), alignment) for item in candidate_objects
    ]
    canonical_actors = [item.as_actor() for item in canonical_objects]
    scale = _scale_cm([*candidate_actors, *canonical_actors])
    locator_result = PairwiseIdentityRetrievalResult(backend="disabled")
    if identity_locator is not None:
        backend_name = str(
            getattr(identity_locator, "name", type(identity_locator).__name__)
        )
        try:
            compatible_pairs = _locator_compatible_pairs(
                candidate_objects,
                candidate_actors,
                canonical_objects,
                canonical_actors,
                scale=scale,
            )
            located = identity_locator.locate(
                candidate_objects,
                canonical_objects,
                compatible_pairs=compatible_pairs,
            )
            if not isinstance(located, PairwiseIdentityRetrievalResult):
                raise TypeError(
                    "identity locator must return PairwiseIdentityRetrievalResult"
                )
            locator_result = located
        except Exception as exception:  # noqa: BLE001 - fail closed to exact tiers
            locator_result = PairwiseIdentityRetrievalResult(
                backend=backend_name,
                backend_error=f"{type(exception).__name__}: {str(exception)[:1200]}",
            )

    matched, assignment_stats = _assign_identity_aware(
        candidate_objects,
        candidate_actors,
        canonical_objects,
        canonical_actors,
        scale,
        locator_result,
    )

    candidate_actor_objects = [
        _make_object(f"candidate:actor:{index:04d}", [actor], "single_actor")
        for index, actor in enumerate(sorted(
            (value for value in candidate if _content_actor(value)),
            key=_actor_sort_key,
        ))
    ]
    canonical_actor_objects = [
        _make_object(f"canonical:actor:{index:04d}", [actor], "single_actor")
        for index, actor in enumerate(sorted(
            (value for value in canonical if _content_actor(value)),
            key=_actor_sort_key,
        ))
    ]
    raw_candidate_actor_evidence = [
        item.as_actor() for item in candidate_actor_objects
    ]
    aligned_candidate_actor_evidence = [
        _aligned_actor(item.as_actor(), alignment)
        for item in candidate_actor_objects
    ]
    canonical_actor_evidence = [
        item.as_actor() for item in canonical_actor_objects
    ]
    actor_matched, actor_assignment_stats = _assign_identity_aware(
        candidate_actor_objects,
        aligned_candidate_actor_evidence,
        canonical_actor_objects,
        canonical_actor_evidence,
        scale,
        PairwiseIdentityRetrievalResult(backend="disabled"),
    )
    attribute_metrics, attribute_audit = _attribute_comparison(
        candidate_actor_objects, canonical_actor_objects, actor_matched
    )
    structured_identity_matches = [
        value
        for row, column, _, _ in actor_matched
        if (value := _structured_identity_match(
            candidate_actor_objects[row], canonical_actor_objects[column]
        )) is not None
    ]

    paired_actors = [
        (
            aligned_candidate_actor_evidence[row],
            canonical_actor_evidence[column],
        )
        for row, column, _, _ in actor_matched
    ]
    positions = [
        math.dist(center_cm(left), center_cm(right))
        for left, right in paired_actors
    ]
    raw_paired_actors = [
        (raw_candidate_actor_evidence[row], canonical_actor_evidence[column])
        for row, column, _, _ in actor_matched
    ]
    absolute_location_errors = [
        math.dist(center_cm(left), center_cm(right))
        for left, right in raw_paired_actors
    ]
    absolute_axis_errors = {
        axis: [
            abs(center_cm(left)[index] - center_cm(right)[index])
            for left, right in raw_paired_actors
        ]
        for index, axis in enumerate(("x", "y", "z"))
    }
    aligned_axis_errors = {
        axis: [
            abs(center_cm(left)[index] - center_cm(right)[index])
            for left, right in paired_actors
        ]
        for index, axis in enumerate(("x", "y", "z"))
    }
    rotation_pairs = [
        (
            aligned_candidate_actor_evidence[row],
            canonical_actor_evidence[column],
        )
        for row, column, _, _ in actor_matched
    ]
    rotations = [_rotation_error(left, right) for left, right in rotation_pairs]
    scale_pairs = [
        (
            aligned_candidate_actor_evidence[row],
            canonical_actor_evidence[column],
        )
        for row, column, _, _ in actor_matched
    ]
    scales = [
        value
        for left, right in scale_pairs
        if (value := _log_rmse(
            [abs(item) for item in _scale_of(left)],
            [abs(item) for item in _scale_of(right)],
        )) is not None
    ]
    geometry_paired_actors = [
        (candidate_actors[row], canonical_actors[column])
        for row, column, _, _ in matched
    ]
    logical_positions = [
        math.dist(center_cm(left), center_cm(right))
        for left, right in geometry_paired_actors
    ]
    bounds_measured = bool(geometry_paired_actors) and all(
        has_bounds(left) and has_bounds(right)
        for left, right in geometry_paired_actors
    )
    sizes = [
        value
        for left, right in geometry_paired_actors
        if (value := _log_rmse(_size_cm(left), _size_cm(right))) is not None
    ]
    ious = [
        _footprint_iou(left, right) for left, right in geometry_paired_actors
    ] if bounds_measured else []
    pairwise, pairwise_count = _pairwise_layout(geometry_paired_actors, scale)

    over_fragmented, under_segmented, fragmentation_denominator, fragment_audit = (
        _fragmentation_by_asset(candidate, canonical)
    )
    matched_candidate = {row for row, _, _, _ in matched}
    matched_canonical = {column for _, column, _, _ in matched}
    matched_candidate_actors = {row for row, _, _, _ in actor_matched}
    matched_canonical_actors = {column for _, column, _, _ in actor_matched}
    center_translation = alignment.center_translation_cm
    origin_translation = alignment.origin_translation_cm
    global_translation = math.hypot(*origin_translation)
    global_center_offset = math.hypot(*center_translation)
    global_yaw = (
        cyclic_degrees(alignment.yaw_deg, 0.0)
        if len(alignment.anchor_pairs) >= 2
        else None
    )

    metrics = {
        "transform_correspondence_coverage": rounded(
            len(actor_matched)
            / max(len(candidate_actor_objects), len(canonical_actor_objects), 1)
        ),
        "logical_object_correspondence_coverage": rounded(
            len(matched) / max(len(candidate_objects), len(canonical_objects), 1)
        ),
        "absolute_location_error_cm": _summary(absolute_location_errors),
        "absolute_location_axis_error_cm": {
            axis: _summary(values)
            for axis, values in absolute_axis_errors.items()
        },
        "aligned_location_error_cm": _summary(positions),
        "aligned_location_axis_error_cm": {
            axis: _summary(values)
            for axis, values in aligned_axis_errors.items()
        },
        "cyclic_rotation_error_deg": _summary(rotations),
        "scale_log_error": _summary(scales),
        **attribute_metrics,
        "candidate_exported_actor_count": candidate_population["exported_actor_count"],
        "canonical_exported_actor_count": canonical_population["exported_actor_count"],
        "candidate_content_actor_count": candidate_population["content_actor_count"],
        "canonical_content_actor_count": canonical_population["content_actor_count"],
        "raw_actor_count_delta": (
            candidate_population["content_actor_count"]
            - canonical_population["content_actor_count"]
        ),
        "candidate_logical_object_count": len(candidate_objects),
        "canonical_logical_object_count": len(canonical_objects),
        "assignment_cost_matrix_cell_count": assignment_stats[
            "cost_matrix_cell_count"
        ],
        "assignment_cost_matrix_reduction_ratio": assignment_stats[
            "cost_matrix_reduction_ratio"
        ],
        "actor_assignment_cost_matrix_cell_count": actor_assignment_stats[
            "cost_matrix_cell_count"
        ],
        "actor_assignment_cost_matrix_reduction_ratio": actor_assignment_stats[
            "cost_matrix_reduction_ratio"
        ],
        "matched_logical_object_count": len(matched),
        "missing_logical_object_count": len(canonical_objects) - len(matched_canonical),
        "extra_logical_object_count": len(candidate_objects) - len(matched_candidate),
        "identity_comparison_pair_count": len(structured_identity_matches),
        "identity_mismatch_count": structured_identity_matches.count(False),
        "identity_match_rate": rounded(
            structured_identity_matches.count(True)
            / len(structured_identity_matches)
            if structured_identity_matches
            else None
        ),
        "over_fragmented_actor_count": over_fragmented,
        "under_segmented_actor_count": under_segmented,
        "fragmentation_error_rate": rounded(
            (over_fragmented + under_segmented) / fragmentation_denominator
            if fragmentation_denominator
            else None
        ),
        "global_center_offset_vector_cm": [
            rounded(value) for value in center_translation
        ],
        "global_center_offset_cm": rounded(global_center_offset),
        "global_translation_vector_cm": [
            rounded(value) for value in origin_translation
        ],
        "global_translation_error_cm": rounded(global_translation),
        "global_translation_normalized": rounded(global_translation / scale),
        "global_yaw_error_deg": (
            rounded(global_yaw) if global_yaw is not None else None
        ),
        "alignment_anchor_count": len(alignment.anchor_pairs),
        "alignment_anchor_rmse_cm": rounded(math.sqrt(
            sum(
                math.dist(
                    center_cm(_aligned_actor(left.as_actor(), alignment)),
                    center_cm(right.as_actor()),
                ) ** 2
                for left, right in alignment.anchor_pairs
            ) / len(alignment.anchor_pairs)
        )) if alignment.anchor_pairs else None,
        "normalization_scale_cm": rounded(scale, 3),
        "aligned_position_rmse_cm": rounded(math.sqrt(
            sum(value * value for value in logical_positions)
            / len(logical_positions)
        )) if logical_positions else None,
        "aligned_position_mean_cm": rounded(
            sum(logical_positions) / len(logical_positions)
        ) if logical_positions else None,
        "aligned_rotation_mean_deg": rounded(
            sum(rotations) / len(rotations)
        ) if rotations else None,
        "rotation_pair_count": len(rotation_pairs),
        "scale_log_rmse": rounded(
            math.sqrt(sum(value * value for value in scales) / len(scales))
        ) if scales else None,
        "bounds_measured": bounds_measured,
        "bounds_size_log_rmse": rounded(
            math.sqrt(sum(value * value for value in sizes) / len(sizes))
        ) if sizes and bounds_measured else None,
        "mean_footprint_iou": rounded(
            sum(ious) / len(ious)
        ) if ious else None,
        "pairwise_layout_error": rounded(pairwise),
        "pairwise_layout_pair_count": pairwise_count,
        "matched_actor_count": len(actor_matched),
        "missing_actor_count": (
            len(canonical_actor_objects) - len(matched_canonical_actors)
        ),
        "extra_actor_count": (
            len(candidate_actor_objects) - len(matched_candidate_actors)
        ),
        "position_rmse_cm": rounded(math.sqrt(
            sum(value * value for value in logical_positions)
            / len(logical_positions)
        )) if logical_positions else None,
        "position_mean_cm": rounded(
            sum(logical_positions) / len(logical_positions)
        ) if logical_positions else None,
        "rotation_mean_deg": rounded(
            sum(rotations) / len(rotations)
        ) if rotations else None,
    }

    logical_audit_matches = []
    for row, column, identity, identity_source in matched:
        candidate_actor = candidate_actors[row]
        canonical_actor = canonical_actors[column]
        bounds_pair = (
            has_bounds(candidate_actor) and has_bounds(canonical_actor)
        )
        logical_audit_matches.append({
            "candidate": _object_summary(candidate_objects[row]),
            "canonical": _object_summary(canonical_objects[column]),
            "identity_cost": rounded(identity),
            "identity_source": identity_source,
            "structured_identity_match": _structured_identity_match(
                candidate_objects[row], canonical_objects[column]
            ),
            "aligned_center_distance_cm": rounded(
                math.dist(center_cm(candidate_actor), center_cm(canonical_actor))
            ),
            "aligned_rotation_error_deg": (
                rounded(_rotation_error(candidate_actor, canonical_actor))
                if candidate_objects[row].rotation_available
                and canonical_objects[column].rotation_available
                else None
            ),
            "bounds_size_log_rmse": rounded(
                _log_rmse(_size_cm(candidate_actor), _size_cm(canonical_actor))
            ) if bounds_pair else None,
            "footprint_iou": rounded(
                _footprint_iou(candidate_actor, canonical_actor)
            ) if bounds_pair else None,
        })
    actor_audit_matches = []
    for row, column, identity, identity_source in actor_matched:
        aligned_actor = aligned_candidate_actor_evidence[row]
        raw_actor = raw_candidate_actor_evidence[row]
        canonical_actor = canonical_actor_evidence[column]
        bounds_pair = has_bounds(aligned_actor) and has_bounds(canonical_actor)
        actor_audit_matches.append({
            "candidate": _object_summary(candidate_actor_objects[row]),
            "canonical": _object_summary(canonical_actor_objects[column]),
            "identity_cost": rounded(identity),
            "identity_source": identity_source,
            "structured_identity_match": _structured_identity_match(
                candidate_actor_objects[row], canonical_actor_objects[column]
            ),
            "absolute_center_distance_cm": rounded(
                math.dist(center_cm(raw_actor), center_cm(canonical_actor))
            ),
            "aligned_center_distance_cm": rounded(
                math.dist(center_cm(aligned_actor), center_cm(canonical_actor))
            ),
            "aligned_rotation_error_deg": rounded(
                _rotation_error(aligned_actor, canonical_actor)
            ),
            "scale_log_error": rounded(
                _log_rmse(
                    [abs(item) for item in _scale_of(aligned_actor)],
                    [abs(item) for item in _scale_of(canonical_actor)],
                )
            ),
            "bounds_size_log_rmse": rounded(
                _log_rmse(_size_cm(aligned_actor), _size_cm(canonical_actor))
            ) if bounds_pair else None,
            "footprint_iou": rounded(
                _footprint_iou(aligned_actor, canonical_actor)
            ) if bounds_pair else None,
        })
    audit = {
        "schema_version": "gt-geometry-correspondence.v5",
        "attribute_comparison": attribute_audit,
        "assignment": assignment_stats,
        "actor_assignment": actor_assignment_stats,
        "identity_locator": locator_result.to_audit(),
        "population": {
            "candidate": candidate_population,
            "canonical": canonical_population,
        },
        "alignment": {
            "method": alignment.method,
            "anchor_count": len(alignment.anchor_pairs),
            "source_center_cm": [
                rounded(value, 3) for value in alignment.source_center_cm
            ],
            "target_center_cm": [
                rounded(value, 3) for value in alignment.target_center_cm
            ],
            "center_translation_cm": [
                rounded(value, 3) for value in center_translation
            ],
            "origin_transform_translation_cm": [
                rounded(value, 3) for value in alignment.origin_translation_cm
            ],
            "yaw_deg": rounded(alignment.yaw_deg),
            "yaw_measured": len(alignment.anchor_pairs) >= 2,
            "anchors": [
                {
                    "candidate_object_id": left.object_id,
                    "canonical_object_id": right.object_id,
                    "asset_path": left.assets[0] if left.assets else None,
                }
                for left, right in alignment.anchor_pairs
            ],
        },
        "fragmented_objects": {
            "candidate": [
                _object_summary(item)
                for item in candidate_objects if item.raw_actor_count > 1
            ],
            "canonical": [
                _object_summary(item)
                for item in canonical_objects if item.raw_actor_count > 1
            ],
        },
        "fragmentation_by_asset": fragment_audit,
        # `matches` remains the v4-compatible logical-object audit key.
        "matches": logical_audit_matches,
        "actor_matches": actor_audit_matches,
        "unmatched_candidate": [
            _object_summary(item)
            for index, item in enumerate(candidate_objects)
            if index not in matched_candidate
        ],
        "unmatched_canonical": [
            _object_summary(item)
            for index, item in enumerate(canonical_objects)
            if index not in matched_canonical
        ],
        "unmatched_candidate_actors": [
            _object_summary(item)
            for index, item in enumerate(candidate_actor_objects)
            if index not in matched_candidate_actors
        ],
        "unmatched_canonical_actors": [
            _object_summary(item)
            for index, item in enumerate(canonical_actor_objects)
            if index not in matched_canonical_actors
        ],
    }
    return {"metrics": metrics, "audit": audit}


def compare(
    candidate: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
    *,
    identity_locator: PairwiseIdentityLocator | None = None,
) -> dict[str, Any]:
    """Pure metric API retained for callers that do not need the audit."""

    return compare_detailed(
        candidate,
        canonical,
        identity_locator=identity_locator,
    )["metrics"]


__all__ = [
    "AUTO_GROUP_GAP_CM",
    "AUTO_GROUP_GAP_CAP_CM",
    "AUTO_GROUP_GAP_RATIO",
    "IDENTITY_GATE",
    "MATCH_WEIGHTS",
    "GeometryObject",
    "actor_pair_attribute_metrics",
    "actor_pair_metrics",
    "build_logical_objects",
    "compare",
    "compare_detailed",
    "estimate_alignment",
]
