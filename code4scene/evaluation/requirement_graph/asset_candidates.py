"""Asset candidate discovery for camera targeting.

The preferred runtime path inventories actors directly from the current UE
render world, ranks their live ActorLabels against every claim as a diagnostic
search signal, and
reads geometry from those exact actor wrappers.  It never needs an IR
``placed[]`` row or a label-to-inventory existence join.  The older IR ranking
and exact live-actor join APIs remain available for replay compatibility.

Every score in this module is useful only for deciding *where to look*.  It is
never semantic evidence and must not be converted into a claim verdict.  A
ranked actor must still be photographed and verified from RGB by the visual
evaluation pipeline.

There are no Unreal or SPEAR imports here.  The small session adapter uses
duck-typed objects, which keeps parsing, ranking, joining, and pose planning
available to offline tests and replay tools.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from .actions import clamp_camera_pose
from .contracts import (
    CameraPose,
    ClaimType,
    PromptClaim,
    SceneBounds,
    coerce_claim_type,
)


DEFAULT_TOP_K = 8
DEFAULT_MIN_SCORE = 0.30
DEFAULT_CLUSTER_DISTANCE_CM = 250.0

_TECHNICAL_TOKENS = frozenset(
    {
        "sm",
        "bp",
        "sk",
        "skm",
        "staticmesh",
        "skeletalmesh",
        "blueprint",
        "mesh",
    }
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "beside",
        "by",
        "contains",
        "containing",
        "for",
        "from",
        "has",
        "have",
        "in",
        "inside",
        "is",
        "near",
        "next",
        "of",
        "on",
        "scene",
        "shows",
        "the",
        "there",
        "to",
        "with",
    }
)
_GROUND_TOKENS = frozenset(
    {
        "floor",
        "ground",
        "landscape",
        "pavement",
        "road",
        "sidewalk",
        "terrain",
    }
)
_RELATION_PREDICATE_TOKENS = frozenset(
    {
        "above",
        "below",
        "behind",
        "between",
        "front",
        "outside",
        "over",
        "under",
        "underneath",
        "within",
    }
)
_COMPOUND_WORDS = {
    "lamppost": "lamp post",
    "snowbank": "snow bank",
    "snowdrift": "snow drift",
    "snowisland": "snow island",
    "snowpile": "snow pile",
    "streetlamp": "street lamp",
    "streetlight": "street light",
    "trashbin": "trash bin",
}

# The sets are intentionally small and concrete.  They improve localization
# recall without pretending to be an ontology or a semantic judge.
_SYNONYM_GROUPS = (
    ("lantern", "lamp", "street light", "street lamp", "lamp post"),
    ("snow pile", "snow bank", "snow drift", "snow island"),
    ("sofa", "couch", "settee"),
    ("trash bin", "garbage can", "waste bin", "rubbish bin"),
    ("rock", "boulder", "stone"),
    ("car", "automobile", "vehicle"),
    ("crate", "box"),
    ("barrel", "drum"),
    ("statue", "sculpture"),
    ("bench", "seat"),
)

_FIELD_WEIGHTS = {
    "name": 1.0,
    "asset_id": 0.97,
    "path_basename": 0.94,
    "id": 0.90,
    "category": 0.78,
}


class IrSceneError(ValueError):
    """Raised when an IR document cannot provide a valid ``placed[]`` list."""


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector3(value: Any, *, name: str) -> tuple[float, float, float]:
    value = _complete_future(value)
    if isinstance(value, Mapping):
        lowered = {str(key).casefold(): item for key, item in value.items()}
        if all(axis in lowered for axis in ("x", "y", "z")):
            values = (lowered["x"], lowered["y"], lowered["z"])
        elif "location" in lowered:
            return _vector3(lowered["location"], name=name)
        elif "location_cm" in lowered:
            return _vector3(lowered["location_cm"], name=name)
        else:
            raise ValueError(f"{name} mapping must contain X/Y/Z")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 3:
            raise ValueError(f"{name} must contain three coordinates")
        values = value[:3]
    else:
        values = None
        for axes in (("x", "y", "z"), ("X", "Y", "Z")):
            if all(hasattr(value, axis) for axis in axes):
                values = tuple(getattr(value, axis) for axis in axes)
                break
        if values is None:
            raise ValueError(f"{name} must contain X/Y/Z")
    result = tuple(_finite_float(item, name=name) for item in values)
    return result  # type: ignore[return-value]


def _optional_vector3(value: Any, *, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return _vector3(value, name=name)


def asset_path_basename(path: str) -> str:
    """Return the object/filename portion of an Unreal-style asset path."""

    leaf = _clean_text(path).replace("\\", "/").rsplit("/", 1)[-1]
    if not leaf:
        return ""
    # ``/Game/Foo/SM_Chair.SM_Chair`` names the package and object.  The object
    # side is the most specific portion; ordinary extensions use the stem.
    pieces = [piece for piece in leaf.split(".") if piece]
    if len(pieces) >= 2 and pieces[-1].casefold() not in {
        "uasset",
        "umap",
        "fbx",
        "obj",
    }:
        return pieces[-1]
    return pieces[0] if pieces else leaf


@lru_cache(maxsize=16_384)
def _normalize_asset_string(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^0-9A-Za-z]+", " ", text).casefold()

    tokens: list[str] = []
    for token in text.split():
        token = re.sub(r"(?<=[a-z])\d+$", "", token)
        if not token or token.isdigit() or token in _TECHNICAL_TOKENS:
            continue
        replacement = _COMPOUND_WORDS.get(token, token)
        tokens.extend(part for part in replacement.split() if part)
    return " ".join(tokens)


def normalize_asset_text(value: Any) -> str:
    """Normalize an IR/name field for metadata retrieval.

    The transformation is deterministic and intentionally lexical: Unicode
    case folding, CamelCase/separator splitting, common UE type-token removal,
    compound splitting, and removal of numeric instance suffixes.
    """

    return _normalize_asset_string(_clean_text(value))


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalized_term(value: str) -> str:
    return " ".join(_singular(token) for token in normalize_asset_text(value).split())


def expand_claim_terms(claim: str | PromptClaim) -> tuple[str, ...]:
    """Return normalized lexical and simple-synonym search terms for a claim."""

    text = claim.text if isinstance(claim, PromptClaim) else str(claim)
    normalized = normalize_asset_text(text)
    ignored_tokens = set(_STOPWORDS)
    if (
        isinstance(claim, PromptClaim)
        and _coerced_claim_type(claim) is ClaimType.SPATIAL_RELATION
    ):
        # A relation predicate is not an actor identity.  For example,
        # ``outside a gothic cathedral`` should retrieve the cathedral rather
        # than an ``OutsideVolume`` helper whose label happens to contain the
        # relation word.
        ignored_tokens.update(_RELATION_PREDICATE_TOKENS)
    content_tokens = [
        _singular(token)
        for token in normalized.split()
        if token not in ignored_tokens and len(token) > 1
    ]

    terms: list[str] = []

    def add(value: str) -> None:
        value = _normalized_term(value)
        if value and value not in terms:
            terms.append(value)

    if content_tokens:
        add(" ".join(content_tokens))
    for token in content_tokens:
        add(token)
    for size in (2, 3):
        for index in range(max(0, len(content_tokens) - size + 1)):
            add(" ".join(content_tokens[index : index + size]))

    token_set = set(content_tokens)
    compact_claim = "".join(content_tokens)
    for group in _SYNONYM_GROUPS:
        normalized_group = tuple(_normalized_term(value) for value in group)
        matched = False
        for value in normalized_group:
            value_tokens = value.split()
            if set(value_tokens).issubset(token_set) or "".join(value_tokens) in compact_claim:
                matched = True
                break
            # A small typo in a concrete object noun should still unlock its
            # synonym set (for example ``lantren`` -> lantern -> lamp).  Keep
            # the threshold high and compare individual tokens so unrelated
            # descriptive prose does not fan out into broad metadata queries.
            if any(
                len(claim_token) >= 5
                and len(group_token) >= 5
                and SequenceMatcher(None, claim_token, group_token).ratio() >= 0.82
                for claim_token in content_tokens
                for group_token in value_tokens
            ):
                matched = True
                break
        if matched:
            for value in normalized_group:
                add(value)
                for token in value.split():
                    add(token)
    return tuple(terms)


def _coerced_claim_type(
    claim: PromptClaim | ClaimType | str,
) -> ClaimType | None:
    """Return a valid claim type without letting routing errors escape."""

    if isinstance(claim, PromptClaim):
        claim_type: ClaimType | str = claim.type
    else:
        claim_type = claim
    try:
        return coerce_claim_type(claim_type)
    except (TypeError, ValueError):
        return None


def claim_supports_asset_discovery(
    claim: PromptClaim | ClaimType | str,
) -> bool:
    """Whether a claim may use live ActorLabel matches as search diagnostics.

    Discovery is intentionally broad: a relation such as ``outside a gothic
    cathedral`` can retrieve a cathedral anchor, and a surface claim such as
    ``wet cobblestone`` can retrieve a cobblestone actor.  A match remains only
    an acquisition hint and is never a semantic verdict.
    """

    return _coerced_claim_type(claim) is not None


def claim_supports_asset_search(claim: PromptClaim | ClaimType | str) -> bool:
    """Whether a live actor may anchor RGB acquisition for this claim.

    Every valid claim may use a lexical live-actor match to decide where to
    photograph.  This is deliberately broader than semantic confirmation: a
    fog volume may anchor an atmosphere view and a cathedral actor may anchor a
    relation view, but neither metadata match proves the claim.  The final
    decision still comes exclusively from RGB.
    """

    return _coerced_claim_type(claim) is not None


def claim_supports_candidate_preverification(
    claim: PromptClaim | ClaimType | str,
) -> bool:
    """Whether one actor's context/close pair may skip ordinary target search.

    A single actor cannot prove a scene identity, atmosphere, spatial relation,
    or surface-wide condition.  Those claims may still use live geometry as a
    camera anchor, but must continue through the normal target phase.
    """

    return _coerced_claim_type(claim) in {
        ClaimType.OBJECT,
        ClaimType.OBJECT_ATTRIBUTE,
    }


@dataclass(frozen=True, slots=True)
class PlacedAsset:
    """One sanitized planning record from ``ir_scene.json:placed[]``."""

    id: str
    asset_id: str = ""
    name: str = ""
    category: str = ""
    path: str = ""
    location_cm: tuple[float, float, float] | None = None
    yaw_deg: float | None = None
    footprint_m: tuple[float, float] | None = None
    radius_m: float | None = None
    measured: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    is_ground: bool = False
    group: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        placed_id = _clean_text(self.id)
        if not placed_id:
            raise ValueError("placed asset id must be non-empty")
        object.__setattr__(self, "id", placed_id)
        for name in ("asset_id", "name", "category", "path"):
            object.__setattr__(self, name, _clean_text(getattr(self, name)))
        object.__setattr__(
            self,
            "location_cm",
            _optional_vector3(self.location_cm, name="location_cm"),
        )
        if self.yaw_deg is not None:
            object.__setattr__(self, "yaw_deg", _finite_float(self.yaw_deg, name="yaw_deg"))
        if self.footprint_m is not None:
            if len(self.footprint_m) != 2:
                raise ValueError("footprint_m must contain width and depth")
            footprint = tuple(
                _finite_float(value, name="footprint_m") for value in self.footprint_m
            )
            if any(value < 0 for value in footprint):
                raise ValueError("footprint_m values must be non-negative")
            object.__setattr__(self, "footprint_m", footprint)
        if self.radius_m is not None:
            radius = _finite_float(self.radius_m, name="radius_m")
            if radius < 0:
                raise ValueError("radius_m must be non-negative")
            object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "is_ground", bool(self.is_ground))
        object.__setattr__(self, "group", _clean_text(self.group) or None)

    @property
    def path_basename(self) -> str:
        return asset_path_basename(self.path)

    @property
    def metadata_fields(self) -> Mapping[str, str]:
        return {
            "name": self.name,
            "asset_id": self.asset_id,
            "path_basename": self.path_basename,
            "category": self.category,
            "id": self.id,
        }

    @property
    def asset_key(self) -> str:
        for value in (self.asset_id, self.path_basename, self.name, self.id):
            normalized = normalize_asset_text(value)
            if normalized:
                return normalized
        return self.id.casefold()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "asset_id": self.asset_id,
            "name": self.name,
            "category": self.category,
            "path": self.path,
            "location_cm": list(self.location_cm) if self.location_cm is not None else None,
            "yaw_deg": self.yaw_deg,
            "footprint_m": list(self.footprint_m) if self.footprint_m is not None else None,
            "radius_m": self.radius_m,
            "is_ground": self.is_ground,
            "group": self.group,
            "measured": dict(self.measured) if self.measured is not None else None,
        }


def _placed_asset_from_mapping(value: Mapping[str, Any]) -> PlacedAsset:
    footprint = value.get("footprint")
    footprint_m: tuple[float, float] | None = None
    if isinstance(footprint, Mapping):
        width = footprint.get("width", footprint.get("w_m"))
        depth = footprint.get("depth", footprint.get("d_m"))
        if width is not None and depth is not None:
            footprint_m = (width, depth)  # type: ignore[assignment]
    elif isinstance(footprint, Sequence) and not isinstance(
        footprint, (str, bytes, bytearray)
    ):
        if len(footprint) >= 2:
            footprint_m = (footprint[0], footprint[1])  # type: ignore[assignment]
    return PlacedAsset(
        id=_clean_text(value.get("id")),
        asset_id=_clean_text(value.get("asset_id", value.get("assetId"))),
        name=_clean_text(value.get("name")),
        category=_clean_text(value.get("category")),
        path=_clean_text(value.get("path", value.get("asset_path"))),
        location_cm=value.get("location", value.get("location_cm")),
        yaw_deg=value.get("yaw_deg", value.get("yaw")),
        footprint_m=footprint_m,
        radius_m=value.get("radius"),
        measured=value.get("measured") if isinstance(value.get("measured"), Mapping) else None,
        is_ground=bool(value.get("isGround", value.get("is_ground", False))),
        group=_clean_text(value.get("group")) or None,
        raw=dict(value),
    )


def parse_placed_assets(
    document: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> tuple[PlacedAsset, ...]:
    """Parse ``placed[]`` from a decoded IR document.

    A malformed individual planning record is skipped by default because one
    bad optional placement should not disable localization for the entire
    scene.  ``strict=True`` is available to validators and unit tests.
    """

    if isinstance(document, Mapping):
        if "placed" not in document:
            raise IrSceneError("IR document does not contain placed[]")
        values = document["placed"]
    else:
        values = document
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise IrSceneError("IR placed must be a list")

    assets: list[PlacedAsset] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            errors.append(f"placed[{index}] is not an object")
            continue
        try:
            asset = _placed_asset_from_mapping(value)
        except (TypeError, ValueError) as exc:
            errors.append(f"placed[{index}]: {exc}")
            continue
        # Duplicate planning rows are not additional live actors.  Preserve the
        # first stable id so actor joining remains deterministic.
        stable_key = asset.id.casefold()
        if stable_key in seen_ids:
            errors.append(f"placed[{index}] duplicates id {asset.id!r}")
            continue
        seen_ids.add(stable_key)
        assets.append(asset)

    if strict and errors:
        raise IrSceneError("; ".join(errors))
    if values and not assets:
        detail = f": {errors[0]}" if errors else ""
        raise IrSceneError(f"IR placed[] contains no valid entries{detail}")
    return tuple(assets)


def load_ir_scene(path: str | Path, *, strict: bool = False) -> tuple[PlacedAsset, ...]:
    """Load and parse placed assets from an IR JSON file."""

    source = Path(path).expanduser()
    try:
        with source.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IrSceneError(f"could not read scene IR {source}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise IrSceneError("IR root must be a JSON object")
    return parse_placed_assets(document, strict=strict)


@dataclass(frozen=True, slots=True)
class AssetCandidate:
    """One ranked metadata candidate; ``score`` is never a visual verdict."""

    asset: PlacedAsset
    score: float
    matched_fields: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    cluster_member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        score = _finite_float(self.score, name="candidate score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("candidate score must be between 0 and 1")
        object.__setattr__(self, "score", score)
        members = tuple(
            dict.fromkeys((self.asset.id, *(str(value) for value in self.cluster_member_ids)))
        )
        object.__setattr__(self, "cluster_member_ids", members)
        for name in ("matched_fields", "matched_terms", "reasons"):
            object.__setattr__(
                self,
                name,
                tuple(
                    dict.fromkeys(
                        _clean_text(value)
                        for value in getattr(self, name)
                        if _clean_text(value)
                    )
                ),
            )

    @property
    def candidate_id(self) -> str:
        return self.asset.id

    @property
    def metadata_score(self) -> float:
        """Explicit alias emphasizing that this value is ranking-only."""

        return self.score

    @property
    def cluster_size(self) -> int:
        return len(self.cluster_member_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "asset": self.asset.to_dict(),
            "metadata_score": self.metadata_score,
            "matched_fields": list(self.matched_fields),
            "matched_terms": list(self.matched_terms),
            "reasons": list(self.reasons),
            "cluster_member_ids": list(self.cluster_member_ids),
            "cluster_size": self.cluster_size,
        }


def _field_similarity(term: str, field_value: str) -> tuple[float, str]:
    # Both values originate in ``expand_claim_terms``/``normalize_asset_text``.
    # Avoid re-running Unicode and regex normalization for every term/field
    # pair: large generated scenes routinely contain more than 1,000 rows.
    term = " ".join(_singular(token) for token in term.split())
    field_value = " ".join(_singular(token) for token in field_value.split())
    if not term or not field_value:
        return 0.0, ""
    if term == field_value:
        return 1.0, "exact"

    term_tokens = term.split()
    field_tokens = field_value.split()
    if term in field_tokens:
        return 0.97, "token_exact"
    term_compact = "".join(term_tokens)
    field_compact = "".join(field_tokens)
    if min(len(term_compact), len(field_compact)) >= 4 and (
        term_compact in field_compact or field_compact in term_compact
    ):
        return 0.91, "substring"

    left = set(term_tokens)
    right = set(field_tokens)
    overlap = len(left & right)
    best_score = 0.0
    best_reason = ""
    if overlap:
        containment = overlap / max(1, min(len(left), len(right)))
        jaccard = overlap / max(1, len(left | right))
        best_score = 0.58 + 0.25 * containment + 0.12 * jaccard
        best_reason = "token_overlap"

    # Fuzzy retrieval is for noun typos, not for comparing every piece of a
    # sentence to every metadata phrase.  Requiring a single query token,
    # similar lengths, and the same leading character both reduces false
    # positives and keeps 1,000+-placement IR ranking comfortably interactive.
    fuzzy_pairs = [
        (term_compact, token)
        for token in field_tokens
        if len(term_tokens) == 1
        and len(term_compact) >= 4
        and len(token) >= 4
        and term_compact[0] == token[0]
        and abs(len(term_compact) - len(token)) <= 2
    ]
    fuzzy = max(
        (
            SequenceMatcher(None, left_value, right_value).ratio()
            for left_value, right_value in fuzzy_pairs
            if max(len(left_value), len(right_value)) >= 4
        ),
        default=0.0,
    )
    if fuzzy >= 0.62 and fuzzy * 0.82 > best_score:
        best_score = fuzzy * 0.82
        best_reason = "fuzzy"
    return min(1.0, best_score), best_reason


def _score_asset(asset: PlacedAsset, terms: Sequence[str]) -> AssetCandidate:
    field_scores: dict[str, tuple[float, str, str]] = {}
    for field_name, raw_value in asset.metadata_fields.items():
        normalized = normalize_asset_text(raw_value)
        best = (0.0, "", "")
        for term in terms:
            score, reason = _field_similarity(term, normalized)
            weighted = score * _FIELD_WEIGHTS[field_name]
            if weighted > best[0]:
                best = (weighted, reason, term)
        if best[0] > 0:
            field_scores[field_name] = best

    if not field_scores:
        return AssetCandidate(asset=asset, score=0.0)
    ordered = sorted(field_scores.items(), key=lambda item: (-item[1][0], item[0]))
    primary_score = ordered[0][1][0]
    corroboration = min(0.06, 0.015 * (len(ordered) - 1))
    score = min(1.0, primary_score + corroboration)
    return AssetCandidate(
        asset=asset,
        score=score,
        matched_fields=tuple(name for name, _ in ordered),
        matched_terms=tuple(value[2] for _, value in ordered if value[2]),
        reasons=tuple(f"{name}:{value[1]}" for name, value in ordered if value[1]),
    )


def _is_ground_asset(asset: PlacedAsset) -> bool:
    if asset.is_ground:
        return True
    tokens = set(normalize_asset_text(asset.category).split())
    return bool(tokens & _GROUND_TOKENS)


def _distance_cm(left: PlacedAsset, right: PlacedAsset) -> float | None:
    if left.location_cm is None or right.location_cm is None:
        return None
    return math.dist(left.location_cm, right.location_cm)


def _same_asset(left: PlacedAsset, right: PlacedAsset) -> bool:
    return left.asset_key == right.asset_key


def _merge_candidates(left: AssetCandidate, right: AssetCandidate) -> AssetCandidate:
    return replace(
        left,
        score=max(left.score, right.score),
        matched_fields=tuple(dict.fromkeys((*left.matched_fields, *right.matched_fields))),
        matched_terms=tuple(dict.fromkeys((*left.matched_terms, *right.matched_terms))),
        reasons=tuple(dict.fromkeys((*left.reasons, *right.reasons))),
        cluster_member_ids=tuple(
            dict.fromkeys((*left.cluster_member_ids, *right.cluster_member_ids))
        ),
    )


def cluster_asset_candidates(
    candidates: Sequence[AssetCandidate],
    *,
    cluster_distance_cm: float = DEFAULT_CLUSTER_DISTANCE_CM,
    max_per_asset: int = 3,
    max_ground_candidates: int = 2,
) -> tuple[AssetCandidate, ...]:
    """Cluster near duplicates and cap repetitive same/ground assets.

    Overflow ids remain in the representative's ``cluster_member_ids``.  Live
    joining therefore tries every clustered stable id rather than trusting that
    the representative planning row was actually spawned.
    """

    distance_limit = _finite_float(cluster_distance_cm, name="cluster_distance_cm")
    if distance_limit < 0:
        raise ValueError("cluster_distance_cm must be non-negative")
    if max_per_asset < 1 or max_ground_candidates < 1:
        raise ValueError("candidate caps must be positive")

    ranked = sorted(candidates, key=lambda value: (-value.score, value.asset.id.casefold()))
    spatial: list[AssetCandidate] = []
    for candidate in ranked:
        merged = False
        for index, representative in enumerate(spatial):
            distance = _distance_cm(candidate.asset, representative.asset)
            if candidate.asset.id.casefold() == representative.asset.id.casefold():
                should_merge = True
            elif distance is None:
                should_merge = False
            else:
                very_close = distance <= min(50.0, distance_limit)
                same_near = (
                    _same_asset(candidate.asset, representative.asset)
                    and distance <= distance_limit
                )
                ground_near = (
                    _is_ground_asset(candidate.asset)
                    and _is_ground_asset(representative.asset)
                    and distance <= distance_limit * 1.5
                )
                should_merge = very_close or same_near or ground_near
            if should_merge:
                spatial[index] = _merge_candidates(representative, candidate)
                merged = True
                break
        if not merged:
            spatial.append(candidate)

    selected: list[AssetCandidate] = []
    per_asset: dict[str, int] = {}
    ground_count = 0
    for candidate in spatial:
        key = candidate.asset.asset_key
        ground = _is_ground_asset(candidate.asset)
        if per_asset.get(key, 0) < max_per_asset and (
            not ground or ground_count < max_ground_candidates
        ):
            selected.append(candidate)
            per_asset[key] = per_asset.get(key, 0) + 1
            if ground:
                ground_count += 1
            continue

        eligible = [
            (index, existing)
            for index, existing in enumerate(selected)
            if existing.asset.asset_key == key
            or (ground and _is_ground_asset(existing.asset))
        ]
        if not eligible:
            # This is reachable only when a caller uses unusual caps alongside
            # an empty selected set; retaining one target is safer than losing
            # every possible observation position.
            selected.append(candidate)
            continue
        index, representative = min(
            eligible,
            key=lambda item: _distance_cm(candidate.asset, item[1].asset)
            if _distance_cm(candidate.asset, item[1].asset) is not None
            else math.inf,
        )
        selected[index] = _merge_candidates(representative, candidate)

    return tuple(sorted(selected, key=lambda value: (-value.score, value.asset.id.casefold())))


def rank_asset_candidates(
    claim: str | PromptClaim,
    placed_assets: Sequence[PlacedAsset | Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    cluster_distance_cm: float = DEFAULT_CLUSTER_DISTANCE_CM,
    max_per_asset: int = 3,
    max_ground_candidates: int = 2,
) -> tuple[AssetCandidate, ...]:
    """Rank and diversify top-k IR candidates for an object-like claim."""

    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if top_k == 0:
        return ()
    threshold = _finite_float(min_score, name="min_score")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    if isinstance(claim, PromptClaim) and not claim_supports_asset_search(claim):
        return ()

    assets = tuple(
        value if isinstance(value, PlacedAsset) else _placed_asset_from_mapping(value)
        for value in placed_assets
    )
    terms = expand_claim_terms(claim)
    if not terms:
        return ()
    ranked = tuple(
        candidate
        for candidate in (_score_asset(asset, terms) for asset in assets)
        if candidate.score >= threshold
    )
    clustered = cluster_asset_candidates(
        ranked,
        cluster_distance_cm=cluster_distance_cm,
        max_per_asset=max_per_asset,
        max_ground_candidates=max_ground_candidates,
    )
    return clustered[:top_k]


def find_asset_candidates(
    claim: str | PromptClaim,
    ir_scene: str | Path | Mapping[str, Any] | Sequence[PlacedAsset | Mapping[str, Any]],
    **ranking_options: Any,
) -> tuple[AssetCandidate, ...]:
    """Convenience API combining IR loading/parsing with candidate ranking."""

    if isinstance(ir_scene, (str, Path)):
        assets = load_ir_scene(ir_scene)
    elif isinstance(ir_scene, Mapping):
        assets = parse_placed_assets(ir_scene)
    else:
        assets = tuple(
            value if isinstance(value, PlacedAsset) else _placed_asset_from_mapping(value)
            for value in ir_scene
        )
    return rank_asset_candidates(claim, assets, **ranking_options)


@dataclass(frozen=True, slots=True)
class LiveActorTarget:
    """A ranked candidate joined to a fresh live actor transform."""

    candidate: AssetCandidate
    placed_id: str
    stable_name: str
    location_cm: tuple[float, float, float]
    bounds_center_cm: tuple[float, float, float]
    extent_cm: tuple[float, float, float]
    actor_label: str | None = None
    unreal_name: str | None = None
    actor: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("placed_id", "stable_name"):
            value = _clean_text(getattr(self, name))
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "location_cm", _vector3(self.location_cm, name="location_cm"))
        object.__setattr__(
            self,
            "bounds_center_cm",
            _vector3(self.bounds_center_cm, name="bounds_center_cm"),
        )
        extent = _vector3(self.extent_cm, name="extent_cm")
        if any(value < 0 for value in extent):
            raise ValueError("extent_cm values must be non-negative")
        object.__setattr__(self, "extent_cm", extent)
        object.__setattr__(self, "actor_label", _clean_text(self.actor_label) or None)
        object.__setattr__(self, "unreal_name", _clean_text(self.unreal_name) or None)

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def live_actor_id(self) -> str:
        """Authoritative UE inventory identity (legacy name: ``placed_id``)."""

        return self.placed_id

    def to_dict(self) -> dict[str, Any]:
        # The reflected actor wrapper is purposefully excluded from artifacts.
        payload = {
            "candidate_id": self.candidate_id,
            "placed_id": self.placed_id,
            "metadata_score": self.candidate.metadata_score,
            "stable_name": self.stable_name,
            "actor_label": self.actor_label,
            "unreal_name": self.unreal_name,
            "live_location_cm": list(self.location_cm),
            "live_bounds_center_cm": list(self.bounds_center_cm),
            "live_extent_cm": list(self.extent_cm),
        }
        if self.candidate.asset.raw.get("source") == "ue_live_inventory":
            payload.update(
                live_actor_id=self.live_actor_id,
                live_id_match_score=self.candidate.metadata_score,
                geometry_available=True,
                reason="live_actor_target_ready",
            )
        return payload


@dataclass(frozen=True, slots=True)
class ActorJoinDiagnostic:
    candidate_id: str
    metadata_score: float
    attempted_placed_ids: tuple[str, ...]
    joined: bool
    reason: str
    joined_placed_id: str | None = None
    stable_name: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "metadata_score": self.metadata_score,
            "attempted_placed_ids": list(self.attempted_placed_ids),
            "joined": self.joined,
            "reason": self.reason,
            "joined_placed_id": self.joined_placed_id,
            "stable_name": self.stable_name,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ActorJoinResult:
    targets: tuple[LiveActorTarget, ...]
    diagnostics: tuple[ActorJoinDiagnostic, ...]
    inventory_available: bool
    inventory_count: int | None
    fallback_reason: str | None = None

    @property
    def joined(self) -> bool:
        return bool(self.targets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": [target.to_dict() for target in self.targets],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "inventory_available": self.inventory_available,
            "inventory_count": self.inventory_count,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True, slots=True)
class LiveActorCandidate:
    """One claim-ranked actor from the authoritative live UE inventory.

    ``live_actor_key`` is the full key returned by
    ``find_actors_as_dict(include_unreal_name=True)`` and is used only as the
    unique actor identity.  ``match_text`` is the live ActorLabel used for
    lexical claim ranking.  Keeping the actor wrapper on this record avoids a
    label-to-inventory lookup after ranking; duplicate labels therefore remain
    valid, distinct actor instances.
    """

    live_actor_key: str
    match_text: str
    match_score: float
    stable_name: str
    actor_label: str | None = None
    unreal_name: str | None = None
    matched_terms: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    actor: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("live_actor_key", "match_text", "stable_name"):
            value = _clean_text(getattr(self, name))
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        score = _finite_float(self.match_score, name="live actor match score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("live actor match score must be between 0 and 1")
        object.__setattr__(self, "match_score", score)
        object.__setattr__(self, "actor_label", _clean_text(self.actor_label) or None)
        object.__setattr__(self, "unreal_name", _clean_text(self.unreal_name) or None)
        for name in ("matched_terms", "reasons"):
            object.__setattr__(
                self,
                name,
                tuple(
                    dict.fromkeys(
                        _clean_text(value)
                        for value in getattr(self, name)
                        if _clean_text(value)
                    )
                ),
            )

    @property
    def candidate_id(self) -> str:
        """Compatibility alias used by the existing camera-search records."""

        return self.live_actor_key

    @property
    def live_actor_id(self) -> str:
        return self.live_actor_key

    @property
    def metadata_score(self) -> float:
        """Compatibility alias; the value is a live-ID lexical match score."""

        return self.match_score

    def as_asset_candidate(self) -> AssetCandidate:
        """Adapt a live candidate to the legacy camera-target container."""

        asset = PlacedAsset(
            id=self.live_actor_key,
            name=self.match_text,
            raw={
                "source": "ue_live_inventory",
                "live_actor_key": self.live_actor_key,
                "actor_label": self.actor_label,
                "unreal_name": self.unreal_name,
            },
        )
        return AssetCandidate(
            asset=asset,
            score=self.match_score,
            matched_fields=("live_actor_label",),
            matched_terms=self.matched_terms,
            reasons=self.reasons,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "live_actor_id": self.live_actor_id,
            "live_actor_key": self.live_actor_key,
            "match_text": self.match_text,
            "live_id_match_score": self.match_score,
            "stable_name": self.stable_name,
            "actor_label": self.match_text,
            "reported_actor_label": self.actor_label,
            "unreal_name": self.unreal_name,
            "matched_terms": list(self.matched_terms),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class LiveActorInventoryDiagnostic:
    live_actor_key: str
    match_text: str | None
    actor_label: str | None
    unreal_name: str | None
    usable: bool
    reason: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "live_actor_id": self.live_actor_key,
            "live_actor_key": self.live_actor_key,
            "match_text": self.match_text,
            "actor_label": self.actor_label,
            "unreal_name": self.unreal_name,
            "usable": self.usable,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class LiveCandidateDiagnostic:
    claim_id: str
    live_actor_key: str
    match_text: str
    live_id_match_score: float
    targeted: bool
    reason: str
    error: str | None = None

    @property
    def live_actor_id(self) -> str:
        return self.live_actor_key

    @property
    def geometry_available(self) -> bool:
        return self.targeted

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "live_actor_id": self.live_actor_id,
            "live_actor_key": self.live_actor_key,
            "match_text": self.match_text,
            "actor_label": self.match_text,
            "live_id_match_score": self.live_id_match_score,
            "geometry_available": self.geometry_available,
            "targeted": self.targeted,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class LiveClaimCandidates:
    claim_id: str
    claim_text: str
    ranked_candidates: tuple[LiveActorCandidate, ...]
    targets: tuple[LiveActorTarget, ...]
    diagnostics: tuple[LiveCandidateDiagnostic, ...]
    fallback_reason: str | None = None

    @property
    def candidates(self) -> tuple[LiveActorCandidate, ...]:
        return self.ranked_candidates

    @property
    def joined(self) -> bool:
        return bool(self.targets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "ranked_candidates": [
                candidate.to_dict() for candidate in self.ranked_candidates
            ],
            "targets": [target.to_dict() for target in self.targets],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True, slots=True)
class LiveActorDiscoveryResult:
    claims: tuple[LiveClaimCandidates, ...]
    inventory_diagnostics: tuple[LiveActorInventoryDiagnostic, ...]
    inventory_available: bool
    inventory_count: int | None
    fallback_reason: str | None = None
    error: str | None = None

    @property
    def joined(self) -> bool:
        return any(claim.targets for claim in self.claims)

    @property
    def targets(self) -> tuple[LiveActorTarget, ...]:
        unique: dict[str, LiveActorTarget] = {}
        for claim in self.claims:
            for target in claim.targets:
                unique.setdefault(target.placed_id, target)
        return tuple(unique.values())

    def for_claim(self, claim_id: str) -> LiveClaimCandidates | None:
        key = _clean_text(claim_id).casefold()
        for claim in self.claims:
            if claim.claim_id.casefold() == key:
                return claim
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [claim.to_dict() for claim in self.claims],
            "inventory_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.inventory_diagnostics
            ],
            "inventory_available": self.inventory_available,
            "inventory_count": self.inventory_count,
            "fallback_reason": self.fallback_reason,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class _ActorRecord:
    stable_name: str
    actor: Any
    aliases: tuple[str, ...]
    actor_label: str | None = None
    unreal_name: str | None = None


def _complete_future(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter()
        except TypeError:
            pass
    result = getattr(value, "result", None)
    if callable(result):
        try:
            return result()
        except TypeError:
            pass
    return value


def _join_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", _clean_text(value).casefold())


def _stable_name_variants(value: Any) -> tuple[str, ...]:
    text = _clean_text(value)
    if not text:
        return ()
    variants = [text]
    for separator in (":", "|", ";"):
        if separator in text:
            variants.append(text.split(separator, 1)[0])
    return tuple(dict.fromkeys(variant for variant in variants if variant))


def _safe_actor_text(actor: Any, names: Sequence[str]) -> str | None:
    if isinstance(actor, Mapping):
        for name in names:
            value = actor.get(name)
            if value is not None and not callable(value):
                text = _clean_text(value)
                if text:
                    return text
    try:
        instance_values = vars(actor)
    except TypeError:
        instance_values = {}
    for name in names:
        explicitly_declared = name in instance_values or any(
            name in base.__dict__ for base in type(actor).__mro__
        )
        try:
            value = getattr(actor, name)
        except (AttributeError, KeyError, TypeError):
            continue
        if callable(value):
            try:
                value = _complete_future(value())
            except Exception:
                continue
        elif not explicitly_declared:
            # SPEAR UnrealObject creates a reflected property proxy for every
            # unknown attribute via ``__getattr__``.  Stringifying that proxy
            # would manufacture a bogus actor label; only accept concrete
            # instance/class attributes or successful methods.
            continue
        text = _clean_text(value)
        if text:
            return text
    return None


def _request_live_actor_label(actor: Any, *, prefer_async: bool = False) -> Any:
    """Issue a reflected actor-label call without completing its future.

    SPEAR's stable-name fallback in a standalone game world is the Unreal
    object name, even when an editor label was serialized into the map.  The
    label UFunction remains available in the development Editor executable,
    so it is the authoritative bridge back to ``placed.id`` for those maps.

    Unknown attributes on ``spear.UnrealObject`` are property proxies rather
    than raising ``AttributeError``.  Consult reflected function metadata when
    present so a missing label function fails closed instead of invoking a
    bogus proxy.
    """

    call_owner = actor
    if prefer_async:
        async_owner = getattr(actor, "call_async", None)
        if async_owner is not None:
            call_owner = async_owner

    derived_state = getattr(actor, "derived_state", None)
    reflected_functions: Mapping[str, Any] | None = None
    if isinstance(derived_state, Mapping):
        value = derived_state.get("unqualified_function_descs")
        if isinstance(value, Mapping):
            reflected_functions = value

    for name in ("GetActorLabel", "K2_GetActorLabel", "get_actor_label"):
        if reflected_functions is not None and name not in reflected_functions:
            continue
        try:
            label_call = getattr(call_owner, name)
        except (AttributeError, KeyError, TypeError):
            continue
        if not callable(label_call):
            continue
        try:
            # Do not manufacture a default label when the serialized label is
            # absent.  Older/fake interfaces may expose a no-argument method.
            return label_call(bCreateIfNone=False)
        except TypeError:
            try:
                return label_call(False)
            except TypeError:
                return label_call()
    raise AttributeError("actor has no reflected GetActorLabel")


def _actor_records(
    actors: Mapping[str, Any] | Sequence[Any] | Iterable[Any],
    *,
    inspect_actor_text: bool = True,
    strict_inventory_keys: bool = False,
) -> tuple[_ActorRecord, ...]:
    actors_are_mapping = isinstance(actors, Mapping)
    if actors_are_mapping:
        pairs: Iterable[tuple[str | None, Any]] = actors.items()
    else:
        pairs = ((None, actor) for actor in actors)

    records: list[_ActorRecord] = []
    for supplied_name, value in pairs:
        if (
            strict_inventory_keys
            and actors_are_mapping
            and not isinstance(supplied_name, str)
        ):
            raise TypeError("live actor inventory keys must be strings")
        metadata = value if isinstance(value, Mapping) else None
        actor = metadata.get("actor", value) if metadata is not None else value
        stable_name = _clean_text(supplied_name)
        reflected_state = getattr(actor, "derived_state", None)
        is_reflected_actor = isinstance(reflected_state, Mapping) and getattr(
            actor, "call_async", None
        ) is not None
        if not stable_name:
            stable_name = _safe_actor_text(
                value,
                ("stable_name", "stableName", "label", "actor_label", "name", "id"),
            ) or ""
        stable_parts = stable_name.split(":", 1)
        actor_label = stable_parts[0] if stable_name else None
        unreal_name = stable_parts[1] if len(stable_parts) == 2 else None

        # ``find_actors_as_dict(include_unreal_name=True)`` already places the
        # stable label and Unreal name in its map key.  Avoid issuing thousands
        # of redundant GetActorLabel/GetName RPCs while inventorying a dense
        # reflected world.  Plain fakes/records still support method or field
        # labels in addition to their supplied key.
        if inspect_actor_text and not is_reflected_actor:
            actor_label = _safe_actor_text(
                value,
                (
                    "actor_label",
                    "label",
                    "get_actor_label",
                    "GetActorLabel",
                    "K2_GetActorLabel",
                ),
            ) or actor_label
            if actor is not value:
                actor_label = _safe_actor_text(
                    actor,
                    (
                        "actor_label",
                        "label",
                        "get_actor_label",
                        "GetActorLabel",
                        "K2_GetActorLabel",
                    ),
                ) or actor_label
            unreal_name = _safe_actor_text(
                value,
                ("unreal_name", "unrealName", "get_name", "GetName", "name"),
            ) or unreal_name
            if actor is not value:
                unreal_name = _safe_actor_text(
                    actor, ("unreal_name", "get_name", "GetName", "name")
                ) or unreal_name
        if not stable_name:
            stable_name = actor_label or unreal_name or f"actor_{len(records)}"
        aliases: list[str] = []
        for item in (stable_name, actor_label, unreal_name):
            aliases.extend(_stable_name_variants(item))
        records.append(
            _ActorRecord(
                stable_name=stable_name,
                actor=actor,
                aliases=tuple(dict.fromkeys(aliases)),
                actor_label=actor_label,
                unreal_name=unreal_name,
            )
        )
    return tuple(records)


def _record_has_standalone_name_fallback(record: _ActorRecord) -> bool:
    """Return whether SPEAR supplied only the Unreal object name as identity."""

    if not record.actor_label or not record.unreal_name:
        return False
    return _join_key(record.actor_label) == _join_key(record.unreal_name)


def _hydrate_actor_labels(
    records: Sequence[_ActorRecord], pending: Mapping[int, Any | Exception]
) -> tuple[_ActorRecord, ...]:
    """Complete requested labels and add successful values as join aliases."""

    hydrated: list[_ActorRecord] = []
    for record in records:
        value = pending.get(id(record.actor))
        if value is None or isinstance(value, Exception):
            hydrated.append(record)
            continue
        try:
            actor_label = _clean_text(_complete_future(value))
        except Exception:
            hydrated.append(record)
            continue
        if not actor_label:
            hydrated.append(record)
            continue
        aliases = list(record.aliases)
        aliases.extend(_stable_name_variants(actor_label))
        hydrated.append(
            replace(
                record,
                aliases=tuple(dict.fromkeys(aliases)),
                actor_label=actor_label,
            )
        )
    return tuple(hydrated)


def _extract_bounds(
    value: Any, *, fallback_center: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    value = _complete_future(value)
    center: Any = None
    extent: Any = None
    if isinstance(value, Mapping):
        lowered = {str(key).casefold(): item for key, item in value.items()}
        for key in ("origin", "center", "bounds_center", "bounds_center_cm"):
            if key in lowered:
                center = lowered[key]
                break
        for key in ("boxextent", "box_extent", "extent", "extent_cm"):
            if key in lowered:
                extent = lowered[key]
                break
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) >= 2:
            center, extent = value[0], value[1]
    else:
        for key in ("Origin", "origin", "Center", "center"):
            if hasattr(value, key):
                center = getattr(value, key)
                break
        for key in ("BoxExtent", "box_extent", "Extent", "extent"):
            if hasattr(value, key):
                extent = getattr(value, key)
                break
    if extent is None:
        raise ValueError("GetActorBounds did not return BoxExtent")
    center_xyz = fallback_center if center is None else _vector3(center, name="bounds center")
    extent_xyz = _vector3(extent, name="bounds extent")
    if any(value < 0 for value in extent_xyz):
        raise ValueError("bounds extent must be non-negative")
    return center_xyz, extent_xyz


def _mapping_value(value: Mapping[str, Any], names: Sequence[str]) -> Any:
    lowered = {str(key).casefold(): item for key, item in value.items()}
    for name in names:
        if name.casefold() in lowered:
            return lowered[name.casefold()]
    return None


_LiveGeometry = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _request_live_geometry(
    actor: Any, *, prefer_async: bool = False
) -> tuple[Any, Any]:
    """Issue location/bounds calls without completing possible futures."""

    location_raw: Any = None
    bounds_raw: Any = None
    if isinstance(actor, Mapping):
        location_raw = _mapping_value(actor, ("location_cm", "location"))
        bounds_raw = _mapping_value(actor, ("bounds", "actor_bounds"))

    if location_raw is None:
        call_owner = actor
        if prefer_async:
            async_owner = getattr(actor, "call_async", None)
            if async_owner is not None:
                call_owner = async_owner
        location_call = getattr(call_owner, "K2_GetActorLocation", None)
        if not callable(location_call):
            raise AttributeError("actor has no K2_GetActorLocation")
        location_raw = location_call()

    if bounds_raw is None:
        call_owner = actor
        if prefer_async:
            async_owner = getattr(actor, "call_async", None)
            if async_owner is not None:
                call_owner = async_owner
        bounds_call = getattr(call_owner, "GetActorBounds", None)
        if not callable(bounds_call):
            raise AttributeError("actor has no GetActorBounds")
        try:
            # ``Origin`` and ``BoxExtent`` are reflected output parameters,
            # not the formal return value of AActor::GetActorBounds.  SPEAR
            # exposes output parameters only when ``as_dict`` is requested.
            bounds_raw = bounds_call(
                bOnlyCollidingComponents=False,
                as_dict=True,
            )
        except TypeError:
            try:
                bounds_raw = bounds_call(bOnlyCollidingComponents=False)
            except TypeError:
                try:
                    bounds_raw = bounds_call(False)
                except TypeError:
                    bounds_raw = bounds_call()
    return location_raw, bounds_raw


def _complete_live_geometry(raw: tuple[Any, Any]) -> _LiveGeometry:
    location_raw, bounds_raw = raw
    location = _vector3(location_raw, name="live actor location")
    center, extent = _extract_bounds(bounds_raw, fallback_center=location)
    return location, center, extent


def _read_live_geometry(actor: Any) -> _LiveGeometry:
    return _complete_live_geometry(_request_live_geometry(actor))


def _record_indexes(
    records: Sequence[_ActorRecord],
) -> tuple[dict[str, list[_ActorRecord]], dict[str, list[_ActorRecord]]]:
    exact: dict[str, list[_ActorRecord]] = {}
    compact: dict[str, list[_ActorRecord]] = {}
    for record in records:
        for alias in record.aliases:
            exact.setdefault(alias.casefold(), []).append(record)
            key = _join_key(alias)
            if key:
                compact.setdefault(key, []).append(record)
    return exact, compact


def _match_candidate_record(
    candidate: AssetCandidate,
    exact: Mapping[str, Sequence[_ActorRecord]],
    compact: Mapping[str, Sequence[_ActorRecord]],
) -> tuple[_ActorRecord | None, str | None, bool]:
    ambiguous = False
    for placed_id in candidate.cluster_member_ids:
        matches = exact.get(placed_id.casefold(), ())
        unique = {id(value.actor): value for value in matches}
        if not unique:
            matches = compact.get(_join_key(placed_id), ())
            unique = {id(value.actor): value for value in matches}
        if len(unique) == 1:
            return next(iter(unique.values())), placed_id, ambiguous
        if len(unique) > 1:
            ambiguous = True
    return None, None, ambiguous


def _join_failure_result(
    candidates: Sequence[AssetCandidate],
    *,
    reason: str,
    error: str | None = None,
    inventory_available: bool = False,
) -> ActorJoinResult:
    diagnostics = tuple(
        ActorJoinDiagnostic(
            candidate_id=candidate.candidate_id,
            metadata_score=candidate.metadata_score,
            attempted_placed_ids=candidate.cluster_member_ids,
            joined=False,
            reason=reason,
            error=error,
        )
        for candidate in candidates
    )
    return ActorJoinResult(
        targets=(),
        diagnostics=diagnostics,
        inventory_available=inventory_available,
        inventory_count=None,
        fallback_reason=reason,
    )


def _join_actor_records(
    candidates: Sequence[AssetCandidate],
    records: Sequence[_ActorRecord],
    *,
    max_targets: int | None,
    geometry_reader: Callable[[_ActorRecord], _LiveGeometry],
) -> ActorJoinResult:
    exact, compact = _record_indexes(records)
    targets: list[LiveActorTarget] = []
    diagnostics: list[ActorJoinDiagnostic] = []
    used_actor_ids: set[int] = set()
    for candidate in candidates:
        if max_targets is not None and len(targets) >= max_targets:
            diagnostics.append(
                ActorJoinDiagnostic(
                    candidate_id=candidate.candidate_id,
                    metadata_score=candidate.metadata_score,
                    attempted_placed_ids=candidate.cluster_member_ids,
                    joined=False,
                    reason="target_limit_reached",
                )
            )
            continue

        match, matched_id, ambiguous = _match_candidate_record(
            candidate, exact, compact
        )
        if match is None:
            diagnostics.append(
                ActorJoinDiagnostic(
                    candidate_id=candidate.candidate_id,
                    metadata_score=candidate.metadata_score,
                    attempted_placed_ids=candidate.cluster_member_ids,
                    joined=False,
                    reason="ambiguous_live_actor_name" if ambiguous else "live_actor_not_found",
                )
            )
            continue
        if id(match.actor) in used_actor_ids:
            diagnostics.append(
                ActorJoinDiagnostic(
                    candidate_id=candidate.candidate_id,
                    metadata_score=candidate.metadata_score,
                    attempted_placed_ids=candidate.cluster_member_ids,
                    joined=False,
                    reason="duplicate_live_actor",
                    joined_placed_id=matched_id,
                    stable_name=match.stable_name,
                )
            )
            continue
        try:
            location, center, extent = geometry_reader(match)
        except Exception as exc:
            diagnostics.append(
                ActorJoinDiagnostic(
                    candidate_id=candidate.candidate_id,
                    metadata_score=candidate.metadata_score,
                    attempted_placed_ids=candidate.cluster_member_ids,
                    joined=False,
                    reason="live_actor_geometry_unavailable",
                    joined_placed_id=matched_id,
                    stable_name=match.stable_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        used_actor_ids.add(id(match.actor))
        targets.append(
            LiveActorTarget(
                candidate=candidate,
                placed_id=matched_id or candidate.candidate_id,
                stable_name=match.stable_name,
                actor_label=match.actor_label,
                unreal_name=match.unreal_name,
                location_cm=location,
                bounds_center_cm=center,
                extent_cm=extent,
                actor=match.actor,
            )
        )
        diagnostics.append(
            ActorJoinDiagnostic(
                candidate_id=candidate.candidate_id,
                metadata_score=candidate.metadata_score,
                attempted_placed_ids=candidate.cluster_member_ids,
                joined=True,
                reason="joined",
                joined_placed_id=matched_id,
                stable_name=match.stable_name,
            )
        )

    fallback_reason = None if targets else (
        "live_actor_inventory_empty" if not records else "no_live_candidates_joined"
    )
    return ActorJoinResult(
        targets=tuple(targets),
        diagnostics=tuple(diagnostics),
        inventory_available=True,
        inventory_count=len(records),
        fallback_reason=fallback_reason,
    )


def join_live_actor_candidates(
    candidates: Sequence[AssetCandidate],
    actors: Mapping[str, Any] | Sequence[Any] | Iterable[Any],
    *,
    max_targets: int | None = None,
) -> ActorJoinResult:
    """Join ranked IR candidates to a generic live/fake actor inventory.

    Joining is by exact stable name/label (case-insensitive), with a compact
    separator-insensitive fallback only when it identifies one unique actor.
    It never falls back to fuzzy asset-name matching.
    """

    if max_targets is not None and max_targets < 0:
        raise ValueError("max_targets must be non-negative")
    candidate_values = tuple(candidates)
    try:
        records = _actor_records(actors)
    except Exception as exc:
        return _join_failure_result(
            candidate_values,
            reason="live_actor_inventory_invalid",
            error=f"{type(exc).__name__}: {exc}",
        )
    return _join_actor_records(
        candidate_values,
        records,
        max_targets=max_targets,
        geometry_reader=lambda record: _read_live_geometry(record.actor),
    )


def join_candidates_in_session(
    candidates: Sequence[AssetCandidate],
    session: Any,
    *,
    max_targets: int | None = None,
) -> ActorJoinResult:
    """Inventory and join actors in the render world's global-sync frame.

    The authoritative service is strictly ``session.game.unreal_service``.
    ``session.instance`` is intentionally never consulted because it may still
    refer to the pre-travel world.
    """

    candidate_values = tuple(candidates)
    ensure_world = getattr(session, "ensure_unreal_world", None)
    if callable(ensure_world):
        try:
            ensure_world()
        except Exception as exc:
            return _join_failure_result(
                candidate_values,
                reason="live_actor_inventory_error",
                error=f"{type(exc).__name__}: {exc}",
            )
    run_frame = getattr(session, "_run_frame", None)
    game = getattr(session, "game", None)
    unreal_service = getattr(game, "unreal_service", None)
    find_actors = getattr(unreal_service, "find_actors_as_dict", None)
    if not callable(run_frame) or not callable(find_actors):
        return _join_failure_result(
            candidate_values,
            reason="live_actor_inventory_unavailable",
            error="session.game.unreal_service.find_actors_as_dict is unavailable",
        )

    def body() -> tuple[
        tuple[_ActorRecord, ...],
        dict[int, tuple[Any, Any] | Exception],
        dict[int, Any | Exception],
    ]:
        actors = find_actors(
            include_unreal_name=True,
            as_unreal_object=True,
            with_sp_funcs=True,
        )
        if actors is None:
            actors = {}
        records = _actor_records(actors)
        exact, compact = _record_indexes(records)
        pending: dict[int, tuple[Any, Any] | Exception] = {}
        pending_labels: dict[int, Any | Exception] = {}
        if max_targets != 0:
            for candidate in candidate_values:
                match, _matched_id, _ambiguous = _match_candidate_record(
                    candidate, exact, compact
                )
                if match is None or id(match.actor) in pending:
                    continue
                try:
                    # ``call_async`` batches both RPCs in this one global-sync
                    # frame when exposed by a real SPEAR UnrealObject.  Generic
                    # fakes/direct actors follow the same request boundary.
                    pending[id(match.actor)] = _request_live_geometry(
                        match.actor,
                        prefer_async=True,
                    )
                except Exception as exc:
                    pending[id(match.actor)] = exc
            # In ``UnrealEditor -game`` (a standalone game world), SPEAR's
            # stable-name fallback is ``AActor::GetName`` rather than the
            # serialized editor label.  Only activate the more expensive
            # label inventory when the fast stable-name path found no
            # candidate at all and the keys exhibit that exact fallback.
            if candidate_values and not pending:
                for record in records:
                    if not _record_has_standalone_name_fallback(record):
                        continue
                    try:
                        pending_labels[id(record.actor)] = _request_live_actor_label(
                            record.actor,
                            prefer_async=True,
                        )
                    except Exception as exc:
                        pending_labels[id(record.actor)] = exc
        return records, pending, pending_labels

    try:
        records, pending, pending_labels = run_frame(body)
    except Exception as exc:
        return _join_failure_result(
            candidate_values,
            reason="live_actor_inventory_error",
            error=f"{type(exc).__name__}: {exc}",
        )

    if pending_labels:
        # Async labels become readable only after the inventory frame ends.
        # Re-index those live labels, then use a second global-sync frame for
        # geometry RPCs on matched actors only.  Any unavailable/duplicate
        # label remains a normal fail-closed join miss.
        records = _hydrate_actor_labels(records, pending_labels)

        def geometry_body() -> dict[int, tuple[Any, Any] | Exception]:
            exact, compact = _record_indexes(records)
            requested: dict[int, tuple[Any, Any] | Exception] = {}
            if max_targets != 0:
                for candidate in candidate_values:
                    match, _matched_id, _ambiguous = _match_candidate_record(
                        candidate, exact, compact
                    )
                    if match is None or id(match.actor) in requested:
                        continue
                    try:
                        requested[id(match.actor)] = _request_live_geometry(
                            match.actor,
                            prefer_async=True,
                        )
                    except Exception as exc:
                        requested[id(match.actor)] = exc
            return requested

        try:
            pending = run_frame(geometry_body)
        except Exception as exc:
            return _join_failure_result(
                candidate_values,
                reason="live_actor_inventory_error",
                error=f"{type(exc).__name__}: {exc}",
            )

    def read_requested_geometry(record: _ActorRecord) -> _LiveGeometry:
        value = pending.get(id(record.actor))
        if value is None:
            raise RuntimeError("live actor geometry was not requested")
        if isinstance(value, Exception):
            raise value
        # SPEAR async futures become complete only after ``end_frame``.  This
        # function runs after ``session._run_frame`` returns, mirroring
        # SimWorldClientSession.call_async_actor's completion semantics.
        return _complete_live_geometry(value)

    return _join_actor_records(
        candidate_values,
        records,
        max_targets=max_targets,
        geometry_reader=read_requested_geometry,
    )


_GENERIC_LIVE_LABEL_RE = re.compile(
    r"^(?:actor|staticmeshactor|skeletalmeshactor|cameraactor|cinecameraactor|"
    r"instancedfoliageactor|levelinstance|brush|worldsettings)(?:_\d+)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _LiveActorIdentity:
    live_actor_key: str
    match_text: str
    record: _ActorRecord


def _generic_live_label(value: Any) -> bool:
    text = re.sub(r"[^0-9A-Za-z_]+", "", _clean_text(value))
    return bool(text and _GENERIC_LIVE_LABEL_RE.fullmatch(text))


def _complete_live_actor_labels(
    records: Sequence[_ActorRecord],
    pending: Mapping[int, Any | Exception],
) -> tuple[tuple[_ActorRecord, ...], dict[int, str]]:
    """Complete label futures after ``end_frame`` and retain failures."""

    completed: list[_ActorRecord] = []
    errors: dict[int, str] = {}
    for record in records:
        actor_key = id(record.actor)
        value = pending.get(actor_key)
        if value is None:
            errors[actor_key] = "GetActorLabel was not requested"
            completed.append(record)
            continue
        if isinstance(value, Exception):
            errors[actor_key] = f"{type(value).__name__}: {value}"
            completed.append(record)
            continue
        try:
            actor_label = _clean_text(_complete_future(value))
        except Exception as exc:
            errors[actor_key] = f"{type(exc).__name__}: {exc}"
            completed.append(record)
            continue
        if not actor_label:
            errors[actor_key] = "GetActorLabel returned an empty label"
            completed.append(record)
            continue
        aliases = list(record.aliases)
        aliases.extend(_stable_name_variants(actor_label))
        completed.append(
            replace(
                record,
                actor_label=actor_label,
                aliases=tuple(dict.fromkeys(aliases)),
            )
        )
    return tuple(completed), errors


def _record_match_text(record: _ActorRecord) -> tuple[str | None, str]:
    """Choose semantic UE text without treating generic object names as IDs."""

    stable_label = record.stable_name.split(":", 1)[0]
    values = tuple(
        dict.fromkeys(
            text
            for text in (_clean_text(record.actor_label), _clean_text(stable_label))
            if text
        )
    )
    saw_generic = False
    for text in values:
        if _generic_live_label(text):
            saw_generic = True
            continue
        return text, "live_actor_label"
    return None, "generic_live_actor_label" if saw_generic else "live_actor_label_unavailable"


def _live_actor_identities(
    records: Sequence[_ActorRecord],
    *,
    label_errors: Mapping[int, str] | None = None,
) -> tuple[tuple[_LiveActorIdentity, ...], tuple[LiveActorInventoryDiagnostic, ...]]:
    """Validate authoritative inventory keys while allowing duplicate labels."""

    label_errors = label_errors or {}
    match_values = [_record_match_text(record) for record in records]
    key_groups: dict[str, list[int]] = {}
    actor_groups: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        # The full inventory key is the UE service's authoritative identity.
        # Do not apply the separator-insensitive legacy IR join normalization:
        # punctuation-distinct keys can represent distinct live actors.
        live_actor_key = _clean_text(record.stable_name)
        if live_actor_key:
            key_groups.setdefault(live_actor_key, []).append(index)
        actor_groups.setdefault(id(record.actor), []).append(index)

    duplicate_keys = {
        index
        for indexes in key_groups.values()
        if len(indexes) > 1
        for index in indexes
    }
    duplicate_wrappers = {
        index
        for indexes in actor_groups.values()
        if len(indexes) > 1
        for index in indexes
    }

    identities: list[_LiveActorIdentity] = []
    diagnostics: list[LiveActorInventoryDiagnostic] = []
    for index, record in enumerate(records):
        live_actor_key = _clean_text(record.stable_name)
        match_text, match_reason = match_values[index]
        error = label_errors.get(id(record.actor))
        if not _join_key(live_actor_key):
            usable = False
            reason = "live_actor_key_unavailable"
        elif index in duplicate_keys:
            usable = False
            reason = "ambiguous_live_actor_key"
        elif index in duplicate_wrappers:
            usable = False
            reason = "duplicate_live_actor_wrapper"
        elif match_text is None:
            usable = False
            reason = match_reason
        else:
            usable = True
            reason = "live_actor_ready_with_label_error" if error else "live_actor_ready"
            identities.append(
                _LiveActorIdentity(
                    live_actor_key=live_actor_key,
                    match_text=match_text,
                    record=record,
                )
            )
        diagnostics.append(
            LiveActorInventoryDiagnostic(
                live_actor_key=live_actor_key,
                match_text=match_text,
                actor_label=record.actor_label,
                unreal_name=record.unreal_name,
                usable=usable,
                reason=reason,
                error=error,
            )
        )
    return tuple(identities), tuple(diagnostics)


def _validate_live_discovery_inputs(
    claims: Sequence[PromptClaim],
    *,
    top_k: int,
    min_score: float,
    max_targets_per_claim: int | None,
) -> tuple[tuple[PromptClaim, ...], float]:
    claim_values = tuple(claims)
    if any(not isinstance(claim, PromptClaim) for claim in claim_values):
        raise TypeError("claims must contain PromptClaim values")
    claim_ids = [claim.id.casefold() for claim in claim_values]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("claim ids must be unique")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    threshold = _finite_float(min_score, name="min_score")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    if max_targets_per_claim is not None and max_targets_per_claim < 0:
        raise ValueError("max_targets_per_claim must be non-negative")
    return claim_values, threshold


def _rank_live_actor_ids(
    claim: PromptClaim,
    identities: Sequence[_LiveActorIdentity],
    *,
    top_k: int,
    min_score: float,
) -> tuple[LiveActorCandidate, ...]:
    if top_k == 0 or not claim_supports_asset_discovery(claim):
        return ()
    terms = expand_claim_terms(claim)
    if not terms:
        return ()

    candidates: list[LiveActorCandidate] = []
    for identity in identities:
        normalized = normalize_asset_text(identity.match_text)
        best_score = 0.0
        best_reason = ""
        best_term = ""
        for term in terms:
            score, reason = _field_similarity(term, normalized)
            if score > best_score:
                best_score = score
                best_reason = reason
                best_term = term
        if best_score < min_score:
            continue
        record = identity.record
        candidates.append(
            LiveActorCandidate(
                live_actor_key=identity.live_actor_key,
                match_text=identity.match_text,
                match_score=best_score,
                stable_name=record.stable_name,
                actor_label=record.actor_label,
                unreal_name=record.unreal_name,
                matched_terms=(best_term,) if best_term else (),
                reasons=(f"live_actor_label:{best_reason}",) if best_reason else (),
                actor=record.actor,
            )
        )
    candidates.sort(
        key=lambda candidate: (-candidate.match_score, candidate.live_actor_key.casefold())
    )
    return tuple(candidates[:top_k])


def _rank_live_claims(
    claims: Sequence[PromptClaim],
    identities: Sequence[_LiveActorIdentity],
    *,
    top_k: int,
    min_score: float,
) -> dict[str, tuple[LiveActorCandidate, ...]]:
    return {
        claim.id: _rank_live_actor_ids(
            claim,
            identities,
            top_k=top_k,
            min_score=min_score,
        )
        for claim in claims
    }


def _targeted_live_candidates(
    claims: Sequence[PromptClaim],
    ranked_by_claim: Mapping[str, Sequence[LiveActorCandidate]],
    *,
    max_targets_per_claim: int | None,
) -> tuple[LiveActorCandidate, ...]:
    unique: dict[int, LiveActorCandidate] = {}
    for claim in claims:
        candidates = tuple(ranked_by_claim.get(claim.id, ()))
        limit = len(candidates) if max_targets_per_claim is None else max_targets_per_claim
        for candidate in candidates[:limit]:
            unique.setdefault(id(candidate.actor), candidate)
    return tuple(unique.values())


def _finalize_live_actor_discovery(
    claims: Sequence[PromptClaim],
    ranked_by_claim: Mapping[str, Sequence[LiveActorCandidate]],
    geometry_by_actor: Mapping[int, _LiveGeometry | Exception],
    *,
    inventory_diagnostics: Sequence[LiveActorInventoryDiagnostic],
    inventory_count: int,
    max_targets_per_claim: int | None,
    result_error: str | None = None,
) -> LiveActorDiscoveryResult:
    claim_results: list[LiveClaimCandidates] = []
    any_ranked = False
    for claim in claims:
        ranked = tuple(ranked_by_claim.get(claim.id, ()))
        any_ranked = any_ranked or bool(ranked)
        limit = len(ranked) if max_targets_per_claim is None else max_targets_per_claim
        targets: list[LiveActorTarget] = []
        diagnostics: list[LiveCandidateDiagnostic] = []
        for index, candidate in enumerate(ranked):
            if index >= limit:
                diagnostics.append(
                    LiveCandidateDiagnostic(
                        claim_id=claim.id,
                        live_actor_key=candidate.live_actor_key,
                        match_text=candidate.match_text,
                        live_id_match_score=candidate.match_score,
                        targeted=False,
                        reason="target_limit_reached",
                    )
                )
                continue
            geometry = geometry_by_actor.get(id(candidate.actor))
            if geometry is None:
                geometry = RuntimeError("live actor geometry was not requested")
            if isinstance(geometry, Exception):
                diagnostics.append(
                    LiveCandidateDiagnostic(
                        claim_id=claim.id,
                        live_actor_key=candidate.live_actor_key,
                        match_text=candidate.match_text,
                        live_id_match_score=candidate.match_score,
                        targeted=False,
                        reason="live_actor_geometry_unavailable",
                        error=f"{type(geometry).__name__}: {geometry}",
                    )
                )
                continue
            location, center, extent = geometry
            targets.append(
                LiveActorTarget(
                    candidate=candidate.as_asset_candidate(),
                    placed_id=candidate.live_actor_key,
                    stable_name=candidate.stable_name,
                    actor_label=candidate.match_text,
                    unreal_name=candidate.unreal_name,
                    location_cm=location,
                    bounds_center_cm=center,
                    extent_cm=extent,
                    actor=candidate.actor,
                )
            )
            diagnostics.append(
                LiveCandidateDiagnostic(
                    claim_id=claim.id,
                    live_actor_key=candidate.live_actor_key,
                    match_text=candidate.match_text,
                    live_id_match_score=candidate.match_score,
                    targeted=True,
                    reason="live_actor_target_ready",
                )
            )

        if targets:
            fallback_reason = None
        elif not claim_supports_asset_discovery(claim):
            fallback_reason = "claim_not_asset_discoverable"
        elif not ranked:
            fallback_reason = "no_live_id_candidates"
        elif limit == 0:
            fallback_reason = "target_limit_reached"
        else:
            fallback_reason = "no_live_candidate_geometry"
        claim_results.append(
            LiveClaimCandidates(
                claim_id=claim.id,
                claim_text=claim.text,
                ranked_candidates=ranked,
                targets=tuple(targets),
                diagnostics=tuple(diagnostics),
                fallback_reason=fallback_reason,
            )
        )

    if any(result.targets for result in claim_results):
        fallback_reason = None
    elif not inventory_count:
        fallback_reason = "live_actor_inventory_empty"
    elif not any(diagnostic.usable for diagnostic in inventory_diagnostics):
        fallback_reason = "no_usable_live_actor_ids"
    elif not any_ranked:
        fallback_reason = "no_live_id_candidates"
    elif max_targets_per_claim == 0:
        fallback_reason = "target_limit_reached"
    else:
        fallback_reason = "no_live_candidate_geometry"
    return LiveActorDiscoveryResult(
        claims=tuple(claim_results),
        inventory_diagnostics=tuple(inventory_diagnostics),
        inventory_available=True,
        inventory_count=inventory_count,
        fallback_reason=fallback_reason,
        error=result_error,
    )


def _live_discovery_failure(
    claims: Sequence[PromptClaim],
    *,
    reason: str,
    error: str,
) -> LiveActorDiscoveryResult:
    return LiveActorDiscoveryResult(
        claims=tuple(
            LiveClaimCandidates(
                claim_id=claim.id,
                claim_text=claim.text,
                ranked_candidates=(),
                targets=(),
                diagnostics=(),
                fallback_reason=reason,
            )
            for claim in claims
        ),
        inventory_diagnostics=(),
        inventory_available=False,
        inventory_count=None,
        fallback_reason=reason,
        error=error,
    )


def discover_live_actor_candidates(
    claims: Sequence[PromptClaim],
    actors: Mapping[str, Any] | Sequence[Any] | Iterable[Any],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    max_targets_per_claim: int | None = None,
) -> LiveActorDiscoveryResult:
    """Rank and geometrize actors from a generic live/fake inventory.

    This non-session adapter mirrors :func:`discover_live_actor_candidates_in_session`
    for replay tools and tests.  Real SPEAR callers should use the session API
    so requests are issued in global-sync frames.
    """

    claim_values, threshold = _validate_live_discovery_inputs(
        claims,
        top_k=top_k,
        min_score=min_score,
        max_targets_per_claim=max_targets_per_claim,
    )
    try:
        records = _actor_records(actors, strict_inventory_keys=True)
    except Exception as exc:
        return _live_discovery_failure(
            claim_values,
            reason="live_actor_inventory_invalid",
            error=f"{type(exc).__name__}: {exc}",
        )
    identities, inventory_diagnostics = _live_actor_identities(records)
    ranked_by_claim = _rank_live_claims(
        claim_values,
        identities,
        top_k=top_k,
        min_score=threshold,
    )
    requested = _targeted_live_candidates(
        claim_values,
        ranked_by_claim,
        max_targets_per_claim=max_targets_per_claim,
    )
    geometry_by_actor: dict[int, _LiveGeometry | Exception] = {}
    for candidate in requested:
        try:
            geometry_by_actor[id(candidate.actor)] = _read_live_geometry(candidate.actor)
        except Exception as exc:
            geometry_by_actor[id(candidate.actor)] = exc
    return _finalize_live_actor_discovery(
        claim_values,
        ranked_by_claim,
        geometry_by_actor,
        inventory_diagnostics=inventory_diagnostics,
        inventory_count=len(records),
        max_targets_per_claim=max_targets_per_claim,
    )


def discover_live_actor_candidates_in_session(
    claims: Sequence[PromptClaim],
    session: Any,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    max_targets_per_claim: int | None = None,
) -> LiveActorDiscoveryResult:
    """Discover targets directly from the current render-world inventory.

    Frame one inventories actors and batches ``GetActorLabel`` for every actor.
    Labels are completed only after that frame ends, then claims are ranked.
    Frame two batches location and bounds for the union of selected actors;
    those futures are likewise completed only after ``end_frame``.  Candidates
    retain their original actor wrappers throughout, so no label-based
    existence lookup is performed.
    """

    claim_values, threshold = _validate_live_discovery_inputs(
        claims,
        top_k=top_k,
        min_score=min_score,
        max_targets_per_claim=max_targets_per_claim,
    )
    ensure_world = getattr(session, "ensure_unreal_world", None)
    if callable(ensure_world):
        try:
            ensure_world()
        except Exception as exc:
            return _live_discovery_failure(
                claim_values,
                reason="live_actor_inventory_error",
                error=f"{type(exc).__name__}: {exc}",
            )
    run_frame = getattr(session, "_run_frame", None)
    game = getattr(session, "game", None)
    unreal_service = getattr(game, "unreal_service", None)
    find_actors = getattr(unreal_service, "find_actors_as_dict", None)
    if not callable(run_frame) or not callable(find_actors):
        return _live_discovery_failure(
            claim_values,
            reason="live_actor_inventory_unavailable",
            error="session.game.unreal_service.find_actors_as_dict is unavailable",
        )

    def inventory_body() -> tuple[tuple[_ActorRecord, ...], dict[int, Any | Exception]]:
        actors = find_actors(
            include_unreal_name=True,
            as_unreal_object=True,
        )
        if actors is None:
            actors = {}
        # Do not call actor methods while parsing the mapping.  Every reflected
        # label request below is explicit and remains inside this frame.
        records = _actor_records(
            actors,
            inspect_actor_text=False,
            strict_inventory_keys=True,
        )
        pending_labels: dict[int, Any | Exception] = {}
        for record in records:
            actor_key = id(record.actor)
            if actor_key in pending_labels:
                continue
            try:
                pending_labels[actor_key] = _request_live_actor_label(
                    record.actor,
                    prefer_async=True,
                )
            except Exception as exc:
                pending_labels[actor_key] = exc
        return records, pending_labels

    try:
        records, pending_labels = run_frame(inventory_body)
    except Exception as exc:
        return _live_discovery_failure(
            claim_values,
            reason="live_actor_inventory_error",
            error=f"{type(exc).__name__}: {exc}",
        )

    records, label_errors = _complete_live_actor_labels(records, pending_labels)
    identities, inventory_diagnostics = _live_actor_identities(
        records,
        label_errors=label_errors,
    )
    ranked_by_claim = _rank_live_claims(
        claim_values,
        identities,
        top_k=top_k,
        min_score=threshold,
    )
    requested = _targeted_live_candidates(
        claim_values,
        ranked_by_claim,
        max_targets_per_claim=max_targets_per_claim,
    )
    pending_geometry: dict[int, tuple[Any, Any] | Exception] = {}
    geometry_frame_error: str | None = None
    if requested:

        def geometry_body() -> dict[int, tuple[Any, Any] | Exception]:
            pending: dict[int, tuple[Any, Any] | Exception] = {}
            for candidate in requested:
                actor_key = id(candidate.actor)
                if actor_key in pending:
                    continue
                try:
                    pending[actor_key] = _request_live_geometry(
                        candidate.actor,
                        prefer_async=True,
                    )
                except Exception as exc:
                    pending[actor_key] = exc
            return pending

        try:
            pending_geometry = run_frame(geometry_body)
        except Exception as exc:
            geometry_frame_error = f"{type(exc).__name__}: {exc}"
            pending_geometry = {id(candidate.actor): exc for candidate in requested}

    geometry_by_actor: dict[int, _LiveGeometry | Exception] = {}
    for candidate in requested:
        actor_key = id(candidate.actor)
        value = pending_geometry.get(actor_key)
        if value is None:
            geometry_by_actor[actor_key] = RuntimeError(
                "live actor geometry was not requested"
            )
        elif isinstance(value, Exception):
            geometry_by_actor[actor_key] = value
        else:
            try:
                # Async futures are completed only after geometry_body's
                # global-sync frame returned and end_frame ran.
                geometry_by_actor[actor_key] = _complete_live_geometry(value)
            except Exception as exc:
                geometry_by_actor[actor_key] = exc

    return _finalize_live_actor_discovery(
        claim_values,
        ranked_by_claim,
        geometry_by_actor,
        inventory_diagnostics=inventory_diagnostics,
        inventory_count=len(records),
        max_targets_per_claim=max_targets_per_claim,
        result_error=geometry_frame_error,
    )


def _look_at_pose(
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    bounds: SceneBounds,
    *,
    expansion_fraction: float = 0.1,
) -> CameraPose:
    provisional = clamp_camera_pose(
        CameraPose(location[0], location[1], location[2], 0.0, 0.0),
        bounds,
        expansion_fraction=expansion_fraction,
    )
    dx = target[0] - provisional.x
    dy = target[1] - provisional.y
    dz = target[2] - provisional.z
    horizontal = math.hypot(dx, dy)
    yaw = math.degrees(math.atan2(dy, dx)) if horizontal > 1e-6 else 0.0
    pitch = math.degrees(math.atan2(dz, horizontal))
    return clamp_camera_pose(
        CameraPose(provisional.x, provisional.y, provisional.z, pitch, yaw),
        bounds,
        expansion_fraction=expansion_fraction,
    )


@dataclass(frozen=True, slots=True)
class _AabbProjection:
    """Screen-space footprint of all eight corners of a live actor AABB.

    Width and height are fractions of the full viewport dimensions, so a
    value of ``1.0`` exactly spans that dimension.  NDC overflow is kept
    separately because a large, cropped target must not look like a good
    close-up merely because its projected footprint is near the desired size.
    """

    width_fraction: float
    height_fraction: float
    center_offset: float
    edge_overflow: float

    @property
    def longest_fraction(self) -> float:
        return max(self.width_fraction, self.height_fraction)


@dataclass(frozen=True, slots=True)
class _CandidatePoseOption:
    pose: CameraPose
    projection: _AabbProjection
    azimuth_degrees: float
    elevation_degrees: float
    distance_cm: float


@dataclass(frozen=True, slots=True)
class BoundsCameraPoseAssessment:
    """Geometry-only target-framing health for one planned camera pose."""

    target_in_front: bool
    longest_viewport_fraction: float | None
    center_offset: float | None
    edge_overflow: float | None

    @property
    def healthy(self) -> bool:
        return (
            self.target_in_front
            and self.longest_viewport_fraction is not None
            and 0.05 <= self.longest_viewport_fraction <= 0.95
            and self.center_offset is not None
            and self.center_offset <= 0.55
            and self.edge_overflow is not None
            and self.edge_overflow <= 0.02
        )


def _aabb_corners(
    center: tuple[float, float, float],
    extent: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            center[0] + sign_x * extent[0],
            center[1] + sign_y * extent[1],
            center[2] + sign_z * extent[2],
        )
        for sign_x in (-1.0, 1.0)
        for sign_y in (-1.0, 1.0)
        for sign_z in (-1.0, 1.0)
    )


def _project_aabb_corners(
    corners: Sequence[tuple[float, float, float]],
    pose: CameraPose,
    *,
    horizontal_fov_radians: float,
    vertical_fov_radians: float,
) -> _AabbProjection | None:
    """Project AABB corners through an Unreal-style, roll-free camera."""

    pitch = math.radians(pose.pitch)
    yaw = math.radians(pose.yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    forward = (
        cos_pitch * cos_yaw,
        cos_pitch * sin_yaw,
        sin_pitch,
    )
    right = (-sin_yaw, cos_yaw, 0.0)
    up = (-sin_pitch * cos_yaw, -sin_pitch * sin_yaw, cos_pitch)
    tan_horizontal = math.tan(horizontal_fov_radians / 2.0)
    tan_vertical = math.tan(vertical_fov_radians / 2.0)
    projected_x: list[float] = []
    projected_y: list[float] = []
    for corner in corners:
        delta = (
            corner[0] - pose.x,
            corner[1] - pose.y,
            corner[2] - pose.z,
        )
        depth = sum(
            component * axis
            for component, axis in zip(delta, forward, strict=True)
        )
        # A pose intersecting the AABB cannot provide a valid framing estimate.
        if depth <= 1e-4:
            return None
        camera_x = sum(
            component * axis
            for component, axis in zip(delta, right, strict=True)
        )
        camera_y = sum(
            component * axis
            for component, axis in zip(delta, up, strict=True)
        )
        projected_x.append(camera_x / (depth * tan_horizontal))
        projected_y.append(camera_y / (depth * tan_vertical))

    min_x, max_x = min(projected_x), max(projected_x)
    min_y, max_y = min(projected_y), max(projected_y)
    max_abs_x = max(abs(min_x), abs(max_x))
    max_abs_y = max(abs(min_y), abs(max_y))
    return _AabbProjection(
        # NDC spans [-1, 1], hence division by two converts to a viewport
        # dimension fraction.
        width_fraction=(max_x - min_x) / 2.0,
        height_fraction=(max_y - min_y) / 2.0,
        center_offset=math.hypot((min_x + max_x) / 2.0, (min_y + max_y) / 2.0),
        edge_overflow=max(0.0, max_abs_x - 0.98)
        + max(0.0, max_abs_y - 0.98),
    )


def _angle_separation_degrees(left: float, right: float) -> float:
    return abs(((left - right + 180.0) % 360.0) - 180.0)


def _candidate_pose_score(
    option: _CandidatePoseOption,
    *,
    desired_fill: float,
    acceptable_fill: tuple[float, float],
    preferred_elevation: float,
    base_azimuth: float,
) -> float:
    fill = max(option.projection.longest_fraction, 1e-6)
    lower, upper = acceptable_fill
    outside_band = max(lower - fill, 0.0, fill - upper)
    return (
        abs(math.log(fill / desired_fill))
        + 3.0 * outside_band
        + 12.0 * option.projection.edge_overflow
        + 0.4 * option.projection.center_offset
        + 0.035
        * (_angle_separation_degrees(option.azimuth_degrees, base_azimuth) / 180.0)
        + 0.025 * (abs(option.elevation_degrees - preferred_elevation) / 24.0)
    )


def assess_bounds_camera_pose(
    bounds_center_cm: Sequence[float],
    extent_cm: Sequence[float],
    pose: CameraPose,
    fov_degrees: float = 90.0,
) -> BoundsCameraPoseAssessment:
    """Check that a target AABB is in front, centered, and usefully sized.

    This deterministic frustum/occupancy gate does not claim to detect world
    occlusion.  Alternate views and an UNKNOWN visual verdict handle that case.
    """

    if not isinstance(pose, CameraPose):
        raise TypeError("pose must be a CameraPose")
    fov = _finite_float(fov_degrees, name="fov_degrees")
    if not 10.0 <= fov <= 160.0:
        raise ValueError("fov_degrees must be between 10 and 160")
    center = _vector3(bounds_center_cm, name="bounds_center_cm")
    extent = tuple(max(1.0, value) for value in _vector3(extent_cm, name="extent_cm"))
    half_horizontal = math.radians(fov / 2.0)
    vertical_fov = 2.0 * math.atan(math.tan(half_horizontal) / (16.0 / 9.0))
    projection = _project_aabb_corners(
        _aabb_corners(center, extent),
        pose,
        horizontal_fov_radians=2.0 * half_horizontal,
        vertical_fov_radians=vertical_fov,
    )
    if projection is None:
        return BoundsCameraPoseAssessment(False, None, None, None)
    return BoundsCameraPoseAssessment(
        True,
        projection.longest_fraction,
        projection.center_offset,
        projection.edge_overflow,
    )


def plan_bounds_camera_poses(
    bounds_center_cm: Sequence[float],
    extent_cm: Sequence[float],
    scene_bounds: SceneBounds,
    fov_degrees: float = 90.0,
    *,
    camera_expansion_fraction: float = 0.1,
    minimum_context_elevation_degrees: float = 0.0,
) -> tuple[CameraPose, ...]:
    """Plan context, close, and alternate views from geometry alone.

    The planner samples twelve azimuths and three elevations.  At each view it
    projects all eight corners of the actor's live AABB using the real 16:9
    horizontal/vertical FOV, rather than estimating framing from a sphere.
    Importantly, projection and scoring happen *after* the evaluator's standard
    camera-envelope clamp, so a target near an edge is framed using the pose UE
    will actually receive.  The three returned poses preserve the historical
    ``(context, close, alternate)`` API and favor different azimuths.
    """

    fov = _finite_float(fov_degrees, name="fov_degrees")
    if not 10.0 <= fov <= 160.0:
        raise ValueError("fov_degrees must be between 10 and 160")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")
    expansion = _finite_float(
        camera_expansion_fraction,
        name="camera_expansion_fraction",
    )
    if expansion < 0.0:
        raise ValueError("camera_expansion_fraction must be non-negative")
    minimum_context_elevation = _finite_float(
        minimum_context_elevation_degrees,
        name="minimum_context_elevation_degrees",
    )
    if not 0.0 <= minimum_context_elevation <= 24.0:
        raise ValueError(
            "minimum_context_elevation_degrees must be between 0 and 24"
        )

    center = _vector3(bounds_center_cm, name="bounds_center_cm")
    extent = tuple(max(1.0, value) for value in _vector3(extent_cm, name="extent_cm"))
    corners = _aabb_corners(center, extent)
    half_horizontal = math.radians(fov / 2.0)
    vertical_fov = 2.0 * math.atan(math.tan(half_horizontal) / (16.0 / 9.0))

    scene_center = scene_bounds.center_cm
    outward_x = center[0] - scene_center[0]
    outward_y = center[1] - scene_center[1]
    if math.hypot(outward_x, outward_y) < 1e-6:
        base_azimuth = 225.0
    else:
        base_azimuth = math.degrees(math.atan2(outward_y, outward_x)) % 360.0

    radius = math.sqrt(sum(component * component for component in extent))
    tan_vertical = max(1e-6, math.tan(vertical_fov / 2.0))
    azimuth_offsets = (
        0.0,
        30.0,
        -30.0,
        60.0,
        -60.0,
        90.0,
        -90.0,
        120.0,
        -120.0,
        150.0,
        -150.0,
        180.0,
    )
    elevation_samples = (0.0, 12.0, 24.0)
    options_by_pose: dict[tuple[float, ...], _CandidatePoseOption] = {}
    for azimuth_offset in azimuth_offsets:
        azimuth_degrees = (base_azimuth + azimuth_offset) % 360.0
        azimuth = math.radians(azimuth_degrees)
        for elevation_degrees in elevation_samples:
            elevation = math.radians(elevation_degrees)
            cos_elevation = math.cos(elevation)
            direction = (
                math.cos(azimuth) * cos_elevation,
                math.sin(azimuth) * cos_elevation,
                math.sin(elevation),
            )
            # Start just outside the AABB face along this exact view vector.
            # This remains scale-relative, avoiding the old 180 cm floor that
            # made candle-sized targets occupy only a few pixels.
            support_distance = sum(
                abs(axis) * component
                for axis, component in zip(direction, extent, strict=True)
            )
            near_horizontal = max(1.0, support_distance * 1.03 * cos_elevation)
            far_horizontal = max(
                near_horizontal * 1.1,
                radius * max(40.0, 6.0 / tan_vertical),
            )
            log_near = math.log(near_horizontal)
            log_span = math.log(far_horizontal) - log_near
            for distance_index in range(73):
                horizontal_distance = math.exp(
                    log_near + log_span * distance_index / 72.0
                )
                raw_location = (
                    center[0] + math.cos(azimuth) * horizontal_distance,
                    center[1] + math.sin(azimuth) * horizontal_distance,
                    center[2] + math.tan(elevation) * horizontal_distance,
                )
                # _look_at_pose first clamps the location and then recomputes
                # look-at rotation.  Projection below therefore evaluates the
                # exact post-clamp transform, not the requested raw transform.
                pose = _look_at_pose(
                    raw_location,
                    center,
                    scene_bounds,
                    expansion_fraction=expansion,
                )
                projection = _project_aabb_corners(
                    corners,
                    pose,
                    horizontal_fov_radians=2.0 * half_horizontal,
                    vertical_fov_radians=vertical_fov,
                )
                if projection is None:
                    continue
                dx = pose.x - center[0]
                dy = pose.y - center[1]
                dz = pose.z - center[2]
                horizontal = math.hypot(dx, dy)
                if horizontal <= 1e-6:
                    continue
                actual_azimuth = math.degrees(math.atan2(dy, dx)) % 360.0
                actual_elevation = math.degrees(math.atan2(dz, horizontal))
                option = _CandidatePoseOption(
                    pose=pose,
                    projection=projection,
                    azimuth_degrees=actual_azimuth,
                    elevation_degrees=actual_elevation,
                    distance_cm=math.sqrt(dx * dx + dy * dy + dz * dz),
                )
                key = tuple(
                    round(value, 4)
                    for value in (
                        pose.x,
                        pose.y,
                        pose.z,
                        pose.pitch,
                        pose.yaw,
                    )
                )
                options_by_pose[key] = option

    options = tuple(options_by_pose.values())
    if not options:
        # The only way every sampled projection can be invalid with validated
        # inputs is for the permitted camera envelope to be contained by (or
        # otherwise unable to see) an exceptionally large/off-bounds AABB.
        # There is no honest framing solution in that case.  Return three
        # distinct best-effort envelope views instead of placing all three
        # cameras at the actor center, where they would be identical and
        # guaranteed to sit inside its bounds.
        limits = scene_bounds.expanded(expansion)
        fallback_z = min(max(center[2], limits.min_z), limits.max_z)
        fallback_locations = (
            (limits.min_x, limits.min_y, fallback_z),
            (limits.max_x, limits.min_y, fallback_z),
            (limits.max_x, limits.max_y, fallback_z),
        )
        return tuple(
            _look_at_pose(
                location,
                center,
                scene_bounds,
                expansion_fraction=expansion,
            )
            for location in fallback_locations
        )

    def rank_options(
        desired_fill: float,
        acceptable_fill: tuple[float, float],
        preferred_elevation: float,
    ) -> list[_CandidatePoseOption]:
        return sorted(
            options,
            key=lambda option: _candidate_pose_score(
                option,
                desired_fill=desired_fill,
                acceptable_fill=acceptable_fill,
                preferred_elevation=preferred_elevation,
                base_azimuth=base_azimuth,
            ),
        )

    def is_acceptably_framed(
        option: _CandidatePoseOption,
        acceptable_fill: tuple[float, float],
    ) -> bool:
        lower, upper = acceptable_fill
        return (
            lower <= option.projection.longest_fraction <= upper
            and option.projection.edge_overflow <= 1e-6
        )

    def diversity_adjusted_score(
        option: _CandidatePoseOption,
        *,
        desired_fill: float,
        acceptable_fill: tuple[float, float],
        preferred_elevation: float,
        references: Sequence[tuple[float, float]],
    ) -> float:
        # Diversity is a modest tie-breaker when the requested angle cannot be
        # achieved with sound framing.  Projection quality remains dominant,
        # preventing a severely cropped pose from winning solely by azimuth.
        deficit = sum(
            max(
                0.0,
                minimum_separation
                - _angle_separation_degrees(
                    option.azimuth_degrees, reference_azimuth
                ),
            )
            / minimum_separation
            for reference_azimuth, minimum_separation in references
        )
        return _candidate_pose_score(
            option,
            desired_fill=desired_fill,
            acceptable_fill=acceptable_fill,
            preferred_elevation=preferred_elevation,
            base_azimuth=base_azimuth,
        ) + 0.18 * deficit

    context_spec = (0.30, (0.20, 0.40), 12.0)
    # A camera on the Actor centerline often sees only the underside of chairs,
    # tables, and other supported props.  Keep the same tight occupancy, but
    # prefer a visibly elevated close view so shape and orientation remain
    # judgeable.  The sampled 12/24-degree elevations still let the frustum and
    # scene-envelope constraints choose the feasible member.
    close_spec = (0.60, (0.45, 0.75), 18.0)
    alternate_spec = (0.48, (0.35, 0.65), 24.0)
    context_ranked = rank_options(*context_spec)
    context_acceptable = tuple(
        option
        for option in context_ranked
        if is_acceptably_framed(option, context_spec[1])
    )
    context_elevated = tuple(
        option
        for option in context_acceptable
        if option.elevation_degrees >= minimum_context_elevation - 1e-6
    )
    context_pool = (
        context_elevated
        if minimum_context_elevation > 0.0
        else context_acceptable
    )
    context = (context_pool or context_acceptable or tuple(context_ranked))[0]

    close_ranked = rank_options(*close_spec)
    close_acceptable = tuple(
        option
        for option in close_ranked
        if is_acceptably_framed(option, close_spec[1])
    )
    close_elevated = tuple(
        option
        for option in close_acceptable
        if option.elevation_degrees >= 12.0 - 1e-6
    )
    close_framed = close_elevated or close_acceptable
    close_diverse = tuple(
        option
        for option in close_framed
        if _angle_separation_degrees(
            option.azimuth_degrees, context.azimuth_degrees
        )
        >= 30.0 - 1e-6
    )
    close_preferred = tuple(
        option
        for option in close_diverse
        if option.projection.longest_fraction
        > context.projection.longest_fraction + 0.05
        and option.distance_cm < context.distance_cm
    )
    if close_preferred or close_diverse:
        # Angular separation is forced only within the zero-overflow,
        # role-appropriate framing set.
        close = (close_preferred or close_diverse)[0]
    else:
        close_pool = close_framed or tuple(close_ranked)
        unused_close = tuple(
            option for option in close_pool if option.pose != context.pose
        )
        close = min(
            unused_close or close_pool,
            key=lambda option: diversity_adjusted_score(
                option,
                desired_fill=close_spec[0],
                acceptable_fill=close_spec[1],
                preferred_elevation=close_spec[2],
                references=((context.azimuth_degrees, 30.0),),
            ),
        )

    alternate_ranked = rank_options(*alternate_spec)
    alternate_acceptable = tuple(
        option
        for option in alternate_ranked
        if is_acceptably_framed(option, alternate_spec[1])
    )
    alternate_elevated = tuple(
        option
        for option in alternate_acceptable
        if option.elevation_degrees >= minimum_context_elevation - 1e-6
    )
    alternate_framed = (
        alternate_elevated
        if minimum_context_elevation > 0.0
        else alternate_acceptable
    ) or alternate_acceptable
    alternate_diverse = tuple(
        option
        for option in alternate_framed
        if _angle_separation_degrees(
            option.azimuth_degrees, context.azimuth_degrees
        )
        >= 50.0
        and _angle_separation_degrees(
            option.azimuth_degrees, close.azimuth_degrees
        )
        >= 50.0
    )
    if alternate_diverse:
        alternate = alternate_diverse[0]
    else:
        alternate_pool = alternate_framed or tuple(alternate_ranked)
        unused_alternate = tuple(
            option
            for option in alternate_pool
            if option.pose not in {context.pose, close.pose}
        )
        alternate = min(
            unused_alternate or alternate_pool,
            key=lambda option: diversity_adjusted_score(
                option,
                desired_fill=alternate_spec[0],
                acceptable_fill=alternate_spec[1],
                preferred_elevation=alternate_spec[2],
                references=(
                    (context.azimuth_degrees, 50.0),
                    (close.azimuth_degrees, 50.0),
                ),
            ),
        )
    return (context.pose, close.pose, alternate.pose)


def plan_bounds_camera_recovery_poses(
    bounds_center_cm: Sequence[float],
    extent_cm: Sequence[float],
    scene_bounds: SceneBounds,
    fov_degrees: float = 90.0,
    *,
    maximum_poses: int = 15,
    camera_expansion_fraction: float = 0.1,
) -> tuple[CameraPose, ...]:
    """Generate diverse fallback poses for unresolved live camera traces.

    The normal planner returns three role-specific views. This recovery API
    deliberately explores more azimuth, distance, and elevation combinations.
    Broadside views are tried first for strongly elongated AABBs so thin ropes,
    signs, decals, rails, and similar sparse geometry are not inspected only
    end-on.
    """

    if (
        isinstance(maximum_poses, bool)
        or not isinstance(maximum_poses, int)
        or not 1 <= maximum_poses <= 24
    ):
        raise ValueError("maximum_poses must be between 1 and 24")
    center = _vector3(bounds_center_cm, name="bounds_center_cm")
    extent = tuple(max(1.0, value) for value in _vector3(extent_cm, name="extent_cm"))
    horizontal_ratio = max(extent[0], extent[1]) / max(
        1.0, min(extent[0], extent[1])
    )
    if horizontal_ratio >= 3.0 and extent[0] >= extent[1]:
        azimuths = (90.0, 270.0, 60.0, 120.0, 240.0, 300.0, 0.0, 180.0)
    elif horizontal_ratio >= 3.0:
        azimuths = (0.0, 180.0, 30.0, 150.0, 210.0, 330.0, 90.0, 270.0)
    else:
        azimuths = tuple(float(value) for value in range(0, 360, 45))
    elevations = (8.0, 20.0, 35.0)
    desired_fills = (0.42, 0.62, 0.28)
    half_horizontal = math.radians(
        _finite_float(fov_degrees, name="fov_degrees") / 2.0
    )
    vertical_fov = 2.0 * math.atan(
        math.tan(half_horizontal) / (16.0 / 9.0)
    )
    tan_vertical = max(1e-6, math.tan(vertical_fov / 2.0))
    radius = math.sqrt(sum(value * value for value in extent))
    result: list[CameraPose] = []
    seen: set[tuple[float, ...]] = set()
    for azimuth_degrees in azimuths:
        azimuth = math.radians(azimuth_degrees)
        for elevation_degrees, desired_fill in zip(
            elevations,
            desired_fills,
            strict=True,
        ):
            elevation = math.radians(elevation_degrees)
            direction = (
                math.cos(azimuth) * math.cos(elevation),
                math.sin(azimuth) * math.cos(elevation),
                math.sin(elevation),
            )
            support = sum(
                abs(axis) * component
                for axis, component in zip(direction, extent, strict=True)
            )
            horizontal_distance = max(
                support * 1.08,
                radius / (tan_vertical * desired_fill),
            )
            raw_location = (
                center[0] + math.cos(azimuth) * horizontal_distance,
                center[1] + math.sin(azimuth) * horizontal_distance,
                center[2] + math.tan(elevation) * horizontal_distance,
            )
            pose = _look_at_pose(
                raw_location,
                center,
                scene_bounds,
                expansion_fraction=camera_expansion_fraction,
            )
            assessment = assess_bounds_camera_pose(
                center,
                extent,
                pose,
                fov_degrees,
            )
            if (
                not assessment.healthy
                or assessment.edge_overflow is None
                or assessment.edge_overflow > 0.05
                or assessment.longest_viewport_fraction is None
                or not 0.1 <= assessment.longest_viewport_fraction <= 0.92
            ):
                continue
            key = tuple(
                round(value, 3)
                for value in (pose.x, pose.y, pose.z, pose.pitch, pose.yaw)
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(pose)
            if len(result) >= maximum_poses:
                return tuple(result)
    for pose in plan_bounds_camera_poses(
        center,
        extent,
        scene_bounds,
        fov_degrees,
        camera_expansion_fraction=camera_expansion_fraction,
    ):
        key = tuple(
            round(value, 3)
            for value in (pose.x, pose.y, pose.z, pose.pitch, pose.yaw)
        )
        if key not in seen:
            seen.add(key)
            result.append(pose)
            if len(result) >= maximum_poses:
                break
    return tuple(result)


def plan_candidate_camera_poses(
    live_target: LiveActorTarget,
    scene_bounds: SceneBounds,
    fov_degrees: float = 90.0,
    *,
    camera_expansion_fraction: float = 0.1,
) -> tuple[CameraPose, ...]:
    """Backward-compatible wrapper around the geometry-only AABB planner."""

    return plan_bounds_camera_poses(
        live_target.bounds_center_cm,
        live_target.extent_cm,
        scene_bounds,
        fov_degrees,
        camera_expansion_fraction=camera_expansion_fraction,
    )


# Readable aliases for callers which phrase the operation around assets.
match_asset_candidates = rank_asset_candidates
join_asset_candidates = join_live_actor_candidates
join_asset_candidates_from_session = join_candidates_in_session


__all__ = [
    "ActorJoinDiagnostic",
    "ActorJoinResult",
    "AssetCandidate",
    "BoundsCameraPoseAssessment",
    "DEFAULT_CLUSTER_DISTANCE_CM",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_TOP_K",
    "IrSceneError",
    "LiveActorCandidate",
    "LiveActorDiscoveryResult",
    "LiveActorInventoryDiagnostic",
    "LiveActorTarget",
    "LiveCandidateDiagnostic",
    "LiveClaimCandidates",
    "PlacedAsset",
    "asset_path_basename",
    "assess_bounds_camera_pose",
    "claim_supports_asset_discovery",
    "claim_supports_asset_search",
    "claim_supports_candidate_preverification",
    "cluster_asset_candidates",
    "discover_live_actor_candidates",
    "discover_live_actor_candidates_in_session",
    "expand_claim_terms",
    "find_asset_candidates",
    "join_asset_candidates",
    "join_asset_candidates_from_session",
    "join_candidates_in_session",
    "join_live_actor_candidates",
    "load_ir_scene",
    "match_asset_candidates",
    "normalize_asset_text",
    "parse_placed_assets",
    "plan_bounds_camera_recovery_poses",
    "plan_bounds_camera_poses",
    "plan_candidate_camera_poses",
    "rank_asset_candidates",
]
