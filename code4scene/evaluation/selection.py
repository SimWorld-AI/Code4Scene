"""Which Actors a selector picks out of a Candidate scene.

Every metric that names a population resolves it here. Two atoms that
both select ``candidate_all`` must select exactly the same Actors, or they are
measuring two different scenes and their numbers cannot be read together.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .values import as_text


def normalize_asset_path(value: Any) -> str | None:
    text = as_text(value)
    if text is None:
        return None
    slash = text.find("/")
    if slash >= 0:
        text = text[slash:]
    text = re.sub(r"[\"']+$", "", text)
    last_slash = text.rfind("/")
    object_separator = text.find(".", last_slash)
    if object_separator > last_slash:
        text = text[:object_separator]
    return text.lower() or None


def normalize_category(value: Any) -> str | None:
    text = as_text(value)
    return re.sub(r"[\s_-]+", "_", text.lower()) if text else None


def normalize_class(value: Any) -> str | None:
    text = as_text(value)
    if text is None:
        return None
    text = text.lower()
    text = text[max(text.rfind("."), text.rfind("/")) + 1 :]
    return re.sub(r"_c$", "", text) or None


def _normalize_lower(value: Any) -> str | None:
    text = as_text(value)
    return text.lower() if text else None


def category_from_actor(actor: Mapping[str, Any]) -> Any:
    """What kind of thing this Actor is, from the fields, never from the tags.

    This used to fall back to scanning raw `actor_tags` for an
    `asset_category=` entry. On a Candidate that is metadata the AGENT wrote —
    `execute_python_script` can set any tag on any Actor — so the subject
    selection a rubric scores by was authorable by the subject. The tags the
    exporter legitimately reads are already lifted into the fields below by
    `ue/export_scene_snapshot.py`; nothing that came from an authored level is
    lost by refusing to read them a second time here.

    Where the category SHOULD come from is `asset_catalog`, keyed by the asset
    path, which the agent cannot assert — resolved once during evidence
    collection so every selector downstream reads the same answer.
    """
    for key in ("asset_category", "semantic_category", "category", "semantic_concept"):
        if actor.get(key):
            return actor[key]
    return None


def _matches_any(value: Any, allowed: Any, normalize: Any) -> bool:
    if allowed is None or allowed == []:
        return True
    if not isinstance(allowed, list):
        raise ValueError("selector values must be arrays")
    observed = normalize(value)
    return observed is not None and observed in {normalize(item) for item in allowed}


def _matches_selector(actor: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    checks = (
        (actor.get("asset_path"), selector.get("allowed_asset_paths"),
         normalize_asset_path),
        (category_from_actor(actor), selector.get("allowed_categories"),
         normalize_category),
        (actor.get("class"), selector.get("allowed_classes"), normalize_class),
        (actor.get("label"), selector.get("labels"), _normalize_lower),
        (actor.get("stable_actor_id"), selector.get("stable_actor_ids"), as_text),
        (actor.get("logical_object_id"), selector.get("logical_object_ids"), as_text),
        (actor.get("actor_role"), selector.get("actor_roles"), _normalize_lower),
        (actor.get("actor_origin"), selector.get("actor_origins"), _normalize_lower),
    )
    if not all(_matches_any(value, allowed, normalize) for value, allowed, normalize in checks):
        return False
    exclusions = (
        (actor.get("asset_path"), selector.get("excluded_asset_paths"),
         normalize_asset_path),
        (category_from_actor(actor), selector.get("excluded_categories"),
         normalize_category),
        (actor.get("class"), selector.get("excluded_classes"), normalize_class),
        (actor.get("label"), selector.get("excluded_labels"), _normalize_lower),
        (actor.get("stable_actor_id"), selector.get("excluded_stable_actor_ids"),
         as_text),
        (actor.get("logical_object_id"),
         selector.get("excluded_logical_object_ids"), as_text),
    )
    for value, denied, normalize in exclusions:
        if denied is None or denied == []:
            continue
        if not isinstance(denied, list):
            raise ValueError("selector exclusion values must be arrays")
        observed = normalize(value)
        if observed is not None and observed in {normalize(item) for item in denied}:
            return False
    required = selector.get("required_tags")
    if required is not None and not isinstance(required, list):
        raise ValueError("selector required_tags must be an array")
    if required:
        tags = {_normalize_lower(tag) for tag in actor.get("actor_tags") or []}
        return all(_normalize_lower(tag) in tags for tag in required)
    return True


def select_candidate_actors(
    actors: Sequence[Mapping[str, Any]], selector: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """The candidate_all selector semantics shared by collection and scoring."""
    if not isinstance(selector, Mapping):
        raise ValueError("actor selector must be an object")
    return [actor for actor in actors if _matches_selector(actor, selector)]


def actor_identifier(actor: Mapping[str, Any], index: int) -> str:
    return (
        as_text(actor.get("stable_actor_id"))
        or as_text(actor.get("actor_path"))
        or as_text(actor.get("label"))
        or f"candidate_actor_{index}"
    )

__all__ = ["actor_identifier", "category_from_actor", "normalize_asset_path",
           "normalize_category", "normalize_class", "select_candidate_actors"]
