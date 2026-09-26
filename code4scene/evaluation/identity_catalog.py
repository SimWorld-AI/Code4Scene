"""Shared, verdict-free identity catalog for scene actors and objects.

Semantic verification and GT correspondence both need the same normalized
identity vocabulary. They must not, however, share a verdict: this module
only groups already-sanitized descriptors and exposes provenance-safe payloads
for locator backends.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .requirement_graph.actor_inventory import (
    ActorDescriptor,
    IdentitySource,
    IdentityTerm,
)
from .requirement_graph.contracts import SceneBounds

_SOURCE_PAYLOAD_KEYS = {
    IdentitySource.ASSET_PATH: "asset_terms",
    IdentitySource.ACTOR_CLASS: "actor_class_terms",
    IdentitySource.UNREAL_NAME: "unreal_name_terms",
    IdentitySource.STRUCTURED_TAG: "structured_tag_terms",
    IdentitySource.ACTOR_LABEL: "actor_label_terms",
}


@dataclass(frozen=True, slots=True)
class SceneIdentityGroup:
    """Opaque catalog row plus controller-only object/actor identifiers."""

    group_id: str
    actor_ids: tuple[str, ...]
    identity_terms: tuple[IdentityTerm, ...]
    locator_terms: tuple[IdentityTerm, ...]

    @property
    def primary_identity_term(self) -> str:
        values = self.identity_terms or self.locator_terms
        return values[0].term

    @property
    def identity_sources(self) -> tuple[IdentitySource, ...]:
        return tuple(
            dict.fromkeys(
                IdentitySource.coerce(value.source)
                for value in (*self.identity_terms, *self.locator_terms)
            )
        )

    @property
    def raw_values(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                value.raw_value
                for value in (*self.identity_terms, *self.locator_terms)
                if value.raw_value
            )
        )

    def llm_payload(self) -> dict[str, Any]:
        """Return identity provenance without leaking controller identifiers."""

        by_source: dict[IdentitySource, list[str]] = defaultdict(list)
        for term in self.identity_terms:
            source = IdentitySource.coerce(term.source)
            value = term.raw_value or term.term
            if value not in by_source[source]:
                by_source[source].append(value[:160])
        payload: dict[str, Any] = {
            "group_id": self.group_id,
            "actor_count": len(self.actor_ids),
        }
        for source, key in _SOURCE_PAYLOAD_KEYS.items():
            payload[key] = by_source.get(source, ())[:8]
        payload["component_locator_terms"] = [
            {
                "source": IdentitySource.coerce(term.source).value,
                "term": (term.raw_value or term.term)[:160],
            }
            for term in self.locator_terms[:8]
        ]
        return payload


def _term_signature(term: IdentityTerm, *, channel: str) -> tuple[str, str, str]:
    return channel, IdentitySource.coerce(term.source).value, term.term


def build_scene_identity_groups(
    actors: Iterable[ActorDescriptor],
    *,
    scene_bounds: SceneBounds,
) -> tuple[SceneIdentityGroup, ...]:
    """Group eligible descriptors by their complete normalized identity."""

    grouped: dict[
        tuple[tuple[str, str, str], ...],
        tuple[
            list[str],
            dict[tuple[str, str], IdentityTerm],
            dict[tuple[str, str], IdentityTerm],
        ],
    ] = {}
    for actor in actors:
        if not isinstance(actor, ActorDescriptor):
            raise TypeError("actors must contain ActorDescriptor values")
        if not actor.eligible_for_stage1(scene_bounds):
            continue
        signature = tuple(
            sorted(
                {
                    *(
                        _term_signature(term, channel="identity")
                        for term in actor.identity_terms
                    ),
                    *(
                        _term_signature(term, channel="locator")
                        for term in actor.locator_terms
                    ),
                }
            )
        )
        if not signature:
            continue
        actor_ids, identities, locators = grouped.setdefault(
            signature, ([], {}, {})
        )
        actor_ids.append(actor.live_actor_id)
        for term in actor.identity_terms:
            identities.setdefault(
                (IdentitySource.coerce(term.source).value, term.term), term
            )
        for term in actor.locator_terms:
            locators.setdefault(
                (IdentitySource.coerce(term.source).value, term.term), term
            )

    result: list[SceneIdentityGroup] = []
    for index, signature in enumerate(sorted(grouped), start=1):
        actor_ids, identities, locators = grouped[signature]
        identity_values = tuple(
            sorted(
                identities.values(),
                key=lambda value: (
                    -IdentitySource.coerce(value.source).priority,
                    value.term,
                ),
            )
        )
        locator_values = tuple(
            sorted(
                locators.values(),
                key=lambda value: (
                    -IdentitySource.coerce(value.source).priority,
                    value.term,
                ),
            )
        )
        result.append(
            SceneIdentityGroup(
                group_id=f"g{index:04d}",
                actor_ids=tuple(sorted(set(actor_ids))),
                identity_terms=identity_values,
                locator_terms=locator_values,
            )
        )
    return tuple(result)


__all__ = ["SceneIdentityGroup", "build_scene_identity_groups"]
