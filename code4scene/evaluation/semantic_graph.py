"""The scene as OBJECTS, not as Actors, and how confident that reading is.

A prompt says "one dining table and four chairs". A level says five hundred
StaticMeshActors. The table is a top and four legs; the chairs might each be
one Actor or three. Counting Actors and calling the answer an object count is
how a correct scene fails a count rule, so every deterministic rule is evaluated over
LOGICAL OBJECTS: Actors sharing a `logical_object_id` are one object, and an
Actor without one is an object by itself.

The second half of this module is about how much that reading can be trusted.
A deterministic rule that selects on semantic category is only as good as the
categories the exporter wrote, and a level where half the Actors carry none
can produce a count of two that means "two, or possibly nine". So a selection
carries its own COVERAGE, and a rule whose evidence is incomplete reports
`not_evaluated` — except where the incompleteness cannot change the verdict,
which is the useful case: if a rule needs at least one chair and four are
already found, more unclassified Actors cannot make that false.

**Not ported** from the vendored evaluator: its ontology (a hand-maintained
synonym and hypernym table mapping prompt words to asset categories) and its
architecture/functional-region resolvers. Selection here is by the same
normalised asset path, category and class matching the rest of the scoring
layer uses, so a rule names what a case spec would name. Rules that needed
the ontology to resolve a word are `not_evaluated` rather than guessed at.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .scene_geometry import aabb, center_cm, extent_cm
from .selection import category_from_actor, select_candidate_actors
from .values import as_text


@dataclass(frozen=True)
class LogicalObject:
    """One thing in the scene, made of one or more Actors."""

    id: str
    actors: tuple[Mapping[str, Any], ...]

    @property
    def label(self) -> str | None:
        return as_text(self.actors[0].get("label")) if self.actors else None

    @property
    def categorised(self) -> bool:
        """Does at least one member say what kind of thing this is."""
        return any(category_from_actor(actor) for actor in self.actors)

    def bounds(self) -> dict[str, list[float]]:
        """The union AABB, so a compound object is measured as one thing."""
        boxes = [aabb(actor) for actor in self.actors]
        minimum = [min(box[0][axis] for box in boxes) for axis in range(3)]
        maximum = [max(box[1][axis] for box in boxes) for axis in range(3)]
        return {"min": minimum, "max": maximum,
                "origin_cm": [(minimum[i] + maximum[i]) / 2 for i in range(3)],
                "extent_cm": [(maximum[i] - minimum[i]) / 2 for i in range(3)]}

    def as_actor(self) -> dict[str, Any]:
        """The object in an Actor's shape, so geometry code can read it.

        A compound object relates to other objects as a whole — the chair is
        near the table, not the chair's left rear leg — and every relation in
        `spatial_relations` takes an Actor. This is that adapter, and it is
        marked so a report can say the row was about an object.
        """
        bounds = self.bounds()
        first = self.actors[0] if self.actors else {}
        return {**first, "label": self.label, "bounds":
                {"origin_cm": bounds["origin_cm"], "extent_cm": bounds["extent_cm"]},
                "transform": {"location_cm": bounds["origin_cm"],
                              "rotation_deg": (first.get("transform") or {}).get(
                                  "rotation_deg") or [0.0, 0.0, 0.0],
                              "scale": (first.get("transform") or {}).get("scale")
                              or [1.0, 1.0, 1.0]},
                "evaluator_object_id": self.id, "is_logical_object": True,
                "raw_actor_count": len(self.actors)}

    def value_at(self, field_name: str) -> dict[str, Any]:
        """One attribute of the object, or why it does not have exactly one.

        Members can disagree — a two-tone sofa has two material sets — and a
        rule about "the sofa's colour" over an object with two answers has no
        answer. Reported as ambiguous rather than resolved to the first.
        """
        values, present = [], False
        for actor in self.actors:
            source = actor
            if field_name not in actor and isinstance(actor.get("properties"), Mapping):
                source = actor["properties"]
            if field_name in source:
                present = True
                value = source[field_name]
                if value not in values:
                    values.append(value)
        if not present:
            return {"available": False, "ambiguous": False, "value": None,
                    "values": []}
        return {"available": True, "ambiguous": len(values) > 1,
                "value": values[0] if len(values) == 1 else None,
                "values": values}


@dataclass(frozen=True)
class Selection:
    """The objects a deterministic rule selected, and how sure that is."""

    objects: tuple[LogicalObject, ...]
    selector: Mapping[str, Any]
    #: Objects the selector could not decide about because they carry no
    #: category. They are neither in nor out; they are why a count may be
    #: inconclusive.
    unclassified: int = 0
    #: Actors that named a logical object nothing else in the scene named. A
    #: compound object exported with only some of its parts is a grouping this
    #: reading cannot trust.
    grouping_complete: bool = True
    fields: tuple[str, ...] = ()

    @property
    def category_complete(self) -> bool:
        return self.unclassified == 0

    @property
    def complete(self) -> bool:
        return self.category_complete and self.grouping_complete

    def coverage(self) -> dict[str, Any]:
        return {"category_complete": self.category_complete,
                "grouping_complete": self.grouping_complete,
                "unclassified_object_count": self.unclassified,
                "selected_object_count": len(self.objects),
                "selector_fields": list(self.fields)}


@dataclass
class SemanticGraph:
    """A scene read as objects."""

    objects: tuple[LogicalObject, ...] = ()
    actors: tuple[Mapping[str, Any], ...] = ()
    partial_groups: tuple[str, ...] = ()
    scoped_graphs: Mapping[str, SemanticGraph] = field(default_factory=dict)

    def select(self, subject: Any) -> Selection:
        """The objects a deterministic rule subject names.

        A subject is the same shape as a case spec's actor selector —
        `allowed_categories`, `allowed_asset_paths`, `allowed_classes`,
        `labels` — because a benchmark with two selector vocabularies is a
        benchmark where the same words mean two things.
        """
        selector = subject if isinstance(subject, Mapping) else {}
        scope = as_text(selector.get("scope"))
        if scope and scope in self.scoped_graphs:
            return self.scoped_graphs[scope].select(
                {key: value for key, value in selector.items() if key != "scope"}
            )
        fields = tuple(sorted(key for key, value in selector.items()
                              if value and key != "scope"))
        chosen, unclassified = [], 0
        for item in self.objects:
            members = select_candidate_actors(list(item.actors), selector)
            if members:
                chosen.append(item)
            elif selector.get("allowed_categories") and not item.categorised:
                # It might have matched; nothing recorded what it is.
                unclassified += 1
        return Selection(objects=tuple(chosen), selector=selector,
                         unclassified=unclassified,
                         grouping_complete=not self.partial_groups, fields=fields)


def build(
    actors: Sequence[Mapping[str, Any]],
    *,
    scopes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> SemanticGraph:
    """Group a scene's Actors into logical objects."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    loose: list[LogicalObject] = []
    for index, actor in enumerate(actors):
        identifier = as_text(actor.get("logical_object_id"))
        if identifier:
            groups.setdefault(identifier, []).append(actor)
        else:
            key = (as_text(actor.get("stable_actor_id"))
                   or as_text(actor.get("actor_path"))
                   or as_text(actor.get("label")) or f"actor_{index}")
            loose.append(LogicalObject(id=key, actors=(actor,)))
    # A logical object whose members declare no role is a group the exporter
    # only half described; the count is still one object, but a rule that
    # depends on the grouping being right is told the grouping is not certain.
    partial = tuple(identifier for identifier, members in groups.items()
                    if any(not as_text(actor.get("actor_role"))
                           for actor in members))
    grouped = [LogicalObject(id=identifier, actors=tuple(members))
               for identifier, members in groups.items()]
    scoped_graphs = {
        str(name): build(values)
        for name, values in (scopes or {}).items()
    }
    return SemanticGraph(objects=tuple(grouped + loose),
                         actors=tuple(actors), partial_groups=partial,
                         scoped_graphs=scoped_graphs)


def object_evidence(item: LogicalObject) -> dict[str, Any]:
    bounds = item.bounds()
    return {"evaluator_object_id": item.id, "label": item.label,
            "actor_count": len(item.actors),
            "category": category_from_actor(item.actors[0]) if item.actors else None,
            "location_cm": [round(value, 3) for value in bounds["origin_cm"]]}


def attribute_matches(value: Any, expected: Mapping[str, Any]) -> bool:
    """Does one observed attribute value satisfy a rule's expectation."""
    if "equals" in expected:
        wanted, observed = expected["equals"], value
        if isinstance(wanted, str) and isinstance(observed, str):
            return wanted.strip().lower() == observed.strip().lower()
        return wanted == observed
    if "one_of" in expected:
        options = expected.get("one_of") or []
        return any(attribute_matches(value, {"equals": option}) for option in options)
    minimum, maximum = expected.get("minimum"), expected.get("maximum")
    if minimum is None and maximum is None:
        return value is not None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return ((minimum is None or number >= float(minimum))
            and (maximum is None or number <= float(maximum)))


__all__ = ["LogicalObject", "Selection", "SemanticGraph", "attribute_matches",
           "build", "center_cm", "extent_cm", "object_evidence"]
