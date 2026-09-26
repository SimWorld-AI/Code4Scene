"""What kind of thing an Actor is, read off the asset rather than the Actor.

Every semantic verifier selects on `asset_category`, and until now that field
had one source: `simcodearena.*` Actor tags, lifted by
`ue/export_scene_snapshot.py`. Nothing in this repository writes those tags.
The studio MCP's `spawn_actor` and `spawn_blueprint_actor` take a name, a mesh
and a transform — there is no tag parameter — so on a task where the agent
BUILDS the scene the field is null for every Actor, always. Measured over the
generation grid: 0 of 6299.

That made the whole prompt-rubric family — the one part of the layer designed
for open-ended tasks, which have no answer key — silently unrunnable on
exactly the tasks it was written for. It worked on the scene-repair family
only because those levels are authored offline, with the tags baked in.

The deeper problem is not coverage. A category the CANDIDATE carries is
metadata the agent wrote, and the agent has `execute_python_script`: it can
set `asset_category=cathedral` on a grey cube, and `category_from_actor` used
to read raw Actor tags looking for exactly that key. A benchmark whose subject
selection can be authored by the subject is not measuring the subject.

So the category comes from the asset catalog, keyed by asset path. The path is
not something an agent can assert — it is what it actually spawned — and the
catalog is the same document `search_assets` already ranks against, so the
category a verifier scores by is the category the agent was offered.

**A catalog answers, or nothing does.** When one is configured, an Actor whose
asset it does not know has NO category, rather than the category its own
metadata claims. A cathedral built out of `/Engine/BasicShapes/Cube` gets
whatever the catalog says a cube is — `static_meshes`, in the palette shipped
with the studio server — and so does not satisfy a rubric asking for a
cathedral. Which is the answer: not "uncategorised, so assume it counts", and
not "the label says Cathedral_Central".

Accepted shapes, because the catalogs that exist are not one shape:

    {"buildings": {"items": ["/Game/.../BP_Building_01.BP_Building_01"]}}
    {"buildings": ["/Game/.../BP_Building_01.BP_Building_01"]}
    {"assets": [{"path": "/Game/...", "category": "buildings"}]}

Keys beginning with `_` are metadata (`assets_full.json` carries a `_meta`)
and never a category. Paths are compared through
`selection.normalize_asset_path`, so a Blueprint's class path
(`....BP_Tree1_C`) and the catalog's object path (`....BP_Tree1`) are the same
asset, which they are.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .selection import normalize_asset_path
from .values import as_text

#: Written onto each Candidate Actor beside its category, so a reader can tell
#: a scored category from an unanswered one without re-deriving it.
SOURCE_KEY = "asset_category_source"
FROM_CATALOG = "asset_catalog"


class AssetCatalogError(Exception):
    """The catalog document is not one this build can read."""


@dataclass(frozen=True)
class AssetCatalog:
    """A path -> category map, and the id that says which catalog answered."""

    id: str
    categories: Mapping[str, str]

    def __len__(self) -> int:
        return len(self.categories)

    def category_for(self, asset_path: Any) -> str | None:
        key = normalize_asset_path(asset_path)
        return self.categories.get(key) if key else None

    def apply(self, actors: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        """Resolve every Actor's category from the catalog, in place.

        Replaces rather than fills: an Actor that arrived carrying a category
        the catalog cannot confirm loses it. That is the point — see the
        module docstring — and it is why this returns the counts, so the
        evidence envelope can say how much of the scene the catalog could
        actually name.
        """
        resolved = unknown = 0
        for actor in actors:
            if not isinstance(actor, dict):
                continue
            category = self.category_for(actor.get("asset_path"))
            if category is None:
                for path in actor.get("component_asset_paths") or []:
                    category = self.category_for(path)
                    if category is not None:
                        break
            actor["asset_category"] = category
            actor[SOURCE_KEY] = FROM_CATALOG if category else None
            # `selection.category_from_actor` falls back through these keys
            # when `asset_category` is empty, and they are candidate metadata
            # the agent can write. Left in place they answer AFTER the catalog
            # refused to — the exact bypass the catalog exists to close.
            for fallback in ("semantic_category", "category", "semantic_concept"):
                actor.pop(fallback, None)
            if category:
                resolved += 1
            else:
                unknown += 1
        return {"categories_resolved": resolved, "categories_unknown": unknown}


def load(document: Any, identifier: str | None = None) -> AssetCatalog:
    """Read a catalog document into a path -> category map."""
    if not isinstance(document, Mapping):
        raise AssetCatalogError(
            "an asset catalog must be a JSON object mapping categories to the "
            "assets in them")
    categories: dict[str, str] = {}
    flat = document.get("assets")
    if isinstance(flat, list):
        for entry in flat:
            if not isinstance(entry, Mapping):
                continue
            key = normalize_asset_path(entry.get("path"))
            category = as_text(entry.get("category"))
            if key and category:
                categories[key] = category
    for name, value in document.items():
        if name.startswith("_") or name == "assets":
            continue
        items = value.get("items") if isinstance(value, Mapping) else value
        if not isinstance(items, list):
            continue
        for entry in items:
            path = entry.get("path") if isinstance(entry, Mapping) else entry
            key = normalize_asset_path(path)
            if key:
                categories[key] = str(name)
    if not categories:
        raise AssetCatalogError(
            "the asset catalog names no assets; scoring a scene against an "
            "empty catalog would report every object as an unknown kind")
    name = as_text(document.get("_meta", {}).get("id")
                   if isinstance(document.get("_meta"), Mapping) else None)
    return AssetCatalog(id=name or identifier or "asset-catalog",
                        categories=categories)


__all__ = ["FROM_CATALOG", "SOURCE_KEY", "AssetCatalog", "AssetCatalogError",
           "load"]
