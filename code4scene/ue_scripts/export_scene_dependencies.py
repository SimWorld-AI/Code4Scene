"""Editor-python: every package the saved level needs, and whether it is there.

A ``.umap`` stores REFERENCES. The material graph, the mesh, the texture are
separate packages, and the map records only their paths. So a level that
renders perfectly in the editor that built it can open to nothing in the next
one — and the run record, which holds the map and a screenshot, says the scene
was fine.

Three real ways that happens, all observed on the grid:

* the agent CREATED an asset and never saved it. It lives in the editor's
  memory for the rest of the episode, the screenshot is correct, and the
  package never reaches disk. Sixteen materials went that way in one run,
  including the one under the whole square.
* the two environments mounted DIFFERENT content. The path resolved when the
  scene was built and does not resolve when it is scored, and nothing in the
  record distinguishes that from an agent inventing a path.
* the pack itself is incomplete — a material instance that ships without the
  texture it samples. Present in both environments, missing in both.

The first is the agent's failure, the second is ours, and the third is the
content store's. They are indistinguishable from a low score, which is why
this probe records the dependency set rather than leaving it implied.

Written to ``SCENE_DEPENDENCIES_OUTPUT`` as JSON. ``SCENE_DEPENDENCIES_MAP``
names the package to walk; empty means the level currently open.
"""

import json
import os
import re

import unreal


#: How deep to follow the reference graph. A material instance's texture is at
#: depth 2 from the map, and a mesh's material's texture at depth 3; beyond
#: that the graph reaches engine content that is mounted everywhere and adds
#: thousands of rows nobody reads.
MAX_DEPTH = 4

#: Prefixes whose packages ship with the engine or the plugins and are present
#: in every environment by construction. Recording them would bury the ones
#: that actually vary.
SKIP_PREFIXES = ("/Engine/", "/Script/", "/Temp/", "/Paper2D/")


def _out_path():
    return globals().get("SCENE_DEPENDENCIES_OUTPUT") or ""


def _root_package():
    named = globals().get("SCENE_DEPENDENCIES_MAP") or ""
    if named:
        return str(named).replace("\\", "/").rsplit(".umap", 1)[0]
    world = unreal.EditorLevelLibrary.get_editor_world()
    if world is None:
        return ""
    return str(world.get_path_name()).split(".", 1)[0]


#: ``+PackageRedirects=(OldName="/Game/a",NewName="/Game/b")`` as the editor's
#: own config spells it.
_REDIRECT = re.compile(r'\+PackageRedirects=\(OldName="([^"]+)",\s*NewName="([^"]+)"')


def _redirects():
    """The package redirects THIS editor booted with.

    Read from the running instance's config rather than passed in, so the set
    reported here is by construction the set the loader is applying. A
    dependency probe that disagreed with the editor about which references
    resolve would be worse than no probe.

    The packs were flattened into ``Content/<Pack>/`` and kept their vendor's
    absolute paths, so a material asks for ``/Game/Textures/T_metal_BC`` while
    the file is at ``/Game/CityDowntown/Textures/T_metal_BC``. The redirect is
    what makes that load; without it every one of those reads as content the
    store is missing.
    """
    try:
        config = unreal.Paths.convert_relative_path_to_full(
            unreal.Paths.project_config_dir())
    except Exception:
        return {}
    found = {}
    for name in ("DefaultEngine.ini", "Engine.ini"):
        path = os.path.join(config, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as handle:
                for old, new in _REDIRECT.findall(handle.read()):
                    found.setdefault(old, new)
        except OSError:
            continue
    return found


def _on_disk(package):
    for suffix in (".umap", ".uasset"):
        try:
            found = unreal.Paths.convert_relative_path_to_full(
                unreal.PackageTools.package_name_to_filename(package, suffix))
        except Exception:
            continue
        if found and os.path.isfile(found):
            return found
    return ""


def _file_for(package, redirects=None):
    """The .uasset/.umap on disk for a package path, or '' if there is none.

    A redirected package resolves to its TARGET's file: the reference the
    registry recorded is stale, the loader follows the redirect, and reporting
    it unresolved would describe a failure that does not happen.
    """
    direct = _on_disk(package)
    if direct:
        return direct
    target = (redirects or {}).get(package)
    return _on_disk(target) if target else ""


def _dependencies(registry, package):
    options = unreal.AssetRegistryDependencyOptions(
        include_soft_package_references=True,
        include_hard_package_references=True,
        include_searchable_names=False,
        include_soft_management_references=False,
        include_hard_management_references=False)
    try:
        found = registry.get_dependencies(package, options)
    except Exception:
        return []
    return [str(value) for value in (found or [])]


def _walk(registry, root, redirects=None):
    """Breadth-first over the reference graph, recording each package once."""
    seen = {}
    frontier = [(root, 0)]
    while frontier:
        package, depth = frontier.pop(0)
        if package in seen or package.startswith(SKIP_PREFIXES):
            continue
        path = _file_for(package, redirects)
        entry = {"package": package, "depth": depth, "resolved": bool(path)}
        if not _on_disk(package) and path:
            entry["redirected_to"] = redirects[package]
        if path:
            try:
                entry["bytes"] = os.path.getsize(path)
            except OSError:
                entry["bytes"] = None
            # Not hashed here. Hashing hundreds of packages on the game thread
            # stalls the editor, and the harness has the paths and can hash
            # them off-thread — see the harness dependency collector.
            entry["file"] = path
        seen[package] = entry
        if depth < MAX_DEPTH:
            for child in _dependencies(registry, package):
                if child not in seen:
                    frontier.append((child, depth + 1))
    return seen


def main():
    out = _out_path()
    root = _root_package()
    # The instance root AS THE EDITOR SEES IT. Every `file` below is under
    # it, and the harness reading this manifest is on the other side of a
    # container boundary where that prefix does not exist — so it is reported
    # rather than left for the reader to guess, and the harness swaps it for
    # its own. Getting this wrong is silent: the paths look absolutely fine
    # and simply are not there.
    try:
        instance_root = unreal.Paths.convert_relative_path_to_full(
            unreal.Paths.project_content_dir()).rstrip("/")
    except Exception:
        instance_root = ""
    payload = {"root": root, "max_depth": MAX_DEPTH,
               "content_root": instance_root,
               "skipped_prefixes": list(SKIP_PREFIXES)}
    try:
        if not root:
            raise RuntimeError("no level is open and no map was named")
        registry = unreal.AssetRegistryHelpers.get_asset_registry()
        redirects = _redirects()
        payload["package_redirects"] = len(redirects)
        entries = _walk(registry, root, redirects)
        # `unresolved` is the answer this probe exists for, so it is its own
        # key rather than something a reader has to filter for.
        payload["packages"] = [entries[key] for key in sorted(entries)]
        payload["unresolved"] = sorted(
            key for key, value in entries.items() if not value["resolved"])
        payload["status"] = "success"
    except Exception as error:                      # noqa: BLE001 — reported
        payload["status"] = "error"
        payload["error"] = "{}: {}".format(type(error).__name__, error)
    if out:
        with open(out, "w") as handle:
            json.dump(payload, handle)


main()
