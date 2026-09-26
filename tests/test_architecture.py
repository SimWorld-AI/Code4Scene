"""The layering, enforced.

The package is a small core, the task schema, the verifier layer and, above
it, the paper's scoring protocol. A layering that only lives in a document
drifts, so the rule is a test: it reads every import in the package and fails
when a layer reaches somewhere it may not.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


PKG = Path(importlib.import_module("code4scene").__file__).resolve().parent

#: layer -> the layers it may import from. Absent means "nothing in the package".
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "tasks": {"core"},
    # Verifiers and result assembly publish the protocol's numbers.
    "evaluation": {"core", "tasks", "protocol"},
    # The protocol builds on a fixed set of measurement primitives only; see
    # PROTOCOL_PRIMITIVES and the acyclicity test below.
    "protocol": {"core", "tasks", "evaluation"},
    # Top-level modules (cli, bundle, scoring, resources) may reach anything.
    "": {"core", "tasks", "evaluation", "protocol"},
}

#: The edges that carry a decision rather than a preference.
LOAD_BEARING = {
    ("core", "evaluation"):
        "core is plumbing; plumbing that needs the scoring layer is not plumbing",
    ("core", "protocol"):
        "core is plumbing; plumbing that needs the scoring rules is not plumbing",
}

#: The only evaluation modules the scoring protocol may import: measurement
#: primitives, never verifiers or result assembly. Together with the
#: acyclicity test this keeps one direction at module level: primitives ->
#: protocol -> verifiers/result assembly.
PROTOCOL_PRIMITIVES = {
    "code4scene.evaluation.assignment",
    "code4scene.evaluation.measure_payload",
    "code4scene.evaluation.repair_success",
    "code4scene.evaluation.repair_target_scope",
    "code4scene.evaluation.requirement_graph.repair_target_authoring",
    "code4scene.evaluation.scalar_score",
    "code4scene.evaluation.scene_diff",
}


def layer_of(module: str) -> str:
    """`code4scene.evaluation.verifiers.vlm_as_judge.rubric` -> `evaluation`; `code4scene.cli` -> ``."""
    parts = module.split(".")
    if len(parts) < 2:
        return ""
    head = parts[1]
    return head if head in ALLOWED else ""


def module_name(path: Path) -> str:
    rel = path.relative_to(PKG).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["code4scene", *parts])


def imports_of(path: Path, module: str) -> list[tuple[str, int]]:
    """Every `code4scene.*` module this file imports, with its line number."""
    tree = ast.parse(path.read_text(), str(path))
    package = module.rsplit(".", 1)[0] if path.name != "__init__.py" else module
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                target = ".".join([*base, *(node.module.split(".") if node.module else [])])
            else:
                target = node.module or ""
            if target.startswith("code4scene"):
                found.append((target, node.lineno))
            # The module alone is not the whole import. `from code4scene
            # import evaluation` binds a layer while `node.module` names none,
            # so recording only the module let that exact line cross any
            # boundary unseen. Each bound name is registered too — but only
            # when it lands in a different layer than the module did, so a
            # plain `from code4scene.core.bridge import Bridge` is not
            # reported twice.
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = f"{target}.{alias.name}" if target else alias.name
                if (bound.startswith("code4scene")
                        and layer_of(bound) != layer_of(target)):
                    found.append((bound, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("code4scene"):
                    found.append((alias.name, node.lineno))
    return found


#: Editor-side scripts are data sent over the bridge, never imported.
NOT_IMPORTED = ("ue_scripts",)


def source_files() -> list[Path]:
    return [p for p in sorted(PKG.rglob("*.py"))
            if "__pycache__" not in str(p)
            and p.relative_to(PKG).parts[0] not in NOT_IMPORTED]


def test_editor_scripts_are_never_imported_by_the_package():
    for path in source_files():
        for target, line in imports_of(path, module_name(path)):
            assert not target.startswith("code4scene.ue_scripts"), (
                f"{path.name}:{line} imports an editor-side script")


def test_every_file_belongs_to_a_layer():
    """A new top-level package is a fifth layer, and that is a decision."""
    tops = {p.relative_to(PKG).parts[0] for p in source_files()
            if len(p.relative_to(PKG).parts) > 1}
    assert tops <= set(ALLOWED) - {""}, (
        f"unplaced top-level packages: {sorted(tops - set(ALLOWED))}. Either it "
        f"belongs in one of {sorted(set(ALLOWED) - {''})}, or the package "
        f"needs a new layer and a reason.")


def test_no_layer_reaches_where_it_may_not():
    crossings = []
    for path in source_files():
        module = module_name(path)
        here = layer_of(module)
        for target, line in imports_of(path, module):
            there = layer_of(target)
            if there in ("", here):
                continue
            if there not in ALLOWED[here]:
                rel = path.relative_to(PKG.parent.parent)
                crossings.append(f"{rel}:{line}  {here} -> {there}  ({target})")
    assert not crossings, (
        "these imports cross a layer boundary:\n  " + "\n  ".join(crossings)
        + "\n\nIf the edge is genuinely needed, the thing being imported "
          "probably belongs in `core`.")


@pytest.mark.parametrize(("frm", "to"), sorted(LOAD_BEARING))
def test_the_two_edges_that_carry_a_decision(frm, to):
    """The edges whose absence is a design decision."""
    assert to not in ALLOWED[frm], LOAD_BEARING[(frm, to)]
    offenders = []
    for path in source_files():
        module = module_name(path)
        if layer_of(module) != frm:
            continue
        for target, line in imports_of(path, module):
            if layer_of(target) == to:
                offenders.append(f"{path.relative_to(PKG.parent.parent)}:{line} -> {target}")
    assert not offenders, f"{frm} must not import {to}: {LOAD_BEARING[(frm, to)]}\n" + "\n".join(offenders)


def _module_file(module: str) -> Path | None:
    parts = module.split(".")[1:]
    candidate = PKG.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    package = PKG.joinpath(*parts) / "__init__.py"
    return package if package.is_file() else None


def _imported_modules(path: Path, module: str) -> list[tuple[str, int]]:
    """Like imports_of, but `from pkg import mod` names the module itself."""
    tree = ast.parse(path.read_text(), str(path))
    package = module.rsplit(".", 1)[0] if path.name != "__init__.py" else module
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                target = ".".join([*base, *(node.module.split(".") if node.module else [])])
            else:
                target = node.module or ""
            if not target.startswith("code4scene"):
                continue
            for alias in node.names:
                full = f"{target}.{alias.name}"
                found.append((full if _module_file(full) else target, node.lineno))
        elif isinstance(node, ast.Import):
            found.extend((a.name, node.lineno) for a in node.names
                         if a.name.startswith("code4scene"))
    return found


def test_the_protocol_uses_measurement_primitives_only():
    offenders = []
    for path in source_files():
        module = module_name(path)
        if layer_of(module) != "protocol":
            continue
        for target, line in _imported_modules(path, module):
            if layer_of(target) == "evaluation" and target not in PROTOCOL_PRIMITIVES:
                offenders.append(f"{path.name}:{line} -> {target}")
    assert not offenders, (
        "the scoring protocol may only use measurement primitives:\n" + "\n".join(offenders))


def test_the_primitives_never_reach_back_into_the_protocol():
    """Module-level acyclicity: nothing the protocol imports imports it back."""
    seen, stack = set(), sorted(PROTOCOL_PRIMITIVES)
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_file(module)
        if path is None:
            continue
        for target, _line in _imported_modules(path, module):
            assert layer_of(target) != "protocol", (
                f"{module} imports {target}: the protocol's primitives must not "
                "depend on the protocol")
            if target.startswith("code4scene.") and _module_file(target) is not None:
                stack.append(target)


def test_core_depends_on_nothing_above_it():
    """If `core` grows an upward edge, something in it has a policy and belongs
    in the layer that owns that policy."""
    assert ALLOWED["core"] == set()
    for path in source_files():
        module = module_name(path)
        if layer_of(module) != "core":
            continue
        for target, line in imports_of(path, module):
            assert layer_of(target) in ("", "core"), (
                f"{path.name}:{line} imports {target}; core is plumbing, and "
                f"plumbing that needs a layer is not plumbing")


def test_the_schema_and_the_implementations_name_the_same_verifiers():
    """The vocabulary lives with the task schema; the code implements it.

    `task.py` refuses an unknown kind at load, before an environment is
    leased — which used to mean the schema imported the scoring layer, so what
    is asked depended on what scores it. The names live in `tasks` now and
    `evaluation` must cover exactly them; this is what keeps the two honest
    without the import.
    """
    from code4scene.evaluation import verifiers
    from code4scene.tasks.task import VERIFIER_KINDS

    assert set(verifiers.REGISTRY) == set(VERIFIER_KINDS)
    assert set(verifiers.CLASSES) == set(VERIFIER_KINDS), (
        "every kind needs a class — open-ended, gt, or declared per task")


def test_the_package_carries_everything_it_reads_at_run_time():
    """An installed (non-editable) copy must be able to score on its own.

    Policies used to be found by walking up from a module to the repository's
    ``configs`` directory: the repository layout, not the installed one. They
    are package data now and resolved through importlib.resources.
    """
    import tomllib

    from code4scene.resources import config_file

    required = {
        "configs/judge-policies/visual-gt-paired-formal.yaml": "the frozen judge policy",
        "configs/rubrics/visual-gt-paired.yaml": "the paired visual rubric",
        "configs/rubrics/scene-quality.yaml": "the scene-quality rubric",
        "configs/score-policies/text-to-scene-human-aligned.yaml": "the T2S score policy",
        "ue_scripts/measure_scene_physics.py": "the in-editor physics probe",
        "ue_scripts/measure_reachability.py": "the in-editor navmesh probe",
        "ue_scripts/export_scene_snapshot.py": "the in-editor scene exporter",
    }
    for relative, why in required.items():
        assert (PKG / relative).exists(), f"{relative} is not in the package — {why}"
    assert config_file("judge-policies", "visual-gt-paired-formal.yaml").is_file()

    pyproject = PKG.parent / "pyproject.toml"
    if not pyproject.is_file():
        return  # an installed copy: the files above are present, which is the point
    declared = tomllib.loads(pyproject.read_text())
    globs = declared["tool"]["setuptools"]["package-data"]["code4scene"]
    for relative in required:
        path = Path(relative)
        covered = any(path.match(g) for g in globs)
        assert covered, (
            f"{relative} is in the package tree but no package-data glob ships "
            f"it, so `pip install` produces a copy that cannot score. Globs: {globs}")


def test_every_public_verifier_is_one_registry_entry_with_a_declared_class():
    """Only canonical entry points form the public verifier surface.

    Leaf algorithms may live beside them as implementation modules, but they
    are not accepted task IDs and never enter the registry.
    """
    from code4scene.evaluation import verifiers
    from code4scene.tasks.task import VERIFIER_KINDS

    assert set(verifiers.CLASSES) == set(VERIFIER_KINDS)
    for kind in verifiers.CLASSES:
        module = importlib.import_module(f"code4scene.evaluation.verifiers.{kind}")
        assert callable(getattr(module, "verify", None)), (
            f"{kind} exports no verify(context)")
        assert getattr(module, "CLASS", None) in verifiers.CLASSES_ALLOWED, (
            f"{kind} must declare CLASS as one of {verifiers.CLASSES_ALLOWED}")

    root = PKG / "evaluation" / "verifiers"
    assert not (root / "plausibility.py").exists()
    assert not (root / "reference_image_alignment.py").exists()
    assert (root / "source_preservation.py").exists()


def test_the_checker_sees_a_layer_however_the_import_spells_it(tmp_path):
    """The checker is itself load-bearing, so its parser gets a test.

    Every row here is a way a file inside `core` could reach `evaluation`,
    and the last two are the ones that used to slip through: importing the
    layer as a NAME from the package root puts nothing in `node.module`, so a
    parser that records only the module saw `code4scene` — no layer — and let
    the line cross.
    """
    source = "\n".join([
        "import code4scene.evaluation",                        # 1
        "from code4scene.evaluation import contracts",         # 2
        "from code4scene.evaluation.contracts import x",       # 3
        "from ...evaluation import contracts",                 # 4
        "from code4scene import evaluation",                   # 5: the escape
        "from ... import evaluation",                          # 6: its relative twin
        "import json",                                         # not ours
        "from json import loads",                              # not ours
    ])
    path = tmp_path / "offender.py"
    path.write_text(source)

    # As if the file sat at code4scene/core/inner/offender.py.
    found = imports_of(path, "code4scene.core.inner.offender")

    reached: dict[int, set[str]] = {}
    for target, line in found:
        assert target.startswith("code4scene"), f"{target} is not ours to police"
        reached.setdefault(line, set()).add(layer_of(target))
    assert set(reached) == set(range(1, 7)), "the stdlib lines record nothing"
    for line in range(1, 7):
        assert "evaluation" in reached[line], (
            f"line {line} reaches evaluation and the checker did not see it")


def test_a_verifier_that_reads_the_answer_key_says_gt():
    """The declaration is one line, so it can be wrong. This is what stops it.

    `classify` is a lookup and every consumer trusts it: a verifier that opens
    the answer key while calling itself open-ended puts the answer into the
    answerless column, which is the one number the two-class split exists to
    keep honest. The answer key is reached exactly one way — `read_label` /
    `LABEL_SUFFIX` from `evaluation.context` — so the check is exact.
    """
    from code4scene.evaluation import verifiers

    expected_gt = {"gt_repair", "scene_diff"}
    assert {
        kind for kind, declared in verifiers.CLASSES.items() if declared == "gt"
    } == expected_gt
    assert all(
        declared == ("gt" if kind in expected_gt else "open_ended")
        for kind, declared in verifiers.CLASSES.items()
    )
