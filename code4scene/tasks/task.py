"""Task files: load, validate, hash.

A task is a YAML file binding inputs; the framework defines no concrete tasks.
See the benchmark's task-file documentation for the contract. Validation is deliberately shallow:
required keys must exist, everything else is preserved verbatim and passed
through to the runner and verifiers, so task authors can extend the format
(milestones, checkpoints, custom verifier arguments) without touching code.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from code4scene.core import scene

from .verifier_schema import (
    CANONICAL_VERIFIER_KINDS,
    validate_verifier_set,
)

#: ``assets`` is required for the same reason ``init_map`` is, one step
#: further out. A task is not only the level it starts on but the content it is
#: built out of, and an environment holding different content is not running
#: the same task. It is also what makes the pairing CHECKABLE before an
#: episode: a task asking for shipping containers, on an instance provisioned
#: without any, used to be indistinguishable from an agent that could not
#: build — same zero, same clean record, forty-three minutes later.
REQUIRED_TOP = ("id", "kind", "inputs", "verifiers", "assets")
#: ``init_map`` is required, not optional. A task does not have to start from
#: an empty world — some start part-way through one — so the level an episode
#: begins on is part of what the task IS, and two tasks that begin on different
#: levels are not comparable however alike their prompts read. Leaving it
#: implicit meant the editor started on whatever it happened to boot with.
REQUIRED_INPUTS = ("prompt", "init_map")

#: How the target scene is specified.  This is deliberately independent from
#: the task's generation/repair ``kind``: a generation can be guided by prose
#: or images, and an edit can eventually be guided by either as well.
CASE_TYPES: frozenset[str] = frozenset({"prompt_to_scene", "image_to_scene"})

#: Physical scene topology used by verifier-owned acquisition policies.  This
#: is deliberately explicit task metadata: an enclosed room and an outdoor
#: level need different safe camera searches, and neither a directory name nor
#: prompt wording is stable enough to choose a scoring protocol.
SCENE_ENVIRONMENTS: frozenset[str] = frozenset({"indoor", "outdoor"})

#: Optional authored close-view manifest accepted by the Indoor repair-target
#: camera planner.  The manifest remains task input metadata; only its camera
#: location, look-at target, and FOV are consumed by scoring.
CASE_CAMERA_MANIFEST_SCHEMA = "scenebench.indoor_review_case_cameras.v1"
CASE_CAMERA_MANIFEST_SCHEMAS = frozenset({
    "scenebench.indoor_case_camera.v3",
    "scenebench.indoor_case_camera.v4",
    "scenebench.indoor_redesign_case_cameras.v2",
    "scenebench.indoor_review_case_cameras.v1",
    "scenebench.indoor_scene_pool_cameras.v1",
})


#: The verifiers a task file may name.
#:
#: The vocabulary lives with the schema rather than with the implementations,
#: so that loading a task does not require importing the scoring layer — the
#: test set is what is asked, and what is asked cannot depend on what scores
#: it. `evaluation.verifiers` must cover exactly this set, and
#: `tests/test_architecture.py` fails if the two ever disagree.
VERIFIER_KINDS: frozenset[str] = CANONICAL_VERIFIER_KINDS


class TaskError(Exception):
    """The task file is missing required keys or is not well-formed."""


def _case_camera_manifest_reference(
    task_path: Path,
    data: dict[str, Any],
) -> tuple[Path, str] | None:
    source = data.get("source")
    if not isinstance(source, dict) or source.get("camera_manifest") is None:
        return None
    declared = source["camera_manifest"]
    if isinstance(declared, str):
        manifest_value = declared.strip()
        integrity = source.get("integrity")
        digest = (
            integrity.get("camera_manifest_sha256")
            if isinstance(integrity, dict)
            else None
        )
    elif isinstance(declared, dict):
        manifest_value = str(declared.get("path") or "").strip()
        digest = declared.get("sha256")
        integrity = source.get("integrity")
        integrity_digest = (
            integrity.get("camera_manifest_sha256")
            if isinstance(integrity, dict)
            else None
        )
        if digest is None:
            digest = integrity_digest
        elif integrity_digest is not None and str(digest) != str(integrity_digest):
            raise TaskError(
                f"{task_path}: source.camera_manifest.sha256 disagrees with "
                "source.integrity.camera_manifest_sha256"
            )
    else:
        raise TaskError(
            f"{task_path}: source.camera_manifest must be a path or a "
            "mapping with path and sha256"
        )
    digest_value = str(digest or "").strip().casefold()
    if not manifest_value:
        raise TaskError(f"{task_path}: source.camera_manifest path must be non-empty")
    if len(digest_value) != 64 or any(
        value not in "0123456789abcdef" for value in digest_value
    ):
        raise TaskError(
            f"{task_path}: camera manifest needs an exact SHA-256 digest"
        )
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = task_path.parent / manifest_path
    return manifest_path.resolve(), digest_value


def _case_camera_manifest(
    task_path: Path,
    data: dict[str, Any],
) -> dict[str, Any] | None:
    reference = _case_camera_manifest_reference(task_path, data)
    if reference is None:
        return None
    manifest_path, expected_sha256 = reference
    try:
        raw = manifest_path.read_bytes()
    except OSError as exception:
        raise TaskError(
            f"{task_path}: cannot read camera manifest {manifest_path}: {exception}"
        ) from exception
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise TaskError(
            f"{task_path}: camera manifest SHA-256 mismatch for {manifest_path}: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise TaskError(
            f"{task_path}: invalid camera manifest {manifest_path}: {exception}"
        ) from exception
    if not isinstance(manifest, dict):
        raise TaskError(f"{task_path}: camera manifest must be a JSON object")
    if manifest.get("schema_version") not in CASE_CAMERA_MANIFEST_SCHEMAS:
        raise TaskError(
            f"{task_path}: unsupported camera manifest schema "
            f"{manifest.get('schema_version')!r}"
        )
    task_id = str(data.get("id") or "")
    embedded_id = manifest.get("task_id") or manifest.get("case_id")
    identity_matches = (
        str(embedded_id) == task_id
        if embedded_id is not None
        else manifest_path.stem == task_id
    )
    if not identity_matches:
        raise TaskError(
            f"{task_path}: camera manifest identity must equal the task id"
        )
    views = manifest.get("views")
    if not isinstance(views, list) or len(views) != 2:
        raise TaskError(
            f"{task_path}: camera manifest must declare exactly two views"
        )
    names: set[str] = set()
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            raise TaskError(
                f"{task_path}: camera manifest views[{index}] must be an object"
            )
        name = str(view.get("name") or "").strip()
        if not name or name in names:
            raise TaskError(
                f"{task_path}: camera manifest view names must be non-empty "
                "and unique"
            )
        names.add(name)
        for field_name in ("location_cm", "target_cm"):
            vector = view.get(field_name)
            if (
                not isinstance(vector, list)
                or len(vector) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in vector
                )
            ):
                raise TaskError(
                    f"{task_path}: camera manifest views[{index}].{field_name} "
                    "must be three finite numbers"
                )
        if view["location_cm"] == view["target_cm"]:
            raise TaskError(
                f"{task_path}: camera manifest views[{index}] camera and target "
                "locations must differ"
            )
        fov = view.get("fov_deg")
        if (
            isinstance(fov, bool)
            or not isinstance(fov, (int, float))
            or not math.isfinite(float(fov))
            or not 10.0 <= float(fov) <= 160.0
        ):
            raise TaskError(
                f"{task_path}: camera manifest views[{index}].fov_deg must be "
                "finite and between 10 and 160"
            )
    return manifest


def _paired_render_qa_reference(
    task_path: Path,
    data: dict[str, Any],
) -> tuple[Path, str] | None:
    source = data.get("source")
    if not isinstance(source, dict) or source.get("paired_render_qa") is None:
        return None
    declared = source["paired_render_qa"]
    if isinstance(declared, str):
        qa_value = declared.strip()
        integrity = source.get("integrity")
        digest = (
            integrity.get("paired_render_qa_sha256")
            if isinstance(integrity, dict)
            else None
        )
    elif isinstance(declared, dict):
        qa_value = str(declared.get("path") or "").strip()
        digest = declared.get("sha256")
        integrity = source.get("integrity")
        integrity_digest = (
            integrity.get("paired_render_qa_sha256")
            if isinstance(integrity, dict)
            else None
        )
        if digest is None:
            digest = integrity_digest
        elif integrity_digest is not None and str(digest) != str(integrity_digest):
            raise TaskError(
                f"{task_path}: source.paired_render_qa.sha256 disagrees with "
                "source.integrity.paired_render_qa_sha256"
            )
    else:
        raise TaskError(
            f"{task_path}: source.paired_render_qa must be a path or a "
            "mapping with path and sha256"
        )
    digest_value = str(digest or "").strip().casefold()
    if not qa_value:
        raise TaskError(
            f"{task_path}: source.paired_render_qa path must be non-empty"
        )
    if len(digest_value) != 64 or any(
        value not in "0123456789abcdef" for value in digest_value
    ):
        raise TaskError(
            f"{task_path}: paired render QA needs an exact SHA-256 digest"
        )
    qa_path = Path(qa_value)
    if not qa_path.is_absolute():
        qa_path = task_path.parent / qa_path
    return qa_path.resolve(), digest_value


def _paired_render_qa(
    task_path: Path,
    data: dict[str, Any],
) -> dict[str, Any] | None:
    reference = _paired_render_qa_reference(task_path, data)
    if reference is None:
        return None
    camera_reference = _case_camera_manifest_reference(task_path, data)
    if camera_reference is None:
        raise TaskError(
            f"{task_path}: source.paired_render_qa requires a camera manifest"
        )
    camera_path, camera_sha256 = camera_reference
    camera_manifest = _case_camera_manifest(task_path, data)
    if camera_manifest is None:  # pragma: no cover - guarded above
        raise TaskError(f"{task_path}: paired QA camera manifest is unavailable")

    qa_path, expected_sha256 = reference
    try:
        raw = qa_path.read_bytes()
    except OSError as exception:
        raise TaskError(
            f"{task_path}: cannot read paired render QA {qa_path}: {exception}"
        ) from exception
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise TaskError(
            f"{task_path}: paired render QA SHA-256 mismatch for {qa_path}: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    try:
        qa = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise TaskError(
            f"{task_path}: invalid paired render QA {qa_path}: {exception}"
        ) from exception
    if not isinstance(qa, dict):
        raise TaskError(f"{task_path}: paired render QA must be a JSON object")

    task_id = str(data.get("id") or "")
    if str(qa.get("case_id") or qa.get("task_id") or "") != task_id:
        raise TaskError(
            f"{task_path}: paired render QA identity must equal the task id"
        )
    qa_camera = qa.get("camera_manifest")
    if (
        not isinstance(qa_camera, dict)
        or str(qa_camera.get("sha256") or "").casefold() != camera_sha256
    ):
        raise TaskError(
            f"{task_path}: paired render QA does not bind the task camera manifest"
        )
    declared_camera_path = qa_camera.get("path")
    if isinstance(declared_camera_path, str) and declared_camera_path.strip():
        bound_path = Path(declared_camera_path)
        if not bound_path.is_absolute():
            bound_path = qa_path.parent.parent / bound_path
        if bound_path.resolve() != camera_path:
            raise TaskError(
                f"{task_path}: paired render QA camera path does not match the task"
            )

    gates = qa.get("gates")
    required_global_gates = (
        "all_camera_similarity_guards_passed",
        "all_gt_repeat_controls_passed",
        "all_locality_guards_passed",
        "at_least_one_strongly_visible_view",
        "both_views_weakly_visible",
        "camera_summaries_match",
        "hard_gates_passed",
    )
    if (
        qa.get("passed") is not True
        or str(qa.get("status") or "").upper() != "PASS"
        or not isinstance(gates, dict)
        or any(gates.get(name) is not True for name in required_global_gates)
    ):
        raise TaskError(
            f"{task_path}: paired render QA did not pass every required hard gate"
        )

    capture_maps = qa.get("capture_maps")
    ground_truth = resolve_ground_truth_map(task_path, data.get("verifiers") or [])
    if not isinstance(capture_maps, dict) or ground_truth is None:
        raise TaskError(
            f"{task_path}: paired render QA lacks Input/GT capture-map identity"
        )
    if (
        normalize_ground_truth_map(str(capture_maps.get("input") or ""))
        != normalize_ground_truth_map(str(data["inputs"]["init_map"]))
        or normalize_ground_truth_map(str(capture_maps.get("gt") or ""))
        != ground_truth
    ):
        raise TaskError(
            f"{task_path}: paired render QA Input/GT maps do not match the task"
        )

    expected_names = [str(value["name"]) for value in camera_manifest["views"]]
    per_view = qa.get("per_view")
    if (
        not isinstance(per_view, list)
        or [str(value.get("view") or "") for value in per_view] != expected_names
    ):
        raise TaskError(
            f"{task_path}: paired render QA views do not match the camera manifest"
        )
    for index, view in enumerate(per_view):
        view_gates = view.get("gates")
        robust_delta = view.get("robust_delta")
        fraction = (
            robust_delta.get("fraction")
            if isinstance(robust_delta, dict)
            else None
        )
        if (
            not isinstance(view_gates, dict)
            or any(
                view_gates.get(name) is not True
                for name in (
                    "camera_similarity",
                    "gt_repeat_stable",
                    "locality",
                    "weak_visibility",
                )
            )
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or float(fraction) <= 0.0
        ):
            raise TaskError(
                f"{task_path}: paired render QA view {index} lacks a passing "
                "GT/Input pixel-observability proof"
            )
    return qa


def normalize_ground_truth_map(value: str) -> str:
    """Normalize a UE level object path to its package identity."""

    raw = str(value).strip()
    if "'" in raw and raw.endswith("'"):
        raw = raw.split("'", 1)[1][:-1]
    if not raw.startswith("/Game/"):
        raise TaskError(f"ground-truth map must be a /Game package, got {value!r}")
    return raw.split(".", 1)[0]


def resolve_ground_truth_map(
    path: str | Path,
    verifiers: list[dict[str, Any]],
) -> str | None:
    """Resolve every declared GT source and reject normalized disagreement."""

    task_path = Path(path)
    label_path = task_path.with_suffix(".label.json")
    label: dict[str, Any] = {}
    if label_path.exists():
        try:
            loaded = json.loads(label_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exception:
            raise TaskError(f"cannot read GT label {label_path}: {exception}") from exception
        if not isinstance(loaded, dict):
            raise TaskError(f"GT label must be an object: {label_path}")
        label = loaded
    nested = label.get("gt") if isinstance(label.get("gt"), dict) else {}
    declared: list[tuple[str, str]] = []
    for source, value in (
        ("label.canonical_map", label.get("canonical_map")),
        ("label.source_level", label.get("source_level")),
        ("label.gt.source_map", nested.get("source_map")),
        ("label.gt.map", nested.get("map")),
    ):
        if isinstance(value, str) and value.strip():
            declared.append((source, normalize_ground_truth_map(value)))
    for spec in verifiers:
        for gt_field in ("ground_truth", "canonical_map"):
            value = spec.get(gt_field)
            if isinstance(value, str) and value.strip():
                declared.append(
                    (
                        f"verifier[{spec.get('name')}].{gt_field}",
                        normalize_ground_truth_map(value),
                    )
                )
    values = {value for _source, value in declared}
    if len(values) > 1:
        detail = ", ".join(f"{source}={value}" for source, value in declared)
        raise TaskError(f"conflicting ground-truth maps: {detail}")
    return next(iter(values), None)


@dataclass(frozen=True)
class Task:
    """A loaded task file. ``data`` keeps the document verbatim."""

    path: Path
    data: dict[str, Any]
    sha256: str = field(repr=False)

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def kind(self) -> str:
        return str(self.data["kind"])

    @property
    def case_type(self) -> str:
        """Input modality, defaulting old tasks to the established text path."""

        return str(self.data.get("case_type") or "prompt_to_scene")

    @property
    def scene_environment(self) -> str | None:
        """Explicit camera-routing environment for image-to-scene tasks."""

        value = self.data.get("scene_environment")
        if value is None:
            return None
        text = str(value).strip().casefold()
        return text or None

    @property
    def inputs(self) -> dict[str, Any]:
        return self.data["inputs"]

    @property
    def verifiers(self) -> list[dict[str, Any]]:
        return self.data["verifiers"]

    @property
    def prompt(self) -> str:
        return str(self.inputs["prompt"]).strip()

    @property
    def size_m(self) -> float | None:
        """Legacy no-GT world-plate edge length, if the task declares one."""
        value = self.inputs.get("size_m")
        return float(value) if value is not None else None

    @property
    def half_extent_m(self) -> float | None:
        """Legacy no-GT plate half-extent used when no canonical GT exists."""
        size = self.size_m
        return size / 2 if size is not None else None

    @property
    def init_map(self) -> str:
        """The level this episode starts from.

        A task input rather than an environment constant: what an agent is
        asked to build on is part of the task, and two tasks that start from
        different levels are not comparable no matter how alike their prompts
        read. The environment still needs SOME level to boot — it starts
        before any task exists — so this is opened at reset, on top of
        whatever the editor came up with.

        The packs the level references must be among the task's own assets.
        A level whose dependencies were not provisioned loads with unresolved
        references and saves nothing, which is not obviously a content problem
        from the outside: the editor reports the missing packages and then
        carries on.
        """
        return str(self.inputs["init_map"]).strip()

    @property
    def packs(self) -> tuple[str, ...]:
        """The explicit physical content roots this task is built out of.

        An environment does not choose what content a task gets: it is handed
        these names plus the physical roots resolved from :attr:`source_pack`,
        mounts exactly that union, and refuses to come up if it cannot. The
        alias expansion lives in environment provisioning because
        ``source.pack`` is an authored dataset name, not necessarily a
        ``/Game`` root.

        The block these come from was already written by the task builders and
        read by nothing; it recorded what a task was made of, which is exactly
        the fact provisioning needs.
        """
        return tuple(str(name).strip()
                     for name in (self.data["assets"].get("packs") or []))

    @property
    def pack_paths(self) -> dict[str, Path]:
        """Optional task-pinned physical directories for explicit packs.

        Portable tasks normally resolve :attr:`packs` from machine-provided
        ``--source`` roots. Local artifact tasks may instead pin selected pack
        directories here; provisioning consumes those paths before considering
        machine sources for the remaining packs.
        """

        return {
            str(name).strip(): Path(value.strip())
            for name, value in (self.data["assets"].get("pack_paths") or {}).items()
        }

    @property
    def source_pack(self) -> str | None:
        """Authored logical pack name whose UE roots must be provisioned.

        This is provenance and a provisioning input. It is deliberately not
        guessed from Candidate content: the required environment is frozen by
        the task before the Candidate exists.
        """

        source = self.data.get("source")
        if not isinstance(source, dict) or source.get("pack") is None:
            return None
        value = str(source["pack"]).strip()
        return value or None

    @property
    def asset_roots(self) -> tuple[str, ...]:
        """:attr:`packs` as ``/Game`` prefixes, each ending in a slash.

        Trailing slash always: these are compared by prefix, and a root without
        one both fails to match itself and matches its neighbours — ``/Game/Winter``
        would admit ``/Game/WinterTown``.
        """
        return tuple(f"/Game/{name}/" for name in self.packs)

    @property
    def reference_views(self) -> list[Path]:
        """Images of the intended scene, if the task ships any.

        A reference-guided task's prompt names these ("match the provided
        reference views"), so an agent that is never shown them is being asked
        for something it cannot see — and scored on the result. They live under
        ``source`` because they come from the release a task was imported from,
        not from the harness. Missing files are dropped rather than raised on:
        a release whose renders were never produced is a real state the
        importer already reports, and it must not stop the run.
        """
        return [path for path in self.declared_reference_views if path.is_file()]

    @property
    def declared_reference_views(self) -> list[Path]:
        """Every path the task declared, including missing scoring evidence.

        The agent-facing staging path historically omits files a release never
        rendered.  A verifier cannot silently omit one image from its
        denominator, so scoring receives this complete list and validates each
        path itself.
        """

        source = self.data.get("source") or {}
        views = source.get("reference_views") or []
        paths = []
        for value in views:
            path = Path(str(value))
            if not path.is_absolute():
                path = self.path.parent / path
            paths.append(path.resolve())
        return paths

    @property
    def case_camera_manifest_path(self) -> Path | None:
        """Integrity-pinned authored close-view manifest, when declared."""

        reference = _case_camera_manifest_reference(self.path, self.data)
        return reference[0] if reference is not None else None

    @property
    def case_camera_manifest_sha256(self) -> str | None:
        """Expected digest of :attr:`case_camera_manifest_path`."""

        reference = _case_camera_manifest_reference(self.path, self.data)
        return reference[1] if reference is not None else None

    @property
    def case_camera_manifest(self) -> dict[str, Any] | None:
        """Validated authored close views; photometric fields stay ignored."""

        return _case_camera_manifest(self.path, self.data)

    @property
    def paired_render_qa_path(self) -> Path | None:
        """Integrity-pinned GT-A/Input/GT-B camera validation, when declared."""

        reference = _paired_render_qa_reference(self.path, self.data)
        return reference[0] if reference is not None else None

    @property
    def paired_render_qa_sha256(self) -> str | None:
        """Expected digest of :attr:`paired_render_qa_path`."""

        reference = _paired_render_qa_reference(self.path, self.data)
        return reference[1] if reference is not None else None

    @property
    def paired_render_qa(self) -> dict[str, Any] | None:
        """Validated deterministic GT/Input pixel-observability evidence."""

        return _paired_render_qa(self.path, self.data)

    def budget(self, key: str) -> Any:
        return (self.inputs.get("budget") or {}).get(key)

    @property
    def ground_truth_map(self) -> str | None:
        return resolve_ground_truth_map(self.path, self.verifiers)


def pack_of(game_path: str) -> str:
    """The content pack a ``/Game`` path belongs to.

    One reading, shared: a task declares packs, provisioning resolves packs,
    and the level check compares packs — three places that must agree about
    where the name ends, which is at the first slash after ``/Game/``.
    """
    text = str(game_path).strip()
    if not text.startswith("/Game/"):
        raise TaskError(f"not a /Game path: {game_path!r}")
    return text[len("/Game/"):].split("/", 1)[0]


def load(path: str | Path) -> Task:
    """Load and validate one task file."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise TaskError(f"cannot read task file {p}: {e}") from e
    try:
        data = yaml.safe_load(raw.decode())
    except yaml.YAMLError as e:
        raise TaskError(f"invalid YAML in {p}: {e}") from e
    if not isinstance(data, dict):
        raise TaskError(f"task file must be a mapping: {p}")
    if "evaluation_policy" in data:
        raise TaskError(
            f"{p}: task-level evaluation_policy is unsupported; the release "
            "derives one current policy from case_type"
        )

    missing = [key for key in REQUIRED_TOP if key not in data]
    missing += [f"inputs.{key}" for key in REQUIRED_INPUTS
                if isinstance(data.get("inputs"), dict) and key not in data["inputs"]]
    if missing:
        raise TaskError(f"{p}: missing required key(s): {', '.join(missing)}")
    if not isinstance(data["inputs"], dict):
        raise TaskError(f"{p}: 'inputs' must be a mapping")
    if not isinstance(data["verifiers"], list):
        raise TaskError(f"{p}: 'verifiers' must be a list")

    case_type = str(data.get("case_type") or "prompt_to_scene")
    if case_type not in CASE_TYPES:
        raise TaskError(
            f"{p}: case_type must be one of {', '.join(sorted(CASE_TYPES))}, "
            f"got {case_type!r}"
        )
    scene_environment = data.get("scene_environment")
    if scene_environment is not None:
        normalized_environment = str(scene_environment).strip().casefold()
        if normalized_environment not in SCENE_ENVIRONMENTS:
            raise TaskError(
                f"{p}: scene_environment must be one of "
                f"{', '.join(sorted(SCENE_ENVIRONMENTS))}, got "
                f"{scene_environment!r}"
            )
        if case_type != "image_to_scene":
            raise TaskError(
                f"{p}: scene_environment is only valid for image_to_scene tasks"
            )
    verifier_names = {
        str(spec.get("name"))
        for spec in data["verifiers"]
        if isinstance(spec, dict)
    }
    gt_repair_only = (
        "gt_repair" in verifier_names
        and "semantic_requirements" not in verifier_names
    )
    if "gt_repair" in verifier_names:
        source = data.get("source")
        source_pack = source.get("pack") if isinstance(source, dict) else None
        if not isinstance(source_pack, str) or not source_pack.strip():
            raise TaskError(
                f"{p}: gt_repair needs a non-empty source.pack so the scoring "
                "environment can resolve and mount the authored content "
                "before the verifier/editor starts"
            )
    if case_type == "image_to_scene" and not gt_repair_only:
        source = data.get("source")
        if not isinstance(source, dict):
            raise TaskError(
                f"{p}: image_to_scene needs a source mapping with reference_views"
            )
        references = source.get("reference_views")
        if not isinstance(references, list) or not references:
            raise TaskError(
                f"{p}: image_to_scene needs a non-empty source.reference_views list"
            )
        invalid_reference = any(
            not isinstance(value, str) or not value.strip()
            for value in references
        )
        if invalid_reference:
            raise TaskError(
                f"{p}: source.reference_views entries must be non-empty paths"
            )

    if _case_camera_manifest_reference(p, data) is not None:
        if not gt_repair_only or case_type != "image_to_scene":
            raise TaskError(
                f"{p}: source.camera_manifest is only valid for image_to_scene "
                "tasks scored solely by gt_repair"
            )
        if str(scene_environment or "").strip().casefold() != "indoor":
            raise TaskError(
                f"{p}: source.camera_manifest requires scene_environment: indoor"
            )
        _case_camera_manifest(p, data)
        if _paired_render_qa_reference(p, data) is not None:
            _paired_render_qa(p, data)
    elif _paired_render_qa_reference(p, data) is not None:
        raise TaskError(
            f"{p}: source.paired_render_qa requires source.camera_manifest"
        )

    for i, spec in enumerate(data["verifiers"]):
        if not isinstance(spec, dict) or "name" not in spec:
            raise TaskError(
                f"{p}: verifiers[{i}] must be a mapping with a 'name'. The key "
                f"used to be `kind`, which the file already uses at the top "
                f"level for the TASK's kind — two different vocabularies under "
                f"one word.")
        # Checked for MEANING, not just shape. The shape check was here on its
        # own, so a task could name a verifier nothing implements and still run
        # to a clean score — which is how a generated task's answer key went
        # unopened. Refused at load, before an environment is leased, rather
        # than after an agent has spent an hour.
        if spec["name"] not in CANONICAL_VERIFIER_KINDS:
            raise TaskError(
                f"{p}: verifiers[{i}] names {spec['name']!r}, which nothing "
                "implements; available IDs: "
                f"{', '.join(sorted(CANONICAL_VERIFIER_KINDS))}")

    try:
        validate_verifier_set(data["verifiers"], case_type=case_type)
        resolve_ground_truth_map(p, data["verifiers"])
    except (ValueError, TaskError) as exception:
        raise TaskError(f"{p}: {exception}") from exception

    assets = data["assets"]
    if not isinstance(assets, dict):
        raise TaskError(f"{p}: 'assets' must be a mapping with a 'packs' list")
    packs = assets.get("packs")
    if not isinstance(packs, list) or not packs:
        raise TaskError(
            f"{p}: 'assets.packs' must be a non-empty list of content pack "
            f"names — what this task is built out of, which the environment "
            f"provisions and nothing else decides")
    bad = [name for name in packs
           if not isinstance(name, str) or not name.strip() or "/" in name]
    if bad:
        raise TaskError(
            f"{p}: 'assets.packs' takes pack NAMES, not paths: {bad}. A name "
            f"is what provisioning resolves against its content sources; "
            f"'/Game/X/' is what it becomes once mounted")

    # The level an episode opens must come from content the task asked for.
    # Checked here rather than at provision time because it needs nothing but
    # the file, and a task that contradicts itself should never reach a host.
    init_map = str(data["inputs"]["init_map"]).strip()
    names = {str(name).strip() for name in packs}
    pack_paths = assets.get("pack_paths", {})
    if not isinstance(pack_paths, dict):
        raise TaskError(f"{p}: 'assets.pack_paths' must be a mapping")
    unknown_pack_paths = sorted(
        str(name) for name in pack_paths if str(name).strip() not in names
    )
    if unknown_pack_paths:
        raise TaskError(
            f"{p}: 'assets.pack_paths' may only name declared assets.packs; "
            f"unknown: {', '.join(unknown_pack_paths)}"
        )
    for name, value in pack_paths.items():
        pack_name = str(name).strip()
        if not isinstance(name, str) or not pack_name or "/" in pack_name:
            raise TaskError(
                f"{p}: 'assets.pack_paths' has invalid pack name {name!r}"
            )
        if not isinstance(value, str) or not value.strip():
            raise TaskError(
                f"{p}: assets.pack_paths.{pack_name} must be a non-empty "
                "absolute directory path"
            )
        pack_path = Path(value.strip())
        if not pack_path.is_absolute():
            raise TaskError(
                f"{p}: assets.pack_paths.{pack_name} must be absolute, got "
                f"{value!r}"
            )
        if pack_path.name != pack_name:
            raise TaskError(
                f"{p}: assets.pack_paths.{pack_name} must point directly to "
                f"the {pack_name!r} directory, got {value!r}"
            )
    # By pack SEGMENT, the same way provisioning reads it — a prefix test would
    # miss `/Game/X`, a package sitting directly under its own pack root.
    owned = [name for name in names if name in scene.INSTANCE_OWNED_ROOTS]
    if owned:
        raise TaskError(
            f"{p}: 'assets.packs' names {', '.join(sorted(owned))}, which an "
            f"instance owns rather than mounts — provisioning makes those, so "
            f"there is no source to ask for them")

    if init_map != scene.BLANK_STAGE and init_map.startswith("/Game/"):
        pack = init_map[len("/Game/"):].split("/", 1)[0]
        if pack in scene.INSTANCE_OWNED_ROOTS:
            pack = None                 # a level written INTO the instance
        if pack is not None and pack not in names:
            raise TaskError(
                f"{p}: init_map {init_map} lives in pack {pack!r}, which is "
                f"not among this task's assets ({', '.join(sorted(names))}). "
                f"A task's assets must include the level it starts on, or the "
                f"editor opens an untitled world, spawning works, and only "
                f"the save fails — after a whole episode has run")

    return Task(path=p, data=data, sha256=hashlib.sha256(raw).hexdigest())


def load_dir(path: str | Path) -> list[Task]:
    """Load every ``*.yaml`` under a directory, sorted by file name."""
    return [load(p) for p in sorted(Path(path).glob("*.yaml"))]
