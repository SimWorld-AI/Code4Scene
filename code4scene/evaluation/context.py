"""What a verifier is given, and the two answers it can give without one.

Split out of the old ``verifiers.py`` so that a verifier module imports the
shape it needs and nothing else. Everything in here is shared by every
verifier; anything specific to one of them belongs in that one's file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import contracts

#: Suffix of the answer key written beside a generated task.
LABEL_SUFFIX = ".label.json"


class VerifierError(Exception):
    """A declared verifier could not be run."""


@dataclass
class Context:
    """Everything a verifier may look at."""

    record: dict[str, Any]              # the harness's run record
    task: Any                           # code4scene.tasks.task.Task
    ids: dict[str, str]                 # task_bundle_id / episode_id
    bridge: Any = None                  # live editor, when there is one
    #: A SECOND editor the agent never drove, and the only place a canonical
    #: scene may be opened. Verifiers that compare a candidate against the
    #: scene it was cut from need one: the canonical maps cannot be in the
    #: agent's instance, because `execute_python_script` is arbitrary code and
    #: no path filter survives it — measured, a guard that blocked the literal
    #: path fell to string concatenation, to chr(), to base64, and to
    #: `list_assets('/Game', True, True)`, which enumerates the answers without
    #: naming a path at all. None means no such environment was provided, and a
    #: verifier that needs one must report that it did not run.
    scoring: Any = None
    #: Renders of the finished scene. The judge scores images; without these
    #: it can only decline.
    images: list[str] = field(default_factory=list)
    #: Task-provided reference images, visible to the agent and therefore
    #: open-ended evidence rather than a hidden answer key.  Kept separate
    #: from candidate renders so the two roles cannot be swapped accidentally.
    reference_images: list[Any] = field(default_factory=list)
    #: Candidate-only renders addressed by camera and channel. This remains
    #: separate from ``renders`` because the latter may contain a GT scene.
    visual_renders: Any = None
    #: Renders addressed by scene, camera and channel — see eval.render
    #: RenderSet. An image-comparison verifier works pairwise by camera, which
    #: a flat list cannot express.
    renders: Any = None
    #: Formal RenderSets keyed by their complete evidence protocol. Canonical
    #: visual verifiers must select their own key and never consume another
    #: verifier's capture merely because both contain RGB images.
    render_evidence: dict[str, Any] = field(default_factory=dict)
    #: Where artifacts that must travel between environments are written. Both
    #: roles mount the same volume there; per-instance storage is invisible to
    #: the other pod.
    artifacts_dir: Any = None
    out_dir: Any = None                 # where this episode's artifacts live
    judge_verdict: dict[str, Any] | None = None
    spec: dict[str, Any] = field(default_factory=dict)   # this verifier's entry
    #: Scratch shared by verifiers that read the SAME evidence. Two metrics
    #: taken from one comparison should not export the level twice; whoever
    #: gets there first stores the result under its own key.
    cache: dict[str, Any] = field(default_factory=dict)

    def renders_for(self, protocol: str) -> Any:
        return self.render_evidence.get(protocol)


def error(kind: str, context: Context, reason: str) -> dict[str, Any]:
    """A verifier that could not run says so, and never scores."""
    return {**contracts.base(kind, context.ids), "status": contracts.ERROR,
            "score": None, "failure_reason": reason, "metrics": {},
            "evidence": {}, "artifacts": {}, "probes_used": (kind,)}


def read_label(context: Context) -> dict[str, Any] | None:
    """The answer key beside a generated task, if there is one.

    None means the task ships no key. A key that EXISTS and cannot be parsed
    raises instead: reporting it as "no answer key" charged a corrupt file to
    the task's design, and the caller's error report — the verifier framework
    turns the exception into an unscored error — is the right attribution.
    """

    path = getattr(context.task, "path", None)
    if not path:
        return None
    label = Path(path).with_suffix(LABEL_SUFFIX)
    if not label.is_file():
        # A generated task is written as <id>.yaml with <id>.label.json beside
        # it; with_suffix only replaces the last one, so a task named
        # `a.b.yaml` needs the fallback.
        label = Path(str(Path(path).with_suffix("")) + LABEL_SUFFIX)
    try:
        value = json.loads(label.read_text())
    except OSError:
        return None
    except ValueError as e:
        raise VerifierError(
            f"answer key JSON invalid at {label}: {e}") from e
    if not isinstance(value, dict):
        return value
    descriptor = value.get("canonical_scene")
    if value.get("canonical_actors") is not None or descriptor is None:
        return value
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "path",
        "sha256",
    }:
        raise VerifierError(
            "answer key canonical_scene must contain exactly path and sha256"
        )
    raw_path = descriptor.get("path")
    expected = descriptor.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise VerifierError("answer key canonical_scene path must be non-empty")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected)
    ):
        raise VerifierError(
            "answer key canonical_scene sha256 must be 64 hex characters"
        )
    scene_path = Path(raw_path)
    if not scene_path.is_absolute():
        scene_path = label.parent / scene_path
    scene_path = scene_path.resolve()
    try:
        payload = scene_path.read_bytes()
    except OSError as error:
        raise VerifierError(
            f"cannot read answer key canonical_scene at {scene_path}: {error}"
        ) from error
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected.casefold():
        raise VerifierError(
            "answer key canonical_scene sha256 mismatch at "
            f"{scene_path}: expected {expected}, got {observed}"
        )
    try:
        canonical_scene = json.loads(payload)
    except ValueError as error:
        raise VerifierError(
            f"answer key canonical_scene JSON invalid at {scene_path}: {error}"
        ) from error
    actors = (
        canonical_scene.get("actors")
        if isinstance(canonical_scene, Mapping)
        else None
    )
    if not isinstance(actors, list) or not actors or any(
        not isinstance(actor, Mapping) for actor in actors
    ):
        raise VerifierError(
            "answer key canonical_scene must contain a non-empty actors array"
        )
    value["canonical_actors"] = actors
    return value
