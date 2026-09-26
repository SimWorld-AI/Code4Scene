"""Rubrics: the canonical definition of what a judge is asked.

A rubric is a file, not code, for the same reason skill text is: its wording
changes the scores, so it has to be hashable. A verdict that does not record
which rubric content produced it is not comparable to anything.

Scoring policy lives here too — the metric ceilings. Where the deterministic
measurements cap the judged score is a policy decision, and putting it in the
rubric keeps it content-addressed alongside the questions rather than buried
in code where it can change silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class RubricError(Exception):
    """The rubric is missing something or is internally inconsistent."""


@dataclass(frozen=True)
class Criterion:
    id: str
    question: str
    weight: float = 1.0
    guidance: str = ""
    #: Which render channels are valid evidence for this question.  A depth
    #: image can support a spatial-layout score, but it cannot establish a
    #: material palette; keeping that distinction in the packaged rubric
    #: prevents the model from treating every image as interchangeable.
    channels: tuple[str, ...] = ("rgb",)


@dataclass(frozen=True)
class Ceiling:
    """A measured defect that caps the judged score.

    ``metric`` above (or below) ``threshold`` limits the final score to
    ``max_score``, expressed as a fraction of the rubric's range.
    """

    metric: str
    above: float | None = None
    below: float | None = None
    max_score: float = 1.0
    reason: str = ""

    def applies_to(self, metrics: dict[str, Any]) -> bool:
        """Whether this cap binds. An ABSENT metric binds it.

        This returned False when the metric was missing, so a measurement that
        failed did not cap the judge — it removed every cap the rubric had. The
        scene nobody could measure was the one the judge was free to score
        highest, which is exactly backwards: the ceilings exist because a scene
        that measures badly must not be rescued by photographing well, and a
        scene that measures not at all is not evidence of quality.
        """
        value = metrics.get(self.metric)
        if value is None:
            return True
        # A metric that is not a number cannot be compared with a threshold.
        # It used to raise from inside the comparison, so a malformed record
        # reached the report as `TypeError: '<' not supported between ...`
        # instead of a judged score or a stated reason. An unusable metric
        # binds the ceiling, for the same reason an absent one does: the cap
        # exists so a scene that measures badly cannot photograph its way out,
        # and "we could not read the measurement" is not evidence of quality.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return True
        if self.above is not None and value > self.above:
            return True
        return self.below is not None and value < self.below


@dataclass(frozen=True)
class ScoreCeiling:
    """A low semantic score that limits the overall judged score.

    This is deliberately deterministic.  A scene that is internally coherent
    but depicts the wrong kind of place must not make up for a failed setting
    identity by accumulating style points in other criteria.
    """

    criterion: str
    at_or_below: int
    max_score: float = 1.0
    reason: str = ""

    def applies_to(self, scores: dict[str, int]) -> bool:
        return scores[self.criterion] <= self.at_or_below


@dataclass(frozen=True)
class Rubric:
    id: str
    criteria: list[Criterion]
    scale_max: int = 4
    ceilings: list[Ceiling] = field(default_factory=list)
    score_ceilings: list[ScoreCeiling] = field(default_factory=list)
    instructions: str = ""
    channels: tuple[str, ...] = ("rgb",)

    @property
    def name(self) -> str:
        return self.id

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)

    def ceiling_for(self, metrics: dict[str, Any]) -> tuple[float, list[str]]:
        """The lowest applicable ceiling and why it applies."""
        applied = [c for c in self.ceilings if c.applies_to(metrics)]
        if not applied:
            return 1.0, []
        strictest = min(applied, key=lambda c: c.max_score)
        # Stripped: a reason written as a folded YAML block carries a trailing
        # newline, and these go verbatim into the verdict a report renders.
        return strictest.max_score, [
            (c.reason or
             f"{c.metric} beyond {c.above if c.above is not None else c.below}").strip()
            for c in applied]

    def score_ceiling_for(self, scores: dict[str, int]) -> tuple[float, list[str]]:
        """The lowest rubric-score ceiling and every reason that applies."""
        applied = [c for c in self.score_ceilings if c.applies_to(scores)]
        if not applied:
            return 1.0, []
        strictest = min(applied, key=lambda c: c.max_score)
        return strictest.max_score, [
            (c.reason or
             f"{c.criterion} scored at or below {c.at_or_below}").strip()
            for c in applied]


def load_rubric(path: str | Path) -> Rubric:
    """Load and validate a rubric under the single current contract."""
    p = Path(path)
    try:
        raw = p.read_bytes()
        data = yaml.safe_load(raw.decode())
    except (OSError, yaml.YAMLError) as e:
        raise RubricError(f"cannot read rubric {p}: {e}") from e
    if not isinstance(data, dict):
        raise RubricError(f"rubric {p} must be a mapping")
    for key in ("id", "criteria"):
        if key not in data:
            raise RubricError(f"rubric {p} is missing '{key}'")

    known_channels = ("rgb", "base_color", "scene_depth")
    raw_channels = data.get("channels", ["rgb"])
    if (not isinstance(raw_channels, list) or not raw_channels
            or any(not isinstance(value, str) for value in raw_channels)):
        raise RubricError(f"rubric {p} 'channels' must be a non-empty list")
    channels = tuple(raw_channels)
    if len(set(channels)) != len(channels):
        raise RubricError(f"rubric {p} declares a render channel twice")
    unknown_channels = [value for value in channels if value not in known_channels]
    if unknown_channels:
        raise RubricError(
            f"rubric {p} has unknown render channel(s): "
            f"{', '.join(unknown_channels)}")

    criteria, seen = [], set()
    for index, entry in enumerate(data["criteria"]):
        if not isinstance(entry, dict) or "id" not in entry or "question" not in entry:
            raise RubricError(f"criteria[{index}] needs an 'id' and a 'question'")
        if entry["id"] in seen:
            raise RubricError(f"criterion '{entry['id']}' declared twice")
        seen.add(entry["id"])
        weight = float(entry.get("weight", 1.0))
        if weight <= 0:
            raise RubricError(f"criterion '{entry['id']}' needs a positive weight")
        raw_criterion_channels = entry.get("channels", list(channels))
        if (not isinstance(raw_criterion_channels, list)
                or not raw_criterion_channels
                or any(not isinstance(value, str)
                       for value in raw_criterion_channels)):
            raise RubricError(
                f"criterion '{entry['id']}' channels must be a non-empty list")
        criterion_channels = tuple(raw_criterion_channels)
        outside = [value for value in criterion_channels if value not in channels]
        if outside:
            raise RubricError(
                f"criterion '{entry['id']}' uses channel(s) not declared by "
                f"the rubric: {', '.join(outside)}")
        criteria.append(Criterion(
            id=str(entry["id"]), question=str(entry["question"]),
            weight=weight, guidance=str(entry.get("guidance", "")),
            channels=criterion_channels))
    if not criteria:
        raise RubricError(f"rubric {p} declares no criteria")

    ceilings = []
    for index, entry in enumerate(data.get("ceilings") or []):
        if "metric" not in entry or ("above" not in entry and "below" not in entry):
            raise RubricError(
                f"ceilings[{index}] needs a 'metric' and one of 'above' / 'below'")
        max_score = float(entry.get("max_score", 1.0))
        if not 0.0 <= max_score <= 1.0:
            raise RubricError(f"ceilings[{index}].max_score must be within 0..1")
        ceilings.append(Ceiling(
            metric=str(entry["metric"]),
            above=float(entry["above"]) if "above" in entry else None,
            below=float(entry["below"]) if "below" in entry else None,
            max_score=max_score, reason=str(entry.get("reason", ""))))

    score_ceilings = []
    for index, entry in enumerate(data.get("score_ceilings") or []):
        if (not isinstance(entry, dict) or "criterion" not in entry
                or "at_or_below" not in entry):
            raise RubricError(
                f"score_ceilings[{index}] needs 'criterion' and 'at_or_below'")
        criterion = str(entry["criterion"])
        if criterion not in seen:
            raise RubricError(
                f"score_ceilings[{index}] names unknown criterion {criterion!r}")
        threshold = int(entry["at_or_below"])
        scale_max = int(data.get("scale_max", 4))
        if not 0 <= threshold <= scale_max:
            raise RubricError(
                f"score_ceilings[{index}].at_or_below must be within "
                f"0..{scale_max}")
        max_score = float(entry.get("max_score", 1.0))
        if not 0.0 <= max_score <= 1.0:
            raise RubricError(
                f"score_ceilings[{index}].max_score must be within 0..1")
        score_ceilings.append(ScoreCeiling(
            criterion=criterion, at_or_below=threshold,
            max_score=max_score, reason=str(entry.get("reason", ""))))

    return Rubric(id=str(data["id"]), criteria=criteria,
                  scale_max=int(data.get("scale_max", 4)),
                  ceilings=ceilings, score_ceilings=score_ceilings,
                  instructions=str(data.get("instructions", "")),
                  channels=channels)
