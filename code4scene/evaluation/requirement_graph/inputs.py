"""Input normalization for the standalone scene evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .contracts import SceneBounds


def first_prompt_paragraph(message: str) -> str:
    """Return the first non-empty paragraph from a generation request."""

    normalized = str(message).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("request message is empty")
    paragraphs = re.split(r"\n\s*\n", normalized, maxsplit=1)
    prompt = paragraphs[0].strip()
    if not prompt:
        raise ValueError("request message has no semantic prompt paragraph")
    return prompt


def prompt_from_request_json(path: str | Path) -> str:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"request json not found: {source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid request json {source}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), str):
        raise ValueError(f"request json {source} must contain a string 'message'")
    return first_prompt_paragraph(payload["message"])


def resolve_semantic_prompt(
    *,
    prompt: str | None,
    request_json: str | Path | None,
) -> str:
    """Resolve CLI prompt precedence: explicit prompt wins over request JSON."""

    if prompt is not None and prompt.strip():
        return prompt.strip()
    if request_json is not None:
        return prompt_from_request_json(request_json)
    raise ValueError("provide either --prompt or --request-json")


def parse_scene_bounds(value: str) -> SceneBounds:
    pieces = [part.strip() for part in str(value).split(",")]
    if len(pieces) != 6:
        raise ValueError(
            "scene bounds must contain six comma-separated numbers: "
            "xmin,ymin,zmin,xmax,ymax,zmax"
        )
    try:
        numbers = tuple(float(part) for part in pieces)
    except ValueError as exc:
        raise ValueError("scene bounds must contain only finite numbers") from exc
    return SceneBounds(min_cm=numbers[:3], max_cm=numbers[3:])


def map_run_name(map_path: str) -> str:
    clean = str(map_path).split("?", 1)[0].rstrip("/")
    name = clean.rsplit("/", 1)[-1] or "scene"
    parent = clean.rsplit("/", 2)[-2] if clean.count("/") >= 2 else ""
    return f"{parent}_{name}" if parent and parent != name else name


__all__ = [
    "first_prompt_paragraph",
    "map_run_name",
    "parse_scene_bounds",
    "prompt_from_request_json",
    "resolve_semantic_prompt",
]
