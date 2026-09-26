"""Provider-neutral structured VLM contracts used by the merged graph engine.

The original graph engine loaded these types from its gym environment at
import time.  Code4Scene owns its own model transport, so the merged engine
keeps only the small data contract here and injects a client from the existing
judge configuration.  This module deliberately has no provider or repository
path dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class LLMMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @classmethod
    def text(cls, role: str, text: str) -> LLMMessage:
        return cls(role=role, content=[{"type": "text", "text": str(text)}])  # type: ignore[arg-type]

    @classmethod
    def user_with_image(cls, text: str, image: Any) -> LLMMessage:
        return cls(
            role="user",
            content=[{"type": "text", "text": str(text)},
                     {"type": "image", "image": image}],
        )

    @classmethod
    def user_with_images(
        cls,
        text: str,
        images: list[Any],
        captions: list[str] | None = None,
    ) -> LLMMessage:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": str(text)}]
        for index, image in enumerate(images):
            if captions and index < len(captions) and captions[index]:
                blocks.append({"type": "text", "text": captions[index]})
            blocks.append({"type": "image", "image": image})
        return cls(role="user", content=blocks)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_action_dict(self) -> dict[str, Any]:
        return {"tool": self.name, "params": self.arguments}


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall]
    reasoning: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    name: str
    model: str

    def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


__all__ = ["LLMClient", "LLMMessage", "LLMResponse", "ToolCall"]
