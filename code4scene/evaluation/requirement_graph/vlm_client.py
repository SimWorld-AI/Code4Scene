"""Native tool-call clients backed by Code4Scene's existing VLM config."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code4scene.evaluation import vlm_model_config as model_config
from code4scene.evaluation.vlm_concurrency import (
    request_with_adaptive_concurrency,
    retryable_http_error,
)

from .existing_llm import LLMMessage, LLMResponse, ToolCall


@dataclass
class _ToolBackend:
    """Minimal shared-endpoint transport without importing verifier registry."""

    model: str
    base_url: str
    api_key: str
    max_tokens: int
    timeout_s: float
    temperature: float
    seed: int | None
    enable_thinking: bool | None

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = model_config.require_base_url(self.base_url)
        url = f"{endpoint}{path}"
        payload = json.dumps(body).encode()

        def post() -> dict[str, Any]:
            request = urllib.request.Request(
                url,
                data=payload,
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode())

        try:
            return request_with_adaptive_concurrency(
                endpoint,
                post,
                retryable=retryable_http_error,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise ValueError(
                f"VLM endpoint returned {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ValueError(
                f"cannot reach the VLM endpoint at {self.base_url}: {exc}"
            ) from exc


def _image_payload(value: Any) -> tuple[bytes, str]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        media_type = media_types.get(path.suffix.casefold())
        if media_type is None:
            raise ValueError(f"unsupported VLM image format: {path.suffix or path.name}")
        return path.read_bytes(), media_type
    if isinstance(value, bytes):
        return value, "image/png"
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise ValueError(
            "array-backed VLM frames need the optional imaging extra: "
            "pip install code4scene"
        ) from error
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("VLM images must be HxWx3 or HxWx4 arrays")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array).convert("RGB")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue(), "image/png"


def _openai_content(message: LLMMessage) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in message.content:
        if block.get("type") == "text":
            result.append({"type": "text", "text": str(block.get("text") or "")})
        elif block.get("type") == "image":
            payload, media_type = _image_payload(block.get("image"))
            encoded = base64.b64encode(payload).decode()
            result.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                }
            )
        else:
            raise ValueError(f"unsupported LLM content block {block.get('type')!r}")
    return result


class OpenAICompatibleToolClient:
    name = "scenebench_openai_compatible_tools"
    _strict_tool_calls = True
    _text_action_mode = False

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.model = backend.model
        # The production pipeline must use the same generation limit as the
        # single packaged VLM deployment.  Stage-specific adapters retain
        # small standalone defaults for isolated callers and tests.
        self.max_tokens = int(backend.max_tokens)

    def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": message.role, "content": _openai_content(message)}
                for message in messages
            ],
            "tools": [
                {"type": "function", "function": dict(tool)} for tool in tools
            ],
        }
        if len(tools) == 1:
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": tools[0]["name"]},
            }
        if self.backend.seed is not None:
            body["seed"] = self.backend.seed
        if self.backend.enable_thinking is not None:
            body["chat_template_kwargs"] = {
                "enable_thinking": self.backend.enable_thinking
            }
        payload = self.backend._post("/chat/completions", body)
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("the VLM endpoint returned no choices")
        choice = choices[0]
        if choice.get("finish_reason") in {"length", "content_filter"}:
            raise ValueError(
                f"VLM request ended with {choice.get('finish_reason')}"
            )
        message = choice.get("message") or {}
        parsed: list[ToolCall] = []
        for index, value in enumerate(message.get("tool_calls") or ()):
            function = value.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool-call arguments must be a JSON object")
            parsed.append(
                ToolCall(
                    id=str(value.get("id") or f"tool_{index}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                )
            )
        return LLMResponse(
            text=message.get("content") if isinstance(message.get("content"), str) else None,
            tool_calls=parsed,
            reasoning=message.get("reasoning"),
            usage=dict(payload.get("usage") or {}),
            raw=payload,
        )


def tool_client_from_env() -> OpenAICompatibleToolClient:
    backend = _ToolBackend(
        model=model_config.MODEL,
        base_url=model_config.base_url(),
        api_key=model_config.api_key(),
        max_tokens=model_config.MAX_TOKENS,
        timeout_s=model_config.TIMEOUT_S,
        temperature=model_config.TEMPERATURE,
        seed=model_config.SEED,
        enable_thinking=model_config.ENABLE_THINKING,
    )
    return OpenAICompatibleToolClient(backend)


__all__ = [
    "OpenAICompatibleToolClient",
    "tool_client_from_env",
]
