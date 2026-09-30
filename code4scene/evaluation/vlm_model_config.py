"""The one model configuration used by every Code4Scene VLM call.

This lives with shared evaluation machinery because RequirementGraph Stage 0,
Stage 2/3, repair diagnostics, and the holistic visual verifier all consume it.
Keeping it below ``evaluation/verifiers`` would make shared code import the
verifier registry and creates a circular import during offline authoring.

The judge model and its decoding parameters are fixed by the benchmark (the
paper used a self-hosted Qwen3.8-27B served through an OpenAI-compatible
endpoint). The release ships NO endpoint: the serving location is always
supplied by the user, either through the ``code4scene`` CLI flags or through
the environment variables below. The endpoint is transport, not scoring
policy, so it is recorded in provenance but never frozen into a policy file.

Environment:

``CODE4SCENE_VLM_BASE_URL``
    OpenAI-compatible base URL of the judge (for example ``http://host:port/v1``).
``CODE4SCENE_VLM_MODEL``
    Served model name. Defaults to ``Qwen/Qwen3.8-27B``.
``CODE4SCENE_VLM_API_KEY``
    Optional bearer token for the judge and embedding endpoints.
``CODE4SCENE_EMBED_BASE_URL`` / ``CODE4SCENE_EMBED_MODEL``
    OpenAI-compatible embedding endpoint used only by the report-only GT
    caption-similarity diagnostic.
"""

from __future__ import annotations

import os

BACKEND = "openai_compat"
BASE_URL_ENV = "CODE4SCENE_VLM_BASE_URL"
MODEL_ENV = "CODE4SCENE_VLM_MODEL"
API_KEY_ENV = "CODE4SCENE_VLM_API_KEY"
EMBED_BASE_URL_ENV = "CODE4SCENE_EMBED_BASE_URL"
EMBED_MODEL_ENV = "CODE4SCENE_EMBED_MODEL"

#: The judge model used for every paper result.
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
#: Public name of the embedding model used by the report-only caption metric.
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
#: No endpoint is shipped. An empty value means "not configured"; a client
#: that needs the judge fails with :class:`JudgeEndpointNotConfigured`.
DEFAULT_BASE_URL = ""


class JudgeEndpointNotConfigured(RuntimeError):
    """A VLM-dependent leaf was requested but no endpoint was supplied."""


def _normalized(value: str | None) -> str:
    return str(value or "").strip().rstrip("/")


def base_url() -> str:
    """The judge endpoint from the environment, or ``""`` when unset."""

    return _normalized(os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL))


def model() -> str:
    return str(os.environ.get(MODEL_ENV) or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def api_key() -> str:
    """Optional secret; never part of provenance."""

    return os.environ.get(API_KEY_ENV, "")


def embed_base_url() -> str:
    return _normalized(os.environ.get(EMBED_BASE_URL_ENV, ""))


def embed_model() -> str:
    return str(os.environ.get(EMBED_MODEL_ENV) or DEFAULT_EMBED_MODEL).strip()


def require_base_url(value: str | None = None) -> str:
    """Return a usable endpoint or explain how to configure one."""

    selected = _normalized(value) or base_url()
    if not selected:
        raise JudgeEndpointNotConfigured(
            "this leaf needs the VLM judge; pass --vlm-base-url or set "
            f"{BASE_URL_ENV} (score with --no-vlm to skip VLM-dependent leaves)"
        )
    return selected


#: Import-time snapshots kept for modules that record configuration in their
#: provenance. The CLI sets the environment before importing evaluation code.
BASE_URL = base_url()
MODEL = model()
STRUCTURED_OUTPUT = True
MAX_TOKENS = 16000
TIMEOUT_S = 600.0
TEMPERATURE = 0.0
SEED = 0
ENABLE_THINKING = False

#: The judge behind every paper result: model and decoding parameters.
PAPER_JUDGE = {"model": DEFAULT_MODEL, "structured_output": True, "max_tokens": 16000,
               "temperature": 0.0, "seed": 0, "enable_thinking": False}


def judge_differences() -> list[str]:
    """The settings of the configured judge that differ from the paper's."""

    current = {"model": model(), "structured_output": STRUCTURED_OUTPUT,
               "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "seed": SEED,
               "enable_thinking": ENABLE_THINKING}
    return [key for key, value in PAPER_JUDGE.items() if current[key] != value]


__all__ = [
    "API_KEY_ENV",
    "BACKEND",
    "BASE_URL",
    "BASE_URL_ENV",
    "DEFAULT_BASE_URL",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_MODEL",
    "EMBED_BASE_URL_ENV",
    "EMBED_MODEL_ENV",
    "ENABLE_THINKING",
    "JudgeEndpointNotConfigured",
    "MAX_TOKENS",
    "MODEL",
    "MODEL_ENV",
    "PAPER_JUDGE",
    "SEED",
    "STRUCTURED_OUTPUT",
    "TEMPERATURE",
    "TIMEOUT_S",
    "api_key",
    "base_url",
    "embed_base_url",
    "embed_model",
    "judge_differences",
    "model",
    "require_base_url",
]
