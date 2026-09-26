"""Compatibility import for the shared, canonical VLM configuration."""

from ...vlm_model_config import (  # noqa: F401
    BACKEND,
    BASE_URL,
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    ENABLE_THINKING,
    MAX_TOKENS,
    MODEL,
    SEED,
    STRUCTURED_OUTPUT,
    TEMPERATURE,
    TIMEOUT_S,
    JudgeEndpointNotConfigured,
    api_key,
    base_url,
    require_base_url,
)

__all__ = [
    "BACKEND", "BASE_URL", "BASE_URL_ENV", "DEFAULT_BASE_URL",
    "ENABLE_THINKING", "MAX_TOKENS", "MODEL", "SEED",
    "STRUCTURED_OUTPUT", "TEMPERATURE", "TIMEOUT_S",
    "JudgeEndpointNotConfigured", "api_key", "base_url", "require_base_url",
]
