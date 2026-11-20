"""Explicit provider selection."""

from __future__ import annotations

from typing import Any

from .base import LLMProvider, ProviderError
from .claude_code import ClaudeCodeProvider
from .opencode_cli import OpenCodeProvider
from .opencode_zen import OpenCodeZenProvider
from .static import StaticProvider

PROVIDERS = ("claude-code", "opencode", "opencode-zen", "static")


def create_provider(name: str, config: dict[str, Any]) -> LLMProvider:
    model = config.get("model")
    timeout = int(config.get("timeout", 900))
    if name == "claude-code":
        return ClaudeCodeProvider(model=model, timeout=timeout, executable=config.get("executable") or "claude")
    if name == "opencode":
        return OpenCodeProvider(model=model, timeout=timeout, executable=config.get("executable") or "opencode")
    if name == "opencode-zen":
        return OpenCodeZenProvider(
            model=model, timeout=timeout,
            api_format=config.get("zen_api_format"), base_url=config.get("zen_base_url"),
        )
    if name == "static":
        return StaticProvider()
    raise ProviderError("config", f"unsupported provider {name!r}; choose one of: {', '.join(PROVIDERS)}")
