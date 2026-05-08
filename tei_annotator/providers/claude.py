"""Anthropic Claude connector."""

from __future__ import annotations

import os
from typing import Callable

from .base import Connector, _post_json


class ClaudeConnector(Connector):
    """Anthropic Claude via the Messages API."""

    _BASE_URL = "https://api.anthropic.com/v1/messages"
    _API_VERSION = "2023-06-01"
    _MODELS = [
        "claude-haiku-4-5-20251001",
        "*claude-sonnet-4-6",
        "*claude-opus-4-6",
    ]

    @property
    def id(self) -> str:
        return "claude"

    @property
    def name(self) -> str:
        return "Anthropic Claude"

    @property
    def description(self) -> str:
        return "Anthropic Claude models via the Messages API (requires ANTHROPIC_API_KEY)."

    def is_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": self._API_VERSION,
        }

        def call_fn(prompt: str) -> str:
            payload = {
                "model": model_id,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
            result = _post_json(self._BASE_URL, payload, headers, timeout)
            return result["content"][0]["text"]

        call_fn.__name__ = f"claude/{model_id}"
        return call_fn
