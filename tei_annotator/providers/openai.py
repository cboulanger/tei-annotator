"""OpenAI connector."""

from __future__ import annotations

import os
from typing import Callable

from .base import Connector, _post_json


class OpenAIConnector(Connector):
    """OpenAI chat completions API."""

    _BASE_URL = "https://api.openai.com/v1"
    _MODELS = [
        "gpt-5-nano",
        "*o4-mini",
    ]

    @property
    def id(self) -> str:
        return "openai"

    @property
    def name(self) -> str:
        return "OpenAI"

    @property
    def description(self) -> str:
        return "OpenAI GPT models via the chat completions API (requires OPENAI_API_KEY)."

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        url = f"{self._BASE_URL}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        def call_fn(prompt: str) -> str:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            result = _post_json(url, payload, headers, timeout)
            return result["choices"][0]["message"]["content"]

        call_fn.__name__ = f"openai/{model_id}"
        return call_fn
