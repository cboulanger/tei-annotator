"""Ollama local inference connector."""

from __future__ import annotations

import os
from typing import Callable

from .base import Connector, _post_json


class OllamaConnector(Connector):
    """Local Ollama instance via its OpenAI-compatible chat completions API."""

    _MODELS: list[str] = []  # unused — model is read dynamically from env

    @property
    def id(self) -> str:
        return "ollama"

    @property
    def name(self) -> str:
        return "Ollama (local)"

    @property
    def description(self) -> str:
        return "Local Ollama instance (set OLLAMA_BASE_URL to enable)."

    def is_available(self) -> bool:
        return bool(os.environ.get("OLLAMA_BASE_URL"))

    def _model(self) -> str:
        return os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

    def models(self) -> list[str]:
        return [self._model()]

    def standard_models(self) -> list[str]:
        return [self._model()]

    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        def call_fn(prompt: str) -> str:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "stream": False,
            }
            result = _post_json(url, payload, headers, timeout)
            return result["choices"][0]["message"]["content"]

        call_fn.__name__ = f"ollama/{model_id}"
        return call_fn
