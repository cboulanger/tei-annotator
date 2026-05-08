"""Google Gemini connector."""

from __future__ import annotations

import os
from typing import Callable

from .base import Connector, _post_json


class GeminiConnector(Connector):
    """Google Gemini via the generateContent REST API."""

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    _MODELS = [
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "*gemini-2.5-pro",
    ]

    @property
    def id(self) -> str:
        return "gemini"

    @property
    def name(self) -> str:
        return "Google Gemini"

    @property
    def description(self) -> str:
        return "Google Gemini models via the generateContent API (requires GEMINI_API_KEY)."

    def is_available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        url = f"{self._BASE_URL}/{model_id}:generateContent?key={api_key}"

        # Disable thinking for 2.5 models (thinkingBudget=0); not valid for older models.
        gen_config: dict = {"temperature": 0.1}
        if "2.5" in model_id:
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}

        def call_fn(prompt: str) -> str:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": gen_config,
            }
            result = _post_json(url, payload, {"Content-Type": "application/json"}, timeout)
            return result["candidates"][0]["content"]["parts"][0]["text"]

        call_fn.__name__ = f"gemini/{model_id}"
        return call_fn
