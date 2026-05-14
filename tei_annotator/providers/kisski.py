"""KISSKI academic cloud connector."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

from .base import Connector, _RateLimiter, _post_json


class KISSKIConnector(Connector):
    """KISSKI academic cloud — OpenAI-compatible chat completions."""

    _DEFAULT_BASE_URL = "https://chat-ai.academiccloud.de/v1"
    # Fallback used when the /models endpoint is unreachable.
    _FALLBACK_MODELS = ["llama-3.3-70b-instruct", "mistral-large-instruct"]

    _rate_limiter = _RateLimiter(rate_per_minute=10)

    def __init__(self) -> None:
        self._cached_models: list[str] | None = None
        self._cached_model_data: list[dict] = []

    @property
    def id(self) -> str:
        return "kisski"

    @property
    def name(self) -> str:
        return "KISSKI"

    @property
    def description(self) -> str:
        return "KISSKI academic cloud models via OpenAI-compatible API (requires KISSKI_API_KEY)."

    def is_available(self) -> bool:
        return bool(os.environ.get("KISSKI_API_KEY"))

    def models(self) -> list[str]:
        if self._cached_models is None:
            self._cached_models = self._fetch_models()
        return list(self._cached_models)

    def standard_models(self) -> list[str]:
        # All KISSKI models are standard (no premium tier).
        return self.models()

    def _fetch_models(self) -> list[str]:
        """Fetch model list from /models and keep only chat-capable ones."""
        api_key = os.environ.get("KISSKI_API_KEY", "")
        base_url = os.environ.get("KISSKI_BASE_URL", self._DEFAULT_BASE_URL)
        req = urllib.request.Request(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            entries = [
                m for m in data.get("data", [])
                if "text" in m.get("input", [])
                and any(o in m.get("output", []) for o in ("text", "thought"))
            ]
            self._cached_model_data = entries
            return [m["id"] for m in entries]
        except Exception:
            self._cached_model_data = []
            return list(self._FALLBACK_MODELS)

    def get_model_statuses(self) -> list[dict]:
        """Return demand metrics from /models response or KISSKI_STATUS_URL if configured."""
        if self._cached_models is None:
            self.models()
        statuses = [
            {"model": m["id"], "demand": int(m.get("demand", 0)), "status": m.get("status", "")}
            for m in self._cached_model_data
            if "demand" in m
        ]
        if statuses:
            return statuses
        status_url = os.environ.get("KISSKI_STATUS_URL")
        if not status_url:
            return []
        api_key = os.environ.get("KISSKI_API_KEY", "")
        try:
            return _post_json(
                status_url,
                {},
                {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=5,
            ).get("data", [])
        except Exception:
            return []

    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        api_key = os.environ.get("KISSKI_API_KEY", "")
        base_url = os.environ.get("KISSKI_BASE_URL", self._DEFAULT_BASE_URL)
        url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        def call_fn(prompt: str) -> str:
            self._rate_limiter.acquire()
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            result = _post_json(url, payload, headers, timeout)
            return result["choices"][0]["message"]["content"]

        call_fn.__name__ = f"kisski/{model_id}"
        return call_fn
