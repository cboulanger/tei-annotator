"""
Connector abstraction for LLM inference providers.

Each connector wraps one provider (HuggingFace Router, Gemini, KISSKI …) and
exposes a uniform interface.  Connectors self-report availability based on
whether the required environment variable is set; only available connectors
are exposed to the API and UI.

Adding a new provider: subclass `Connector`, implement the abstract methods,
and append an instance to `_ALL_CONNECTORS`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 300) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class Connector(ABC):
    """Base class for all LLM provider connectors."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Short machine-readable identifier, e.g. 'hf', 'gemini', 'kisski'."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name shown in the UI dropdown group label."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description of the provider."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True iff the required credentials are present in the environment."""

    @abstractmethod
    def models(self) -> list[str]:
        """Return the list of model IDs available for this connector."""

    @property
    def default_model(self) -> str:
        """The model pre-selected in the UI (override to customise)."""
        return self.models()[0]

    @abstractmethod
    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        """Return a call_fn(prompt: str) -> str for the given model."""


# ---------------------------------------------------------------------------
# HuggingFace Inference Router
# ---------------------------------------------------------------------------


class HFConnector(Connector):
    """HuggingFace Inference Router — OpenAI-compatible chat completions."""

    _BASE_URL = "https://router.huggingface.co/v1"

    _MODELS = [
        "meta-llama/Llama-3.1-70B-Instruct",
        "Qwen/Qwen3-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    ]

    @property
    def id(self) -> str:
        return "hf"

    @property
    def name(self) -> str:
        return "HuggingFace Inference Router"

    @property
    def description(self) -> str:
        return "Open models via router.huggingface.co (requires HF_TOKEN)."

    def is_available(self) -> bool:
        return bool(os.environ.get("HF_TOKEN"))

    def models(self) -> list[str]:
        return list(self._MODELS)

    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        token = os.environ.get("HF_TOKEN", "")
        url = f"{self._BASE_URL}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        def call_fn(prompt: str) -> str:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            result = _post_json(url, payload, headers, timeout)
            return result["choices"][0]["message"]["content"]

        call_fn.__name__ = f"hf/{model_id}"
        return call_fn


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GeminiConnector(Connector):
    """Google Gemini via the generateContent REST API."""

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    _MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
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

    def models(self) -> list[str]:
        return list(self._MODELS)

    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        url = f"{self._BASE_URL}/{model_id}:generateContent?key={api_key}"

        def call_fn(prompt: str) -> str:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1},
            }
            result = _post_json(url, payload, {"Content-Type": "application/json"}, timeout)
            return result["candidates"][0]["content"]["parts"][0]["text"]

        call_fn.__name__ = f"gemini/{model_id}"
        return call_fn


# ---------------------------------------------------------------------------
# KISSKI
# ---------------------------------------------------------------------------


class KISSKIConnector(Connector):
    """KISSKI academic cloud — OpenAI-compatible chat completions."""

    _DEFAULT_BASE_URL = "https://chat-ai.academiccloud.de/v1"
    # Fallback used when the /models endpoint is unreachable.
    _FALLBACK_MODELS = ["llama-3.3-70b-instruct", "mistral-large-instruct"]

    def __init__(self) -> None:
        self._cached_models: list[str] | None = None

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
            return [
                m["id"] for m in data.get("data", [])
                if "text" in m.get("input", [])
                and any(o in m.get("output", []) for o in ("text", "thought"))
            ]
        except Exception:
            return list(self._FALLBACK_MODELS)

    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        api_key = os.environ.get("KISSKI_API_KEY", "")
        base_url = os.environ.get("KISSKI_BASE_URL", self._DEFAULT_BASE_URL)
        url = f"{base_url}/chat/completions"
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

        call_fn.__name__ = f"kisski/{model_id}"
        return call_fn


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL_CONNECTORS: list[Connector] = [
    HFConnector(),
    GeminiConnector(),
    KISSKIConnector(),
]


def get_available_connectors() -> list[Connector]:
    """Return all connectors whose required credentials are present."""
    return [c for c in _ALL_CONNECTORS if c.is_available()]


def get_connector(connector_id: str) -> Connector | None:
    """Look up a connector by its id string."""
    return next((c for c in _ALL_CONNECTORS if c.id == connector_id), None)
