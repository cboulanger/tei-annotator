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
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Enforce a minimum interval between calls (thread-safe)."""

    def __init__(self, rate_per_minute: int) -> None:
        self._interval = 60.0 / rate_per_minute
        self._lock = threading.Lock()
        self._last: float = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


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

    # Subclasses declare their model list here.
    # Prefix a model name with '*' to mark it as premium-only.
    # The '*' is stripped before the ID is passed to any API call.
    _MODELS: list[str] = []

    @abstractmethod
    def is_available(self) -> bool:
        """Return True iff the required credentials are present in the environment."""

    def models(self) -> list[str]:
        """Return all model IDs (premium and standard), with '*' stripped."""
        return [m.lstrip("*") for m in self._MODELS]

    def standard_models(self) -> list[str]:
        """Return only non-premium model IDs (those not prefixed with '*')."""
        return [m for m in self._MODELS if not m.startswith("*")]

    @property
    def default_model(self) -> str:
        """The model pre-selected in the UI (override to customise)."""
        return self.standard_models()[0] if self.standard_models() else self.models()[0]

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
        "Qwen/Qwen3-14B",
        "*meta-llama/Llama-3.1-70B-Instruct",
        "*deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
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


# ---------------------------------------------------------------------------
# KISSKI
# ---------------------------------------------------------------------------


class KISSKIConnector(Connector):
    """KISSKI academic cloud — OpenAI-compatible chat completions."""

    _DEFAULT_BASE_URL = "https://chat-ai.academiccloud.de/v1"
    # Fallback used when the /models endpoint is unreachable.
    _FALLBACK_MODELS = ["llama-3.3-70b-instruct", "mistral-large-instruct"]

    _rate_limiter = _RateLimiter(rate_per_minute=10)

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


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIConnector(Connector):
    """OpenAI chat completions API."""

    _BASE_URL = "https://api.openai.com/v1"
    _MODELS = [
        "gpt-5-nano",
        "*o4-mini"
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


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL_CONNECTORS: list[Connector] = [
    HFConnector(),
    GeminiConnector(),
    KISSKIConnector(),
    OpenAIConnector(),
    ClaudeConnector(),
]


def get_available_connectors() -> list[Connector]:
    """Return all connectors whose required credentials are present."""
    return [c for c in _ALL_CONNECTORS if c.is_available()]


def get_connector(connector_id: str) -> Connector | None:
    """Look up a connector by its id string."""
    return next((c for c in _ALL_CONNECTORS if c.id == connector_id), None)
