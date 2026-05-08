"""Shared utilities and abstract base class for LLM connectors."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable


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


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 300) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


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
