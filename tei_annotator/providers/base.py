"""Shared utilities and abstract base class for LLM connectors."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable

_DEBUG_HTTP = os.environ.get("TEI_DEBUG_HTTP", "") == "1"


def _dbg(msg: str) -> None:
    if _DEBUG_HTTP:
        print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class ProviderTimeoutError(RuntimeError):
    """Raised when a provider's HTTP call exceeds its timeout."""

    def __init__(self, url: str, timeout: float, elapsed: float) -> None:
        super().__init__(
            f"Provider did not respond within {timeout:.0f}s "
            f"(waited {elapsed:.0f}s) — {url}"
        )
        self.url = url
        self.timeout = timeout
        self.elapsed = elapsed


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


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    """POST JSON with a hard total-elapsed timeout.

    urllib's ``timeout=`` is a per-read inactivity timeout — a server that
    trickles back bytes never trips it.  We enforce a total wall-clock cap
    via a daemon-thread watchdog and raise :class:`ProviderTimeoutError`
    when it fires.
    """
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    _dbg(f"_post_json POST {url} (timeout={timeout}s, body={len(body)} bytes)")
    t0 = time.monotonic()

    result: dict = {}

    def _worker() -> None:
        try:
            # urllib's inactivity timeout slightly above the watchdog so
            # the watchdog wins the race and produces a clean message.
            with urllib.request.urlopen(req, timeout=timeout + 30) as resp:
                result["status"] = resp.status
                result["raw"] = resp.read()
        except BaseException as exc:  # noqa: BLE001 — pass any failure up
            result["exc"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        elapsed = time.monotonic() - t0
        _dbg(f"_post_json WATCHDOG TIMEOUT after {elapsed:.1f}s (limit={timeout}s)")
        # The daemon thread keeps running; it will either complete on its
        # own or die with the process. urllib will time out at timeout+30s.
        raise ProviderTimeoutError(url, timeout, elapsed)

    exc = result.get("exc")
    if exc is not None:
        elapsed = time.monotonic() - t0
        if isinstance(exc, urllib.error.HTTPError):
            detail = exc.read().decode(errors="replace")
            _dbg(f"_post_json HTTPError {exc.code} after {elapsed:.1f}s: {detail[:200]}")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        if isinstance(exc, (socket.timeout, TimeoutError)):
            _dbg(f"_post_json TIMEOUT after {elapsed:.1f}s (limit={timeout}s)")
            raise ProviderTimeoutError(url, timeout, elapsed) from exc
        if isinstance(exc, urllib.error.URLError) and isinstance(
            exc.reason, (socket.timeout, TimeoutError)
        ):
            _dbg(f"_post_json TIMEOUT (URLError) after {elapsed:.1f}s")
            raise ProviderTimeoutError(url, timeout, elapsed) from exc
        _dbg(f"_post_json EXC after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
        raise exc

    status, raw = result["status"], result["raw"]
    _dbg(f"_post_json POST {url} -> {status} in {time.monotonic()-t0:.1f}s "
         f"({len(raw)} bytes)")
    return json.loads(raw)


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

    def get_model_statuses(self) -> list[dict]:
        """
        Return [{"model": str, "demand": int, "status": str}, ...].
        demand: 0 = free, 1-2 = moderate, 3+ = busy.
        Returns [] if this provider does not expose status information.
        """
        return []

    @abstractmethod
    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        """Return a call_fn(prompt: str) -> str for the given model."""
