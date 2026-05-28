"""
base.py — Abstract base classes for experiment tracking.

ExperimentTracker   — factory that creates RunContext objects.
RunContext          — active run; accumulates records, flushes on exit.
RecordEntry         — per-record data logged during evaluation.
_NullRunContext     — no-op implementation used when tracking is disabled.

Usage pattern in run_evaluation():

    ctx = tracker.start_run(run_name, params) if tracker else _NullRunContext()
    with ctx:
        if log_prompts:
            call_fn = ctx.wrap_call_fn(call_fn)
        for record in records:
            result = evaluate(record)
            ctx.log_record(RecordEntry(...))
        ctx.log_summary(overall_result)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tei_annotator.evaluation.metrics import EvaluationResult


@dataclass
class RecordEntry:
    """Per-record data captured during an evaluation run."""

    idx: int
    snippet: str                   # first 80 chars of plain text (for display)
    micro_f1: float
    missed_tags: list[str]         # element names in unmatched_gold
    spurious_tags: list[str]       # element names in unmatched_pred
    elapsed_seconds: float | None = None   # wall-clock time for the LLM call
    prompt: str | None = None      # full LLM prompt, if --log-prompts


class RunContext(ABC):
    """
    Represents an active experiment run.  Use as a context manager.

    Concrete subclasses must implement:
    - log_record(entry)    — called after each evaluated record
    - log_summary(result)  — called once with the aggregate EvaluationResult
    - _finish()            — called on __exit__ to flush to the backend

    The default wrap_call_fn() stores the last intercepted prompt in
    self._last_prompt so callers can pass it to log_record().
    """

    def __init__(self) -> None:
        self._last_prompt: str | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def log_record(self, entry: RecordEntry) -> None:
        """Accumulate a single record's evaluation result."""

    @abstractmethod
    def log_summary(self, result: "EvaluationResult") -> None:
        """Log aggregate metrics after the evaluation loop ends."""

    @abstractmethod
    def _finish(self) -> None:
        """Flush accumulated data to the backend and close the run."""

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self._finish()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[tracking] Error finalising run: {exc}", stacklevel=2)
        return None  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Prompt capture helper
    # ------------------------------------------------------------------

    def wrap_call_fn(self, fn: Callable[[str], str]) -> Callable[[str], str]:
        """
        Return a wrapped call_fn that stores the last prompt in
        self._last_prompt before each LLM call.

        For records that result in multiple chunks the last chunk's prompt
        is retained; callers read self._last_prompt immediately after
        evaluate_element() returns.
        """
        ctx = self

        def _wrapped(prompt: str) -> str:
            ctx._last_prompt = prompt
            return fn(prompt)

        _wrapped.__name__ = getattr(fn, "__name__", "call_fn")
        return _wrapped


class _NullRunContext(RunContext):
    """No-op RunContext used when tracking is disabled (tracker=None)."""

    def log_record(self, entry: RecordEntry) -> None:
        pass

    def log_summary(self, result: "EvaluationResult") -> None:
        pass

    def _finish(self) -> None:
        pass


class ExperimentTracker(ABC):
    """
    Factory for RunContext objects.  One ExperimentTracker instance per
    backend (W&B, MLFlow, …); subclass and register in __init__.py.

    Required environment variables are declared in each subclass's
    is_available() method.  If not set, the tracker is silently skipped.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Short machine-readable ID, e.g. 'wandb', 'mlflow'."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name, e.g. 'Weights & Biases'."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True iff all required env vars are present."""

    @abstractmethod
    def start_run(self, run_name: str, params: dict) -> RunContext:
        """
        Initialise a new experiment run and return its RunContext.

        Parameters
        ----------
        run_name : str
            Display name for this run in the tracking UI.
        params : dict
            Hyper-parameters / configuration to log (schema, model, etc.).
        """
