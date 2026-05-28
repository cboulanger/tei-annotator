"""
Unit tests for tei_annotator.tracking.

All tests are fully mocked — no real W&B or MLFlow calls are made.
The wandb and mlflow packages are not required to be installed; they are
mocked at the sys.modules level so lazy imports inside the connectors work.
"""

from __future__ import annotations

import sys
import warnings
from unittest.mock import MagicMock, patch

import pytest

from tei_annotator.tracking import get_available_trackers, get_tracker, _ALL_TRACKERS
from tei_annotator.tracking.base import (
    RecordEntry,
    RunContext,
    _NullRunContext,
)
from tei_annotator.tracking.mlflow_tracker import MLFlowRunContext, MLFlowTracker
from tei_annotator.tracking.wandb_tracker import WandbRunContext, WandbTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_eval_result(micro_f1: float = 0.9, per_element: dict | None = None):
    """Build a minimal EvaluationResult-like mock."""
    r = MagicMock()
    r.micro_precision = micro_f1
    r.micro_recall = micro_f1
    r.micro_f1 = micro_f1
    r.micro_tp = 100
    r.micro_fp = 10
    r.micro_fn = 10
    r.macro_precision = micro_f1
    r.macro_recall = micro_f1
    r.macro_f1 = micro_f1
    r.per_element = per_element or {}
    return r


def _entry(
    idx: int = 1,
    f1: float = 0.9,
    elapsed: float | None = None,
    prompt: str | None = None,
) -> RecordEntry:
    return RecordEntry(
        idx=idx,
        snippet=f"entry {idx}",
        micro_f1=f1,
        missed_tags=["date"] if f1 < 1.0 else [],
        spurious_tags=[],
        elapsed_seconds=elapsed,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# RecordEntry
# ---------------------------------------------------------------------------


def test_record_entry_fields():
    entry = RecordEntry(
        idx=1,
        snippet="Some bibliographic text",
        micro_f1=0.875,
        missed_tags=["date"],
        spurious_tags=["author"],
    )
    assert entry.idx == 1
    assert entry.snippet == "Some bibliographic text"
    assert entry.micro_f1 == 0.875
    assert entry.missed_tags == ["date"]
    assert entry.spurious_tags == ["author"]
    assert entry.elapsed_seconds is None  # default
    assert entry.prompt is None  # default


def test_record_entry_with_elapsed():
    entry = RecordEntry(
        idx=3,
        snippet="Timed entry",
        micro_f1=0.95,
        missed_tags=[],
        spurious_tags=[],
        elapsed_seconds=1.234,
    )
    assert entry.elapsed_seconds == 1.234


def test_record_entry_with_prompt():
    entry = RecordEntry(
        idx=2,
        snippet="Another entry",
        micro_f1=1.0,
        missed_tags=[],
        spurious_tags=[],
        prompt="Please annotate this text…",
    )
    assert entry.prompt == "Please annotate this text…"


# ---------------------------------------------------------------------------
# _NullRunContext
# ---------------------------------------------------------------------------


def test_null_run_context_noop():
    ctx = _NullRunContext()
    ctx.log_record(_entry())
    ctx.log_summary(_fake_eval_result())
    ctx._finish()  # no exception


def test_null_run_context_as_context_manager():
    ctx = _NullRunContext()
    with ctx as c:
        assert c is ctx
        c.log_record(_entry())
    # __exit__ → _finish() is a no-op


def test_null_run_context_wrap_call_fn():
    ctx = _NullRunContext()
    calls = []

    def original(prompt: str) -> str:
        calls.append(prompt)
        return "response"

    wrapped = ctx.wrap_call_fn(original)
    result = wrapped("hello prompt")
    assert result == "response"
    assert calls == ["hello prompt"]
    assert ctx._last_prompt == "hello prompt"


# ---------------------------------------------------------------------------
# RunContext.wrap_call_fn — last-prompt capture
# ---------------------------------------------------------------------------


def test_wrap_call_fn_captures_last_prompt():
    ctx = _NullRunContext()

    def original(prompt: str) -> str:
        return f"response"

    wrapped = ctx.wrap_call_fn(original)
    wrapped("first prompt")
    assert ctx._last_prompt == "first prompt"
    wrapped("second prompt")
    assert ctx._last_prompt == "second prompt"  # last wins


def test_wrap_call_fn_preserves_name():
    ctx = _NullRunContext()

    def my_call_fn(prompt: str) -> str:
        return prompt

    my_call_fn.__name__ = "gemini/gemini-2.0-flash"
    wrapped = ctx.wrap_call_fn(my_call_fn)
    assert wrapped.__name__ == "gemini/gemini-2.0-flash"


# ---------------------------------------------------------------------------
# WandbTracker — availability
# ---------------------------------------------------------------------------


def test_wandb_tracker_not_available_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert not WandbTracker().is_available()


def test_wandb_tracker_available_with_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key-123")
    assert WandbTracker().is_available()


def test_wandb_tracker_id_and_name():
    tracker = WandbTracker()
    assert tracker.id == "wandb"
    assert "Weights" in tracker.name or "W&B" in tracker.name


# ---------------------------------------------------------------------------
# WandbTracker — start_run / log_record / log_summary / _finish
# ---------------------------------------------------------------------------


def _mock_wandb_module():
    """Build a minimal mock for the wandb module."""
    m = MagicMock()
    m.init.return_value = MagicMock()  # mock run
    m.Table.return_value = MagicMock()
    return m


def test_wandb_run_context_accumulates_records(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")

    mock_wandb = _mock_wandb_module()
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracker = WandbTracker()
        ctx = tracker.start_run("test-run", {"schema": "bibl"})

        mock_wandb.init.assert_called_once_with(
            project="test-project",
            entity=None,
            name="test-run",
            config={"schema": "bibl"},
            reinit=True,
        )

        ctx.log_record(_entry(1, 0.9))
        ctx.log_record(_entry(2, 1.0))
        assert len(ctx._records) == 2


def test_wandb_log_summary_calls_wandb_log(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    mock_wandb = _mock_wandb_module()
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracker = WandbTracker()
        ctx = tracker.start_run("run", {})
        ctx.log_summary(_fake_eval_result(micro_f1=0.85))

        mock_wandb.log.assert_called_once()
        logged = mock_wandb.log.call_args[0][0]
        assert logged["micro_f1"] == 0.85
        assert "micro_precision" in logged


def test_wandb_log_summary_includes_timing_metrics(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    mock_wandb = _mock_wandb_module()
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracker = WandbTracker()
        ctx = tracker.start_run("run", {})
        ctx.log_record(_entry(1, 0.9, elapsed=1.2))
        ctx.log_record(_entry(2, 1.0, elapsed=0.8))
        ctx.log_summary(_fake_eval_result(micro_f1=0.95))

        logged = mock_wandb.log.call_args[0][0]
        assert "mean_elapsed_seconds" in logged
        assert abs(logged["mean_elapsed_seconds"] - 1.0) < 0.01
        assert "total_elapsed_seconds" in logged
        assert abs(logged["total_elapsed_seconds"] - 2.0) < 0.01


def test_wandb_log_summary_no_timing_when_absent(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    mock_wandb = _mock_wandb_module()
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracker = WandbTracker()
        ctx = tracker.start_run("run", {})
        # No records with elapsed_seconds
        ctx.log_summary(_fake_eval_result(micro_f1=0.95))

        logged = mock_wandb.log.call_args[0][0]
        assert "mean_elapsed_seconds" not in logged


def test_wandb_finish_builds_table_and_calls_finish(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    mock_wandb = _mock_wandb_module()
    mock_table = MagicMock()
    mock_wandb.Table.return_value = mock_table

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracker = WandbTracker()
        ctx = tracker.start_run("run", {})
        ctx.log_record(_entry(1, 0.9))
        ctx._finish()

        mock_wandb.Table.assert_called_once()
        mock_table.add_data.assert_called_once()
        mock_wandb.finish.assert_called_once()


def test_wandb_finish_includes_elapsed_column_when_present(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    mock_wandb = _mock_wandb_module()
    mock_table = MagicMock()
    mock_wandb.Table.return_value = mock_table

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracker = WandbTracker()
        ctx = tracker.start_run("run", {})
        ctx.log_record(_entry(1, 0.8, elapsed=1.5))
        ctx._finish()

        cols_arg = mock_wandb.Table.call_args[1]["columns"]
        assert "elapsed_seconds" in cols_arg


def test_wandb_finish_includes_prompt_column_when_present(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    mock_wandb = _mock_wandb_module()
    mock_table = MagicMock()
    mock_wandb.Table.return_value = mock_table

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracker = WandbTracker()
        ctx = tracker.start_run("run", {})
        ctx.log_record(_entry(1, 0.8, prompt="Please annotate…"))
        ctx._finish()

        cols_arg = mock_wandb.Table.call_args[1]["columns"]
        assert "prompt" in cols_arg


def test_wandb_context_manager_calls_finish_on_exit(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    mock_wandb = _mock_wandb_module()
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracker = WandbTracker()
        with tracker.start_run("run", {}):
            pass
        mock_wandb.finish.assert_called_once()


# ---------------------------------------------------------------------------
# MLFlowTracker — availability
# ---------------------------------------------------------------------------


def test_mlflow_tracker_not_available_without_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert not MLFlowTracker().is_available()


def test_mlflow_tracker_available_with_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://gitlab.com/api/v4/projects/1/ml/mlflow")
    assert MLFlowTracker().is_available()


def test_mlflow_tracker_id_and_name():
    tracker = MLFlowTracker()
    assert tracker.id == "mlflow"
    assert "MLFlow" in tracker.name or "mlflow" in tracker.name.lower()


# ---------------------------------------------------------------------------
# MLFlowTracker — start_run / log_record / log_summary / _finish
# ---------------------------------------------------------------------------


def _mock_mlflow_module():
    m = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "run-abc123"
    m.start_run.return_value = mock_run
    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp-1"
    m.get_experiment_by_name.return_value = mock_exp
    return m


def test_mlflow_start_run_sets_uri_and_logs_params(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.com/mlflow")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")

    mock_mlflow = _mock_mlflow_module()
    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        tracker = MLFlowTracker()
        params = {"schema": "bibl", "provider": "gemini"}
        tracker.start_run("my-run", params)

        mock_mlflow.set_tracking_uri.assert_called_once_with("https://example.com/mlflow")
        mock_mlflow.log_params.assert_called_once_with(params)


def test_mlflow_log_summary_calls_log_metrics(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.com/mlflow")

    mock_mlflow = _mock_mlflow_module()
    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        tracker = MLFlowTracker()
        ctx = tracker.start_run("run", {})
        ctx.log_summary(_fake_eval_result(micro_f1=0.75))

        mock_mlflow.log_metrics.assert_called_once()
        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert logged["micro_f1"] == 0.75


def test_mlflow_finish_logs_artifact_and_ends_run(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.com/mlflow")

    mock_mlflow = _mock_mlflow_module()
    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        tracker = MLFlowTracker()
        ctx = tracker.start_run("run", {})
        ctx.log_record(_entry(1, 0.8))
        ctx._finish()

        mock_mlflow.log_artifact.assert_called_once()
        mock_mlflow.end_run.assert_called_once()


def test_mlflow_creates_experiment_when_not_found(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.com/mlflow")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "new-exp")

    mock_mlflow = _mock_mlflow_module()
    mock_mlflow.get_experiment_by_name.return_value = None  # experiment doesn't exist yet
    mock_mlflow.create_experiment.return_value = "new-exp-id"

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        tracker = MLFlowTracker()
        tracker.start_run("run", {})
        mock_mlflow.create_experiment.assert_called_once_with("new-exp")


def test_mlflow_context_manager_calls_end_run(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.com/mlflow")

    mock_mlflow = _mock_mlflow_module()
    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        tracker = MLFlowTracker()
        with tracker.start_run("run", {}):
            pass
        mock_mlflow.end_run.assert_called_once()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_get_tracker_returns_correct_instance():
    t = get_tracker("wandb")
    assert t is not None
    assert t.id == "wandb"

    t2 = get_tracker("mlflow")
    assert t2 is not None
    assert t2.id == "mlflow"


def test_get_tracker_returns_none_for_unknown():
    assert get_tracker("nonexistent_backend_xyz") is None


def test_get_available_trackers_respects_env(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    available = get_available_trackers()
    assert all(t.is_available() for t in available)
    assert len(available) == 0


def test_get_available_trackers_with_wandb_key(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "some-key")
    available = get_available_trackers()
    ids = [t.id for t in available]
    assert "wandb" in ids
    assert "mlflow" not in ids


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


def test_finish_exception_is_warned_not_raised(monkeypatch):
    """_finish() swallows backend exceptions and emits a warning, never raises."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    mock_wandb = _mock_wandb_module()
    # Make wandb.finish() fail — this is always called even without records
    mock_wandb.finish.side_effect = RuntimeError("W&B connection failed")

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracker = WandbTracker()
        ctx = tracker.start_run("run", {})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Call _finish() directly: exception must be swallowed as a warning
            ctx._finish()

        assert len(caught) >= 1
        messages = " ".join(str(w.message) for w in caught)
        assert any(
            kw in messages.lower()
            for kw in ("tracking", "finish", "failed", "connection")
        )


def test_log_summary_exception_is_warned_not_raised(monkeypatch):
    """log_summary() swallows exceptions and emits a warning."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    mock_wandb = _mock_wandb_module()
    mock_wandb.log.side_effect = RuntimeError("W&B connection failed")

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        tracker = WandbTracker()
        ctx = tracker.start_run("run", {})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctx.log_summary(_fake_eval_result())

        assert len(caught) >= 1
