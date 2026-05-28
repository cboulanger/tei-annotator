"""
wandb_tracker.py — Experiment tracker backed by classic Weights & Biases.

Required env vars:
    WANDB_API_KEY   — enables this tracker

Optional env vars:
    WANDB_PROJECT   — W&B project name (default: "tei-annotator")
    WANDB_ENTITY    — W&B team or user (default: personal workspace)

Install:
    uv sync --extra wandb
    # or: pip install "tei-annotator[wandb]"
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

from .base import ExperimentTracker, RecordEntry, RunContext

if TYPE_CHECKING:
    from tei_annotator.evaluation.metrics import EvaluationResult


class WandbRunContext(RunContext):
    """Active W&B run.  Accumulates records; flushes table on _finish()."""

    def __init__(self, run, log_table: bool = True) -> None:  # run: wandb.sdk.wandb_run.Run
        super().__init__()
        self._run = run
        self._log_table = log_table
        self._records: list[RecordEntry] = []

    def log_record(self, entry: RecordEntry) -> None:
        self._records.append(entry)

    def log_summary(self, result: "EvaluationResult") -> None:
        try:
            import wandb  # noqa: PLC0415

            metrics: dict[str, float] = {
                "micro_precision": result.micro_precision,
                "micro_recall": result.micro_recall,
                "micro_f1": result.micro_f1,
                "micro_tp": result.micro_tp,
                "micro_fp": result.micro_fp,
                "micro_fn": result.micro_fn,
                "macro_precision": result.macro_precision,
                "macro_recall": result.macro_recall,
                "macro_f1": result.macro_f1,
            }
            # Per-element metrics
            for elem, m in result.per_element.items():
                metrics[f"element/{elem}/precision"] = m.precision
                metrics[f"element/{elem}/recall"] = m.recall
                metrics[f"element/{elem}/f1"] = m.f1
            # Inference timing (from accumulated records)
            elapsed = [r.elapsed_seconds for r in self._records if r.elapsed_seconds is not None]
            if elapsed:
                metrics["mean_elapsed_seconds"] = round(sum(elapsed) / len(elapsed), 3)
                metrics["total_elapsed_seconds"] = round(sum(elapsed), 3)

            wandb.log(metrics)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[tracking/wandb] log_summary failed: {exc}", stacklevel=2)

    def _finish(self) -> None:
        try:
            import wandb  # noqa: PLC0415

            if self._log_table and self._records:
                has_prompts = any(r.prompt is not None for r in self._records)
                has_elapsed = any(r.elapsed_seconds is not None for r in self._records)
                columns = ["idx", "snippet", "f1", "missed_tags", "spurious_tags"]
                if has_elapsed:
                    columns.append("elapsed_seconds")
                if has_prompts:
                    columns.append("prompt")

                table = wandb.Table(columns=columns)
                for r in self._records:
                    row = [
                        r.idx,
                        r.snippet,
                        round(r.micro_f1, 4),
                        ", ".join(r.missed_tags),
                        ", ".join(r.spurious_tags),
                    ]
                    if has_elapsed:
                        row.append(r.elapsed_seconds)
                    if has_prompts:
                        row.append(r.prompt or "")
                    table.add_data(*row)

                wandb.log({"per_record": table})

            wandb.finish()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[tracking/wandb] _finish failed: {exc}", stacklevel=2)


class WandbTracker(ExperimentTracker):
    """
    Experiment tracker that logs to classic W&B (wandb.init / wandb.log).

    Enabled when WANDB_API_KEY is set in the environment.
    """

    @property
    def id(self) -> str:
        return "wandb"

    @property
    def name(self) -> str:
        return "Weights & Biases"

    def is_available(self) -> bool:
        return bool(os.environ.get("WANDB_API_KEY"))

    def start_run(self, run_name: str, params: dict) -> WandbRunContext:
        try:
            import wandb  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Run: uv sync --extra wandb"
            ) from exc

        project = os.environ.get("WANDB_PROJECT", "tei-annotator")
        entity = os.environ.get("WANDB_ENTITY") or None

        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=params,
            reinit=True,
        )
        return WandbRunContext(run)
