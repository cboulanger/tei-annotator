"""
mlflow_tracker.py — Experiment tracker backed by MLFlow.

Supports GitLab-hosted MLFlow:
    https://gitlab.com/api/v4/projects/{project_id}/ml/mlflow

Required env vars:
    MLFLOW_TRACKING_URI    — MLFlow tracking server URL (enables this tracker)
    MLFLOW_TRACKING_TOKEN  — authentication token (GitLab PAT or bearer token)
                             Alternatively, use the standard MLFlow vars:
                             MLFLOW_TRACKING_USERNAME + MLFLOW_TRACKING_PASSWORD

Optional env vars:
    MLFLOW_EXPERIMENT_NAME — experiment display name (default: "tei-annotator")

Install:
    uv sync --extra mlflow
    # or: pip install "tei-annotator[mlflow]"
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
import warnings
from typing import TYPE_CHECKING

from .base import ExperimentTracker, RecordEntry, RunContext

if TYPE_CHECKING:
    from tei_annotator.evaluation.metrics import EvaluationResult


class MLFlowRunContext(RunContext):
    """Active MLFlow run.  Accumulates records; flushes CSV artifact on _finish()."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id
        self._records: list[RecordEntry] = []

    def log_record(self, entry: RecordEntry) -> None:
        self._records.append(entry)

    def log_summary(self, result: "EvaluationResult") -> None:
        try:
            import mlflow  # noqa: PLC0415

            metrics: dict[str, float] = {
                "micro_precision": result.micro_precision,
                "micro_recall": result.micro_recall,
                "micro_f1": result.micro_f1,
                "micro_tp": float(result.micro_tp),
                "micro_fp": float(result.micro_fp),
                "micro_fn": float(result.micro_fn),
                "macro_precision": result.macro_precision,
                "macro_recall": result.macro_recall,
                "macro_f1": result.macro_f1,
            }
            # Per-element metrics (MLFlow requires flat metric names)
            for elem, m in result.per_element.items():
                safe = elem.replace("/", "_").replace(" ", "_")
                metrics[f"element_{safe}_precision"] = m.precision
                metrics[f"element_{safe}_recall"] = m.recall
                metrics[f"element_{safe}_f1"] = m.f1
            # Inference timing (from accumulated records)
            elapsed = [r.elapsed_seconds for r in self._records if r.elapsed_seconds is not None]
            if elapsed:
                metrics["mean_elapsed_seconds"] = round(sum(elapsed) / len(elapsed), 3)
                metrics["total_elapsed_seconds"] = round(sum(elapsed), 3)

            mlflow.log_metrics(metrics)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[tracking/mlflow] log_summary failed: {exc}", stacklevel=2)

    def _finish(self) -> None:
        try:
            import mlflow  # noqa: PLC0415

            if self._records:
                has_prompts = any(r.prompt is not None for r in self._records)
                has_elapsed = any(r.elapsed_seconds is not None for r in self._records)
                fieldnames = ["idx", "snippet", "f1", "missed_tags", "spurious_tags"]
                if has_elapsed:
                    fieldnames.append("elapsed_seconds")
                if has_prompts:
                    fieldnames.append("prompt")

                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=fieldnames)
                writer.writeheader()
                for r in self._records:
                    row: dict = {
                        "idx": r.idx,
                        "snippet": r.snippet,
                        "f1": round(r.micro_f1, 4),
                        "missed_tags": ", ".join(r.missed_tags),
                        "spurious_tags": ", ".join(r.spurious_tags),
                    }
                    if has_elapsed:
                        row["elapsed_seconds"] = r.elapsed_seconds
                    if has_prompts:
                        row["prompt"] = r.prompt or ""
                    writer.writerow(row)

                # Write to a temp file for mlflow.log_artifact
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".csv",
                    prefix="per_record_",
                    delete=False,
                    encoding="utf-8",
                ) as fh:
                    fh.write(buf.getvalue())
                    tmp_path = fh.name

                mlflow.log_artifact(tmp_path, artifact_path="evaluation")

                import os  # noqa: PLC0415
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            mlflow.end_run()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[tracking/mlflow] _finish failed: {exc}", stacklevel=2)


class MLFlowTracker(ExperimentTracker):
    """
    Experiment tracker that logs to MLFlow (including GitLab-hosted instances).

    Enabled when MLFLOW_TRACKING_URI is set in the environment.
    Authentication via MLFLOW_TRACKING_TOKEN (bearer) or
    MLFLOW_TRACKING_USERNAME + MLFLOW_TRACKING_PASSWORD.
    """

    @property
    def id(self) -> str:
        return "mlflow"

    @property
    def name(self) -> str:
        return "MLFlow"

    def is_available(self) -> bool:
        return bool(os.environ.get("MLFLOW_TRACKING_URI"))

    def start_run(self, run_name: str, params: dict) -> MLFlowRunContext:
        try:
            import mlflow  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "mlflow is not installed. Run: uv sync --extra mlflow"
            ) from exc

        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        token = os.environ.get("MLFLOW_TRACKING_TOKEN")
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "tei-annotator")

        mlflow.set_tracking_uri(tracking_uri)

        # Bearer-token auth via environment variable (MLFlow reads this natively
        # for HTTP-backed stores when MLFLOW_TRACKING_TOKEN is set)
        if token:
            os.environ.setdefault("MLFLOW_TRACKING_TOKEN", token)

        # Resolve or create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        run = mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment_id,
        )
        mlflow.log_params(params)

        return MLFlowRunContext(run.info.run_id)
