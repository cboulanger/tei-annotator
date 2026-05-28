"""
tei_annotator.tracking — Experiment tracking for evaluation runs.

Provides an abstract ExperimentTracker / RunContext API backed by
Weights & Biases (classic wandb SDK) and MLFlow (including GitLab-hosted).

Quick-start
-----------
Trackers are enabled by environment variables:

    # Weights & Biases
    WANDB_API_KEY=<your-key>
    WANDB_PROJECT=tei-annotator        # optional

    # MLFlow / GitLab
    MLFLOW_TRACKING_URI=https://gitlab.com/api/v4/projects/<id>/ml/mlflow
    MLFLOW_TRACKING_TOKEN=<gitlab-pat>
    MLFLOW_EXPERIMENT_NAME=tei-annotator  # optional

Install optional packages:
    uv sync --extra wandb
    uv sync --extra mlflow

Usage
-----
    from tei_annotator.tracking import get_available_trackers, get_tracker
    from tei_annotator.tracking.base import RecordEntry, _NullRunContext

    # all configured trackers
    for t in get_available_trackers():
        with t.start_run("my-run", {"schema": "bibl", ...}) as ctx:
            for record in records:
                result = evaluate(record)
                ctx.log_record(RecordEntry(...))
            ctx.log_summary(overall)

    # single tracker by id
    tracker = get_tracker("wandb")
"""

from .base import ExperimentTracker, RecordEntry, RunContext, _NullRunContext
from .mlflow_tracker import MLFlowTracker
from .wandb_tracker import WandbTracker

__all__ = [
    "ExperimentTracker",
    "RunContext",
    "RecordEntry",
    "_NullRunContext",
    "WandbTracker",
    "MLFlowTracker",
    "get_available_trackers",
    "get_tracker",
]

_ALL_TRACKERS: list[ExperimentTracker] = [
    WandbTracker(),
    MLFlowTracker(),
]


def get_available_trackers() -> list[ExperimentTracker]:
    """Return all trackers whose required env vars are present."""
    return [t for t in _ALL_TRACKERS if t.is_available()]


def get_tracker(tracker_id: str) -> ExperimentTracker | None:
    """Return the tracker with the given id, or None if not found."""
    return next((t for t in _ALL_TRACKERS if t.id == tracker_id), None)
