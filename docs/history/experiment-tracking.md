# Experiment Tracking Implementation

## Motivation

The evaluation harness (`scripts/evaluate_llm.py` and `/api/evaluate`) printed metrics to stdout and discarded them.  There was no persistent record of evaluation runs across schema changes, model updates, or prompt iterations, making it impossible to track whether a schema edit actually improved F1 or just moved errors around.

Adding experiment tracking provides:
- A time-series of F1 scores across the project's history
- Per-record tables showing exactly which records failed and why
- LLM prompt capture for prompt engineering workflows
- Comparison of model/schema combinations in a browser dashboard

---

## What was built

### `tei_annotator/tracking/`

```
tei_annotator/tracking/
├── __init__.py        # get_available_trackers(), get_tracker(), re-exports
├── base.py            # ExperimentTracker, RunContext, RecordEntry, _NullRunContext
├── wandb_tracker.py   # WandbTracker + WandbRunContext
└── mlflow_tracker.py  # MLFlowTracker + MLFlowRunContext
```

#### Abstract API (`base.py`)

`RecordEntry` is a plain dataclass capturing the per-record data that backends receive:

```python
@dataclass
class RecordEntry:
    idx: int
    snippet: str                     # first 80 chars of plain text (for display)
    micro_f1: float
    missed_tags: list[str]           # element names in unmatched_gold
    spurious_tags: list[str]         # element names in unmatched_pred
    elapsed_seconds: float | None    # wall-clock time for the LLM call
    prompt: str | None               # full LLM prompt, if --log-prompts
```

`RunContext` is the abstract base for an active run.  The key design choice is `_NullRunContext` — a concrete no-op subclass used when tracking is disabled.  This lets `run_evaluation()` always write:

```python
ctx = tracker.start_run(...) if tracker else _NullRunContext()
with ctx:
    ...
    ctx.log_record(entry)
    ctx.log_summary(overall)
```

without any `if tracker:` guards in the loop body.  The `with ctx:` pattern calls `_finish()` on exit via `__exit__`, which flushes the artifact table and closes the run.

`ExperimentTracker` is the abstract factory.  `is_available()` checks environment variables; if not set the tracker is silently skipped.

#### W&B backend (`wandb_tracker.py`)

- **Enabled by**: `WANDB_API_KEY`
- **Optional env**: `WANDB_PROJECT` (default `"tei-annotator"`), `WANDB_ENTITY`
- `log_summary()` calls `wandb.log()` with scalar metrics plus `element/{tag}/f1` per element type
- `_finish()` builds a `wandb.Table` with the accumulated records and logs it as `per_record`
- Both `elapsed_seconds` and `prompt` columns are conditional — they appear only when at least one record carries the value (`has_elapsed = any(r.elapsed_seconds is not None ...)`)
- Package: `wandb>=0.17` installed via `uv sync --extra wandb`

#### MLFlow backend (`mlflow_tracker.py`)

- **Enabled by**: `MLFLOW_TRACKING_URI`; supports GitLab-hosted instances (`https://gitlab.com/api/v4/projects/{id}/ml/mlflow`)
- **Auth**: `MLFLOW_TRACKING_TOKEN` (bearer token / GitLab PAT); MLFlow also reads `MLFLOW_TRACKING_USERNAME` + `MLFLOW_TRACKING_PASSWORD` natively
- **Optional env**: `MLFLOW_EXPERIMENT_NAME` (default `"tei-annotator"`)
- `log_summary()` calls `mlflow.log_metrics()` with the same metric set as W&B
- `_finish()` writes the per-record table as a CSV to a temp file and uploads it via `mlflow.log_artifact()`
- Package: `mlflow>=2.12` installed via `uv sync --extra mlflow`

**Note on `uv sync` behaviour**: `uv sync --extra wandb` makes the virtual environment match *exactly* what was requested — any extra not listed is removed.  To install both backends: `uv sync --extra wandb --extra mlflow`.

---

### CLI additions (`scripts/evaluate_llm.py`)

Three new flags:

| Flag | Description |
|------|-------------|
| `--track {wandb,mlflow,all}` | Enable tracking; `all` uses the first available backend |
| `--run-name NAME` | Custom run name; default: `{schema}-{provider}-{timestamp}` |
| `--log-prompts` | Capture the LLM prompt per record into the artifact table |

`run_evaluation()` gained a `tracker`, `run_name`, and `log_prompts` parameter.  The tracking run is opened at the top and the `with ctx:` block wraps the entire evaluation loop.

Prompt capture works via `ctx.wrap_call_fn(call_fn)`: the wrapper stores the last prompt string in `ctx._last_prompt` before each LLM call.  After `evaluate_element()` returns, `_last_prompt` holds the prompt used for that record (or the last chunk's prompt for multi-chunk records).

---

### Webservice integration (`webservice/main.py`)

- `EvaluateRequest` gained a `track: bool = False` field
- `_run_evaluation()` gains a `track: bool` parameter; when `True`, it iterates over `get_available_trackers()` and opens a run context per tracker
- `/api/config` response includes a `tracking_backends` list so the frontend knows which backends are active

---

### Inference timing (`elapsed_seconds`)

Wall-clock time for the LLM call is captured at the point closest to the actual inference:

- **CLI, batch_size=1**: `evaluate_element()` in `evaluator.py` wraps the `annotate()` call with `time.monotonic()` and sets `eval_result.elapsed_seconds = round(_elapsed, 3)`
- **CLI, batch_size>1**: `_evaluate_batch()` in `evaluate_llm.py` divides total batch wall-clock time by the number of non-empty records, giving a per-record *estimate*
- **Webservice, batch_size=1**: `_evaluate_one()` in `main.py` wraps `annotate()` and sets `result.elapsed_seconds`
- **Webservice, batch_size>1**: `_evaluate_batch_group()` in `main.py` divides total batch time by record count

In the trackers, `log_summary()` computes `mean_elapsed_seconds` and `total_elapsed_seconds` from the accumulated `RecordEntry` list (not from the aggregate `EvaluationResult`, which intentionally leaves `elapsed_seconds=None`).

**Bug fixed during integration**: The webservice's internal evaluation functions (`_evaluate_one`, `_evaluate_batch_group`) called `compute_metrics()` directly and never set `elapsed_seconds` on the result, so it was always `None`.  The tracker's `has_elapsed` check then evaluated to `False` and silently omitted all timing output.  Fixed by adding `time.monotonic()` timing to both functions and assigning `result.elapsed_seconds`.

---

### Tests (`tests/test_tracking.py`)

All tests use mocking; no real W&B or MLFlow calls are made.  Key techniques:

- **Module-level mock**: `patch.dict(sys.modules, {"wandb": MagicMock()})` rather than `patch("wandb.init")`, because `wandb` is imported inside methods and may not be installed in CI
- **`_NullRunContext` isolation**: tests that exercise the null path require no mocking at all
- **Timing tests**: two records with known `elapsed_seconds` values; assert `mean_elapsed_seconds ≈ expected` using `pytest.approx`

Tests added:
- `test_record_entry_fields` — default `elapsed_seconds=None`
- `test_record_entry_with_elapsed` — `elapsed_seconds` round-trips correctly
- `test_wandb_log_summary_includes_timing_metrics` — mean/total appear when records have elapsed
- `test_wandb_log_summary_no_timing_when_absent` — mean/total absent when no records have elapsed
- `test_wandb_finish_includes_elapsed_column_when_present` — column present in table

---

## What is logged per run

| Item | W&B | MLFlow |
|------|-----|--------|
| Run parameters (schema, provider, model, …) | `wandb.init(config=…)` | `mlflow.log_params()` |
| Scalar metrics | `wandb.log()` | `mlflow.log_metrics()` |
| Per-record table | `wandb.Table` named `per_record` | CSV artifact under `evaluation/` |
| `elapsed_seconds` column | conditional | conditional |
| `prompt` column | conditional (`--log-prompts`) | conditional |

Scalar metrics logged: `micro_precision`, `micro_recall`, `micro_f1`, `micro_tp`, `micro_fp`, `micro_fn`, `macro_precision`, `macro_recall`, `macro_f1`, `element/{tag}/precision`, `element/{tag}/recall`, `element/{tag}/f1`, `mean_elapsed_seconds`, `total_elapsed_seconds`.
