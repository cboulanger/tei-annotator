# tei_annotator.tracking

Experiment tracking for evaluation runs.  Logs metrics, per-run parameters, and per-record artifact tables to **Weights & Biases** or **MLFlow** (including GitLab-hosted instances).

---

## Quick-start

### 1. Install the optional package

```bash
uv sync --extra wandb    # Weights & Biases
uv sync --extra mlflow   # MLFlow
```

### 2. Set environment variables

**Weights & Biases:**

| Variable | Required | Default | Description |
|---|---|---|---|
| `WANDB_API_KEY` | ✅ | — | Enables the W&B tracker |
| `WANDB_PROJECT` | | `"tei-annotator"` | W&B project name |
| `WANDB_ENTITY` | | user default | W&B team or user |

**MLFlow / GitLab:**

| Variable | Required | Default | Description |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | ✅ | — | MLFlow tracking server URL (enables tracker) |
| `MLFLOW_TRACKING_TOKEN` | | — | Auth token (GitLab PAT or bearer token) |
| `MLFLOW_EXPERIMENT_NAME` | | `"tei-annotator"` | Experiment display name |

For GitLab-hosted MLFlow, the tracking URI looks like:
```
https://gitlab.com/api/v4/projects/<project-id>/ml/mlflow
```

### 3. Run evaluation with tracking

```bash
# Log to W&B
uv run scripts/evaluate_llm.py \
    --schema bibl --provider gemini --track wandb

# Log to MLFlow, capturing prompts in the artifact table
uv run scripts/evaluate_llm.py \
    --schema bibl --provider gemini --track mlflow --log-prompts

# Log to all configured backends
uv run scripts/evaluate_llm.py \
    --schema bibl --provider gemini --track all

# Custom run name
uv run scripts/evaluate_llm.py \
    --schema bibl --provider gemini --track wandb \
    --run-name "bibl-gemini-flash-v1.8"
```

---

## What is logged

### Parameters (logged at run start)

| Key | Example |
|---|---|
| `schema` | `"bibl"` |
| `provider` | `"Gemini / gemini-2.0-flash"` |
| `match_mode` | `"text"` |
| `batch_size` | `1` |
| `n_records` | `120` |
| `gliner_model` | `"disabled"` |

### Scalar metrics (logged at run end)

`micro_precision`, `micro_recall`, `micro_f1`, `micro_tp`, `micro_fp`, `micro_fn`,
`macro_precision`, `macro_recall`, `macro_f1`,
plus `element/{tag}/precision`, `element/{tag}/recall`, `element/{tag}/f1` for each element type.

### Per-record artifact table

A table with columns: `idx`, `snippet`, `f1`, `missed_tags`, `spurious_tags`.
When `--log-prompts` is passed: an additional `prompt` column with the full LLM prompt.

- **W&B**: logged as a `wandb.Table` named `per_record`
- **MLFlow**: logged as a CSV file artifact under `evaluation/per_record_*.csv`

---

## API reference

```python
from tei_annotator.tracking import get_available_trackers, get_tracker
from tei_annotator.tracking.base import ExperimentTracker, RunContext, RecordEntry

# List all configured trackers
trackers = get_available_trackers()   # → [WandbTracker(), ...]

# Look up by ID
tracker = get_tracker("wandb")        # → WandbTracker instance or None

# Use as a context manager
with tracker.start_run("my-run", params_dict) as ctx:
    for record in records:
        result = evaluate(record)
        ctx.log_record(RecordEntry(
            idx=1,
            snippet=record[:80],
            micro_f1=result.micro_f1,
            missed_tags=[s.element for s in result.unmatched_gold],
            spurious_tags=[s.element for s in result.unmatched_pred],
            prompt=None,  # set to full LLM prompt if --log-prompts
        ))
    ctx.log_summary(aggregate_result)
# __exit__ → flushes table + closes run
```

### Prompt capture

To include the LLM prompt in the artifact table, wrap the `call_fn` before creating
the `EndpointConfig`:

```python
call_fn = ctx.wrap_call_fn(call_fn)
# ... annotate() is called ...
# After each annotation, ctx._last_prompt holds the most recent prompt
```

`wrap_call_fn()` returns a wrapped function that records the last prompt in
`ctx._last_prompt`, overwritten on each call.  For chunked records, this is the
last chunk's prompt.

---

## Adding a new tracker

1. Create `tei_annotator/tracking/mytracker.py`
2. Subclass `RunContext` and `ExperimentTracker` from `base.py`
3. Implement `log_record()`, `log_summary()`, `_finish()`, `id`, `name`, `is_available()`, `start_run()`
4. Add your instance to `_ALL_TRACKERS` in `__init__.py`

All backend calls must be wrapped in `try/except` so evaluation is never interrupted by
tracking failures.  Use `warnings.warn()` to surface errors to the user.
