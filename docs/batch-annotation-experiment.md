# Batch Annotation Experiment

## Background

The evaluation script (`scripts/evaluate_llm.py`) originally called the LLM once per `<bibl>` record — one round-trip per citation. For a gold standard of 162 records this means 162 sequential API calls, creating considerable latency.

The question: **does sending multiple records in a single LLM call degrade annotation quality, and does the answer differ by model?**

Research (BatchPrompt, ICLR 2024) suggests batch prompting can maintain competitive accuracy while reducing token overhead by 18–30 %. The primary known risk is the *lost in the middle* effect: models tend to perform less reliably on items that appear in the middle of a long context window.

## Implementation

A `--batch-size N` flag was added to `evaluate_llm.py` (default `1`, fully backward-compatible).
A `--timeout SECONDS` flag controls the HTTP read timeout per API call (default `120`).

When `batch_size > 1` the evaluator:

1. Collects the plain text of N consecutive `<bibl>` elements.
2. Joins them with a distinctive ASCII sentinel:

   ```text
   \n---RECORD|||SEP|||BOUNDARY---\n
   ```

3. Passes the combined text to the existing `annotate()` pipeline as a single call.
4. Splits the annotated XML output on the same sentinel to recover N per-record fragments.
5. Evaluates each fragment against its gold standard independently.
6. Falls back to empty predictions for the entire batch if the separator count mismatches.

The separator is guaranteed to survive the annotation pass unchanged because `inject_xml()` never modifies text characters — it only inserts XML tags. However, very complex prompts can cause a model to omit or duplicate the separator, triggering the fallback.

## Experiment Setup

- **Gold standard:** `tests/fixtures/blbl-examples.tei.xml` (162 `<bibl>` records)
- **Match mode:** `text` (normalised whitespace)
- **GLiNER:** disabled

Two models were tested:

| Model | Provider | Inference speed |
| --- | --- | --- |
| Gemini 2.0 Flash | Google | Fast (~3–6 s/record) |
| llama-3.3-70b-instruct | KISSKI / AcademicCloud | Slow (~14–20 s/record) |

> **Note on concurrency:** Runs against the same KISSKI API key must be **sequential** — concurrent jobs cause GPU resource contention, inflating response times and triggering timeouts.

---

## Results summary

| Model | Batch size | Records | Timeout | Micro F1 | Macro F1 | API calls |
| --- | --- | --- | --- | --- | --- | --- |
| Gemini 2.0 Flash | 1 | 30 | 120 s | 0.825 | 0.754 | 30 |
| Gemini 2.0 Flash | 10 | 30 | 120 s | 0.847 | 0.779 | 3 |
| Gemini 2.0 Flash | 1 | 162 | 120 s | — | — | — |
| Gemini 2.0 Flash | 10 | 162 | 120 s | **0.744** | 0.627 | 17 |
| KISSKI Llama 3.3 70b | 1 | 162 | 120 s | **0.867** | 0.636 | 162 |
| KISSKI Llama 3.3 70b | 10 | 162 | 120 s | timed out | — | — |
| KISSKI Llama 3.3 70b | 10 | 162 | **600 s** | **0.854** | 0.619 | 17 |

> A full Gemini batch-1 run on 162 records has not been executed.

---

## Detailed results: Gemini 2.0 Flash

### 30-record pilot (batch sizes 1 and 10)

| Metric | Batch-1 | Batch-10 | Δ |
| --- | --- | --- | --- |
| Micro F1 | 0.825 | **0.847** | +0.022 |
| Micro Precision | 0.807 | **0.825** | +0.018 |
| Micro Recall | 0.843 | **0.870** | +0.027 |
| Macro F1 | 0.754 | **0.779** | +0.025 |
| API calls | 30 | 3 | −90 % |
| Wall time | ~2:14 | ~2:21 | +7 s |

On the 30-record pilot, batch-10 was slightly *better* across every metric.

### Full 162-record run (batch-10)

| Metric | 30-rec pilot (batch-10) | 162-rec full (batch-10) | Δ |
| --- | --- | --- | --- |
| Micro F1 | 0.847 | **0.744** | −0.103 |
| Micro Precision | 0.825 | 0.799 | −0.026 |
| Micro Recall | 0.870 | 0.696 | **−0.174** |
| Macro F1 | 0.779 | 0.627 | −0.152 |

Quality degraded significantly on the full dataset. Inspection of the worst records reveals that five consecutive entries (#141–145, all within the same batch) scored F1=0.000 with every gold span missed and nothing predicted. This is the **batch split failure** pattern: the model omitted the separator in that prompt, the mismatch fallback triggered, and all 10 records in the batch received empty predictions.

### Per-element breakdown (Gemini, 162 records, batch-10)

| Element | Batch-10 F1 | TP | FP | FN |
| --- | --- | --- | --- | --- |
| `surname` | **0.878** | 258 | 6 | 66 |
| `label` | 0.842 | 8 | 0 | 3 |
| `date` | 0.832 | 131 | 16 | 37 |
| `forename` | 0.820 | 242 | 27 | 79 |
| `pubPlace` | 0.794 | 54 | 15 | 13 |
| `ptr` | 0.750 | 3 | 0 | 2 |
| `biblScope` | 0.758 | 111 | 12 | 59 |
| `publisher` | 0.750 | 45 | 10 | 20 |
| `title` | 0.634 | 149 | 55 | 117 |
| `author` | 0.568 | 96 | 88 | 58 |
| `idno` | 0.667 | 1 | 0 | 1 |
| `note` | 0.359 | 7 | 10 | 15 |
| `orgName` | 0.300 | 6 | 23 | 5 |
| `editor` | 0.448 | 15 | 21 | 16 |

---

## Detailed results: KISSKI llama-3.3-70b-instruct

### Full 162-record run, all batch sizes

| Batch size | Timeout | Records completed | Micro F1 | Macro F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | 120 s | **162 / 162** | **0.867** | **0.636** | Full, reliable |
| 10 | 120 s | 22 / 162 | 0.843\* | 0.672\* | 14/16 batches timed out |
| 50 | 120 s | 0 / 162 | — | — | All timed out |
| 10 | **600 s** | **162 / 162** | 0.854 | 0.619 | Full run with raised timeout |

\* Biased sample — only 3 of 16 batches completed; not representative.

### Batch-1 vs batch-10 (600 s) — full 162 records

| Metric | Batch-1 | Batch-10 (600 s) | Δ |
| --- | --- | --- | --- |
| Micro F1 | **0.867** | 0.854 | −0.013 |
| Micro Precision | 0.876 | 0.869 | −0.007 |
| Micro Recall | **0.858** | 0.839 | −0.019 |
| Macro F1 | **0.636** | 0.619 | −0.017 |
| API calls | 162 | 17 | −90 % |
| Wall time | ~43 min | ~57 min | +14 min |

A small but consistent quality drop across the board. No complete batch failures (no batch scored all-zero), confirming Llama respects the separator more reliably than Gemini on this dataset.

### Per-element comparison (KISSKI, 162 records)

| Element | Batch-1 F1 | Batch-10 F1 | Δ |
| --- | --- | --- | --- |
| `forename` | **0.954** | 0.945 | −0.009 |
| `surname` | **0.983** | 0.972 | −0.011 |
| `date` | **0.949** | 0.913 | −0.036 |
| `editor` | 0.793 | **0.828** | +0.035 |
| `biblScope` | 0.847 | **0.804** | −0.043 |
| `pubPlace` | 0.853 | **0.855** | +0.002 |
| `publisher` | **0.825** | 0.794 | −0.031 |
| `author` | **0.778** | 0.778 | 0 |
| `title` | **0.743** | 0.733 | −0.010 |
| `note` | 0.485 | 0.485 | 0 |
| `orgName` | 0.421 | **0.457** | +0.036 |
| `label` | **0.235** | 0.222 | −0.013 |

Most regressions are small (< 5 pp). `editor` and `orgName` actually improved slightly in batch mode.

---

## Conclusions

### Batch size is viable but model-dependent

| Model | Batch-10 outcome | Recommendation |
| --- | --- | --- |
| Gemini 2.0 Flash | −10 pp F1 at scale; risk of complete batch failure | Use batch-1 for full runs; batch-10 acceptable for quick pilots |
| KISSKI Llama 3.3 70b | −1.3 pp F1; no batch failures; requires 600 s timeout | Use `--batch-size 10 --timeout 600` |

### Key findings

1. **Small batches on small datasets look deceptively good.** Gemini batch-10 on 30 records was *better* than batch-1 (+2 pp). On 162 records it was significantly worse (−10 pp). The 30-record pilot was not representative because it happened to avoid the complex, multi-author batches near the end of the corpus.

2. **Batch failure is catastrophic, not graceful.** When Gemini loses the separator in a complex batch, all 10 records score 0.000 — a hard floor that drags down corpus-level metrics far more than occasional per-record errors.

3. **Llama respects the separator more reliably.** With adequate timeout (600 s), all 162 batches completed and there were no separator-loss events. The quality drop is small (−1.3 pp) and the latency gain is real (162 → 17 API calls).

4. **Wall time does not improve with batching for slow models.** Llama batch-10 took *longer* than batch-1 (~57 min vs ~43 min) because each batch generates ~10× the tokens in a single call. The benefit is fewer API calls, which matters for rate-limit management, not raw speed.

5. **The `--timeout` flag is essential for slow models.** Without `--timeout 600`, Llama batch-10 timed out on 87 % of batches.

## Recommendations

| Scenario | Command |
| --- | --- |
| Gemini, quality-critical full run | `--provider gemini --batch-size 1` |
| Gemini, quick pilot (≤ 30 records) | `--provider gemini --batch-size 10` |
| KISSKI Llama, full run (balanced) | `--provider kisski --batch-size 10 --timeout 600` |
| KISSKI Llama, full run (highest quality) | `--provider kisski --batch-size 1` |
