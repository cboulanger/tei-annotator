# Batch Annotation Experiment

## Background

The evaluation script (`scripts/evaluate_llm.py`) originally called the LLM once per `<bibl>` record — one round-trip per citation. For a gold standard of 162 records this means 162 sequential API calls, creating considerable latency.

The question: **does sending multiple records in a single LLM call degrade annotation quality?**

Research (BatchPrompt, ICLR 2024) suggests batch prompting can maintain competitive accuracy while reducing token overhead by 18–30 %. The primary known risk is the *lost in the middle* effect: models tend to perform less reliably on items that appear in the middle of a long context window.

## Implementation

A `--batch-size N` flag was added to `evaluate_llm.py` (default `1`, fully backward-compatible).

When `batch_size > 1` the evaluator:

1. Collects the plain text of N consecutive `<bibl>` elements.
2. Joins them with a distinctive ASCII sentinel:
   ```
   \n---RECORD|||SEP|||BOUNDARY---\n
   ```
3. Passes the combined text to the existing `annotate()` pipeline as a single call.
4. Splits the annotated XML output on the same sentinel to recover N per-record fragments.
5. Evaluates each fragment against its gold standard independently.

The separator is guaranteed to survive the annotation pass unchanged because `inject_xml()` never modifies text characters — it only inserts XML tags.

## Experiment

- **Model:** Gemini 2.0 Flash
- **Gold records:** first 30 entries from `tests/fixtures/blbl-examples.tei.xml`
- **Match mode:** `text` (normalised whitespace)
- **GLiNER:** disabled

Two runs were executed in parallel:

| Run | `--batch-size` | API calls |
|-----|---------------|-----------|
| Baseline | 1 | 30 |
| Batch-10 | 10 | 3 |

## Results

### Overall metrics

| Metric | Baseline (1) | Batch-10 | Δ |
|--------|-------------|----------|---|
| Micro F1 | 0.825 | **0.847** | +0.022 |
| Micro Precision | 0.807 | **0.825** | +0.018 |
| Micro Recall | 0.843 | **0.870** | +0.027 |
| Macro F1 | 0.754 | **0.779** | +0.025 |
| Wall time | ~2:14 | ~2:21 | +7 s |

Batch-10 was slightly *better* across every aggregate metric — the feared *lost in the middle* degradation did not materialise.

### Per-element breakdown

| Element | Baseline F1 | Batch-10 F1 | Δ |
|---------|------------|------------|---|
| `date` | 0.896 | **1.000** | +0.104 |
| `label` | 0.800 | **1.000** | +0.200 |
| `author` | 0.575 | **0.719** | +0.144 |
| `title` | 0.695 | **0.742** | +0.047 |
| `publisher` | 0.870 | 0.870 | 0 |
| `editor` | 0.800 | 0.800 | 0 |
| `biblScope` | 0.919 | **0.932** | +0.013 |
| `pubPlace` | **0.839** | 0.813 | −0.026 |
| `surname` | **1.000** | 0.980 | −0.020 |
| `forename` | **0.911** | 0.865 | −0.046 |
| `orgName` | 0.250 | 0.154 | −0.096 |
| `note` | 0.250 | 0.250 | 0 |
| `idno` | 1.000 | 1.000 | 0 |

**Gains** are largest for elements whose correct tag can be inferred from context across entries (`date`, `label`, `author`). Seeing a sequence of citations helps the model establish the citation style.

**Regressions** in `forename` and `surname` are minor (−2 to −5 pp). The `orgName` regression is notable but both values are already low — this element is consistently difficult regardless of batch size.

### Latency

Wall time was essentially identical (~2:14 vs ~2:21). For 30 records, latency is dominated by token generation time, not round-trip overhead. The benefit of batching would become visible at scale:

- At 162 records: baseline = 162 calls; batch-10 = 17 calls — meaningful when approaching rate limits.
- Cost savings are proportional to the reduction in calls (shared prompt/schema overhead amortised across N records).

## Conclusions

1. **Batch annotation at size 10 is safe.** Quality is at least as good as single-record annotation on this dataset.
2. **Context helps for some elements.** `date`, `label`, and `author` all improved, likely because the model calibrates its interpretation of citation style from surrounding entries.
3. **No evidence of *lost in the middle* degradation** at batch size 10 (combined text ~1,500–3,000 chars, well within Gemini's context window).
4. **Latency benefit is most relevant at scale** (rate-limit avoidance, cost amortisation) rather than in raw wall time for small runs.

## Recommendation

Use `--batch-size 10` as the default for full evaluation runs. Run a follow-up experiment on the full 162-record gold standard to confirm that quality holds at larger scale and that no records are adversely affected by their position within a batch.
