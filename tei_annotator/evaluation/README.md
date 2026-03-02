# Evaluation

The `evaluation` module measures how accurately the annotator reproduces a hand-annotated (gold-standard) TEI XML file. It computes **precision**, **recall**, and **F1 score** per element type and in aggregate.

---

## Evaluation flow

For each gold element (e.g. a `<biblStruct>` record in the gold file):

```
Gold XML element
      │
      ▼
 1. extract_spans()   →  gold EvaluationSpans + plain text   (extractor.py)
      │
      ▼
 2. annotate()        →  predicted XML string                (pipeline.py)
      │
      ▼
 3. extract_spans()   →  predicted EvaluationSpans           (extractor.py)
      │
      ▼
 4. compute_metrics() →  EvaluationResult                    (metrics.py)
```

Because the annotator receives *exactly the same plain text* that the gold spans are anchored to, character offsets align without any additional normalisation.

---

## Span extraction (`extractor.py`)

### `EvaluationSpan`

```python
@dataclass
class EvaluationSpan:
    tag: str    # element name, e.g. "author"
    start: int  # start offset in the element's plain text (inclusive)
    end: int    # end offset (exclusive)
    text: str   # raw text content
```

`normalized_text` is a computed property that collapses runs of internal whitespace to a single space and strips leading/trailing whitespace. It is used by the `TEXT` match mode to compare spans independent of formatting differences.

### `extract_spans(element)`

Walks the lxml element tree depth-first. At each descendant element it records the cumulative plain-text length seen so far as `start`, then adds the element's own text content to get `end`. Tail text (text following a closing tag but before the next sibling's opening tag) is accumulated but not itself attributed to a span.

Returns `(plain_text: str, spans: list[EvaluationSpan])` — the plain text is what `annotate()` will receive; the spans are what it should ideally produce.

---

## Match modes (`metrics.py`)

Three strategies determine when a predicted span "matches" a gold span:

| Mode | Match condition |
|------|-----------------|
| `EXACT` | Same element tag **and** identical `(start, end)` offsets |
| `TEXT` *(default)* | Same element tag **and** `normalized_text` is equal |
| `OVERLAP` | Same element tag **and** intersection-over-union of offset ranges ≥ threshold (default 0.5) |

`TEXT` mode is the most useful in practice: it is invariant to small character-position differences (which can arise from whitespace normalisation or fuzzy span matching) while still requiring the annotator to have found the right text.

`EXACT` is strictest and useful for regression testing. `OVERLAP` is the most lenient and is appropriate when partial matches should count as correct.

### Greedy matching algorithm

`match_spans(gold, predicted, mode)`:

1. Enumerate all `(gold_span, predicted_span)` candidate pairs that share the same element tag.
2. Score each pair:
   - `EXACT` / `TEXT` → 1.0 if matching, 0.0 otherwise.
   - `OVERLAP` → IoU value in [0, 1].
3. Sort by score descending, then greedily assign: take the highest-scoring unmatched pair, mark both spans as matched.
4. Return `SpanMatch` objects for matched pairs, plus lists of unmatched gold and unmatched predicted spans.

---

## Metrics (`metrics.py`)

### `ElementMetrics`

Per-element-type counts (`tp`, `fp`, `fn`) and derived rates:

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2 · P · R / (P + R)

### `EvaluationResult`

Aggregates `SpanMatch` objects and per-element `ElementMetrics`. Exposes two averaging strategies:

| Strategy | How computed | When to prefer |
|----------|-------------|----------------|
| **Micro** | Sum TP / FP / FN across all element types, then compute P / R / F1 | Imbalanced element distributions — frequent types dominate the score, which reflects real-world impact |
| **Macro** | Compute P / R / F1 per element type, then average | All element types weighted equally regardless of frequency — highlights performance on rare types |

`result.report()` prints a formatted table with both averages and a per-element breakdown.

### `aggregate(results)`

Merges a list of `EvaluationResult` objects (one per document record) into a single corpus-level result by summing TP / FP / FN counts across all records before computing rates.

---

## High-level API (`evaluator.py`)

### `evaluate_element(element, schema, endpoint, match_mode)`

Evaluates annotation of a single lxml `_Element`. Handles:

- XML parsing errors in the annotated output (caught and counted as zero predictions).
- Literal `<` / `>` characters in the annotated text that are not valid XML tags: `_escape_nonschema_brackets()` escapes any angle bracket that is not part of a known schema element tag, so lxml can parse the result without error.

### `evaluate_file(gold_xml_path, schema, endpoint, match_mode, max_items)`

Evaluates an entire TEI XML file:

1. Parses the XML file with lxml.
2. Finds all first-level child elements of the root.
3. Calls `evaluate_element()` on each, up to `max_items`.
4. Returns `(list[EvaluationResult], EvaluationResult)` — individual results per record and the corpus-level aggregate.

---

## Example output

```
=== Evaluation Results ===
Micro  P=0.821  R=0.754  F1=0.786  (TP=83  FP=18  FN=27)
Macro  P=0.834  R=0.762  F1=0.791

Per-element breakdown:
  author               P=0.923  R=0.960  F1=0.941  (TP=24  FP=2  FN=1)
  biblScope            P=0.750  R=0.600  F1=0.667  (TP=12  FP=4  FN=8)
  date                 P=0.867  R=0.867  F1=0.867  (TP=13  FP=2  FN=2)
  title                P=0.900  R=0.818  F1=0.857  (TP=18  FP=2  FN=4)
  ...
```

The `matched`, `unmatched_gold`, and `unmatched_pred` lists on each `EvaluationResult` are available for detailed error analysis beyond the summary table.

### Terminology

| Term         | Field            | Meaning                                                                   |
|--------------|------------------|---------------------------------------------------------------------------|
| **missed**   | `unmatched_gold` | False negatives — gold spans the model **failed to annotate**             |
| **spurious** | `unmatched_pred` | False positives — predicted spans that have **no counterpart in the gold** |

Missed spans hurt **recall**; spurious spans hurt **precision**.
