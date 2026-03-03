---
name: optimize-element-descriptions
description: Iteratively improve TEIElement descriptions in _build_schema() to maximise F1 against the gold standard. Use when annotation quality is low or when evaluation shows missed or spurious spans.
disable-model-invocation: true
argument-hint: [--max-items N] [--provider gemini|kisski|all]
allowed-tools: Read, Edit, Bash
---

# optimize-element-descriptions

Iteratively improve the `TEIElement` descriptions in `scripts/evaluate_llm.py::_build_schema()` to maximise F1 score against the gold standard.

Before writing any descriptions, read the guidelines in [docs/tei-element-descriptions.md](../../../docs/tei-element-descriptions.md).

Extra arguments passed to this skill (e.g. `--max-items 10 --provider gemini`) are forwarded to `evaluate_llm.py` where applicable.

---

## Workflow

### Step 1 — Baseline evaluation

Run a full evaluation with `--verbose` and `--match-mode overlap` to capture missed and spurious spans for every failing record:

```bash
uv run scripts/evaluate_llm.py --verbose --match-mode overlap $ARGUMENTS
```

Record the overall Micro F1, per-element F1, and the text of the lowest-scoring records.

---

### Step 2 — Diagnose failure patterns

For each record where F1 < 1.0, analyse the `missed=` and `spurious=` lists alongside the Gold and Annotation lines shown by `--verbose`.

Group failures into patterns such as:

| Pattern | Typical cause |
|---|---|
| Span emitted as wrong element (spurious + missed same text) | Conflicting or missing negative constraint in description |
| Required parent span missing (e.g. `author` around `orgName`) | Parent–child relationship not described from both sides |
| Multiple instances merged into one span | No explicit "one span per …" instruction |
| Span boundary includes surrounding punctuation | Span boundary not specified in description |
| Positional trigger missed (e.g. editor after "in") | Contextual keyword triggers absent from description |

Focus on patterns that affect **multiple records or both models**: single-record anomalies may be gold-standard issues, not description issues.

---

### Step 3 — Improve descriptions

Read `scripts/evaluate_llm.py` to see the current descriptions, then edit `_build_schema()` following the guidelines in [docs/tei-element-descriptions.md](../../../docs/tei-element-descriptions.md).

Key principles (summary):
- Phrase everything as "emit a span", not "wrap in a tag"
- State multiplicity explicitly: "a separate span for each distinct …"
- Describe parent–child direction from both sides with a concrete example
- Add negative constraints: "never tag X as Y"
- Include textual triggers (keywords, position) and inline surface-form examples
- Prefix critical constraints with `CRITICAL:`

Only edit descriptions for elements where you identified a clear failure pattern.

---

### Step 4 — Targeted re-evaluation with `--grep`

Build a grep pattern from the text of the failing records identified in Step 1, then re-run only those records:

```bash
uv run scripts/evaluate_llm.py --verbose --match-mode overlap \
    --grep "pattern1|pattern2|..." $ARGUMENTS
```

Compare the new F1 values against the Step 1 baseline for each affected record.

---

### Step 5 — Decide: iterate or stop

**Iterate (go to Step 2)** if:
- At least one record improved and no regressions were introduced, AND
- Remaining failures still show patterns addressable by description changes

**Stop** if any of the following apply:
- No improvement across two consecutive rounds
- Remaining failures appear to be gold-standard annotation issues (flag these for human review)
- Failures are caused by model-level reasoning limits that description changes cannot fix (e.g. a model consistently ignoring a rule that is already clearly stated)

---

### Step 6 — Full re-evaluation (final)

Once iterations are complete, run a full evaluation without `--grep` to confirm that overall F1 has not regressed on records that were previously correct:

```bash
uv run scripts/evaluate_llm.py --verbose --match-mode overlap $ARGUMENTS
```

Report the final Micro F1 and per-element breakdown, noting which elements improved and which remain problematic.
