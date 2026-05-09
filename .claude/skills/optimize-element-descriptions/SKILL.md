---
name: optimize-element-descriptions
description: Iteratively improve TEIElement descriptions and schema rules to maximise F1 against the gold standard. Use when annotation quality is low or when evaluation shows missed or spurious spans.
disable-model-invocation: true
argument-hint: "--max-items N --provider gemini|kisski|all"
---

# optimize-element-descriptions

Iteratively improve the `TEIElement` descriptions and `TEISchema.rules` in the relevant schema file under `tei_annotator/schemas/` to maximise F1 score against the gold standard.

Schema files:
- `tei_annotator/schemas/bibl.py` — `build_bibl_schema()`
- `tei_annotator/schemas/bibl_reference_segmenter.py` — `build_bibl_reference_segmenter_schema()`

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

Read the relevant schema file under `tei_annotator/schemas/` to see the current descriptions, then edit the builder function following the guidelines in [docs/tei-element-descriptions.md](../../../docs/tei-element-descriptions.md).

Key principles (summary):
- Phrase everything as "emit a span", not "wrap in a tag"
- State multiplicity explicitly: "a separate span for each distinct …"
- Describe parent–child direction from both sides with a concrete example
- Add negative constraints: "never tag X as Y"
- Include textual triggers (keywords, position) and inline surface-form examples
- Prefix critical constraints with `CRITICAL:`
- If a failure pattern affects **multiple element types**, add the constraint to `TEISchema.rules` instead of duplicating it in each element description — the prompt renders `rules` as a numbered "General Rules" section before all element descriptions.

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
- Remaining failures appear to be gold-standard annotation issues (flag these for human review; see Step 5a)
- Failures are caused by model-level reasoning limits that description changes cannot fix (e.g. a model consistently ignoring a rule that is already clearly stated)

---

### Step 5a — Handle editorial ambiguities with `cert="low"`

If a failure pattern **persists across model families** after two or more rule iterations and the boundary in question reflects a genuine editorial choice (either split or merged would be defensible), do **not** continue iterating on the prompt. Instead, update the gold file:

1. Split the merged gold span into two adjacent spans with **no tail text** between them.
2. Set `cert="low"` on the **second** span.

```xml
<!-- before -->
<bibl><label>5</label> Commentary mentioning Althusser; see Bunn (2015).<lb/> </bibl>

<!-- after -->
<bibl><label>5</label> Commentary mentioning Althusser;</bibl><bibl cert="low">see Bunn (2015).<lb/> </bibl>
```

The evaluator's union-match pass then accepts either model behaviour (split or merged) as correct. See [tei_annotator/evaluation/README.md](../../../tei_annotator/evaluation/README.md#uncertain-boundary-gold-spans-certlow) for the full specification.

---

### Step 6 — Full re-evaluation (final)

Once iterations are complete, run a full evaluation without `--grep` to confirm that overall F1 has not regressed on records that were previously correct:

```bash
uv run scripts/evaluate_llm.py --verbose --match-mode overlap $ARGUMENTS
```

Report the final Micro F1 and per-element breakdown, noting which elements improved and which remain problematic.
