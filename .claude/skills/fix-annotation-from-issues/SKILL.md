---
name: fix-annotation-from-issues
description: Use when a GitHub issue reports a missed, spurious, or wrong-element annotation. Reproduces the failure with debug_annotation.py, diagnoses the root cause, and improves schema descriptions or rules.
disable-model-invocation: true
argument-hint: "--issue N [--provider gemini|kisski|all]"
---

# fix-annotation-from-issues

Fix annotation failures reported in GitHub issues by reproducing them with
`debug_annotation.py`, diagnosing the root cause, and improving `TEIElement`
descriptions or `TEISchema.rules` in the relevant schema file.

This skill is complementary to `optimize-element-descriptions`, which improves
descriptions by maximising F1 against a gold standard. Use this skill when
the failure is known from a user-reported issue rather than from an evaluation
run. The description-writing guidelines and schema file locations are the same —
see [docs/tei-element-descriptions.md](../../../docs/tei-element-descriptions.md)
and [optimize-element-descriptions](../optimize-element-descriptions/SKILL.md).

Extra arguments (e.g. `--provider kisski`) are forwarded to `debug_annotation.py`
where applicable.

---

## Step 1 — Fetch the issue

Pull the issue body and all comments so you have every failing snippet:

```bash
gh issue view $ISSUE --comments
```

Issues may contain:

- **XML source text** — a `<bibl>` block as it appears in the gold file; this
  is the input to the annotator.
- **XML labelled text** — the annotator's actual output; this is what went wrong.
- **Screenshots** — diff images embedded as `<img>` tags. Screenshots cannot be
  read programmatically, but the surrounding text usually describes the error.
  Note their position in the issue for context.

Extract every `<bibl>` block from the issue that demonstrates a failure. Strip
the outer `<bibl>…</bibl>` wrapper to get the raw text the annotator sees.

---

## Step 2 — Reproduce with `debug_annotation.py`

Run the failing snippet through the full pipeline:

```bash
uv run scripts/debug_annotation.py \
    --text "<raw text from issue>" \
    --show-prompt \
    [--provider gemini] [--schema bibl-reference-segmenter]
```

The script prints every pipeline stage. Read **top-to-bottom** and identify the
**first stage** where the output diverges from what is expected:

| Stage | What to look for | Likely fix |
|---|---|---|
| **Parsed spans** | Wrong element, wrong text, or missing span | Improve element description or schema rules |
| **Resolved spans** | Span parsed correctly but not resolved | Prompt or context instructions |
| **Validated spans** | Resolved but rejected | Element name or attribute value list wrong |
| **Final XML** | All spans correct but nesting wrong | `inject_xml` / injector issue |

Only improve schema descriptions or rules for **Parsed spans** failures.

---

## Step 3 — Diagnose the failure pattern

Compare the issue's "source text" against its "labelled text" (or the screenshot
description) and the debug output. Classify the failure:

| Pattern | Typical cause |
|---|---|
| Span emitted as wrong element | Missing negative constraint in description |
| Required span entirely missing | Trigger condition or surface-form example absent |
| Span boundary too wide or too narrow | Boundary rule not stated |
| Multiple references merged into one span | No "one span per …" instruction |
| Parent span missing around child span | Parent–child relationship not described from both sides |
| Span emitted for only part of the reference | Instruction to cover the full reference absent |

Check whether the same pattern appears in **multiple comments or multiple
issues**. A cross-issue pattern belongs in `TEISchema.rules`, not in a single
element description.

---

## Step 4 — Improve descriptions

Read the current schema file, then edit it following the guidelines in
[docs/tei-element-descriptions.md](../../../docs/tei-element-descriptions.md).

Key principles (summary — see the full guidelines for detail):
- Phrase everything as "emit a span", not "wrap in a tag".
- State multiplicity explicitly: "a separate span for each distinct …"
- Describe parent–child relationships from both sides with a concrete example.
- Add negative constraints: "never tag X as Y".
- Include textual triggers (keywords, position) and inline surface-form examples.
- Prefix critical constraints with `CRITICAL:`.
- Cross-element patterns → `TEISchema.rules` (rendered before all element
  descriptions); single-element patterns → the element's description.

Only edit descriptions for elements directly implicated by the failure.

---

## Step 5 — Re-run the debug script

Re-run `debug_annotation.py` on **every failing snippet** extracted in Step 1:

```bash
uv run scripts/debug_annotation.py \
    --text "<same raw text>" \
    [--provider gemini] [--schema bibl-reference-segmenter]
```

Confirm that the parsed spans now match the expected annotation. If the fix
introduced a regression on a different snippet, diagnose and resolve before
continuing.

---

## Step 6 — Check for regressions on the gold standard

Run a targeted evaluation over records whose text overlaps with the fixed
snippet to catch regressions:

```bash
uv run scripts/evaluate_llm.py --verbose --match-mode overlap \
    --grep "keyword_from_fixed_text" [--provider gemini]
```

If overall F1 is unchanged or improved, the fix is safe.

---

## Step 7 — Close or comment on the issue

Once the fix is confirmed:

```bash
# Add a resolution comment
gh issue comment $ISSUE --body "Fixed in <commit hash>: <one-line description of the change>"

# Close the issue if fully resolved
gh issue close $ISSUE
```

If only some examples in the issue are fixed, leave the remaining failing
snippets as a comment and keep the issue open.

---

## When to stop

Stop and flag for human review if:

- The failure persists after two rounds of description changes and the issue
  appears across multiple model families — this may be a fundamental model
  reasoning limit rather than a prompt-quality problem.
- The expected annotation is itself ambiguous (either split or merged would be
  defensible). In that case, consider adding a `cert="low"` span to the gold
  file as described in `optimize-element-descriptions` Step 5a.
