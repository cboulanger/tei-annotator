# Evaluation: Gemini 2.0 Flash vs KISSKI/Qwen3-Coder — bibl and bibl-reference-segmenter

**Date:** 2026-05-08  
**Script:** `scripts/evaluate_llm.py --max-items 5 --verbose`  
**Match mode:** text  
**Batch size:** 1  
**GLiNER:** disabled

---

## Setup

| Provider | Model | ID |
| --- | --- | --- |
| Google Gemini | gemini-2.0-flash | `gemini` |
| KISSKI | qwen3-coder-30b-a3b-instruct | `kisski` |

Note: KISSKI's default model was fetched live from the `/models` endpoint. The previous hardcoded default was `llama-3.3-70b-instruct`; the live default is now `qwen3-coder-30b-a3b-instruct`. Future runs should pin the model with `--model llama-3.3-70b-instruct` to compare apples-to-apples.

---

## Summary

| Schema | Gemini | KISSKI/Qwen3-Coder |
| --- | --- | --- |
| `bibl` (5 items) | Micro F1 = **0.707** | Micro F1 = **0.732** |
| `bibl-reference-segmenter` (5 items) | Micro F1 = **0.883** | Micro F1 = **0.567** |

---

## bibl schema — per-element breakdown

| Element | Gemini F1 | KISSKI F1 | Notes |
| --- | --- | --- | --- |
| surname | 1.000 | 0.909 | KISSKI missed one (editor without surname) |
| pubPlace | 1.000 | 1.000 | — |
| publisher | 1.000 | 1.000 | — |
| editor | 1.000 | 1.000 | — |
| forename | 0.800 | 0.800 | Gemini: "JJ." → "JJ" (no trailing period); KISSKI same |
| biblScope | 0.800 | 0.667 | KISSKI absorbed trailing "p" into page span |
| date | 0.667 | **0.833** | Gemini wraps parentheses into span: `<date>(2013)</date>` |
| author | 0.545 | 0.545 | **Both** split multi-author spans (see below) |
| orgName | **0.000** | 1.000 | Gemini inverts nesting: `<author>` inside `<orgName>` |
| title | 0.429 | 0.429 | **Both** fragment titles and mishandle `<italic>` (see below) |
| note | — | 0.000 | KISSKI hallucinated `<note type="report">` for "Brochure APMEP n°198" |

---

## bibl-reference-segmenter — per-element breakdown

| Element | Gemini F1 | KISSKI F1 | Notes |
| --- | --- | --- | --- |
| label | **1.000** | 0.522 | KISSKI emits `<bibl>` blocks but leaves labels as bare text |
| bibl | 0.791 | 0.591 | Both fail on multi-citation footnotes; KISSKI also misses labelled wrapping |

---

## Failure analysis

### F1 — `author` (both models, F1=0.545)

**Pattern:** "Russell, D.A. and Michael Winterbottom" is split into two `<author>` spans instead of one.

**Gold:**
```xml
<author><surname>Russell</surname>, <forename>D.A.</forename> and
<forename>Michael</forename> <surname>Winterbottom</surname></author>
```
**Both models produced:** `<author>Russell, D.A.</author> and <author>Michael Winterbottom</author>`

**Rule that should catch this:** "All contiguous authors MUST always be placed inside a SINGLE 'author' span — conjunctions ('and', '&', 'et') and commas between names do NOT create separate spans."

**Suggestion:** The rule is present but the word "contiguous" may be too weak. Restate as a CRITICAL rule with a concrete counter-example: *"'Russell, D.A. and Michael Winterbottom' is ONE author span — the conjunction 'and' is NOT a separator."*

---

### F2 — `title` (both models, F1=0.429)

Two sub-patterns both reduce title F1:

**F2a — Title split at internal period (subtitle separator):**  
"Classical Literary Criticism. Oxford World Classics" should be one `<title level="m">` span.  
Both models split at the period, emitting two title spans or absorbing "Oxford World Classics" as `<pubPlace>` / extra `<title level="s">`.

**Rule that should catch this:** "Do NOT split a title at an internal period or subtitle separator — 'Classical Literary Criticism. Oxford World Classics' is ONE title span."

**Suggestion:** The rule is present in the bibl schema. Neither model follows it reliably. Consider adding a concrete example directly in the rule text, e.g.: *"Example: 'Classical Literary Criticism. Oxford World Classics' → ONE `<title level="m">` span covering the whole string."*

**F2b — `<italic>` text from gold XML becomes an element in predictions:**  
Gold text for Doyle (1998) contains the literal string `&lt;italic&gt;Trends in Plant Science&lt;/italic&gt;` (decoded from an XML entity in the gold file). The annotator sees `<italic>Trends in Plant Science</italic>` in the plain text and wraps the journal title in `<italic>` tags, then `<title level="j">` inside, producing a nested/split result.

**Root cause:** The gold file encodes `<italic>` as an XML entity escape, which `extract_spans()` decodes into literal `<italic>…</italic>` text. The annotator then mis-parses those angle brackets as real tags.

**Suggestion (pre-processing):** Strip or normalise residual formatting markup (`<italic>`, `<bold>`, `<emph>`) from the plain text before passing to the annotator — either in `extract_spans()` or in the pipeline's tag-stripping step. The evaluator already escapes these via `_escape_nonschema_brackets()`, but the plain text still contains the literal `<italic>` string which confuses the LLM.

---

### F3 — `orgName` nesting (Gemini only, F1=0.000)

**Gold:**
```xml
<author><orgName>Commission Inter-IREM Collège</orgName> &amp;
<orgName>Commission Inter-IREM Statistiques et Probabilités</orgName></author>
```
**Gemini produced:**
```xml
<orgName><author>Commission Inter-IREM Collège & Commission Inter-IREM
Statistiques et Probabilités</author></orgName>
```

**Rule that should catch this:** "When an organisation acts as author or editor, emit BOTH an 'orgName' span AND an enclosing 'author' span. The 'author'/'editor' span MUST enclose the 'orgName' span — NEVER put an 'author' or 'editor' span inside an 'orgName' span."

**Suggestion:** The CRITICAL rule is in place. Gemini inverts the nesting despite it. Try reinforcing with a short positive example in the rule: *"Correct: `<author><orgName>…</orgName></author>`. Wrong: `<orgName><author>…</author></orgName>`."* Alternatively, add this as a separate rule closer to the orgName element description.

---

### F4 — `date` with enclosing parentheses (Gemini only)

**Gold:** `(<date>2013</date>)` — parentheses are outside the span.  
**Gemini:** `<date>(2013)</date>` — parentheses absorbed into span.

**Suggestion:** Add a rule: *"Do NOT include surrounding punctuation (parentheses, brackets, commas) inside a 'date' span — only the year or date string itself."*

---

### F5 — `label` omitted when `bibl` is present (KISSKI bibl-reference-segmenter, label F1=0.522)

KISSKI correctly identifies the `<bibl>` boundaries for most records but consistently emits the numeric label as bare text rather than wrapping it in `<label>`. For record 1 (five labelled footnotes: 2, 4, 5, 6, 7), all five `<bibl>` spans were correct but all five `<label>` spans were missing.

**Suggestion:** The label rule is present and detailed. The model may not be attending to it strongly enough given it is focused on the harder `bibl`-segmentation task. Consider elevating `<label>` to the first rule and adding an explicit reminder in the `bibl` element description: *"If the reference begins with a numeric or alphanumeric label, the very first child of the 'bibl' span MUST be a 'label' span."*

---

### F6 — Multi-citation footnotes: first `<bibl>` wrapper dropped (both models)

**Pattern (record 2):** Footnotes like `1. See Doe (2020); Foo (2021); Bar (2022)` should produce three `<bibl>` spans with `<label>1</label>` on the first.

**Gemini:** emitted the label and subsequent `<bibl>` spans correctly, but wrapped the first citation inside those `<bibl>` spans (nested bibl) rather than as a peer bibl. Result: first citation had no bibl wrapper.

**KISSKI:** dropped all bibl wrappers and all label spans for the entire footnote.

**Suggestion:** The rule "CRITICAL: A footnote citing multiple works → MULTIPLE 'bibl' spans, label on FIRST ONLY" is present. Neither model reliably applies it to the first citation in the sequence.

Consider adding a step-by-step worked example directly in the rule text:
```
Input: "1. Doe (2020), 45; Foo (2021), 123."
Output:
  <bibl><label>1</label> Doe (2020), 45;</bibl>
  <bibl>Foo (2021), 123.</bibl>
```
The example makes the structure of the first `<bibl>` (which must contain the `<label>`) unambiguous.

---

## Action items

| Priority | Change | Target |
| --- | --- | --- |
| High | Add counter-example to multi-author CRITICAL rule | `bibl.py` rules |
| High | Add worked example to multi-citation footnote CRITICAL rule | `bibl_reference_segmenter.py` rules |
| High | Elevate `<label>` into `<bibl>` description for ref-segmenter | `bibl_reference_segmenter.py` |
| Medium | Strip residual `<italic>`/`<bold>` markup from plain text before LLM | `pipeline.py` or `extractor.py` |
| Medium | Add "no parentheses in date span" rule | `bibl.py` rules |
| Medium | Add positive/negative example for orgName nesting | `bibl.py` rules |
| Low | Re-run KISSKI with `--model llama-3.3-70b-instruct` to compare against prior default | evaluation |
| Low | Re-run with full fixture (all records) once fast-run issues found above are addressed | evaluation |
