# Evaluation: KISSKI 4-model comparison — bibl-reference-segmenter

**Date:** 2026-05-08  
**Script:** `scripts/evaluate_llm.py --max-items 5 --verbose --timeout 300`  
**Match mode:** text  
**Batch size:** 1  
**GLiNER:** disabled  
**Reference:** [2026-05-08 Gemini vs KISSKI/Qwen3-Coder](2026-05-08-gemini-kisski-bibl-refseg.md)

---

## Setup

Models were selected from the live `/models` endpoint as the 4 most promising for instruction-following annotation tasks (large parameter count or strong structured-output track record):

| # | Model | ID |
| --- | --- | --- |
| 1 | Mistral Large 3 675B Instruct | `mistral-large-3-675b-instruct-2512` |
| 2 | Qwen 3.5 122B A10B (MoE + thinking) | `qwen3.5-122b-a10b` |
| 3 | DeepSeek R1 Distill Llama 70B (reasoning) | `deepseek-r1-distill-llama-70b` |
| 4 | Gemma 4 31B Instruct | `gemma-4-31b-it` |

Reference results from prior run included for comparison:

| # | Model | ID |
| --- | --- | --- |
| R1 | Gemini 2.0 Flash | `gemini` |
| R2 | Qwen3-Coder 30B A3B | `qwen3-coder-30b-a3b-instruct` |

---

## Summary — bibl-reference-segmenter (5 items)

| Model | Micro F1 | bibl F1 | label F1 | Notes |
| --- | --- | --- | --- | --- |
| **Gemma 4 31B** | **0.889** | 0.936 | **0.824** | Best overall; all 5 records completed |
| Gemini 2.0 Flash *(ref)* | 0.883 | 0.791 | 1.000 | Reference |
| Mistral Large 3 675B | 0.815 | 0.936 | 0.647 | Label boundary errors |
| Qwen 3.5 122B (MoE) | 0.792 | 0.867 | 0.667 | 2/5 records timed out (300 s) |
| Qwen3-Coder 30B *(ref)* | 0.567 | 0.591 | 0.522 | Reference |
| **DeepSeek R1 Distill 70B** | **0.000** | 0.000 | 0.000 | Parse failure on all 5 records |

---

## Per-record F1

| Record | Mistral 675B | Qwen 122B | DeepSeek R1 | Gemma 4 31B |
| --- | --- | --- | --- | --- |
| 1 — Law-Related Education (5 bibls) | 1.000 | *(timeout)* | 0.000 | 1.000 |
| 2 — Robins / multi-citation (8 bibls) | 0.880 | 0.880 | 0.000 | 0.880 |
| 3 — Creed "17." labels (3 bibls) | 0.500 | 0.364 | 0.000 | 0.500 |
| 4 — "[1]" bracket labels (3 bibls) | 0.500 | *(timeout)* | 0.000 | 1.000 |
| 5 — "(1)" paren labels (3 bibls) | 1.000 | 1.000 | 0.000 | 1.000 |

---

## Failure analysis

### F1 — DeepSeek R1: parse failure on every record

**Root cause:** DeepSeek R1 Distill outputs a verbose chain-of-thought block before the JSON answer. The pipeline's JSON parser cannot locate a valid JSON object in the response and discards the entire chunk with `"Could not parse LLM response"`. This yields empty annotations (F1=0.000) across all 5 records — not a schema-understanding failure but a response-format incompatibility.

**Pattern observed:**
```
<think>
…several hundred tokens of reasoning…
</think>

[plain text, no JSON object]
```

**Suggestion:** Add a pre-processing step in the response parser (or in the KISSKI call function) that strips `<think>…</think>` blocks before JSON extraction. Alternatively, pass a `"thinking": {"type": "disabled"}` parameter if the API supports it.

---

### F2 — Label boundary: period/bracket stripped from label text

Both Mistral and Gemma (and the prior Qwen3-Coder run) extract the numeric part of a label without its trailing punctuation:

| Gold | Model output | Effect |
| --- | --- | --- |
| `<label>17.</label>` | `<label>17</label>.` | text mismatch → FP + FN |
| `<label>[1]</label>` | `[<label>1</label>]` | brackets outside span → FP + FN |

This accounts for all 6 missed+spurious label spans on records 3 and 4 for Mistral, and 3 of the 6 for Gemma (Gemma correctly handles `[1]` but not `17.`).

**Suggestion:** Add explicit label-boundary examples to the schema rules: *"The label span MUST include ALL formatting characters that are part of the label — trailing period, brackets, and parentheses — e.g. `<label>17.</label>`, `<label>[1]</label>`, `<label>(1)</label>`."*

---

### F3 — Multi-citation footnote: first bibl wrapper dropped (all models)

Record 2 contains footnotes that cite 3 or 8 works. All models emit one fewer `<bibl>` than gold for the multi-citation block. This is the same failure (F6) documented in the prior run. It persists across model family and size.

**Pattern (record 2, footnote 1):**
- Gold: 3 `<bibl>` spans (label on first)
- All models: 3 `<bibl>` spans — but combined into 2 (first citation merged with second, or first missing wrapper)

**Suggestion:** The schema rule for multi-citation footnotes needs a worked example. This was already flagged as a high-priority action item in the prior run; still unaddressed.

---

### F4 — Qwen 3.5 122B timeouts

Records 1 and 4 timed out at 300 s each. The MoE model activates a thinking pass before responding, significantly increasing latency. The final score (F1=0.792) is computed over only 3 of 5 records and is therefore not directly comparable to others.

**Suggestion:** Increase `--timeout` to 600 s when running `qwen3.5-122b-a10b`, or pass `enable_thinking=false` if the API supports it.

---

## Key findings

1. **Gemma 4 31B is the top KISSKI model** for `bibl-reference-segmenter`, narrowly edging Gemini 2.0 Flash on micro F1 (0.889 vs 0.883). It is also markedly cheaper than Mistral 675B and requires no timeout increase.

2. **DeepSeek R1 Distill is incompatible** with the current pipeline without parser changes. Its reasoning-chain output format prevents any JSON extraction.

3. **Label boundary punctuation** is a consistent weak point across all models (except Gemini 2.0 Flash, which scored F1=1.000 on `label`). A single concrete rule with examples should close most of the gap.

4. **Multi-citation footnote segmentation** remains the hardest case. All models miss at least one `<bibl>` boundary on these records regardless of model size.

---

## Action items

| Priority | Change | Target |
| --- | --- | --- |
| High | Strip `<think>…</think>` blocks before JSON parsing (or disable thinking at call site) | `providers/kisski.py` or `postprocessing/parser.py` |
| High | Add punctuation-inclusive label examples to schema rules | `bibl_reference_segmenter.py` |
| High | Add worked multi-citation footnote example (carry-over from prior run) | `bibl_reference_segmenter.py` |
| Medium | Re-run Qwen 3.5 122B with `--timeout 600` for a valid 5-record comparison | evaluation |
| Low | Confirm whether KISSKI API supports `enable_thinking=false` for Qwen/DeepSeek MoE models | evaluation |
