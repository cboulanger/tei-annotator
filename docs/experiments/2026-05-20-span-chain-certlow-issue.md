# Span-chain `cert="low"` evaluation limitation

**Date:** 2026-05-20  
**Schema:** bibl-reference-segmenter  
**Related record:** Group 2 listBibl (Robins/Morgan/Althusser footnotes)

---

## The problem

The `cert="low"` union-match mechanism in `compute_metrics()` only handles **adjacent pairs**:

> Find every pair `(G1, G2)` where `G2.cert == "low"` AND both are still unmatched.  
> Union text = `" ".join((G1.text + G2.text).split())`.  
> Match against unmatched predicted spans.

When a model merges **three or more** consecutive references into a single span, no union matches fire, so all merged sub-spans score as FN + the merged predicted span as FP.

---

## Concrete failure

Gold (Group 2, record with Simon Robins):

```xml
<bibl><label>1</label> Simon Robins, ... (Routledge 2013); </bibl>
<bibl cert="low">Pauline Boss, ... (Harvard University Press 2000);<lb/> </bibl>
<bibl cert="low">Iosif Kovras, ... (Cambridge University Press 2017).<lb/> </bibl>
```

Both Gemini and KISSKI **always** produce a single merged bibl:

```xml
<bibl><label>1</label> Simon Robins, ... Boss, ... Kovras, ...</bibl>
```

What happens in the evaluator:

1. **Standard greedy pass:** The merged bibl overlaps G1 (Robins) at IoU ≈ 33% — below the 0.5 threshold — so nothing matches.
2. **Union-match pass:**
   - Pair (G1, G2=Boss cert=low): union = Robins+Boss text ≠ merged-all text → no match.
   - Pair (G2=Boss, G3=Kovras cert=low): union = Boss+Kovras text ≠ merged-all text → no match.
3. Result: merged bibl = 1 FP, G1/G2/G3 = 3 FN. F1 ≈ 0.42 for that record.

The same pattern applies to the **Morgan group** (5 sub-references, label 3) in the same listBibl — no cert="low" marks there at all, so the 5-way merge is entirely uncompensated.

---

## Proposed fix: span-chain union match

Extend `compute_metrics()` in [tei_annotator/evaluation/metrics.py](../../tei_annotator/evaluation/metrics.py) with a **chain pass** after the existing pair pass:

1. Collect all maximal runs of consecutive unmatched gold spans where every span after the first has `cert="low"`.  
   Example: `[G1, G2(cert=low), G3(cert=low)]` is one chain.
2. For each chain of length ≥ 2, compute the full union text: `" ".join("".join(g.text for g in chain).split())`.
3. Search unmatched predicted spans for a span whose `normalized_text` equals the full union text.
4. If found: credit **all** spans in the chain as TP and remove the merged predicted span from FP.

This makes the evaluator accept any contiguous merge of cert="low"-bounded sub-references as correct, not just two-span merges.

### Edge cases to handle

- Chain must start with a span **without** `cert="low"` (the anchor).
- All subsequent spans in the chain must have `cert="low"` and be adjacent with no tail text.
- Only the longest chain match should be tried if nested subsets also exist.
- The fix is purely in the evaluator; no schema or gold-file changes needed beyond what already exists.

---

## Files to change

| File | Change |
|------|--------|
| [tei_annotator/evaluation/metrics.py](../../tei_annotator/evaluation/metrics.py) | Add chain union-match pass after existing pair pass in `compute_metrics()` |
| [tests/test_evaluation.py](../../tests/test_evaluation.py) | Add test cases for 3-way and 5-way chain merges |

---

## Current workaround

`cert="low"` was applied to the Boss and Kovras sub-references (commit `52bbc73`). This helps when a model merges exactly two of the three (pair match fires), but does nothing when all three are merged. The Morgan group (5 sub-references) is uncompensated entirely.
