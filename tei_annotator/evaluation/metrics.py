"""
metrics.py — Precision, recall, and F1 for TEI span annotation.

The core idea is simple:
- A *true positive* (TP) is a predicted span that matches a gold span.
- A *false positive* (FP) is a predicted span with no matching gold span.
- A *false negative* (FN) is a gold span with no matching predicted span.

Three match modes are supported (``MatchMode``):
- EXACT  — same element tag + identical character offsets
- TEXT   — same element tag + identical normalised text content (default)
- OVERLAP — same element tag + intersection-over-union ≥ *overlap_threshold*

Matching is greedy: candidate pairs are sorted by score (highest first) and
each gold / predicted span is consumed at most once.

Metrics are reported both per-element-type and aggregated:
- *micro* — aggregate TP/FP/FN counts across all types, then compute P/R/F1
- *macro* — average P/R/F1 across element types present in the gold set
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .extractor import EvaluationSpan


class MatchMode(Enum):
    EXACT = "exact"      # same element + exact (start, end) offsets
    TEXT = "text"        # same element + normalised text content
    OVERLAP = "overlap"  # same element + IoU ≥ overlap_threshold


@dataclass
class SpanMatch:
    """A matched (gold, predicted) pair."""

    gold: EvaluationSpan
    pred: EvaluationSpan
    score: float = 1.0   # 1.0 for exact/text, IoU for overlap mode


@dataclass
class ElementMetrics:
    """Counts and derived metrics for a single element type."""

    element: str
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvaluationResult:
    """
    Full evaluation result: matched pairs, unmatched spans, and metrics.

    Attributes
    ----------
    matched : list[SpanMatch]
        True positives — gold/predicted pairs that were matched.
    unmatched_gold : list[EvaluationSpan]
        False negatives — gold spans not found in the prediction.
    unmatched_pred : list[EvaluationSpan]
        False positives — predicted spans absent from the gold.
    per_element : dict[str, ElementMetrics]
        Per-element-type breakdown.
    """

    matched: list[SpanMatch]
    unmatched_gold: list[EvaluationSpan]   # false negatives
    unmatched_pred: list[EvaluationSpan]   # false positives
    per_element: dict[str, ElementMetrics] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Micro-averaged (aggregate raw counts, then compute rates)
    # ------------------------------------------------------------------

    @property
    def micro_tp(self) -> int:
        return sum(m.true_positives for m in self.per_element.values())

    @property
    def micro_fp(self) -> int:
        return sum(m.false_positives for m in self.per_element.values())

    @property
    def micro_fn(self) -> int:
        return sum(m.false_negatives for m in self.per_element.values())

    @property
    def micro_precision(self) -> float:
        denom = self.micro_tp + self.micro_fp
        return self.micro_tp / denom if denom else 0.0

    @property
    def micro_recall(self) -> float:
        denom = self.micro_tp + self.micro_fn
        return self.micro_tp / denom if denom else 0.0

    @property
    def micro_f1(self) -> float:
        p, r = self.micro_precision, self.micro_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    # ------------------------------------------------------------------
    # Macro-averaged (average per-element rates; only gold-present types)
    # ------------------------------------------------------------------

    @property
    def macro_precision(self) -> float:
        vals = [
            m.precision
            for m in self.per_element.values()
            if (m.true_positives + m.false_negatives) > 0
        ]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_recall(self) -> float:
        vals = [
            m.recall
            for m in self.per_element.values()
            if (m.true_positives + m.false_negatives) > 0
        ]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_f1(self) -> float:
        vals = [
            m.f1
            for m in self.per_element.values()
            if (m.true_positives + m.false_negatives) > 0
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def report(self, title: str = "Evaluation Results") -> str:
        """Return a human-readable summary string."""
        lines = [
            f"=== {title} ===",
            f"Micro  P={self.micro_precision:.3f}  R={self.micro_recall:.3f}"
            f"  F1={self.micro_f1:.3f}"
            f"  (TP={self.micro_tp}  FP={self.micro_fp}  FN={self.micro_fn})",
            f"Macro  P={self.macro_precision:.3f}  R={self.macro_recall:.3f}"
            f"  F1={self.macro_f1:.3f}",
            "",
            "Per-element breakdown:",
        ]
        for elem, m in sorted(self.per_element.items()):
            lines.append(
                f"  {elem:<20}  P={m.precision:.3f}  R={m.recall:.3f}"
                f"  F1={m.f1:.3f}"
                f"  (TP={m.true_positives}  FP={m.false_positives}"
                f"  FN={m.false_negatives})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iou(gold: EvaluationSpan, pred: EvaluationSpan) -> float:
    """Intersection-over-union for two character-offset spans."""
    inter_start = max(gold.start, pred.start)
    inter_end = min(gold.end, pred.end)
    if inter_end <= inter_start:
        return 0.0
    inter = inter_end - inter_start
    union = max(gold.end, pred.end) - min(gold.start, pred.start)
    return inter / union if union > 0 else 0.0


def _pair_score(
    gold: EvaluationSpan,
    pred: EvaluationSpan,
    mode: MatchMode,
    overlap_threshold: float,
) -> float:
    """Score a (gold, pred) candidate pair; 0.0 means no match."""
    if gold.element != pred.element:
        return 0.0
    if mode == MatchMode.EXACT:
        return 1.0 if (gold.start == pred.start and gold.end == pred.end) else 0.0
    if mode == MatchMode.TEXT:
        return 1.0 if gold.normalized_text == pred.normalized_text else 0.0
    if mode == MatchMode.OVERLAP:
        iou = _iou(gold, pred)
        return iou if iou >= overlap_threshold else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_spans(
    gold_spans: list[EvaluationSpan],
    pred_spans: list[EvaluationSpan],
    mode: MatchMode = MatchMode.TEXT,
    overlap_threshold: float = 0.5,
) -> tuple[list[SpanMatch], list[EvaluationSpan], list[EvaluationSpan]]:
    """
    Greedily match *gold_spans* to *pred_spans*.

    Each gold span and each predicted span is matched at most once.
    Candidate pairs are ranked by score (highest first); ties are broken
    by gold index then pred index (i.e. document order).

    Parameters
    ----------
    gold_spans, pred_spans :
        Lists of :class:`EvaluationSpan` to compare.
    mode :
        Matching criterion (EXACT / TEXT / OVERLAP).
    overlap_threshold :
        Minimum IoU required for OVERLAP mode (ignored for other modes).

    Returns
    -------
    (matched, unmatched_gold, unmatched_pred)
        ``matched``         — list of :class:`SpanMatch` (true positives)
        ``unmatched_gold``  — false negatives
        ``unmatched_pred``  — false positives
    """
    # Build all candidate pairs with non-zero score
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold_spans):
        for pi, p in enumerate(pred_spans):
            score = _pair_score(g, p, mode, overlap_threshold)
            if score > 0.0:
                candidates.append((score, gi, pi))

    # Greedy assignment: highest score first
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    matched: list[SpanMatch] = []

    for score, gi, pi in candidates:
        if gi not in matched_gold and pi not in matched_pred:
            matched.append(SpanMatch(gold=gold_spans[gi], pred=pred_spans[pi], score=score))
            matched_gold.add(gi)
            matched_pred.add(pi)

    unmatched_gold = [g for i, g in enumerate(gold_spans) if i not in matched_gold]
    unmatched_pred = [p for i, p in enumerate(pred_spans) if i not in matched_pred]

    return matched, unmatched_gold, unmatched_pred


def compute_metrics(
    gold_spans: list[EvaluationSpan],
    pred_spans: list[EvaluationSpan],
    mode: MatchMode = MatchMode.TEXT,
    overlap_threshold: float = 0.5,
) -> EvaluationResult:
    """
    Compute precision, recall, and F1 by matching gold against predicted spans.

    Parameters
    ----------
    gold_spans :
        Ground-truth spans extracted from a manually annotated XML element.
    pred_spans :
        Spans extracted from the annotator's XML output.
    mode :
        How to decide whether a (gold, pred) pair is a match.
    overlap_threshold :
        Minimum IoU for OVERLAP mode (ignored otherwise).

    Returns
    -------
    :class:`EvaluationResult`
        Contains matched pairs, unmatched spans, and per-element metrics.
    """
    matched, unmatched_gold, unmatched_pred = match_spans(
        gold_spans, pred_spans, mode, overlap_threshold
    )

    # Collect all element types seen in gold or pred
    all_elements: set[str] = set()
    for s in gold_spans:
        all_elements.add(s.element)
    for s in pred_spans:
        all_elements.add(s.element)

    tp: dict[str, int] = {e: 0 for e in all_elements}
    fp: dict[str, int] = {e: 0 for e in all_elements}
    fn: dict[str, int] = {e: 0 for e in all_elements}

    for m in matched:
        tp[m.gold.element] += 1
    for s in unmatched_gold:
        fn[s.element] += 1
    for s in unmatched_pred:
        fp[s.element] += 1

    per_element = {
        elem: ElementMetrics(
            element=elem,
            true_positives=tp[elem],
            false_positives=fp[elem],
            false_negatives=fn[elem],
        )
        for elem in sorted(all_elements)
    }

    return EvaluationResult(
        matched=matched,
        unmatched_gold=unmatched_gold,
        unmatched_pred=unmatched_pred,
        per_element=per_element,
    )


def aggregate(results: list[EvaluationResult]) -> EvaluationResult:
    """
    Merge a list of per-record :class:`EvaluationResult` objects into one.

    Per-element TP/FP/FN counts are summed; the ``matched`` and ``unmatched``
    lists are concatenated.  This is the correct way to compute corpus-level
    micro/macro metrics without re-running span matching across records.
    """
    all_matched = [m for r in results for m in r.matched]
    all_unmatched_gold = [s for r in results for s in r.unmatched_gold]
    all_unmatched_pred = [s for r in results for s in r.unmatched_pred]

    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}

    for r in results:
        for elem, m in r.per_element.items():
            tp[elem] = tp.get(elem, 0) + m.true_positives
            fp[elem] = fp.get(elem, 0) + m.false_positives
            fn[elem] = fn.get(elem, 0) + m.false_negatives

    per_element = {
        elem: ElementMetrics(
            element=elem,
            true_positives=tp[elem],
            false_positives=fp.get(elem, 0),
            false_negatives=fn.get(elem, 0),
        )
        for elem in sorted(tp.keys())
    }

    return EvaluationResult(
        matched=all_matched,
        unmatched_gold=all_unmatched_gold,
        unmatched_pred=all_unmatched_pred,
        per_element=per_element,
    )
