# CLAUDE.md

## Package manager

Uses `uv`. Run tests with `uv run pytest`. Install deps with `uv sync` (add `--extra gliner` or `--extra webservice` for optional extras). API keys go in `.env` (copy from `.env.template`).

---

## Project layout

```
tei_annotator/          core library
  models/               TEIAttribute, TEIElement, TEISchema; SpanDescriptor, ResolvedSpan
  inference/            EndpointConfig, EndpointCapability
  chunking/             chunk_text()
  detection/            detect_spans() — GLiNER pre-detection (needs [gliner] extra)
  prompting/            build_prompt(), make_correction_prompt(); Jinja2 templates
  postprocessing/       parse_response(), resolve_spans(), validate_spans(), inject_xml()
  schemas/              build_bibl_schema(), build_bibl_reference_segmenter_schema()
    registry.py         SCHEMA_REGISTRY — maps schema key → build fn + root/child elements
  providers/            LLM connectors: hf / gemini / kisski / openai / claude
  evaluation/           EvaluationSpan, extract_spans(), compute_metrics(), evaluate_file()
  pipeline.py           annotate() — top-level entry point
  tei.py                create_schema() — parse RNG → TEISchema

scripts/
  evaluate_llm.py       run any provider against a gold-standard TEI file
  smoke_test_llm.py     quick connectivity check
  smoke_test_webservice.py

tests/
  test_*.py             unit tests (fully mocked, < 0.5 s) — run with: uv run pytest
  integration/          real GLiNER / end-to-end tests (excluded from CI by default)
  fixtures/             gold TEI XML files used by evaluate_llm.py and test_evaluation.py

webservice/             FastAPI JSON API + browser UI

docs/                   see Documentation section below
```

---

## Key design rules

- The LLM prompt talks about **spans** (emit a span / cover a span), never XML tags. Schema descriptions must match this vocabulary.
- `SpanDescriptor` is always **flat** — no nesting. `ResolvedSpan.children` is populated later by the injector.
- Source text is **never modified** by any model call.
- Cross-element constraints belong in `TEISchema.rules` (rendered as numbered "General Rules" before element descriptions), not duplicated inside individual element descriptions.

---

## Running the evaluator

```bash
# quick run: 5 records, gemini, bibl-reference-segmenter schema
uv run scripts/evaluate_llm.py \
    --provider gemini --schema bibl-reference-segmenter --max-items 5 --verbose

# re-run only failing records
uv run scripts/evaluate_llm.py --verbose --match-mode overlap \
    --grep "Creed|Robins" --provider kisski

# all providers, all records
uv run scripts/evaluate_llm.py --schema bibl --output-file results.txt
```

Key flags: `--provider`, `--model`, `--schema`, `--gold-file`, `--max-items`,
`--batch-size`, `--match-mode`, `--verbose`, `--grep`, `--shuffle`, `--timeout`.

---

## Skills

**`/optimize-element-descriptions`** — iterative workflow for improving schema prompt rules and element descriptions to maximise F1 against a gold standard. Includes guidance for handling genuinely ambiguous gold boundaries via `cert="low"`. See [.claude/skills/optimize-element-descriptions/SKILL.md](.claude/skills/optimize-element-descriptions/SKILL.md).

---

## Documentation

### Module READMEs

| Path | Topic |
|------|-------|
| [tei_annotator/models/README.md](tei_annotator/models/README.md) | TEISchema, TEIElement, TEIAttribute; SpanDescriptor, ResolvedSpan |
| [tei_annotator/detection/README.md](tei_annotator/detection/README.md) | GLiNER pre-detection |
| [tei_annotator/chunking/README.md](tei_annotator/chunking/README.md) | Text chunking strategy |
| [tei_annotator/prompting/README.md](tei_annotator/prompting/README.md) | Prompt templates and builder |
| [tei_annotator/inference/README.md](tei_annotator/inference/README.md) | EndpointConfig; provider setup examples |
| [tei_annotator/postprocessing/README.md](tei_annotator/postprocessing/README.md) | Parse → resolve → validate → inject pipeline |
| [tei_annotator/schemas/README.md](tei_annotator/schemas/README.md) | Built-in schemas, registry, adding a new schema |
| [tei_annotator/providers/README.md](tei_annotator/providers/README.md) | LLM connectors, adding a new provider |
| [tei_annotator/evaluation/README.md](tei_annotator/evaluation/README.md) | Evaluation flow, match modes, metrics, `cert="low"` uncertain-boundary handling |
| [webservice/README.md](webservice/README.md) | FastAPI webservice setup and API |

### Guides

| Path | Topic |
|------|-------|
| [docs/tei-element-descriptions.md](docs/tei-element-descriptions.md) | Evidence-based guidelines for writing effective TEIElement descriptions |
| [docs/huggingface-deployment.md](docs/huggingface-deployment.md) | Deploying `app.py` to HuggingFace Spaces |

### Experiments

| Path | Summary |
|------|---------|
| [docs/experiments/evaluation-results.md](docs/experiments/evaluation-results.md) | Running evaluation results table across models and schemas |
| [docs/experiments/batch-annotation-experiment.md](docs/experiments/batch-annotation-experiment.md) | Batching multiple records per LLM call to reduce latency |
| [docs/experiments/2026-05-08-gemini-kisski-bibl-refseg.md](docs/experiments/2026-05-08-gemini-kisski-bibl-refseg.md) | Gemini 2.0 Flash vs KISSKI/Qwen3-Coder on bibl and bibl-reference-segmenter |
| [docs/experiments/2026-05-08-kisski-model-comparison-bibl-refseg.md](docs/experiments/2026-05-08-kisski-model-comparison-bibl-refseg.md) | KISSKI 4-model comparison on bibl-reference-segmenter |

### History

| Path | Topic |
|------|-------|
| [docs/history/implementation-plan.md](docs/history/implementation-plan.md) | Original design and implementation plan (historical) |
