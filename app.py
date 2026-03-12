"""
TEI Annotator — Gradio demo for HuggingFace Spaces.

Set HF_TOKEN as a Space secret.  All inference calls use that token;
visitors use the app without any login or token input.

Deploy on HF Spaces (SDK: gradio).  Required secret: HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Config (mirrors webservice/main.py — keep in sync if you change models)
# ---------------------------------------------------------------------------

_HF_TOKEN = os.environ.get("HF_TOKEN", "")

_HF_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct:nscale",
    "meta-llama/Llama-3.1-70B-Instruct:scaleway",
    "Qwen/Qwen3-32B:nscale",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B:nscale",
    "Qwen/Qwen2.5-Coder-32B-Instruct:nscale",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:nscale",
    "meta-llama/Llama-3.1-8B-Instruct:nscale",
    "Qwen/Qwen3-14B:nscale",
    "openai/gpt-oss-20b:nscale",
    "Qwen/QwQ-32B:nscale",
]

_HF_BASE_URL = "https://router.huggingface.co/v1"
_FIXTURE_PATH = Path(__file__).parent / "tests" / "fixtures" / "blbl-examples.tei.xml"

# Build the schema once at startup — it's immutable and expensive-ish to rebuild per request.
from tei_annotator.schemas.blbl import build_blbl_schema as _build_blbl_schema
_SCHEMA = _build_blbl_schema()

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 300) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _make_call_fn(model: str, timeout: int = 300):
    url = f"{_HF_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_HF_TOKEN}"}

    def call_fn(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        result = _post_json(url, payload, headers, timeout)
        return result["choices"][0]["message"]["content"]

    call_fn.__name__ = f"hf/{model}"
    return call_fn


# ---------------------------------------------------------------------------
# Action functions
# ---------------------------------------------------------------------------


def do_annotate(text: str, model: str):
    if not _HF_TOKEN:
        return "", "HF_TOKEN is not set. Add it as a Space secret."
    if not text.strip():
        return "", "Please enter some text to annotate."

    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate

    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=_make_call_fn(model),
    )
    schema = _SCHEMA
    t0 = time.monotonic()
    try:
        result = annotate(text=text, schema=schema, endpoint=endpoint, gliner_model=None)
    except Exception as exc:
        return "", f"Error: {exc}"
    elapsed = round(time.monotonic() - t0, 1)
    return result.xml, f"Done in {elapsed}s"


def do_load_samples(n: int):
    from lxml import etree

    if not _FIXTURE_PATH.exists():
        return "Fixture file not found. Make sure tests/fixtures/ is present."
    tree = etree.parse(str(_FIXTURE_PATH))
    bibls = tree.findall(".//{http://www.tei-c.org/ns/1.0}bibl")
    samples = random.sample(bibls, min(int(n), len(bibls)))

    from tei_annotator.evaluation.extractor import extract_spans

    return "\n\n".join(extract_spans(el)[0] for el in samples)


_BATCH_SEP = "\n---RECORD|||SEP|||BOUNDARY---\n"


def do_evaluate(model: str, n: int, batch_size: int = 1):
    if not _HF_TOKEN:
        return None, "HF_TOKEN is not set. Add it as a Space secret."

    from lxml import etree

    from tei_annotator.evaluation.extractor import extract_spans
    from tei_annotator.evaluation.metrics import MatchMode, aggregate, compute_metrics
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate

    if not _FIXTURE_PATH.exists():
        return None, "Fixture file not found. Make sure tests/fixtures/ is present."

    tree = etree.parse(str(_FIXTURE_PATH))
    bibls = tree.findall(".//{http://www.tei-c.org/ns/1.0}bibl")
    samples = random.sample(bibls, min(int(n), len(bibls)))
    schema = _SCHEMA
    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=_make_call_fn(model),
    )
    batch_size = max(1, int(batch_size))
    parser = etree.XMLParser(recover=True)

    def _annotate_one(el):
        plain_text, gold_spans = extract_spans(el)
        try:
            ann_result = annotate(plain_text, schema, endpoint, gliner_model=None)
            pred_el = etree.fromstring(f"<bibl>{ann_result.xml}</bibl>".encode(), parser)
        except Exception:
            pred_el = etree.Element("bibl")
        _, pred_spans = extract_spans(pred_el)
        return compute_metrics(gold_spans, pred_spans, mode=MatchMode.TEXT)

    def _annotate_batch(batch_els):
        plain_texts = []
        gold_spans_list = []
        for el in batch_els:
            pt, gs = extract_spans(el)
            plain_texts.append(pt)
            gold_spans_list.append(gs)

        if any(_BATCH_SEP in t for t in plain_texts):
            return [_annotate_one(el) for el in batch_els]

        combined = _BATCH_SEP.join(plain_texts)
        try:
            ann_result = annotate(combined, schema, endpoint, gliner_model=None)
        except Exception as exc:
            raise RuntimeError(f"Error during batch annotation: {exc}") from exc

        pieces = ann_result.xml.split(_BATCH_SEP)
        if len(pieces) != len(batch_els):
            return [
                compute_metrics(gs, [], mode=MatchMode.TEXT)
                for gs in gold_spans_list
            ]

        results = []
        for piece, gs in zip(pieces, gold_spans_list):
            try:
                pred_el = etree.fromstring(f"<bibl>{piece}</bibl>".encode(), parser)
            except Exception:
                pred_el = etree.Element("bibl")
            _, pred_spans = extract_spans(pred_el)
            results.append(compute_metrics(gs, pred_spans, mode=MatchMode.TEXT))
        return results

    per_result = []
    t0 = time.monotonic()
    try:
        if batch_size <= 1:
            for el in samples:
                per_result.append(_annotate_one(el))
        else:
            for start in range(0, len(samples), batch_size):
                per_result.extend(_annotate_batch(samples[start : start + batch_size]))
    except Exception as exc:
        return None, f"Error during annotation: {exc}"

    elapsed = round(time.monotonic() - t0, 1)
    agg = aggregate(per_result)

    rows = [
        [
            tag,
            round(m.precision, 3),
            round(m.recall, 3),
            round(m.f1, 3),
            m.true_positives,
            m.false_positives,
            m.false_negatives,
        ]
        for tag, m in sorted(agg.per_element.items(), key=lambda kv: -kv[1].f1)
    ]
    summary = (
        f"n={len(samples)} samples | "
        f"micro P={agg.micro_precision:.3f}  R={agg.micro_recall:.3f}  F1={agg.micro_f1:.3f} | "
        f"{elapsed}s"
    )
    return rows, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="TEI Annotator") as demo:
    gr.Markdown(
        "# TEI Annotator\n"
        "[GitHub Repo](https://github.com/cboulanger/tei-annotator)\n"
        "This demo annotates bibliographic plain text with TEI XML tags (tei:author, tei:title etc.) "
        "using open LLMs via the HuggingFace Inference Router."
    )
    if not _HF_TOKEN:
        gr.Markdown("> **Setup required:** Set `HF_TOKEN` as a Space secret and restart.")

    with gr.Tabs():
        # ── Annotation tab ────────────────────────────────────────────────
        with gr.Tab("Annotate"):
            model_dd = gr.Dropdown(
                choices=_HF_MODELS, value=_HF_MODELS[0], label="Model"
            )
            text_in = gr.Textbox(
                lines=5,
                label="Input text",
                placeholder="Paste a bibliographic reference here…",
            )
            annotate_btn = gr.Button("Annotate", variant="primary")
            xml_out = gr.Code(label="XML output", language="html", interactive=False)
            ann_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

            annotate_btn.click(
                do_annotate,
                inputs=[text_in, model_dd],
                outputs=[xml_out, ann_status],
            )

        # ── Evaluation tab ────────────────────────────────────────────────
        with gr.Tab("Evaluate"):
            eval_model_dd = gr.Dropdown(
                choices=_HF_MODELS, value=_HF_MODELS[0], label="Model"
            )
            n_slider = gr.Slider(1, 20, value=5, step=1, label="Number of samples")
            batch_size_slider = gr.Slider(
                1, 20, value=min(n_slider.value, 5), step=1, label="Batch size",
                info="Records per LLM call. Values > 1 reduce latency but may reduce quality.",
            )
            with gr.Row():
                sample_btn = gr.Button("Load sample texts")
                eval_btn = gr.Button("Run evaluation", variant="primary")
            sample_out = gr.Textbox(
                lines=8,
                label="Sampled texts (from gold standard)",
                interactive=False,
            )
            eval_table = gr.Dataframe(
                headers=["Element", "Precision", "Recall", "F1", "TP", "FP", "FN"],
                label="Per-element metrics (sorted by F1)",
            )
            eval_status = gr.Textbox(label="Summary", interactive=False, max_lines=2)

            n_slider.change(
                lambda n: min(int(n), 5),
                inputs=[n_slider],
                outputs=[batch_size_slider],
            )
            sample_btn.click(do_load_samples, inputs=[n_slider], outputs=[sample_out])
            eval_btn.click(
                do_evaluate,
                inputs=[eval_model_dd, n_slider, batch_size_slider],
                outputs=[eval_table, eval_status],
            )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
