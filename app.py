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

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _make_call_fn(model: str, timeout: int = 120):
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
    from tei_annotator.schemas.blbl import build_blbl_schema

    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=_make_call_fn(model),
    )
    schema = build_blbl_schema()
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


def do_evaluate(model: str, n: int):
    if not _HF_TOKEN:
        return None, "HF_TOKEN is not set. Add it as a Space secret."

    from lxml import etree

    from tei_annotator.evaluation.extractor import extract_spans
    from tei_annotator.evaluation.metrics import MatchMode, aggregate, compute_metrics
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate
    from tei_annotator.schemas.blbl import build_blbl_schema

    if not _FIXTURE_PATH.exists():
        return None, "Fixture file not found. Make sure tests/fixtures/ is present."

    tree = etree.parse(str(_FIXTURE_PATH))
    bibls = tree.findall(".//{http://www.tei-c.org/ns/1.0}bibl")
    samples = random.sample(bibls, min(int(n), len(bibls)))
    schema = build_blbl_schema()
    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=_make_call_fn(model),
    )

    per_result = []
    t0 = time.monotonic()
    for el in samples:
        plain_text, gold_spans = extract_spans(el)
        try:
            ann_result = annotate(plain_text, schema, endpoint, gliner_model=None)
        except Exception as exc:
            return None, f"Error during annotation: {exc}"
        try:
            parser = etree.XMLParser(recover=True)
            pred_el = etree.fromstring(f"<bibl>{ann_result.xml}</bibl>".encode(), parser)
        except Exception:
            pred_el = etree.Element("bibl")
        _, pred_spans = extract_spans(pred_el)
        per_result.append(compute_metrics(gold_spans, pred_spans, mode=MatchMode.TEXT))

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
        "Annotate bibliographic plain text with TEI XML tags using open LLMs "
        "via the HuggingFace Inference Router."
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

            sample_btn.click(do_load_samples, inputs=[n_slider], outputs=[sample_out])
            eval_btn.click(
                do_evaluate,
                inputs=[eval_model_dd, n_slider],
                outputs=[eval_table, eval_status],
            )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
