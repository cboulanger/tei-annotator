"""HuggingFace Inference Router connector."""

from __future__ import annotations

import os
from typing import Callable

from .base import Connector, _post_json


class HFConnector(Connector):
    """HuggingFace Inference Router — OpenAI-compatible chat completions."""

    _BASE_URL = "https://router.huggingface.co/v1"

    _MODELS = [
        # ── fast ────────────────────────────────────────────────────────────
        "Qwen/Qwen3-8B",                              # 8B, multilingual, fast
        "meta-llama/Llama-3.1-8B-Instruct",           # 8B, reliable
        "Qwen/Qwen3-30B-A3B",                         # MoE: only 3B active params
        # ── balanced ────────────────────────────────────────────────────────
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",  # sparse MoE, good quality
        "Qwen/Qwen3-14B",                             # solid quality/speed trade-off
        # ── high quality (Pro / premium token) ──────────────────────────────
        "*meta-llama/Llama-3.3-70B-Instruct",
        "*Qwen/Qwen2.5-72B-Instruct",
    ]

    @property
    def id(self) -> str:
        return "hf"

    @property
    def name(self) -> str:
        return "HuggingFace Inference Router"

    @property
    def description(self) -> str:
        return "Open models via router.huggingface.co (requires HF_TOKEN)."

    def is_available(self) -> bool:
        return bool(os.environ.get("HF_TOKEN"))

    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        token = os.environ.get("HF_TOKEN", "")
        url = f"{self._BASE_URL}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        def call_fn(prompt: str) -> str:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            result = _post_json(url, payload, headers, timeout)
            return result["choices"][0]["message"]["content"]

        call_fn.__name__ = f"hf/{model_id}"
        return call_fn
