from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class EndpointCapability(Enum):
    TEXT_GENERATION = "text_generation"  # plain LLM, JSON via prompt only
    JSON_ENFORCED = "json_enforced"      # constrained decoding guaranteed
    EXTRACTION = "extraction"            # GLiNER2/NuExtract-style native


@dataclass
class EndpointConfig:
    capability: EndpointCapability
    call_fn: Callable[[str], str]
    # call_fn signature: takes a prompt string, returns a response string.
    # Caller is responsible for auth, model selection, and retries.
