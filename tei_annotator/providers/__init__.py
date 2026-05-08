"""
LLM provider connectors for tei-annotator.

To add a new provider:
  1. Create a new module in this package (e.g. ``myprovider.py``).
  2. Subclass :class:`Connector` and implement the abstract methods.
  3. Append an instance to :data:`_ALL_CONNECTORS` below.
"""

from .base import Connector, _RateLimiter, _post_json
from .claude import ClaudeConnector
from .gemini import GeminiConnector
from .hf import HFConnector
from .kisski import KISSKIConnector
from .openai import OpenAIConnector

_ALL_CONNECTORS: list[Connector] = [
    HFConnector(),
    GeminiConnector(),
    KISSKIConnector(),
    OpenAIConnector(),
    ClaudeConnector(),
]


def get_available_connectors() -> list[Connector]:
    """Return all connectors whose required credentials are present."""
    return [c for c in _ALL_CONNECTORS if c.is_available()]


def get_connector(connector_id: str) -> Connector | None:
    """Look up a connector by its id string."""
    return next((c for c in _ALL_CONNECTORS if c.id == connector_id), None)
