from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TEIAttribute:
    name: str
    description: str
    required: bool = False
    allowed_values: list[str] | None = None


@dataclass
class TEIElement:
    tag: str
    description: str
    allowed_children: list[str] = field(default_factory=list)
    attributes: list[TEIAttribute] = field(default_factory=list)


@dataclass
class TEISchema:
    elements: list[TEIElement] = field(default_factory=list)

    def get(self, tag: str) -> TEIElement | None:
        for elem in self.elements:
            if elem.tag == tag:
                return elem
        return None
