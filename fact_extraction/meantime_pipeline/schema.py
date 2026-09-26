from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Token:
    id: str
    number: int
    sentence_id: str
    text: str


@dataclass
class Mention:
    id: str
    kind: str
    token_ids: list[str]
    text: str
    attributes: dict[str, str]

    @property
    def is_zero_width(self) -> bool:
        return not self.token_ids


@dataclass
class Relation:
    id: str
    type: str
    sources: list[str]
    targets: list[str]
    attributes: dict[str, str]


@dataclass
class Document:
    id: str
    name: str
    language: str
    source_path: str
    attributes: dict[str, str]
    tokens: list[Token]
    mentions: list[Mention]
    relations: list[Relation]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
