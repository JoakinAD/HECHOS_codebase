# fact_extraction/meantime_pipeline/parser.py
"""Lossless-enough parser for the MEANTIME XML actually distributed here."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

from .schema import Document, Mention, Relation, Token

MENTION_KINDS = {
    "ENTITY_MENTION", "EVENT_MENTION", "TIMEX3", "VALUE", "SIGNAL", "C-SIGNAL",
    "ENTITY", "EVENT",
}


def _surface(token_ids: list[str], by_id: dict[str, Token]) -> str:
    return " ".join(by_id[t].text for t in token_ids if t in by_id)


def parse_document(path: str | Path, root: str | Path | None = None) -> Document:
    """Parse one XML document without changing IDs, anchors, or attributes."""
    path = Path(path)
    xml_root = ET.parse(path).getroot()
    tokens = [
        Token(t.attrib["t_id"], int(t.attrib.get("number", "0")), t.attrib["sentence"], t.text or "")
        for t in xml_root.findall("token")
    ]
    by_id = {token.id: token for token in tokens}
    mentions = []
    markables = xml_root.find("Markables")
    for element in markables if markables is not None else []:
        if element.tag not in MENTION_KINDS:
            continue
        anchors = [anchor.attrib["t_id"] for anchor in element.findall("token_anchor")]
        attributes = dict(element.attrib)
        mention_id = attributes.pop("m_id")
        mentions.append(Mention(mention_id, element.tag, anchors, _surface(anchors, by_id), attributes))
    relations = []
    relations_node = xml_root.find("Relations")
    for element in relations_node if relations_node is not None else []:
        attributes = dict(element.attrib)
        relation_id = attributes.pop("r_id")
        relations.append(Relation(
            relation_id, element.tag,
            [node.attrib["m_id"] for node in element.findall("source")],
            [node.attrib["m_id"] for node in element.findall("target")], attributes,
        ))
    attrs = dict(xml_root.attrib)
    return Document(
        id=attrs.pop("doc_id"), name=attrs.pop("doc_name"), language=attrs.pop("lang"),
        source_path=str(path.relative_to(root)) if root else str(path), attributes=attrs,
        tokens=tokens, mentions=mentions, relations=relations,
    )


def parse_corpus(root: str | Path) -> list[Document]:
    root = Path(root)
    documents = [parse_document(path, root) for path in sorted(root.glob("**/*.xml"))]
    ids = [document.id for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Document IDs must be unique in one corpus view")
    return documents


def sentence_tokens(document: Document) -> dict[str, list[Token]]:
    grouped: dict[str, list[Token]] = defaultdict(list)
    for token in document.tokens:
        grouped[token.sentence_id].append(token)
    return dict(grouped)


def validate_document(document: Document) -> list[str]:
    """Return schema-integrity errors rather than discarding unusual annotations."""
    node_ids = {mention.id for mention in document.mentions}
    token_ids = {token.id for token in document.tokens}
    errors = []
    for mention in document.mentions:
        bad = set(mention.token_ids) - token_ids
        if bad:
            errors.append(f"mention {mention.id} references missing tokens {sorted(bad)}")
    for relation in document.relations:
        bad = set(relation.sources + relation.targets) - node_ids
        if bad:
            errors.append(f"relation {relation.id} references missing nodes {sorted(bad)}")
    return errors
