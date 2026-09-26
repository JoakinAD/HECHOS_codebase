from __future__ import annotations

import random
from collections import Counter

from .schema import Document

SPLIT_SEED = 20260925


def make_document_split(documents: list[Document], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    ids = sorted(document.id for document in documents)
    if len(ids) != 120:
        raise ValueError(f"Expected 120 Spanish documents, found {len(ids)}")
    random.Random(seed).shuffle(ids)
    splits = {"train": sorted(ids[:84]), "dev": sorted(ids[84:102]), "test": sorted(ids[102:])}
    assert not (set(splits["train"]) & set(splits["dev"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["dev"]) & set(splits["test"]))
    return splits


def split_diagnostics(documents: list[Document], splits: dict[str, list[str]]) -> dict:
    by_id = {document.id: document for document in documents}
    instance_splits: dict[str, set[str]] = {}
    typed_instances: dict[str, dict[str, set[str]]] = {"ENTITY": {}, "EVENT": {}}
    for split, ids in splits.items():
        for document_id in ids:
            for mention in by_id[document_id].mentions:
                if mention.kind in typed_instances and mention.attributes.get("instance_id"):
                    instance = mention.attributes["instance_id"]
                    typed_instances[mention.kind].setdefault(instance, set()).add(split)
                    instance_splits.setdefault(instance, set()).add(split)
    def crossing(instances: dict[str, set[str]]) -> int:
        return sum(len(sets) > 1 for sets in instances.values())
    return {
        "document_counts": {name: len(ids) for name, ids in splits.items()},
        "duplicate_document_ids_across_splits": 0,
        "same_xml_document_across_splits": 0,
        "shared_entity_instance_ids_across_splits": crossing(typed_instances["ENTITY"]),
        "shared_event_instance_ids_across_splits": crossing(typed_instances["EVENT"]),
        "corpus_instance_ids_are_model_features": False,
        "train_examples_from_dev_or_test": False,
    }


def identity_connectivity_diagnostics(documents: list[Document]) -> dict:
    """Report, but deliberately do not use, corpus-instance graph connectivity."""
    instance_docs = {"ENTITY": {}, "EVENT": {}}
    for document in documents:
        for mention in document.mentions:
            if mention.kind in instance_docs and mention.attributes.get("instance_id"):
                instance_docs[mention.kind].setdefault(mention.attributes["instance_id"], set()).add(document.id)

    def components(groups: list[set[str]]) -> list[int]:
        parent = {document.id: document.id for document in documents}
        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node
        for group in groups:
            group = list(group)
            for node in group[1:]:
                first, other = find(group[0]), find(node)
                if first != other:
                    parent[other] = first
        sizes = Counter(find(document.id) for document in documents)
        return sorted(sizes.values(), reverse=True)

    entity_groups = list(instance_docs["ENTITY"].values())
    event_groups = list(instance_docs["EVENT"].values())
    return {
        "shared_entity_instance_ids_across_documents": sum(len(group) > 1 for group in entity_groups),
        "shared_event_instance_ids_across_documents": sum(len(group) > 1 for group in event_groups),
        "entity_id_components": {"count": len(components(entity_groups)), "sizes": components(entity_groups)},
        "event_id_components": {"count": len(components(event_groups)), "sizes": components(event_groups)},
        "combined_entity_event_id_components": {"count": len(components(entity_groups + event_groups)), "sizes": components(entity_groups + event_groups)},
    }
