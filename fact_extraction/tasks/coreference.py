"""Shared implementation for the two document-local coreference tasks."""
from collections import defaultdict

from fact_extraction.common.metrics import coreference_metrics


def within_document_coreference(documents, mention_kind):
    gold_clusters, predicted_clusters = [], []
    for document in documents:
        mentions = {m.id: m for m in document.mentions if m.kind == mention_kind and m.token_ids}
        target_to_sources = defaultdict(set)
        for relation in document.relations:
            if relation.type == "REFERS_TO" and relation.targets:
                for source in relation.sources:
                    if source in mentions:
                        target_to_sources[relation.targets[0]].add(f"{document.id}:{source}")
        gold_clusters.extend(cluster for cluster in target_to_sources.values() if cluster)
        by_text = defaultdict(set)
        for source_id in {item for cluster in target_to_sources.values() for item in cluster}:
            mention = mentions[source_id.split(":", 1)[1]]
            by_text[" ".join(mention.text.lower().split())].add(source_id)
        predicted_clusters.extend(cluster for cluster in by_text.values() if cluster)
    return coreference_metrics(predicted_clusters, gold_clusters)
