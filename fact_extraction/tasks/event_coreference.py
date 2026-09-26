"""Within-document event coreference diagnostic."""
from .coreference import within_document_coreference


def evaluate(documents):
    return within_document_coreference(documents, "EVENT_MENTION")
