"""Within-document entity coreference diagnostic."""
from .coreference import within_document_coreference


def evaluate(documents):
    return within_document_coreference(documents, "ENTITY_MENTION")
