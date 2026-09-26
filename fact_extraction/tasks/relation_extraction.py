"""Train-only document-level relation candidate classifier."""
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from fact_extraction.tasks.mention_extraction import MENTION_KINDS, SEED


def _node_key(mention):
    return f"{mention.kind}:{','.join(mention.token_ids)}"


def _examples(documents):
    rows, labels_by_pair = [], defaultdict(set)
    valid = {"HAS_PARTICIPANT", "TLINK", "SLINK", "GLINK", "CLINK"}
    for document in documents:
        mentions = {m.id: m for m in document.mentions if m.kind in MENTION_KINDS and m.token_ids}
        token_sentence = {token.id: int(token.sentence_id) for token in document.tokens}
        for relation in document.relations:
            if relation.type not in valid:
                continue
            label = relation.type + (":" + relation.attributes["sem_role"] if relation.type == "HAS_PARTICIPANT" else "")
            for source in relation.sources:
                for target in relation.targets:
                    if source in mentions and target in mentions:
                        labels_by_pair[(document.id, source, target)].add(label)
        for source, sm in mentions.items():
            for target, tm in mentions.items():
                if source == target:
                    continue
                distance = abs(token_sentence[sm.token_ids[0]] - token_sentence[tm.token_ids[0]])
                if distance > 1:
                    continue
                feature = {f"source_kind={sm.kind}": 1, f"target_kind={tm.kind}": 1, f"sentence_distance={distance}": 1, f"source_text={sm.text.lower()}": 1, f"target_text={tm.text.lower()}": 1}
                rows.append((document.id, sm, tm, feature, labels_by_pair[(document.id, source, target)]))
    return rows


def train(documents):
    rows = _examples(documents)
    vectorizer = DictVectorizer()
    features = vectorizer.fit_transform([row[3] for row in rows])
    labels = sorted({label for row in rows for label in row[4]})
    matrix = np.array([[int(label in row[4]) for label in labels] for row in rows])
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=200, class_weight="balanced", random_state=SEED))
    classifier.fit(features, matrix)
    return vectorizer, classifier, labels, {"candidate_pairs": len(rows), "positive_pairs": int(matrix.any(axis=1).sum()), "labels": Counter(label for row in rows for label in row[4])}


def predict(documents, vectorizer, classifier, labels):
    rows = _examples(documents)
    output = []
    for row, probabilities in zip(rows, classifier.predict_proba(vectorizer.transform([row[3] for row in rows]))):
        for label, probability in zip(labels, probabilities):
            if probability >= 0.5:
                relation_type, _, role = label.partition(":")
                output.append({"document_id": row[0], "type": relation_type, "source": _node_key(row[1]), "target": _node_key(row[2]), "role": role, "confidence": float(probability)})
    return output


def gold(documents):
    output = []
    for document in documents:
        mentions = {m.id: m for m in document.mentions if m.kind in MENTION_KINDS and m.token_ids}
        for relation in document.relations:
            if relation.type == "REFERS_TO":
                continue
            for source in relation.sources:
                for target in relation.targets:
                    if source in mentions and target in mentions:
                        output.append({"document_id": document.id, "type": relation.type, "source": _node_key(mentions[source]), "target": _node_key(mentions[target]), "role": relation.attributes.get("sem_role", ""), "confidence": 1.0})
    return output
