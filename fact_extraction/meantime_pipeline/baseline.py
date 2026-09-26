# fact_extraction/meantime_pipeline/baseline.py
"""Small, train-only Transformer baseline for document-level MEANTIME extraction."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoModel, AutoTokenizer

from .metrics import coreference_metrics, mention_metrics, relation_metrics
from .parser import sentence_tokens
from .schema import Document, Mention

MENTION_KINDS = ["ENTITY_MENTION", "EVENT_MENTION", "VALUE", "TIMEX3"]
EVENT_ATTRIBUTES = ["aspect", "certainty", "modality", "polarity", "pos", "pred", "special_cases", "tense", "time"]
SEED = 20260925


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _sentence_examples(documents: list[Document]):
    examples, skipped = [], Counter()
    for document in documents:
        by_token = {token.id: token for token in document.tokens}
        for sentence_id, tokens in sentence_tokens(document).items():
            token_ids = [token.id for token in tokens]
            labels = {kind: [0] * len(tokens) for kind in MENTION_KINDS}
            occupied = {kind: set() for kind in MENTION_KINDS}
            index = {token_id: i for i, token_id in enumerate(token_ids)}
            for mention in document.mentions:
                if mention.kind not in labels or not mention.token_ids:
                    continue
                if any(token_id not in index for token_id in mention.token_ids):
                    continue
                positions = [index[token_id] for token_id in mention.token_ids]
                if positions != list(range(min(positions), max(positions) + 1)):
                    skipped["discontinuous_training_span"] += 1
                    continue
                if set(positions) & occupied[mention.kind]:
                    skipped["overlapping_same_kind_training_span"] += 1
                    continue
                occupied[mention.kind].update(positions)
                labels[mention.kind][positions[0]] = 1
                for position in positions[1:]:
                    labels[mention.kind][position] = 2
            examples.append((document.id, sentence_id, token_ids, [token.text for token in tokens], labels))
    return examples, dict(skipped)


class TokenClassifier(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, len(MENTION_KINDS) * 3)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(hidden)).view(hidden.shape[0], hidden.shape[1], len(MENTION_KINDS), 3)


def _batch(tokenizer, examples, device):
    words = [example[3] for example in examples]
    encoded = tokenizer(words, is_split_into_words=True, padding=True, truncation=True, max_length=256, return_tensors="pt")
    labels = torch.full((len(examples), encoded.input_ids.shape[1], len(MENTION_KINDS)), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        previous = None
        for position, word_id in enumerate(encoded.word_ids(row)):
            if word_id is not None and word_id != previous:
                for kind_index, kind in enumerate(MENTION_KINDS):
                    labels[row, position, kind_index] = example[4][kind][word_id]
            previous = word_id
    return {key: value.to(device) for key, value in encoded.items()}, labels.to(device), encoded


def train_mention_model(train_documents: list[Document], model_name: str, output_dir: Path, epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-5):
    seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenClassifier(model_name).to(device)
    examples, skipped = _sentence_examples(train_documents)
    checkpoint = output_dir / "checkpoints" / "mention_model.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        return model, tokenizer, skipped
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    model.train()
    for _ in range(epochs):
        random.shuffle(examples)
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            inputs, labels, _ = _batch(tokenizer, batch, device)
            logits = model(**inputs)
            loss = sum(loss_fn(logits[:, :, kind, :].reshape(-1, 3), labels[:, :, kind].reshape(-1)) for kind in range(len(MENTION_KINDS)))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return model, tokenizer, skipped


def predict_mentions(documents: list[Document], model, tokenizer, batch_size: int = 8) -> list[dict]:
    examples, _ = _sentence_examples(documents)
    device = next(model.parameters()).device
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            inputs, _, encoded = _batch(tokenizer, batch, device)
            probabilities = model(**inputs).softmax(-1).cpu()
            for row, example in enumerate(batch):
                word_scores = [None for _ in example[2]]
                for position, word_id in enumerate(encoded.word_ids(row)):
                    if word_id is not None and word_scores[word_id] is None:
                        word_scores[word_id] = probabilities[row, position]
                for kind_index, kind in enumerate(MENTION_KINDS):
                    active = []
                    for word_index, scores in enumerate(word_scores):
                        if scores is None:  # Truncated word: do not fabricate a prediction.
                            if active:
                                predictions.append(_mention_record(example, kind, active, word_scores, kind_index))
                                active = []
                            continue
                        tag = int(scores[kind_index].argmax())
                        if tag == 1 or (tag == 2 and not active):
                            if active:
                                predictions.append(_mention_record(example, kind, active, word_scores, kind_index))
                            active = [word_index]
                        elif tag == 2 and active:
                            active.append(word_index)
                        elif active:
                            predictions.append(_mention_record(example, kind, active, word_scores, kind_index))
                            active = []
                    if active:
                        predictions.append(_mention_record(example, kind, active, word_scores, kind_index))
    return predictions


def _mention_record(example, kind, indexes, scores, kind_index):
    token_ids = [example[2][index] for index in indexes]
    confidence = float(np.mean([float(scores[index][kind_index, 1:].max()) for index in indexes]))
    return {"document_id": example[0], "sentence_id": example[1], "kind": kind, "token_ids": token_ids,
            "text": " ".join(example[3][index] for index in indexes), "confidence": confidence}


def gold_mentions(documents: list[Document]) -> list[dict]:
    result = []
    for document in documents:
        tokens = {token.id: token for token in document.tokens}
        for mention in document.mentions:
            if mention.kind not in MENTION_KINDS or not mention.token_ids:
                continue
            sentence_id = tokens[mention.token_ids[0]].sentence_id
            result.append({"document_id": document.id, "sentence_id": sentence_id, "kind": mention.kind,
                           "token_ids": mention.token_ids, "text": mention.text, "confidence": 1.0,
                           "attributes": mention.attributes, "mention_id": mention.id})
    return result


def train_attribute_majorities(documents: list[Document]) -> dict[str, str]:
    values = {attribute: Counter() for attribute in EVENT_ATTRIBUTES}
    for document in documents:
        for mention in document.mentions:
            if mention.kind == "EVENT_MENTION":
                for attribute in EVENT_ATTRIBUTES:
                    values[attribute][mention.attributes.get(attribute, "")] += 1
    return {attribute: counts.most_common(1)[0][0] for attribute, counts in values.items()}


def attribute_metrics(dev_documents: list[Document], majorities: dict[str, str]) -> dict:
    result = {}
    for attribute, prediction in majorities.items():
        gold = [mention.attributes.get(attribute, "") for document in dev_documents for mention in document.mentions if mention.kind == "EVENT_MENTION"]
        labels = sorted(set(gold) | {prediction})
        classes = {}
        for label in labels:
            tp = sum(value == label and prediction == label for value in gold)
            fp = sum(value != label and prediction == label for value in gold)
            fn = sum(value == label and prediction != label for value in gold)
            p, r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
            classes[label] = {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0, "support": gold.count(label)}
        result[attribute] = {"accuracy": sum(value == prediction for value in gold) / len(gold),
                             "macro_f1": sum(item["f1"] for item in classes.values()) / len(classes), "classes": classes}
    return result


def _node_key(document_id: str, mention: Mention) -> str:
    return f"{mention.kind}:{','.join(mention.token_ids)}"


def _relation_examples(documents: list[Document], include_labels=True):
    rows, labels_by_pair = [], defaultdict(set)
    valid = {"HAS_PARTICIPANT", "TLINK", "SLINK", "GLINK", "CLINK"}
    for document in documents:
        mentions = {mention.id: mention for mention in document.mentions if mention.kind in MENTION_KINDS and mention.token_ids}
        token_sentence = {token.id: int(token.sentence_id) for token in document.tokens}
        for relation in document.relations:
            if relation.type not in valid:
                continue
            label = relation.type + (":" + relation.attributes["sem_role"] if relation.type == "HAS_PARTICIPANT" else "")
            for source in relation.sources:
                for target in relation.targets:
                    if source in mentions and target in mentions:
                        labels_by_pair[(document.id, source, target)].add(label)
        ids = list(mentions)
        for source in ids:
            for target in ids:
                if source == target:
                    continue
                sm, tm = mentions[source], mentions[target]
                distance = abs(token_sentence[sm.token_ids[0]] - token_sentence[tm.token_ids[0]])
                if distance > 1:  # current sentence plus adjacent sentence context
                    continue
                feature = {"source_kind=" + sm.kind: 1, "target_kind=" + tm.kind: 1,
                           "sentence_distance=" + str(distance): 1,
                           "source_text=" + sm.text.lower(): 1, "target_text=" + tm.text.lower(): 1}
                rows.append((document.id, source, target, sm, tm, feature, labels_by_pair[(document.id, source, target)]))
    return rows


def train_relation_model(train_documents: list[Document]):
    rows = _relation_examples(train_documents)
    vectorizer = DictVectorizer()
    features = vectorizer.fit_transform([row[5] for row in rows])
    labels = sorted({label for row in rows for label in row[6]})
    matrix = np.array([[int(label in row[6]) for label in labels] for row in rows])
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=200, class_weight="balanced", random_state=SEED))
    classifier.fit(features, matrix)
    return vectorizer, classifier, labels, {"candidate_pairs": len(rows), "positive_pairs": int(matrix.any(axis=1).sum()), "labels": Counter(label for row in rows for label in row[6])}


def predict_relations(documents, vectorizer, classifier, labels):
    rows = _relation_examples(documents, include_labels=False)
    matrix = classifier.predict_proba(vectorizer.transform([row[5] for row in rows]))
    output = []
    for row, probabilities in zip(rows, matrix):
        for label, probability in zip(labels, probabilities):
            if probability < 0.5:
                continue
            relation_type, _, role = label.partition(":")
            output.append({"document_id": row[0], "type": relation_type, "source": _node_key(row[0], row[3]),
                           "target": _node_key(row[0], row[4]), "role": role, "confidence": float(probability)})
    return output


def gold_relations(documents):
    output = []
    for document in documents:
        mentions = {mention.id: mention for mention in document.mentions if mention.kind in MENTION_KINDS and mention.token_ids}
        for relation in document.relations:
            if relation.type == "REFERS_TO":
                continue
            for source in relation.sources:
                for target in relation.targets:
                    if source in mentions and target in mentions:
                        output.append({"document_id": document.id, "type": relation.type,
                                       "source": _node_key(document.id, mentions[source]), "target": _node_key(document.id, mentions[target]),
                                       "role": relation.attributes.get("sem_role", ""), "confidence": 1.0})
    return output


def within_document_coreference(documents: list[Document], mention_kind: str) -> dict:
    """Train-free lexical clustering baseline; IDs remain evaluation-only gold data."""
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


def run(train_documents: list[Document], dev_documents: list[Document], output_dir: Path, model_name="xlm-roberta-base") -> tuple[dict, list[dict], list[dict]]:
    model, tokenizer, skipped = train_mention_model(train_documents, model_name, output_dir)
    predicted_mentions = predict_mentions(dev_documents, model, tokenizer)
    gold = gold_mentions(dev_documents)
    majors = train_attribute_majorities(train_documents)
    vectorizer, classifier, labels, distribution = train_relation_model(train_documents)
    predicted_relations = predict_relations(dev_documents, vectorizer, classifier, labels)
    gold_relation_records = gold_relations(dev_documents)
    metrics = {
        "run": {"seed": SEED, "model": model_name, "epochs": 3, "batch_size": 4, "learning_rate": 2e-5,
                "checkpoint_criterion": "fixed three-epoch initial baseline; DEV is not used for training", "device": str(next(model.parameters()).device)},
        "mention_detection": mention_metrics(predicted_mentions, gold, MENTION_KINDS),
        "event_attributes_gold_spans": attribute_metrics(dev_documents, majors),
        "relations_oracle_gold_nodes": relation_metrics(predicted_relations, gold_relation_records),
        "relations_end_to_end": {"status": "not yet decoded from predicted nodes; oracle candidate classifier reported separately"},
        "graph_diagnostics": {"node_exact_span": mention_metrics(predicted_mentions, gold, MENTION_KINDS)["micro"],
                              "edge_oracle": relation_metrics(predicted_relations, gold_relation_records)["micro"],
                              "predicted_nodes": len(predicted_mentions), "gold_nodes": len(gold),
                              "predicted_edges_oracle": len(predicted_relations), "gold_edges": len(gold_relation_records)},
        "relation_class_distribution_train": {key: int(value) for key, value in distribution["labels"].items()} | {"candidate_pairs": distribution["candidate_pairs"], "positive_pairs": distribution["positive_pairs"]},
        "timex_normalization": {"status": "unresolved: HeidelTime unavailable in the project environment; no normalized dates invented"},
        "entity_coreference": {"gold_mention_within_document": within_document_coreference(dev_documents, "ENTITY_MENTION"), "cross_document": "DEFERRED"},
        "event_coreference": {"gold_mention_within_document": within_document_coreference(dev_documents, "EVENT_MENTION"), "cross_document": "DEFERRED"},
        "training_span_exclusions": skipped,
    }
    return metrics, predicted_mentions + predicted_relations, gold
