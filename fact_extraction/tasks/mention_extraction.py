# fact_extraction/meantime_pipeline/baseline.py
"""Small, train-only Transformer baseline for document-level MEANTIME extraction."""
from __future__ import annotations

from collections import Counter, defaultdict
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from fact_extraction.common.parser import sentence_tokens
from fact_extraction.common.schema import Document

MENTION_KINDS = ["ENTITY_MENTION", "EVENT_MENTION", "VALUE", "TIMEX3"]
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

