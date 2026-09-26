"""Controlled XLM-R/Longformer and BIO/CRF mention-extraction experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer

from fact_extraction.common.metrics import mention_metrics
from fact_extraction.common.parser import parse_corpus, sentence_tokens
from fact_extraction.tasks.mention_extraction import MENTION_KINDS, SEED, _batch, _sentence_examples, seed_everything


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "MEANTIME" / "v2" / "meantime_newsreader_spanish_nov15" / "intra_cross-doc_annotation"
SPLITS = ROOT / "artifacts" / "splits" / "splits.json"
OUTPUT = ROOT / "artifacts" / "mention_extraction" / "experiment_01"
CONFIG = {"learning_rate": 2e-5, "epochs": 3, "dropout": 0.1, "batch_size": 4, "max_length": 256, "seed": SEED}
ENCODERS = {
    "xlm_r": {"model": "xlm-roberta-base", "label": "XLM-R", "pretraining": "multilingual", "tokenizer_kwargs": {}},
    "longformer": {"model": "allenai/longformer-base-4096", "label": "Longformer", "pretraining": "English", "tokenizer_kwargs": {"add_prefix_space": True}},
    "beto": {"model": "dccuchile/bert-base-spanish-wwm-cased", "label": "BETO", "pretraining": "Spanish-specific BERT", "tokenizer_kwargs": {}},
    "bertin": {"model": "bertin-project/bertin-roberta-base-spanish", "label": "BERTIN", "pretraining": "Spanish-specific RoBERTa", "tokenizer_kwargs": {"add_prefix_space": True}},
    "mdeberta_v3": {"model": "microsoft/mdeberta-v3-base", "label": "mDeBERTa-v3", "pretraining": "multilingual DeBERTa", "tokenizer_kwargs": {}},
    "spanish_longformer": {"model": "mrm8488/longformer-base-4096-spanish", "label": "Spanish Longformer", "pretraining": "Spanish-adapted Longformer checkpoint", "tokenizer_kwargs": {"add_prefix_space": True}},
}
SYSTEMS = {f"{key}_{decoder}": (key, decoder) for key in ENCODERS for decoder in ("bio", "crf")}
ORIGINAL_SYSTEMS = ("xlm_r_bio", "xlm_r_crf", "longformer_bio", "longformer_crf")
NEW_SYSTEMS = tuple(name for name in SYSTEMS if name not in ORIGINAL_SYSTEMS)


class MentionModel(torch.nn.Module):
    def __init__(self, model_name: str, decoder: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(CONFIG["dropout"])
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, len(MENTION_KINDS) * 3)
        self.decoder = decoder
        self.crfs = torch.nn.ModuleList([CRF(3, batch_first=True) for _ in MENTION_KINDS]) if decoder == "crf" else None

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        return self.head(self.dropout(hidden)).view(hidden.shape[0], hidden.shape[1], len(MENTION_KINDS), 3)


def _tokenizer(encoder_key):
    encoder = ENCODERS[encoder_key]
    return AutoTokenizer.from_pretrained(encoder["model"], **encoder["tokenizer_kwargs"])


def _word_level(logits, labels):
    """Compact first-subword emissions so CRFs see only MEANTIME token positions."""
    positions = [[index for index, value in enumerate(row[:, 0].tolist()) if value != -100] for row in labels]
    width = max(map(len, positions))
    emissions = logits.new_zeros((len(positions), width, len(MENTION_KINDS), 3))
    tags = labels.new_zeros((len(positions), width, len(MENTION_KINDS)))
    mask = torch.zeros((len(positions), width), dtype=torch.bool, device=logits.device)
    for row, indexes in enumerate(positions):
        emissions[row, :len(indexes)] = logits[row, indexes]
        tags[row, :len(indexes)] = labels[row, indexes]
        mask[row, :len(indexes)] = True
    return emissions, tags, mask, positions


def _loss(model, logits, labels):
    if model.decoder == "bio":
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        return sum(loss_fn(logits[:, :, kind, :].reshape(-1, 3), labels[:, :, kind].reshape(-1)) for kind in range(len(MENTION_KINDS)))
    emissions, tags, mask, _ = _word_level(logits, labels)
    return sum(-model.crfs[kind](emissions[:, :, kind], tags[:, :, kind], mask=mask, reduction="mean") for kind in range(len(MENTION_KINDS)))


def _evaluable_gold(documents):
    """Apply exactly the BIO representability rules used by every system."""
    records, exclusions = [], Counter()
    for document in documents:
        token_by_id = {token.id: token for token in document.tokens}
        indexes = {sentence_id: {token.id: position for position, token in enumerate(tokens)} for sentence_id, tokens in sentence_tokens(document).items()}
        occupied = {sentence_id: {kind: set() for kind in MENTION_KINDS} for sentence_id in indexes}
        for mention in document.mentions:
            if mention.kind not in MENTION_KINDS:
                continue
            if not mention.token_ids:
                exclusions["zero_width"] += 1
                continue
            sentence_ids = {token_by_id[token_id].sentence_id for token_id in mention.token_ids}
            if len(sentence_ids) != 1:
                exclusions["cross_sentence"] += 1
                continue
            sentence_id = sentence_ids.pop()
            positions = [indexes[sentence_id][token_id] for token_id in mention.token_ids]
            if positions != list(range(min(positions), max(positions) + 1)):
                exclusions["discontinuous"] += 1
                continue
            if set(positions) & occupied[sentence_id][mention.kind]:
                exclusions["overlapping_same_type"] += 1
                continue
            occupied[sentence_id][mention.kind].update(positions)
            records.append({"document_id": document.id, "sentence_id": sentence_id, "kind": mention.kind,
                            "token_ids": mention.token_ids, "text": mention.text, "confidence": 1.0})
    return records, dict(exclusions)


def _record(example, kind, indexes, scores):
    return {"document_id": example[0], "sentence_id": example[1], "kind": kind,
            "token_ids": [example[2][index] for index in indexes],
            "text": " ".join(example[3][index] for index in indexes), "confidence": float(np.mean(scores))}


def _decode_tags(example, kind, tags, confidences):
    output, active, scores = [], [], []
    for index, tag in enumerate(tags):
        if tag == 1 or (tag == 2 and not active):
            if active:
                output.append(_record(example, kind, active, scores))
            active, scores = [index], [confidences[index]]
        elif tag == 2 and active:
            active.append(index)
            scores.append(confidences[index])
        elif active:
            output.append(_record(example, kind, active, scores))
            active, scores = [], []
    if active:
        output.append(_record(example, kind, active, scores))
    return output


def _predict(documents, model, tokenizer):
    examples, _ = _sentence_examples(documents)
    device, predictions = next(model.parameters()).device, []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), CONFIG["batch_size"]):
            batch = examples[start:start + CONFIG["batch_size"]]
            inputs, labels, encoded = _batch(tokenizer, batch, device)
            logits = model(**inputs)
            if model.decoder == "crf":
                emissions, _, mask, positions = _word_level(logits, labels)
                decoded = [model.crfs[kind].decode(emissions[:, :, kind], mask=mask) for kind in range(len(MENTION_KINDS))]
            probabilities = logits.softmax(-1).cpu()
            for row, example in enumerate(batch):
                word_scores = [None] * len(example[2])
                for position, word_id in enumerate(encoded.word_ids(row)):
                    if word_id is not None and word_scores[word_id] is None:
                        word_scores[word_id] = probabilities[row, position]
                for kind_index, kind in enumerate(MENTION_KINDS):
                    available = [index for index, score in enumerate(word_scores) if score is not None]
                    scores = [float(word_scores[index][kind_index, 1:].max()) for index in available]
                    if model.decoder == "bio":
                        tags = [int(word_scores[index][kind_index].argmax()) for index in available]
                    else:
                        tags = decoded[kind_index][row]
                    predictions.extend(_decode_tags(example, kind, tags, scores))
    return predictions


def _errors(predictions, gold, documents):
    sentence_text = {(document.id, sentence_id): " ".join(token.text for token in tokens) for document in documents for sentence_id, tokens in sentence_tokens(document).items()}
    predicted = {(x["document_id"], x["kind"], tuple(x["token_ids"])): x for x in predictions}
    actual = {(x["document_id"], x["kind"], tuple(x["token_ids"])): x for x in gold}
    records = []
    for key, item in actual.items():
        if key in predicted:
            continue
        overlaps = [candidate for candidate in predictions if candidate["document_id"] == item["document_id"] and candidate["kind"] == item["kind"] and set(candidate["token_ids"]) & set(item["token_ids"])]
        error_type = "missed mention"
        if overlaps:
            candidate = overlaps[0]
            same_left = candidate["token_ids"][0] == item["token_ids"][0]
            same_right = candidate["token_ids"][-1] == item["token_ids"][-1]
            error_type = "right-boundary error" if same_left else "left-boundary error" if same_right else "both-boundaries wrong"
        records.append({"document_id": item["document_id"], "sentence_id": item["sentence_id"], "sentence_text": sentence_text[(item["document_id"], item["sentence_id"])], "task": "mention_extraction", "gold_item": item, "predicted_item": overlaps[0] if overlaps else None, "confidence": overlaps[0]["confidence"] if overlaps else None, "error_type": error_type})
    for key, item in predicted.items():
        if key not in actual:
            records.append({"document_id": item["document_id"], "sentence_id": item["sentence_id"], "sentence_text": sentence_text[(item["document_id"], item["sentence_id"])], "task": "mention_extraction", "gold_item": None, "predicted_item": item, "confidence": item["confidence"], "error_type": "spurious mention"})
    return records


def _train_and_evaluate(name, encoder_key, decoder, train_documents, dev_documents):
    seed_everything()
    encoder = ENCODERS[encoder_key]
    model_name = encoder["model"]
    system_dir = OUTPUT / name
    checkpoint = system_dir / "checkpoint" / "mention_model.pt"
    system_dir.joinpath("checkpoint").mkdir(parents=True, exist_ok=True)
    tokenizer = _tokenizer(encoder_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MentionModel(model_name, decoder).to(device)
    train_examples, training_exclusions = _sentence_examples(train_documents)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
        model.train()
        rng = np.random.default_rng(SEED)
        for _ in range(CONFIG["epochs"]):
            rng.shuffle(train_examples)
            for start in range(0, len(train_examples), CONFIG["batch_size"]):
                inputs, labels, _ = _batch(tokenizer, train_examples[start:start + CONFIG["batch_size"]], device)
                loss = _loss(model, model(**inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        torch.save(model.state_dict(), checkpoint)
    gold, evaluation_exclusions = _evaluable_gold(dev_documents)
    predictions = _predict(dev_documents, model, tokenizer)
    metrics = mention_metrics(predictions, gold, MENTION_KINDS)
    metrics["macro_f1"] = sum(metrics[kind]["f1"] for kind in MENTION_KINDS) / len(MENTION_KINDS)
    metrics["configuration"] = CONFIG | {"system": name, "encoder": model_name, "encoder_label": encoder["label"], "pretraining": encoder["pretraining"], "decoder": decoder, "device": str(device), "input_context": "sentence", "first_subword_supervision": True, "tokenizer_workaround": encoder["tokenizer_kwargs"] or "none", "crf_library": "pytorch-crf==0.7.2" if decoder == "crf" else "not applicable", "crf_transition_constraints": "not supported by pytorch-crf; no custom CRF mathematics added" if decoder == "crf" else "not applicable"}
    metrics["span_exclusions"] = {"train": training_exclusions, "dev_evaluation": evaluation_exclusions}
    system_dir.joinpath("metrics_dev.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with system_dir.joinpath("predictions_dev.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with system_dir.joinpath("errors_dev.jsonl").open("w", encoding="utf-8") as handle:
        for record in _errors(predictions, gold, dev_documents):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics


def _verify_encoder(encoder_key, train_documents):
    """Required preflight: real-token alignment plus one BIO and CRF backward pass."""
    encoder = ENCODERS[encoder_key]
    tokenizer = _tokenizer(encoder_key)
    examples, _ = _sentence_examples(train_documents)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for decoder in ("bio", "crf"):
        seed_everything()
        model = MentionModel(encoder["model"], decoder).to(device)
        inputs, labels, encoded = _batch(tokenizer, examples[:1], device)
        word_ids = encoded.word_ids(0)
        expected = list(range(len(examples[0][2])))
        observed = sorted({word_id for word_id in word_ids if word_id is not None})
        if observed != expected:
            raise ValueError(f"{encoder_key} word_ids alignment mismatch: {observed} != {expected}")
        loss = _loss(model, model(**inputs), labels)
        loss.backward()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"model": encoder["model"], "tokenizer": type(tokenizer).__name__, "word_ids_alignment": "passed", "bio_backward": "passed", "crf_backward": "passed"}


def _load_metrics(name):
    metrics = json.loads((OUTPUT / name / "metrics_dev.json").read_text(encoding="utf-8"))
    config = metrics["configuration"]
    encoder_key, decoder = SYSTEMS[name]
    encoder = ENCODERS[encoder_key]
    # Original Experiment 01 files are immutable; enrich only the aggregate view.
    config.setdefault("encoder_label", encoder["label"])
    config.setdefault("pretraining", encoder["pretraining"])
    config.setdefault("decoder", decoder)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="*", choices=SYSTEMS, default=list(NEW_SYSTEMS))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    documents = parse_corpus(CORPUS)
    split_ids = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]
    by_id = {document.id: document for document in documents}
    train_documents = [by_id[document_id] for document_id in split_ids["train"]]
    dev_documents = [by_id[document_id] for document_id in split_ids["dev"]]
    verification = {key: _verify_encoder(key, train_documents) for key in {SYSTEMS[name][0] for name in args.systems}}
    if args.verify_only:
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return
    for name in args.systems:
        _train_and_evaluate(name, *SYSTEMS[name], train_documents, dev_documents)
    results = {name: _load_metrics(name) for name in SYSTEMS}
    comparison = {"experiment": "Mention Extraction Experiment 01", "train_documents": len(train_documents), "dev_documents": len(dev_documents), "test_evaluated": False, "fairness": "All systems use sentence input and identical labels/settings. Original Longformer is English-pretrained. Spanish Longformer checkpoint exposes a RoBERTa config with an attention-window field, so architecture and pretraining remain distinct factors.", "crf": {"library": "pytorch-crf==0.7.2", "transition_constraints": "not supported directly; no custom CRF mathematics added"}, "encoders": ENCODERS, "preflight": verification, "systems": results}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.joinpath("comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Mention Extraction Experiment 01", "", "Encoder screening only: all systems use sentence-level MEANTIME input, identical labels, split, seed, and optimization settings. Original Longformer is English-pretrained. The Spanish Longformer checkpoint has a RoBERTa config with an attention-window field; this is reported as published rather than assumed to be native Longformer attention.", "", "| Encoder | Decoder | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Macro F1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        config = result["configuration"]
        lines.append(f"| {config['encoder_label']} | {config['decoder'].upper()} | {result['ENTITY_MENTION']['f1']:.3f} | {result['EVENT_MENTION']['f1']:.3f} | {result['VALUE']['f1']:.3f} | {result['TIMEX3']['f1']:.3f} | {result['micro']['f1']:.3f} | {result['macro_f1']:.3f} |")
    lines.extend(["", "## Encoder Summary", "", "| Encoder | Best BIO/CRF Decoder | Best Micro F1 |", "|---|---|---:|"])
    for encoder_key, encoder in ENCODERS.items():
        candidates = [(result["micro"]["f1"], result["configuration"]["decoder"]) for result in results.values() if result["configuration"]["encoder_label"] == encoder["label"]]
        best, decoder = max(candidates)
        lines.append(f"| {encoder['label']} ({encoder['pretraining']}) | {decoder.upper()} | {best:.3f} |")
    OUTPUT.joinpath("comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
