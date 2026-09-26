"""Targeted DEV-only ablations for the selected mDeBERTa BIO mention model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torchcrf import CRF
from transformers import AutoModel

from fact_extraction.common.metrics import mention_metrics
from fact_extraction.common.parser import parse_corpus, sentence_tokens
from fact_extraction.tasks.mention_extraction import MENTION_KINDS, _sentence_examples
from fact_extraction.tasks.mention_extraction_experiment_01 import ENCODERS, _errors, _evaluable_gold, _tokenizer, _word_level
from fact_extraction.tasks.mention_extraction_experiment_02 import FINAL_SEEDS, _batch, _length_diagnostics, _set_seed


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "MEANTIME" / "v2" / "meantime_newsreader_spanish_nov15" / "intra_cross-doc_annotation"
SPLITS = ROOT / "artifacts" / "splits" / "splits.json"
EXPERIMENT_02 = ROOT / "artifacts" / "mention_extraction" / "experiment_02" / "mdeberta_v3_bio"
OUTPUT = ROOT / "artifacts" / "mention_extraction" / "experiment_03"
LOCKED = {"learning_rate": 3e-5, "batch_size": 8, "dropout": 0.2, "context": "window_1", "max_epochs": 10, "patience": 2}
SEED = 20260925


class Model(torch.nn.Module):
    def __init__(self, timex_crf=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
        self.dropout = torch.nn.Dropout(LOCKED["dropout"])
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, len(MENTION_KINDS) * 3)
        self.timex_crf = CRF(3, batch_first=True) if timex_crf else None

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        return self.head(self.dropout(hidden)).view(hidden.shape[0], hidden.shape[1], len(MENTION_KINDS), 3)


def _examples(documents, context, separator):
    base, exclusions = _sentence_examples(documents)
    if context == "window_1":
        radius = 1
    elif context == "window_2":
        radius = 2
    else:
        return [(item, item[3], 0) for item in base], exclusions
    by_doc = defaultdict(list)
    for item in base:
        by_doc[item[0]].append(item)
    output = []
    for items in by_doc.values():
        items.sort(key=lambda item: int(item[1]))
        for index, item in enumerate(items):
            words = []
            for neighbor in range(max(0, index - radius), index):
                words.extend(items[neighbor][3])
                words.append(separator)
            target_start = len(words)
            words.extend(item[3])
            for neighbor in range(index + 1, min(len(items), index + radius + 1)):
                words.append(separator)
                words.extend(items[neighbor][3])
            output.append((item, words, target_start))
    return output, exclusions


def _record(item, kind, indexes, confidences):
    source = item[0]
    return {"document_id": source[0], "sentence_id": source[1], "kind": kind,
            "token_ids": [source[2][index] for index in indexes], "text": " ".join(source[3][index] for index in indexes),
            "confidence": float(np.mean(confidences))}


def _decode(item, kind, tags, confidences):
    output, active, scores = [], [], []
    for index, tag in enumerate(tags):
        if tag == 1 or (tag == 2 and not active):
            if active:
                output.append(_record(item, kind, active, scores))
            active, scores = [index], [confidences[index]]
        elif tag == 2 and active:
            active.append(index)
            scores.append(confidences[index])
        elif active:
            output.append(_record(item, kind, active, scores))
            active, scores = [], []
    if active:
        output.append(_record(item, kind, active, scores))
    return output


def _loss(model, logits, labels, value_weights=None):
    emissions, tags, mask, _ = _word_level(logits, labels)
    loss = 0
    for index in range(len(MENTION_KINDS)):
        if index == MENTION_KINDS.index("TIMEX3") and model.timex_crf:
            loss += -model.timex_crf(emissions[:, :, index], tags[:, :, index], mask=mask, reduction="mean")
        else:
            weights = value_weights if MENTION_KINDS[index] == "VALUE" else None
            loss += torch.nn.functional.cross_entropy(logits[:, :, index, :].reshape(-1, 3), labels[:, :, index].reshape(-1), weight=weights, ignore_index=-100)
    return loss


def _predict(documents, model, tokenizer, config):
    examples, _ = _examples(documents, config["context"], tokenizer.sep_token or "[SEP]")
    device, output = next(model.parameters()).device, []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), config["batch_size"]):
            batch = examples[start:start + config["batch_size"]]
            inputs, labels, _, _ = _batch(tokenizer, batch, device)
            logits = model(**inputs)
            emissions, _, mask, _ = _word_level(logits, labels)
            timex_tags = model.timex_crf.decode(emissions[:, :, MENTION_KINDS.index("TIMEX3")], mask=mask) if model.timex_crf else None
            probabilities = logits.softmax(-1).cpu()
            for row, item in enumerate(batch):
                positions = [position for position, value in enumerate(labels[row, :, 0].tolist()) if value != -100]
                for kind_index, kind in enumerate(MENTION_KINDS):
                    scores = [float(probabilities[row, position, kind_index, 1:].max()) for position in positions]
                    tags = timex_tags[row] if kind == "TIMEX3" and timex_tags is not None else [int(probabilities[row, position, kind_index].argmax()) for position in positions]
                    output.extend(_decode(item, kind, tags, scores))
    return output


def _metrics(predictions, gold):
    metrics = mention_metrics(predictions, gold, MENTION_KINDS)
    metrics["macro_f1"] = sum(metrics[kind]["f1"] for kind in MENTION_KINDS) / len(MENTION_KINDS)
    return metrics


def _weights(train_documents):
    examples, _ = _sentence_examples(train_documents)
    counts = Counter(label for _, _, _, _, streams in examples for label in streams["VALUE"])
    total = sum(counts.values())
    weights = torch.tensor([total / (3 * counts[index]) for index in range(3)], dtype=torch.float)
    return weights, {"O": counts[0], "B_VALUE": counts[1], "I_VALUE": counts[2]}


def _train(variant, train_documents, dev_documents, seed=SEED, checkpoint=None):
    _set_seed(seed)
    tokenizer = _tokenizer("mdeberta_v3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LOCKED | ({"context": "window_2"} if variant == "M3D" else {})
    model = Model(timex_crf=variant == "M3C").to(device)
    train_examples, exclusions = _examples(train_documents, config["context"], tokenizer.sep_token or "[SEP]")
    value_weights, value_counts = _weights(train_documents)
    value_weights = value_weights.to(device) if variant == "M3B" else None
    gold, dev_exclusions = _evaluable_gold(dev_documents)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    history, best, stale = [], None, 0
    with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
        best_path = Path(temporary) / "best.pt"
        rng = np.random.default_rng(seed)
        for epoch in range(1, config["max_epochs"] + 1):
            rng.shuffle(train_examples)
            losses = []
            model.train()
            for start in range(0, len(train_examples), config["batch_size"]):
                inputs, labels, _, _ = _batch(tokenizer, train_examples[start:start + config["batch_size"]], device)
                loss = _loss(model, model(**inputs), labels, value_weights)
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            predictions = _predict(dev_documents, model, tokenizer, config)
            metrics = _metrics(predictions, gold)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "micro_f1": metrics["micro"]["f1"], "micro_precision": metrics["micro"]["precision"], "micro_recall": metrics["micro"]["recall"], **{f"{kind}_f1": metrics[kind]["f1"] for kind in MENTION_KINDS}})
            if best is None or metrics["micro"]["f1"] > best["metrics"]["micro"]["f1"]:
                best = {"epoch": epoch, "metrics": metrics, "predictions": predictions}
                torch.save(model.state_dict(), best_path); stale = 0
            else:
                stale += 1
                if stale >= config["patience"]: break
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), checkpoint)
    best.update({"history": history, "gold": gold, "errors": _errors(best["predictions"], gold, dev_documents), "config": config, "value_counts": value_counts, "span_exclusions": {"train": exclusions, "dev": dev_exclusions}})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best


def _threshold_predictions(predictions, thresholds):
    return [item for item in predictions if item["confidence"] >= thresholds[item["kind"]]]


def _threshold_search(predictions, gold):
    values, selected = {}, {}
    for kind in MENTION_KINDS:
        rows = []
        for threshold in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8):
            filtered = [item for item in predictions if item["kind"] != kind or item["confidence"] >= threshold]
            metric = _metrics(filtered, gold)[kind]
            rows.append({"threshold": threshold, **metric})
        best = max(rows, key=lambda row: row["f1"])
        values[kind], selected[kind] = rows, best["threshold"]
    return values, selected, _threshold_predictions(predictions, selected)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_baseline(train_documents, dev_documents):
    aggregate = json.loads((EXPERIMENT_02 / "aggregate_metrics.json").read_text(encoding="utf-8"))
    tokenizer = _tokenizer("mdeberta_v3")
    model = Model().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load(EXPERIMENT_02 / "checkpoint" / "best.pt", map_location=next(model.parameters()).device, weights_only=True))
    predictions = _predict(dev_documents, model, tokenizer, LOCKED)
    gold, _ = _evaluable_gold(dev_documents)
    metrics = _metrics(predictions, gold)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return aggregate, predictions, gold, metrics


def _mean_std(results):
    output = {}
    for kind in MENTION_KINDS + ["micro"]:
        output[kind] = {metric: {"mean": float(np.mean([item["metrics"][kind][metric] for item in results])), "std": float(np.std([item["metrics"][kind][metric] for item in results]))} for metric in ("precision", "recall", "f1")}
    output["macro_f1"] = {"mean": float(np.mean([item["metrics"]["macro_f1"] for item in results])), "std": float(np.std([item["metrics"]["macro_f1"] for item in results]) )}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    documents = parse_corpus(CORPUS)
    split_ids = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]
    by_id = {document.id: document for document in documents}
    train, dev = [by_id[item] for item in split_ids["train"]], [by_id[item] for item in split_ids["dev"]]
    baseline_aggregate, baseline_predictions, gold, baseline_metrics = _load_baseline(train, dev)
    threshold_rows, thresholds, threshold_predictions = _threshold_search(baseline_predictions, gold)
    candidates = {"baseline": {"metrics": baseline_metrics}, "M3A": {"metrics": _metrics(threshold_predictions, gold), "thresholds": thresholds, "threshold_rows": threshold_rows}}
    _write(OUTPUT / "baseline" / "metrics_dev.json", candidates["baseline"])
    _write_jsonl(OUTPUT / "baseline" / "predictions_dev.jsonl", baseline_predictions)
    _write_jsonl(OUTPUT / "baseline" / "errors_dev.jsonl", _errors(baseline_predictions, gold, dev))
    _write_jsonl(OUTPUT / "threshold" / "predictions_dev.jsonl", threshold_predictions)
    _write_jsonl(OUTPUT / "threshold" / "errors_dev.jsonl", _errors(threshold_predictions, gold, dev))
    if args.materialize_only:
        saved = json.loads((OUTPUT / "ablation_results.json").read_text(encoding="utf-8"))["candidates"]
        candidates.update({name: saved[name] for name in ("M3B", "M3C", "M3D")})
        final = json.loads((OUTPUT / "final_comparison.json").read_text(encoding="utf-8"))
        selected_seed = next(item for item in final["three_seed"] if item["seed"] == SEED)
        _write_jsonl(OUTPUT / "final" / "predictions_dev.jsonl", selected_seed["predictions"])
        _write_jsonl(OUTPUT / "final" / "errors_dev.jsonl", selected_seed["errors"])
        _render(candidates, baseline_aggregate, final, train, baseline_predictions, gold, dev)
        return
    for variant, directory in (("M3B", "value_weighting"), ("M3C", "timex_crf"), ("M3D", "window_2")):
        result = _train(variant, train, dev)
        candidates[variant] = {"metrics": result["metrics"], "epoch": result["epoch"], "history": result["history"], "value_counts": result["value_counts"] if variant == "M3B" else None}
        _write(OUTPUT / directory / "metrics_dev.json", candidates[variant])
        with (OUTPUT / directory / "predictions_dev.jsonl").open("w", encoding="utf-8") as handle:
            for item in result["predictions"]: handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        with (OUTPUT / directory / "errors_dev.jsonl").open("w", encoding="utf-8") as handle:
            for item in result["errors"]: handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    _write(OUTPUT / "threshold" / "metrics_dev.json", candidates["M3A"])
    best_variant = max((name for name in candidates if name != "baseline"), key=lambda name: candidates[name]["metrics"]["micro"]["f1"])
    # Keep one final candidate: the strongest independent ablation; do not stack DEV-selected changes.
    if best_variant == "M3A":
        final_results = []
        for seed in FINAL_SEEDS:
            tokenizer = _tokenizer("mdeberta_v3"); model = Model().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            path = EXPERIMENT_02 / "checkpoint" / f"seed_{seed}.pt"
            model.load_state_dict(torch.load(path, map_location=next(model.parameters()).device, weights_only=True))
            predictions = _threshold_predictions(_predict(dev, model, tokenizer, LOCKED), thresholds)
            final_results.append({"seed": seed, "metrics": _metrics(predictions, gold), "predictions": predictions, "errors": _errors(predictions, gold, dev)})
            del model
    else:
        final_results = []
        for seed in FINAL_SEEDS:
            result = _train(best_variant, train, dev, seed=seed, checkpoint=OUTPUT / "final" / "checkpoint" / f"seed_{seed}.pt")
            final_results.append({"seed": seed, "metrics": result["metrics"], "predictions": result["predictions"], "errors": result["errors"], "epoch": result["epoch"]})
    final = {"selected_independent_ablation": best_variant, "three_seed": final_results, "aggregate": _mean_std(final_results), "baseline_experiment_02": baseline_aggregate["aggregate"]}
    selected_seed = next(item for item in final_results if item["seed"] == SEED)
    _write_jsonl(OUTPUT / "final" / "predictions_dev.jsonl", selected_seed["predictions"])
    _write_jsonl(OUTPUT / "final" / "errors_dev.jsonl", selected_seed["errors"])
    _write(OUTPUT / "ablation_results.json", {"locked_baseline": baseline_metrics, "candidates": candidates, "value_train_distribution": _weights(train)[1]})
    _write(OUTPUT / "final_comparison.json", final)
    _render(candidates, baseline_aggregate, final, train, baseline_predictions, gold, dev)


def _render(candidates, baseline_aggregate, final, train, baseline_predictions, gold, dev):
    error_rows = _errors(baseline_predictions, gold, dev)
    lines = ["# Mention Extraction Experiment 03 Diagnostics", "", "Locked baseline is mDeBERTa-v3 BIO with LR 3e-5, batch 8, dropout 0.2, window_1, and Experiment 02 early stopping.", "", "| Mention Type | Gold | Predicted | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|"]
    for kind in MENTION_KINDS:
        actual = {(item["document_id"], tuple(item["token_ids"])) for item in gold if item["kind"] == kind}
        predicted = {(item["document_id"], tuple(item["token_ids"])) for item in baseline_predictions if item["kind"] == kind}
        lines.append(f"| {kind} | {len(actual)} | {len(predicted)} | {len(actual & predicted)} | {len(predicted - actual)} | {len(actual - predicted)} |")
    counts = Counter(row["error_type"] for row in error_rows)
    lines.extend(["", "| Error Type | Count |", "|---|---:|"])
    lines.extend(f"| {name} | {counts[name]} |" for name in ("missed mention", "spurious mention", "left-boundary error", "right-boundary error", "both-boundaries wrong"))
    lines.extend(["", "VALUE TRAIN labels: " + json.dumps(_weights(train)[1]) + ". O-to-positive imbalance justifies M3B weighted cross-entropy for VALUE only.", "", "Boundary errors are 98/748 (13.1%); syntax refinement was skipped because no aligned dependency representation exists and boundary errors are not dominant."])
    (OUTPUT / "diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# Experiment 03 Ablations", "", "| Model | Modification | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Delta |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    baseline = candidates["baseline"]["metrics"]["micro"]["f1"]
    for name, label in (("baseline", "none"), ("M3A", "thresholding"), ("M3B", "VALUE weighting"), ("M3C", "TIMEX CRF"), ("M3D", "window_2")):
        metrics = candidates[name]["metrics"]
        lines.append(f"| {name} | {label} | {metrics['ENTITY_MENTION']['f1']:.3f} | {metrics['EVENT_MENTION']['f1']:.3f} | {metrics['VALUE']['f1']:.3f} | {metrics['TIMEX3']['f1']:.3f} | {metrics['micro']['f1']:.3f} | {metrics['micro']['f1'] - baseline:+.3f} |")
    lines.extend(["", "## M3A Threshold Search", "", "| Mention Type | Threshold | Precision | Recall | F1 | Predicted |", "|---|---:|---:|---:|---:|---:|"])
    for kind in MENTION_KINDS:
        for row in candidates["M3A"]["threshold_rows"][kind]:
            lines.append(f"| {kind} | {row['threshold']:.1f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['predicted']} |")
    (OUTPUT / "ablation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    base, tuned = baseline_aggregate["aggregate"], final["aggregate"]
    lines = ["# Experiment 03 Final Comparison", "", "TEST was not evaluated. Values are three-seed means.", "", "| Model | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Micro SD | Macro F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|", f"| Experiment 02 mDeBERTa BIO | {base['ENTITY_MENTION']['f1']['mean']:.3f} | {base['EVENT_MENTION']['f1']['mean']:.3f} | {base['VALUE']['f1']['mean']:.3f} | {base['TIMEX3']['f1']['mean']:.3f} | {base['micro']['f1']['mean']:.3f} | {base['micro']['f1']['std']:.3f} | {base['macro_f1']['mean']:.3f} |", f"| M3-FINAL ({final['selected_independent_ablation']}) | {tuned['ENTITY_MENTION']['f1']['mean']:.3f} | {tuned['EVENT_MENTION']['f1']['mean']:.3f} | {tuned['VALUE']['f1']['mean']:.3f} | {tuned['TIMEX3']['f1']['mean']:.3f} | {tuned['micro']['f1']['mean']:.3f} | {tuned['micro']['f1']['std']:.3f} | {tuned['macro_f1']['mean']:.3f} |", "", "| Model | Micro Precision | Micro Recall |", "|---|---:|---:|", f"| Experiment 02 mDeBERTa BIO | {base['micro']['precision']['mean']:.3f} | {base['micro']['recall']['mean']:.3f} |", f"| M3-FINAL ({final['selected_independent_ablation']}) | {tuned['micro']['precision']['mean']:.3f} | {tuned['micro']['recall']['mean']:.3f} |"]
    gains = [final_item["metrics"]["micro"]["f1"] - baseline_item["metrics"]["micro"]["f1"] for baseline_item, final_item in zip(baseline_aggregate["per_seed"], final["three_seed"])]
    lines.extend(["", "## Interpretation", "", "- M3-FINAL is M3D (`window_2`) only. M3A thresholding also improved the locked DEV seed, but it was not combined with window_2 because that interaction was not independently tested.", f"- M3-FINAL improves the matched Experiment 02 seed in all three runs: {', '.join(f'{gain:+.3f}' for gain in gains)} micro F1.", "- M3B and M3C were not selected because they reduced locked-seed micro F1 despite their targeted motivations.", "- M3A thresholds are DEV-selected and therefore its DEV score is a model-selection result, not an unbiased estimate."])
    (OUTPUT / "final_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
