"""TRAIN-CV threshold calibration and centered-context ablations for mDeBERTa BIO."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from fact_extraction.common.metrics import mention_metrics
from fact_extraction.common.parser import parse_corpus
from fact_extraction.tasks.mention_extraction import MENTION_KINDS, _sentence_examples
from fact_extraction.tasks.mention_extraction_experiment_01 import _errors, _evaluable_gold, _tokenizer
from fact_extraction.tasks.mention_extraction_experiment_02 import FINAL_SEEDS, _set_seed
from fact_extraction.tasks.mention_extraction_experiment_03 import EXPERIMENT_02, Model, _decode, _loss, _mean_std, _record


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "MEANTIME" / "v2" / "meantime_newsreader_spanish_nov15" / "intra_cross-doc_annotation"
SPLITS = ROOT / "artifacts" / "splits" / "splits.json"
EXPERIMENT_03 = ROOT / "artifacts" / "mention_extraction" / "experiment_03"
OUTPUT = ROOT / "artifacts" / "mention_extraction" / "experiment_04"
CONFIG = {"learning_rate": 3e-5, "batch_size": 8, "dropout": 0.2, "max_epochs": 10, "patience": 2}
SEED = 20260925
GRID = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8)


def _metrics(predictions, gold):
    metrics = mention_metrics(predictions, gold, MENTION_KINDS)
    metrics["macro_f1"] = sum(metrics[kind]["f1"] for kind in MENTION_KINDS) / len(MENTION_KINDS)
    return metrics


def _encoded_length(tokenizer, words):
    return len(tokenizer([words], is_split_into_words=True, truncation=False)["input_ids"][0])


def _examples(documents, mode, tokenizer, max_length):
    base, exclusions = _sentence_examples(documents)
    by_doc = defaultdict(list)
    for item in base:
        by_doc[item[0]].append(item)
    sep = tokenizer.sep_token or "[SEP]"
    output, diagnostics = [], []
    for items in by_doc.values():
        items.sort(key=lambda item: int(item[1]))
        for index, item in enumerate(items):
            if mode == "w2":
                before = items[max(0, index - 2):index]
                after = items[index + 1:index + 3]
                words = [word for sentence in before for word in sentence[3] + [sep]] + item[3] + [word for sentence in after for word in [sep] + sentence[3]]
                start = sum(len(sentence[3]) + 1 for sentence in before)
                info = {"previous_sentences": len(before), "following_sentences": len(after), "context_truncated": _encoded_length(tokenizer, words) > max_length, "target_truncated": _encoded_length(tokenizer, item[3]) > max_length}
            else:
                words, start, info = _max_context(items, index, tokenizer, max_length, sep)
            output.append((item, words, start, info))
            diagnostics.append({"encoded_length": min(_encoded_length(tokenizer, words), max_length), **info, "document_id": item[0], "sentence_id": item[1]})
    return output, exclusions, diagnostics


def _max_context(items, index, tokenizer, max_length, sep):
    target = items[index][3]
    if _encoded_length(tokenizer, target) > max_length:
        return target, 0, {"previous_sentences": 0, "following_sentences": 0, "context_truncated": False, "target_truncated": True}
    budget = max_length - _encoded_length(tokenizer, target)
    before, after, left, right = [], [], index - 1, index + 1
    # Add complete nearest sentences while respecting roughly half of available encoded budget per side.
    for side in ("left", "right"):
        while (left >= 0 if side == "left" else right < len(items)):
            candidate = items[left][3] if side == "left" else items[right][3]
            proposal_before = [candidate] + before if side == "left" else before
            proposal_after = after if side == "left" else after + [candidate]
            words = [word for sentence in proposal_before for word in sentence + [sep]] + target + [word for sentence in proposal_after for word in [sep] + sentence]
            if _encoded_length(tokenizer, words) > _encoded_length(tokenizer, target) + budget // 2:
                break
            before, after = proposal_before, proposal_after
            if side == "left": left -= 1
            else: right += 1
    # Let either side consume unused capacity, preserving closest-first document order.
    progress = True
    while progress:
        progress = False
        for side in ("left", "right"):
            if (left < 0 if side == "left" else right >= len(items)): continue
            candidate = items[left][3] if side == "left" else items[right][3]
            proposal_before = [candidate] + before if side == "left" else before
            proposal_after = after if side == "left" else after + [candidate]
            words = [word for sentence in proposal_before for word in sentence + [sep]] + target + [word for sentence in proposal_after for word in [sep] + sentence]
            if _encoded_length(tokenizer, words) <= max_length:
                before, after, progress = proposal_before, proposal_after, True
                if side == "left": left -= 1
                else: right += 1
    words = [word for sentence in before for word in sentence + [sep]] + target + [word for sentence in after for word in [sep] + sentence]
    return words, sum(len(sentence) + 1 for sentence in before), {"previous_sentences": len(before), "following_sentences": len(after), "context_truncated": left >= 0 or right < len(items), "target_truncated": False}


def _batch(tokenizer, examples, device, max_length):
    encoded = tokenizer([item[1] for item in examples], is_split_into_words=True, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = torch.full((len(examples), encoded.input_ids.shape[1], len(MENTION_KINDS)), -100, dtype=torch.long)
    for row, (item, _, start, _) in enumerate(examples):
        end, first = start + len(item[2]), set()
        for position, word_id in enumerate(encoded.word_ids(row)):
            if word_id is not None and start <= word_id < end and word_id not in first:
                first.add(word_id)
                for kind_index, kind in enumerate(MENTION_KINDS): labels[row, position, kind_index] = item[4][kind][word_id - start]
    return {key: value.to(device) for key, value in encoded.items()}, labels.to(device)


def _predict(documents, model, tokenizer, mode, max_length):
    examples, _, diagnostics = _examples(documents, mode, tokenizer, max_length)
    device, output = next(model.parameters()).device, []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), CONFIG["batch_size"]):
            batch = examples[start:start + CONFIG["batch_size"]]
            inputs, labels = _batch(tokenizer, batch, device, max_length)
            probabilities = model(**inputs).softmax(-1).cpu()
            for row, item in enumerate(batch):
                positions = [position for position, value in enumerate(labels[row, :, 0].tolist()) if value != -100]
                for kind_index, kind in enumerate(MENTION_KINDS):
                    tags = [int(probabilities[row, pos, kind_index].argmax()) for pos in positions]
                    scores = [float(probabilities[row, pos, kind_index, 1:].max()) for pos in positions]
                    output.extend(_decode(item, kind, tags, scores))
    return output, diagnostics


def _train(train_documents, dev_documents, mode, max_length, seed=SEED, checkpoint=None):
    _set_seed(seed)
    tokenizer = _tokenizer("mdeberta_v3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    train_examples, train_exclusions, _ = _examples(train_documents, mode, tokenizer, max_length)
    gold, dev_exclusions = _evaluable_gold(dev_documents)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    history, best, stale = [], None, 0
    with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
        path = Path(temporary) / "best.pt"; rng = np.random.default_rng(seed)
        for epoch in range(1, CONFIG["max_epochs"] + 1):
            rng.shuffle(train_examples); model.train(); losses = []
            for start in range(0, len(train_examples), CONFIG["batch_size"]):
                inputs, labels = _batch(tokenizer, train_examples[start:start + CONFIG["batch_size"]], device, max_length)
                loss = _loss(model, model(**inputs), labels); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
            predictions, _ = _predict(dev_documents, model, tokenizer, mode, max_length); metrics = _metrics(predictions, gold)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "micro_f1": metrics["micro"]["f1"], "micro_precision": metrics["micro"]["precision"], "micro_recall": metrics["micro"]["recall"], **{f"{kind}_f1": metrics[kind]["f1"] for kind in MENTION_KINDS}})
            if best is None or metrics["micro"]["f1"] > best["metrics"]["micro"]["f1"]:
                best = {"epoch": epoch, "metrics": metrics, "predictions": predictions}; torch.save(model.state_dict(), path); stale = 0
            else:
                stale += 1
                if stale >= CONFIG["patience"]: break
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        if checkpoint: checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), checkpoint)
    best.update({"history": history, "gold": gold, "errors": _errors(best["predictions"], gold, dev_documents), "exclusions": {"train": train_exclusions, "dev": dev_exclusions}})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best


def _filter(predictions, thresholds): return [item for item in predictions if item["confidence"] >= thresholds[item["kind"]]]


def _folds(train_ids):
    ids = sorted(train_ids); random.Random(SEED).shuffle(ids)
    return [ids[index::4] for index in range(4)]


def _verify_train_cv_split(train_ids, dev_ids):
    train, dev = set(train_ids), set(dev_ids)
    assert train and dev and not train.intersection(dev), "TRAIN and DEV must be disjoint"
    held_out = [doc_id for fold in _folds(train_ids) for doc_id in fold]
    assert len(held_out) == len(train) and set(held_out) == train, "each TRAIN document must be OOF once"
    assert not set(held_out).intersection(dev), "DEV cannot enter TRAIN-CV calibration"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _thresholds(oof_predictions, oof_gold):
    rows, chosen = {}, {}
    for kind in MENTION_KINDS:
        values = []
        for threshold in GRID:
            filtered = [item for item in oof_predictions if item["kind"] != kind or item["confidence"] >= threshold]
            metric = _metrics(filtered, oof_gold)[kind]
            values.append({"threshold": threshold, "kept": metric["predicted"], "rejected": sum(item["kind"] == kind and item["confidence"] < threshold for item in oof_predictions), **metric})
        best = max(values, key=lambda item: (item["f1"], -item["threshold"]))
        rows[kind], chosen[kind] = values, best["threshold"]
    return rows, chosen


def _load_e3(dev_documents):
    tokenizer = _tokenizer("mdeberta_v3"); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device); model.load_state_dict(torch.load(EXPERIMENT_03 / "final" / "checkpoint" / "seed_20260925.pt", map_location=device, weights_only=True))
    predictions, diagnostics = _predict(dev_documents, model, tokenizer, "w2", 256); gold, _ = _evaluable_gold(dev_documents)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return predictions, gold, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    documents = parse_corpus(CORPUS); split = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]; by_id = {doc.id: doc for doc in documents}
    train_ids, dev_ids = split["train"], split["dev"]; train, dev = [by_id[x] for x in train_ids], [by_id[x] for x in dev_ids]
    _verify_train_cv_split(train_ids, dev_ids)
    if args.render_only:
        thresholds = json.loads((OUTPUT / "train_cv_thresholds" / "thresholds.json").read_text(encoding="utf-8"))
        raw_predictions, gold, _ = _load_e3(dev); filtered = _filter(raw_predictions, thresholds)
        raw_metrics, threshold_metrics = _metrics(raw_predictions, gold), _metrics(filtered, gold)
        raw_keys = {(item["document_id"], item["kind"], tuple(item["token_ids"])) for item in raw_predictions}
        filtered_keys = {(item["document_id"], item["kind"], tuple(item["token_ids"])) for item in filtered}
        gold_keys = {(item["document_id"], item["kind"], tuple(item["token_ids"])) for item in gold}
        threshold_effect = {"false_positives_removed": len((raw_keys - gold_keys) - (filtered_keys - gold_keys)), "true_positives_removed": len((raw_keys & gold_keys) - (filtered_keys & gold_keys))}
        w2_512_data = json.loads((OUTPUT / "context_window_2" / "metrics_dev.json").read_text(encoding="utf-8"))
        max_data = json.loads((OUTPUT / "context_max" / "metrics_dev.json").read_text(encoding="utf-8"))
        final = json.loads((OUTPUT / "final_comparison.json").read_text(encoding="utf-8"))
        usage = json.loads((OUTPUT / "diagnostics.json").read_text(encoding="utf-8"))
        _render(raw_metrics, threshold_metrics, thresholds, w2_512_data, max_data, final, usage, threshold_effect)
        return
    folds = _folds(train_ids); _write(OUTPUT / "train_cv_thresholds" / "folds.json", {"seed": SEED, "folds": folds})
    oof_predictions, oof_gold = [], []
    for index, held_ids in enumerate(folds):
        held = [by_id[x] for x in held_ids]; fitted = [by_id[x] for x in train_ids if x not in set(held_ids)]
        result = _train(fitted, held, "w2", 256); oof_predictions.extend(result["predictions"]); oof_gold.extend(result["gold"])
        _write(OUTPUT / "train_cv_thresholds" / f"fold_{index}.json", {"held_out": held_ids, "best_epoch": result["epoch"], "metrics": result["metrics"]})
    search, thresholds = _thresholds(oof_predictions, oof_gold)
    _write(OUTPUT / "train_cv_thresholds" / "threshold_search.json", search); _write(OUTPUT / "train_cv_thresholds" / "thresholds.json", thresholds)
    # 4A: use the same historical full-TRAIN W2@256 checkpoint, not a threshold-specific retraining.
    raw_predictions, gold, w2_diagnostics = _load_e3(dev); filtered = _filter(raw_predictions, thresholds)
    raw_metrics, threshold_metrics = _metrics(raw_predictions, gold), _metrics(filtered, gold)
    raw_keys = {(item["document_id"], item["kind"], tuple(item["token_ids"])) for item in raw_predictions}
    filtered_keys = {(item["document_id"], item["kind"], tuple(item["token_ids"])) for item in filtered}
    gold_keys = {(item["document_id"], item["kind"], tuple(item["token_ids"])) for item in gold}
    threshold_effect = {"false_positives_removed": len((raw_keys - gold_keys) - (filtered_keys - gold_keys)), "true_positives_removed": len((raw_keys & gold_keys) - (filtered_keys & gold_keys))}
    _write(OUTPUT / "threshold_ablation" / "metrics_dev.json", {"thresholds": thresholds, "before": raw_metrics, "after": threshold_metrics})
    _jsonl(OUTPUT / "threshold_ablation" / "predictions_dev.jsonl", filtered); _jsonl(OUTPUT / "threshold_ablation" / "errors_dev.jsonl", _errors(filtered, gold, dev))
    # 4B0 and 4B use fresh single-seed training. The only variable in each paired comparison is stated in its report.
    w2_512 = _train(train, dev, "w2", 512); maximum = _train(train, dev, "max", 512)
    for name, result in (("context_window_2", w2_512), ("context_max", maximum)):
        _write(OUTPUT / name / "metrics_dev.json", {"best_epoch": result["epoch"], "history": result["history"], "metrics": result["metrics"], "exclusions": result["exclusions"]})
        _jsonl(OUTPUT / name / "predictions_dev.jsonl", result["predictions"]); _jsonl(OUTPUT / name / "errors_dev.jsonl", result["errors"])
    tokenizer = _tokenizer("mdeberta_v3"); w2_examples, _, w2_usage = _examples(train + dev, "w2", tokenizer, 256)
    usage = {"model_max_positions": 512, "effective_historical_max_length": 256, "window_2": {"mean": float(np.mean([x["encoded_length"] for x in w2_usage])), "median": float(np.median([x["encoded_length"] for x in w2_usage])), "max": max(x["encoded_length"] for x in w2_usage), "truncated": sum(x["context_truncated"] for x in w2_usage), "count": len(w2_usage), "mean_unused_at_256": float(np.mean([256 - x["encoded_length"] for x in w2_usage]) )}}
    _, _, max_usage = _examples(train + dev, "max", tokenizer, 512); usage["maximum_centered"] = {"mean_previous_sentences": float(np.mean([x["previous_sentences"] for x in max_usage])), "mean_following_sentences": float(np.mean([x["following_sentences"] for x in max_usage])), "min_context_sentences": min(x["previous_sentences"] + x["following_sentences"] for x in max_usage), "max_context_sentences": max(x["previous_sentences"] + x["following_sentences"] for x in max_usage), "mean_encoded_length": float(np.mean([x["encoded_length"] for x in max_usage])), "near_limit_percent": 100 * sum(x["encoded_length"] >= 480 for x in max_usage) / len(max_usage), "context_truncated_percent": 100 * sum(x["context_truncated"] for x in max_usage) / len(max_usage)}
    _write(OUTPUT / "diagnostics.json", usage)
    # Do not combine frozen thresholds with maximum context: calibration is from W2@256 OOF models.
    selected = "M4B-MAX" if maximum["metrics"]["micro"]["f1"] > w2_512["metrics"]["micro"]["f1"] else "M4B-W2"
    mode, length = ("max", 512) if selected == "M4B-MAX" else ("w2", 512)
    seeds = []
    for seed in FINAL_SEEDS:
        result = _train(train, dev, mode, length, seed=seed, checkpoint=OUTPUT / "final" / "checkpoint" / f"seed_{seed}.pt")
        seeds.append({"seed": seed, "metrics": result["metrics"], "best_epoch": result["epoch"], "predictions": result["predictions"], "errors": result["errors"]})
    final = {"selected": selected, "three_seed": seeds, "aggregate": _mean_std(seeds), "experiment_03_reference": json.loads((EXPERIMENT_03 / "final_comparison.json").read_text(encoding="utf-8"))["aggregate"]}
    _write(OUTPUT / "final_comparison.json", final); _jsonl(OUTPUT / "final" / "predictions_dev.jsonl", seeds[0]["predictions"]); _jsonl(OUTPUT / "final" / "errors_dev.jsonl", seeds[0]["errors"])
    _render(raw_metrics, threshold_metrics, thresholds, w2_512, maximum, final, usage, threshold_effect)


def _render(raw, threshold, thresholds, w2, maximum, final, usage, threshold_effect):
    lines = ["# Experiment 04A: Frozen TRAIN-CV Thresholds", "", "Thresholds were selected only from four-fold out-of-fold TRAIN predictions; DEV was not used for selection.", "", "| Type | Threshold | Precision Before | Precision After | Recall Before | Recall After | F1 Before | F1 After |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in MENTION_KINDS: lines.append(f"| {kind} | {thresholds[kind]:.1f} | {raw[kind]['precision']:.3f} | {threshold[kind]['precision']:.3f} | {raw[kind]['recall']:.3f} | {threshold[kind]['recall']:.3f} | {raw[kind]['f1']:.3f} | {threshold[kind]['f1']:.3f} |")
    lines.extend(["", f"Micro F1: {raw['micro']['f1']:.3f} -> {threshold['micro']['f1']:.3f}.", f"False positives removed: {threshold_effect['false_positives_removed']}; true positives removed: {threshold_effect['true_positives_removed']}."])
    (OUTPUT / "experiment_4a.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# Experiment 04B: Token Budget And Context", "", "| Model | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro P | Micro R | Micro F1 | Macro F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in (("M4B0-W2 @ 256 historical", {"metrics": raw}), ("M4B-W2 @ 512", w2), ("M4B-MAX @ 512", maximum)):
        m=result["metrics"]; lines.append(f"| {name} | {m['ENTITY_MENTION']['f1']:.3f} | {m['EVENT_MENTION']['f1']:.3f} | {m['VALUE']['f1']:.3f} | {m['TIMEX3']['f1']:.3f} | {m['micro']['precision']:.3f} | {m['micro']['recall']:.3f} | {m['micro']['f1']:.3f} | {m['macro_f1']:.3f} |")
    lines.extend(["", "```json", json.dumps(usage, indent=2), "```"])
    (OUTPUT / "experiment_4b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ref, tuned = final["experiment_03_reference"], final["aggregate"]
    lines = ["# Experiment 04 Final Comparison", "", "Thresholds are not combined with maximum context because they were calibrated from W2@256 out-of-fold confidence distributions.", "", "## Experiment 4A", "", "| System | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Delta |", "|---|---:|---:|---:|---:|---:|---:|", f"| window_2 @ 256 | {raw['ENTITY_MENTION']['f1']:.3f} | {raw['EVENT_MENTION']['f1']:.3f} | {raw['VALUE']['f1']:.3f} | {raw['TIMEX3']['f1']:.3f} | {raw['micro']['f1']:.3f} | - |", f"| + frozen TRAIN-CV thresholds | {threshold['ENTITY_MENTION']['f1']:.3f} | {threshold['EVENT_MENTION']['f1']:.3f} | {threshold['VALUE']['f1']:.3f} | {threshold['TIMEX3']['f1']:.3f} | {threshold['micro']['f1']:.3f} | {threshold['micro']['f1'] - raw['micro']['f1']:+.3f} |", "", "## Experiment 4B", "", "| System | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Delta |", "|---|---:|---:|---:|---:|---:|---:|", f"| window_2 @ 512 | {w2['metrics']['ENTITY_MENTION']['f1']:.3f} | {w2['metrics']['EVENT_MENTION']['f1']:.3f} | {w2['metrics']['VALUE']['f1']:.3f} | {w2['metrics']['TIMEX3']['f1']:.3f} | {w2['metrics']['micro']['f1']:.3f} | - |", f"| maximum centered @ 512 | {maximum['metrics']['ENTITY_MENTION']['f1']:.3f} | {maximum['metrics']['EVENT_MENTION']['f1']:.3f} | {maximum['metrics']['VALUE']['f1']:.3f} | {maximum['metrics']['TIMEX3']['f1']:.3f} | {maximum['metrics']['micro']['f1']:.3f} | {maximum['metrics']['micro']['f1'] - w2['metrics']['micro']['f1']:+.3f} |", "", "## Three-Seed Final", "", "| Final System | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Micro SD |", "|---|---:|---:|---:|---:|---:|---:|", f"| Experiment 03 M3-FINAL | {ref['ENTITY_MENTION']['f1']['mean']:.3f} | {ref['EVENT_MENTION']['f1']['mean']:.3f} | {ref['VALUE']['f1']['mean']:.3f} | {ref['TIMEX3']['f1']['mean']:.3f} | {ref['micro']['f1']['mean']:.3f} | {ref['micro']['f1']['std']:.3f} |", f"| {final['selected']} | {tuned['ENTITY_MENTION']['f1']['mean']:.3f} | {tuned['EVENT_MENTION']['f1']['mean']:.3f} | {tuned['VALUE']['f1']['mean']:.3f} | {tuned['TIMEX3']['f1']['mean']:.3f} | {tuned['micro']['f1']['mean']:.3f} | {tuned['micro']['f1']['std']:.3f} |"]
    (OUTPUT / "final_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
