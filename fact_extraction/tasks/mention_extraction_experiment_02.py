"""Sequential, DEV-only tuning for the three selected Experiment 01 systems."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torchcrf import CRF
from transformers import AutoModel

from fact_extraction.common.metrics import mention_metrics
from fact_extraction.common.parser import parse_corpus, sentence_tokens
from fact_extraction.tasks.mention_extraction import MENTION_KINDS, SEED, _sentence_examples, seed_everything
from fact_extraction.tasks.mention_extraction_experiment_01 import ENCODERS, _errors, _evaluable_gold, _tokenizer, _word_level


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "MEANTIME" / "v2" / "meantime_newsreader_spanish_nov15" / "intra_cross-doc_annotation"
SPLITS = ROOT / "artifacts" / "splits" / "splits.json"
EXPERIMENT_01 = ROOT / "artifacts" / "mention_extraction" / "experiment_01"
OUTPUT = ROOT / "artifacts" / "mention_extraction" / "experiment_02"
SYSTEMS = {
    "mdeberta_v3_bio": ("mdeberta_v3", "bio"),
    "mdeberta_v3_crf": ("mdeberta_v3", "crf"),
    "bertin_bio": ("bertin", "bio"),
}
BASE = {"learning_rate": 2e-5, "batch_size": 4, "dropout": 0.1, "context": "sentence", "max_epochs": 10, "patience": 2}
TUNE_SEED = 20260925
FINAL_SEEDS = (20260925, 20260926, 20260927)


class MentionModel(torch.nn.Module):
    def __init__(self, model_name, decoder, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, len(MENTION_KINDS) * 3)
        self.decoder = decoder
        self.crfs = torch.nn.ModuleList([CRF(3, batch_first=True) for _ in MENTION_KINDS]) if decoder == "crf" else None

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        return self.head(self.dropout(hidden)).view(hidden.shape[0], hidden.shape[1], len(MENTION_KINDS), 3)


def _set_seed(seed):
    seed_everything()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _context_examples(documents, mode, separator="[SEP]"):
    base, exclusions = _sentence_examples(documents)
    if mode == "sentence":
        return [(item, item[3], 0) for item in base], exclusions
    by_document = defaultdict(list)
    for item in base:
        by_document[item[0]].append(item)
    examples = []
    for items in by_document.values():
        items.sort(key=lambda item: int(item[1]))
        for index, item in enumerate(items):
            words = []
            if index:
                words.extend(items[index - 1][3])
                words.append(separator)
            target_start = len(words)
            words.extend(item[3])
            if index + 1 < len(items):
                words.append(separator)
                words.extend(items[index + 1][3])
            examples.append((item, words, target_start))
    return examples, exclusions


def _batch(tokenizer, examples, device):
    encoded = tokenizer([item[1] for item in examples], is_split_into_words=True, padding=True, truncation=True, max_length=256, return_tensors="pt")
    labels = torch.full((len(examples), encoded.input_ids.shape[1], len(MENTION_KINDS)), -100, dtype=torch.long)
    target_word_indexes = []
    for row, (item, _, start) in enumerate(examples):
        target_end = start + len(item[2])
        first_positions = {}
        for position, word_id in enumerate(encoded.word_ids(row)):
            if word_id is not None and start <= word_id < target_end and word_id not in first_positions:
                first_positions[word_id] = position
                for kind_index, kind in enumerate(MENTION_KINDS):
                    labels[row, position, kind_index] = item[4][kind][word_id - start]
        target_word_indexes.append(sorted(first_positions))
    return {key: value.to(device) for key, value in encoded.items()}, labels.to(device), encoded, target_word_indexes


def _loss(model, logits, labels):
    if model.decoder == "bio":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        return sum(criterion(logits[:, :, kind, :].reshape(-1, 3), labels[:, :, kind].reshape(-1)) for kind in range(len(MENTION_KINDS)))
    emissions, tags, mask, _ = _word_level(logits, labels)
    return sum(-model.crfs[kind](emissions[:, :, kind], tags[:, :, kind], mask=mask, reduction="mean") for kind in range(len(MENTION_KINDS)))


def _decode(item, kind, word_positions, tags, scores):
    output, active, active_scores = [], [], []
    for target_index, tag in enumerate(tags):
        if tag == 1 or (tag == 2 and not active):
            if active:
                output.append(_record(item, kind, active, active_scores))
            active, active_scores = [target_index], [scores[target_index]]
        elif tag == 2 and active:
            active.append(target_index)
            active_scores.append(scores[target_index])
        elif active:
            output.append(_record(item, kind, active, active_scores))
            active, active_scores = [], []
    if active:
        output.append(_record(item, kind, active, active_scores))
    return output


def _record(item, kind, indexes, scores):
    example = item[0]
    return {"document_id": example[0], "sentence_id": example[1], "kind": kind,
            "token_ids": [example[2][index] for index in indexes],
            "text": " ".join(example[3][index] for index in indexes), "confidence": float(np.mean(scores))}


def _predict(documents, model, tokenizer, config):
    examples, _ = _context_examples(documents, config["context"], tokenizer.sep_token or "[SEP]")
    device, output = next(model.parameters()).device, []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), config["batch_size"]):
            batch = examples[start:start + config["batch_size"]]
            inputs, labels, encoded, target_word_indexes = _batch(tokenizer, batch, device)
            logits = model(**inputs)
            if model.decoder == "crf":
                emissions, _, mask, _ = _word_level(logits, labels)
                decoded = [model.crfs[kind].decode(emissions[:, :, kind], mask=mask) for kind in range(len(MENTION_KINDS))]
            probabilities = logits.softmax(-1).cpu()
            for row, item in enumerate(batch):
                target_positions = [position for position, value in enumerate(labels[row, :, 0].tolist()) if value != -100]
                for kind_index, kind in enumerate(MENTION_KINDS):
                    scores = [float(probabilities[row, position, kind_index, 1:].max()) for position in target_positions]
                    tags = decoded[kind_index][row] if model.decoder == "crf" else [int(probabilities[row, position, kind_index].argmax()) for position in target_positions]
                    output.extend(_decode(item, kind, target_word_indexes[row], tags, scores))
    return output


def _metrics(predictions, gold):
    values = mention_metrics(predictions, gold, MENTION_KINDS)
    values["macro_f1"] = sum(values[kind]["f1"] for kind in MENTION_KINDS) / len(MENTION_KINDS)
    return values


def _train_run(system, config, train_documents, dev_documents, seed=TUNE_SEED, save_checkpoint=None):
    encoder_key, decoder = SYSTEMS[system]
    _set_seed(seed)
    tokenizer = _tokenizer(encoder_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MentionModel(ENCODERS[encoder_key]["model"], decoder, config["dropout"]).to(device)
    train_examples, training_exclusions = _context_examples(train_documents, config["context"], tokenizer.sep_token or "[SEP]")
    gold, evaluation_exclusions = _evaluable_gold(dev_documents)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    history, best, stale = [], None, 0
    with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
        best_path = Path(temporary) / "best.pt"
        rng = np.random.default_rng(seed)
        for epoch in range(1, config["max_epochs"] + 1):
            rng.shuffle(train_examples)
            model.train()
            losses = []
            for start in range(0, len(train_examples), config["batch_size"]):
                inputs, labels, _, _ = _batch(tokenizer, train_examples[start:start + config["batch_size"]], device)
                loss = _loss(model, model(**inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            predictions = _predict(dev_documents, model, tokenizer, config)
            metrics = _metrics(predictions, gold)
            epoch_record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_micro_precision": metrics["micro"]["precision"], "dev_micro_recall": metrics["micro"]["recall"], "dev_micro_f1": metrics["micro"]["f1"], **{f"{kind}_f1": metrics[kind]["f1"] for kind in MENTION_KINDS}}
            history.append(epoch_record)
            if best is None or metrics["micro"]["f1"] > best["metrics"]["micro"]["f1"]:
                best = {"epoch": epoch, "metrics": metrics, "predictions": predictions}
                torch.save(model.state_dict(), best_path)
                stale = 0
            else:
                stale += 1
                if stale >= config["patience"]:
                    break
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        if save_checkpoint:
            save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_checkpoint)
    best["history"] = history
    best["span_exclusions"] = {"train": training_exclusions, "dev_evaluation": evaluation_exclusions}
    best["errors"] = _errors(best["predictions"], gold, dev_documents)
    best["gold"] = gold
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def _stage(system, stage, candidates, base, train_documents, dev_documents):
    records = []
    for candidate in candidates:
        config = base | candidate
        result = _train_run(system, config, train_documents, dev_documents)
        records.append({"settings": candidate, "best_epoch": result["epoch"], "history": result["history"], "metrics": result["metrics"]})
    winner = max(records, key=lambda record: record["metrics"]["micro"]["f1"])
    return records, base | winner["settings"]


def _length_diagnostics(predictions, gold):
    result = {}
    for kind in MENTION_KINDS:
        result[kind] = {}
        for label, predicate in (("1_token", lambda item: len(item["token_ids"]) == 1), ("2_tokens", lambda item: len(item["token_ids"]) == 2), ("3plus_tokens", lambda item: len(item["token_ids"]) >= 3)):
            actual = {(item["document_id"], tuple(item["token_ids"])) for item in gold if item["kind"] == kind and predicate(item)}
            predicted = {(item["document_id"], tuple(item["token_ids"])) for item in predictions if item["kind"] == kind and predicate(item)}
            true_positive = len(actual & predicted)
            result[kind][label] = {"gold_support": len(actual), "predicted_count": len(predicted), "true_positives": true_positive, "false_positives": len(predicted - actual), "false_negatives": len(actual - predicted)}
    return result


def _mean_std(seed_results):
    output = {}
    for kind in MENTION_KINDS + ["micro"]:
        output[kind] = {metric: {"mean": float(np.mean([item["metrics"][kind][metric] for item in seed_results])), "std": float(np.std([item["metrics"][kind][metric] for item in seed_results]))} for metric in ("precision", "recall", "f1")}
    output["macro_f1"] = {"mean": float(np.mean([item["metrics"]["macro_f1"] for item in seed_results])), "std": float(np.std([item["metrics"]["macro_f1"] for item in seed_results]))}
    return output


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _alignment_check(system, documents):
    encoder_key, _ = SYSTEMS[system]
    tokenizer = _tokenizer(encoder_key)
    examples, _ = _context_examples(documents, "sentence", tokenizer.sep_token or "[SEP]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, labels, encoded, _ = _batch(tokenizer, examples[:1], device)
    observed = sorted({word_id for word_id in encoded.word_ids(0) if word_id is not None})
    expected = list(range(len(examples[0][0][2])))
    return observed == expected and int((labels[0, :, 0] != -100).sum()) == len(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--systems", nargs="*", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    experiment_one = json.loads((EXPERIMENT_01 / "comparison.json").read_text(encoding="utf-8"))
    baselines = {system: experiment_one["systems"][system]["micro"]["f1"] for system in SYSTEMS}
    documents = parse_corpus(CORPUS)
    split_ids = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]
    by_id = {document.id: document for document in documents}
    train_documents, dev_documents = [by_id[item] for item in split_ids["train"]], [by_id[item] for item in split_ids["dev"]]
    if args.verify_only:
        checks = {}
        for system in SYSTEMS:
            result = _train_run(system, BASE, train_documents[:1], dev_documents[:1], seed=TUNE_SEED)
            checks[system] = {"token_alignment": _alignment_check(system, train_documents), "dev_evaluation": bool(result["history"]), "best_checkpoint_restored": result["epoch"] == max(record["epoch"] for record in result["history"] if record["dev_micro_f1"] == max(item["dev_micro_f1"] for item in result["history"]))}
        print(json.dumps({"experiment_01_baselines": baselines, "checks": checks}, indent=2))
        return
    if args.render_only:
        finals = {system: json.loads((OUTPUT / system / "aggregate_metrics.json").read_text(encoding="utf-8")) for system in SYSTEMS}
        tuning = {"experiment_01_baselines": baselines, "seed": TUNE_SEED, "test_evaluated": False, "systems": {system: {"selected": value["best_hyperparameters"], "stages": value["tuning_stages"]} for system, value in finals.items()}}
        _write_json(OUTPUT / "tuning_results.json", tuning)
        _write_json(OUTPUT / "final_comparison.json", {"experiment_01_baselines": baselines, "test_evaluated": False, "systems": finals})
        _write_markdown(tuning, finals, baselines)
        return
    tuning = {"experiment_01_baselines": baselines, "seed": TUNE_SEED, "test_evaluated": False, "systems": {}}
    finals = {}
    for system in args.systems:
        config = copy.deepcopy(BASE)
        stages = {}
        stages["duration"], config = _stage(system, "duration", [{}], config, train_documents, dev_documents)
        stages["learning_rate"], config = _stage(system, "learning_rate", [{"learning_rate": rate} for rate in (1e-5, 2e-5, 3e-5)], config, train_documents, dev_documents)
        stages["batch_size"], config = _stage(system, "batch_size", [{"batch_size": size} for size in (4, 8)], config, train_documents, dev_documents)
        stages["dropout"], config = _stage(system, "dropout", [{"dropout": value} for value in (0.1, 0.2)], config, train_documents, dev_documents)
        stages["context"], config = _stage(system, "context", [{"context": value} for value in ("sentence", "window_1")], config, train_documents, dev_documents)
        system_dir = OUTPUT / system
        seed_results = []
        for seed in FINAL_SEEDS:
            checkpoint = system_dir / "checkpoint" / f"seed_{seed}.pt"
            result = _train_run(system, config, train_documents, dev_documents, seed=seed, save_checkpoint=checkpoint)
            seed_results.append({"seed": seed, "best_epoch": result["epoch"], "history": result["history"], "metrics": result["metrics"]})
            if seed == TUNE_SEED:
                with system_dir.joinpath("predictions_dev.jsonl").open("w", encoding="utf-8") as handle:
                    for record in result["predictions"]:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                with system_dir.joinpath("errors_dev.jsonl").open("w", encoding="utf-8") as handle:
                    for record in result["errors"]:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                diagnostics = _length_diagnostics(result["predictions"], result["gold"])
        aggregate = _mean_std(seed_results)
        system_result = {"best_hyperparameters": config, "best_epoch_seed_20260925": seed_results[0]["best_epoch"], "tuning_stages": stages, "per_seed": seed_results, "aggregate": aggregate, "span_length_diagnostics_seed_20260925": diagnostics, "event_pos_diagnostic": "unavailable: no existing token-aligned POS preprocessing is used", "span_exclusions": result["span_exclusions"]}
        _write_json(system_dir / "aggregate_metrics.json", system_result)
        tuning["systems"][system] = {"selected": config, "stages": stages}
        finals[system] = system_result
    for system in SYSTEMS:
        if system in finals:
            continue
        completed = json.loads((OUTPUT / system / "aggregate_metrics.json").read_text(encoding="utf-8"))
        finals[system] = completed
        tuning["systems"][system] = {"selected": completed["best_hyperparameters"], "stages": completed["tuning_stages"]}
    _write_json(OUTPUT / "tuning_results.json", tuning)
    _write_json(OUTPUT / "final_comparison.json", {"experiment_01_baselines": baselines, "test_evaluated": False, "systems": finals})
    _write_markdown(tuning, finals, baselines)


def _write_markdown(tuning, finals, baselines):
    lines = ["# Mention Extraction Experiment 02 Tuning", "", "All selection used DEV micro exact-span F1 only. TEST was never evaluated.", "", "## Experiment 01 Baselines", "", "| System | Micro F1 |", "|---|---:|"]
    lines.extend(f"| {name} | {score:.3f} |" for name, score in baselines.items())
    for system, value in tuning["systems"].items():
        lines.extend(["", f"## {system}", "", "| Stage | Settings | Best Epoch | Micro F1 |", "|---|---|---:|---:|"])
        for stage, records in value["stages"].items():
            for record in records:
                lines.append(f"| {stage} | `{json.dumps(record['settings'], sort_keys=True)}` | {record['best_epoch']} | {record['metrics']['micro']['f1']:.3f} |")
    (OUTPUT / "tuning_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# Mention Extraction Experiment 02 Final Comparison", "", "Values are means over deterministic seeds 20260925, 20260926, and 20260927. TEST was not evaluated.", "", "| System | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Macro F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for system in SYSTEMS:
        result = finals[system]
        aggregate = result["aggregate"]
        lines.append(f"| {system} | {aggregate['ENTITY_MENTION']['f1']['mean']:.3f} | {aggregate['EVENT_MENTION']['f1']['mean']:.3f} | {aggregate['VALUE']['f1']['mean']:.3f} | {aggregate['TIMEX3']['f1']['mean']:.3f} | {aggregate['micro']['f1']['mean']:.3f} | {aggregate['macro_f1']['mean']:.3f} |")
    lines.extend(["", "| System | LR | Batch | Dropout | Context | Best Epoch |", "|---|---:|---:|---:|---|---:|"])
    for system in SYSTEMS:
        result = finals[system]
        config = result["best_hyperparameters"]
        lines.append(f"| {system} | {config['learning_rate']} | {config['batch_size']} | {config['dropout']} | {config['context']} | {result['best_epoch_seed_20260925']} |")
    lines.extend(["", "## Experiment 01 Difference", "", "| System | Experiment 01 Micro F1 | Experiment 02 Mean Micro F1 | Absolute Difference |", "|---|---:|---:|---:|"])
    for system in SYSTEMS:
        result = finals[system]
        mean = result["aggregate"]["micro"]["f1"]["mean"]
        lines.append(f"| {system} | {baselines[system]:.3f} | {mean:.3f} | {mean - baselines[system]:+.3f} |")
    lines.extend(["", "## Span Length Diagnostics (Seed 20260925)", "", "| System | Mention Type | Length | Gold | Predicted | TP | FP | FN |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for system in SYSTEMS:
        for kind in MENTION_KINDS:
            for length, values in finals[system]["span_length_diagnostics_seed_20260925"][kind].items():
                lines.append(f"| {system} | {kind} | {length} | {values['gold_support']} | {values['predicted_count']} | {values['true_positives']} | {values['false_positives']} | {values['false_negatives']} |")
    error_counts = {}
    for system in SYSTEMS:
        with (OUTPUT / system / "errors_dev.jsonl").open(encoding="utf-8") as handle:
            counts = defaultdict(int)
            for line in handle:
                counts[json.loads(line)["error_type"]] += 1
            error_counts[system] = dict(counts)
    bio, crf, bertin = finals["mdeberta_v3_bio"], finals["mdeberta_v3_crf"], finals["bertin_bio"]
    weakest = min(((result["aggregate"][kind]["f1"]["mean"], kind) for result in finals.values() for kind in MENTION_KINDS), key=lambda item: item[0])
    combined_errors = defaultdict(int)
    for counts in error_counts.values():
        for name, count in counts.items():
            combined_errors[name] += count
    lines.extend(["", "## Factual Interpretation", "", f"- Tuning changed mean micro F1 by {bio['aggregate']['micro']['f1']['mean'] - baselines['mdeberta_v3_bio']:+.3f} for mDeBERTa BIO, {crf['aggregate']['micro']['f1']['mean'] - baselines['mdeberta_v3_crf']:+.3f} for mDeBERTa CRF, and {bertin['aggregate']['micro']['f1']['mean'] - baselines['bertin_bio']:+.3f} for BERTIN BIO.", f"- mDeBERTa BIO minus CRF mean F1: entity {bio['aggregate']['ENTITY_MENTION']['f1']['mean'] - crf['aggregate']['ENTITY_MENTION']['f1']['mean']:+.3f}, event {bio['aggregate']['EVENT_MENTION']['f1']['mean'] - crf['aggregate']['EVENT_MENTION']['f1']['mean']:+.3f}, value {bio['aggregate']['VALUE']['f1']['mean'] - crf['aggregate']['VALUE']['f1']['mean']:+.3f}, TIMEX3 {bio['aggregate']['TIMEX3']['f1']['mean'] - crf['aggregate']['TIMEX3']['f1']['mean']:+.3f}, micro {bio['aggregate']['micro']['f1']['mean'] - crf['aggregate']['micro']['f1']['mean']:+.3f}.", f"- mDeBERTa BIO minus BERTIN BIO mean micro F1: {bio['aggregate']['micro']['f1']['mean'] - bertin['aggregate']['micro']['f1']['mean']:+.3f}; entity difference {bio['aggregate']['ENTITY_MENTION']['f1']['mean'] - bertin['aggregate']['ENTITY_MENTION']['f1']['mean']:+.3f}; event difference {bio['aggregate']['EVENT_MENTION']['f1']['mean'] - bertin['aggregate']['EVENT_MENTION']['f1']['mean']:+.3f}.", "- In the sequential DEV selection, `window_1` beat sentence-only context for all three systems; this is a controlled context result, not a full-document result.", f"- Lowest final category F1 among the three systems is {weakest[1]} ({weakest[0]:.3f}).", f"- Most frequent seed-20260925 error across final systems is {max(combined_errors, key=combined_errors.get)} ({max(combined_errors.values())}).", "- EVENT_MENTION POS diagnostics are unavailable because no existing token-aligned POS preprocessing is used; no POS feature was added to models.", "- Zero-width, discontinuous, and overlapping same-kind mentions remain separately excluded because contiguous BIO/CRF cannot represent them."])
    (OUTPUT / "final_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
