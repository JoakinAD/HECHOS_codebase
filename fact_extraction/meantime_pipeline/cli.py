# fact_extraction/meantime_pipeline/cli.py
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform

from .baseline import MENTION_KINDS, gold_mentions, run
from .parser import parse_corpus, sentence_tokens, validate_document
from .splits import SPLIT_SEED, identity_connectivity_diagnostics, make_document_split, split_diagnostics


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = PACKAGE_ROOT.parent / "MEANTIME" / "v2" / "meantime_newsreader_spanish_nov15" / "intra_cross-doc_annotation"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def corpus_statistics(documents):
    counts = Counter()
    attributes = defaultdict(Counter)
    relation_types = Counter()
    discontinuous = Counter()
    zero_width = Counter()
    instance_docs = {"ENTITY": defaultdict(set), "EVENT": defaultdict(set)}
    for document in documents:
        counts["sentences"] += len({token.sentence_id for token in document.tokens})
        counts["tokens"] += len(document.tokens)
        for mention in document.mentions:
            counts[mention.kind] += 1
            for key, value in mention.attributes.items():
                attributes[f"{mention.kind}.{key}"][value] += 1
            if mention.kind in {"ENTITY_MENTION", "EVENT_MENTION", "TIMEX3", "VALUE", "SIGNAL", "C-SIGNAL"} and not mention.token_ids:
                zero_width[mention.kind] += 1
            elif mention.kind in {"ENTITY_MENTION", "EVENT_MENTION", "TIMEX3", "VALUE", "SIGNAL", "C-SIGNAL"} and [int(x) for x in mention.token_ids] != list(range(int(mention.token_ids[0]), int(mention.token_ids[-1]) + 1)):
                discontinuous[mention.kind] += 1
            if mention.kind in instance_docs and mention.attributes.get("instance_id"):
                instance_docs[mention.kind][mention.attributes["instance_id"]].add(document.id)
        relation_types.update(relation.type for relation in document.relations)
    return {
        "canonical_input": "intra_cross-doc_annotation",
        "documents": len(documents), **dict(counts), "relation_types": dict(relation_types),
        "event_attribute_values": {key: dict(value) for key, value in attributes.items() if key.startswith("EVENT_MENTION.")},
        "zero_width_mentions": dict(zero_width), "discontinuous_mentions": dict(discontinuous),
        "shared_instance_ids_across_documents": {kind: sum(len(docs) > 1 for docs in values.values()) for kind, values in instance_docs.items()},
        "validation_errors": [error for document in documents for error in validate_document(document)],
    }


def write_notes(output: Path, stats: dict) -> None:
    output.write_text(f"""# Spanish MEANTIME v2 Dataset Notes

Canonical input is `intra_cross-doc_annotation/`, containing {stats['documents']} Spanish XML documents, {stats['sentences']} sentences, and {stats['tokens']} tokens. The parallel `intra-doc_annotation/` view is not used by this pipeline.

## XML schema observed

Each `Document` stores `doc_id`, `doc_name`, language, optional URL, ordered `token` elements, `Markables`, and `Relations`. Tokens have document-local `t_id`, zero-based `number`, sentence ID, and surface string. No character offsets occur in this release.

Markables include entity, event, temporal, value, signal, and abstract entity/event nodes: `ENTITY_MENTION`, `EVENT_MENTION`, `TIMEX3`, `VALUE`, `SIGNAL`, `C-SIGNAL`, `ENTITY`, and `EVENT`. `ENTITY` and `EVENT` nodes can retain corpus `instance_id` values. All original attributes and token anchors are retained by the parser.

Relations observed are `REFERS_TO`, `HAS_PARTICIPANT`, `TLINK`, `SLINK`, `GLINK`, and `CLINK`; direction is retained exactly. `HAS_PARTICIPANT` includes a PropBank `sem_role`. The distributed XSD omits `GLINK` and `CLINK`, although both occur in actual XML.

## Span behavior

Most anchors are contiguous, but this is not guaranteed. Counts are in `dataset_stats.json`. There are {stats['zero_width_mentions'].get('ENTITY_MENTION', 0)} zero-width entity mentions ({stats['zero_width_mentions'].get('ENTITY_MENTION', 0) / stats['ENTITY_MENTION']:.2%} of entity mentions), treated as unsupported implicit mentions by the initial token-span detector and reported separately.

## Scope

Corpus instance IDs are preserved for future T3.2 work but never supplied as model features or labels. T3.1 evaluates document-local extraction and document-local `REFERS_TO` only. Cross-document entity and event coreference are deferred.
""", encoding="utf-8")


def markdown_metrics(metrics: dict, diagnostics: dict) -> str:
    lines = ["# DEV Metrics", "", "This is an initial T3.1 baseline, not final thesis evaluation.", "", "## Mention Detection", "", "| Type | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for kind, value in metrics["mention_detection"].items():
        lines.append(f"| {kind} | {value['precision']:.3f} | {value['recall']:.3f} | {value['f1']:.3f} | {value['support']} |")
    lines.extend(["", "## Event Attributes (Gold Event Spans)", "", "| Attribute | Accuracy | Macro F1 |", "|---|---:|---:|"])
    for attribute, value in metrics["event_attributes_gold_spans"].items():
        lines.append(f"| {attribute} | {value['accuracy']:.3f} | {value['macro_f1']:.3f} |")
    lines.extend(["", "## Relation Extraction (Oracle Gold Nodes)", "", "| Type | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
    for name, value in metrics["relations_oracle_gold_nodes"].items():
        if isinstance(value, dict) and "f1" in value:
            lines.append(f"| {name} | {value['precision']:.3f} | {value['recall']:.3f} | {value['f1']:.3f} | {value['support']} |")
    lines.extend(["", "## Within-Document Coreference (Gold Mentions)", "", "| Task | MUC F1 | B3 F1 | CEAF_e F1 | CoNLL F1 |", "|---|---:|---:|---:|---:|"])
    for task in ("entity_coreference", "event_coreference"):
        value = metrics[task]["gold_mention_within_document"]
        lines.append(f"| {task} | {value['MUC']['f1']:.3f} | {value['B3']['f1']:.3f} | {value['CEAF_e']['f1']:.3f} | {value['CoNLL_F1']:.3f} |")
    graph = metrics["graph_diagnostics"]
    lines.extend(["", "## Graph Diagnostics", "", f"- Node exact-span micro F1: {graph['node_exact_span']['f1']:.3f}", f"- Oracle edge micro F1: {graph['edge_oracle']['f1']:.3f}", f"- Predicted/gold nodes: {graph['predicted_nodes']}/{graph['gold_nodes']}", f"- Oracle predicted/gold edges: {graph['predicted_edges_oracle']}/{graph['gold_edges']}"])
    lines.extend(["", "## Split Limitations", "", "MEANTIME Spanish has corpus-level entity/event identity links. Entity links and combined entity/event links form globally connected graphs, so strict connected-component splitting is impossible. This experiment instead evaluates document-level extraction only. Corpus instance IDs are never exposed to the model. Cross-document coreference is intentionally excluded from current metrics; final T3.2 evaluation requires a story/event-cluster-level protocol.", "", f"The deterministic document split uses seed `{SPLIT_SEED}`: {diagnostics['document_counts']['train']} train, {diagnostics['document_counts']['dev']} DEV, and {diagnostics['document_counts']['test']} untouched TEST documents.", "", "## Deferred", "", "- CROSS_DOCUMENT_ENTITY_COREFERENCE: DEFERRED", "- CROSS_DOCUMENT_EVENT_COREFERENCE: DEFERRED", "- TIMEX normalization: unresolved because HeidelTime is not installed in the project environment; no values were invented.", ""])
    return "\n".join(lines)


def errors(predicted, gold, documents):
    pred_index = {(x["document_id"], x.get("kind"), tuple(x.get("token_ids", []))): x for x in predicted if "kind" in x}
    gold_index = {(x["document_id"], x["kind"], tuple(x["token_ids"])): x for x in gold}
    sentence_text = {}
    for document in documents:
        for sentence_id, tokens in sentence_tokens(document).items():
            sentence_text[(document.id, sentence_id)] = " ".join(token.text for token in tokens)
    records = []
    for key, item in gold_index.items():
        if key not in pred_index:
            records.append({"document_id": item["document_id"], "sentence_id": item["sentence_id"], "sentence_text": sentence_text[(item["document_id"], item["sentence_id"])], "task": "mention_detection", "gold_item": item, "predicted_item": None, "confidence": None, "error_type": "missed span"})
    for key, item in pred_index.items():
        if key not in gold_index:
            records.append({"document_id": item["document_id"], "sentence_id": item["sentence_id"], "sentence_text": sentence_text[(item["document_id"], item["sentence_id"])], "task": "mention_detection", "gold_item": None, "predicted_item": item, "confidence": item["confidence"], "error_type": "spurious span"})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inspect", "split", "run"])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT)
    args = parser.parse_args()
    documents = parse_corpus(args.corpus)
    if args.command == "inspect":
        stats = corpus_statistics(documents)
        write_json(args.output / "dataset_stats.json", stats)
        write_notes(args.output / "DATASET_NOTES.md", stats)
        return
    splits = make_document_split(documents)
    diagnostics = split_diagnostics(documents, splits)
    write_json(args.output / "splits.json", {"seed": SPLIT_SEED, "policy": "document-level T3.1", "splits": splits, "diagnostics": diagnostics, "corpus_identity_diagnostics": identity_connectivity_diagnostics(documents)})
    if args.command == "split":
        return
    by_id = {document.id: document for document in documents}
    metrics, predictions, gold = run([by_id[x] for x in splits["train"]], [by_id[x] for x in splits["dev"]], args.output)
    write_json(args.output / "metrics_dev.json", metrics)
    (args.output / "metrics_dev.md").write_text(markdown_metrics(metrics, diagnostics), encoding="utf-8")
    with (args.output / "predictions_dev.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    with (args.output / "errors_dev.jsonl").open("w", encoding="utf-8") as handle:
        error_records = errors(predictions, gold, [by_id[x] for x in splits["dev"]])
        for error in error_records:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    error_counts = Counter(record["error_type"] for record in error_records)
    (args.output / "ERROR_SUMMARY.md").write_text("# DEV Error Summary\n\n" + "\n".join(f"- {name}: {count}" for name, count in sorted(error_counts.items())) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
