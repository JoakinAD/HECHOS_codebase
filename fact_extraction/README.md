# MEANTIME Spanish T3.1 Baseline

This package is an initial, inspectable document-level fact-extraction baseline for the professionally translated Spanish MEANTIME v2 corpus. It uses only `MEANTIME/v2/meantime_newsreader_spanish_nov15/intra_cross-doc_annotation/`; it does not read, import, or depend on `literature_pipeline`.

## Implemented

- Loss-preserving XML parser for tokens, all markables, abstract nodes, attributes, anchors, and directed relations.
- Deterministic document split: 84 TRAIN, 18 DEV, 18 untouched TEST, seed `20260925`.
- Shared `xlm-roberta-base` token encoder with four independent BIO heads for `ENTITY_MENTION`, `EVENT_MENTION`, `VALUE`, and `TIMEX3` spans.
- Train-only majority event-attribute baseline, gold-node relation candidate classifier, and lexical within-document coreference diagnostic.
- JSON metrics, predictions, error records, parser validation, and focused tests.

## Current Scope

The parser preserves `SIGNAL`, `C-SIGNAL`, abstract `ENTITY`/`EVENT`, and all relation types. The automatic baseline currently evaluates explicit contiguous spans. Zero-width entity mentions (implicit subjects) and discontinuous spans are retained in parsed gold but are not representable by the BIO decoder and are reported as exclusions.

`REFERS_TO` is retained in full. Within-document gold-mention coreference diagnostics are reported. `CROSS_DOCUMENT_ENTITY_COREFERENCE` and `CROSS_DOCUMENT_EVENT_COREFERENCE` are explicitly **DEFERRED**. Corpus `instance_id` values are never model inputs or training labels.

HeidelTime is not installed in the project virtual environment, so temporal normalization is intentionally unresolved. The Transformer does detect TIMEX3 spans; it does not invent normalized values.

## Architecture

`xlm-roberta-base` is fine-tuned with a shared contextual encoder and one 3-class BIO head per overlapping mention category. The relation baseline is a train-only one-vs-rest logistic candidate-pair classifier using surface/type/distance features, constrained to same or adjacent sentences. Event attributes use a train-only majority classifier to isolate the plumbing before attribute heads are tuned. Within-document coreference is currently an exact-normalized-surface diagnostic baseline.

spaCy is not used as NER. The current baseline does not require spaCy parsing because MEANTIME tokenization is authoritative; parser-feature alignment is deferred to the next tuning iteration.

## Reproduce

From the repository root:

```bash
PYTHONPATH=. fact_extraction/.venv/bin/python -m fact_extraction.run inspect
PYTHONPATH=. fact_extraction/.venv/bin/python -m fact_extraction.run split
PYTHONPATH=. fact_extraction/.venv/bin/python -m fact_extraction.run run
PYTHONPATH=. fact_extraction/.venv/bin/python -m unittest discover -s fact_extraction/tests -v
```

`run` trains only on TRAIN and writes DEV-only outputs. It never evaluates TEST. The model checkpoint is `artifacts/mention_extraction/checkpoints/mention_model.pt`; delete it deliberately to retrain.

## Mention Extraction Experiment 01

Run the controlled XLM-R/Longformer and BIO/CRF comparison with:

```bash
PYTHONPATH=. fact_extraction/.venv/bin/python -m fact_extraction.tasks.mention_extraction_experiment_01
```

Outputs are under `artifacts/mention_extraction/experiment_01/`. Longformer uses English-pretrained `allenai/longformer-base-4096`; its results therefore confound encoder architecture with pretraining language and should not be interpreted as a language-neutral architecture comparison.

## Outputs

- `artifacts/dataset/DATASET_NOTES.md`, `artifacts/dataset/dataset_stats.json`
- `artifacts/splits/splits.json`
- `artifacts/mention_extraction/metrics_dev.json`, `metrics_dev.md`
- `artifacts/mention_extraction/predictions_dev.jsonl`, `errors_dev.jsonl`

## Limitations And Next Steps

DEV values are baseline/tuning-stage results, not final thesis results. Corpus-level identity links connect the corpus globally, so this T3.1 document split permits shared real-world entities/events across splits but never exposes their IDs to models. T3.2 needs a story/event-cluster split for cross-document coreference.

Priority tuning work: add token-aligned Spanish parser features; model implicit subjects separately; add contextual event-attribute and relation heads; decode relations against predicted nodes for end-to-end relation metrics; integrate HeidelTime after verifying its Java resources; and replace lexical within-document coreference with trained contextual pair scoring.

Run metadata is recorded in `metrics_dev.json`: seed, model, epochs (3), batch size (4), learning rate (`2e-5`), device, and checkpoint criterion.
