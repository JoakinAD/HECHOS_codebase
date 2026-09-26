# DEV Metrics

This is an initial T3.1 baseline, not final thesis evaluation.

## Mention Detection

| Type | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ENTITY_MENTION | 0.621 | 0.415 | 0.497 | 576 |
| EVENT_MENTION | 0.606 | 0.434 | 0.506 | 389 |
| VALUE | 0.242 | 0.151 | 0.186 | 53 |
| TIMEX3 | 0.422 | 0.559 | 0.481 | 68 |
| micro | 0.577 | 0.418 | 0.485 | 1086 |

## Event Attributes (Gold Event Spans)

| Attribute | Accuracy | Macro F1 |
|---|---:|---:|
| aspect | 0.645 | 0.157 |
| certainty | 0.882 | 0.234 |
| modality | 0.992 | 0.249 |
| polarity | 0.913 | 0.318 |
| pos | 0.483 | 0.163 |
| pred | 0.995 | 0.499 |
| special_cases | 0.928 | 0.193 |
| tense | 0.286 | 0.056 |
| time | 0.824 | 0.301 |

## Relation Extraction (Oracle Gold Nodes)

| Type | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| CLINK | 0.023 | 0.333 | 0.043 | 3 |
| GLINK | 0.120 | 0.543 | 0.196 | 46 |
| HAS_PARTICIPANT | 0.102 | 0.498 | 0.170 | 287 |
| SLINK | 0.113 | 0.677 | 0.194 | 31 |
| TLINK | 0.120 | 0.393 | 0.184 | 308 |
| micro | 0.109 | 0.461 | 0.177 | 675 |

## Within-Document Coreference (Gold Mentions)

| Task | MUC F1 | B3 F1 | CEAF_e F1 | CoNLL F1 |
|---|---:|---:|---:|---:|
| entity_coreference | 0.501 | 0.625 | 0.509 | 0.545 |
| event_coreference | 0.480 | 0.927 | 0.891 | 0.766 |

## Graph Diagnostics

- Node exact-span micro F1: 0.485
- Oracle edge micro F1: 0.177
- Predicted/gold nodes: 787/1088
- Oracle predicted/gold edges: 2876/680

## Split Limitations

MEANTIME Spanish has corpus-level entity/event identity links. Entity links and combined entity/event links form globally connected graphs, so strict connected-component splitting is impossible. This experiment instead evaluates document-level extraction only. Corpus instance IDs are never exposed to the model. Cross-document coreference is intentionally excluded from current metrics; final T3.2 evaluation requires a story/event-cluster-level protocol.

The deterministic document split uses seed `20260925`: 84 train, 18 DEV, and 18 untouched TEST documents.

## Deferred

- CROSS_DOCUMENT_ENTITY_COREFERENCE: DEFERRED
- CROSS_DOCUMENT_EVENT_COREFERENCE: DEFERRED
- TIMEX normalization: unresolved because HeidelTime is not installed in the project environment; no values were invented.
