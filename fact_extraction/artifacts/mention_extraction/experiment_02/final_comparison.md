# Mention Extraction Experiment 02 Final Comparison

Values are means over deterministic seeds 20260925, 20260926, and 20260927. TEST was not evaluated.

| System | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| mdeberta_v3_bio | 0.526 | 0.603 | 0.469 | 0.610 | 0.560 | 0.552 |
| mdeberta_v3_crf | 0.516 | 0.598 | 0.465 | 0.627 | 0.556 | 0.551 |
| bertin_bio | 0.483 | 0.570 | 0.431 | 0.578 | 0.523 | 0.516 |

| System | LR | Batch | Dropout | Context | Best Epoch |
|---|---:|---:|---:|---|---:|
| mdeberta_v3_bio | 3e-05 | 8 | 0.2 | window_1 | 9 |
| mdeberta_v3_crf | 2e-05 | 8 | 0.1 | window_1 | 7 |
| bertin_bio | 2e-05 | 8 | 0.1 | window_1 | 3 |

## Experiment 01 Difference

| System | Experiment 01 Micro F1 | Experiment 02 Mean Micro F1 | Absolute Difference |
|---|---:|---:|---:|
| mdeberta_v3_bio | 0.513 | 0.560 | +0.047 |
| mdeberta_v3_crf | 0.503 | 0.556 | +0.053 |
| bertin_bio | 0.502 | 0.523 | +0.022 |

## Span Length Diagnostics (Seed 20260925)

| System | Mention Type | Length | Gold | Predicted | TP | FP | FN |
|---|---|---|---:|---:|---:|---:|---:|
| mdeberta_v3_bio | ENTITY_MENTION | 1_token | 173 | 218 | 129 | 89 | 44 |
| mdeberta_v3_bio | ENTITY_MENTION | 2_tokens | 96 | 140 | 72 | 68 | 24 |
| mdeberta_v3_bio | ENTITY_MENTION | 3plus_tokens | 145 | 151 | 62 | 89 | 83 |
| mdeberta_v3_bio | EVENT_MENTION | 1_token | 381 | 292 | 210 | 82 | 171 |
| mdeberta_v3_bio | EVENT_MENTION | 2_tokens | 0 | 0 | 0 | 0 | 0 |
| mdeberta_v3_bio | EVENT_MENTION | 3plus_tokens | 3 | 1 | 1 | 0 | 2 |
| mdeberta_v3_bio | VALUE | 1_token | 18 | 18 | 13 | 5 | 5 |
| mdeberta_v3_bio | VALUE | 2_tokens | 15 | 17 | 6 | 11 | 9 |
| mdeberta_v3_bio | VALUE | 3plus_tokens | 20 | 14 | 8 | 6 | 12 |
| mdeberta_v3_bio | TIMEX3 | 1_token | 17 | 15 | 11 | 4 | 6 |
| mdeberta_v3_bio | TIMEX3 | 2_tokens | 13 | 15 | 6 | 9 | 7 |
| mdeberta_v3_bio | TIMEX3 | 3plus_tokens | 38 | 30 | 23 | 7 | 15 |
| mdeberta_v3_crf | ENTITY_MENTION | 1_token | 173 | 174 | 99 | 75 | 74 |
| mdeberta_v3_crf | ENTITY_MENTION | 2_tokens | 96 | 96 | 60 | 36 | 36 |
| mdeberta_v3_crf | ENTITY_MENTION | 3plus_tokens | 145 | 118 | 51 | 67 | 94 |
| mdeberta_v3_crf | EVENT_MENTION | 1_token | 381 | 423 | 250 | 173 | 131 |
| mdeberta_v3_crf | EVENT_MENTION | 2_tokens | 0 | 0 | 0 | 0 | 0 |
| mdeberta_v3_crf | EVENT_MENTION | 3plus_tokens | 3 | 0 | 0 | 0 | 3 |
| mdeberta_v3_crf | VALUE | 1_token | 18 | 17 | 10 | 7 | 8 |
| mdeberta_v3_crf | VALUE | 2_tokens | 15 | 10 | 5 | 5 | 10 |
| mdeberta_v3_crf | VALUE | 3plus_tokens | 20 | 10 | 7 | 3 | 13 |
| mdeberta_v3_crf | TIMEX3 | 1_token | 17 | 14 | 10 | 4 | 7 |
| mdeberta_v3_crf | TIMEX3 | 2_tokens | 13 | 13 | 8 | 5 | 5 |
| mdeberta_v3_crf | TIMEX3 | 3plus_tokens | 38 | 28 | 24 | 4 | 14 |
| bertin_bio | ENTITY_MENTION | 1_token | 173 | 215 | 112 | 103 | 61 |
| bertin_bio | ENTITY_MENTION | 2_tokens | 96 | 100 | 55 | 45 | 41 |
| bertin_bio | ENTITY_MENTION | 3plus_tokens | 145 | 130 | 43 | 87 | 102 |
| bertin_bio | EVENT_MENTION | 1_token | 381 | 301 | 193 | 108 | 188 |
| bertin_bio | EVENT_MENTION | 2_tokens | 0 | 0 | 0 | 0 | 0 |
| bertin_bio | EVENT_MENTION | 3plus_tokens | 3 | 0 | 0 | 0 | 3 |
| bertin_bio | VALUE | 1_token | 18 | 20 | 10 | 10 | 8 |
| bertin_bio | VALUE | 2_tokens | 15 | 11 | 6 | 5 | 9 |
| bertin_bio | VALUE | 3plus_tokens | 20 | 16 | 7 | 9 | 13 |
| bertin_bio | TIMEX3 | 1_token | 17 | 32 | 10 | 22 | 7 |
| bertin_bio | TIMEX3 | 2_tokens | 13 | 16 | 6 | 10 | 7 |
| bertin_bio | TIMEX3 | 3plus_tokens | 38 | 28 | 22 | 6 | 16 |

## Factual Interpretation

- Tuning changed mean micro F1 by +0.047 for mDeBERTa BIO, +0.053 for mDeBERTa CRF, and +0.022 for BERTIN BIO.
- mDeBERTa BIO minus CRF mean F1: entity +0.009, event +0.005, value +0.004, TIMEX3 -0.017, micro +0.004.
- mDeBERTa BIO minus BERTIN BIO mean micro F1: +0.037; entity difference +0.043; event difference +0.032.
- In the sequential DEV selection, `window_1` beat sentence-only context for all three systems; this is a controlled context result, not a full-document result.
- Lowest final category F1 among the three systems is VALUE (0.431).
- Most frequent seed-20260925 error across final systems is spurious mention (1154).
- EVENT_MENTION POS diagnostics are unavailable because no existing token-aligned POS preprocessing is used; no POS feature was added to models.
- Zero-width, discontinuous, and overlapping same-kind mentions remain separately excluded because contiguous BIO/CRF cannot represent them.
