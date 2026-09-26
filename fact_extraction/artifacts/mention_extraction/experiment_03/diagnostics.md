# Mention Extraction Experiment 03 Diagnostics

Locked baseline is mDeBERTa-v3 BIO with LR 3e-5, batch 8, dropout 0.2, window_1, and Experiment 02 early stopping.

| Mention Type | Gold | Predicted | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| ENTITY_MENTION | 414 | 509 | 263 | 246 | 151 |
| EVENT_MENTION | 384 | 293 | 211 | 82 | 173 |
| VALUE | 53 | 49 | 27 | 22 | 26 |
| TIMEX3 | 68 | 60 | 40 | 20 | 28 |

| Error Type | Count |
|---|---:|
| missed mention | 280 |
| spurious mention | 370 |
| left-boundary error | 38 |
| right-boundary error | 38 |
| both-boundaries wrong | 22 |

VALUE TRAIN labels: {"O": 29999, "B_VALUE": 272, "I_VALUE": 317}. O-to-positive imbalance justifies M3B weighted cross-entropy for VALUE only.

Boundary errors are 98/748 (13.1%); syntax refinement was skipped because no aligned dependency representation exists and boundary errors are not dominant.
