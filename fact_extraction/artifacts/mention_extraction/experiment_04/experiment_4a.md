# Experiment 04A: Frozen TRAIN-CV Thresholds

Thresholds were selected only from four-fold out-of-fold TRAIN predictions; DEV was not used for selection.

| Type | Threshold | Precision Before | Precision After | Recall Before | Recall After | F1 Before | F1 After |
|---|---:|---:|---:|---:|---:|---:|---:|
| ENTITY_MENTION | 0.8 | 0.596 | 0.673 | 0.614 | 0.551 | 0.605 | 0.606 |
| EVENT_MENTION | 0.0 | 0.676 | 0.676 | 0.695 | 0.695 | 0.685 | 0.685 |
| VALUE | 0.8 | 0.536 | 0.675 | 0.566 | 0.509 | 0.550 | 0.581 |
| TIMEX3 | 0.8 | 0.873 | 0.941 | 0.706 | 0.706 | 0.780 | 0.807 |

Micro F1: 0.647 -> 0.654.
False positives removed: 78; true positives removed: 29.
