# Experiment 03 Final Comparison

TEST was not evaluated. Values are three-seed means.

| Model | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Micro SD | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Experiment 02 mDeBERTa BIO | 0.526 | 0.603 | 0.469 | 0.610 | 0.560 | 0.031 | 0.552 |
| M3-FINAL (M3D) | 0.572 | 0.696 | 0.585 | 0.764 | 0.637 | 0.013 | 0.654 |

| Model | Micro Precision | Micro Recall |
|---|---:|---:|
| Experiment 02 mDeBERTa BIO | 0.577 | 0.545 |
| M3-FINAL (M3D) | 0.636 | 0.639 |

## Interpretation

- M3-FINAL is M3D (`window_2`) only. M3A thresholding also improved the locked DEV seed, but it was not combined with window_2 because that interaction was not independently tested.
- M3-FINAL improves the matched Experiment 02 seed in all three runs: +0.056, +0.074, +0.101 micro F1.
- M3B and M3C were not selected because they reduced locked-seed micro F1 despite their targeted motivations.
- M3A thresholds are DEV-selected and therefore its DEV score is a model-selection result, not an unbiased estimate.
