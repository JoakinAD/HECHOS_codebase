# Experiment 04 Final Comparison

Thresholds are not combined with maximum context because they were calibrated from W2@256 out-of-fold confidence distributions.

## Experiment 4A

| System | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| window_2 @ 256 | 0.605 | 0.685 | 0.550 | 0.780 | 0.647 | - |
| + frozen TRAIN-CV thresholds | 0.606 | 0.685 | 0.581 | 0.807 | 0.654 | +0.006 |

## Experiment 4B

| System | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| window_2 @ 512 | 0.582 | 0.660 | 0.535 | 0.756 | 0.623 | - |
| maximum centered @ 512 | 0.602 | 0.787 | 0.704 | 0.830 | 0.701 | +0.078 |

## Three-Seed Final

| Final System | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Micro SD |
|---|---:|---:|---:|---:|---:|---:|
| Experiment 03 M3-FINAL | 0.572 | 0.696 | 0.585 | 0.764 | 0.637 | 0.013 |
| M4B-MAX | 0.629 | 0.786 | 0.662 | 0.820 | 0.709 | 0.022 |
