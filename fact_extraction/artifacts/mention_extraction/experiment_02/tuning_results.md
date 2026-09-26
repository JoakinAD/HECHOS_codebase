# Mention Extraction Experiment 02 Tuning

All selection used DEV micro exact-span F1 only. TEST was never evaluated.

## Experiment 01 Baselines

| System | Micro F1 |
|---|---:|
| mdeberta_v3_bio | 0.513 |
| mdeberta_v3_crf | 0.503 |
| bertin_bio | 0.502 |

## mdeberta_v3_bio

| Stage | Settings | Best Epoch | Micro F1 |
|---|---|---:|---:|
| batch_size | `{"batch_size": 4}` | 3 | 0.513 |
| batch_size | `{"batch_size": 8}` | 9 | 0.521 |
| context | `{"context": "sentence"}` | 6 | 0.516 |
| context | `{"context": "window_1"}` | 8 | 0.576 |
| dropout | `{"dropout": 0.1}` | 4 | 0.505 |
| dropout | `{"dropout": 0.2}` | 5 | 0.517 |
| duration | `{}` | 6 | 0.530 |
| learning_rate | `{"learning_rate": 1e-05}` | 7 | 0.510 |
| learning_rate | `{"learning_rate": 2e-05}` | 3 | 0.512 |
| learning_rate | `{"learning_rate": 3e-05}` | 4 | 0.514 |

## mdeberta_v3_crf

| Stage | Settings | Best Epoch | Micro F1 |
|---|---|---:|---:|
| batch_size | `{"batch_size": 4}` | 3 | 0.516 |
| batch_size | `{"batch_size": 8}` | 9 | 0.518 |
| context | `{"context": "sentence"}` | 8 | 0.530 |
| context | `{"context": "window_1"}` | 10 | 0.574 |
| dropout | `{"dropout": 0.1}` | 8 | 0.535 |
| dropout | `{"dropout": 0.2}` | 8 | 0.519 |
| duration | `{}` | 3 | 0.502 |
| learning_rate | `{"learning_rate": 1e-05}` | 5 | 0.498 |
| learning_rate | `{"learning_rate": 2e-05}` | 7 | 0.529 |
| learning_rate | `{"learning_rate": 3e-05}` | 3 | 0.512 |

## bertin_bio

| Stage | Settings | Best Epoch | Micro F1 |
|---|---|---:|---:|
| batch_size | `{"batch_size": 4}` | 5 | 0.502 |
| batch_size | `{"batch_size": 8}` | 7 | 0.502 |
| context | `{"context": "sentence"}` | 7 | 0.502 |
| context | `{"context": "window_1"}` | 10 | 0.543 |
| dropout | `{"dropout": 0.1}` | 7 | 0.502 |
| dropout | `{"dropout": 0.2}` | 4 | 0.491 |
| duration | `{}` | 5 | 0.502 |
| learning_rate | `{"learning_rate": 1e-05}` | 9 | 0.495 |
| learning_rate | `{"learning_rate": 2e-05}` | 5 | 0.502 |
| learning_rate | `{"learning_rate": 3e-05}` | 3 | 0.496 |
