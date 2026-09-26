# Mention Extraction Experiment 01

Encoder screening only: all systems use sentence-level MEANTIME input, identical labels, split, seed, and optimization settings. Original Longformer is English-pretrained. The Spanish Longformer checkpoint has a RoBERTa config with an attention-window field; this is reported as published rather than assumed to be native Longformer attention.

| Encoder | Decoder | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro F1 | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| XLM-R | BIO | 0.483 | 0.533 | 0.342 | 0.532 | 0.497 | 0.473 |
| XLM-R | CRF | 0.478 | 0.541 | 0.386 | 0.510 | 0.499 | 0.479 |
| Longformer | BIO | 0.438 | 0.519 | 0.416 | 0.469 | 0.469 | 0.460 |
| Longformer | CRF | 0.422 | 0.525 | 0.381 | 0.494 | 0.465 | 0.455 |
| BETO | BIO | 0.436 | 0.401 | 0.517 | 0.528 | 0.435 | 0.471 |
| BETO | CRF | 0.461 | 0.537 | 0.505 | 0.563 | 0.501 | 0.516 |
| BERTIN | BIO | 0.481 | 0.517 | 0.505 | 0.541 | 0.502 | 0.511 |
| BERTIN | CRF | 0.462 | 0.509 | 0.393 | 0.558 | 0.483 | 0.480 |
| mDeBERTa-v3 | BIO | 0.465 | 0.567 | 0.447 | 0.604 | 0.513 | 0.521 |
| mDeBERTa-v3 | CRF | 0.458 | 0.547 | 0.434 | 0.622 | 0.503 | 0.515 |
| Spanish Longformer | BIO | 0.427 | 0.530 | 0.391 | 0.549 | 0.470 | 0.474 |
| Spanish Longformer | CRF | 0.405 | 0.525 | 0.384 | 0.540 | 0.456 | 0.464 |

## Encoder Summary

| Encoder | Best BIO/CRF Decoder | Best Micro F1 |
|---|---|---:|
| XLM-R (multilingual) | CRF | 0.499 |
| Longformer (English) | BIO | 0.469 |
| BETO (Spanish-specific BERT) | CRF | 0.501 |
| BERTIN (Spanish-specific RoBERTa) | BIO | 0.502 |
| mDeBERTa-v3 (multilingual DeBERTa) | BIO | 0.513 |
| Spanish Longformer (Spanish-adapted Longformer checkpoint) | BIO | 0.470 |
