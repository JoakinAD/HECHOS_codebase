# Experiment 04B: Token Budget And Context

| Model | Entity F1 | Event F1 | Value F1 | TIMEX3 F1 | Micro P | Micro R | Micro F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M4B0-W2 @ 256 historical | 0.605 | 0.685 | 0.550 | 0.780 | 0.643 | 0.652 | 0.647 | 0.655 |
| M4B-W2 @ 512 | 0.582 | 0.660 | 0.535 | 0.756 | 0.592 | 0.657 | 0.623 | 0.633 |
| M4B-MAX @ 512 | 0.602 | 0.787 | 0.704 | 0.830 | 0.704 | 0.699 | 0.701 | 0.731 |

```json
{
  "effective_historical_max_length": 256,
  "maximum_centered": {
    "context_truncated_percent": 63.10615989515072,
    "max_context_sentences": 17,
    "mean_encoded_length": 455.69069462647445,
    "mean_following_sentences": 4.815858453473132,
    "mean_previous_sentences": 5.951507208387942,
    "min_context_sentences": 3,
    "near_limit_percent": 51.8348623853211
  },
  "model_max_positions": 512,
  "window_2": {
    "count": 1526,
    "max": 256,
    "mean": 182.5137614678899,
    "mean_unused_at_256": 73.4862385321101,
    "median": 187.0,
    "truncated": 187
  }
}
```
