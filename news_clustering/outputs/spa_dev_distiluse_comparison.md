# Untuned Development Baselines

These are untuned development baselines. The weights, temporal sigma, HAC threshold, HDBSCAN parameters, and BIRCH parameters have not been selected through systematic development-set tuning. No held-out test-set run was performed.

`hybrid_distance` compares HAC and HDBSCAN on the same `D_hybrid`. `concat_features` compares HDBSCAN and BIRCH on the same `X_concat`. Results across those representation families, including HAC versus BIRCH, are not a pure algorithm-only comparison.

| run_name | representation | clusterer | encoder | AMI | ARI | B-Cubed Precision | B-Cubed Recall | B-Cubed F1 | gold_clusters | predicted_clusters | predicted_singletons | documents_in_singleton_clusters | raw_noise_points | raw_noise_percentage | documents |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spa_dev_distiluse_distance_hac | hybrid_distance | hac | distiluse_multilingual | 0.861150 | 0.720057 | 0.980426 | 0.683111 | 0.805200 | 416 | 1020 | 421 | 421 | N/A | N/A | 4527 |
| spa_dev_distiluse_distance_hdbscan | hybrid_distance | hdbscan | distiluse_multilingual | 0.607328 | 0.318924 | 0.994460 | 0.400851 | 0.571386 | 416 | 1940 | 983 | 983 | 983 | 21.714159% | 4527 |
| spa_dev_distiluse_concat_hdbscan | concat_features | hdbscan | distiluse_multilingual | 0.652261 | 0.383228 | 0.988840 | 0.454281 | 0.622555 | 416 | 1762 | 853 | 853 | 853 | 18.842501% | 4527 |
| spa_dev_distiluse_concat_birch | concat_features | birch | distiluse_multilingual | 0.875245 | 0.800458 | 0.820596 | 0.828328 | 0.824444 | 416 | 630 | 235 | 235 | N/A | N/A | 4527 |
