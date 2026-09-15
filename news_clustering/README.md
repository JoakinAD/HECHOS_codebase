# News clustering experiment

This prototype compares simultaneous news coverage as news-story clustering. The multilingual benchmark is suitable for validation, but its story labels are not asserted to be identical to the thesis's stricter physical-event definition; that requires manually annotated HECHOS data.

## Setup

```bash
python -m pip install -r requirements.txt
python -m spacy download es_core_news_md
python cluster_documents.py --config config.json
```

Copy and edit `config.example.json`; its values are starting values, not optimal or paper-derived optima. Use `dataset.dev.json` for selecting every configuration choice, freeze it, then perform one held-out evaluation using `dataset.test.json`.

## Inputs and encoders

The `priberam` adapter loads `id`, `title`, `text`, `date`, `source`, and `cluster`, filtering `lang == "spa"` by default. `cluster` is evaluation only. It never uses `event_id`, `bag_id`, `duplicate`, source, link, or bias as clustering inputs. The `hechos` adapter maps `headline`, `body`, `DD-MM-YYYY` date, and `newspaper`; link and bias are output metadata only and are not evaluated unless an annotated `cluster` field is supplied.

Both TF-IDF and the encoder use exactly `title + "\n" + text`. TF-IDF is L2-normalized word unigrams. The registry contains only reproducible identifiers:

| Key | Model |
|---|---|
| `distiluse_multilingual` | `sentence-transformers/distiluse-base-multilingual-cased-v2` |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` |
| `distilroberta` | `sentence-transformers/all-distilroberta-v1` |

## Two separate controlled comparisons

| Representation | HAC | HDBSCAN | BIRCH |
|---|---:|---:|---:|
| `hybrid_distance` | yes | yes | no |
| `concat_features` | no | yes | yes |

Experiment A is `D_hybrid -> HAC vs HDBSCAN`. Separate cosine-distance matrices are combined as `D_hybrid = ws*D_semantic + wl*D_tfidf + wt*D_time`, where `D_time = 1 - exp(-((delta_days - mu_days)^2)/(2*sigma_days^2))`. HAC uses complete linkage and a distance threshold; HDBSCAN receives exactly this precomputed matrix. Desai & Nagwanshi motivate this weighted distance-matrix fusion, which must not be conflated with concatenation.

Experiment B is `X_concat -> HDBSCAN vs BIRCH`. It requires `mu_days == 0` and constructs `z_i = (t_i-t_ref)/(sqrt(2)*sigma)`, then:

`x_i = [sqrt(ws/2)e_i || sqrt(wl/2)v_i || sqrt(wt)z_i]`.

HDBSCAN (Euclidean) and BIRCH (`n_clusters=None`) receive the same sparse CSR matrix. Its squared Euclidean difference is `ws*D_semantic + wl*D_tfidf + wt*delta_days^2/(2*sigma^2)`. This is intentionally not the bounded Gaussian temporal distance in Experiment A, so `X_concat` is not mathematically equivalent to `D_hybrid`. Cross-representation HAC-vs-BIRCH results are not a pure algorithm-only comparison.

Kömeçoğlu & Yilmaz compare BIRCH and HDBSCAN on a common Node2Vec-derived event vector. This implementation follows the same controlled-comparison principle but uses a different common vector representation: weighted concatenation of normalized Sentence-Transformer, TF-IDF, and temporal features. It is therefore not a reproduction of their event-graph representation.

## Evaluation and outputs

Gold runs report AMI, ARI, B-Cubed precision/recall/F1, cluster counts, singleton statistics, and document count. Raw HDBSCAN noise (`-1`) becomes a distinct predicted singleton per document for evaluation and reporting, never one shared cluster.

Each run creates `outputs/<run_name>/` with `config.json`, `metrics.json`, `assignments.json`, UTF-8 `clusters.txt`, and `clusters_graph.html`. Existing run directories cause an error. The report's `Tema: Entidades(...)` comes from post-clustering `es_core_news_md` NER, is document-presence based, and never affects representations or assignments. The graph has synthetic cluster hubs and membership edges only.
