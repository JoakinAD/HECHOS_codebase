"""Configurable news-story clustering experiments.

Experiment A uses a hybrid pairwise distance; Experiment B uses concatenated
features. They deliberately have different temporal geometry.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize

ENCODERS = {
    "distiluse_multilingual": "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "distilroberta": "sentence-transformers/all-distilroberta-v1",
}
VALID = {
    "hybrid_distance": {"hac", "hdbscan"},
    "concat_features": {"hdbscan", "birch"},
}


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    date: datetime
    source: str | None
    gold_cluster: str | None
    link: str | None = None
    bias: str | None = None

    @property
    def content(self) -> str:
        return f"{self.title}\n{self.text}"


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing configuration field: {key}")
    return mapping[key]


def validate_config(config: dict[str, Any]) -> None:
    representation = _require(config, "representation")
    if representation not in VALID:
        raise ValueError(f"Unknown representation: {representation}")
    encoder = _require(config, "encoder")
    if encoder not in ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder}")
    clusterer = _require(config, "clusterer")
    if not isinstance(clusterer, dict):
        raise ValueError("clusterer must be an object")
    name = _require(clusterer, "name")
    if name not in {"hac", "hdbscan", "birch"}:
        raise ValueError(f"Unknown clusterer: {name}")
    if name not in VALID[representation]:
        raise ValueError(f"Incompatible combination: {representation} + {name}")
    weights = _require(config, "weights")
    values = [weights.get(key) for key in ("semantic", "tfidf", "temporal")]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("All weights must be finite non-negative numbers")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Weights must sum to 1.0")
    temporal = _require(config, "temporal")
    sigma = temporal.get("sigma_days")
    mu = temporal.get("mu_days")
    if not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("temporal.sigma_days must be finite and > 0")
    if not isinstance(mu, (int, float)) or not math.isfinite(mu):
        raise ValueError("temporal.mu_days must be finite")
    if representation == "concat_features" and not math.isclose(mu, 0.0, abs_tol=1e-12):
        raise ValueError("concat_features requires temporal.mu_days == 0")


def load_documents(dataset: dict[str, Any]) -> list[Document]:
    path = Path(_require(dataset, "path"))
    dataset_type = _require(dataset, "type")
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("Dataset root must be a JSON array")
    language = dataset.get("language")
    documents: list[Document] = []
    for row in rows:
        if dataset_type == "priberam":
            if language is not None and row.get("lang") != language:
                continue
            documents.append(Document(str(row["id"]), str(row["title"]), str(row["text"]),
                datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S"), row.get("source"),
                str(row["cluster"]) if row.get("cluster") is not None else None))
        elif dataset_type == "hechos":
            documents.append(Document(str(row["id"]), str(row["headline"]), str(row["body"]),
                datetime.strptime(row["date"], "%d-%m-%Y"), row.get("newspaper"),
                str(row["cluster"]) if row.get("cluster") is not None else None,
                row.get("link"), row.get("bias")))
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    if not documents:
        raise ValueError("No documents loaded after filtering")
    return documents


def build_tfidf(documents: list[Document]) -> sparse.csr_matrix:
    return TfidfVectorizer(norm="l2").fit_transform([document.content for document in documents]).tocsr()


def build_embeddings(documents: list[Document], encoder_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENCODERS[encoder_name])
    embeddings = model.encode([document.content for document in documents], normalize_embeddings=True, show_progress_bar=True)
    return normalize(np.asarray(embeddings, dtype=np.float32), norm="l2").astype(np.float32)


def cosine_distance_matrix(features: np.ndarray | sparse.spmatrix) -> np.ndarray:
    similarity = (features @ features.T).toarray() if sparse.issparse(features) else features @ features.T
    distance = 1.0 - np.clip(np.asarray(similarity, dtype=np.float32), -1.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def temporal_distance_matrix(documents: list[Document], mu_days: float, sigma_days: float) -> np.ndarray:
    days = np.array([(document.date - documents[0].date).total_seconds() / 86400.0 for document in documents])
    delta = np.abs(days[:, None] - days[None, :])
    distance = 1.0 - np.exp(-((delta - mu_days) ** 2) / (2.0 * sigma_days**2))
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32)


def hybrid_distance(semantic: np.ndarray, tfidf: sparse.csr_matrix, documents: list[Document], weights: dict[str, float], temporal: dict[str, float]) -> np.ndarray:
    result = (weights["semantic"] * cosine_distance_matrix(semantic) +
              weights["tfidf"] * cosine_distance_matrix(tfidf) +
              weights["temporal"] * temporal_distance_matrix(documents, temporal["mu_days"], temporal["sigma_days"]))
    result = np.asarray(result, dtype=np.float32)
    np.fill_diagonal(result, 0.0)
    if result.shape != (len(documents), len(documents)) or not np.isfinite(result).all() or not np.allclose(result, result.T, atol=1e-6):
        raise ValueError("Invalid hybrid distance matrix")
    return result


def concat_features(semantic: np.ndarray, tfidf: sparse.csr_matrix, documents: list[Document], weights: dict[str, float], sigma_days: float) -> sparse.csr_matrix:
    reference = min(document.date for document in documents)
    z = np.array([(document.date - reference).total_seconds() / 86400.0 for document in documents], dtype=np.float32)
    z /= math.sqrt(2.0) * sigma_days
    # hdbscan requires float64 for its sparse Euclidean implementation; both
    # Experiment B clusterers receive this exact common matrix.
    matrix = sparse.hstack([
        sparse.csr_matrix(semantic * math.sqrt(weights["semantic"] / 2.0)),
        tfidf * math.sqrt(weights["tfidf"] / 2.0),
        sparse.csr_matrix((z * math.sqrt(weights["temporal"])).reshape(-1, 1)),
    ], format="csr", dtype=np.float64)
    if not np.isfinite(matrix.data).all():
        raise ValueError("Invalid concatenated feature matrix")
    return matrix


def route_clusterer(representation: str, clusterer: dict[str, Any], data: np.ndarray | sparse.csr_matrix) -> np.ndarray:
    name = clusterer["name"]
    if name == "hac":
        return AgglomerativeClustering(metric="precomputed", linkage="complete", n_clusters=None,
            distance_threshold=clusterer["hac"]["distance_threshold"]).fit_predict(data)
    if name == "hdbscan":
        import hdbscan
        metric = "precomputed" if representation == "hybrid_distance" else "euclidean"
        params = clusterer["hdbscan"]
        # hdbscan's precomputed implementation requires a float64 buffer.
        if representation == "hybrid_distance":
            data = np.asarray(data, dtype=np.float64)
        return hdbscan.HDBSCAN(metric=metric, min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"]).fit_predict(data)
    params = clusterer["birch"]
    return Birch(threshold=params["threshold"], branching_factor=params["branching_factor"], n_clusters=None).fit_predict(data)


def normalize_noise_labels(labels: np.ndarray | list[int]) -> list[str]:
    result: list[str] = []
    for index, label in enumerate(labels):
        result.append(f"noise-{index}" if int(label) == -1 else str(int(label)))
    return result


def bcubed(gold: list[str], predicted: list[str]) -> tuple[float, float, float]:
    gold_groups: dict[str, set[int]] = defaultdict(set)
    predicted_groups: dict[str, set[int]] = defaultdict(set)
    for index, (gold_label, predicted_label) in enumerate(zip(gold, predicted)):
        gold_groups[gold_label].add(index)
        predicted_groups[predicted_label].add(index)
    precision = sum(len(gold_groups[g] & predicted_groups[p]) / len(predicted_groups[p]) for g, p in zip(gold, predicted)) / len(gold)
    recall = sum(len(gold_groups[g] & predicted_groups[p]) / len(gold_groups[g]) for g, p in zip(gold, predicted)) / len(gold)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(documents: list[Document], labels: list[str]) -> dict[str, Any] | None:
    if any(document.gold_cluster is None for document in documents):
        return None
    gold = [document.gold_cluster for document in documents]
    precision, recall, f1 = bcubed(gold, labels)
    sizes = Counter(labels)
    singleton_count = sum(size == 1 for size in sizes.values())
    return {"ami": float(adjusted_mutual_info_score(gold, labels)), "ari": float(adjusted_rand_score(gold, labels)),
        "bcubed_precision": precision, "bcubed_recall": recall, "bcubed_f1": f1,
        "gold_clusters": len(set(gold)), "predicted_clusters": len(sizes), "predicted_singletons": singleton_count,
        "predicted_singleton_percentage": 100.0 * singleton_count / len(sizes), "documents": len(documents)}


def cluster_entities(documents: list[Document]) -> list[str]:
    try:
        import spacy
        nlp = spacy.load("es_core_news_md")
    except OSError as error:
        raise RuntimeError("Install Spanish NER with: python -m spacy download es_core_news_md") from error
    counts: Counter[str] = Counter()
    for document in documents:
        counts.update({entity.text.strip().lower() for entity in nlp(document.content).ents if entity.text.strip()})
    return [entity for entity, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]]


def write_outputs(output: Path, config: dict[str, Any], documents: list[Document], labels: list[str], metrics: dict[str, Any] | None) -> None:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for document, label in zip(documents, labels): grouped[label].append(document)
    summaries: dict[str, dict[str, Any]] = {}
    for label, members in grouped.items():
        members.sort(key=lambda item: (item.date, item.id))
        entities = cluster_entities(members)
        summaries[label] = {"members": members, "entities": entities}
    with (output / "clusters.txt").open("w", encoding="utf-8") as handle:
        for label in sorted(grouped, key=lambda item: (grouped[item][0].date, item)):
            members, entities = summaries[label]["members"], summaries[label]["entities"]
            start, end = members[0].date.date(), members[-1].date.date()
            sources = dict(sorted(Counter(member.source for member in members if member.source).items()))
            topic = ", ".join(entities) if entities else "no detectadas"
            handle.write(f"🔹 Cluster {label} ({int(len(members))} noticias):\nFechas: {start} a {end} ({(end - start).days} días)\nPeriódicos: {sources}\nTema: Entidades({topic})\n\n")
            for member in members:
                handle.write(f"  - [{member.id}] [{member.source or 'Sin fuente'}] [{member.date:%Y-%m-%d}] {member.title}\n")
            handle.write("\n")
    assignments = [{"id": document.id, "predicted_cluster": label, "gold_cluster": document.gold_cluster,
        "title": document.title, "date": document.date.strftime("%Y-%m-%d"), "source": document.source,
        **({"link": document.link, "bias": document.bias} if document.link is not None or document.bias is not None else {})}
        for document, label in zip(documents, labels)]
    (output / "assignments.json").write_text(json.dumps(assignments, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_payload = {"representation": config["representation"], "clusterer": config["clusterer"]["name"], "encoder": config["encoder"], "metrics": metrics}
    (output / "metrics.json").write_text(json.dumps(metric_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_graph(output / "clusters_graph.html", summaries)


def write_graph(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    from pyvis.network import Network
    graph = Network(height="900px", width="100%", directed=False, notebook=False)
    for number, label in enumerate(sorted(summaries)):
        members, entities = summaries[label]["members"], summaries[label]["entities"]
        hub = f"cluster-{label}"
        dates = f"{members[0].date:%Y-%m-%d} a {members[-1].date:%Y-%m-%d}"
        graph.add_node(hub, label=f"Cluster {label}", shape="dot", size=30, color="#222222",
            title=f"cluster id: {label}<br>article count: {len(members)}<br>date range: {dates}<br>top entities: {', '.join(entities) or 'no detectadas'}")
        color = f"hsl({(number * 47) % 360}, 65%, 55%)"
        for member in members:
            node = f"article-{member.id}"
            graph.add_node(node, label=member.title[:80], shape="box", color=color,
                title=f"document id: {member.id}<br>source: {member.source or ''}<br>date: {member.date:%Y-%m-%d}<br>title: {member.title}")
            graph.add_edge(hub, node)
    graph.write_html(str(path), open_browser=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    # Resolve paths relative to the configuration file, making example configs portable.
    dataset_path = Path(config["dataset"]["path"])
    if not dataset_path.is_absolute(): config["dataset"]["path"] = str((config_path.parent / dataset_path).resolve())
    output_root = Path(config["output_dir"])
    if not output_root.is_absolute(): output_root = config_path.parent / output_root
    output = output_root / config["run_name"]
    if output.exists(): raise FileExistsError(f"Output directory already exists: {output}. Choose a new run_name or remove it deliberately.")
    documents = load_documents(config["dataset"])
    print(f"Dataset: {config['dataset']['path']}\nDocuments loaded: {len(documents)}\nLanguage filter: {config['dataset'].get('language', 'none')}\nEncoder: {config['encoder']}\nRepresentation: {config['representation']}\nWeights: semantic={config['weights']['semantic']}, tfidf={config['weights']['tfidf']}, temporal={config['weights']['temporal']}\nTemporal parameters: {config['temporal']}\nClusterer: {config['clusterer']['name']}\nGold labels available: {'yes' if all(d.gold_cluster is not None for d in documents) else 'no'}")
    print("Embedding..."); semantic = build_embeddings(documents, config["encoder"])
    print("TF-IDF..."); tfidf = build_tfidf(documents)
    print("Representation construction...")
    if config["representation"] == "hybrid_distance":
        data = hybrid_distance(semantic, tfidf, documents, config["weights"], config["temporal"])
        print(f"Hybrid matrix shape: {data.shape}")
    else:
        data = concat_features(semantic, tfidf, documents, config["weights"], config["temporal"]["sigma_days"])
        print(f"Feature matrix shape: {data.shape}; sparse: {sparse.issparse(data)}")
    print("Clustering..."); labels = normalize_noise_labels(route_clusterer(config["representation"], config["clusterer"], data))
    print("Evaluation..."); metrics = evaluate(documents, labels)
    if metrics: print(json.dumps(metrics, indent=2))
    output.mkdir(parents=True)
    print("Output generation..."); write_outputs(output, config, documents, labels, metrics)


if __name__ == "__main__":
    main()
