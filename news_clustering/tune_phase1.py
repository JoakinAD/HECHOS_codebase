#news_clustering/tune_phase1.py
"""Development-only Phase 1 clusterer parameter tuning.

This runner deliberately creates only compact metrics and plots. It does not
call the full report, NER, graph, or assignment output code.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cluster_documents as clustering

PROJECT_DIR = Path(__file__).resolve().parent
RESULT_FIELDS = [
    "trial_id", "representation", "clusterer", "encoder", "w_semantic", "w_tfidf", "w_temporal",
    "sigma_days", "distance_threshold", "min_cluster_size", "min_samples", "birch_threshold",
    "branching_factor", "bcubed_precision", "bcubed_recall", "bcubed_f1", "ami", "ari",
    "gold_clusters", "predicted_clusters", "predicted_singletons", "documents_in_singleton_clusters",
    "raw_noise_points", "raw_noise_percentage", "runtime_seconds",
]
FAMILY_ORDER = [
    ("hybrid_distance", "hac"),
    ("hybrid_distance", "hdbscan"),
    ("concat_features", "hdbscan"),
    ("concat_features", "birch"),
]


def validate_tuning_dataset(path: Path) -> None:
    if path.name != "dataset.dev.json":
        raise ValueError("Tuning accepts only dataset.dev.json; dataset.test.json is not permitted")


def phase1_trials() -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        trials.append({"representation": "hybrid_distance", "clusterer": "hac", "distance_threshold": threshold})
    for representation in ("hybrid_distance", "concat_features"):
        for min_cluster_size in (2, 3, 5, 8, 10):
            for min_samples in (1, 2, 3, 5):
                trials.append({"representation": representation, "clusterer": "hdbscan",
                    "min_cluster_size": min_cluster_size, "min_samples": min_samples})
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        for branching_factor in (25, 50, 100):
            trials.append({"representation": "concat_features", "clusterer": "birch",
                "birch_threshold": threshold, "branching_factor": branching_factor})
    for number, trial in enumerate(trials, start=1):
        trial["trial_id"] = f"phase1-{number:03d}"
    return trials


def simplex_weights(step: float = 0.2) -> list[dict[str, float]]:
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1.0")
    return [{"semantic": semantic * step, "tfidf": tfidf * step, "temporal": temporal * step}
        for semantic in range(units + 1)
        for tfidf in range(units - semantic + 1)
        for temporal in [units - semantic - tfidf]]


def phase2_representation_grid() -> list[dict[str, Any]]:
    return [{"sigma_days": sigma, "weights": weights}
        for sigma in (1, 3, 7, 14) for weights in simplex_weights(0.2)]


def clusterer_config(trial: dict[str, Any]) -> dict[str, Any]:
    name = trial["clusterer"]
    return {"name": name,
        "hac": {"distance_threshold": trial.get("distance_threshold")},
        "hdbscan": {"min_cluster_size": trial.get("min_cluster_size"), "min_samples": trial.get("min_samples")},
        "birch": {"threshold": trial.get("birch_threshold"), "branching_factor": trial.get("branching_factor")}}


def evaluate_trial(documents: list[clustering.Document], trial: dict[str, Any], data: Any) -> dict[str, Any]:
    started = time.perf_counter()
    raw_labels = clustering.route_clusterer(trial["representation"], clusterer_config(trial), data)
    labels = clustering.normalize_noise_labels(raw_labels)
    raw_for_diagnostics = raw_labels if trial["clusterer"] == "hdbscan" else None
    metrics = clustering.evaluate(documents, labels, raw_for_diagnostics)
    if metrics is None:
        raise ValueError("Development tuning requires gold cluster labels")
    result = {field: None for field in RESULT_FIELDS}
    result.update({"trial_id": trial["trial_id"], "representation": trial["representation"],
        "clusterer": trial["clusterer"], "encoder": "distiluse_multilingual",
        "w_semantic": 0.5, "w_tfidf": 0.3, "w_temporal": 0.2, "sigma_days": 3.0,
        "distance_threshold": trial.get("distance_threshold"), "min_cluster_size": trial.get("min_cluster_size"),
        "min_samples": trial.get("min_samples"), "birch_threshold": trial.get("birch_threshold"),
        "branching_factor": trial.get("branching_factor"), "runtime_seconds": time.perf_counter() - started})
    result.update(metrics)
    return result


def rank_by_family(results: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    return {family: sorted((row for row in results if (row["representation"], row["clusterer"]) == family),
        key=lambda row: (-row["bcubed_f1"], row["trial_id"])) for family in FAMILY_ORDER}


def write_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "phase1_clusterers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({field: "N/A" if row[field] is None else row[field] for field in RESULT_FIELDS})
    (output_dir / "phase1_clusterers.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def _family_label(family: tuple[str, str]) -> str:
    return f"{family[0]} + {family[1]}"


def plot_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    ranked = rank_by_family(results)
    best = [ranked[family][0] for family in FAMILY_ORDER]
    labels = [_family_label(family) for family in FAMILY_ORDER]
    positions = np.arange(len(best))
    figure, axis = plt.subplots(figsize=(11, 6))
    width = 0.24
    for offset, metric, title in ((-width, "bcubed_f1", "B-Cubed F1"), (0, "ami", "AMI"), (width, "ari", "ARI")):
        axis.bar(positions + offset, [row[metric] for row in best], width, label=title)
    axis.set(xticks=positions, xticklabels=labels, ylim=(0, 1), ylabel="Score", title="Best Phase 1 development configuration per family")
    axis.tick_params(axis="x", rotation=18)
    axis.legend()
    figure.tight_layout(); figure.savefig(output_dir / "phase1_summary_metrics.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for family, color in zip(FAMILY_ORDER, colors):
        rows = ranked[family]
        axis.scatter([row["bcubed_recall"] for row in rows], [row["bcubed_precision"] for row in rows], label=_family_label(family), color=color, alpha=.7)
        selected = rows[0]
        axis.scatter(selected["bcubed_recall"], selected["bcubed_precision"], color=color, edgecolor="black", s=110, zorder=3)
        axis.annotate(selected["trial_id"], (selected["bcubed_recall"], selected["bcubed_precision"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(xlabel="B-Cubed Recall", ylabel="B-Cubed Precision", xlim=(0, 1), ylim=(0, 1), title="Phase 1 precision-recall trade-off")
    axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(output_dir / "phase1_precision_recall.png", dpi=160); plt.close(figure)

    hac = ranked[("hybrid_distance", "hac")]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot([row["distance_threshold"] for row in hac], [row["bcubed_f1"] for row in hac], marker="o")
    axis.set(xlabel="HAC distance threshold", ylabel="B-Cubed F1", ylim=(0, 1), title="HAC threshold response")
    figure.tight_layout(); figure.savefig(output_dir / "hac_threshold.png", dpi=160); plt.close(figure)

    birch = ranked[("concat_features", "birch")]
    figure, axis = plt.subplots(figsize=(7, 5))
    for branching_factor in (25, 50, 100):
        rows = sorted((row for row in birch if row["branching_factor"] == branching_factor), key=lambda row: row["birch_threshold"])
        axis.plot([row["birch_threshold"] for row in rows], [row["bcubed_f1"] for row in rows], marker="o", label=f"branching_factor={branching_factor}")
    axis.set(xlabel="BIRCH threshold", ylabel="B-Cubed F1", ylim=(0, 1), title="BIRCH threshold response")
    axis.legend(); figure.tight_layout(); figure.savefig(output_dir / "birch_threshold.png", dpi=160); plt.close(figure)

    for family, filename in ((("hybrid_distance", "hdbscan"), "hdbscan_distance_heatmap.png"), (("concat_features", "hdbscan"), "hdbscan_concat_heatmap.png")):
        rows = ranked[family]
        sizes, samples = (2, 3, 5, 8, 10), (1, 2, 3, 5)
        values = np.array([[next(row["bcubed_f1"] for row in rows if row["min_cluster_size"] == size and row["min_samples"] == sample) for sample in samples] for size in sizes])
        figure, axis = plt.subplots(figsize=(6, 6))
        image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis")
        axis.set(xticks=range(len(samples)), xticklabels=samples, yticks=range(len(sizes)), yticklabels=sizes,
            xlabel="min_samples", ylabel="min_cluster_size", title=f"HDBSCAN B-Cubed F1: {family[0]}")
        for row in range(len(sizes)):
            for column in range(len(samples)): axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", color="white" if values[row, column] < .55 else "black", fontsize=8)
        figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(output_dir / filename, dpi=160); plt.close(figure)


def run_phase1() -> list[dict[str, Any]]:
    dataset_path = PROJECT_DIR / "dataset" / "dataset.dev.json"
    validate_tuning_dataset(dataset_path)
    output_dir = PROJECT_DIR / "outputs" / "tuning"
    targets = [output_dir / "phase1_clusterers.csv", output_dir / "phase1_clusterers.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Phase 1 tuning outputs already exist; remove them deliberately before rerunning")
    dataset = {"type": "priberam", "path": str(dataset_path), "language": "spa"}
    documents = clustering.load_documents(dataset)
    print(f"Development tuning documents: {len(documents)}")
    print("Embedding once..."); semantic = clustering.build_embeddings(documents, "distiluse_multilingual")
    print("TF-IDF once..."); tfidf = clustering.build_tfidf(documents)
    print("Pairwise distances once...")
    semantic_distance = clustering.cosine_distance_matrix(semantic)
    tfidf_distance = clustering.cosine_distance_matrix(tfidf)
    temporal_distance = clustering.temporal_distance_matrix(documents, 0.0, 3.0)
    hybrid = np.asarray(.5 * semantic_distance + .3 * tfidf_distance + .2 * temporal_distance, dtype=np.float32)
    np.fill_diagonal(hybrid, 0.0)
    concat = clustering.concat_features(semantic, tfidf, documents, {"semantic": .5, "tfidf": .3, "temporal": .2}, 3.0)
    results: list[dict[str, Any]] = []
    for trial in phase1_trials():
        data = hybrid if trial["representation"] == "hybrid_distance" else concat
        print(f"{trial['trial_id']}: {trial['representation']} + {trial['clusterer']}")
        results.append(evaluate_trial(documents, trial, data))
    write_results(output_dir, results)
    plot_results(output_dir, results)
    for family, rows in rank_by_family(results).items():
        best = rows[0]
        print(f"Best {_family_label(family)}: {best['trial_id']} B-Cubed F1={best['bcubed_f1']:.6f}")
    return results


if __name__ == "__main__":
    run_phase1()
