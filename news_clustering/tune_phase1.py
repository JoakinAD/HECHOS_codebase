#news_clustering/tune_phase1.py
"""Development-only Phase 1 clusterer parameter tuning.

This runner deliberately creates only compact metrics and plots. It does not
call the full report, NER, graph, or assignment output code.
"""
from __future__ import annotations

import argparse
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
PHASE2_FIELDS = [
    "trial_id", "representation_id", "representation", "clusterer", "encoder", "w_semantic", "w_tfidf",
    "w_temporal", "sigma_days", "distance_threshold", "min_cluster_size", "min_samples", "birch_threshold",
    "branching_factor", "bcubed_precision", "bcubed_recall", "bcubed_f1", "ami", "ari", "gold_clusters",
    "predicted_clusters", "predicted_singletons", "predicted_singleton_percentage", "documents_in_singleton_clusters",
    "documents_in_singleton_clusters_percentage", "raw_noise_points", "raw_noise_percentage", "documents", "runtime_seconds",
]
PHASE2_FIXED = {
    ("hybrid_distance", "hac"): {"distance_threshold": 0.65},
    ("hybrid_distance", "hdbscan"): {"min_cluster_size": 5, "min_samples": 1},
    ("concat_features", "hdbscan"): {"min_cluster_size": 5, "min_samples": 3},
    ("concat_features", "birch"): {"birch_threshold": 0.50, "branching_factor": 50},
}
PHASE1_B3F1 = {
    ("hybrid_distance", "hac"): 0.9016142154793614,
    ("hybrid_distance", "hdbscan"): 0.8892443701559221,
    ("concat_features", "hdbscan"): 0.8912527720289849,
    ("concat_features", "birch"): 0.8244441902300357,
}


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
    configurations = []
    for weights in simplex_weights(0.2):
        sigmas = (None,) if weights["temporal"] == 0 else (1.0, 3.0, 7.0, 14.0)
        for sigma in sigmas:
            configurations.append({"weights": weights, "sigma_days": sigma})
    return configurations


def phase2_representation_id(representation: str, weights: dict[str, float], sigma_days: float | None) -> str:
    def unit(value: float) -> str:
        return f"{int(round(value * 100)):03d}"
    sigma = "NA" if sigma_days is None else f"{int(sigma_days):02d}"
    prefix = "hybrid" if representation == "hybrid_distance" else "concat"
    return f"{prefix}-ws{unit(weights['semantic'])}-wl{unit(weights['tfidf'])}-wt{unit(weights['temporal'])}-s{sigma}"


def phase2_trials() -> list[dict[str, Any]]:
    trials = []
    for representation, clusterers in (("hybrid_distance", ("hac", "hdbscan")), ("concat_features", ("hdbscan", "birch"))):
        for configuration in phase2_representation_grid():
            representation_id = phase2_representation_id(representation, configuration["weights"], configuration["sigma_days"])
            for clusterer in clusterers:
                trials.append({"trial_id": f"{representation_id}-{clusterer}", "representation_id": representation_id,
                    "representation": representation, "clusterer": clusterer, "weights": configuration["weights"],
                    "sigma_days": configuration["sigma_days"], **PHASE2_FIXED[(representation, clusterer)]})
    return trials


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


def phase2_clusterer_config(trial: dict[str, Any]) -> dict[str, Any]:
    return {"name": trial["clusterer"],
        "hac": {"distance_threshold": trial.get("distance_threshold")},
        "hdbscan": {"min_cluster_size": trial.get("min_cluster_size"), "min_samples": trial.get("min_samples")},
        "birch": {"threshold": trial.get("birch_threshold"), "branching_factor": trial.get("branching_factor")}}


def evaluate_phase2_trial(documents: list[clustering.Document], trial: dict[str, Any], data: Any) -> dict[str, Any]:
    started = time.perf_counter()
    raw_labels = clustering.route_clusterer(trial["representation"], phase2_clusterer_config(trial), data)
    labels = clustering.normalize_noise_labels(raw_labels)
    metrics = clustering.evaluate(documents, labels, raw_labels if trial["clusterer"] == "hdbscan" else None)
    if metrics is None:
        raise ValueError("Development tuning requires gold cluster labels")
    result = {field: None for field in PHASE2_FIELDS}
    result.update({"trial_id": trial["trial_id"], "representation_id": trial["representation_id"],
        "representation": trial["representation"], "clusterer": trial["clusterer"], "encoder": "distiluse_multilingual",
        "w_semantic": trial["weights"]["semantic"], "w_tfidf": trial["weights"]["tfidf"],
        "w_temporal": trial["weights"]["temporal"], "sigma_days": trial["sigma_days"],
        "distance_threshold": trial.get("distance_threshold"), "min_cluster_size": trial.get("min_cluster_size"),
        "min_samples": trial.get("min_samples"), "birch_threshold": trial.get("birch_threshold"),
        "branching_factor": trial.get("branching_factor"), "runtime_seconds": time.perf_counter() - started})
    result.update(metrics)
    return result


def rank_phase2(results: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    return {family: sorted((row for row in results if (row["representation"], row["clusterer"]) == family),
        key=lambda row: (-row["bcubed_f1"], -row["ami"], -row["ari"], row["trial_id"])) for family in FAMILY_ORDER}


def best_phase2_trials(rows: list[dict[str, Any]], tolerance: float = 1e-12) -> list[dict[str, Any]]:
    best = rows[0]["bcubed_f1"]
    return [row for row in rows if np.isclose(row["bcubed_f1"], best, rtol=0, atol=tolerance)]


def write_phase2_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    with (output_dir / "phase2_representations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PHASE2_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({field: "N/A" if row[field] is None else row[field] for field in PHASE2_FIELDS})
    (output_dir / "phase2_representations.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def phase2_ablation(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    categories = {
        "semantic only": lambda row: row["w_semantic"] == 1 and row["w_tfidf"] == 0 and row["w_temporal"] == 0,
        "TF-IDF only": lambda row: row["w_semantic"] == 0 and row["w_tfidf"] == 1 and row["w_temporal"] == 0,
        "temporal only": lambda row: row["w_semantic"] == 0 and row["w_tfidf"] == 0 and row["w_temporal"] == 1,
        "semantic + TF-IDF, no time": lambda row: row["w_semantic"] > 0 and row["w_tfidf"] > 0 and row["w_temporal"] == 0,
        "all three components active": lambda row: row["w_semantic"] > 0 and row["w_tfidf"] > 0 and row["w_temporal"] > 0,
    }
    return {name: max((row for row in rows if predicate(row)), key=lambda row: (row["bcubed_f1"], row["ami"], row["ari"], row["trial_id"]), default=None)
        for name, predicate in categories.items()}


def write_phase2_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    ranked = rank_phase2(results)
    lines = ["# Phase 2 Representation Tuning", "", "## Search Description", "",
        "Development set only; DistilUSE fixed; 21 coarse weight triples; `sigma_days` in `{1,3,7,14}` when temporal weight is positive; 66 unique representations per representation family; 264 total algorithm trials. B-Cubed F1 is the primary selection metric.",
        "", "## Best Configuration Per Pipeline", "", "| Pipeline | Weights | Sigma | Fixed clusterer parameters | B-Cubed P/R/F1 | AMI | ARI | Predicted clusters | Raw noise % |", "|---|---|---:|---|---|---:|---:|---:|---:|"]
    for family in FAMILY_ORDER:
        for best in best_phase2_trials(ranked[family]):
            params = PHASE2_FIXED[family]
            lines.append(f"| {_family_label(family)} | ({best['w_semantic']:.1f}, {best['w_tfidf']:.1f}, {best['w_temporal']:.1f}) | {best['sigma_days'] if best['sigma_days'] is not None else 'N/A'} | {params} | {best['bcubed_precision']:.6f} / {best['bcubed_recall']:.6f} / {best['bcubed_f1']:.6f} | {best['ami']:.6f} | {best['ari']:.6f} | {best['predicted_clusters']} | {best['raw_noise_percentage'] if best['raw_noise_percentage'] is not None else 'N/A'} |")
    lines.extend(["", "## Top 10 Configurations Per Pipeline", ""])
    for family in FAMILY_ORDER:
        lines.extend([f"### {_family_label(family)}", "", "| trial_id | weights | sigma | B-Cubed F1 | AMI | ARI |", "|---|---|---:|---:|---:|---:|"])
        for row in ranked[family][:10]:
            lines.append(f"| {row['trial_id']} | ({row['w_semantic']:.1f}, {row['w_tfidf']:.1f}, {row['w_temporal']:.1f}) | {row['sigma_days'] if row['sigma_days'] is not None else 'N/A'} | {row['bcubed_f1']:.6f} | {row['ami']:.6f} | {row['ari']:.6f} |")
        lines.append("")
    lines.extend(["## Component-Ablation Results", "", "Observed development metrics only; these results do not establish causation.", ""])
    for family in FAMILY_ORDER:
        lines.extend([f"### {_family_label(family)}", "", "| Ablation | trial_id | B-Cubed F1 |", "|---|---|---:|"])
        for name, row in phase2_ablation(ranked[family]).items():
            lines.append(f"| {name} | {row['trial_id']} | {row['bcubed_f1']:.6f} |" if row else f"| {name} | N/A | N/A |")
        lines.append("")
    lines.extend(["## Phase-1 Comparison", "", "| Pipeline | Phase-1 B-Cubed F1 | Best Phase-2 B-Cubed F1 | Absolute difference |", "|---|---:|---:|---:|"])
    for family in FAMILY_ORDER:
        best = ranked[family][0]
        lines.append(f"| {_family_label(family)} | {PHASE1_B3F1[family]:.6f} | {best['bcubed_f1']:.6f} | {best['bcubed_f1'] - PHASE1_B3F1[family]:+.6f} |")
    lines.extend(["", "## Methodological Warning", "", "Phase 2 tunes representation hyperparameters on the development labels. The resulting best configuration is a development-selected configuration, not a held-out result. The held-out test set remains untouched.", "", "Best configurations selected separately for different clusterers are optimized-pipeline comparisons, not pure algorithm-only comparisons. Controlled clusterer comparisons remain available by comparing algorithms under the same representation_id."])
    (output_dir / "phase2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_controlled_pairs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    fields = ["representation_id", "representation", "hac_bcubed_f1", "hdbscan_bcubed_f1", "birch_bcubed_f1", "delta_bcubed_f1_hac_minus_hdbscan", "delta_bcubed_f1_hdbscan_minus_birch", "hac_ami", "hdbscan_ami", "birch_ami", "hac_ari", "hdbscan_ari", "birch_ari"]
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results: grouped.setdefault(row["representation_id"], {})[row["clusterer"]] = row
    rows = []
    for representation_id, pair in sorted(grouped.items()):
        representation = next(iter(pair.values()))["representation"]
        row = {field: None for field in fields}; row.update({"representation_id": representation_id, "representation": representation})
        if representation == "hybrid_distance":
            hac, hdbscan = pair["hac"], pair["hdbscan"]
            row.update({"hac_bcubed_f1": hac["bcubed_f1"], "hdbscan_bcubed_f1": hdbscan["bcubed_f1"], "delta_bcubed_f1_hac_minus_hdbscan": hac["bcubed_f1"] - hdbscan["bcubed_f1"], "hac_ami": hac["ami"], "hdbscan_ami": hdbscan["ami"], "hac_ari": hac["ari"], "hdbscan_ari": hdbscan["ari"]})
        else:
            hdbscan, birch = pair["hdbscan"], pair["birch"]
            row.update({"hdbscan_bcubed_f1": hdbscan["bcubed_f1"], "birch_bcubed_f1": birch["bcubed_f1"], "delta_bcubed_f1_hdbscan_minus_birch": hdbscan["bcubed_f1"] - birch["bcubed_f1"], "hdbscan_ami": hdbscan["ami"], "birch_ami": birch["ami"], "hdbscan_ari": hdbscan["ari"], "birch_ari": birch["ari"]})
        rows.append(row)
    with (output_dir / "phase2_controlled_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({field: "N/A" if row[field] is None else row[field] for field in fields})


def plot_phase2(output_dir: Path, results: list[dict[str, Any]]) -> None:
    ranked = rank_phase2(results); best = [ranked[family][0] for family in FAMILY_ORDER]; labels = ["Hybrid + HAC", "Hybrid + HDBSCAN", "Concat + HDBSCAN", "Concat + BIRCH"]
    positions = np.arange(4); width = .24
    figure, axis = plt.subplots(figsize=(11, 6))
    for offset, metric, label in ((-width, "bcubed_f1", "B-Cubed F1"), (0, "ami", "AMI"), (width, "ari", "ARI")):
        axis.bar(positions + offset, [row[metric] for row in best], width, label=label)
    axis.set(xticks=positions, xticklabels=labels, ylim=(0, 1), ylabel="Score", title="Phase 2 development-set best configurations"); axis.tick_params(axis="x", rotation=15); axis.legend(); figure.tight_layout(); figure.savefig(output_dir / "phase2_best_metrics.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 7)); colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for family, color in zip(FAMILY_ORDER, colors):
        rows = ranked[family]; axis.scatter([row["bcubed_recall"] for row in rows], [row["bcubed_precision"] for row in rows], color=color, alpha=.55, label=_family_label(family))
        for selected in best_phase2_trials(rows):
            axis.scatter(selected["bcubed_recall"], selected["bcubed_precision"], color=color, edgecolor="black", s=110, zorder=3); axis.annotate(selected["trial_id"], (selected["bcubed_recall"], selected["bcubed_precision"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set(xlabel="B-Cubed Recall", ylabel="B-Cubed Precision", xlim=(0, 1), ylim=(0, 1), title="Phase 2 development-set precision-recall trade-off"); axis.legend(fontsize=8); figure.tight_layout(); figure.savefig(output_dir / "phase2_precision_recall.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 6))
    for family, color in zip(FAMILY_ORDER, colors):
        rows = [row for row in ranked[family] if row["w_temporal"] > 0]
        sigma_best = [max((row for row in rows if row["sigma_days"] == sigma), key=lambda row: row["bcubed_f1"]) for sigma in (1.0, 3.0, 7.0, 14.0)]
        axis.plot([row["sigma_days"] for row in sigma_best], [row["bcubed_f1"] for row in sigma_best], marker="o", color=color, label=_family_label(family))
    axis.set(xlabel="sigma_days", ylabel="Best B-Cubed F1 with temporal weight > 0", ylim=(0, 1), title="Phase 2 sigma sensitivity"); axis.legend(fontsize=8); figure.tight_layout(); figure.savefig(output_dir / "phase2_sigma_sensitivity.png", dpi=160); plt.close(figure)
    categories = list(phase2_ablation(ranked[FAMILY_ORDER[0]])); positions = np.arange(len(categories)); figure, axis = plt.subplots(figsize=(13, 6)); width = .19
    for index, family in enumerate(FAMILY_ORDER):
        ablation = phase2_ablation(ranked[family]); axis.bar(positions + (index - 1.5) * width, [ablation[name]["bcubed_f1"] for name in categories], width, label=_family_label(family))
    axis.set(xticks=positions, xticklabels=categories, ylim=(0, 1), ylabel="Best B-Cubed F1", title="Phase 2 component-ablation comparison"); axis.tick_params(axis="x", rotation=16); axis.legend(fontsize=8); figure.tight_layout(); figure.savefig(output_dir / "phase2_ablation_b3f1.png", dpi=160); plt.close(figure)


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


def run_phase2() -> list[dict[str, Any]]:
    dataset_path = PROJECT_DIR / "dataset" / "dataset.dev.json"
    validate_tuning_dataset(dataset_path)
    configurations = phase2_representation_grid()
    trials = phase2_trials()
    if len(configurations) != 66 or len(trials) != 264:
        raise RuntimeError("Unexpected Phase 2 representation or trial count")
    if any(len([trial for trial in trials if trial["representation_id"] == representation_id]) != 2 for representation_id in {trial["representation_id"] for trial in trials}):
        raise RuntimeError("Phase 2 controlled representation pairing is incomplete")
    output_dir = PROJECT_DIR / "outputs" / "tuning"
    targets = [output_dir / "phase2_representations.csv", output_dir / "phase2_representations.json", output_dir / "phase2_summary.md"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Phase 2 tuning outputs already exist; remove them deliberately before rerunning")
    dataset = {"type": "priberam", "path": str(dataset_path), "language": "spa"}
    documents = clustering.load_documents(dataset)
    gold_clusters = len({document.gold_cluster for document in documents})
    print("Phase: 2\nDataset: development only\nDocuments: {}\nGold clusters: {}\nEncoder: distiluse_multilingual\nWeight triples: 21\nSigma values: [1.0, 3.0, 7.0, 14.0]\nUnique representations per family: 66\nExpected total trials: 264\n\nFixed clusterers:\nHybrid HAC: threshold=0.65\nHybrid HDBSCAN: min_cluster_size=5, min_samples=1\nConcat HDBSCAN: min_cluster_size=5, min_samples=3\nConcat BIRCH: threshold=0.50, branching_factor=50, n_clusters=None".format(len(documents), gold_clusters))
    if len(documents) != 4527 or gold_clusters != 416:
        raise RuntimeError("Unexpected development dataset size or gold-cluster count")
    print("Embedding once..."); semantic = clustering.build_embeddings(documents, "distiluse_multilingual")
    print("TF-IDF once..."); tfidf = clustering.build_tfidf(documents)
    print("Semantic and TF-IDF distances once..."); semantic_distance = clustering.cosine_distance_matrix(semantic); tfidf_distance = clustering.cosine_distance_matrix(tfidf)
    temporal_distances: dict[float, np.ndarray] = {}
    results: list[dict[str, Any]] = []
    for representation in ("hybrid_distance", "concat_features"):
        print(f"Building and evaluating {representation} representations...")
        for configuration in configurations:
            weights, sigma = configuration["weights"], configuration["sigma_days"]
            representation_id = phase2_representation_id(representation, weights, sigma)
            if representation == "hybrid_distance":
                data = weights["semantic"] * semantic_distance + weights["tfidf"] * tfidf_distance
                if weights["temporal"] > 0:
                    if sigma not in temporal_distances:
                        temporal_distances[sigma] = clustering.temporal_distance_matrix(documents, 0.0, sigma)
                    data = data + weights["temporal"] * temporal_distances[sigma]
                data = np.asarray(data, dtype=np.float32); np.fill_diagonal(data, 0.0)
                clusterers = ("hac", "hdbscan")
            else:
                data = clustering.concat_features(semantic, tfidf, documents, weights, sigma if sigma is not None else 1.0)
                clusterers = ("hdbscan", "birch")
            for clusterer in clusterers:
                trial = next(item for item in trials if item["representation_id"] == representation_id and item["clusterer"] == clusterer)
                print(trial["trial_id"])
                results.append(evaluate_phase2_trial(documents, trial, data))
    if len(results) != 264:
        raise RuntimeError(f"Expected 264 Phase 2 trials, got {len(results)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_phase2_results(output_dir, results); write_controlled_pairs(output_dir, results); write_phase2_summary(output_dir, results); plot_phase2(output_dir, results)
    for family, rows in rank_phase2(results).items():
        print(f"Best {_family_label(family)}: {rows[0]['trial_id']} B-Cubed F1={rows[0]['bcubed_f1']:.6f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    run_phase1() if args.phase == 1 else run_phase2()
