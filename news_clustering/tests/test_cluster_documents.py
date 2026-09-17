from datetime import datetime, timedelta
import sys
import types

import numpy as np
import pytest
from scipy import sparse

import cluster_documents as app
import tune_phase1 as tuning


def config(**changes):
    value = {"representation": "hybrid_distance", "encoder": "distiluse_multilingual",
        "weights": {"semantic": .5, "tfidf": .3, "temporal": .2},
        "temporal": {"mu_days": 0., "sigma_days": 3.},
        "clusterer": {"name": "hac"}}
    value.update(changes)
    return value


def docs(days=(0, 1)):
    origin = datetime(2026, 1, 1)
    return [app.Document(str(i), "title", "body", origin + timedelta(days=day), None, "g") for i, day in enumerate(days)]


@pytest.mark.parametrize("change", [
    {"weights": {"semantic": -.1, "tfidf": .9, "temporal": .2}},
    {"weights": {"semantic": .5, "tfidf": .3, "temporal": .1}},
    {"temporal": {"mu_days": 0., "sigma_days": 0.}},
    {"encoder": "unknown"}, {"representation": "unknown"},
    {"clusterer": {"name": "unknown"}},
    {"clusterer": {"name": "birch"}},
])
def test_invalid_config(change):
    with pytest.raises(ValueError): app.validate_config(config(**change))


def test_concat_validation_requires_zero_mu_and_rejects_hac():
    with pytest.raises(ValueError): app.validate_config(config(representation="concat_features", clusterer={"name": "hac"}))
    with pytest.raises(ValueError): app.validate_config(config(representation="concat_features", clusterer={"name": "birch"}, temporal={"mu_days": 1., "sigma_days": 3.}))
    app.validate_config(config(representation="concat_features", clusterer={"name": "birch"}))


def test_temporal_distance_properties():
    distance = app.temporal_distance_matrix(docs((0, 0, 1, 5)), 0., 3.)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0)
    assert distance[0, 1] == 0
    assert distance[0, 2] < distance[0, 3]


def test_hybrid_distance():
    semantic = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    tfidf = sparse.csr_matrix(semantic)
    result = app.hybrid_distance(semantic, tfidf, docs((0, 0)), {"semantic": .5, "tfidf": .3, "temporal": .2}, {"mu_days": 0., "sigma_days": 3.})
    assert result[0, 1] == pytest.approx(.8)
    assert np.allclose(result, result.T) and np.allclose(np.diag(result), 0)


def test_concat_distance_identity():
    semantic = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    tfidf = sparse.csr_matrix(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    weights = {"semantic": .5, "tfidf": .3, "temporal": .2}
    matrix = app.concat_features(semantic, tfidf, docs((0, 3)), weights, 3.)
    assert sparse.isspmatrix_csr(matrix) and matrix.shape[0] == 2 and np.isfinite(matrix.data).all()
    expected = .5 * 1 + .3 * 1 + .2 * (3**2 / (2 * 3**2))
    assert float((matrix[0] - matrix[1]).multiply(matrix[0] - matrix[1]).sum()) == pytest.approx(expected)


def test_bcubed_known_partitions():
    assert app.bcubed(["a", "a", "b"], ["x", "x", "y"]) == pytest.approx((1., 1., 1.))
    p, r, f = app.bcubed(["a", "a"], ["x", "y"])
    assert (p, r, f) == pytest.approx((1., .5, 2 / 3))
    p, r, f = app.bcubed(["a", "b"], ["x", "x"])
    assert (p, r, f) == pytest.approx((.5, 1., 2 / 3))


def test_noise_labels_are_unique_singletons():
    assert app.normalize_noise_labels([0, -1, -1, 1]) == ["0", "noise-1", "noise-2", "1"]


def test_evaluation_reports_raw_noise_and_singleton_documents():
    documents = [app.Document(str(i), "", "", datetime(2026, 1, 1), None, "gold") for i in range(4)]
    metrics = app.evaluate(documents, ["0", "noise-1", "noise-2", "1"], np.array([0, -1, -1, 1]))
    assert metrics["predicted_singletons"] == metrics["documents_in_singleton_clusters"] == 4
    assert metrics["documents_in_singleton_clusters_percentage"] == pytest.approx(100.0)
    assert metrics["raw_noise_points"] == 2
    assert metrics["raw_noise_percentage"] == pytest.approx(50.0)


def test_clusterer_routing(monkeypatch):
    received = []
    class FakeModel:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def fit_predict(self, data): received.append((self.kwargs, data.shape)); return np.zeros(data.shape[0], dtype=int)
    monkeypatch.setattr(app, "AgglomerativeClustering", FakeModel)
    monkeypatch.setattr(app, "Birch", FakeModel)
    monkeypatch.setitem(sys.modules, "hdbscan", types.SimpleNamespace(HDBSCAN=FakeModel))
    distance = np.eye(2, dtype=np.float32)
    features = sparse.csr_matrix(np.eye(2, dtype=np.float32))
    app.route_clusterer("hybrid_distance", {"name": "hac", "hac": {"distance_threshold": .5}}, distance)
    app.route_clusterer("hybrid_distance", {"name": "hdbscan", "hdbscan": {"min_cluster_size": 2, "min_samples": 1}}, distance)
    app.route_clusterer("concat_features", {"name": "hdbscan", "hdbscan": {"min_cluster_size": 2, "min_samples": 1}}, features)
    app.route_clusterer("concat_features", {"name": "birch", "birch": {"threshold": .5, "branching_factor": 50}}, features)
    assert received[0][0]["metric"] == "precomputed" and received[0][1] == (2, 2)
    assert received[1][0]["metric"] == "precomputed" and received[1][1] == (2, 2)
    assert received[2][0]["metric"] == "euclidean" and received[2][1] == received[3][1] == (2, 2)


def test_report_formatting(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "load_spanish_ner", lambda: None)
    monkeypatch.setattr(app, "cluster_entities", lambda *_: ["ana"])
    output = tmp_path / "output"; output.mkdir()
    monkeypatch.setattr(app, "write_graph", lambda *_: None)
    documents = [app.Document("b", "Later", "", datetime(2026, 1, 2), "Paper", None), app.Document("a", "Earlier", "", datetime(2026, 1, 1), "Paper", None)]
    app.write_outputs(output, config(), documents, ["1", "1"], None)
    report = (output / "clusters.txt").read_text(encoding="utf-8")
    assert "np.int64" not in report and "(2 noticias)" in report
    assert report.index("[a]") < report.index("[b]")


def test_phase1_grid_is_deterministic_and_complete():
    trials = tuning.phase1_trials()
    assert len(trials) == 65
    assert [trial["trial_id"] for trial in trials] == [f"phase1-{number:03d}" for number in range(1, 66)]
    assert sum(trial["representation"] == "hybrid_distance" and trial["clusterer"] == "hac" for trial in trials) == 7
    assert sum(trial["representation"] == "hybrid_distance" and trial["clusterer"] == "hdbscan" for trial in trials) == 20
    assert sum(trial["representation"] == "concat_features" and trial["clusterer"] == "hdbscan" for trial in trials) == 20
    assert sum(trial["representation"] == "concat_features" and trial["clusterer"] == "birch" for trial in trials) == 18


def test_tuning_rejects_test_dataset():
    with pytest.raises(ValueError): tuning.validate_tuning_dataset(__import__("pathlib").Path("dataset/dataset.test.json"))
    tuning.validate_tuning_dataset(__import__("pathlib").Path("dataset/dataset.dev.json"))


def test_simplex_weight_generation():
    weights = tuning.simplex_weights(.2)
    assert len(weights) == 21
    assert {"semantic": 1.0, "tfidf": 0.0, "temporal": 0.0} in weights
    assert {"semantic": 0.0, "tfidf": 0.0, "temporal": 1.0} in weights
    assert all(sum(row.values()) == pytest.approx(1.0) for row in weights)
    assert sum(row["temporal"] == 0 for row in weights) == 6
    assert sum(row["temporal"] > 0 for row in weights) == 15
    assert len(tuning.phase2_representation_grid()) == 66
    assert all(row["sigma_days"] is None for row in tuning.phase2_representation_grid() if row["weights"]["temporal"] == 0)


def test_tuning_trial_skips_full_artifacts(monkeypatch):
    documents = [app.Document("a", "", "", datetime(2026, 1, 1), None, "g")]
    trial = {"trial_id": "phase1-001", "representation": "hybrid_distance", "clusterer": "hac", "distance_threshold": .5}
    monkeypatch.setattr(app, "route_clusterer", lambda *_: np.array([0]))
    monkeypatch.setattr(app, "write_outputs", lambda *_: (_ for _ in ()).throw(AssertionError("full outputs must not run")))
    result = tuning.evaluate_trial(documents, trial, np.array([[0.]]))
    assert result["trial_id"] == "phase1-001" and result["raw_noise_points"] is None


def test_tuning_ranking_and_csv_fields(tmp_path):
    results = []
    for number, family in enumerate(tuning.FAMILY_ORDER, start=1):
        results.append({field: None for field in tuning.RESULT_FIELDS} | {"trial_id": f"phase1-{number:03d}", "representation": family[0], "clusterer": family[1], "bcubed_f1": .5, "bcubed_precision": .5, "bcubed_recall": .5})
        results.append({field: None for field in tuning.RESULT_FIELDS} | {"trial_id": f"phase1-{number + 10:03d}", "representation": family[0], "clusterer": family[1], "bcubed_f1": .7, "bcubed_precision": .7, "bcubed_recall": .7})
    assert all(rows[0]["bcubed_f1"] == .7 for rows in tuning.rank_by_family(results).values())
    tuning.write_results(tmp_path, results)
    header = (tmp_path / "phase1_clusterers.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == tuning.RESULT_FIELDS


def test_phase2_trial_grid_pairing_and_fixed_clusterers():
    trials = tuning.phase2_trials()
    assert len(trials) == 264
    hybrid = [trial for trial in trials if trial["representation"] == "hybrid_distance"]
    concat = [trial for trial in trials if trial["representation"] == "concat_features"]
    assert len(hybrid) == len(concat) == 132
    for family_trials, clusterers in ((hybrid, {"hac", "hdbscan"}), (concat, {"hdbscan", "birch"})):
        ids = {trial["representation_id"] for trial in family_trials}
        assert len(ids) == 66
        assert all({trial["clusterer"] for trial in family_trials if trial["representation_id"] == identifier} == clusterers for identifier in ids)
    assert all(trial["distance_threshold"] == .65 for trial in hybrid if trial["clusterer"] == "hac")
    assert all((trial["min_cluster_size"], trial["min_samples"]) == (5, 1) for trial in hybrid if trial["clusterer"] == "hdbscan")
    assert all((trial["min_cluster_size"], trial["min_samples"]) == (5, 3) for trial in concat if trial["clusterer"] == "hdbscan")
    assert all((trial["birch_threshold"], trial["branching_factor"]) == (.5, 50) for trial in concat if trial["clusterer"] == "birch")


def test_phase2_ids_and_primary_selection_are_deterministic():
    weights = {"semantic": .6, "tfidf": .2, "temporal": .2}
    assert tuning.phase2_representation_id("hybrid_distance", weights, 3.) == "hybrid-ws060-wl020-wt020-s03"
    assert tuning.phase2_representation_id("concat_features", {"semantic": .6, "tfidf": .4, "temporal": 0.}, None) == "concat-ws060-wl040-wt000-sNA"
    rows = [{"bcubed_f1": .8, "ami": .99, "ari": .99, "trial_id": "later"}, {"bcubed_f1": .9, "ami": .1, "ari": .1, "trial_id": "best"}]
    assert tuning.best_phase2_trials(sorted(rows, key=lambda row: -row["bcubed_f1"]))[0]["trial_id"] == "best"


def test_phase2_trial_skips_human_outputs(monkeypatch):
    documents = [app.Document("a", "", "", datetime(2026, 1, 1), None, "g")]
    trial = next(trial for trial in tuning.phase2_trials() if trial["clusterer"] == "hac")
    monkeypatch.setattr(app, "route_clusterer", lambda *_: np.array([0]))
    monkeypatch.setattr(app, "write_outputs", lambda *_: (_ for _ in ()).throw(AssertionError("full outputs must not run")))
    monkeypatch.setattr(app, "load_spanish_ner", lambda: (_ for _ in ()).throw(AssertionError("NER must not run")))
    result = tuning.evaluate_phase2_trial(documents, trial, np.array([[0.]]))
    assert result["representation_id"] == trial["representation_id"] and result["raw_noise_points"] is None


def test_phase2b_local_grid_counts_and_boundaries():
    grids = tuning.phase2b_configurations()
    assert len(grids["h1_weights"]) == 19 and len(grids["H1"]) == 83
    assert len(grids["h2_weights"]) == 12 and len(grids["H2"]) == 60
    assert {row["weights"]["tfidf"] for row in grids["h2_weights"]} == {0.0, 0.05, 0.1}
    assert len(grids["c1_units"]) == len(grids["c2_units"]) == 19
    assert len(set(grids["c1_units"]) | set(grids["c2_units"])) == 35
    assert sum(row["weights"]["temporal"] == 0 for row in grids["concat"]) == 3
    assert len(grids["concat"]) == 163
    assert all(row["sigma_days"] is None for row in grids["H1"] + grids["concat"] if row["weights"]["temporal"] == 0)


def test_phase2b_trials_are_paired_and_fixed():
    trials = tuning.phase2b_trials()
    hybrid = [trial for trial in trials if trial["representation"] == "hybrid_distance"]
    concat = [trial for trial in trials if trial["representation"] == "concat_features"]
    assert len(trials) == 612 and len(hybrid) == 286 and len(concat) == 326
    assert len({trial["representation_id"] for trial in trials}) == 306
    for rows, clusterers in ((hybrid, {"hac", "hdbscan"}), (concat, {"hdbscan", "birch"})):
        assert all({trial["clusterer"] for trial in rows if trial["representation_id"] == identifier} == clusterers for identifier in {trial["representation_id"] for trial in rows})
    assert all(trial["distance_threshold"] == .65 for trial in hybrid if trial["clusterer"] == "hac")
    assert all((trial["min_cluster_size"], trial["min_samples"]) == (5, 1) for trial in hybrid if trial["clusterer"] == "hdbscan")
    assert all((trial["min_cluster_size"], trial["min_samples"]) == (5, 3) for trial in concat if trial["clusterer"] == "hdbscan")
    assert all((trial["birch_threshold"], trial["branching_factor"]) == (.5, 50) for trial in concat if trial["clusterer"] == "birch")


def test_phase2b_ids_and_human_output_guard(monkeypatch):
    weights = {"semantic": .8, "tfidf": .05, "temporal": .15}
    assert tuning.phase2b_representation_id("hybrid_distance", "H2", weights, .5) == "p2b-hybrid-h2-ws080-wl005-wt015-s050"
    documents = [app.Document("a", "", "", datetime(2026, 1, 1), None, "g")]
    trial = next(trial for trial in tuning.phase2b_trials() if trial["clusterer"] == "hac")
    monkeypatch.setattr(app, "route_clusterer", lambda *_: np.array([0]))
    monkeypatch.setattr(app, "write_outputs", lambda *_: (_ for _ in ()).throw(AssertionError("full outputs must not run")))
    monkeypatch.setattr(app, "load_spanish_ner", lambda: (_ for _ in ()).throw(AssertionError("NER must not run")))
    result = tuning.evaluate_phase2b_trial(documents, trial, np.array([[0.]]))
    assert result["search_region"] == "H1" and result["raw_noise_points"] is None


def test_phase2b_boundary_statuses():
    assert "upper" in tuning.phase2b_boundary_status({"search_region": "H1", "sigma_days": 28, "representation": "hybrid_distance", "clusterer": "hac", "w_tfidf": .6})
    assert "lower high-sigma" in tuning.phase2b_boundary_status({"search_region": "C2", "sigma_days": 7, "representation": "concat_features", "clusterer": "birch", "w_tfidf": .6})
    assert "lexical-weight" in tuning.phase2b_boundary_status({"search_region": "H2", "sigma_days": 2, "representation": "hybrid_distance", "clusterer": "hdbscan", "w_tfidf": 0.})
