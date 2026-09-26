# fact_extraction/meantime_pipeline/metrics.py
from __future__ import annotations

from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment


def prf(predicted: set, gold: set) -> dict[str, float | int]:
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": len(gold), "predicted": len(predicted)}


def mention_metrics(predictions: list[dict], gold: list[dict], kinds: list[str]) -> dict:
    result = {}
    all_pred, all_gold = set(), set()
    for kind in kinds:
        pred = {(x["document_id"], tuple(x["token_ids"])) for x in predictions if x["kind"] == kind}
        actual = {(x["document_id"], tuple(x["token_ids"])) for x in gold if x["kind"] == kind}
        result[kind] = prf(pred, actual)
        all_pred |= {(kind, *x) for x in pred}
        all_gold |= {(kind, *x) for x in actual}
    result["micro"] = prf(all_pred, all_gold)
    return result


def relation_metrics(predictions: list[dict], gold: list[dict]) -> dict:
    labels = sorted({x["type"] for x in predictions + gold})
    result = {}
    all_pred, all_gold = set(), set()
    for label in labels:
        pred = {tuple(x[k] for k in ("document_id", "type", "source", "target", "role")) for x in predictions if x["type"] == label}
        actual = {tuple(x[k] for k in ("document_id", "type", "source", "target", "role")) for x in gold if x["type"] == label}
        result[label] = prf(pred, actual)
        all_pred |= pred
        all_gold |= actual
    result["micro"] = prf(all_pred, all_gold)
    result["macro_f1"] = sum(result[x]["f1"] for x in labels) / len(labels) if labels else 0.0
    return result


def coreference_metrics(predicted_clusters: list[set[str]], gold_clusters: list[set[str]]) -> dict:
    """MUC, B3, and CEAF_e for a fixed, document-local mention universe."""
    mentions = set().union(*gold_clusters, *predicted_clusters) if gold_clusters or predicted_clusters else set()
    def muc(clusters, other):
        index = {mention: i for i, cluster in enumerate(other) for mention in cluster}
        numerator = denominator = 0
        for cluster in clusters:
            denominator += len(cluster) - 1
            numerator += len(cluster) - len({index.get(mention, ("singleton", mention)) for mention in cluster})
        return numerator / denominator if denominator else 0.0
    muc_p, muc_r = muc(predicted_clusters, gold_clusters), muc(gold_clusters, predicted_clusters)
    p_index = {mention: cluster for cluster in predicted_clusters for mention in cluster}
    g_index = {mention: cluster for cluster in gold_clusters for mention in cluster}
    b_p = sum(len(p_index.get(m, set()) & g_index.get(m, set())) / len(p_index.get(m, {m})) for m in mentions) / len(mentions) if mentions else 0.0
    b_r = sum(len(p_index.get(m, set()) & g_index.get(m, set())) / len(g_index.get(m, {m})) for m in mentions) / len(mentions) if mentions else 0.0
    similarity = np.array([[2 * len(p & g) / (len(p) + len(g)) for g in gold_clusters] for p in predicted_clusters])
    ceaf = similarity[linear_sum_assignment(-similarity)].sum() if similarity.size else 0.0
    ceaf_p = ceaf / len(predicted_clusters) if predicted_clusters else 0.0
    ceaf_r = ceaf / len(gold_clusters) if gold_clusters else 0.0
    def score(p, r): return 2*p*r/(p+r) if p+r else 0.0
    return {"MUC": {"precision": muc_p, "recall": muc_r, "f1": score(muc_p, muc_r)},
            "B3": {"precision": b_p, "recall": b_r, "f1": score(b_p, b_r)},
            "CEAF_e": {"precision": ceaf_p, "recall": ceaf_r, "f1": score(ceaf_p, ceaf_r)},
            "CoNLL_F1": (score(muc_p, muc_r) + score(b_p, b_r) + score(ceaf_p, ceaf_r)) / 3,
            "mentions": len(mentions)}
