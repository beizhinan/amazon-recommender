"""Top-N 排序评估指标：Precision@K、Recall@K、F1@K、NDCG@K、HitRate@K、覆盖率。"""
from __future__ import annotations

import numpy as np


def precision_at_k(rec: np.ndarray, gt: set[int], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    return float(sum(1 for i in rec[:k] if i in gt) / k)


def recall_at_k(rec: np.ndarray, gt: set[int], k: int = 10) -> float:
    if not gt:
        return 0.0
    return float(sum(1 for i in rec[:k] if i in gt) / len(gt))


def f1_at_k(rec: np.ndarray, gt: set[int], k: int = 10) -> float:
    p = precision_at_k(rec, gt, k)
    r = recall_at_k(rec, gt, k)
    return float(2 * p * r / (p + r)) if p + r > 0 else 0.0


def hit_rate_at_k(rec: np.ndarray, gt: set[int], k: int = 10) -> float:
    return float(any(i in gt for i in rec[:k]))


def ndcg_at_k(rec: np.ndarray, gt: set[int], k: int = 10) -> float:
    """二值相关性下的 NDCG@K（IDCG 按 min(|gt|, k) 个理想命中计算）。"""
    if not gt:
        return 0.0
    hits = np.fromiter((1.0 if i in gt else 0.0 for i in rec[:k]),
                       dtype=np.float64, count=min(k, len(rec)))
    if hits.sum() == 0:
        return 0.0
    discount = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    dcg = float((hits * discount).sum())
    ideal_n = min(len(gt), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_n + 2))).sum())
    return dcg / idcg


def evaluate_ranking(recs: list[np.ndarray], gts: list[set[int]], k: int = 10,
                     n_items: int | None = None) -> dict:
    """对每个用户推荐列表求指标后取宏平均。"""
    n = len(recs)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0,
                "hit_rate": 0.0, "coverage": 0.0, "users": 0}

    p = np.array([precision_at_k(r, g, k) for r, g in zip(recs, gts)])
    rc = np.array([recall_at_k(r, g, k) for r, g in zip(recs, gts)])
    f = np.array([f1_at_k(r, g, k) for r, g in zip(recs, gts)])
    nd = np.array([ndcg_at_k(r, g, k) for r, g in zip(recs, gts)])
    hr = np.array([hit_rate_at_k(r, g, k) for r, g in zip(recs, gts)])

    coverage = 0.0
    if n_items:
        uniq = len({int(i) for r in recs for i in r})
        coverage = uniq / n_items

    return {
        "precision": round(float(p.mean()), 6),
        "recall": round(float(rc.mean()), 6),
        "f1": round(float(f.mean()), 6),
        "ndcg": round(float(nd.mean()), 6),
        "hit_rate": round(float(hr.mean()), 6),
        "coverage": round(coverage, 4),
        "users": n,
    }
