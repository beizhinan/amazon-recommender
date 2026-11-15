"""邻域协同过滤：ItemCF / UserCF（稀疏相似度 + 收缩 + top-K 截断）。"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .baselines import build_centered, build_residual
from .common import RatingData, log, topk_rows


def _cosine_normalize(sim: sp.csr_matrix, mat: sp.csr_matrix) -> sp.csr_matrix:
    """cosine 归一化：sim_ij / (||i|| * ||j||)，范数按矩阵列计算。"""
    norm = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=0)).ravel()).astype(np.float64)
    inv = np.zeros_like(norm)
    nz = norm > 0
    inv[nz] = 1.0 / norm[nz]
    d = sp.diags(inv)
    return (d @ sim @ d).tocsr()


def _apply_shrinkage(sim: sp.csr_matrix, counts: sp.csr_matrix,
                     shrink: float) -> sp.csr_matrix:
    """相似度收缩：sim * n_ij / (n_ij + shrink)，抑制偶发共现。"""
    c = counts.tocsr().copy()
    c.data = c.data.astype(np.float32)
    c.data = c.data / (c.data + np.float32(shrink))
    return sim.multiply(c).tocsr()


class ItemCF:
    """基于物品的邻域模型。

    相似度：用户在中心化评分上的余弦相似度 + 共现收缩 + 每物品 top-K。
    预测：μ + b_u + b_i + Σ_j sim_ij * residual_uj / Σ_j |sim_ij|
    """

    name = "ItemCF"

    def __init__(self, k: int = 100, shrink: float = 25.0) -> None:
        self.k = k
        self.shrink = shrink
        self.mu = 0.0
        self.b_u: np.ndarray | None = None
        self.b_i: np.ndarray | None = None
        self.sim: sp.csr_matrix | None = None
        self.res: sp.csr_matrix | None = None

    def fit(self, data: RatingData, mu: float, b_u: np.ndarray,
            b_i: np.ndarray) -> "ItemCF":
        self.mu, self.b_u, self.b_i = mu, b_u, b_i
        t0 = time.time()

        log(f"[ItemCF] 构建用户中心化矩阵...")
        centered = build_centered(data, mu, b_u)
        log(f"[ItemCF] 计算物品相似度矩阵...")
        sim = (centered.T @ centered).tocsr()
        log(f"[ItemCF] 相似度矩阵 nnz={sim.nnz:,d} ({time.time() - t0:.1f}s)")
        counts = (data.B.T @ data.B).tocsr()
        sim = _cosine_normalize(sim, centered)
        sim = _apply_shrinkage(sim, counts, self.shrink)
        log(f"[ItemCF] 余弦归一化+收缩完成，截断 top-{self.k}...")
        self.sim = topk_rows(sim, self.k, keep_positive=True)
        self.res = build_residual(data, mu, b_u, b_i)
        log(f"[ItemCF] 训练完成 用时 {time.time() - t0:.1f}s")
        return self

    def predict(self, data: RatingData) -> tuple[np.ndarray, np.ndarray]:
        sim = self.sim
        sim_indptr, sim_indices, sim_data = sim.indptr, sim.indices, sim.data
        res_indptr, res_indices, res_data = (
            self.res.indptr, self.res.indices, self.res.data,
        )
        baseline = self.mu + self.b_u[data.u_te] + self.b_i[data.i_te]
        pred = np.full(len(data.y_te), np.nan, dtype=np.float64)
        fallback = np.zeros(len(data.y_te), dtype=bool)

        groups = data.test_groups()
        total = len(groups)
        step = max(1, total // 10)
        t0 = time.time()
        for gi, (u, pos) in enumerate(groups):
            if gi % step == 0:
                log(f"[ItemCF] 预测 {gi}/{total} 个用户 ({time.time() - t0:.0f}s)")
            items = data.i_te[pos]
            u_start, u_end = res_indptr[u], res_indptr[u + 1]
            u_items = res_indices[u_start:u_end]
            u_res = res_data[u_start:u_end].astype(np.float64)

            num = np.zeros(len(pos), dtype=np.float64)
            den = np.zeros(len(pos), dtype=np.float64)
            for t, item in enumerate(items):
                s, e = sim_indptr[item], sim_indptr[item + 1]
                idxs = sim_indices[s:e]
                w = sim_data[s:e].astype(np.float64)
                if len(idxs) == 0 or len(u_items) == 0:
                    continue
                loc = np.searchsorted(u_items, idxs)
                loc = np.clip(loc, 0, len(u_items) - 1)
                valid = u_items[loc] == idxs
                num[t] = np.dot(w[valid], u_res[loc[valid]])
                den[t] = np.abs(w[valid]).sum()

            values = np.where(den > 0, baseline[pos] + num / np.where(den > 0, den, 1.0),
                              np.nan)
            fallback[pos] = den <= 0
            pred[pos] = values

        log(f"[ItemCF] 预测完成 {total} 个用户 ({time.time() - t0:.0f}s)")
        pred[fallback] = baseline[fallback]
        return pred, fallback


class UserCF:
    """基于用户的邻域模型（余弦相似度，每用户 top-K，预测时聚合邻居残差）。"""

    name = "UserCF"

    def __init__(self, k: int = 50, shrink: float = 25.0) -> None:
        self.k = k
        self.shrink = shrink
        self.mu = 0.0
        self.b_u: np.ndarray | None = None
        self.b_i: np.ndarray | None = None
        self.sim: sp.csr_matrix | None = None
        self.res: sp.csr_matrix | None = None

    def fit(self, data: RatingData, mu: float, b_u: np.ndarray,
            b_i: np.ndarray) -> "UserCF":
        self.mu, self.b_u, self.b_i = mu, b_u, b_i
        t0 = time.time()

        log("[UserCF] 构建用户中心化矩阵...")
        centered = build_centered(data, mu, b_u)
        log("[UserCF] 计算用户相似度矩阵...")
        sim = (centered @ centered.T).tocsr()
        log(f"[UserCF] 相似度矩阵 nnz={sim.nnz:,d} ({time.time() - t0:.1f}s)")
        counts = (data.B @ data.B.T).tocsr()
        sim = _cosine_normalize(sim, centered.T)
        sim = _apply_shrinkage(sim, counts, self.shrink)
        log(f"[UserCF] 余弦归一化+收缩完成，截断 top-{self.k}...")
        self.sim = topk_rows(sim, self.k, keep_positive=True)
        self.res = build_residual(data, mu, b_u, b_i)
        log(f"[UserCF] 训练完成 用时 {time.time() - t0:.1f}s")
        return self

    def predict(self, data: RatingData) -> tuple[np.ndarray, np.ndarray]:
        sim = self.sim
        sim_indptr, sim_indices, sim_data = sim.indptr, sim.indices, sim.data
        res = self.res
        res_indptr, res_indices, res_data = res.indptr, res.indices, res.data
        baseline = self.mu + self.b_u[data.u_te] + self.b_i[data.i_te]
        pred = np.full(len(data.y_te), np.nan, dtype=np.float64)
        fallback = np.zeros(len(data.y_te), dtype=bool)

        groups = data.test_groups()
        total = len(groups)
        step = max(1, total // 10)
        t0 = time.time()
        for gi, (u, pos) in enumerate(groups):
            if gi % step == 0:
                log(f"[UserCF] 预测 {gi}/{total} 个用户 ({time.time() - t0:.0f}s)")
            s, e = sim_indptr[u], sim_indptr[u + 1]
            neigh = sim_indices[s:e]
            weights = sim_data[s:e].astype(np.float64)
            if len(neigh) == 0:
                fallback[pos] = True
                pred[pos] = baseline[pos]
                continue

            # 邻居的残差子矩阵（K x n_items），只保留测试商品对应的列
            sub = res[neigh].tocoo()
            items = data.i_te[pos]
            order = np.argsort(items, kind="mergesort")
            items_sorted = items[order]
            loc = np.searchsorted(items_sorted, sub.col)
            loc = np.clip(loc, 0, len(items_sorted) - 1)
            valid = items_sorted[loc] == sub.col
            if not valid.any():
                fallback[pos] = True
                pred[pos] = baseline[pos]
                continue

            w = weights[sub.row[valid]]
            num = np.bincount(loc[valid], weights=w * sub.data[valid], minlength=len(pos))
            den = np.bincount(loc[valid], weights=np.abs(w), minlength=len(pos))
            has = den > 0
            target = pos[order]
            fallback[target] = ~has
            pred[target] = baseline[target]
            pred[target[has]] += num[has] / den[has]

        log(f"[UserCF] 预测完成 {total} 个用户 ({time.time() - t0:.0f}s)")
        return pred, fallback
