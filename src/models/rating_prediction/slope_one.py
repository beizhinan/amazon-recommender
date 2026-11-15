"""Slope One（加权版本，top-K 物品对）。

dev_ij = mean_{u∈S_ij}(res_ui - res_uj)，res 为去偏置后的残差；
pred_ui = μ + b_u + b_i + Σ_{j∈R_u} (dev_ij + res_uj) * c_ij / Σ_{j∈R_u} c_ij
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .baselines import build_residual
from .common import RatingData, log, topk_rows


class SlopeOne:
    name = "SlopeOne"

    def __init__(self, k: int = 100, min_count: int = 1) -> None:
        self.k = k
        self.min_count = min_count
        self.dev: sp.csr_matrix | None = None
        self.counts: sp.csr_matrix | None = None
        self.mu = 0.0
        self.b_u: np.ndarray | None = None
        self.b_i: np.ndarray | None = None
        self.res: sp.csr_matrix | None = None

    def fit(self, data: RatingData, mu: float, b_u: np.ndarray,
            b_i: np.ndarray) -> "SlopeOne":
        self.mu, self.b_u, self.b_i = mu, b_u, b_i
        t0 = time.time()
        log("[SlopeOne] 构建残差矩阵与共现统计...")
        res = build_residual(data, mu, b_u, b_i)
        self.res = res

        B = data.B.tocsr()
        C = (B.T @ B).tocsr()                       # 共现次数
        D = (res.T @ B - B.T @ res).tocsr()         # 残差差之和
        log(f"[SlopeOne] 共现矩阵 nnz={C.nnz:,d} ({time.time() - t0:.1f}s)")

        # 按共现次数选 top-K 物品对（比按 dev 更稳健）
        log(f"[SlopeOne] 截断 top-{self.k} 物品对并计算偏差...")
        C_top = topk_rows(C, self.k, keep_positive=True)
        self.counts = C_top

        # 对齐 D 与 C_top 的行结构，计算 dev = D / C
        indptr_c, indices_c, data_c = C_top.indptr, C_top.indices, C_top.data
        indptr_d, indices_d, data_d = D.indptr, D.indices, D.data
        rows, cols, vals = [], [], []
        for i in range(C_top.shape[0]):
            c_idx = indices_c[indptr_c[i]:indptr_c[i + 1]]
            c_val = data_c[indptr_c[i]:indptr_c[i + 1]]
            if len(c_idx) == 0:
                continue
            d_idx = indices_d[indptr_d[i]:indptr_d[i + 1]]
            d_val = data_d[indptr_d[i]:indptr_d[i + 1]]
            pos = np.searchsorted(d_idx, c_idx)
            pos = np.clip(pos, 0, len(d_idx) - 1)
            hit = d_idx[pos] == c_idx if len(d_idx) else np.zeros(len(c_idx), bool)
            mean = np.zeros(len(c_idx), dtype=np.float32)
            mean[hit] = d_val[pos[hit]] / c_val[hit]
            rows.append(np.full(len(c_idx), i, dtype=np.int32))
            cols.append(c_idx.astype(np.int32))
            vals.append(mean)
        self.dev = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=C_top.shape,
        )
        self.dev.sort_indices()
        log(f"[SlopeOne] 训练完成 用时 {time.time() - t0:.1f}s")
        return self

    def predict(self, data: RatingData) -> tuple[np.ndarray, np.ndarray]:
        cnt = self.counts
        dev = self.dev
        c_indptr, c_indices, c_data = cnt.indptr, cnt.indices, cnt.data
        d_indptr, d_data = dev.indptr, dev.data
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
                log(f"[SlopeOne] 预测 {gi}/{total} 个用户 ({time.time() - t0:.0f}s)")
            items = data.i_te[pos]
            s_u, e_u = res_indptr[u], res_indptr[u + 1]
            u_items = res_indices[s_u:e_u]
            u_res = res_data[s_u:e_u].astype(np.float64)

            num = np.zeros(len(pos), dtype=np.float64)
            den = np.zeros(len(pos), dtype=np.float64)
            for t, item in enumerate(items):
                s, e = c_indptr[item], c_indptr[item + 1]
                cand = c_indices[s:e]
                c_val = c_data[s:e].astype(np.float64)
                if len(cand) == 0 or len(u_items) == 0:
                    continue
                loc = np.searchsorted(u_items, cand)
                loc = np.clip(loc, 0, len(u_items) - 1)
                valid = u_items[loc] == cand
                if self.min_count > 1:
                    valid &= c_val >= self.min_count
                if not valid.any():
                    continue
                # dev 与 counts 的稀疏结构逐行一致（构造时对齐），直接取对应行
                d_val = d_data[d_indptr[item]:d_indptr[item + 1]].astype(np.float64)
                w = c_val[valid]
                num[t] = np.dot(d_val[valid] + u_res[loc[valid]], w)
                den[t] = w.sum()

            has = den > 0
            fallback[pos] = ~has
            pred[pos] = baseline[pos]
            pred[pos[has]] += num[has] / den[has]

        log(f"[SlopeOne] 预测完成 {total} 个用户 ({time.time() - t0:.0f}s)")
        return pred, fallback
