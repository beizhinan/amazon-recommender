"""Top-N 推荐模型：流行度、评分排序、ItemKNN、BPR-MF 与混合排序。

所有模型统一接口：
    fit(...)                       # 训练
    score_chunk(users, B_hist)     # 返回 (len(users), n_items) 的稠密得分
由 run_topn.py 负责排除已交互商品并取 top-K。
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.models.rating_prediction.common import log, topk_rows


class Popularity:
    """按训练集交互次数排序（MostPop）。"""

    name = "MostPop"

    def fit(self, B: sp.csr_matrix) -> "Popularity":
        self.counts = np.asarray(B.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score_chunk(self, users: np.ndarray, B_hist: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.counts, (len(users), 1))


class RatingRanker:
    """按评分预测（μ + b_u + b_i）排序，衡量「只做评分预测」的排序能力。"""

    name = "RatingPred"

    def __init__(self, mu: float, b_u: np.ndarray, b_i: np.ndarray) -> None:
        self.mu = float(mu)
        self.b_u = b_u.astype(np.float32)
        self.b_i = b_i.astype(np.float32)

    def score_chunk(self, users: np.ndarray, B_hist: sp.csr_matrix) -> np.ndarray:
        return (self.mu + self.b_u[users][:, None] + self.b_i[None, :]).astype(np.float32)


class ItemKNN:
    """基于物品的协同过滤（非对称余弦 + 共现收缩 + top-K 截断）。

    非对称余弦（Steck, 2019）：
        s_ij = C_ij / (||w_i||^α · ||w_j||^(1-α))
    α > 0.5 时对热门候选商品施加更强惩罚，缓解流行度偏置。
    输入 W 为带时间衰减的交互权重矩阵，得分 = 用户历史与候选的相似度之和。
    """

    name = "ItemKNN"

    def __init__(self, k: int = 200, shrink: float = 10.0, alpha: float = 0.5) -> None:
        self.k = int(k)
        self.shrink = float(shrink)
        self.alpha = float(alpha)
        self.S: sp.csr_matrix | None = None
        self._C: sp.csr_matrix | None = None
        self._norm: np.ndarray | None = None

    def fit(self, W: sp.csr_matrix) -> "ItemKNN":
        """只计算共现矩阵与物品范数，具体 alpha / k 由 build() 生成。"""
        t0 = time.time()
        log(f"[ItemKNN] 计算共现矩阵 shrink={self.shrink}...")
        C = (W.T @ W).tocsr()
        C.data = C.data.astype(np.float32)
        C.data = C.data / (C.data + np.float32(self.shrink))
        self._C = C
        norm = np.sqrt(np.asarray(W.multiply(W).sum(axis=0)).ravel()).astype(np.float64)
        self._norm = np.maximum(norm, 1e-12)
        log(f"[ItemKNN] 共现 nnz={C.nnz:,d} ({time.time() - t0:.1f}s)")
        return self

    def build(self, alpha: float, k: int) -> "ItemKNN":
        """按给定 alpha 做非对称归一化，并截断到 top-k。"""
        t0 = time.time()
        cnt = np.maximum(self._norm ** 2, 1e-12)   # ||b_i||² = 交互次数
        d_l = sp.diags(np.power(cnt, -alpha))
        d_r = sp.diags(np.power(cnt, -(1.0 - alpha)))
        S = (d_l @ self._C @ d_r).tocsr()
        S.sum_duplicates()
        S.setdiag(0.0)
        S.eliminate_zeros()
        obj = ItemKNN(k=k, shrink=self.shrink, alpha=alpha)
        obj._C, obj._norm = self._C, self._norm
        obj.S = topk_rows(S, k, keep_positive=True, drop_self=True)
        log(f"[ItemKNN] alpha={alpha} k={k} nnz={obj.S.nnz:,d} "
            f"({time.time() - t0:.1f}s)")
        return obj

    def truncate(self, k: int) -> "ItemKNN":
        """从已截断的相似度矩阵再截断到更小的 k（无需重算归一化）。"""
        obj = ItemKNN(k=k, shrink=self.shrink, alpha=self.alpha)
        obj._C, obj._norm, obj.S = self._C, self._norm, topk_rows(
            self.S, k, keep_positive=True, drop_self=True)
        return obj

    def score_sparse(self, users: np.ndarray, B_hist: sp.csr_matrix) -> sp.csr_matrix:
        return (B_hist[users] @ self.S).tocsr()

    def score_chunk(self, users: np.ndarray, B_hist: sp.csr_matrix) -> np.ndarray:
        return self.score_sparse(users, B_hist).toarray().astype(np.float32, copy=False)


class BPR:
    """BPR-MF：成对排序损失 + 矩阵分解（隐式反馈）。"""

    name = "BPR"

    def __init__(self, n_factors: int = 64, lr: float = 1e-3, reg: float = 1e-5,
                 epochs: int = 20, batch_size: int = 8192, seed: int = 42) -> None:
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None

    def fit(self, u: np.ndarray, i: np.ndarray,
            n_users: int, n_items: int) -> "BPR":
        torch.manual_seed(self.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"[BPR] 开始训练 device={dev} pairs={len(u):,d} factors={self.n_factors} "
            f"epochs={self.epochs}")

        P = nn.Embedding(n_users, self.n_factors).to(dev)
        Q = nn.Embedding(n_items, self.n_factors).to(dev)
        nn.init.normal_(P.weight, std=0.01)
        nn.init.normal_(Q.weight, std=0.01)
        opt = torch.optim.Adam(list(P.parameters()) + list(Q.parameters()), lr=self.lr)

        u_t = torch.as_tensor(u, dtype=torch.long, device=dev)
        i_t = torch.as_tensor(i, dtype=torch.long, device=dev)
        n = len(u_t)
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(n, device=dev)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                ub, ib = u_t[idx], i_t[idx]
                jb = torch.randint(0, n_items, (len(idx),), device=dev)
                pu, qi, qj = P(ub), Q(ib), Q(jb)
                x_ui = (pu * qi).sum(dim=1)
                x_uj = (pu * qj).sum(dim=1)
                loss = -torch.nn.functional.logsigmoid(x_ui - x_uj).mean()
                loss = loss + self.reg * (pu.pow(2).sum() + qi.pow(2).sum()
                                          + qj.pow(2).sum()) / len(idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(idx)
            log(f"[BPR] epoch {epoch:2d}/{self.epochs} loss={total / n:.4f} "
                f"({time.time() - t0:.0f}s)")

        self.P = P.weight.detach().cpu().numpy().astype(np.float32)
        self.Q = Q.weight.detach().cpu().numpy().astype(np.float32)
        log(f"[BPR] 训练完成 用时 {time.time() - t0:.1f}s")
        return self

    def score_chunk(self, users: np.ndarray, B_hist: sp.csr_matrix) -> np.ndarray:
        return self.P[users] @ self.Q.T


class SegmentedHybrid:
    """按「用户活跃度桶 × 商品流行度桶」分段设定融合权重的两路融合。

    score = z(BPR) + λ[u_bucket, i_bucket] · z(ItemKNN)
    分段权重在验证集上用坐标搜索学习，避免全局单一 λ 的低频/长尾偏差。
    """

    name = "Hybrid_Seg"

    def __init__(self, base, aux, lam_table: np.ndarray,
                 u_bucket: np.ndarray, i_bucket: np.ndarray) -> None:
        self.base = base
        self.aux = aux
        self.lam_table = lam_table
        self.u_bucket = u_bucket
        self.i_bucket = i_bucket

    def score_chunk(self, users: np.ndarray, B_hist: sp.csr_matrix) -> np.ndarray:
        # z-score 的均值是每用户常数，不影响排序，只保留尺度归一
        a = self.base.score_chunk(users, B_hist)
        a /= (a.std(axis=1, keepdims=True) + 1e-8)

        sub = self.aux.score_sparse(users, B_hist)
        n = a.shape[1]
        sum_k = np.asarray(sub.sum(axis=1)).ravel()
        sumsq = np.asarray(sub.multiply(sub).sum(axis=1)).ravel()
        mean_k = sum_k / n
        std_k = np.sqrt(np.maximum(sumsq / n - mean_k ** 2, 0.0)) + 1e-8

        lam_rows = self.lam_table[self.u_bucket[users]]          # (m, n_ib)
        for i in range(len(users)):
            s, e = sub.indptr[i], sub.indptr[i + 1]
            if e <= s:
                continue
            cols = sub.indices[s:e]
            coef = lam_rows[i].take(self.i_bucket[cols]) / std_k[i]
            a[i, cols] += (coef * sub.data[s:e]).astype(np.float32, copy=False)
        return a


class Hybrid:
    """两路协同得分的 z-score 加权融合：score = z(base) + λ · z(aux)。

    用于融合 BPR（隐因子）与 ItemKNN（邻域），二者覆盖与偏好结构互补。
    """

    name = "Hybrid"

    def __init__(self, base, aux, lam: float = 0.5) -> None:
        self.base = base
        self.aux = aux
        self.lam = lam

    def score_chunk(self, users: np.ndarray, B_hist: sp.csr_matrix) -> np.ndarray:
        # z-score 中的均值是每用户的常数，不影响排序，故只做尺度归一
        a = self.base.score_chunk(users, B_hist)
        a /= (a.std(axis=1, keepdims=True) + 1e-8)

        if hasattr(self.aux, "score_sparse"):
            # 稀疏分支只在非零位置写入，避免整块稠密矩阵
            sub = self.aux.score_sparse(users, B_hist)
            n = a.shape[1]
            sum_k = np.asarray(sub.sum(axis=1)).ravel()
            sumsq = np.asarray(sub.multiply(sub).sum(axis=1)).ravel()
            mean_k = sum_k / n
            std_k = np.sqrt(np.maximum(sumsq / n - mean_k ** 2, 0.0)) + 1e-8
            coeff = (self.lam / std_k).astype(np.float32)
            for i in range(len(users)):
                s, e = sub.indptr[i], sub.indptr[i + 1]
                if e > s:
                    a[i, sub.indices[s:e]] += coeff[i] * sub.data[s:e]
        else:
            b = self.aux.score_chunk(users, B_hist)
            b = (b - b.mean(axis=1, keepdims=True)) / (b.std(axis=1, keepdims=True) + 1e-8)
            a += self.lam * b
        return a.astype(np.float32, copy=False)


class MetaItemKNN:
    """基于商品元数据共购图（also_bought / also_viewed）的物品邻域推荐。

    与 ItemKNN（评分共现）互补：评分共现图只覆盖约 19% 的商品，元数据图
    可覆盖冷门 / 无共现的商品。相似度同样用非对称余弦 + top-k 截断：
        s_ij = A_ij / (deg_i^α · deg_j^(1-α))，deg 为图上的度。
    """

    name = "MetaItemKNN"

    def __init__(self, k: int = 200, alpha: float = 0.5) -> None:
        self.k = int(k)
        self.alpha = float(alpha)
        self.S: sp.csr_matrix | None = None

    def fit(self, A: sp.csr_matrix) -> "MetaItemKNN":
        """A 为 item-item 邻接矩阵（已对称化、非负）。"""
        t0 = time.time()
        A = A.tocsr()
        deg = np.maximum(np.asarray(A.sum(axis=1)).ravel().astype(np.float64), 1e-12)
        d_l = sp.diags(np.power(deg, -self.alpha))
        d_r = sp.diags(np.power(deg, -(1.0 - self.alpha)))
        S = (d_l @ A @ d_r).tocsr()
        S.sum_duplicates()
        S.setdiag(0.0)
        S.eliminate_zeros()
        self.S = topk_rows(S, self.k, keep_positive=True, drop_self=True)
        log(f"[MetaItemKNN] alpha={self.alpha} k={self.k} nnz={self.S.nnz:,d} "
            f"({time.time() - t0:.1f}s)")
        return self

    def score_sparse(self, users: np.ndarray, B_hist: sp.csr_matrix) -> sp.csr_matrix:
        return (B_hist[users] @ self.S).tocsr()

    def score_chunk(self, users: np.ndarray, B_hist: sp.csr_matrix) -> np.ndarray:
        return self.score_sparse(users, B_hist).toarray().astype(np.float32, copy=False)


class TripleHybrid:
    """三路融合：score = z(base) + λ1·z(aux1) + λ2·z(aux2)。

    用于 BPR（隐因子）+ 评分共现 ItemKNN + 元数据共购 MetaItemKNN 的融合，
    两个 aux 均走稀疏分支（须有 score_sparse 接口）。
    """

    name = "TripleHybrid"

    def __init__(self, base, aux1, aux2, lam1: float = 1.0, lam2: float = 1.0) -> None:
        self.base = base
        self.aux1 = aux1
        self.aux2 = aux2
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)

    def score_chunk(self, users: np.ndarray, B_hist: sp.csr_matrix) -> np.ndarray:
        a = self.base.score_chunk(users, B_hist)
        a /= (a.std(axis=1, keepdims=True) + 1e-8)
        n = a.shape[1]
        for aux, lam in ((self.aux1, self.lam1), (self.aux2, self.lam2)):
            sub = aux.score_sparse(users, B_hist)
            sum_k = np.asarray(sub.sum(axis=1)).ravel()
            sumsq = np.asarray(sub.multiply(sub).sum(axis=1)).ravel()
            mean_k = sum_k / n
            std_k = np.sqrt(np.maximum(sumsq / n - mean_k ** 2, 0.0)) + 1e-8
            coeff = (lam / std_k).astype(np.float32)
            for i in range(len(users)):
                s, e = sub.indptr[i], sub.indptr[i + 1]
                if e > s:
                    a[i, sub.indices[s:e]] += coeff[i] * sub.data[s:e]
        return a.astype(np.float32, copy=False)
