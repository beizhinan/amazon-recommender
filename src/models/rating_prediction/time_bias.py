"""TAB：时序—活跃度增强偏置模型（Time-Activity Bias）。

设计动机（基于 Books 5-core 实测结论）：
    - 双偏置 μ+b_u+b_i 已达 0.9023 RMSE，且 63%/81% 的邻域预测无有效邻居；
    - 每用户训练评分中位数仅 6 条，fixed λ 对小样本用户收缩不足；
    - 评分跨 1996-2018，用户打分尺度与商品口碑随时间漂移。

模型：
    r̂_ui(t) = μ + b_u + α_u · dev_u(t) + b_i + b_{i,Bin(t)} + g_{Bin(t)}
    dev_u(t) = sign(t - t_u) · |t - t_u|^0.4     （Koren 2009 的漂移基函数）

求解：ALS 坐标下降。每个参数块都用「残差回加自身贡献」的技巧做闭式更新，
        全程只依赖 np.bincount，单轮复杂度 O(nnz)。
"""
from __future__ import annotations

import time

import numpy as np

from .common import RatingData, log


class TimeAwareBias:
    """带时间漂移、活跃度自适应收缩与商品时间分箱的偏置模型。"""

    name = "TAB"

    def __init__(
        self,
        reg_u: float = 3.0,
        reg_i: float = 3.0,
        reg_alpha: float = 50.0,
        reg_it: float = 20.0,
        reg_g: float = 5.0,
        n_bins: int = 12,
        use_global_drift: bool = True,
        use_user_drift: bool = True,
        use_item_time: bool = True,
        adaptive_gamma: float = 0.0,
        n_iter: int = 20,
        clip: tuple[float, float] = (1.0, 5.0),
        verbose: bool = True,
    ) -> None:
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.reg_alpha = reg_alpha
        self.reg_it = reg_it
        self.reg_g = reg_g
        self.n_bins = n_bins
        self.use_global_drift = use_global_drift
        self.use_user_drift = use_user_drift
        self.use_item_time = use_item_time
        self.adaptive_gamma = adaptive_gamma
        self.n_iter = n_iter
        self.clip = clip
        self.verbose = verbose

        self.mu = 0.0
        self.b_u = self.alpha_u = self.b_i = self.b_it = self.g_bin = None
        self.t_user = None
        self.t_scale = 1.0

    # ---------- 时间特征 ----------
    def _bins(self, t: np.ndarray) -> np.ndarray:
        idx = (t / self.t_scale * self.n_bins).astype(np.int32)
        return np.clip(idx, 0, self.n_bins - 1)

    @staticmethod
    def _dev(dt: np.ndarray) -> np.ndarray:
        return np.sign(dt) * np.abs(dt) ** 0.4

    # ---------- 训练 ----------
    def fit(self, data: RatingData) -> "TimeAwareBias":
        if data.t_tr is None:
            raise ValueError("TimeAwareBias 需要时间戳，请用 load_rating_data(with_time=True)")
        t0 = time.time()
        u, i, r = data.u_tr, data.i_tr, data.r_tr.astype(np.float64)
        t = data.t_tr.astype(np.float64)
        n_users, n_items = data.R.shape

        self.mu = float(r.mean())
        self.t_scale = max(float(t.max()), 1e-6)
        bins = self._bins(t)
        n_bins = self.n_bins

        cnt_u = np.bincount(u, minlength=n_users).astype(np.float64)
        lam_u = np.full(n_users, self.reg_u, dtype=np.float64)
        if self.adaptive_gamma > 0:
            # 经验贝叶斯式自适应收缩：样本越少的用户收缩越强
            med = float(np.median(cnt_u[cnt_u > 0]))
            lam_u = self.reg_u * np.power(med / np.maximum(cnt_u, 1.0), self.adaptive_gamma)
            lam_u = np.clip(lam_u, self.reg_u * 0.25, self.reg_u * 4.0)

        # 用户平均时间（用于 dev_u）
        sum_t_u = np.bincount(u, weights=t, minlength=n_users)
        self.t_user = sum_t_u / np.maximum(cnt_u, 1.0)
        dev = self._dev(t - self.t_user[u])

        code_it = i * n_bins + bins
        n_it = n_items * n_bins

        self.b_u = np.zeros(n_users)
        self.alpha_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.b_it = np.zeros(n_it)
        self.g_bin = np.zeros(n_bins)

        e = r - self.mu

        # 各参数块的设计值及是否启用
        blocks = [("b_u", 1)]
        if self.use_user_drift:
            blocks.append(("alpha_u", 2))
        blocks.append(("b_i", 3))
        if self.use_item_time:
            blocks.append(("b_it", 4))
        if self.use_global_drift:
            blocks.append(("g_bin", 5))

        for it in range(1, self.n_iter + 1):
            for kind, _ in blocks:
                if kind == "b_u":
                    num = np.bincount(u, weights=e + self.b_u[u], minlength=n_users)
                    new = num / (cnt_u + lam_u)
                    e -= (new - self.b_u)[u]
                    self.b_u = new
                elif kind == "alpha_u":
                    a2 = np.bincount(u, weights=dev * dev, minlength=n_users)
                    ae = np.bincount(u, weights=dev * e, minlength=n_users)
                    new = (ae + self.alpha_u * a2) / (a2 + self.reg_alpha)
                    e -= (new - self.alpha_u)[u] * dev
                    self.alpha_u = new
                elif kind == "b_i":
                    num = np.bincount(i, weights=e + self.b_i[i], minlength=n_items)
                    cnt_i = np.bincount(i, minlength=n_items).astype(np.float64)
                    new = num / (cnt_i + self.reg_i)
                    e -= (new - self.b_i)[i]
                    self.b_i = new
                elif kind == "b_it":
                    num = np.bincount(code_it, weights=e + self.b_it[code_it], minlength=n_it)
                    cnt = np.bincount(code_it, minlength=n_it).astype(np.float64)
                    new = num / (cnt + self.reg_it)
                    new[cnt == 0] = 0.0
                    e -= (new - self.b_it)[code_it]
                    self.b_it = new
                elif kind == "g_bin":
                    num = np.bincount(bins, weights=e + self.g_bin[bins], minlength=n_bins)
                    cnt = np.bincount(bins, minlength=n_bins).astype(np.float64)
                    new = num / (cnt + self.reg_g)
                    e -= (new - self.g_bin)[bins]
                    self.g_bin = new

            if self.verbose and (it % 5 == 0 or it == self.n_iter):
                rmse = float(np.sqrt(np.mean(e ** 2)))
                log(f"TAB iter {it:2d}/{self.n_iter} train_rmse={rmse:.4f}")
        log(f"TAB 训练完成 {time.time() - t0:.1f}s")
        return self

    # ---------- 预测 ----------
    def predict(self, data: RatingData, return_parts: bool = False):
        t = data.t_te.astype(np.float64)
        bins = self._bins(t)
        pred = np.full(len(t), self.mu, dtype=np.float64)

        pred += self.b_u[data.u_te]
        if self.use_user_drift:
            dev = self._dev(t - self.t_user[data.u_te])
            pred += self.alpha_u[data.u_te] * dev
        pred += self.b_i[data.i_te]
        if self.use_item_time:
            pred += self.b_it[data.i_te * self.n_bins + bins]
        if self.use_global_drift:
            pred += self.g_bin[bins]

        pred = np.clip(pred, *self.clip)
        if return_parts:
            return pred, {"mu": self.mu, "b_u": self.b_u[data.u_te],
                          "b_i": self.b_i[data.i_te]}
        return pred
