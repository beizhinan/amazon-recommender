"""基线模型：全局均值、用户/商品均值、正则化 Bias Baseline（ALS 求解）。"""
from __future__ import annotations

import numpy as np

from .common import RatingData, log


class GlobalMean:
    """所有预测均为训练集全局均值。"""

    name = "GlobalMean"

    def __init__(self, data: RatingData) -> None:
        self.mu = float(data.R.sum() / data.R.nnz)

    def predict(self, data: RatingData) -> np.ndarray:
        return np.full(len(data.y_te), self.mu, dtype=np.float64)


class BiasBaseline:
    """μ + b_u + b_i，交替最小二乘（ALS）求解，L2 正则。

    b_u = Σ_{i∈R_u}(r_ui - μ - b_i) / (|R_u| + λ_u)
    b_i = Σ_{u∈R_i}(r_ui - μ - b_u) / (|R_i| + λ_i)
    """

    name = "BiasBaseline"

    def __init__(self, reg_u: float = 10.0, reg_i: float = 10.0, n_iter: int = 15,
                 verbose: bool = True) -> None:
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.n_iter = n_iter
        self.verbose = verbose
        self.mu = 0.0
        self.b_u = None
        self.b_i = None

    def fit(self, data: RatingData) -> "BiasBaseline":
        R = data.R
        B = data.B
        self.mu = float(R.sum() / R.nnz)
        n_users, n_items = R.shape

        self.b_u = np.zeros(n_users, dtype=np.float64)
        self.b_i = np.zeros(n_items, dtype=np.float64)

        sum_u = np.asarray(R.sum(axis=1)).ravel().astype(np.float64)
        cnt_u = np.diff(R.indptr).astype(np.float64)
        sum_i = np.asarray(R.sum(axis=0)).ravel().astype(np.float64)
        cnt_i = np.asarray(B.sum(axis=0)).ravel().astype(np.float64)

        for it in range(1, self.n_iter + 1):
            self.b_u = (sum_u - self.mu * cnt_u - B @ self.b_i) / (cnt_u + self.reg_u)
            self.b_i = (sum_i - self.mu * cnt_i - B.T @ self.b_u) / (cnt_i + self.reg_i)
            if self.verbose and (it % 5 == 0 or it == self.n_iter):
                residual = data.r_tr - (self.mu + self.b_u[data.u_tr]
                                       + self.b_i[data.i_tr])
                rmse = float(np.sqrt(np.mean(residual.astype(np.float64) ** 2)))
                log(f"bias iter {it:2d}/{self.n_iter} train_rmse={rmse:.4f}")
        return self

    def predict(self, data: RatingData) -> np.ndarray:
        """对测试样本返回 μ + b_u + b_i。"""
        return (self.mu + self.b_u[data.u_te] + self.b_i[data.i_te]).astype(np.float64)


class UserMean(GlobalMean):
    """全局均值 + 用户偏置（只建模用户效应，用于消融/对比）。"""

    name = "UserMean"

    def __init__(self, data: RatingData, reg_u: float = 10.0) -> None:
        super().__init__(data)
        cnt = np.diff(data.R.indptr)
        sums = np.asarray(data.R.sum(axis=1)).ravel()
        self.b_u = (sums - self.mu * cnt) / (cnt + reg_u)

    def predict(self, data: RatingData) -> np.ndarray:
        return self.mu + self.b_u[data.u_te]


def build_centered(data: RatingData, mu: float, b_u: np.ndarray) -> "sp.csr_matrix":
    """返回用户中心化后的评分矩阵：r - μ - b_u。"""
    import scipy.sparse as sp

    mat = data.R.copy().astype(np.float32)
    coo = mat.tocoo()
    mat = sp.csr_matrix(
        (mat.data - (mu + b_u[coo.row]).astype(np.float32), (coo.row, coo.col)),
        shape=mat.shape,
    )
    mat.sort_indices()
    return mat


def build_residual(data: RatingData, mu: float, b_u: np.ndarray,
                   b_i: np.ndarray) -> "sp.csr_matrix":
    """返回残差矩阵：r - μ - b_u - b_i（邻域模型预测时使用）。"""
    import scipy.sparse as sp

    mat = data.R.copy().astype(np.float32)
    coo = mat.tocoo()
    mat = sp.csr_matrix(
        (mat.data - (mu + b_u[coo.row] + b_i[coo.col]).astype(np.float32),
         (coo.row, coo.col)),
        shape=mat.shape,
    )
    mat.sort_indices()
    return mat
