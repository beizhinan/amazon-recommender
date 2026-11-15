"""FAR：特征增强残差模型（Feature-Augmented Residual）。

设计依据（Books 5-core 实测结论）：
    - 双偏置 μ+b_u+b_i 已达 0.9023 RMSE，邻域/隐因子模型都无法再降；
    - 残差不是「线性可加效应」，而是与**样本量、离散度、时间位置**强相关的异方差结构：
      冷门商品与低频用户的偏差估计噪声大、应被更强收缩；
    - 固定 λ 的偏置模型对这些情形一视同仁，因此存在系统性改进空间。

做法：
    1. 用训练集统计量构造特征，训练样本一律使用**留一法（LOO）**，
       绝不把目标评分泄漏进特征（避免 target encoding 泄漏）；
    2. 目标为原始评分，让模型同时学习主效应与非线性收缩；
    3. HistGradientBoosting 拟合（分段常数的树天然具备自适应收缩能力）；
    4. 预测裁剪到 [1, 5]。
"""
from __future__ import annotations

import numpy as np

from .common import RatingData

FEATURE_NAMES = [
    "mean_u",      # 用户平均分（留一）
    "mean_i",      # 商品平均分（留一）
    "std_u",       # 用户评分离散度（留一）
    "std_i",       # 商品评分离散度（留一）
    "log_cnt_u",   # 用户活跃度
    "log_cnt_i",   # 商品流行度
    "t_year",      # 评分时间（年）
    "dev_user",    # 相对该用户平均评分时间的偏离
    "dev_item",    # 相对该商品平均评分时间的偏离
    "item_span",   # 该商品评分跨越的时间跨度
]


def _agg(idx: np.ndarray, r: np.ndarray, t: np.ndarray, size: int):
    """分组求和：计数 / 评分和 / 评分平方和 / 时间和 / 时间平方和。"""
    return (
        np.bincount(idx, minlength=size).astype(np.float64),
        np.bincount(idx, weights=r, minlength=size),
        np.bincount(idx, weights=r * r, minlength=size),
        np.bincount(idx, weights=t, minlength=size),
        np.bincount(idx, weights=t * t, minlength=size),
    )


def _row_stats(cnt, s, s2, st, st2, idx, r=None, t=None, mu=0.0):
    """按行取统计量；给出 r 时做留一（扣除该行自身贡献）。"""
    c, sm, sm2, stm, stm2 = cnt[idx], s[idx], s2[idx], st[idx], st2[idx]
    if r is not None:
        c, sm, sm2 = c - 1.0, sm - r, sm2 - r * r
        stm, stm2 = stm - t, stm2 - t * t
    safe = np.maximum(c, 1.0)
    mean = np.where(c > 0, sm / safe, mu)
    var = np.where(c > 1, sm2 / safe - mean ** 2, 0.0)
    std = np.sqrt(np.maximum(var, 0.0))
    t_mean = np.where(c > 0, stm / safe, np.nan)
    t_var = np.where(c > 1, stm2 / safe - t_mean ** 2, 0.0)
    return mean, std, t_mean, 2.0 * np.sqrt(np.maximum(t_var, 0.0)), np.maximum(c, 0.0)


def build_features(data: RatingData, fit_mask: np.ndarray,
                   u: np.ndarray, i: np.ndarray, t: np.ndarray,
                   r: np.ndarray | None = None) -> np.ndarray:
    """构造特征矩阵。r 非空表示这些行属于训练集，统计量按留一法计算。"""
    uf, i_f = data.u_tr[fit_mask], data.i_tr[fit_mask]
    rf = data.r_tr[fit_mask].astype(np.float64)
    tf = data.t_tr[fit_mask].astype(np.float64)
    mu = float(rf.mean())
    n_users, n_items = data.R.shape

    agg_u = _agg(uf, rf, tf, n_users)
    agg_i = _agg(i_f, rf, tf, n_items)

    mean_u, std_u, tmean_u, _, cnt_u = _row_stats(*agg_u, u, r=r, t=t, mu=mu)
    mean_i, std_i, tmean_i, span_i, cnt_i = _row_stats(*agg_i, i, r=r, t=t, mu=mu)
    tmean_u = np.where(np.isnan(tmean_u), t, tmean_u)
    tmean_i = np.where(np.isnan(tmean_i), t, tmean_i)

    return np.column_stack([
        mean_u, mean_i, std_u, std_i,
        np.log1p(cnt_u), np.log1p(cnt_i),
        t, t - tmean_u, t - tmean_i, span_i,
    ]).astype(np.float32)
