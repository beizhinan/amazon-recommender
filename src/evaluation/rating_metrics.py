"""评分预测评估指标：MAE、RMSE（可选裁剪到评分区间）。"""
from __future__ import annotations

import numpy as np

RATING_RANGE = (1.0, 5.0)


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(
    y_true, y_pred, clip: tuple[float, float] | None = RATING_RANGE,
    fallback_mask=None,
) -> dict:
    """返回 MAE/RMSE 等指标；clip 用于把预测限制在合法评分区间。"""
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # 无法给出预测的样本（无邻居等）记为缺失，不参与指标计算
    valid = np.isfinite(y_pred)
    if clip is not None and valid.any():
        lo, hi = clip
        y_pred = np.where(valid, np.clip(y_pred, lo, hi), y_pred)

    n_total = int(len(y_true))
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mse": float("nan"),
                "n": n_total, "n_evaluated": 0, "coverage": 0.0,
                "fallback_rate": 1.0}

    metrics = {
        "mae": round(mae(np.asarray(y_true)[valid], y_pred[valid]), 6),
        "rmse": round(rmse(np.asarray(y_true)[valid], y_pred[valid]), 6),
        "mse": round(float(np.mean((np.asarray(y_true)[valid] - y_pred[valid]) ** 2)), 6),
        "n": n_total,
        "n_evaluated": n_valid,
        "coverage": round(n_valid / n_total, 4),
        "fallback_rate": round(float(fallback_mask.mean()), 4) if fallback_mask is not None else 0.0,
    }
    return metrics
