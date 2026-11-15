"""评分预测公共模块：数据加载、ID 映射、稀疏矩阵构造、top-K 截断、结果记录。"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.config import PROJECT_ROOT, ensure_dir, load_config

LOG_DIR = PROJECT_ROOT / "experiments" / "logs" / "rating_prediction"
RESULT_CSV = LOG_DIR / "results.csv"
RESULT_COLUMNS = [
    "model", "config", "mse", "mae", "rmse", "n_evaluated", "coverage",
    "fallback_rate", "fit_seconds", "predict_seconds", "total_seconds",
    "run_time", "note",
]


def log(msg: str) -> None:
    """带时间戳的进度输出，强制刷新以便实时查看。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class RatingData:
    """训练/测试集及其索引化表示。"""

    train: pd.DataFrame
    test: pd.DataFrame
    users: np.ndarray          # 索引 -> user_id
    items: np.ndarray          # 索引 -> item_id
    u_tr: np.ndarray           # 训练集用户索引
    i_tr: np.ndarray           # 训练集商品索引
    r_tr: np.ndarray           # 训练集评分
    u_te: np.ndarray           # 测试集用户索引
    i_te: np.ndarray           # 测试集商品索引
    y_te: np.ndarray           # 测试集真实评分
    R: sp.csr_matrix           # users x items 评分矩阵
    B: sp.csr_matrix           # users x items 二值矩阵
    t_tr: np.ndarray | None = None   # 训练集时间（单位：年，从最早时间起算）
    t_te: np.ndarray | None = None   # 测试集时间（同一坐标）

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def user_row(self, u: int) -> tuple[np.ndarray, np.ndarray]:
        """返回用户 u 在训练集中评过分的 (商品索引, 评分)。索引有序。"""
        start, end = self.R.indptr[u], self.R.indptr[u + 1]
        return self.R.indices[start:end], self.R.data[start:end].astype(np.float64)

    def test_groups(self) -> list[tuple[int, np.ndarray]]:
        """把测试集按用户分组，返回 [(user_idx, 该用户在测试集中的行位置)]。"""
        order = np.argsort(self.u_te, kind="mergesort")
        sorted_u = self.u_te[order]
        boundaries = np.flatnonzero(np.diff(sorted_u)) + 1
        groups_u = np.split(sorted_u, boundaries)
        groups_pos = np.split(order, boundaries)
        return [(int(u[0]), pos) for u, pos in zip(groups_u, groups_pos)]


def load_rating_data(cfg: dict | None = None, sample_users: int | None = None,
                     seed: int = 42, with_time: bool = False) -> RatingData:
    """读取 train/test parquet，构造索引与稀疏矩阵。

    sample_users 仅用于 smoke test：随机抽取若干用户（训练与测试同时过滤）。
    with_time=True 时额外加载时间戳并换算为「年」（用于时序建模）。
    """
    cfg = cfg or load_config()
    paths = cfg["paths"]
    cols = ["user_id", "item_id", "rating"]
    if with_time:
        cols.append("timestamp")
    log(f"读取数据: {paths['train_parquet']}")
    train = pd.read_parquet(paths["train_parquet"], columns=cols)
    test = pd.read_parquet(paths["test_parquet"], columns=cols)
    log(f"原始划分: train={len(train):,d} test={len(test):,d}")

    if sample_users is not None and sample_users < train["user_id"].nunique():
        rng = np.random.default_rng(seed)
        all_users = np.union1d(train["user_id"].unique(), test["user_id"].unique())
        picked = rng.choice(all_users, size=sample_users, replace=False)
        train = train[train["user_id"].isin(picked)].reset_index(drop=True)
        test = test[test["user_id"].isin(picked)].reset_index(drop=True)
        log(f"抽样后: train={len(train):,d} test={len(test):,d} (sample_users={sample_users})")

    users = np.sort(pd.unique(pd.concat([train["user_id"], test["user_id"]], ignore_index=True)))
    items = np.sort(pd.unique(pd.concat([train["item_id"], test["item_id"]], ignore_index=True)))
    u_index = pd.Index(users)
    i_index = pd.Index(items)

    u_tr = u_index.get_indexer(train["user_id"].values)
    i_tr = i_index.get_indexer(train["item_id"].values)
    r_tr = train["rating"].to_numpy(dtype=np.float32)
    u_te = u_index.get_indexer(test["user_id"].values)
    i_te = i_index.get_indexer(test["item_id"].values)
    y_te = test["rating"].to_numpy(dtype=np.float32)

    R = sp.csr_matrix(
        (r_tr, (u_tr, i_tr)), shape=(len(users), len(items)), dtype=np.float32
    )
    R.sort_indices()
    B = R.copy()
    B.data = np.ones_like(B.data, dtype=np.float32)
    log(f"稀疏矩阵完成: users={len(users):,d} items={len(items):,d} nnz={R.nnz:,d}")

    t_tr = t_te = None
    if with_time:
        ts_all = pd.concat([train["timestamp"], test["timestamp"]], ignore_index=True)
        t0 = float(pd.to_numeric(ts_all, errors="coerce").min())
        seconds_per_year = 365.25 * 24 * 3600
        t_tr = ((pd.to_numeric(train["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
                 - t0) / seconds_per_year).astype(np.float32)
        t_te = ((pd.to_numeric(test["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
                 - t0) / seconds_per_year).astype(np.float32)
        log(f"时间跨度: {float(np.nanmin(t_tr)):.2f} ~ {float(np.nanmax(t_te)):.2f} 年")

    return RatingData(
        train=train, test=test, users=users, items=items,
        u_tr=u_tr, i_tr=i_tr, r_tr=r_tr, u_te=u_te, i_te=i_te, y_te=y_te,
        R=R, B=B, t_tr=t_tr, t_te=t_te,
    )


def topk_rows(mat: sp.csr_matrix, k: int, keep_positive: bool = True,
              drop_self: bool = True) -> sp.csr_matrix:
    """对 CSR 矩阵逐行保留权重最大的 k 个元素（常用于相似度截断）。"""
    rows, cols, vals = [], [], []
    for i in range(mat.shape[0]):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        idx = mat.indices[start:end]
        dat = mat.data[start:end]
        if keep_positive:
            mask = dat > 0
            idx, dat = idx[mask], dat[mask]
        if drop_self:
            keep = idx != i
            idx, dat = idx[keep], dat[keep]
        if len(dat) > k:
            sel = np.argpartition(dat, -k)[-k:]
            idx, dat = idx[sel], dat[sel]
        order = np.argsort(idx)
        rows.append(np.full(len(idx), i, dtype=np.int32))
        cols.append(idx.astype(np.int32))
        vals.append(dat.astype(np.float32))
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=mat.shape,
    )


def make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


class Timer:
    """简易计时器，记录多个阶段耗时。"""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> float:
        elapsed = time.time() - self.t0
        self.marks[name] = round(elapsed, 1)
        return elapsed

    def elapsed(self) -> float:
        return round(time.time() - self.t0, 1)


def save_result(model: str, config: dict, metrics: dict, timer: Timer,
                note: str = "") -> None:
    """把一次实验追加写入 results.csv，同时落盘一份 json 方便追溯。"""
    ensure_dir(LOG_DIR)
    payload = {
        "model": model,
        "config": json.dumps(config, ensure_ascii=False, sort_keys=True),
        "mse": metrics.get("mse"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "n_evaluated": metrics.get("n_evaluated"),
        "coverage": metrics.get("coverage"),
        "fallback_rate": metrics.get("fallback_rate"),
        "fit_seconds": timer.marks.get("fit"),
        "predict_seconds": timer.marks.get("predict"),
        "total_seconds": timer.elapsed(),
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
    }
    new_file = not RESULT_CSV.exists()
    if not new_file:
        with open(RESULT_CSV, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != RESULT_COLUMNS:
            backup = RESULT_CSV.with_name(
                f"results_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            RESULT_CSV.rename(backup)
            new_file = True
    with open(RESULT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(payload)

    detail = LOG_DIR / f"{model.replace('/', '_')}" / f"{int(time.time())}.json"
    ensure_dir(detail.parent)
    with open(detail, "w", encoding="utf-8") as f:
        json.dump({**payload, "metrics": metrics}, f, ensure_ascii=False, indent=2)
    log(f"[result] {model} mae={payload['mae']} rmse={payload['rmse']} "
        f"coverage={payload['coverage']} time={payload['total_seconds']}s")
