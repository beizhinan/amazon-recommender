"""Surprise SVD（带偏置的矩阵分解，SGD 优化）。"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from surprise import AlgoBase, Dataset, Reader, SVD

from .common import RatingData, log


class SurpriseSVD:
    name = "SVD"

    def __init__(self, n_factors: int = 100, n_epochs: int = 30,
                 lr_all: float = 0.005, reg_all: float = 0.1,
                 random_state: int = 42, verbose: bool = True) -> None:
        self.params = dict(
            n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all,
            reg_all=reg_all, random_state=random_state, verbose=verbose,
        )
        self.algo: AlgoBase | None = None

    def fit(self, data: RatingData) -> "SurpriseSVD":
        t0 = time.time()
        log(f"[SVD] 开始训练 {self.params}")
        df = pd.DataFrame({
            "user": data.train["user_id"].values,
            "item": data.train["item_id"].values,
            "rating": data.train["rating"].values,
        })
        reader = Reader(rating_scale=(1, 5))
        trainset = Dataset.load_from_df(df, reader).build_full_trainset()
        self.algo = SVD(**self.params)
        self.algo.fit(trainset)
        log(f"[SVD] 训练完成 用时 {time.time() - t0:.1f}s")
        return self

    def predict(self, data: RatingData) -> tuple[np.ndarray, np.ndarray]:
        test = data.test
        pred = np.empty(len(test), dtype=np.float64)
        algo = self.algo
        uids = test["user_id"].values
        iids = test["item_id"].values
        total = len(test)
        step = max(1, total // 10)
        t0 = time.time()
        for idx in range(total):
            pred[idx] = algo.predict(uids[idx], iids[idx]).est
            if (idx + 1) % step == 0 or idx == total - 1:
                log(f"[SVD] 预测 {idx + 1}/{total} ({time.time() - t0:.0f}s)")
        fallback = np.zeros(len(test), dtype=bool)
        return pred, fallback
