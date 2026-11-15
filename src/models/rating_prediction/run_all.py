"""评分预测实验入口：批量运行基线模型，记录并汇总 MAE / RMSE。

用法（项目根目录）:
    python -m src.models.rating_prediction.run_all --models all
    python -m src.models.rating_prediction.run_all --models itemcf,usercf --sample-users 20000 --quick
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import pandas as pd

from src.evaluation.rating_metrics import evaluate
from src.models.rating_prediction.baselines import (
    BiasBaseline, GlobalMean, UserMean,
)
from src.models.rating_prediction.common import (
    LOG_DIR, RatingData, Timer, load_config, load_rating_data, log, save_result,
)
from src.utils.config import ensure_dir

# 小网格搜索空间（--quick 时自动裁剪）
GRIDS: dict[str, list[dict]] = {
    "global": [{}],
    "usermean": [{"reg_u": 10}, {"reg_u": 25}],
    "bias": [
        {"reg_u": 0.5, "reg_i": 0.5},
        {"reg_u": 1, "reg_i": 1},
        {"reg_u": 2, "reg_i": 2},
        {"reg_u": 3, "reg_i": 3},
        {"reg_u": 5, "reg_i": 5},
        {"reg_u": 8, "reg_i": 8},
        {"reg_u": 10, "reg_i": 10},
        {"reg_u": 25, "reg_i": 25},
        {"reg_u": 10, "reg_i": 25},
    ],
    "itemcf": [
        {"k": 50, "shrink": 25},
        {"k": 100, "shrink": 10},
        {"k": 100, "shrink": 25},
        {"k": 100, "shrink": 50},
        {"k": 200, "shrink": 25},
    ],
    "usercf": [
        {"k": 20, "shrink": 25},
        {"k": 50, "shrink": 10},
        {"k": 50, "shrink": 25},
        {"k": 100, "shrink": 25},
    ],
    "slopeone": [
        {"k": 50},
        {"k": 100},
        {"k": 200},
        {"k": 100, "min_count": 2},
    ],
    "svd": [
        {"n_factors": 50, "n_epochs": 30, "reg_all": 0.1},
        {"n_factors": 100, "n_epochs": 30, "reg_all": 0.05},
        {"n_factors": 100, "n_epochs": 30, "reg_all": 0.1},
        {"n_factors": 150, "n_epochs": 30, "reg_all": 0.1},
    ],
    "ncf": [
        {"gmf_dim": 32, "mlp_dim": 32, "layers": [64, 32, 16], "epochs": 10},
        {"gmf_dim": 64, "mlp_dim": 64, "layers": [128, 64, 32], "epochs": 15},
    ],
}

QUICK_GRIDS: dict[str, list[dict]] = {
    "global": [{}],
    "usermean": [{"reg_u": 10}],
    "bias": [{"reg_u": 10, "reg_i": 10}],
    "itemcf": [{"k": 50, "shrink": 25}],
    "usercf": [{"k": 50, "shrink": 25}],
    "slopeone": [{"k": 100}],
    "svd": [{"n_factors": 50, "n_epochs": 10, "reg_all": 0.1}],
    "ncf": [{"gmf_dim": 32, "mlp_dim": 32, "layers": [64, 32, 16], "epochs": 3}],
}

ALL_MODELS = list(GRIDS.keys())


def _fit_bias(data: RatingData, reg_u: float = 5, reg_i: float = 5) -> BiasBaseline:
    return BiasBaseline(reg_u=reg_u, reg_i=reg_i, n_iter=15).fit(data)


def run_one(name: str, params: dict, data: RatingData, note: str = "") -> dict | None:
    """运行单个模型配置，返回指标字典。"""
    timer = Timer()
    try:
        if name == "global":
            model = GlobalMean(data)
            pred, fallback = model.predict(data), np.zeros(len(data.y_te), dtype=bool)
        elif name == "usermean":
            model = UserMean(data, reg_u=params.get("reg_u", 10))
            pred, fallback = model.predict(data), np.zeros(len(data.y_te), dtype=bool)
        elif name == "bias":
            model = BiasBaseline(reg_u=params["reg_u"], reg_i=params["reg_i"], n_iter=15)
            model.fit(data)
            pred = model.predict(data)
            fallback = np.zeros(len(data.y_te), dtype=bool)
        elif name in ("itemcf", "usercf", "slopeone"):
            bias = _fit_bias(data)
            if name == "itemcf":
                from src.models.rating_prediction.knn import ItemCF
                model = ItemCF(k=params["k"], shrink=params["shrink"])
            elif name == "usercf":
                from src.models.rating_prediction.knn import UserCF
                model = UserCF(k=params["k"], shrink=params["shrink"])
            else:
                from src.models.rating_prediction.slope_one import SlopeOne
                model = SlopeOne(k=params["k"], min_count=params.get("min_count", 1))
            model.fit(data, bias.mu, bias.b_u, bias.b_i)
            pred, fallback = model.predict(data)
        elif name == "svd":
            from src.models.rating_prediction.svd_surprise import SurpriseSVD
            model = SurpriseSVD(**params)
            model.fit(data)
            pred, fallback = model.predict(data)
        elif name == "ncf":
            from src.models.rating_prediction.ncf import NCF
            model = NCF(**params)
            model.fit(data)
            pred, fallback = model.predict(data)
        else:
            raise ValueError(f"未知模型: {name}")

        timer.mark("fit")
        timer.mark("predict")
    except Exception:  # noqa: BLE001 - 记录失败但不中断批量实验
        log(f"[fail] {name} {params}")
        traceback.print_exc()
        return None

    metrics = evaluate(data.y_te, pred, fallback_mask=fallback)
    save_result(name, params, metrics, timer, note=note)
    log(f"[{name}] {params} -> MAE={metrics['mae']} RMSE={metrics['rmse']} "
        f"fallback={metrics['fallback_rate']} time={timer.elapsed()}s")
    return {"model": name, "config": params, **metrics, "seconds": timer.elapsed()}


def print_summary(runs: list[dict], data: RatingData) -> None:
    if not runs:
        log("没有成功的实验")
        return
    df = pd.DataFrame(runs)
    best = df.loc[df.groupby("model")["rmse"].idxmin()].sort_values("rmse")
    log("===== 各模型最佳配置（按 RMSE 排序）=====")
    for _, row in best.iterrows():
        log(f"{row['model']:<12s} RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} "
            f"config={row['config']}")
    out = LOG_DIR / "summary_best.csv"
    ensure_dir(LOG_DIR)
    best.to_csv(out, index=False, encoding="utf-8")
    log(f"[summary] 写入 {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="评分预测基线实验")
    parser.add_argument("--models", default="all",
                        help="逗号分隔的模型列表，或 all")
    parser.add_argument("--sample-users", type=int, default=None,
                        help="仅抽样这么多个用户（smoke test 用）")
    parser.add_argument("--quick", action="store_true", help="使用最小网格")
    parser.add_argument("--note", default="", help="实验备注")
    args = parser.parse_args()

    cfg = load_config()
    data = load_rating_data(cfg, sample_users=args.sample_users)
    note = args.note or (f"sample_users={args.sample_users}" if args.sample_users else "")
    log(f"[data] users={data.n_users:,d} items={data.n_items:,d} "
        f"train={len(data.r_tr):,d} test={len(data.y_te):,d}")

    models = ALL_MODELS if args.models == "all" else [m.strip() for m in args.models.split(",")]
    grid = QUICK_GRIDS if args.quick else GRIDS

    runs: list[dict] = []
    for name in models:
        if name not in grid:
            log(f"[skip] 未知模型 {name}")
            continue
        configs = grid[name]
        log(f"===== {name} ({len(configs)} 个配置) =====")
        for i, params in enumerate(configs, 1):
            log(f"[run] {name} {i}/{len(configs)} {params}")
            result = run_one(name, params, data, note=note)
            if result:
                runs.append(result)

    print_summary(runs, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
