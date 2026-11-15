"""TAB 算法实验入口：消融对比 + 序数校准 + 与偏置基线对比。

用法:
    python -m src.models.rating_prediction.run_improved            # 全量
    python -m src.models.rating_prediction.run_improved --quick    # 小网格
    python -m src.models.rating_prediction.run_improved --sample-users 3000  # 冒烟测试
    python -m src.models.rating_prediction.run_improved --refit-full       # 最佳配置用全量训练重训
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.isotonic import IsotonicRegression

from src.evaluation.rating_metrics import evaluate
from src.utils.config import ensure_dir

from .common import (LOG_DIR, RESULT_COLUMNS, RatingData, load_rating_data, log)
from .time_bias import TimeAwareBias

IMPROVED_CSV = LOG_DIR / "results_improved.csv"


def make_view(data: RatingData, fit_mask: np.ndarray,
              eval_u: np.ndarray, eval_i: np.ndarray,
              eval_y: np.ndarray, eval_t: np.ndarray) -> RatingData:
    """用部分训练行（fit_mask）构造仅含该子集的 RatingData，并把评估集设为 eval_*。"""
    u = data.u_tr[fit_mask]
    i = data.i_tr[fit_mask]
    r = data.r_tr[fit_mask]
    t = data.t_tr[fit_mask]
    R = sp.csr_matrix((r, (u, i)), shape=data.R.shape, dtype=np.float32)
    R.sort_indices()
    B = R.copy()
    B.data = np.ones_like(B.data, dtype=np.float32)
    df_train = pd.DataFrame({
        "user_id": data.users.take(u), "item_id": data.items.take(i),
        "rating": r.astype(np.float32),
    })
    df_test = pd.DataFrame({
        "user_id": data.users.take(eval_u), "item_id": data.items.take(eval_i),
        "rating": eval_y.astype(np.float32),
    })
    return RatingData(
        train=df_train, test=df_test, users=data.users, items=data.items,
        u_tr=u, i_tr=i, r_tr=r, u_te=eval_u, i_te=eval_i, y_te=eval_y,
        R=R, B=B, t_tr=t, t_te=eval_t,
    )


def save_row(model: str, config: dict, metrics: dict, seconds: float, note: str = "") -> None:
    ensure_dir(LOG_DIR)
    payload = {
        "model": model,
        "config": json.dumps(config, ensure_ascii=False, sort_keys=True),
        "mse": metrics.get("mse"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "n_evaluated": metrics.get("n_evaluated"),
        "coverage": metrics.get("coverage"),
        "fallback_rate": 0.0,
        "fit_seconds": round(seconds, 1),
        "predict_seconds": "",
        "total_seconds": round(seconds, 1),
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
    }
    new_file = not IMPROVED_CSV.exists()
    with open(IMPROVED_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(payload)
    log(f"[improved] {model} rmse={payload['rmse']} mae={payload['mae']} {payload['total_seconds']}s")


def build_configs(quick: bool) -> list[dict]:
    """消融阶梯：逐项打开时间项、自适应收缩，最后加序数校准。"""
    base = dict(reg_u=3.0, reg_i=3.0, reg_alpha=50.0, reg_it=20.0, reg_g=5.0,
                n_iter=15, n_bins=12,
                use_global_drift=False, use_user_drift=False,
                use_item_time=False, adaptive_gamma=0.0, calib=False)
    cfgs = [dict(base, tag="A_bias_only")]
    cfgs.append(dict(base, tag="B_+global_drift", use_global_drift=True, reg_g=5.0))
    cfgs.append(dict(base, tag="C_+user_drift", use_global_drift=True, reg_g=5.0,
                     use_user_drift=True, reg_alpha=50.0))
    cfgs.append(dict(base, tag="D_+item_time", use_global_drift=True, reg_g=5.0,
                     use_user_drift=True, reg_alpha=50.0,
                     use_item_time=True, reg_it=20.0))
    cfgs.append(dict(base, tag="E_+adaptive_reg", use_global_drift=True, reg_g=5.0,
                     use_user_drift=True, reg_alpha=50.0,
                     use_item_time=True, reg_it=20.0, adaptive_gamma=0.5))
    cfgs.append(dict(base, tag="F_+calibration", use_global_drift=True, reg_g=5.0,
                     use_user_drift=True, reg_alpha=50.0,
                     use_item_time=True, reg_it=20.0, adaptive_gamma=0.5,
                     calib=True))
    if not quick:
        # 关键超参的轻量网格（都带校准）
        for nb in (6, 24):
            cfgs.append(dict(base, tag=f"G_bins{nb}", use_global_drift=True, reg_g=5.0,
                             use_user_drift=True, reg_alpha=50.0,
                             use_item_time=True, reg_it=20.0,
                             adaptive_gamma=0.5, calib=True, n_bins=nb))
        for ra in (20.0, 100.0):
            cfgs.append(dict(base, tag=f"H_alpha{int(ra)}", use_global_drift=True,
                             reg_g=5.0, use_user_drift=True, reg_alpha=ra,
                             use_item_time=True, reg_it=20.0,
                             adaptive_gamma=0.5, calib=True))
    return cfgs


def fit_one(cfg: dict, fit_data: RatingData):
    model = TimeAwareBias(
        reg_u=cfg["reg_u"], reg_i=cfg["reg_i"],
        reg_alpha=cfg["reg_alpha"], reg_it=cfg["reg_it"], reg_g=cfg["reg_g"],
        n_bins=cfg["n_bins"], use_global_drift=cfg["use_global_drift"],
        use_user_drift=cfg["use_user_drift"], use_item_time=cfg["use_item_time"],
        adaptive_gamma=cfg["adaptive_gamma"], n_iter=cfg["n_iter"], verbose=True,
    ).fit(fit_data)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description="TAB 改进算法实验")
    ap.add_argument("--sample-users", type=int, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="从训练集划出多少比例做校准/验证（默认 0.1）")
    ap.add_argument("--quick", action="store_true", help="只跑消融阶梯，不做超参网格")
    ap.add_argument("--refit-full", action="store_true",
                    help="用最佳配置在 100%% 训练集上重训后再评估测试集")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load_rating_data(sample_users=args.sample_users, seed=args.seed, with_time=True)
    n_tr = len(data.r_tr)

    # 划分训练集为 fit / val（val 仅用于校准与验证，不参与拟合）
    rng = np.random.default_rng(args.seed)
    is_val = rng.random(n_tr) < args.val_ratio
    fit_mask = ~is_val
    log(f"训练集拆分: fit={int(fit_mask.sum()):,d} val={int(is_val.sum()):,d}")

    val_view = make_view(data, fit_mask, data.u_tr[is_val], data.i_tr[is_val],
                         data.r_tr[is_val], data.t_tr[is_val])
    test_view = make_view(data, fit_mask, data.u_te, data.i_te, data.y_te, data.t_te)

    rows = []
    best = (None, float("inf"))
    for cfg in build_configs(args.quick):
        tag = cfg.pop("tag")
        t0 = time.time()
        model = fit_one(cfg, val_view)

        val_pred = model.predict(val_view)          # 在 val 上拟合校准器
        calib = None
        if cfg.get("calib"):
            calib = IsotonicRegression(increasing=True, out_of_bounds="clip")
            calib.fit(val_pred, val_view.y_te.astype(np.float64))

        val_metrics = evaluate(val_view.y_te, val_pred)
        pred = model.predict(test_view)
        if calib is not None:
            pred = calib.predict(pred)
        test_metrics = evaluate(test_view.y_te, pred)
        secs = time.time() - t0

        save_row(f"TAB_{tag}", cfg, test_metrics, secs,
                 note=f"val_rmse={val_metrics['rmse']:.4f} val_mae={val_metrics['mae']:.4f}"
                      f"{' calib' if calib is not None else ''}")
        rows.append((tag, val_metrics["rmse"], val_metrics["mae"],
                     test_metrics["rmse"], test_metrics["mae"], secs))
        # 以未校准的验证集 RMSE 作为选择依据，避免用测试集调参
        if val_metrics["rmse"] < best[1]:
            best = (dict(cfg, tag=tag), val_metrics["rmse"])

    # 汇总
    print("\n" + "=" * 84, flush=True)
    print(f"{'配置':<22}{'val_RMSE':>10}{'val_MAE':>10}{'test_RMSE':>11}{'test_MAE':>10}{'耗时(s)':>10}")
    print("-" * 84, flush=True)
    for tag, vr, vm, tr, tm, secs in rows:
        print(f"{tag:<22}{vr:>10.4f}{vm:>10.4f}{tr:>11.4f}{tm:>10.4f}{secs:>10.1f}", flush=True)
    print("=" * 84, flush=True)
    print(f"对照：已有最优 bias(reg=3) test RMSE=0.9023 MAE=0.6574", flush=True)

    if args.refit_full and best[0] is not None:
        cfg = {k: v for k, v in best[0].items() if k != "tag"}
        log(f"用最佳配置在 100% 训练集重训: {cfg}")
        t0 = time.time()
        full = make_view(data, np.ones(n_tr, dtype=bool), data.u_te, data.i_te,
                         data.y_te, data.t_te)
        model = fit_one(cfg, full)
        pred = model.predict(full)
        if cfg.get("calib"):
            calib = IsotonicRegression(increasing=True, out_of_bounds="clip")
            calib.fit(model.predict(val_view), val_view.y_te.astype(np.float64))
            pred = calib.predict(pred)
        m = evaluate(full.y_te, pred)
        save_row("TAB_full", cfg, m, time.time() - t0, note="refit on 100% train")
        print(f"\n[full] RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} 耗时={time.time() - t0:.1f}s",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
