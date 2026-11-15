"""FAR 实验入口：特征增强残差模型 vs 偏置基线（消融 + 轻量网格）。

用法:
    python -m src.models.rating_prediction.run_far                    # 全量
    python -m src.models.rating_prediction.run_far --sample-users 3000
    python -m src.models.rating_prediction.run_far --quick
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.evaluation.rating_metrics import evaluate

from .baselines import BiasBaseline
from .common import load_rating_data, log
from .far_model import FEATURE_NAMES, build_features
from .run_improved import make_view, save_row


def main() -> int:
    ap = argparse.ArgumentParser(description="FAR 特征增强残差模型")
    ap.add_argument("--sample-users", type=int, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load_rating_data(sample_users=args.sample_users, seed=args.seed, with_time=True)
    n_tr = len(data.r_tr)
    rng = np.random.default_rng(args.seed)
    is_val = rng.random(n_tr) < args.val_ratio
    fit_mask = ~is_val
    log(f"训练集拆分: fit={int(fit_mask.sum()):,d} val={int(is_val.sum()):,d}")

    val_view = make_view(data, fit_mask, data.u_tr[is_val], data.i_tr[is_val],
                         data.r_tr[is_val], data.t_tr[is_val])
    test_view = make_view(data, fit_mask, data.u_te, data.i_te, data.y_te, data.t_te)

    # ---- 参考基线：同一 fit 子集上训练的正则化偏置 ----
    t0 = time.time()
    bias = BiasBaseline(reg_u=3.0, reg_i=3.0).fit(val_view)
    b_val = evaluate(val_view.y_te, bias.predict(val_view))
    b_test = evaluate(test_view.y_te, bias.predict(test_view))
    log(f"[bias] val RMSE={b_val['rmse']:.4f} test RMSE={b_test['rmse']:.4f} "
        f"MAE={b_test['mae']:.4f} ({time.time() - t0:.1f}s)")
    save_row("REF_bias", {"reg_u": 3.0, "reg_i": 3.0}, b_test, time.time() - t0,
             note=f"val_rmse={b_val['rmse']:.4f} 同 fit 子集对照")

    # ---- 构造特征（训练行用留一统计，val/test 用全量统计） ----
    t0 = time.time()
    feats = {
        "train": build_features(data, fit_mask, data.u_tr[fit_mask], data.i_tr[fit_mask],
                                data.t_tr[fit_mask], r=data.r_tr[fit_mask].astype(np.float64)),
        "val": build_features(data, fit_mask, data.u_tr[is_val], data.i_tr[is_val],
                              data.t_tr[is_val]),
        "test": build_features(data, fit_mask, data.u_te, data.i_te, data.t_te),
    }
    log(f"特征构造完成 {time.time() - t0:.1f}s 维度={feats['train'].shape} 特征={FEATURE_NAMES}")

    grids = [
        dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=100,
             l2_regularization=1.0, max_bins=255),
        dict(max_iter=600, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=200,
             l2_regularization=2.0, max_bins=255),
    ]
    if args.quick:
        grids = grids[:1]

    y_tr = data.r_tr[fit_mask].astype(np.float64)
    rows = []
    for gi, params in enumerate(grids):
        t0 = time.time()
        model = HistGradientBoostingRegressor(
            loss="squared_error", early_stopping=True, validation_fraction=0.05,
            n_iter_no_change=30, random_state=args.seed, verbose=0, **params)
        model.fit(feats["train"], y_tr)
        secs = time.time() - t0

        p_val = np.clip(model.predict(feats["val"]), 1.0, 5.0)
        p_test = np.clip(model.predict(feats["test"]), 1.0, 5.0)
        m_val = evaluate(val_view.y_te, p_val)
        m_test = evaluate(test_view.y_te, p_test)

        tag = f"FAR_g{gi}"
        save_row(tag, params, m_test, secs,
                 note=f"val_rmse={m_val['rmse']:.4f} n_iter={model.n_iter_}")
        rows.append((tag, m_val["rmse"], m_val["mae"], m_test["rmse"], m_test["mae"], secs))
        log(f"[{tag}] val RMSE={m_val['rmse']:.4f} test RMSE={m_test['rmse']:.4f} "
            f"MAE={m_test['mae']:.4f} ({secs:.1f}s)")

    print("\n" + "=" * 80, flush=True)
    print(f"{'模型':<18}{'val_RMSE':>10}{'val_MAE':>10}{'test_RMSE':>11}{'test_MAE':>10}{'耗时(s)':>10}")
    print("-" * 80, flush=True)
    print(f"{'bias(同子集)':<18}{b_val['rmse']:>10.4f}{b_val['mae']:>10.4f}"
          f"{b_test['rmse']:>11.4f}{b_test['mae']:>10.4f}{'-':>10}", flush=True)
    for tag, vr, vm, tr, tm, secs in rows:
        print(f"{tag:<18}{vr:>10.4f}{vm:>10.4f}{tr:>11.4f}{tm:>10.4f}{secs:>10.1f}", flush=True)
    print("=" * 80, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
