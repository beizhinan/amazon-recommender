"""TAB-H：门控加权融合（Gated Stacking）——新算法最终版。

设计依据（Books 5-core 实测结论）：
    - bias 的 RMSE 最好（0.9023）但 MAE 不如 ItemCF（0.6531），二者误差结构互补：
      ItemCF 在有邻居时提供协同修正，在无邻居时（63.4%）只能回退到 bias；
    - FAR（特征增强 GBDT）在全量数据上比同子集 bias 低 0.3%~0.4% RMSE，
      但其优势集中在低频/冷门样本，与 CF 的覆盖区域基本互补；
    - 因此按「是否有协同邻居」门控、按「用户活跃度」分段，用验证集学习
      非负权重，可在不引入过拟合的前提下取得 RMSE/MAE 同时改善。

模型：
    r̂ = clip( w0 + w1·bias + w2·cf + w3·svd + w4·far )
    其中 cf 仅在存在邻居时进入（否则该行退化为 bias + svd + far），
    权重 w 由验证集上的非负最小二乘（NNLS）分「活跃度分段 × 是否有邻居」求解。

用法:
    python -m src.models.rating_prediction.run_stack                    # 全量
    python -m src.models.rating_prediction.run_stack --sample-users 3000
    python -m src.models.rating_prediction.run_stack --no-far           # 三路对比
    python -m src.models.rating_prediction.run_stack --refresh          # 忽略缓存
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor

from src.evaluation.rating_metrics import evaluate

from .baselines import BiasBaseline
from .common import LOG_DIR, load_rating_data, log
from .far_model import build_features
from .knn import ItemCF
from .run_improved import make_view, save_row
from .svd_surprise import SurpriseSVD

FAR_GRIDS = {
    # g0: 浅树快版；g1: 深树慢版（全量实测 RMSE 略优）
    "g0": dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
               min_samples_leaf=100, l2_regularization=1.0, max_bins=255),
    "g1": dict(max_iter=600, learning_rate=0.03, max_leaf_nodes=127,
               min_samples_leaf=200, l2_regularization=2.0, max_bins=255),
}

# 用户活跃度分段边界（训练评分数）：[0,5] (5,15] (15,50] >50
EDGES = [5, 15, 50]


def solve_nnls(designs: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """对给定设计矩阵列做非负最小二乘。"""
    A = np.column_stack(designs)
    w, _ = nnls(A, y)
    return w


def apply_weights(preds: dict[str, np.ndarray], w: np.ndarray,
                  comps: list[str]) -> np.ndarray:
    """用已学权重把各路预测线性组合（第 0 列为常数项）。"""
    n = len(preds["bias"])
    X = np.column_stack([np.ones(n)] + [preds[c] for c in comps])
    return X @ w


def design_cols(preds: dict[str, np.ndarray], m: np.ndarray,
                comps: list[str]) -> list[np.ndarray]:
    """取出掩码 m 对应的设计矩阵列（常数项 + 各分支预测）。"""
    n = int(m.sum())
    return [np.ones(n)] + [preds[c][m] for c in comps]


def main() -> int:
    ap = argparse.ArgumentParser(description="TAB-H 门控加权融合")
    ap.add_argument("--sample-users", type=int, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-svd", action="store_true", help="不参与 SVD 分支（更快）")
    ap.add_argument("--no-far", action="store_true", help="不参与 FAR（GBDT）分支")
    ap.add_argument("--far-params", choices=["g0", "g1"], default="g1",
                    help="FAR 的 GBDT 配置（g1 更慢但更准）")
    ap.add_argument("--refresh", action="store_true", help="忽略预测缓存，重新训练各分支")
    ap.add_argument("--k", type=int, default=50, help="ItemCF 邻居数")
    args = ap.parse_args()

    data = load_rating_data(sample_users=args.sample_users, seed=args.seed,
                            with_time=True)
    n_tr = len(data.r_tr)
    rng = np.random.default_rng(args.seed)
    is_val = rng.random(n_tr) < args.val_ratio
    fit_mask = ~is_val
    log(f"训练集拆分: fit={int(fit_mask.sum()):,d} val={int(is_val.sum()):,d}")

    fit_view = make_view(data, fit_mask, data.u_tr[fit_mask], data.i_tr[fit_mask],
                         data.r_tr[fit_mask], data.t_tr[fit_mask])
    val_view = make_view(data, fit_mask, data.u_tr[is_val], data.i_tr[is_val],
                         data.r_tr[is_val], data.t_tr[is_val])
    test_view = make_view(data, fit_mask, data.u_te, data.i_te, data.y_te, data.t_te)

    # 分支预测缓存：按「样本量 / k / 参与分支」区分，避免不同配置互相污染
    sig = (f"s{args.sample_users or 0}_k{args.k}"
           f"_{'nosvd' if args.no_svd else 'svd'}"
           f"_{'nofar' if args.no_far else args.far_params}")
    cache_path = LOG_DIR / f"stack_preds_{sig}.npz"

    if cache_path.exists() and not args.refresh:
        z = np.load(cache_path, allow_pickle=False)
        comps = [str(c) for c in z["comps"]]
        preds = {c: {"val": z[f"{c}_val"], "test": z[f"{c}_test"]} for c in comps}
        has_cf = {"val": z["has_cf_val"].astype(bool),
                  "test": z["has_cf_test"].astype(bool)}
        log(f"[cache] 复用分支预测 {cache_path.name} comps={comps}")

    else:
        # ---- 1) 偏置基线（同时提供 ItemCF 所需的 b_u / b_i）----
        t0 = time.time()
        bias = BiasBaseline(reg_u=3.0, reg_i=3.0).fit(fit_view)
        log(f"[bias] 完成 {time.time() - t0:.1f}s")

        # ---- 2) ItemCF ----
        t0 = time.time()
        cf = ItemCF(k=args.k, shrink=25.0).fit(fit_view, bias.mu, bias.b_u, bias.b_i)
        log(f"[ItemCF] 完成 {time.time() - t0:.1f}s")

        # ---- 3) SVD ----
        svd = None
        if not args.no_svd:
            t0 = time.time()
            svd = SurpriseSVD(n_factors=50, n_epochs=30, reg_all=0.1).fit(fit_view)
            log(f"[SVD] 完成 {time.time() - t0:.1f}s")

        # ---- 4) FAR（特征增强 GBDT，训练行特征用留一法防泄漏）----
        far = None
        if not args.no_far:
            t0 = time.time()
            params = FAR_GRIDS[args.far_params]
            far = HistGradientBoostingRegressor(
                loss="squared_error", early_stopping=True, validation_fraction=0.05,
                n_iter_no_change=30, random_state=args.seed, verbose=0, **params)
            feats_tr = build_features(
                data, fit_mask, data.u_tr[fit_mask], data.i_tr[fit_mask],
                data.t_tr[fit_mask], r=data.r_tr[fit_mask].astype(np.float64))
            far.fit(feats_tr, data.r_tr[fit_mask].astype(np.float64))
            del feats_tr
            log(f"[FAR-{args.far_params}] 完成 n_iter={far.n_iter_} "
                f"{time.time() - t0:.1f}s")

        bysplit: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}
        has_cf = {}
        for name, view in (("val", val_view), ("test", test_view)):
            t0 = time.time()
            bysplit[name]["bias"] = bias.predict(view)
            p_cf, fallback = cf.predict(view)
            bysplit[name]["cf"] = p_cf
            has_cf[name] = ~fallback
            if svd is not None:
                bysplit[name]["svd"] = svd.predict(view)[0]
            if far is not None:
                f = build_features(data, fit_mask, view.u_te, view.i_te, view.t_te)
                bysplit[name]["far"] = np.clip(far.predict(f), 1.0, 5.0)
            log(f"[{name}] n={len(view.y_te):,d} 协同覆盖率={has_cf[name].mean():.2%} "
                f"({time.time() - t0:.1f}s)")

        comps = [c for c in ("bias", "cf", "svd", "far") if c in bysplit["val"]]
        preds = {c: {name: bysplit[name][c] for name in ("val", "test")} for c in comps}

        np.savez(cache_path, comps=np.array(comps),
                 has_cf_val=has_cf["val"], has_cf_test=has_cf["test"],
                 **{f"{c}_{name}": preds[c][name] for c in comps
                    for name in ("val", "test")})
        log(f"[cache] 已写入 {cache_path}")

    y_val = val_view.y_te.astype(np.float64)

    # ---- 5) 单模型分路得分（用于诊断互补性）----
    for name, view in (("val", val_view), ("test", test_view)):
        for c in comps:
            if c == "cf":
                m = has_cf[name]
            else:
                m = np.ones(len(view.y_te), dtype=bool)
            mm = evaluate(view.y_te[m], preds[c][name][m])
            log(f"[单模型 {name}] {c:4s} 覆盖={m.mean():6.2%} "
                f"RMSE={mm['rmse']:.4f} MAE={mm['mae']:.4f}")

    # ---- 6) 分段学习非负权重（验证集）----
    cnt_u = np.diff(fit_view.R.indptr).astype(np.float64)

    def bucket(u_idx: np.ndarray) -> np.ndarray:
        return np.digitize(cnt_u[u_idx], EDGES, right=True)

    b_val = bucket(val_view.u_te)
    b_test = bucket(test_view.u_te)

    rest = [c for c in comps if c not in ("bias", "cf")]

    def comps_for(flag: bool) -> list[str]:
        """有协同邻居时纳入 cf；无邻居时只用全局分支。"""
        return (["bias", "cf"] + rest) if flag else (["bias"] + rest)

    def learn(mask: np.ndarray, flag: bool) -> tuple[np.ndarray, list[str]]:
        cs = comps_for(flag)
        views = {c: preds[c]["val"] for c in cs}
        return solve_nnls(design_cols(views, mask, cs), y_val[mask]), cs

    glob: dict[bool, tuple[np.ndarray, list[str]]] = {}
    for flag in (True, False):
        m = has_cf["val"] == flag
        if m.sum() < 5000:
            continue
        glob[flag] = learn(m, flag)
        w, cs = glob[flag]
        log(f"[全局权重 has_cf={flag}] n={int(m.sum()):,d} comps={cs} "
            f"w={np.round(w, 4)}")

    seg_w: dict[tuple[bool, int], tuple[np.ndarray, list[str]]] = {}
    for flag in (True, False):
        if flag not in glob:
            continue
        for bidx in np.unique(b_val):
            m = (b_val == bidx) & (has_cf["val"] == flag)
            if m.sum() < 2000:
                continue
            seg_w[(flag, int(bidx))] = learn(m, flag)
    log(f"[分段权重] 学到 {len(seg_w)} 组（has_cf × 活跃度分段）")

    # ---- 7) 融合与评估 ----
    def blend(name: str) -> np.ndarray:
        bkt = b_val if name == "val" else b_test
        out = np.empty(len(has_cf[name]), dtype=np.float64)
        for flag in (True, False):
            m_flag = has_cf[name] == flag
            if not m_flag.any():
                continue
            for bidx in np.unique(bkt[m_flag]):
                m = m_flag & (bkt == bidx)
                item = seg_w.get((flag, int(bidx))) or glob.get(flag)
                if item is None:
                    out[m] = preds["bias"][name][m]
                    continue
                w, cs = item
                sub = {c: preds[c][name][m] for c in cs}
                out[m] = apply_weights(sub, w, cs)
        return np.clip(out, 1.0, 5.0)

    p_val, p_test = blend("val"), blend("test")
    m_val = evaluate(val_view.y_te, p_val)
    m_test = evaluate(test_view.y_te, p_test)
    b_ref_val = evaluate(val_view.y_te, preds["bias"]["val"])
    b_ref_test = evaluate(test_view.y_te, preds["bias"]["test"])

    save_row("TAB-H_gated_stack",
             {"k": args.k, "comps": comps,
              "far_params": None if args.no_far else args.far_params,
              "segments": len(seg_w)},
             m_test, 0.0,
             note=f"val_rmse={m_val['rmse']:.4f} val_mae={m_val['mae']:.4f}")

    print("\n" + "=" * 76, flush=True)
    print(f"{'模型':<26}{'val_RMSE':>10}{'val_MAE':>10}{'test_RMSE':>11}{'test_MAE':>10}")
    print("-" * 76, flush=True)
    print(f"{'bias(同子集)':<26}{b_ref_val['rmse']:>10.4f}{b_ref_val['mae']:>10.4f}"
          f"{b_ref_test['rmse']:>11.4f}{b_ref_test['mae']:>10.4f}", flush=True)
    for c in comps:
        if c == "bias":
            continue
        cv = evaluate(val_view.y_te, preds[c]["val"])
        ct = evaluate(test_view.y_te, preds[c]["test"])
        print(f"{c.upper():<26}{cv['rmse']:>10.4f}{cv['mae']:>10.4f}"
              f"{ct['rmse']:>11.4f}{ct['mae']:>10.4f}", flush=True)
    print("-" * 76, flush=True)
    print(f"{'TAB-H 门控融合':<26}{m_val['rmse']:>10.4f}{m_val['mae']:>10.4f}"
          f"{m_test['rmse']:>11.4f}{m_test['mae']:>10.4f}", flush=True)
    print("=" * 76, flush=True)
    print(f"相对同子集 bias: RMSE {(b_ref_test['rmse'] - m_test['rmse']) / b_ref_test['rmse']:+.2%}"
          f"  MAE {(b_ref_test['mae'] - m_test['mae']) / b_ref_test['mae']:+.2%}", flush=True)
    print("对照：历史最优 bias(全量训练) test RMSE=0.9023 MAE=0.6574", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
