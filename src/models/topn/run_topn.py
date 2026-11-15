"""第三步：Top-N 推荐实验入口。

协议：
    - 训练集按 9:1 切成 fit / val（val 仅用于调参，不进入任何模型）；
    - 在 val 用户上做小网格选择（ItemKNN 的 k、混合排序的 λ）；
    - 最终在 test 用户上做全量商品排序（排除用户已交互的训练商品），取 top-10；
    - 指标：Precision@10 / Recall@10 / F1@10 / NDCG@10 / HitRate@10 / 覆盖率。

用法:
    python -m src.models.topn.run_topn                                  # 全量
    python -m src.models.topn.run_topn --users 2000 --tune-users 1000 --quick
    python -m src.models.topn.run_topn --skip-tuning                    # 跳过调参
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.rating_prediction.baselines import BiasBaseline
from src.models.rating_prediction.common import load_rating_data, log
from src.models.rating_prediction.run_improved import make_view
from src.utils.config import PROJECT_ROOT, ensure_dir

from .models import (
    BPR, Hybrid, ItemKNN, MetaItemKNN, Popularity, RatingRanker,
    SegmentedHybrid, TripleHybrid,
)

TOPN_DIR = PROJECT_ROOT / "experiments" / "logs" / "topn"
RESULT_CSV = TOPN_DIR / "results_topn.csv"
RESULT_COLUMNS = [
    "model", "config", "precision", "recall", "f1", "ndcg", "hit_rate",
    "coverage", "users", "seconds", "run_time", "note",
]


def save_row(model: str, config: dict, metrics: dict, seconds: float,
             note: str = "") -> None:
    ensure_dir(TOPN_DIR)
    payload = {
        "model": model,
        "config": json.dumps(config, ensure_ascii=False, sort_keys=True),
        **{k: metrics.get(k) for k in
           ("precision", "recall", "f1", "ndcg", "hit_rate", "coverage", "users")},
        "seconds": round(seconds, 1),
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
    }
    new_file = not RESULT_CSV.exists()
    with open(RESULT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(payload)
    log(f"[result] {model} P@10={payload['precision']} R@10={payload['recall']} "
        f"NDCG@10={payload['ndcg']} ({payload['seconds']}s)")


META_NPZ = PROJECT_ROOT / "data" / "processed" / "Books" / "meta_graph.npz"


def load_meta_graph(path: Path = META_NPZ, buy_weight: float = 1.0,
                    view_weight: float = 0.5) -> sp.csr_matrix | None:
    """加载元数据共购图，合并 also_buy / also_view 为单个 item-item 邻接矩阵。"""
    if not Path(path).exists():
        log(f"[meta] 未找到共购图 {path}，跳过元数据分支")
        return None
    z = np.load(path)
    shape = tuple(int(x) for x in z["shape"].tolist())
    A_buy = sp.csr_matrix(
        (z["buy_data"], z["buy_indices"], z["buy_indptr"]), shape=shape)
    A_view = sp.csr_matrix(
        (z["view_data"], z["view_indices"], z["view_indptr"]), shape=shape)
    A = (buy_weight * A_buy + view_weight * A_view).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    log(f"[meta] 共购图加载完成 nnz={A.nnz:,d} "
        f"(buy={A_buy.nnz:,d} view={A_view.nnz:,d})")
    return A


def build_matrix(users: np.ndarray, items: np.ndarray,
                 shape: tuple[int, int]) -> sp.csr_matrix:
    mat = sp.csr_matrix(
        (np.ones(len(users), dtype=np.float32), (users, items)), shape=shape)
    mat.sum_duplicates()
    mat.sort_indices()
    return mat


def build_weighted(u: np.ndarray, i: np.ndarray, t: np.ndarray,
                   shape: tuple[int, int], half_life: float,
                   scheme: str = "none") -> sp.csr_matrix:
    """构造加权交互矩阵。

    half_life > 0：按「距用户最近活跃时间」做指数衰减（单位：年）。
    scheme="bm25"：再按用户活跃度折算，w_ui /= log2(1+n_u)，
        避免重度用户主导共现统计（BM25 的文档长度归一化思想）。
    """
    if half_life <= 0:
        mat = build_matrix(u, i, shape)
    else:
        last = np.zeros(shape[0], dtype=np.float64)
        np.maximum.at(last, u, t.astype(np.float64))
        w = np.power(0.5, (last[u] - t.astype(np.float64)) / half_life).astype(np.float32)
        mat = sp.csr_matrix((w, (u, i)), shape=shape)
        mat.sum_duplicates()
        mat.sort_indices()

    if scheme == "bm25":
        cnt = np.diff(mat.indptr).astype(np.float32)
        mult = (1.0 / np.log2(1.0 + cnt)).astype(np.float32)
        mat.data = (mat.data * np.repeat(mult, np.diff(mat.indptr))).astype(np.float32)
    return mat


def recommend(scorer, users: np.ndarray, exclude: sp.csr_matrix, hist: sp.csr_matrix,
              k: int, chunk: int = 1000, tag: str = "") -> list[np.ndarray]:
    """对每个用户打分、排除已交互商品、返回 top-K 商品索引。"""
    recs: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(users), chunk):
        us = users[start:start + chunk]
        scores = scorer.score_chunk(us, hist)
        for row, u in enumerate(us):
            seen = exclude.indices[exclude.indptr[u]:exclude.indptr[u + 1]]
            s = scores[row]
            if len(seen):
                s[seen] = -np.inf
            idx = np.argpartition(-s, k)[:k]
            recs.append(idx[np.argsort(-s[idx])])
        if tag and (start // chunk) % 5 == 0:
            log(f"[{tag}] 推荐 {len(recs):,}/{len(users):,} ({time.time() - t0:.0f}s)")
    return recs


def ground_truth(pos: sp.csr_matrix, users: np.ndarray) -> list[set[int]]:
    return [set(pos.indices[pos.indptr[u]:pos.indptr[u + 1]].tolist()) for u in users]


def run_eval(scorer, users: np.ndarray, exclude: sp.csr_matrix,
             pos: sp.csr_matrix, hist: sp.csr_matrix, k: int, n_items: int,
             chunk: int = 1000, tag: str = "") -> dict:
    recs = recommend(scorer, users, exclude, hist, k, chunk=chunk, tag=tag)
    return evaluate_ranking(recs, ground_truth(pos, users), k=k, n_items=n_items)


def main() -> int:
    ap = argparse.ArgumentParser(description="Top-N 推荐实验")
    ap.add_argument("--users", type=int, default=10000, help="测试评估用户数")
    ap.add_argument("--tune-users", type=int, default=3000, help="调参用户数")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=10, help="推荐列表长度")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ks", default="200,400,800", help="ItemKNN 邻居数网格")
    ap.add_argument("--alphas", default="0.5,0.7,0.85",
                    help="非对称余弦的 alpha 网格")
    ap.add_argument("--half-lifes", default="2,0",
                    help="时间衰减半衰期网格（年），<=0 表示不衰减")
    ap.add_argument("--weight-schemes", default="none,bm25",
                    help="交互权重方案：none / bm25（按用户活跃度折算）")
    ap.add_argument("--no-bpr", action="store_true", help="跳过 BPR 及其融合（省时）")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-tuning", action="store_true")
    ap.add_argument("--sample-users", type=int, default=None,
                    help="仅加载部分用户（smoke test 用）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 5)

    data = load_rating_data(sample_users=args.sample_users, seed=args.seed,
                            with_time=True)
    n_users, n_items = data.n_users, data.n_items
    log(f"[data] users={n_users:,d} items={n_items:,d} train={len(data.r_tr):,d} "
        f"test={len(data.y_te):,d}")

    # ---- 1) fit / val 切分（val 只用于调参）----
    rng = np.random.default_rng(args.seed)
    n_tr = len(data.r_tr)
    is_val = rng.random(n_tr) < args.val_ratio
    fit_mask = ~is_val
    shape = (n_users, n_items)
    B_hist = build_matrix(data.u_tr[fit_mask], data.i_tr[fit_mask], shape)
    B_val = build_matrix(data.u_tr[is_val], data.i_tr[is_val], shape)
    B_train = data.B
    B_test = build_matrix(data.u_te, data.i_te, shape)
    log(f"切分: fit={int(fit_mask.sum()):,d} val={int(is_val.sum()):,d}")

    # 分段桶：用户活跃度 4 档 × 商品流行度 2 档（Top 25% 为头部）
    cnt_u = np.asarray(B_hist.sum(axis=1)).ravel()
    u_bucket = np.digitize(cnt_u, [5, 15, 50], right=True).astype(np.int64)
    item_cnt = np.asarray(B_hist.sum(axis=0)).ravel()
    i_bucket = (item_cnt > np.quantile(item_cnt, 0.75)).astype(np.int64)

    # ---- 2) 评分排序分支（μ + b_u + b_i，仅用 fit 数据）----
    fit_view = make_view(data, fit_mask, data.u_tr[fit_mask], data.i_tr[fit_mask],
                         data.r_tr[fit_mask],
                         data.t_tr[fit_mask] if data.t_tr is not None
                         else np.zeros(int(fit_mask.sum()), dtype=np.float32))
    bias = BiasBaseline(reg_u=3.0, reg_i=3.0).fit(fit_view)
    rating = RatingRanker(bias.mu, bias.b_u, bias.b_i)

    # ---- 3) 训练各模型 ----
    pop = Popularity().fit(B_hist)
    half_lifes = [float(x) for x in args.half_lifes.split(",")]
    schemes = [s.strip() for s in args.weight_schemes.split(",") if s.strip()]
    W_hist = build_weighted(data.u_tr[fit_mask], data.i_tr[fit_mask],
                            data.t_tr[fit_mask], shape, half_lifes[0], schemes[0])
    knn_src = ItemKNN(shrink=10.0).fit(W_hist)
    bpr = None
    if not args.no_bpr:
        bpr = BPR(n_factors=args.factors, epochs=args.epochs, seed=args.seed)
        bpr.fit(data.u_tr[fit_mask], data.i_tr[fit_mask], n_users, n_items)

    # ---- 4) 调参（val 用户）----
    all_users = np.arange(n_users)
    tune_users = rng.choice(all_users, size=min(args.tune_users, n_users), replace=False)
    tune_users = np.array([u for u in tune_users if B_val.indptr[u] < B_val.indptr[u + 1]])

    alphas = [0.5] if args.quick else [float(a) for a in args.alphas.split(",")]
    k_grid = [200] if args.quick else sorted({int(x) for x in args.ks.split(",")})
    lam_grid = [0.15, 0.25, 0.5, 1.0] if not args.quick else [0.25]

    best_k, best_alpha, best_k_ndcg = k_grid[0], alphas[0], -1.0
    best_hl, best_scheme = half_lifes[0], schemes[0]
    knn_best = None
    if args.skip_tuning:
        knn_best = knn_src.build(0.5, k_grid[0])
    else:
        # 阶段 1：当前 (half_life, 权重方案) 下扫描 alpha × k（共现矩阵只算一次）
        log(f"[tune] 阶段1 alpha ∈ {alphas}, k ∈ {k_grid}，"
            f"用户数={len(tune_users):,d}")
        for alpha in alphas:
            full = knn_src.build(alpha, max(k_grid))
            for k in k_grid:
                scorer = full if k >= full.k else full.truncate(k)
                m = run_eval(scorer, tune_users, B_hist, B_val, B_hist, args.k,
                             n_items, tag=f"tune-a{alpha}-k{k}")
                log(f"[tune] ItemKNN alpha={alpha} k={k} NDCG@10={m['ndcg']:.4f} "
                    f"R@10={m['recall']:.4f} F1@10={m['f1']:.4f}")
                if m["ndcg"] > best_k_ndcg:
                    best_k_ndcg, best_k, best_alpha, knn_best = (
                        m["ndcg"], k, alpha, scorer)
        log(f"[tune] 阶段1 最优 alpha={best_alpha} k={best_k} "
            f"NDCG@10={best_k_ndcg:.4f}")

        # 阶段 2：(half_life, 权重方案) 坐标搜索，固定阶段 1 选出的 alpha
        for hl in half_lifes:
            for scheme in schemes:
                if hl == half_lifes[0] and scheme == schemes[0]:
                    continue
                log(f"[tune] 阶段2 重建共现: half_life={hl} scheme={scheme}")
                src = ItemKNN(shrink=10.0).fit(
                    build_weighted(data.u_tr[fit_mask], data.i_tr[fit_mask],
                                   data.t_tr[fit_mask], shape, hl, scheme))
                full = src.build(best_alpha, max(best_k, max(k_grid)))
                for k in sorted({best_k, max(k_grid)}):
                    scorer = full if k >= full.k else full.truncate(k)
                    m = run_eval(scorer, tune_users, B_hist, B_val, B_hist, args.k,
                                 n_items, tag=f"tune-hl{hl}-{scheme}-k{k}")
                    log(f"[tune] ItemKNN half_life={hl} scheme={scheme} k={k} "
                        f"NDCG@10={m['ndcg']:.4f}")
                    if m["ndcg"] > best_k_ndcg:
                        best_k_ndcg, best_k = m["ndcg"], k
                        best_hl, best_scheme, knn_best = hl, scheme, scorer

    best_lam, best_lam_ndcg = 0.25, -1.0
    hybrid = None
    lam_tab = np.full((4, 2), best_lam, dtype=np.float32)
    if bpr is None:
        pass
    elif args.skip_tuning:
        hybrid = Hybrid(bpr, knn_best, lam=best_lam)
        lam_tab = np.full((4, 2), best_lam, dtype=np.float32)
    else:
        log(f"[tune] 全局 λ ∈ {lam_grid}，用户数={len(tune_users):,d}")
        for lam in lam_grid:
            cand = Hybrid(bpr, knn_best, lam=lam)
            m = run_eval(cand, tune_users, B_hist, B_val, B_hist, args.k, n_items,
                         tag=f"tune-lam{lam}")
            log(f"[tune] Hybrid λ={lam} NDCG@10={m['ndcg']:.4f} "
                f"R@10={m['recall']:.4f}")
            if m["ndcg"] > best_lam_ndcg:
                best_lam_ndcg, best_lam = m["ndcg"], lam
        hybrid = Hybrid(bpr, knn_best, lam=best_lam)

        # 分段 λ：用户活跃度 4 桶 × 商品流行度 2 桶，坐标搜索
        lam_tab = np.full((4, 2), best_lam, dtype=np.float32)
        best_seg_ndcg = -1.0
        for u in range(4):
            for it in range(2):
                local_best, local_lam = -1.0, lam_tab[u, it]
                for cand in lam_grid:
                    trial = lam_tab.copy()
                    trial[u, it] = cand
                    m = run_eval(SegmentedHybrid(bpr, knn_best, trial, u_bucket,
                                                 i_bucket),
                                 tune_users, B_hist, B_val, B_hist, args.k,
                                 n_items)
                    if m["ndcg"] > local_best:
                        local_best, local_lam = m["ndcg"], cand
                lam_tab[u, it] = local_lam
                log(f"[tune] 段(u={u}, i={it}) λ={local_lam} NDCG@10={local_best:.4f}")
                best_seg_ndcg = max(best_seg_ndcg, local_best)
        log(f"[tune] 分段 λ 表:\n{np.array2string(lam_tab, precision=2)}")

    log(f"[tune] 选择: ItemKNN alpha={best_alpha} k={best_k} "
        f"half_life={best_hl} scheme={best_scheme}, 全局 λ={best_lam}")

    # ---- 4.5) 元数据共购图：独立评测 + 三路融合（补足共现覆盖率）----
    meta_knn = None
    best_lam2 = best_lam
    A_meta = load_meta_graph()
    if A_meta is not None and A_meta.shape[0] == n_items:
        meta_knn = MetaItemKNN(k=best_k, alpha=best_alpha).fit(A_meta)
        if bpr is not None:
            best_tri_ndcg = -1.0
            for lam2 in [0.25, 0.5, 1.0]:
                cand = TripleHybrid(bpr, knn_best, meta_knn,
                                    lam1=best_lam, lam2=lam2)
                m = run_eval(cand, tune_users, B_hist, B_val, B_hist, args.k,
                             n_items, tag=f"tune-tri-{lam2}")
                log(f"[tune] TripleHybrid lam1={best_lam} lam2={lam2} "
                    f"NDCG@10={m['ndcg']:.4f}")
                if m["ndcg"] > best_tri_ndcg:
                    best_tri_ndcg, best_lam2 = m["ndcg"], lam2
            log(f"[tune] TripleHybrid 最优 lam2={best_lam2} "
                f"NDCG@10={best_tri_ndcg:.4f}")

    # ---- 5) test 评估（排除全部训练交互）----
    test_users = np.unique(rng.choice(all_users, size=min(args.users, n_users),
                                      replace=False))
    log(f"[test] 评估用户数={len(test_users):,d}，候选商品数={n_items:,d}")

    models = {
        "MostPop": (pop, {}),
        "RatingPred": (rating, {"reg_u": 3, "reg_i": 3}),
        f"ItemKNN_a{best_alpha}": (knn_best, {"k": best_k, "alpha": best_alpha,
                                              "half_life": best_hl,
                                              "scheme": best_scheme}),
    }
    if meta_knn is not None:
        models["MetaItemKNN"] = (meta_knn, {"k": best_k, "alpha": best_alpha,
                                            "buy_weight": 1.0,
                                            "view_weight": 0.5})
    if bpr is not None:
        hybrid_seg = SegmentedHybrid(bpr, knn_best, lam_tab, u_bucket, i_bucket)
        models["BPR"] = (bpr, {"factors": args.factors, "epochs": args.epochs})
        models["Hybrid_global"] = (hybrid, {"lam": best_lam, "k": best_k,
                                            "alpha": best_alpha,
                                            "half_life": best_hl,
                                            "scheme": best_scheme})
        models["Hybrid_seg"] = (hybrid_seg, {"lam_table": lam_tab.round(2).tolist(),
                                             "k": best_k, "alpha": best_alpha,
                                             "half_life": best_hl,
                                             "scheme": best_scheme})
        if meta_knn is not None:
            models["TripleHybrid"] = (
                TripleHybrid(bpr, knn_best, meta_knn,
                             lam1=best_lam, lam2=best_lam2),
                {"lam1": best_lam, "lam2": best_lam2, "k": best_k,
                 "alpha": best_alpha, "half_life": best_hl,
                 "scheme": best_scheme})

    rows = []
    for name, (scorer, cfg) in models.items():
        t0 = time.time()
        m = run_eval(scorer, test_users, B_train, B_test, B_hist, args.k, n_items,
                     tag=name)
        secs = time.time() - t0
        save_row(name, cfg, m, secs,
                 note=f"fit=90%train eval_users={len(test_users)}")
        rows.append((name, cfg, m, secs))

    rows.sort(key=lambda r: -r[2]["ndcg"])
    print("\n" + "=" * 92, flush=True)
    print(f"{'模型':<16}{'P@10':>9}{'R@10':>9}{'F1@10':>9}{'NDCG@10':>10}"
          f"{'HitRate':>9}{'覆盖率':>9}{'耗时(s)':>9}", flush=True)
    print("-" * 92, flush=True)
    for name, _, m, secs in rows:
        print(f"{name:<16}{m['precision']:>9.4f}{m['recall']:>9.4f}{m['f1']:>9.4f}"
              f"{m['ndcg']:>10.4f}{m['hit_rate']:>9.4f}{m['coverage']:>9.2%}"
              f"{secs:>9.1f}", flush=True)
    print("=" * 92, flush=True)
    base = next(r for r in rows if r[0] == "MostPop")[2]
    best = rows[0]
    print(f"最优 {best[0]}：NDCG@10 相对 MostPop 提升 "
          f"{(best[2]['ndcg'] - base['ndcg']) / base['ndcg']:+.1%}，"
          f"Recall@10 {(best[2]['recall'] - base['recall']) / base['recall']:+.1%}",
          flush=True)
    print(f"结果已写入 {RESULT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
