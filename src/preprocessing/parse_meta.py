"""解析 Amazon 商品元数据（meta_Books.json.gz），提取共购/共视图。

meta 文件每行是一个 Python dict 字面量（单引号），字段示例：
    {'asin': ..., 'related': {'also_bought': [...], 'also_viewed': [...]},
     'categories': [[...]], 'title': ..., 'salesRank': {...}}

本脚本做三件事：
    1) 统计元数据与训练/测试商品的 ASIN 交集覆盖；
    2) 构建 also_bought / also_viewed 两路 item-item 邻接矩阵（对称化）；
    3) 落盘 data/processed/Books/meta_graph.npz，供 run_topn.py 作为第三路
       协同信号（补足共现矩阵覆盖不到的冷门商品）。

用法:
    python src/preprocessing/parse_meta.py
    python src/preprocessing/parse_meta.py --limit 100000   # 仅解析前 N 行做冒烟
"""
from __future__ import annotations

import argparse
import ast
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.config import PROJECT_ROOT, load_config

RAW_META = PROJECT_ROOT / "data" / "raw" / "meta_Books.json.gz"
OUT_NPZ = PROJECT_ROOT / "data" / "processed" / "Books" / "meta_graph.npz"

RELATED_BUY_KEYS = ("also_bought", "also_buy")
RELATED_VIEW_KEYS = ("also_viewed", "also_view", "buy_after_viewing")


def parse_line(line: str) -> dict | None:
    """安全解析一行 Python dict 字面量，失败返回 None。"""
    try:
        return ast.literal_eval(line)
    except (ValueError, SyntaxError, MemoryError):
        return None


def related_lists(d: dict) -> tuple[list, list]:
    """提取 also_bought / also_viewed 列表，兼容顶层 also_buy/also_view。"""
    rel = d.get("related") or {}
    if not isinstance(rel, dict):
        rel = {}
    buy = []
    for k in RELATED_BUY_KEYS:
        v = rel.get(k) if k in rel else d.get(k)
        if isinstance(v, list):
            buy = v
            break
    view = []
    for k in RELATED_VIEW_KEYS:
        v = rel.get(k) if k in rel else d.get(k)
        if isinstance(v, list):
            view = v
            break
    return buy, view


def load_items(cfg: dict) -> tuple[np.ndarray, pd.Index]:
    """加载训练+测试的唯一 item_id（与 common.load_rating_data 相同的排序规则）。"""
    train = pd.read_parquet(cfg["paths"]["train_parquet"], columns=["item_id"])
    test = pd.read_parquet(cfg["paths"]["test_parquet"], columns=["item_id"])
    items = np.sort(pd.unique(pd.concat([train["item_id"], test["item_id"]],
                                        ignore_index=True)))
    return items, pd.Index(items)


def main() -> int:
    ap = argparse.ArgumentParser(description="解析元数据并构建共购图")
    ap.add_argument("--limit", type=int, default=None, help="仅解析前 N 行")
    ap.add_argument("--buy-weight", type=float, default=1.0)
    ap.add_argument("--view-weight", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=OUT_NPZ)
    args = ap.parse_args()

    cfg = load_config()
    items, i_index = load_items(cfg)
    n_items = len(items)
    print(f"[meta] 训练+测试唯一商品数: {n_items:,d}", flush=True)

    total = 0
    n_ok = 0
    n_buy = 0
    n_view = 0
    n_in_items = 0
    # 用列表累积 COO，最后一次性构建 CSR，避免逐行扩列
    buy_rows, buy_cols = [], []
    view_rows, view_cols = [], []
    buy_edges = view_edges = 0
    buy_edges_kept = view_edges_kept = 0

    print(f"[meta] 解析 {RAW_META.name} ...", flush=True)
    with gzip.open(RAW_META, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            if args.limit and total > args.limit:
                break
            d = parse_line(line)
            if d is None or not isinstance(d, dict):
                continue
            n_ok += 1
            asin = d.get("asin")
            if not isinstance(asin, str):
                continue
            if asin in i_index:
                n_in_items += 1
            buy, view = related_lists(d)
            if buy:
                n_buy += 1
            if view:
                n_view += 1
            # 只保留两端都在 items 内的边
            si = i_index.get_loc(asin) if asin in i_index else None
            if si is None:
                continue
            for j in buy:
                if isinstance(j, str) and j in i_index:
                    sj = i_index.get_loc(j)
                    buy_rows.append(si)
                    buy_cols.append(sj)
                    buy_edges_kept += 1
                buy_edges += 1
            for j in view:
                if isinstance(j, str) and j in i_index:
                    sj = i_index.get_loc(j)
                    view_rows.append(si)
                    view_cols.append(sj)
                    view_edges_kept += 1
                view_edges += 1

    print(f"[meta] 总记录数: {total:,d}  成功解析: {n_ok:,d}", flush=True)
    print(f"[meta] 含 also_bought: {n_buy:,d}  含 also_viewed: {n_view:,d}",
          flush=True)
    print(f"[meta] ASIN 命中训练/测试商品的记录: {n_in_items:,d} "
          f"({n_in_items / max(n_ok, 1):.1%})", flush=True)
    print(f"[meta] also_bought 原始边: {buy_edges:,d} 保留: {buy_edges_kept:,d} "
          f"({buy_edges_kept / max(buy_edges, 1):.1%})", flush=True)
    print(f"[meta] also_viewed 原始边: {view_edges:,d} 保留: {view_edges_kept:,d} "
          f"({view_edges_kept / max(view_edges, 1):.1%})", flush=True)

    def to_sym_csr(rows, cols, w, name):
        if not rows:
            print(f"[meta] {name} 无边，返回空矩阵", flush=True)
            return sp.csr_matrix((n_items, n_items), dtype=np.float32)
        m = sp.coo_matrix(
            (np.full(len(rows), w, dtype=np.float32), (rows, cols)),
            shape=(n_items, n_items)).tocsr()
        m = (m + m.T).tocsr()          # 对称化：also_buy 为有向关系，转成无向共购
        m.setdiag(0.0)
        m.eliminate_zeros()
        m.sum_duplicates()
        print(f"[meta] {name} 对称化后 nnz={m.nnz:,d}", flush=True)
        return m

    A_buy = to_sym_csr(buy_rows, buy_cols, args.buy_weight, "also_buy")
    A_view = to_sym_csr(view_rows, view_cols, args.view_weight, "also_view")

    # 覆盖的商品数（图里度数 > 0 的商品）
    deg_buy = np.asarray(A_buy.sum(axis=1)).ravel()
    deg_view = np.asarray(A_view.sum(axis=1)).ravel()
    covered = int((deg_buy > 0).sum() + (deg_view > 0).sum() -
                  ((deg_buy > 0) & (deg_view > 0)).sum())
    print(f"[meta] 共购图覆盖商品数: {covered:,d} "
          f"({covered / n_items:.1%})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        buy_data=A_buy.data, buy_indices=A_buy.indices, buy_indptr=A_buy.indptr,
        view_data=A_view.data, view_indices=A_view.indices, view_indptr=A_view.indptr,
        shape=np.asarray([n_items, n_items], dtype=np.int64),
        items=items,
    )
    print(f"[meta] 已落盘: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
