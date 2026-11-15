"""数据选择与预处理：解析 Amazon 5-core、清洗、去重、按用户 80/20 划分。

流程:
    1. 读取 data/raw/*.json.gz（每行一条用户-商品评分记录）
    2. 字段规范化并保留全部原始信息（helpful 拆为 helpful_no/helpful_yes，其余未知字段存入 extra）
    3. 清洗：丢弃缺失 user/item/rating 的记录，过滤非法评分
    4. 去重：同一 (user, item) 保留时间戳最新的一条，避免训练/测试泄漏
    5. 划分：每个用户随机 80% 训练 / 20% 测试（固定种子，可复现）
    6. 保存为 CSV 与 Parquet，并输出统计文件 stats.json

用法:
    python src/preprocessing/prepare_data.py
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

try:
    import pgzip  # 可选：并行 gzip 解压，比内置 gzip 快数倍
except ImportError:
    pgzip = None

tqdm.pandas()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"

# 原始字段 -> 规范字段
RENAME = {
    "reviewerID": "user_id",
    "asin": "item_id",
    "overall": "rating",
    "reviewText": "review_text",
    "summary": "summary",
    "unixReviewTime": "timestamp",
    "reviewTime": "review_time",
    "reviewerName": "reviewer_name",
}

# 最终输出列（顺序固定）
OUTPUT_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
    "review_text",
    "summary",
    "timestamp",
    "review_time",
    "reviewer_name",
    "helpful_no",
    "helpful_yes",
    "extra",
]


def load_config(path: Path | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 仅解析文件/目录类字段，避免误处理 URL
    for key, value in cfg["dataset"].items():
        if key.endswith("_file") or key.endswith("_dir"):
            cfg["dataset"][key] = str(PROJECT_ROOT / value)
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str(PROJECT_ROOT / value)
    return cfg


def _parse_json_bytes(buf: bytes) -> pd.DataFrame:
    """解析一段 JSON Lines 字节流为 DataFrame（供子进程调用）。"""
    return pd.read_json(io.BytesIO(buf), lines=True)


def read_raw(path: Path, workers: int | None = None) -> pd.DataFrame:
    """读取 jsonl.gz 原始数据，并行解压 + 多进程解析 JSON 行。"""
    print(f"[read] {path}")
    workers = workers or min(16, os.cpu_count() or 1)

    # 并行解压（pgzip 可用时）→ 一次性读入原始字节
    if pgzip is not None:
        with pgzip.open(str(path), "rb") as f:
            raw = f.read()
    else:
        with gzip.open(path, "rb") as f:
            raw = f.read()

    # 按换行切分为 workers 个块，块边界对齐到整行
    lines = raw.split(b"\n")
    del raw
    if lines and lines[-1] == b"":
        lines.pop()
    n = len(lines)
    if workers > 1 and n > workers:
        size = math.ceil(n / workers)
        chunks = [b"\n".join(lines[i:i + size]) for i in range(0, n, size)]
        del lines
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            parts = pool.map(_parse_json_bytes, chunks)
        del chunks
    else:
        parts = [_parse_json_bytes(b"\n".join(lines))]

    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"[read] 原始记录数: {len(df):,d}, 原始字段: {sorted(df.columns)}")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """字段重命名、helpful 拆分、未知字段归入 extra。"""
    df = df.rename(columns=RENAME)

    # helpful: [no, yes]（向量化拆分，比逐行 apply 快得多）
    if "helpful" in df.columns:
        h = df["helpful"]
        df["helpful_no"] = h.str[0]
        df["helpful_yes"] = h.str[1]
        df = df.drop(columns=["helpful"])

    # 补齐缺失的标准字段
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # 其余未知字段统一序列化为 JSON 字符串，保证信息不丢失
    known = set(OUTPUT_COLUMNS)
    extra_cols = [c for c in df.columns if c not in known]
    if extra_cols:
        def _pack(row):
            payload = {c: row[c] for c in extra_cols if pd.notna(row[c])}
            return json.dumps(payload, ensure_ascii=False, default=str) if payload else None

        df["extra"] = df[extra_cols].progress_apply(_pack, axis=1)
        df = df.drop(columns=extra_cols)

    # 类型规范
    df["user_id"] = df["user_id"].astype("string").str.strip()
    df["item_id"] = df["item_id"].astype("string").str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    df["helpful_no"] = pd.to_numeric(df["helpful_no"], errors="coerce").astype("Int64")
    df["helpful_yes"] = pd.to_numeric(df["helpful_yes"], errors="coerce").astype("Int64")
    for col in ("review_text", "summary", "review_time", "reviewer_name", "extra"):
        df[col] = df[col].fillna("").astype("string")

    # 文本净化：Amazon 原文含 NUL(\x00) 与孤立代理字符，
    # 会让 csv.writer 报 "need to escape, but no escapechar set" / UnicodeEncodeError
    df = scrub_text(df)

    return df[OUTPUT_COLUMNS]


def scrub_text(df: pd.DataFrame) -> pd.DataFrame:
    """移除会导致 CSV 写入失败的 NUL 字符与孤立代理字符。"""
    text_cols = ["review_text", "summary", "review_time", "reviewer_name", "extra"]
    n_nul = 0
    for col in text_cols:
        s = df[col]
        if s.isna().all():
            continue
        # NUL：直接删除（替换操作本身即检测，无需额外扫描）
        n_col_nul = int(s.str.count("\x00").sum())
        if n_col_nul:
            n_nul += n_col_nul
            s = s.str.replace("\x00", "", regex=False)
        # 孤立代理字符：用 errors="ignore" 编码丢弃（仅在检出时执行）
        if s.str.contains(r"[\ud800-\udfff]", regex=True).fillna(False).any():
            s = s.str.encode("utf-8", "ignore").str.decode("utf-8")
        df[col] = s
    if n_nul:
        print(f"[scrub] 移除 NUL 字符 {n_nul:,d} 个（否则 CSV 落盘会失败）")
    return df


def clean(df: pd.DataFrame, min_rating: int, max_rating: int) -> tuple[pd.DataFrame, dict]:
    """清洗与去重，返回 (清洗后数据, 移除统计)。"""
    n_raw = len(df)

    # 关键字段缺失 / 空串
    invalid = (
        df["user_id"].isna()
        | df["item_id"].isna()
        | df["rating"].isna()
        | (df["user_id"].str.len() == 0)
        | (df["item_id"].str.len() == 0)
    )
    # 评分越界
    invalid |= ~df["rating"].between(min_rating, max_rating)
    n_invalid = int(invalid.sum())
    df = df.loc[~invalid].copy()

    # 去重：同一 (user, item) 保留时间戳最新的一条（时间缺失按 0 处理）
    sort_key = "_ts_sort"
    df[sort_key] = df["timestamp"].fillna(0)
    df = df.sort_values(["user_id", "item_id", sort_key], ascending=[True, True, False])
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    n_dup = before_dedup - len(df)
    df = df.drop(columns=[sort_key]).reset_index(drop=True)

    removed = {
        "raw_records": n_raw,
        "invalid_removed": n_invalid,
        "duplicates_removed": int(n_dup),
        "clean_records": len(df),
    }
    return df, removed


def split_per_user(
    df: pd.DataFrame, test_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """按用户随机划分：每个用户 test_ratio 比例进入测试集。"""
    rng = np.random.default_rng(seed)

    # 为每条记录生成随机键，再在用户内排序，取前 n_test 条为测试
    df = df.copy()
    df["_rand"] = rng.random(len(df))
    df = df.sort_values(["user_id", "_rand"], kind="mergesort").reset_index(drop=True)

    n = df.groupby("user_id", sort=False)["user_id"].transform("size")
    n_test = np.round(test_ratio * n).astype(int)
    n_test = np.maximum(1, n_test)          # 至少 1 条测试
    n_test = np.minimum(n_test, n - 1)      # 至少保留 1 条训练
    n_test = np.maximum(n_test, 0)

    df["_rank"] = df.groupby("user_id", sort=False).cumcount()
    is_test = df["_rank"] < n_test

    train = df.loc[~is_test].drop(columns=["_rand", "_rank"]).reset_index(drop=True)
    test = df.loc[is_test].drop(columns=["_rand", "_rank"]).reset_index(drop=True)

    info = {
        "users": int(df["user_id"].nunique()),
        "train_records": len(train),
        "test_records": len(test),
        "actual_test_ratio": round(len(test) / len(df), 4) if len(df) else 0.0,
    }
    return train, test, info


def _subset_stats(df: pd.DataFrame) -> dict:
    users = df["user_id"].nunique()
    items = df["item_id"].nunique()
    n = len(df)
    sparsity = 1 - n / (users * items) if users and items else 0.0
    per_user = df.groupby("user_id").size()
    return {
        "records": int(n),
        "users": int(users),
        "items": int(items),
        "sparsity": round(float(sparsity), 6),
        "rating_mean": round(float(df["rating"].mean()), 4) if n else 0.0,
        "per_user_min": int(per_user.min()) if n else 0,
        "per_user_median": float(per_user.median()) if n else 0.0,
        "per_user_mean": round(float(per_user.mean()), 4) if n else 0.0,
        "per_user_max": int(per_user.max()) if n else 0,
    }


def compute_stats(
    df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
    removed: dict, cfg: dict, split_info: dict
) -> dict:
    rating_dist = df["rating"].value_counts().sort_index()
    ts = df["timestamp"].dropna()

    overlap = train.merge(test, on=["user_id", "item_id"], how="inner")
    train_users = set(train["user_id"])
    test_users = set(test["user_id"])

    return {
        "dataset": cfg["dataset"]["name"],
        "seed": cfg["preprocessing"]["seed"],
        "test_ratio": cfg["preprocessing"]["test_ratio"],
        "removed": removed,
        "all": _subset_stats(df),
        "train": _subset_stats(train),
        "test": _subset_stats(test),
        "rating_distribution": {str(k): int(v) for k, v in rating_dist.items()},
        "timestamp": {
            "min": int(ts.min()) if len(ts) else None,
            "max": int(ts.max()) if len(ts) else None,
        },
        "split": split_info,
        "checks": {
            "train_test_pair_overlap": int(len(overlap)),
            "users_without_train": int(len(test_users - train_users)),
            "users_without_test": int(len(train_users - test_users)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Amazon 5-core 数据预处理与划分")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行解析 JSON 的进程数（默认 min(16, CPU 核数)）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre = cfg["preprocessing"]
    paths = cfg["paths"]

    raw_path = Path(cfg["dataset"]["raw_file"])
    if not raw_path.exists():
        print(f"[error] 原始数据不存在: {raw_path}，请先运行 download_data.py")
        return 1

    stages = ["读取数据", "字段规范化", "清洗去重", "按用户划分", "落盘", "统计"]
    with tqdm(total=len(stages), desc="预处理", unit="阶段") as pbar:
        pbar.set_postfix_str(stages[0])
        df = read_raw(raw_path, args.workers)
        pbar.update(1)

        pbar.set_postfix_str(stages[1])
        df = normalize(df)
        pbar.update(1)

        pbar.set_postfix_str(stages[2])
        df, removed = clean(df, pre["min_rating"], pre["max_rating"])
        pbar.update(1)

        pbar.set_postfix_str(stages[3])
        train, test, split_info = split_per_user(df, pre["test_ratio"], pre["seed"])
        pbar.update(1)

        # 落盘
        pbar.set_postfix_str(stages[4])
        for out_dir in {Path(paths["processed_dir"])}:
            out_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(paths["train_csv"], index=False, encoding="utf-8")
        test.to_csv(paths["test_csv"], index=False, encoding="utf-8")
        train.to_parquet(paths["train_parquet"], index=False)
        test.to_parquet(paths["test_parquet"], index=False)
        print(f"[save] {paths['train_csv']}")
        print(f"[save] {paths['test_csv']}")
        print(f"[save] {paths['train_parquet']}")
        print(f"[save] {paths['test_parquet']}")
        pbar.update(1)

        pbar.set_postfix_str(stages[5])
        stats = compute_stats(df, train, test, removed, cfg, split_info)
        with open(paths["stats_json"], "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[save] {paths['stats_json']}")
        pbar.update(1)

    # 摘要
    print("\n===== 预处理摘要 =====")
    print(f"原始记录     : {removed['raw_records']:,d}")
    print(f"移除(非法)   : {removed['invalid_removed']:,d}")
    print(f"移除(重复)   : {removed['duplicates_removed']:,d}")
    print(f"清洗后记录   : {removed['clean_records']:,d}")
    print(f"用户 / 商品  : {stats['all']['users']:,d} / {stats['all']['items']:,d}")
    print(f"训练 / 测试  : {split_info['train_records']:,d} / {split_info['test_records']:,d} "
          f"(测试占比 {split_info['actual_test_ratio']:.2%})")
    print(f"泄漏检查(重叠): {stats['checks']['train_test_pair_overlap']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
