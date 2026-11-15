"""下载 Amazon 商品元数据（meta_*.json.gz）到 data/raw/。

数据源与 download_data.py 保持一致（SNAP Amazon productGraph categoryFiles），
文件名遵循命名约定：reviews_Books_5.json.gz -> meta_Books.json.gz。

用法:
    python src/preprocessing/download_meta.py            # 按 config 推导链接
    python src/preprocessing/download_meta.py --force    # 强制重新下载
    python src/preprocessing/download_meta.py --url URL  # 指定链接
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
CHUNK = 1 << 20


def load_config(path: Path | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 将相对路径统一转为基于项目根目录的绝对路径
    for key, value in cfg["dataset"].items():
        if key.endswith("_file") or key.endswith("_dir"):
            cfg["dataset"][key] = str(PROJECT_ROOT / value)
    return cfg


def derive_meta_url(review_url: str) -> str:
    """由评论文件链接推导元数据链接（reviews_X_5.json.gz -> meta_X.json.gz）。"""
    name = Path(urlparse(review_url).path).name
    if not name.startswith("reviews_"):
        raise ValueError(f"无法从 {name} 推导元数据文件名")
    return review_url.replace(name, "meta_" + name[len("reviews_"):]).replace(
        "_5.json.gz", ".json.gz")


def download(url: str, dst: Path, force: bool = False, retries: int = 5) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"[skip] 已存在: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        return dst

    tmp = dst.with_name(dst.name + ".part")
    if force:
        tmp.unlink(missing_ok=True)
    pos = tmp.stat().st_size if tmp.exists() else 0
    if pos:
        print(f"[resume] 从 {pos / 1e6:.1f} MB 处继续")

    print(f"[download] {url}")
    for attempt in range(1, retries + 1):
        try:
            headers = {"Range": f"bytes={pos}-"} if pos else {}
            with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
                if pos and resp.status_code == 200:
                    # 服务器不支持断点续传，从头写
                    pos = 0
                    tmp.unlink(missing_ok=True)
                resp.raise_for_status()
                size = int(resp.headers.get("Content-Length", 0))
                total = size + pos if size else None
                with open(tmp, "ab" if pos else "wb") as f, tqdm(
                    total=total, initial=pos, unit="B", unit_scale=True,
                    desc=dst.name,
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=CHUNK):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
                            pos += len(chunk)
            break
        except Exception as exc:  # 网络抖动重试
            print(f"[retry {attempt}/{retries}] {type(exc).__name__}: {exc}")
            if attempt == retries:
                raise
            time.sleep(3 * attempt)

    tmp.replace(dst)
    print(f"[done] 保存到: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def verify(path: Path) -> None:
    """校验 gzip 可读，并确认包含共购关系字段。"""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = json.loads(f.readline())
    keys = set(first)
    print(f"[verify] 首条记录字段: {sorted(keys)}")
    related = first.get("related")
    has_graph = "also_buy" in keys or "also_view" in keys or bool(related)
    print(f"[verify] 共购关系字段: {'存在' if has_graph else '未找到'}")
    if not has_graph:
        print("[warn] 该元数据不含 also_buy/also_view/related，无法构建共购图")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 Amazon 商品元数据")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    parser.add_argument("--url", type=str, default=None, help="元数据下载链接")
    parser.add_argument("--out", type=Path, default=None, help="保存路径")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    url = args.url or derive_meta_url(ds["url"])
    dst = args.out or Path(ds["raw_file"]).with_name(Path(urlparse(url).path).name)

    download(url, dst, force=args.force)
    verify(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
