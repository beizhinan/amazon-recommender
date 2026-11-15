"""下载 Amazon 5-core 数据集到 data/raw/。

用法:
    python src/preprocessing/download_data.py
    python src/preprocessing/download_data.py --force   # 强制重新下载
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 将相对路径统一转为基于项目根目录的绝对路径
    for key, value in cfg["dataset"].items():
        if key.endswith("_file") or key.endswith("_dir"):
            cfg["dataset"][key] = str(PROJECT_ROOT / value)
    return cfg


def download(url: str, dst: Path, force: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"[skip] 已存在: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        return dst

    tmp = dst.with_name(dst.name + ".part")
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=dst.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(dst)
    print(f"[done] 保存到: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def verify_gzip(path: Path) -> None:
    """尝试读取 gzip 首个换行，确认文件未损坏。"""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        f.readline()
    print("[verify] gzip 文件可正常读取")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 Amazon 5-core 数据集")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    dst = Path(ds["raw_file"])

    download(ds["url"], dst, force=args.force)
    verify_gzip(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
