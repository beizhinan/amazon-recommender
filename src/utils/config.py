"""配置读取与项目路径。"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """读取 config.yaml，并把相对路径解析为项目内绝对路径。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for section, key in (("dataset", "raw_file"),):
        if cfg.get(section, {}).get(key):
            cfg[section][key] = str(PROJECT_ROOT / cfg[section][key])
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(PROJECT_ROOT / value)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
