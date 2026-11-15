"""对比全部评分预测模型：读取 results.csv，绘制 MAE / RMSE 柱状图。

用法:
    python -m src.evaluation.compare_models
    python -m src.evaluation.compare_models --results <results.csv> --output <png>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import PROJECT_ROOT

DEFAULT_RESULTS = (PROJECT_ROOT / "experiments" / "logs" / "rating_prediction"
                   / "results.csv")
DEFAULT_OUTPUT = DEFAULT_RESULTS.parent / "model_comparison_mae_rmse.png"

DISPLAY = {
    "global": "GlobalMean",
    "usermean": "UserMean",
    "bias": "Bias Baseline",
    "itemcf": "ItemCF",
    "usercf": "UserCF",
    "slopeone": "Slope One",
    "svd": "SVD",
    "ncf": "NCF (NeuMF)",
}


def load_best(results: Path, select: str = "rmse") -> pd.DataFrame:
    """每个模型取 select 指标最优的一次运行。"""
    df = pd.read_csv(results)
    for col in ("mae", "rmse"):
        if col not in df.columns:
            raise ValueError(f"{results} 缺少列 {col}，请先运行 run_all 生成结果")
    df = df.dropna(subset=["mae", "rmse"]).copy()
    best = df.loc[df.groupby("model")[select].idxmin()].copy()
    best["name"] = best["model"].map(lambda m: DISPLAY.get(m, m))
    return best.sort_values(select).reset_index(drop=True)


def _setup_font() -> bool:
    """启用中文字体，返回是否成功（失败则图表用英文标注）。"""
    from matplotlib import font_manager

    candidates = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans SC",
                  "Source Han Sans SC", "WenQuanYi Zen Hei", "Droid Sans Fallback",
                  "SimHei", "Microsoft YaHei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def plot(best: pd.DataFrame, output: Path) -> None:
    zh = _setup_font()
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 0.62 * len(best) + 2.6), sharey=True,
    )
    panels = [
        (axes[0], "mae", "#4C72B0", "MAE（平均绝对误差）" if zh else "MAE"),
        (axes[1], "rmse", "#DD8452", "RMSE（均方根误差）" if zh else "RMSE"),
    ]
    for ax, col, color, title in panels:
        values = best[col].astype(float)
        ax.barh(best["name"], values, color=color, height=0.62, alpha=0.9)
        for y, v in enumerate(values):
            ax.text(v, y, f" {v:.4f}", va="center", fontsize=9)
        ax.set_xlim(0, values.max() * 1.16)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("误差（越低越好）" if zh else "Error (lower is better)")
        ax.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # sharey 只能翻转一次，否则会被翻回；翻转后误差最小的模型排在最上方
    axes[0].invert_yaxis()

    n_eval = int(best["n_evaluated"].max()) if "n_evaluated" in best else 0
    if zh:
        fig.suptitle(f"评分预测模型对比（Books 测试集，n={n_eval:,d}）",
                     fontsize=14, y=1.0)
    else:
        fig.suptitle(f"Rating Prediction Comparison (Books test, n={n_eval:,d})",
                     fontsize=14, y=1.0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="评分预测模型 MAE/RMSE 对比图")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--select", choices=["rmse", "mae"], default="rmse",
                        help="每个模型选最优配置所依据的指标（默认 rmse）")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"[error] 结果文件不存在: {args.results}\n"
              f"        请先运行 python -m src.models.rating_prediction.run_all --models all")
        return 1

    best = load_best(args.results, args.select)
    cols = [c for c in ("model", "mae", "rmse", "coverage", "config")
            if c in best.columns]
    print("\n===== 各模型最佳配置 =====")
    print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot(best, args.output)
    print(f"\n[save] 对比图: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())