# 技术文档：环境与运行指南

> 本文件描述仓库的技术环境、依赖、数据准备与各任务的运行方式，面向需要复现或二次开发的人。
> 项目背景、方法细节与实验结果见 [README.md](README.md)；完整的项目报告见 [report/report.md](report/report.md)。

## 1. 技术栈

| 类别 | 选型 |
|------|------|
| 语言 | Python 3.10 |
| 核心数值 | NumPy / SciPy（稀疏矩阵） / Pandas |
| 机器学习 | scikit-learn、scikit-surprise（SVD） |
| 深度学习 | PyTorch（NCF、BPR） |
| 树模型 | scikit-learn（FAR 用 GradientBoosting 类） |
| 数据/IO | PyArrow（Parquet）、PyYAML、requests、tqdm |
| 可视化 | Matplotlib、Seaborn、Jupyter |

## 2. 环境要求

### 2.1 硬件

| 任务 | 最低内存 | 说明 |
|------|----------|------|
| 数据预处理 | ≥ 8 GB | 889 万评分，流式处理 |
| 评分预测 | ≥ 16 GB | ItemCF/SVD/FAR 全量 |
| Top-N 全量 | **≥ 64 GB（建议）** | 共现矩阵约 8.25 亿 nnz，构建阶段峰值 30–50 GB |

- GPU 可选：NCF / BPR 用 PyTorch，有 CUDA GPU 可显著加速，纯 CPU 也能运行（较慢）。
- 本实验在 128 核 / 1 TB 内存 / 2×A40 的机器上完成，故日志中的耗时偏低负载场景。

### 2.2 软件

- 操作系统：Linux（脚本含 shell 命令，未在 Windows 验证）
- Python：3.10.x
- 包管理：conda + pip

## 3. 环境搭建

```bash
# 1) 创建并激活 conda 环境
conda create -n amazon-recsys python=3.10 -y
conda activate amazon-recsys

# 2) 安装依赖
pip install -r requirements.txt

# 3)（可选）GPU 版 PyTorch，按你的 CUDA 版本安装，例如 cu121：
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`requirements.txt` 中的版本约束已与 `numpy 2.x` / `scipy 1.x` / `pandas 2.x` 兼容（本项目使用 `numpy>=2.2` 系列）。

## 4. 数据准备

### 4.1 评分数据（必需）

```bash
conda activate amazon-recsys

# 下载 Books 5-core 评分（约 826 MB，SNAP 源）
python src/preprocessing/download_data.py

# 清洗 + 去重 + 每用户 80/20 划分 + 落盘
python src/preprocessing/prepare_data.py --workers 8
```

输出到 `data/processed/Books/`：

| 文件 | 说明 |
|------|------|
| `train.csv` / `test.csv` | 训练/测试集（UTF-8 CSV） |
| `train.parquet` / `test.parquet` | 列式存储，加载更快 |
| `stats.json` | 记录数、稀疏度、评分分布、泄漏检查 |

### 4.2 商品元数据（可选，仅共购图尝试用）

```bash
# 下载 meta_Books.json.gz（约 826 MB，SNAP 源，支持断点续传）
python src/preprocessing/download_meta.py

# 解析 also_buy/also_view，构建共购图 data/processed/Books/meta_graph.npz
python src/preprocessing/parse_meta.py
```

## 5. 运行方式

### 5.1 评分预测

```bash
# 基线批量实验（8 类模型 / 31 组配置，约 2 小时）
python -m src.models.rating_prediction.run_all --models all

# 生成 MSE/RMSE 对比图
python -m src.evaluation.compare_models

# 时序偏置消融（先跑 3000 用户子集验证）
python -m src.models.rating_prediction.run_improved --sample-users 3000

# FAR 网格（g0/g1）
python -m src.models.rating_prediction.run_far

# 最终算法 TAB-H（门控分段融合，约 15 分钟）
python -m src.models.rating_prediction.run_stack --far-params g1
python -m src.models.rating_prediction.run_stack --far-params g1 --refresh   # 忽略预测缓存
python -m src.models.rating_prediction.run_stack --no-far                    # 三路快速版
```

### 5.2 Top-N 推荐

```bash
# 全量（含调参 + BPR + 元数据分支，约 1–2 小时，取决于负载）
python -m src.models.topn.run_topn

# 快速冒烟（2000 用户 / 跳过 BPR）
python -m src.models.topn.run_topn --users 2000 --tune-users 1000 --quick --no-bpr

# 完整调参参数（默认值，可显式指定）
python -m src.models.topn.run_topn \
  --users 10000 --tune-users 3000 \
  --ks 200,400,800 \
  --alphas 0.5,0.7,0.85 \
  --half-lifes 2,0 \
  --weight-schemes none,bm25
```

### 5.3 EDA

```bash
jupyter notebook notebooks/01_data_eda_quality.ipynb
```

## 6. 复现关键结果

| 结果 | 命令 | 关键产物 |
|------|------|----------|
| 评分预测 TAB-H（RMSE 0.8939） | `python -m src.models.rating_prediction.run_stack --far-params g1` | `experiments/logs/rating_prediction/stack_full.log` |
| Top-N Hybrid_global（NDCG 0.0746） | `python -m src.models.topn.run_topn` | `experiments/logs/topn/results_topn.csv` |

> 两个实验入口都做了**随机种子固定**（seed=42）与**训练/调参/测试三集分离**，结果可复现。Top-N 的调参集是训练集内再切 10% 的 val，绝不进入最终模型。

## 7. 目录结构（技术视角）

```
amazon-recommender/
├── config/config.yaml        # 数据集路径、预处理参数
├── data/
│   ├── raw/                  # 原始 json.gz（评分 + 元数据）
│   └── processed/Books/      # train/test CSV/Parquet、stats.json、meta_graph.npz
├── src/
│   ├── preprocessing/        # download_*/prepare_*/parse_*
│   ├── models/
│   │   ├── rating_prediction/ # common + baselines/knn/slope_one/svd/ncf + run_*
│   │   └── topn/              # models + run_topn
│   ├── evaluation/            # rating_metrics / ranking_metrics / compare_models
│   └── utils/config.py        # PROJECT_ROOT / ensure_dir 等路径工具
├── notebooks/                # EDA
├── experiments/logs/         # 日志 / 结果 CSV / 对比图 / 预测缓存
├── report/report.md          # 项目报告
└── README.md                 # 项目背景与结果
```

## 8. 关键实现约定

1. **运行方式**：`src/preprocessing/*` 以脚本方式运行（`python src/...`，内部用 `Path(__file__).resolve().parents[2]` 定位项目根）；`src/models/*` 以模块方式运行（`python -m src.models...`，依赖 `src` 作为命名空间包导入）。请始终在项目根目录执行。
2. **路径解析**：`config.yaml` 里的相对路径由 `src/utils/config.py` 统一转为基于项目根的绝对路径。
3. **数据加载**：`src/models/rating_prediction/common.py` 的 `load_rating_data()` 是统一入口，返回 `RatingData`（含 `B` 交互矩阵、`u_tr/i_tr/r_tr` 等字段），评分预测与 Top-N 共用。
4. **防泄漏**：训练样本的统计特征（FAR）一律留一法；Top-N 的调参集与测试集严格分离；去重按时间戳取最新。
5. **评估协议**：Top-N 采用**全量排序**（候选为全部 36.8 万商品，排除训练已交互项），非采样负例；指标为 P/R/F1/NDCG/HitRate@10 + 覆盖率。

## 9. 常见问题

- **内存不足（Top-N）**：共现矩阵构建阶段内存峰值高，若 OOM，减小 `--users` / `--tune-users`，或改用子集冒烟。
- **BPR 极慢**：高负载下 BPR 单 epoch 可能从秒级膨胀到数百秒。可加 `--no-bpr` 跳过（会同时跳过 `Hybrid`/`TripleHybrid` 融合）。
- **下载缓慢**：SNAP 源（斯坦福）跨境带宽低，单连接常只有几十 KB/s。用 aria2c 多连接续传可提速：

  ```bash
  aria2c -c -x 16 -s 16 -k 1M --file-allocation=none \
    -o meta_Books.json.gz.part \
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz"
  ```
- **结果 CSV 被覆盖**：Top-N 结果按 `run_time` 追加写入 `results_topn.csv`，冒烟测试前建议先备份该文件。
