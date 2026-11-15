# 跨媒体推荐系统前沿（Amazon 商品推荐）

基于亚马逊 5-core 数据集实现的评分预测与 Top-N 推荐项目。

> 环境搭建与运行方式见 [DEVELOPMENT.md](DEVELOPMENT.md)；完整项目报告见 [report/report.md](report/report.md)。

## 项目流程

1. **数据选择与预处理**：选定商品领域 → 每用户 80% 评分作训练集、20% 作测试集
2. **评分预测**：协同过滤 / Slope One / 矩阵分解 / 融合模型，用 MAE、RMSE 评估
3. **Top-N 推荐**：生成长度 10 的推荐列表，用 Precision、Recall、F-measure、NDCG 评估

## 目录结构

```
amazon-recommender/
├── data/
│   ├── raw/                  # 原始 5-core 数据（json.gz）
│   └── processed/            # 清洗后、划分好的 train/test 数据（按数据集分目录）
├── src/
│   ├── preprocessing/
│   │   ├── download_data.py   # 下载 5-core 评分数据（SNAP 源）
│   │   ├── prepare_data.py    # 清洗 + 去重 + 80/20 划分 + 落盘
│   │   ├── download_meta.py   # 下载商品元数据 meta_Books.json.gz（SNAP 源）
│   │   └── parse_meta.py      # 解析元数据，构建 also_buy/also_view 共购图
│   ├── models/
│   │   ├── rating_prediction/
│   │   │   ├── common.py         # 数据加载 / RatingData / 日志 / 结果落盘
│   │   │   ├── baselines.py      # GlobalMean / UserMean / BiasBaseline
│   │   │   ├── knn.py            # ItemCF / UserCF
│   │   │   ├── slope_one.py      # Slope One
│   │   │   ├── svd_surprise.py   # SVD（Surprise）
│   │   │   ├── ncf.py            # NeuMF（PyTorch）
│   │   │   ├── run_all.py        # 基线批量实验入口
│   │   │   ├── time_bias.py      # TAB 时序动态偏置（消融负结果）
│   │   │   ├── far_model.py      # FAR 留一特征工程
│   │   │   ├── run_far.py        # FAR 实验入口
│   │   │   ├── run_improved.py   # 消融阶梯 A–F / 数据视图 / 结果落盘
│   │   │   └── run_stack.py      # TAB-H 门控分段融合（最终算法）
│   │   └── topn/
│   │       ├── models.py         # MostPop / RatingPred / ItemKNN / BPR / Hybrid / MetaItemKNN / TripleHybrid
│   │       └── run_topn.py       # Top-N 实验入口（全量排序 + 调参）
│   ├── evaluation/
│   │   ├── rating_metrics.py   # MAE、RMSE、MSE、覆盖率
│   │   ├── ranking_metrics.py  # Precision/Recall/F1/NDCG@K
│   │   └── compare_models.py   # 模型对比作图
│   └── utils/                  # 配置与路径工具（config.py）
├── config/                   # 配置文件（数据集路径、模型参数）
├── notebooks/                # 01_data_eda_quality.ipynb（数据质量与分布分析）
├── experiments/
│   └── logs/                 # 实验日志、结果 CSV、对比图、预测缓存
├── report/
│   └── report.md             # 项目报告
└── slides/                   # 项目幻灯片
```

## 数据获取

- 评分数据 5-core：<http://jmcauley.ucsd.edu/data/amazon/>
- 实际下载地址（Stanford SNAP 镜像）：`https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz`
- 商品元数据（共购图尝试，见 3.8）：`https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz`
- 字段：`reviewerID` / `asin` / `overall`(1-5) / `reviewText` / `summary` / `unixReviewTime` / `helpful`

## 第一步：数据选择与预处理

选定领域为 **Books**（5-core，889 万评分 / 60.4 万用户 / 36.8 万商品），配置见 `config/config.yaml`。产出的数据按数据集分子目录存放，便于后续切换类别；早期在 Movies_and_TV、Home_and_Kitchen 上的结果保留在 `data/processed/` 下。

**处理规则**

- 保留全部原始字段：`reviewText`、`summary`、`unixReviewTime`、`helpful`、`reviewerName` 等；`helpful` 拆为 `helpful_no` / `helpful_yes`。
- 清洗：丢弃缺失 user/item/rating 的记录，过滤评分不在 1–5 的记录；剔除文本中的 NUL 字符（`\x00` 会导致 CSV 写出报 `need to escape`）与孤立代理字符。
- 去重：同一 `(user_id, item_id)` 保留时间戳最新的一条，避免训练/测试泄漏。
- 划分：每个用户随机 20% 评分作测试集、其余作训练集，随机种子固定为 42。

**运行**

```bash
conda activate amazon-recsys
python src/preprocessing/download_data.py   # 下载到 data/raw/
python src/preprocessing/prepare_data.py   # 清洗 + 划分 + 落盘（支持 --workers 并行）
```

**输出（`data/processed/Books/`）**

| 文件 | 说明 |
|------|------|
| `train.csv` / `test.csv` | 训练/测试集，UTF-8 CSV |
| `train.parquet` / `test.parquet` | 同上，列式存储 |
| `stats.json` | 记录数、用户/商品数、稀疏度、评分分布、划分与泄漏检查 |

规范列：`user_id`、`item_id`、`rating`、`review_text`、`summary`、`timestamp`、`review_time`、`reviewer_name`、`helpful_no`、`helpful_yes`、`extra`。

## 第二步：评分预测

### 2.1 数据概况与评估协议

| 项目 | 全部 | 训练集 | 测试集 |
|------|------|--------|--------|
| 评分数 | 8,898,041 | 7,142,099 | 1,755,942 |
| 用户数 | 603,668 | 603,668 | 603,668 |
| 商品数 | 367,982 | 367,965 | 318,935 |
| 稀疏度 | 99.9960% | 99.9968% | 99.9991% |
| 评分均值 | 4.2499 | 4.2501 | 4.2494 |
| 每用户评分中位数 | 8 | 6 | 2 |

评分分布极度偏正：5 星占 56.0%，≥4 星占 81.0%，1–2 星合计仅 8.3%。这决定了「预测偏向高分段」本身就是降低 RMSE 的有效策略，也让用户均值成为极强的基线。

**指标**：`MAE = mean|r̂ − r|`、`RMSE = sqrt(mean(r̂ − r)²)`、`MSE`。预测裁剪到 `[1, 5]`；邻域模型无可用邻居时回退到双偏置基线，回退率单独统计（覆盖率恒为 100%，因此只报回退率）。

**基线定义**：`μ` 为训练集全局均值；`UserMean` 为 `μ + b_u`；`BiasBaseline` 为 `μ + b_u + b_i`，用交替最小二乘（ALS）求解，L2 正则 `reg_u = reg_i = λ`：

```
b_u = Σ_{i∈R_u}(r_ui − μ − b_i) / (|R_u| + λ)
b_i = Σ_{u∈R_i}(r_ui − μ − b_u) / (|R_i| + λ)
```

### 2.2 基线模型实现

| 标识 | 模型 | 核心做法 |
|------|------|----------|
| `global` | GlobalMean | 全部预测为训练集均值 |
| `usermean` | UserMean | 用户偏置（正则收缩） |
| `bias` | BiasBaseline | 用户 + 商品双偏置，ALS 求解 |
| `itemcf` | ItemCF | 用户中心化评分上的余弦相似度 + 共现收缩 + 每物品 top-K；预测 `μ + b_u + b_i + Σ sim·res/Σ|sim|` |
| `usercf` | UserCF | 用户相似度 + 邻居残差加权 |
| `slopeone` | Slope One | 带 top-K 与最小共现数的加权偏差 |
| `svd` | SVD（Surprise） | 带偏置的 SGD 矩阵分解 |
| `ncf` | NeuMF | GMF + MLP 双塔，PyTorch 实现 |

### 2.3 全量实验结果（8 类模型 / 31 组配置）

`results.csv`，测试集 1,755,942 条，按 RMSE 升序：

| 模型 | 配置 | MAE | RMSE | 回退率 | 耗时(s) |
|------|------|-----|------|--------|---------|
| bias | reg=3 | **0.6574** | **0.9023** | 0 | 2.8 |
| bias | reg=2 | 0.6535 | 0.9032 | 0 | 3.0 |
| bias | reg=5 | 0.6655 | 0.9038 | 0 | 3.2 |
| svd | 50×30, reg=0.1 | 0.6672 | 0.9063 | 0 | 229.6 |
| bias | reg=1 | 0.6504 | 0.9066 | 0 | 2.9 |
| svd | 100×30, reg=0.1 | 0.6697 | 0.9080 | 0 | 301.6 |
| ncf | 32/32, [64,32,16], 10ep | 0.6576 | 0.9085 | 0 | 225.3 |
| bias | reg=8 | 0.6764 | 0.9089 | 0 | 3.0 |
| ncf | 64/64, [128,64,32], 15ep | 0.6605 | 0.9089 | 0 | 600.7 |
| svd | 150×30, reg=0.1 | 0.6720 | 0.9097 | 0 | 395.9 |
| bias | reg=0.5 | 0.6497 | 0.9102 | 0 | 3.0 |
| bias | reg=10 | 0.6826 | 0.9126 | 0 | 2.8 |
| svd | 100×30, reg=0.05 | 0.6704 | 0.9127 | 0 | 274.4 |
| bias | reg_u=10, reg_i=25 | 0.6896 | 0.9175 | 0 | 3.4 |
| usercf | k=20, shrink=25 | 0.6688 | 0.9226 | 80.82% | 490.6 |
| itemcf | k=50, shrink=25 | **0.6531** | 0.9265 | 63.44% | 321.5 |
| usercf | k=50, shrink=10 | 0.6698 | 0.9293 | 74.18% | 531.1 |
| usercf | k=50, shrink=25 | 0.6709 | 0.9313 | 72.11% | 504.8 |
| itemcf | k=100, shrink=10 | 0.6543 | 0.9317 | 58.34% | 300.2 |
| itemcf | k=100, shrink=25 | 0.6543 | 0.9334 | 57.03% | 320.8 |
| itemcf | k=100, shrink=50 | 0.6544 | 0.9341 | 56.49% | 379.3 |
| bias | reg=25 | 0.7125 | 0.9356 | 0 | 2.7 |
| usercf | k=100, shrink=25 | 0.6728 | 0.9389 | 64.07% | 561.7 |
| itemcf | k=200, shrink=25 | 0.6552 | 0.9403 | 50.15% | 296.2 |
| usermean | reg=10 | 0.7290 | 0.9580 | 0 | 0.1 |
| slopeone | k=100, min_count=2 | 0.6687 | 0.9603 | 45.73% | 306.1 |
| slopeone | k=50 | 0.6716 | 0.9660 | 47.38% | 292.3 |
| slopeone | k=100 | 0.6743 | 0.9748 | 41.29% | 287.7 |
| usermean | reg=25 | 0.7534 | 0.9774 | 0 | 0.2 |
| slopeone | k=200 | 0.6766 | 0.9825 | 35.71% | 311.8 |
| global | — | 0.8387 | 1.0562 | 0 | 0.1 |

对比图：`experiments/logs/rating_prediction/model_comparison_mse_rmse.png`。

**关键观察**

1. **双偏置（reg=3）RMSE 最优 0.9023**，比 SVD 低 0.44%、比 NCF 低 0.68%，而训练只需 3 秒。
2. **ItemCF 的 MAE 最低（0.6531）但 RMSE 明显劣于 bias（0.9265）**：协同修正幅度大，猜中时很准，猜偏时误差被平方放大。这为后续融合提供了明确动机。
3. **邻域模型回退率极高**（ItemCF 63.4%、UserCF 80.8%），说明 5-core 下共现信息严重不足，大部分预测实际是基线在兜底。
4. `reg` 太小时 RMSE 变差（0.9102 @ 0.5）、MAE 变好，呈典型偏差-方差取舍；RMSE 最优点在 `reg≈3`。
5. bias/SVD/NCF 的 RMSE/MAE 比值集中在 1.36–1.38，邻域模型与 Slope One 达 1.42–1.45（误差重尾，少数大误差主导 RMSE），全局均值最低 1.26。本项目 bias 基线（0.9023 / 0.6574）优于 Amazon 评分预测的常见水平（RMSE 0.95–1.05、MAE 0.70–0.78），主要原因是 Books 评分分布极度偏正（56% 为 5 星）。

### 2.4 结果分析：为什么单模型难以突破 0.90

- **用户效应主导**：`usermean` 已达 0.9580，而 `item_mean` 单独只有 1.0124（比全局均值 1.0562 好得有限）。Books 的商品粒度极细（36.8 万件，平均每件仅 24 条评分），`b_i` 估计噪声大。
- **极端稀疏 + 长尾**：稀疏度 99.9968%，训练集每用户中位数仅 6 条评分，邻居证据不足，CF 的回退与噪声同时上升。
- **评分分布偏斜**：56% 为 5 星，模型的期望输出天然靠近 4.2–4.3；此时「稳定的收缩估计」比「激进的协同修正」更能降低平方误差。
- **结论**：继续堆叠单一模型（更深 MF、更大 K、更多隐因子）收益趋零。需要的是**误差结构互补**的正交信号，而不是更强的同质模型。

### 2.5 文献调研结论

| 方向 | 代表工作 | 在 5-core 上的实际收益 | 决策 |
|------|----------|------------------------|------|
| 评论文本建模 | DeepCoNN / NARRE / MPCNN | 报告增益 0–1.3%，且 DeepCoNN 使用**被测评论本身**作为输入，属 transductive 设置，与「用历史评分预测」的协议不一致，存在泄漏 | 不采用 |
| 时序动态偏置 | Koren, *Collaborative Filtering with Temporal Dynamics* (KDD'09) | Netflix 上 ~1.3%；5-core 每用户中位数仅 6 条、时间跨度内评分极稀，漂移项无法稳定估计 | 已实现并消融，负结果 |
| 序数 / 分位数回归 | RecRec 等序数推荐复现 | 主要改善 MAE 与排序一致性，对 RMSE 无稳定增益 | 不采用 |
| 特征增强 + GBDT | 工业界常用的 target encoding / GBDT 排序 | 需要足够样本量才能超过线性偏置；小样本下显著更差 | 采用（FAR），并验证了样本量依赖 |
| 模型融合 / stacking | 门控加权、按上下文分段 | 在异质模型间稳定有效，是稀疏场景中性价比最高的改进 | **采用（TAB-H）** |

调研的核心判断：在 Amazon 5-core 这种「用户多、每人极少、评分偏正」的数据上，**提升来自信号正交性而非模型容量**。

### 2.6 尝试一：时序动态偏置（TAB，负结果）

实现见 [time_bias.py](file:///data2/malong/media_lab/amazon-recommender/src/models/rating_prediction/time_bias.py)，模型为

```
r̂ = μ + b_u + α_u · dev_u(t) + b_i + b_{i,Bin(t)} + g_{Bin(t)}
dev_u(t) = sign(t − t_u) · |t − t_u|^0.4
```

并加入活跃度自适应正则 `λ_u = reg_u · (median_n / n_u)^γ`（clip 到 0.25×–4×）与等距回归（isotonic）校准。在 3,000 用户子集上逐项开启做消融（`run_improved.py`，验证集选模型）：

| 配置 | test MAE | test RMSE | val RMSE |
|------|----------|-----------|----------|
| A 仅双偏置 | 0.7288 | 0.9638 | 0.9500 |
| B +全局时间漂移 | **0.7271** | 0.9645 | 0.9503 |
| C +用户漂移 | 0.7255 | 0.9639 | 0.9504 |
| D +商品时间偏置 | 0.7262 | 0.9645 | 0.9508 |
| E +自适应正则 | **0.7231** | 0.9649 | 0.9511 |
| F +isotonic 校准 | 0.7271 | 0.9662 | 0.9511 |

结论：MAE 有 0.5%–0.8% 的边际改善，但 **RMSE 全部持平或变差**；漂移项在每用户中位数 6 条评分时无法稳定估计，反而引入方差。校准会破坏 RMSE（等距回归优化的是排序/单调性，不是平方损失）。该分支不进入最终模型。

### 2.7 尝试二：FAR 特征增强残差模型

实现见 [far_model.py](file:///data2/malong/media_lab/amazon-recommender/src/models/rating_prediction/far_model.py)、[run_far.py](file:///data2/malong/media_lab/amazon-recommender/src/models/rating_prediction/run_far.py)。

**设计**：把「双偏置无法表达的异方差」交给树模型——冷门商品、低频用户的偏差估计噪声大，应被更强收缩。10 维特征：用户/商品评分均值与离散度、`log1p(评分数)`、评分时间、相对用户/商品平均时间的偏离、商品时间跨度。

**防泄漏**：训练样本的统计特征一律**留一法**（扣除该行自身的评分与时间贡献），测试样本用全量统计；特征中不含 `bias` 预测值，避免 stacking 泄漏。

**结果（同一 90% fit 子集，验证集选参数）**

| 模型 | val RMSE | test RMSE | test MAE | 耗时(s) |
|------|----------|-----------|----------|---------|
| bias（同子集） | 0.9111 | 0.9059 | 0.6608 | 2.7 |
| FAR-g0（300 树 / lr 0.06 / 63 叶） | 0.9081 | 0.9032 | 0.6592 | 93 |
| FAR-g1（600 树 / lr 0.03 / 127 叶） | 0.9076 | **0.9028** | **0.6588** | 428 |
| FAR-g0（3,000 用户子集） | 0.9870 | 1.0168 | 0.7576 | 6.7 |
| bias（3,000 用户子集） | 0.9500 | 0.9638 | 0.7288 | 0.0 |

**重要现象**：小样本（3 千用户）上 FAR 比 bias 差 5.5%，全量上反而好 0.34%。树模型的非线性收缩需要足够样本才能估准，这也解释了为什么很多论文在小数据上复现不出 GBDT 的收益。FAR 单独只赢 0.34%，但它的优势样本（低频/冷门）与 CF 的覆盖样本几乎不重叠——这正是融合的价值来源。

### 2.8 尝试三：TAB-H 门控分段加权融合（最终算法）

实现见 [run_stack.py](file:///data2/malong/media_lab/amazon-recommender/src/models/rating_prediction/run_stack.py)。

**模型**

```
r̂ = clip( w₀ + w₁·r̂_bias + w₂·r̂_cf + w₃·r̂_svd + w₄·r̂_far , 1, 5 )
```

**三个设计要点**

1. **门控（gating）**：`cf` 只在「该样本存在可用协同邻居」时进入组合。否则 ItemCF 输出的是基线回退值，把它当独立信号会污染权重估计。
2. **分段（segmentation）**：在门控内再按用户训练评分数分桶 `[0,5] / (5,15] / (15,50] / >50`，每段独立求解权重，让「低频用户靠 FAR 收缩、高频用户敢用协同信号」的差异被数据自动学出来。
3. **非负最小二乘（NNLS）**：`w ≥ 0`，避免负权重放大噪声；段内样本 < 2000 时回退到该门控下的全局权重；仅在 train 内 10% 验证集学习，模型选择看未校准的 val RMSE。

**结果（Books 全量，test 1,755,942 条）**

| 模型 | val RMSE | val MAE | test RMSE | test MAE |
|------|----------|---------|-----------|----------|
| bias（同子集） | 0.9111 | 0.6659 | 0.9059 | 0.6608 |
| ItemCF（覆盖 33.15%） | 0.8223 | 0.5168 | 0.8200 | 0.5149 |
| SVD | 0.9156 | 0.6771 | 0.9100 | 0.6711 |
| FAR-g1 | 0.9076 | 0.6636 | 0.9028 | 0.6588 |
| TAB-H 三路（bias + cf + svd） | 0.9052 | 0.6620 | 0.8998 | 0.6569 |
| **TAB-H 四路（+ FAR）** | **0.8991** | **0.6546** | **0.8939** | **0.6496** |

- 相对同子集 bias：**RMSE −1.32%，MAE −1.69%**（两项同时改善，非指标互换）。
- 相对全量训练的 bias 最优（0.9023 / 0.6574）：**RMSE −0.93%，MAE −1.19%**。注意 TAB-H 的各分支只用了 90% 训练行，仍能超过全量训练的 bias，结论不依赖切分带来的优势。
- 全流程耗时约 15 分钟（bias 5s、ItemCF 264s、SVD 180s、FAR 344s、三路预测约 100s、权重求解 <1s）。

**学到的分段权重**

| 门控 | 活跃度 | n | const | bias | cf | svd | far |
|------|--------|---|-------|------|----|-----|-----|
| 有邻居 | [0,5] | 27,107 | 0.302 | 0.000 | 0.321 | 0.000 | 0.628 |
| 有邻居 | (5,15] | 39,415 | 0.229 | 0.062 | 0.317 | 0.000 | 0.583 |
| 有邻居 | (15,50] | 51,679 | 0.162 | 0.115 | 0.262 | 0.094 | 0.501 |
| 有邻居 | >50 | 108,231 | 0.072 | 0.205 | 0.227 | 0.000 | 0.556 |
| 无邻居 | [0,5] | 160,616 | 0.000 | 0.324 | — | 0.000 | 0.672 |
| 无邻居 | (5,15] | 141,081 | 0.000 | 0.395 | — | 0.000 | 0.598 |
| 无邻居 | (15,50] | 98,452 | 0.000 | 0.480 | — | 0.000 | 0.511 |
| 无邻居 | >50 | 86,877 | 0.000 | 0.477 | — | 0.000 | 0.516 |

**解读**

- 用户越活跃，`bias` 权重越高（偏置估计更可靠）；最冷的一端（0–5 条）几乎完全交给 FAR 的强收缩。
- 有邻居时 `cf` 获得 0.23–0.32 的稳定权重，与 FAR 形成互补而非替代。
- **SVD 权重几乎全为 0**：隐因子在 99.9968% 稀疏度下与 bias/FAR 高度冗余，只在 (15,50] 段拿到 0.094。若追求效率可加 `--no-svd` 去掉该分支（省约 3 分钟，指标基本不变）。

### 2.9 复现命令

```bash
conda activate amazon-recsys

# 1) 8 类基线（31 组配置，约 2 小时）
python -m src.models.rating_prediction.run_all --models all

# 2) 生成 MSE / RMSE 对比图
python -m src.evaluation.compare_models

# 3) 时序偏置消融（A–F，建议先用子集验证）
python -m src.models.rating_prediction.run_improved --sample-users 3000

# 4) FAR 网格
python -m src.models.rating_prediction.run_far            # g0/g1 全量

# 5) 最终算法 TAB-H（含 FAR 分支）
python -m src.models.rating_prediction.run_stack --far-params g1
python -m src.models.rating_prediction.run_stack --far-params g1 --refresh  # 忽略预测缓存
python -m src.models.rating_prediction.run_stack --no-far                   # 三路快速版
```

### 2.10 产物清单

| 路径 | 内容 |
|------|------|
| `experiments/logs/rating_prediction/results.csv` | 31 组基线配置的 MAE/RMSE/MSE/回退率/耗时 |
| `experiments/logs/rating_prediction/summary_best.csv` | 每个模型的最优配置（按 RMSE） |
| `experiments/logs/rating_prediction/results_improved.csv` | TAB 消融、FAR、TAB-H 的结果 |
| `experiments/logs/rating_prediction/model_comparison_mse_rmse.png` | 模型对比柱状图 |
| `experiments/logs/rating_prediction/stack_preds_*.npz` | 各分支预测缓存（调权重时免重训） |
| `experiments/logs/rating_prediction/stack_full.log` | TAB-H 全量运行日志 |
| `experiments/logs/eda/Books/` | EDA 图表（`notebooks/01_data_eda_quality.ipynb` 输出） |

### 2.11 小结

- 在 Books 5-core 上，**双偏置是单模型上限**（RMSE 0.9023 / MAE 0.6574），SVD、NCF、CF 均无法超越。
- 时序动态、序数校准在当前稀疏度下无效（负结果已记录）；文本模型存在协议性泄漏风险，未采用。
- **TAB-H 门控分段融合达到 RMSE 0.8939 / MAE 0.6496**，相对同子集 bias 提升 1.32% / 1.69%，且权重可解释（活跃度越高越信偏置，越低越信 FAR 收缩）。
- 进一步的改进空间主要在 Top-N 任务（融合模型的排序能力）与跨域/元数据信号，而非继续压缩评分误差。

## 第三步：Top-N 推荐

### 3.1 评估协议

| 项目 | 设置 |
|------|------|
| 候选集 | 全部 367,982 个商品（全量排序，非采样负例） |
| 排除 | 用户在训练集中已交互的商品（测试集商品不排除） |
| 评估用户 | 随机 10,000 个用户（seed=42），其测试集商品作为 ground truth |
| 列表长度 | K = 10 |
| 指标 | Precision@10、Recall@10、F1@10、NDCG@10、HitRate@10，按用户宏平均 |
| 调参 | 训练集再切 90% fit / 10% val（1,606 个有验证交互的用户）；共现矩阵昂贵且只依赖 (half_life, 权重方案)，故用两阶段坐标搜索：先 α×k，再 (half_life, 权重方案)，最后单独搜索 λ |

Books 的测试集每用户中位数只有 2 个商品，且候选池 36.8 万，因此各项指标的绝对值天然很低（与采样负例的评测不可直接比较）。调参集只有 1,606 个用户，其 NDCG 系统性低于 10k 测试集约 15–20%，因此**只有幅度显著的差异才可采信**，零点几个百分点的比较需谨慎。

### 3.2 模型

| 模型 | 做法 |
|------|------|
| **MostPop** | 按训练集交互次数排序（全局同一列表，仅按用户历史过滤） |
| **RatingPred** | 直接按评分预测 `μ + b_u + b_i` 排序（第二步的最优模型） |
| **ItemKNN** | 非对称余弦 `s_ij = C_ij / (‖w_i‖^α · ‖w_j‖^(1-α))`（Steck, 2019）+ BM25 用户活跃度归一化 `w_ui /= log2(1+n_u)` + 共现收缩 `n/(n+10)` + 每物品 top-k |
| **BPR** | BPR-MF 成对排序损失，64 维隐因子，Adam(1e-3)，20 epoch，负样本均匀采样 |
| **Hybrid_global** | `score = z(BPR) + λ·z(ItemKNN)`，λ 在 val 上选择（最终 λ=0.15）；另有分段版 `Hybrid_seg`（按用户/商品桶分段设 λ，见 3.3） |

α 是对称余弦的推广：α=0.5 即标准余弦；α>0.5 时对**热门候选商品**施加更强惩罚。BM25 项则从另一侧修正流行度偏置——修正的是**重度用户**（读 500 本书的用户原本会主导共现统计）。两者正交。

### 3.3 调参结果（val，1,606 个有效用户）

**阶段一：α × k（half_life=2 年，无 BM25）**

| α \ k | 200 | 400 | 800 |
|-------|-----|-----|-----|
| 0.5 | 0.0254 | 0.0265 | 0.0268 |
| 0.7 | **0.0440** | 0.0432 | 0.0432 |
| 0.85 | 0.0464 | 0.0465 | **0.0466** |

- α 从 0.5 提到 0.7 带来 **+73%**，0.7→0.85 只剩 **+5.5%**，说明「惩罚热门候选」的收益已接近饱和。
- k 从 200 增到 800 的差异不到 1%，**k=200 已足够**；长尾并非卡在「邻居数量不足」上。

**阶段二：half_life × 权重方案（α=0.85，k=800）**

| half_life | 无加权 | BM25 |
|-----------|--------|------|
| 2 年 | 0.0466 | 0.0566 |
| 不衰减（0） | 0.0447 | **0.0595** |

- **BM25 用户活跃度归一化是这一轮第二大增益：+33%**（0.0447 → 0.0595）。
- **时间衰减没有稳定收益**：无加权时它略正（0.0466 vs 0.0447，+4%），与 BM25 组合时反而为负（0.0566 vs 0.0595，−5%），符号随配置翻转，幅度处于抽样噪声量级。最终选择**不衰减**，模型也因此更简单。

**融合系数 λ（全局）**

| λ | 0.15 | 0.25 | 0.5 | 1.0 |
|---|------|------|-----|-----|
| NDCG@10 | **0.0625** | 0.0615 | 0.0601 | 0.0596 |

λ 越小越好且落在网格下界，说明 BPR 只能提供很弱的正向增量。

**分段 λ（用户活跃度 4 档 × 商品流行度 2 档）**：坐标搜索后 val NDCG 0.0643，比全局 λ 高 2.9%；但**测试集上反而略低**（0.0742 vs 0.0746），属于对 1,606 人调参集的过拟合，最终采用全局 λ。

### 3.4 全量测试结果（10,000 用户 × 367,982 候选商品）

| 模型 | P@10 | R@10 | F1@10 | NDCG@10 | HitRate@10 | 覆盖率 | 耗时 |
|------|------|------|-------|---------|------------|--------|------|
| **Hybrid_global** | **0.02307** | **0.10243** | **0.03281** | **0.07456** | **0.1830** | 16.23% | 46s |
| ItemKNN（α0.85, k800, BM25） | 0.02208 | 0.09556 | 0.03094 | 0.07065 | 0.1714 | 19.08% | 28s |
| BPR | 0.00634 | 0.02738 | 0.00882 | 0.01696 | 0.0553 | 2.24% | 25s |
| MostPop | 0.00165 | 0.00888 | 0.00253 | 0.00500 | 0.0160 | 0.00% | 29s |
| RatingPred | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 34s |

**三轮迭代的演进**（同一协议、同一 10k 用户，MostPop 每轮均为 0.0050，可验证口径一致）

| 版本 | 关键改动 | ItemKNN NDCG | Hybrid NDCG |
|------|---------|--------------|-------------|
| v1 | 对称余弦，k=200，无衰减 | 0.0227 | 0.0288 |
| v2 | 非对称余弦 α=0.7，2 年衰减 | 0.0472（+108%） | 0.0581（+102%） |
| **v3** | α=0.85，k=800，BM25 归一化，去衰减 | **0.0707（+50%）** | **0.0746（+28%）** |

> 口径说明：v2 行的 Hybrid NDCG=0.0581 是分段 λ（`Hybrid_seg`），v3 行的 0.0746 是全局 λ（`Hybrid_global`）；v3 判定分段 λ 过拟合后统一回退到全局 λ，故两行的 Hybrid 不可直接等同对比。

- **Hybrid 相对 MostPop：NDCG@10 +1,391%，Recall@10 +1,054%**；相对随机基线（Recall 0.0027%）为 **3,500 倍以上**。
- HitRate@10 = 18.3% 意味着约每 **5.5 个用户有 1 个**能在前 10 命中其测试商品；在 36.8 万候选、每人仅 2–3 个正例的条件下属于正常量级。
- **覆盖率**：BPR 仅 2.24%，推荐高度集中于头部（隐因子在极端稀疏下只能拟合热门商品）；ItemKNN 19.08% 明显更分散；融合后降到 16.23%，是准确率换覆盖的权衡。
- **BPR 始终很弱**：NDCG 仅为 ItemKNN 的 24%，融合权重 0.15 甚至落在搜索网格下界。这与第二步的结论一致——在 99.9968% 稀疏度下，隐因子模型难以学到长尾商品的有效表示。

**RatingPred 为何全为 0**：`μ + b_u + b_i` 中 `b_u` 是每用户常数，不改变商品间排序，因此所有用户得到的是同一个「全局 b_i 最高」的列表，等于一个没有个性化的打分器。这直接证明了**评分预测精度高 ≠ 排序能力强**，排序必须建模成对/相对偏好（BPR）或邻域协同（ItemKNN）。

### 3.5 复现命令

```bash
conda activate amazon-recsys

# 全量 v3（10k 测试用户，约 50 分钟）
python -m src.models.topn.run_topn --users 10000 --tune-users 3000 \
  --ks 200,400,800 --alphas 0.5,0.7,0.85 \
  --half-lifes 2,0 --weight-schemes none,bm25

# 仅验证 ItemKNN（跳过 BPR 与融合，约 40 分钟）
python -m src.models.topn.run_topn --users 10000 --tune-users 3000 \
  --ks 200,400,800 --alphas 0.5,0.7,0.85 \
  --half-lifes 2,0 --weight-schemes none,bm25 --no-bpr

# 快速验证（小样本，几分钟）
python -m src.models.topn.run_topn --sample-users 20000 --users 2000 --tune-users 1000 --quick --no-bpr
```

耗时构成（v3 实测）：初始共现矩阵约 95 秒、BPR 训练约 5 分钟（291.8s）、阶段一 α×k 约 9 分钟、阶段二 3 个 (half_life, 方案) 组合约 25 分钟（每次都要重建 825M nnz 的共现矩阵）、λ 搜索约 5 分钟、最终 6 个模型评测约 4 分钟，加数据加载合计约 50 分钟。

### 3.6 产物

| 路径 | 内容 |
|------|------|
| `experiments/logs/topn/results_topn.csv` | 各模型的 P@10 / R@10 / F1@10 / NDCG@10 / HitRate / 覆盖率 / 耗时；多轮实验按 `run_time` 追加，最新一轮为 v4（19:34 之后的 8 行，含元数据分支） |
| `experiments/logs/topn/topn_v4.log` | v4 全量日志（含元数据共购图分支，负结果） |
| `experiments/logs/topn/topn_v3.log` | v3 全量日志（含两阶段调参轨迹与各阶段耗时） |
| `experiments/logs/topn/topn_full.log` | v1 全量日志（历史对照） |
| `data/processed/Books/meta_graph.npz` | 元数据共购图（also_buy/also_view 加权对称化，覆盖 90.5% 商品） |

### 3.7 小结

- 排序阶段最有效的改动是**非对称余弦**：α 从 0.5 提到 0.85，ItemKNN 的 NDCG 提升 **83%（val 口径）**，它直接压制了 Books 上严重的流行度偏置。
- 其次是 **BM25 用户活跃度归一化**（**+33%**）：修正重度用户对共现统计的支配，与 α 从不同方向解决同一个问题。
- 两个"看起来有用"的改动被判为**无收益**：时间衰减（无加权时 +4%、与 BM25 组合时 −5%，符号不稳定）与分段 λ（val +2.9% 但 test −0.5%，属对 1,606 人调参集的过拟合）。二者均已从最终模型中移除。
- **k 在 200 已饱和**，扩到 800 的收益不足 1%，只会增加内存占用。
- 与评分预测的关键差异：`μ+b_u+b_i` 在排序上完全退化（NDCG=0），而 BPR 虽弱（0.0170）仍显著优于流行度——**排序必须建模相对偏好或邻域协同**。
- 瓶颈已定位：共现信号只能覆盖 **19%** 的商品。尝试用商品元数据共购图补足覆盖，结果为**负**（见 3.8）：also_bought 与评分共现高度冗余且无个性化，MetaItemKNN 比 ItemKNN 低 41%、三路融合低 14%，覆盖率也未提升。当前 Top-N 最优仍是二路 `Hybrid_global`（NDCG@10 = 0.0746）。

### 3.8 尝试：商品元数据共购图（负结果）

**动机**：3.7 指出共现信号只覆盖 19% 的商品。Amazon 元数据中的 `also_bought` / `also_viewed` 提供独立的商品关联图，理论上能补足剩余 81% 商品的邻居证据。

**实现**

- 下载 `meta_Books.json.gz`（826 MB，SNAP 源），解析 `asin` / `also_buy` / `also_view`（[download_meta.py](src/preprocessing/download_meta.py)、[parse_meta.py](src/preprocessing/parse_meta.py)）。
- 元数据 2,370,585 条记录中，ASIN 命中训练/测试商品的 367,982 条；`also_bought` 边保留 10,619,120（52.7%），`also_viewed` 边保留 742,725（50.2%）。
- 构建加权对称共购图 `A = 1.0·buy + 0.5·view`（对称化后 nnz=17,535,362），**覆盖 90.5% 商品**。
- 新增 `MetaItemKNN`（在共购图上跑相同的非对称余弦 + top-k），以及三路融合 `TripleHybrid = BPR + lam1·ItemKNN + lam2·MetaItemKNN`。

**结果（v4，10,000 用户全量，同一协议）**

| 模型 | NDCG@10 | R@10 | 覆盖率 |
|------|---------|------|--------|
| ItemKNN（评分共现，对照） | 0.0707 | 0.0956 | 19.08% |
| MetaItemKNN（元数据共购） | 0.0414 | 0.0633 | 17.34% |
| Hybrid_global（二路，当前最优） | **0.0746** | 0.1024 | 16.23% |
| TripleHybrid（三路融合） | 0.0639 | 0.0968 | 16.79% |

**结论：负结果**。MetaItemKNN 单独比 ItemKNN 低 **41%**，三路融合比二路低 **14%**，覆盖率不升反降（17.34% vs 19.08%）。

**根因**

1. **also_bought 不是正交信号**：它与评分共现图捕获的是同一件事——「商品被共同消费」；区别仅在共现是用户个性化的、共购是跨用户全局的。在 Books 上两者高度冗余，而共购没有个性化维度，质量更低。
2. **α=0.85 惩罚在源侧**：非对称余弦的分母 `‖w_i‖^α · ‖w_j‖^(1-α)` 中，候选侧只有 `‖w_j‖^0.15`，几乎不惩罚热门候选，畅销书依然霸榜，覆盖率起不来。
3. **lam2 单调递减**：三路融合的 lam2 网格 0.25/0.5/1.0 对应 val NDCG 0.0562/0.0497/0.0444，最优卡在下界——MetaItemKNN 的最优权重趋近 0，等价于「这个信号加得越少越好」。

**决策**：不进入最终模型。元数据共购图对评分共现不是有效补充，Top-N 的最优方案仍是 `Hybrid_global`（NDCG@10 = 0.0746）。
