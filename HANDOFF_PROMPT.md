# 交接给 Claude Code 的任务说明

你在我的 macOS（Apple Silicon）上工作，项目目录是当前目录 `industry_etf_rotation/`（git 仓库，uv 管理环境）。请先读 `README.md`、`configs/base.yaml` 和上级目录的 `../../项目实现规划.md`，了解项目后再动手。

## 项目是什么

金融专硕毕业论文：用 LSTM 预测 30 个申万一级行业（剔除"综合"）未来 5 日收益，每期选前 5 个行业，再按"过去 120 日收益相关性 + 流动性"动态映射到行业/主题 ETF，做可交易的轮动回测。**最关键的目标是模型对行业未来收益的预测（排序）能力要足够好。**

代码已经完整可用：数据 → 因子 → 标签 → walk-forward 训练 → 评价 → 信号 → ETF 映射 → 回测 → 报告。常用命令见 README。

## 必须遵守的研究规则

- 区间划分：2015–2020 只用于因子筛选；2021–2022 是开发期（`--phase tune`，默认），用于调参和各种比较；2023–2025 是测试期（`--phase test`），**只能在所有设定冻结并提交 git 之后跑一次**，跑完不得再根据测试期结果修改任何设定。
- 所有选择（因子、超参数、标签、模型、映射门槛）都只依据开发期结果。
- 训练用 GPU：`train.device: auto` 会自动用 MPS。长任务用后台方式运行并定期查看进度，用 `caffeinate -i` 防止睡眠。逐折和逐种子都有缓存（`outputs/_folds/`），中断后重跑同一命令会续跑。
- 每完成一个有意义的步骤就 `git commit`，提交信息用中文写清楚改了什么、为什么。
- 改代码后跑 `uv run pytest`（目前 100 个测试全过），保持 `uv run ruff check src tests` 通过。

## 目前的进展与结论（开发期 2021–2022）

- 对照组（旧的 8 个筛选因子）：岭回归 RankIC 0.038、动量 0.015、GBDT 0.010，均不显著。
- LSTM 第一轮调参（12 组）：hidden=32 在全部 6 组对比中优于 64；seq_len、lr 影响不大。
- LSTM 第二轮调参（16 组，结果在 `outputs/_tuning/lstm_grid2/trials.csv`）：
  - **不做因子筛选、直接用全部 52 个因子**明显更好（前 7 名全是这种）；
  - 最优约为 hidden=32、dropout=0.4、全部因子：调仓日 RankIC ≈ 0.05，HAC t ≈ 2，两年都为正；
  - 损失函数 `ic`（0.054）与 `mse+ic`（0.048）差不多，主线用 `mse+ic`（输出仍是收益率）。
- 岭回归用全部因子：RankIC 0.059（t=2.38），与 LSTM 基本打平；GBDT 用全部因子仍很弱。
- 试过给 LSTM 加线性残差通路（`model.params.linear_skip`），变差（0.023），不采用。
- 这些结论已写入 `configs/base.yaml` 并提交（`features.selection.enabled: false`，hidden 32，dropout 0.4）。

## 你要做的事

### 1. 按我的要求重写因子筛选（`src/rotation/features/selection.py`）

只用 2015–2020 的样本（标签在 2020-12-31 前可观测），步骤：

1. **重要性评分**：对全部候选因子分别训练随机森林和 **XGBoost**，取各自的特征重要性，再加上截面 RankIC 的绝对值，三者按排名合成综合分。不要再用"t≥2 且方向稳定"这种硬门槛一刀切。
2. **相关性筛选**：把 |相关系数| ≥ 0.8 的因子聚成簇。
   - 簇内只有一个因子重要性高：保留它，去掉其余；
   - 簇内有多个因子重要性都高：对这一簇做 **PCA**，用第一主成分合成一个新因子替代它们。PCA 只在 2015–2020 上拟合，载荷方向要让合成因子与标签正相关。
3. 去掉综合分排名最后的因子（阈值做成配置项）。
4. 输出论文可用的因子表：每个因子的 RF 重要性、XGBoost 重要性、RankIC、所属簇、是否入选及原因、PCA 载荷与解释方差。
5. 注意事项：
   - 把 `xgboost` 加进 `pyproject.toml` 依赖；macOS 上如缺 OpenMP 可能需要 `brew install libomp`。
   - XGBoost 与 torch 同进程在 macOS 上可能因 OpenMP 冲突崩溃，建议把 XGBoost 放在子进程里算重要性。
   - PCA 合成因子必须是因果的：只用拟合好的载荷去变换各日数据，不能用未来数据重新拟合。
   - 为新逻辑补测试，包括防泄露测试。

### 2. 在开发期比较三种特征方案

分别用 LSTM（5 个种子）和岭回归，比较以下三种特征方案，按调仓日 RankIC、HAC t 值、分年度稳定性决定主线用哪一种：
- 新筛选方法；
- 全部 52 个因子（当前默认）；
- 旧的 8 个因子。

### 3. 用选定的特征方案跑开发期主实验

```bash
uv run rotation matrix --file configs/matrix/models.yaml                # LSTM/GRU/Transformer/GBDT/岭回归/动量
uv run rotation run -c configs/experiments/lstm_raw_h5.yaml             # 原始收益 vs 超额收益标签
```
根据开发期结果确定主线标签和主模型。论文主角是 LSTM：如果岭回归依然更好，如实记录，不要为了让 LSTM 胜出而在开发期反复调参。

### 4. 冻结设定

把最终选择写进配置，跑测试，更新 README 中与结论相关的部分，然后提交 git，提交信息写"冻结设定"。

### 5. 测试期只跑一次

```bash
caffeinate -i uv run rotation matrix --file configs/matrix/models.yaml --phase test --yes
```
可选：`configs/matrix/robustness.yaml` 的稳健性实验也在测试期跑一次（它只改一个因素，大部分会复用缓存）。

### 6. 写结果报告

在项目根目录写 `RESULTS.md`，包括：
- 开发期的所有尝试和选择依据，以及尝试过的配置数量，以便论文如实说明；
- 测试期结果：预测指标、策略绩效、映射损耗、关卡判定（G1–G4）；
- 局限性。

报告要如实写：结果不好就写不好，不要挑选有利的数字。

每完成一步简要告诉我结果；遇到需要我拍板的研究决策（例如结果与预期明显不符）先停下来问我。
