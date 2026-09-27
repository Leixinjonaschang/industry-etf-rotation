# industry-etf-rotation

用 LSTM 预测 30 个申万一级行业未来 h 日收益，每期选前 5 个行业；再按"过去 120 日收益相关性 + 流动性"动态映射到行业/主题 ETF，做可交易的轮动回测。

```
行业行情 ─► 技术因子 ─► 因子筛选 ─► LSTM(walk-forward) ─► 前 5 行业 ─► ETF 映射 ─► 回测 / 报告
```

## 快速开始（macOS）

```bash
brew install uv                 # 或 curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                         # 自动安装 Python 3.12 与依赖，并生成 uv.lock（建议提交）
uv run rotation device --bench  # 查看 MPS 是否可用，并实测 CPU / MPS 训练速度
uv run rotation run -c configs/experiments/lstm_excess_h5.yaml
```

结果在 `outputs/lstm_excess_h5/tune/summary.md`（同目录下还有 `summary.xlsx` 和 `figures/`）。

**数据位置**：默认读取 `../../未命名文件夹/`（即 `zheng_work/未命名文件夹`）。换位置：

```bash
export ROTATION_DATA_DIR=/path/to/数据目录
```

## GPU / 设备

- `train.device: auto` 的顺序是 CUDA → MPS → CPU，Apple Silicon 上默认用 MPS。
- 模型很小，MPS 不一定比 CPU 快。先跑 `rotation device --bench`，再按建议设置：
  `--set train.device=cpu`，或直接改 `configs/base.yaml`。
- 已自动设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`：MPS 不支持的算子会回退到 CPU。
- 一个实验约 13 折 × 5 个种子。每折结果单独缓存，中断后重跑同一条命令即可续跑。

## 研究流程（别让测试期"泄露"进设计）

| 区间 | 用途 | 怎么跑 |
|---|---|---|
| 2014 | 指标预热 | 自动 |
| 2015–2020 | 因子筛选（IC / t 值 / 稳定性 / 去相关） | 自动（`features` 阶段） |
| 2021–2022 | **tune**：调参，选标签、模型和映射门槛 | 默认阶段，`--phase tune` |
| 2023–2025 | **test**：设定冻结后只跑一次 | `--phase test`（需确认，记入 `outputs/TEST_RUNS.md`） |

tune 阶段会把 2023 年以后的所有价格截掉，代码层面看不到。

**时间约定**：t 日收盘出信号 → t+1 开盘成交 → 持有 h 日。标签默认是 `Open[t+h+1]/Open[t+1]−1`，与成交时点对齐。训练样本只用起点前已可观测的标签（purge）。

## 常用命令

| 目的 | 命令 |
|---|---|
| 跑单个实验 | `uv run rotation run -c configs/experiments/<名>.yaml` |
| 只跑部分阶段 | `... --stages backtest,report`（可选 data, features, predict, evaluate, backtest, report） |
| 临时改参数 | `... --set label.horizon=10 --set model.params.hidden=32` |
| LSTM 调参（tune） | `uv run rotation tune --grid configs/tuning/lstm_grid.yaml` |
| 标签对比 E1 | `uv run rotation matrix --file configs/matrix/labels.yaml` |
| 模型对比 E2 | `uv run rotation matrix --file configs/matrix/models.yaml` |
| 对比已完成实验 | `uv run rotation compare -e lstm_excess_h5 -e momentum_h5` |
| 某日映射全景 | `uv run rotation mapping-table -c configs/base.yaml --date 2022-12-30` |
| 列出全部因子 | `uv run rotation factors` |
| 测试 | `uv run pytest` |

## 配置

- `configs/base.yaml`：全部默认参数，都有注释。
- `configs/experiments/*.yaml`：用 `_base_: ../base.yaml` 继承，只写改动的部分。
- `configs/etf_whitelist.yaml`：行业 → ETF 的语义候选池（关键词规则），冻结于开发期末。

已有的实验：`lstm_excess_h5`（主线）、`lstm_raw_h5`、`gru_*`、`transformer_*`、`gbdt_*`、`ridge_*`、`momentum_h5`。

## 输出（`outputs/<实验>/<tune|test>/`）

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.xlsx` | 关卡判定（G1–G4）、预测指标、策略绩效、映射损耗 |
| `prediction_metrics.csv` | RankIC、HAC t、前 5 命中率、分组收益、R²_OOS |
| `navs.csv` / `strategy_metrics.csv` | ETF 策略、行业指数策略、等权、沪深300、动量的净值与指标 |
| `mapping_*.csv` | 每期映射明细、失败原因、覆盖率、年末全景表 |
| `factor_selection_*.{json,csv}` | 入选因子及其 IC 统计 |
| `folds/` | 逐折预测与训练日志（缓存） |

## 代码结构（`src/rotation/`）

```
data/        读数（行业 Excel、ETF 名单与后复权 CSV）、单位修正、质量报告、缓存
features/    指标 → 因子注册 → 因果预处理 → 开发期因子筛选
labels.py    raw / excess × oo / cc 标签及可观测日
dataset/     Panel 面板、walk-forward 划分（purge）
models/      zero / hist_mean / momentum / ridge / gbdt / lstm / gru / transformer
evaluation/  排序与回归指标、HAC / DM 检验
strategy/    调仓信号（前 N + 缓冲）
mapping/     白名单 → 流动性过滤 → 相关性打分 → 匈牙利唯一分配 → 顺延兜底
backtest/    逐日回测引擎、基准、随机选行业检验、映射损耗分解
reporting/   图表、Excel、多实验对比
pipeline/    实验编排、调参、批量运行
```

## 常见问题

- **`OMP: Error #15`**（macOS 上 OpenMP 重复加载）：先 `export KMP_DUPLICATE_LIB_OK=TRUE`，再运行。
- **想用 LightGBM**：先 `brew install libomp && uv sync --extra lightgbm`，再加 `--set model.params.backend=lightgbm`。默认的 GBDT 用 scikit-learn，不需要它。
- **数据更新后结果没变**：缓存按源文件的修改时间自动失效；也可以直接删掉 `cache/`。
