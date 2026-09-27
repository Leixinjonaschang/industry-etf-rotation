"""实验配置：YAML 分层合并 + pydantic 类型校验。

- 实验文件用 `_base_: ../base.yaml` 继承基础配置，只写需要改动的字段；
- 命令行 `--set a.b.c=value` 可临时覆盖任意字段（value 按 YAML 语法解析）；
- 相对路径一律相对于项目根目录（含 pyproject.toml 的目录）解析；
- 环境变量 ROTATION_DATA_DIR / ROTATION_CACHE_DIR / ROTATION_OUTPUT_DIR 可覆盖对应目录。
"""

from __future__ import annotations

import copy
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(_Strict):
    name: str = "default"
    description: str = ""
    seed: int = 42
    # tune = 开发期伪测试（2021–2022）；test = 最终样本外（2023–2025），须显式指定。
    phase: Literal["tune", "test"] = "tune"


class PathsConfig(_Strict):
    data_dir: str = "../../未命名文件夹"
    industry_file: str = "申万一级指数_原始数据.xlsx"
    etf_industry_list: str = "行业指数ETF.xlsx"
    etf_theme_list: str = "主题指数ETF.xlsx"
    etf_daily_dir: str = "后复权"
    etf_whitelist: str = "configs/etf_whitelist.yaml"
    cache_dir: str = "cache"
    output_dir: str = "outputs"


class DataConfig(_Strict):
    exclude_industries: list[str] = Field(default_factory=lambda: ["综合"])
    load_start: date = date(2014, 1, 1)
    sample_start: date = date(2015, 1, 1)
    select_end: date = date(2020, 12, 31)
    tune_start: date = date(2021, 1, 1)
    dev_end: date = date(2022, 12, 31)
    test_start: date = date(2023, 1, 1)
    test_end: date = date(2025, 12, 31)
    benchmark_codes: list[str] = Field(default_factory=lambda: ["510300"])
    # 数据源在该日之前的 ETF 成交额单位为千元，乘以 multiplier 统一为元。
    etf_amount_unit_change: date = date(2026, 1, 1)
    etf_amount_multiplier_before: float = 1000.0

    @model_validator(mode="after")
    def _check_order(self) -> DataConfig:
        chain = [
            ("load_start", self.load_start),
            ("sample_start", self.sample_start),
            ("select_end", self.select_end),
            ("tune_start", self.tune_start),
            ("dev_end", self.dev_end),
            ("test_start", self.test_start),
            ("test_end", self.test_end),
        ]
        for (name_a, a), (name_b, b) in zip(chain, chain[1:]):
            if not a < b:
                raise ValueError(f"日期顺序错误：{name_a}={a} 应早于 {name_b}={b}")
        return self


class LabelConfig(_Strict):
    horizon: int = Field(5, ge=1, le=60)
    type: Literal["raw", "excess"] = "excess"
    # oo: Open[t+h+1]/Open[t+1]-1（与 t+1 开盘成交对齐）；cc: Close[t+h]/Close[t]-1
    price: Literal["oo", "cc"] = "oo"


class SelectionConfig(_Strict):
    enabled: bool = True
    min_abs_t: float = 2.0
    require_sign_stability: bool = True
    corr_threshold: float = 0.8
    max_factors: int = 15
    min_factors: int = 8
    min_per_category: int = 1
    tree_weight: float = 0.2
    tree_max_samples: int = 30_000


class FeatureConfig(_Strict):
    categories: list[str] = Field(
        default_factory=lambda: ["momentum", "trend", "oscillator", "volume", "volatility"]
    )
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    use_market: bool = True
    normalize: Literal["none", "ts", "cs", "ts+cs", "cs_rank"] = "cs"
    ts_window: int = 60
    market_ts_window: int = 250
    winsor_mad: float = 5.0
    clip: float = 5.0
    ffill_limit: int = 5
    min_valid_ratio: float = 0.9
    pca_components: int = 0  # 0 表示不用 PCA；>0 时在每折训练集上拟合
    selection: SelectionConfig = Field(default_factory=SelectionConfig)


class SplitConfig(_Strict):
    mode: Literal["expanding", "rolling"] = "expanding"
    train_window: int = 750  # 仅 rolling 模式使用
    retrain_every: int = 60
    val_len: int = 250
    min_train: int = 500


class ModelConfig(_Strict):
    # zero | hist_mean | momentum | ridge | gbdt | lstm | gru | transformer
    name: str = "lstm"
    params: dict[str, Any] = Field(default_factory=dict)


class TrainConfig(_Strict):
    device: str = "auto"
    loss: str = "mse+ic"
    ic_weight: float = 0.5
    huber_delta: float = 1.0
    listnet_temperature: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    min_epochs: int = 5
    patience: int = 8
    lr_patience: int = 3
    lr_factor: float = 0.5
    batch_dates: int = 32
    grad_clip: float = 1.0
    n_seeds: int = 5
    early_stop_metric: Literal["rank_ic", "loss"] = "rank_ic"
    num_threads: int = 0  # 0 = PyTorch 默认线程数
    save_checkpoints: bool = False


class StrategyConfig(_Strict):
    top_n: int = 5
    weighting: Literal["equal", "rank"] = "equal"
    buffer: int = 0  # 已持有行业排名仍在 top_n+buffer 内则继续持有


class MappingConfig(_Strict):
    enabled: bool = True
    corr_window: int = 120
    min_corr_obs: int = 110
    min_listing_days: int = 120
    liq_window: int = 20
    min_amount: float = 1.0e7  # 近 liq_window 日日均成交额下限（元）
    min_valid_days: int = 18
    tau_p: float = 0.80  # 原始日收益 Pearson 相关下限
    tau_ex: float = 0.50  # 超额（扣除 30 行业等权）日收益相关下限
    weights: tuple[float, float, float] = (0.25, 0.25, 0.5)  # (pearson, spearman, excess)
    tier_priority: bool = True  # 行业 ETF 层优先于主题 ETF 层
    retention_bonus: float = 0.03
    fallback: Literal["next_rank", "cash", "broad"] = "next_rank"
    max_fallback_rank: int = 12
    broad_code: str = "510300"


class BacktestConfig(_Strict):
    cost_bps: float = 5.0  # 单边：佣金 + 滑点（ETF 免印花税）
    risk_free: float = 0.015
    n_random: int = 1000
    momentum_lookback: int = 20


class Config(_Strict):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    _root: Path = PrivateAttr(default_factory=Path.cwd)
    _source: Path | None = PrivateAttr(default=None)

    @field_validator("experiment")
    @classmethod
    def _name_ok(cls, value: ExperimentConfig) -> ExperimentConfig:
        if not value.name or "/" in value.name:
            raise ValueError("experiment.name 不能为空且不能包含 '/'")
        return value

    # ---------- 路径 ----------
    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, value: str, env_var: str | None = None) -> Path:
        if env_var and os.environ.get(env_var):
            return Path(os.environ[env_var]).expanduser().resolve()
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self._root / path).resolve()

    @property
    def data_dir(self) -> Path:
        return self._resolve(self.paths.data_dir, "ROTATION_DATA_DIR")

    @property
    def cache_dir(self) -> Path:
        return self._resolve(self.paths.cache_dir, "ROTATION_CACHE_DIR")

    @property
    def output_root(self) -> Path:
        return self._resolve(self.paths.output_dir, "ROTATION_OUTPUT_DIR")

    @property
    def whitelist_path(self) -> Path:
        return self._resolve(self.paths.etf_whitelist)

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.experiment.name / self.experiment.phase

    # ---------- 研究区间 ----------
    @property
    def research_end(self) -> pd.Timestamp:
        """当前阶段可见数据的最后一天：tune 阶段看不到 2023 年及以后的任何价格。"""
        end = self.data.dev_end if self.experiment.phase == "tune" else self.data.test_end
        return pd.Timestamp(end)

    @property
    def eval_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        if self.experiment.phase == "tune":
            return pd.Timestamp(self.data.tune_start), pd.Timestamp(self.data.dev_end)
        return pd.Timestamp(self.data.test_start), pd.Timestamp(self.data.test_end)

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并（含 model.params），override 优先。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_layered(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"配置继承出现循环：{path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    base = _load_layered((path.parent / base_ref), seen)
    return _deep_merge(base, raw)


def apply_override(raw: dict[str, Any], expression: str) -> None:
    """`a.b.c=value`，value 按 YAML 解析（数字、布尔、列表、日期均可）。"""
    if "=" not in expression:
        raise ValueError(f"覆盖参数格式应为 key=value：{expression}")
    key, text = expression.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ValueError(f"覆盖参数缺少键名：{expression}")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"无法覆盖 {key}：{part} 不是字典")
    node[parts[-1]] = yaml.safe_load(text)


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    root: Path | None = None,
) -> Config:
    raw: dict[str, Any] = {}
    source = None
    if path is not None:
        source = Path(path)
        raw = _load_layered(source)
    for expression in overrides or []:
        apply_override(raw, expression)
    cfg = Config.model_validate(raw)
    cfg._root = root or find_project_root(source.parent if source else Path.cwd())
    cfg._source = source
    return cfg


def load_default_config(start: Path | None = None) -> Config:
    """项目根目录下的 configs/base.yaml（不存在时使用内置默认值）。"""
    root = find_project_root(start or Path.cwd())
    base = root / "configs" / "base.yaml"
    return load_config(base if base.exists() else None, root=root)


def config_from_dict(raw: dict[str, Any], root: Path | None = None) -> Config:
    cfg = Config.model_validate(raw)
    cfg._root = root or Path.cwd()
    return cfg
