"""朴素基准：零收益、训练集均值、行业动量。"""

from __future__ import annotations

from typing import Any

import numpy as np

from rotation.dataset.panel import Panel
from rotation.models.base import BaseModel


class ZeroModel(BaseModel):
    """预测收益恒为 0（回归指标的最朴素基准）。"""

    name = "zero"

    def fit(self, panel: Panel, train_pos: np.ndarray, val_pos: np.ndarray) -> dict[str, Any]:
        return {}

    def predict(self, panel: Panel, pos: np.ndarray) -> np.ndarray:
        return np.zeros((len(pos), panel.n_industries), dtype=np.float32)


class HistMeanModel(BaseModel):
    """预测收益 = 训练集标签均值（R²_OOS 的基准；截面上无区分度）。"""

    name = "hist_mean"

    def fit(self, panel: Panel, train_pos: np.ndarray, val_pos: np.ndarray) -> dict[str, Any]:
        y = panel.y[train_pos][panel.target_mask(train_pos)]
        self.mean_ = float(np.mean(y)) if len(y) else 0.0
        return {"train_mean": self.mean_}

    def predict(self, panel: Panel, pos: np.ndarray) -> np.ndarray:
        return np.full((len(pos), panel.n_industries), self.mean_, dtype=np.float32)


class MomentumModel(BaseModel):
    """行业动量：分数 = close[t−skip] / close[t−lookback] − 1。"""

    name = "momentum"
    produces_returns = False

    def __init__(self, params: dict[str, Any], train_cfg, seed: int = 42):
        super().__init__(params, train_cfg, seed)
        self.lookback = int(self.params.get("lookback", 20))
        self.skip = int(self.params.get("skip", 0))
        if self.lookback <= self.skip:
            raise ValueError("momentum.lookback 必须大于 skip")

    def fit(self, panel: Panel, train_pos: np.ndarray, val_pos: np.ndarray) -> dict[str, Any]:
        return {"lookback": self.lookback, "skip": self.skip}

    def predict(self, panel: Panel, pos: np.ndarray) -> np.ndarray:
        close = panel.close
        recent = close[np.maximum(pos - self.skip, 0)]
        past = close[np.maximum(pos - self.lookback, 0)]
        with np.errstate(invalid="ignore", divide="ignore"):
            score = recent / past - 1.0
        score[pos - self.lookback < 0] = np.nan
        return score.astype(np.float32)
