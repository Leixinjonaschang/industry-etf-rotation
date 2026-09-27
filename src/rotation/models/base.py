"""模型统一接口：流水线只依赖 fit / predict，新增模型无需改动流水线。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rotation.config import TrainConfig
from rotation.dataset.panel import Panel
from rotation.evaluation.ranking import daily_rank_ic


class BaseModel(ABC):
    name: str = "base"
    #: 输出是否为收益率量纲（False 表示只是排序分数，回归指标不适用）
    produces_returns: bool = True
    #: 神经网络模型需要的历史窗口长度（决定最早可用样本）
    history: int = 1

    def __init__(self, params: dict[str, Any], train_cfg: TrainConfig, seed: int = 42):
        self.params = dict(params)
        self.train_cfg = train_cfg
        self.seed = seed

    @abstractmethod
    def fit(self, panel: Panel, train_pos: np.ndarray, val_pos: np.ndarray) -> dict[str, Any]:
        """训练；返回训练日志（会写入 fold 日志）。"""

    @abstractmethod
    def predict(self, panel: Panel, pos: np.ndarray) -> np.ndarray:
        """返回 [len(pos), N] 的预测值。"""

    @staticmethod
    def validation_rank_ic(panel: Panel, pos: np.ndarray, pred: np.ndarray) -> float:
        if len(pos) == 0:
            return float("nan")
        ic = daily_rank_ic(pred, panel.y[pos], panel.target_mask(pos))
        return float(np.nanmean(ic)) if np.isfinite(ic).any() else float("nan")
