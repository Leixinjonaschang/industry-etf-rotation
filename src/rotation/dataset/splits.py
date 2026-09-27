"""Walk-forward 样本划分（带 purge）。

对每个重训起点 o（一个测试区块的第一天）：
- 候选样本 s：s ≥ train_start，且标签可观测 known_pos[s] ≤ o；
- 验证集：候选中最近的 val_len 个日期；
- 训练集：known_pos[s] ≤ 验证集首日（训练标签区间不与验证期重叠）；rolling 模式只取最近 train_window 个；
- 测试集：[o, o + retrain_every) 内的所有交易日（每天都出预测，评价时再按调仓日抽样）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rotation.config import SplitConfig
from rotation.dataset.panel import Panel


@dataclass(frozen=True)
class Fold:
    index: int
    origin: pd.Timestamp
    train_pos: np.ndarray
    val_pos: np.ndarray
    test_pos: np.ndarray

    def summary(self, dates: pd.DatetimeIndex) -> dict:
        def span(pos: np.ndarray) -> tuple[str, str, int]:
            if len(pos) == 0:
                return ("", "", 0)
            return (str(dates[pos[0]].date()), str(dates[pos[-1]].date()), int(len(pos)))

        tr, va, te = span(self.train_pos), span(self.val_pos), span(self.test_pos)
        return {
            "fold": self.index,
            "origin": str(self.origin.date()),
            "train_start": tr[0],
            "train_end": tr[1],
            "train_days": tr[2],
            "val_start": va[0],
            "val_end": va[1],
            "val_days": va[2],
            "test_start": te[0],
            "test_end": te[1],
            "test_days": te[2],
        }


def make_folds(
    panel: Panel,
    cfg: SplitConfig,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    train_start: pd.Timestamp,
    min_history: int = 0,
) -> list[Fold]:
    dates = panel.dates
    eval_mask = (dates >= eval_start) & (dates <= eval_end)
    eval_pos = np.flatnonzero(eval_mask)
    if len(eval_pos) == 0:
        raise ValueError(f"评价区间 {eval_start.date()} ~ {eval_end.date()} 内没有交易日")
    start_pos = max(int(dates.searchsorted(train_start, side="left")), min_history)
    enough_labels = np.isfinite(panel.y).sum(axis=1) >= max(2, panel.n_industries // 2)

    folds: list[Fold] = []
    for index, block_start in enumerate(range(0, len(eval_pos), cfg.retrain_every)):
        test_pos = eval_pos[block_start : block_start + cfg.retrain_every]
        origin = int(test_pos[0])
        candidates = np.arange(start_pos, origin)
        candidates = candidates[(panel.known_pos[candidates] <= origin) & enough_labels[candidates]]
        if cfg.val_len > 0:
            val_pos = candidates[-cfg.val_len :]
            train_pos = (
                candidates[panel.known_pos[candidates] <= val_pos[0]]
                if len(val_pos)
                else candidates
            )
        else:
            val_pos = candidates[:0]
            train_pos = candidates
        if cfg.mode == "rolling":
            train_pos = train_pos[-cfg.train_window :]
        if len(train_pos) < cfg.min_train:
            raise ValueError(
                f"第 {index} 折训练样本只有 {len(train_pos)} 天（< min_train={cfg.min_train}），"
                "请调整 split 配置或 sample_start"
            )
        folds.append(
            Fold(
                index=index,
                origin=dates[origin],
                train_pos=train_pos,
                val_pos=val_pos,
                test_pos=test_pos,
            )
        )
    return folds
