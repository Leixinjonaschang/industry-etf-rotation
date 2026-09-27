"""预测标签与其"可观测日"。

- price="oo"：y_t = Open[t+h+1] / Open[t+1] − 1，t+h+1 日开盘时可观测（与 t+1 开盘成交对齐）；
- price="cc"：y_t = Close[t+h] / Close[t] − 1，t+h 日收盘时可观测；
- type="excess"：再减去当日 30 个行业标签的等权均值（截面超额收益）。

known_pos[t] 是标签可观测的交易日位置。训练起点 o 只能使用 known_pos ≤ o 的样本。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Labels:
    y: np.ndarray  # [T, N] 训练目标（raw 或 excess）
    y_raw: np.ndarray  # [T, N] 同价格口径的原始收益
    known_pos: np.ndarray  # [T]
    horizon: int
    kind: str
    price: str


def make_labels(
    open_: pd.DataFrame, close: pd.DataFrame, horizon: int, kind: str = "excess", price: str = "oo"
) -> Labels:
    if horizon < 1:
        raise ValueError("horizon 必须 ≥ 1")
    t = len(close)
    if price == "oo":
        raw = open_.shift(-(horizon + 1)) / open_.shift(-1) - 1.0
        known = np.arange(t) + horizon + 1
    elif price == "cc":
        raw = close.shift(-horizon) / close - 1.0
        known = np.arange(t) + horizon
    else:
        raise ValueError(f"未知价格口径：{price}")
    y_raw = raw.to_numpy(dtype=np.float64, copy=True)
    y_raw[~np.isfinite(y_raw)] = np.nan
    if kind == "raw":
        y = y_raw.copy()
    elif kind == "excess":
        count = np.isfinite(y_raw).sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            mean = np.nansum(y_raw, axis=1, keepdims=True) / np.maximum(count, 1)
        y = np.where(count >= max(2, y_raw.shape[1] // 2), y_raw - mean, np.nan)
    else:
        raise ValueError(f"未知标签类型：{kind}")
    return Labels(
        y=y.astype(np.float32),
        y_raw=y_raw.astype(np.float32),
        known_pos=known.astype(np.int64),
        horizon=horizon,
        kind=kind,
        price=price,
    )
