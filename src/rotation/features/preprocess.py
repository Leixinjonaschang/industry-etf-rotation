"""因果预处理：在 [日期 T, 行业 N, 因子 F] 立方体上操作。

每一步只使用 t 日及以前的数据（时序滚动），或只使用 t 日当天的截面数据：
1. 行业内向前填充（有上限）；
2. 可选时序标准化：行业内滚动 z-score；
3. 可选截面处理：当日 MAD 缩尾 + z-score，或截面排名；
4. 截断到 [-clip, clip]，剩余缺失填 0（≈ 截面中位数），并给出有效性掩码。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_EPS = 1e-12


def to_cube(frames: list[pd.DataFrame], dates: pd.DatetimeIndex, columns: list[str]) -> np.ndarray:
    cube = np.stack(
        [f.reindex(index=dates, columns=columns).to_numpy(dtype=np.float64) for f in frames],
        axis=-1,
    )
    cube[~np.isfinite(cube)] = np.nan
    return cube


def ffill_cube(cube: np.ndarray, limit: int) -> np.ndarray:
    t, n, f = cube.shape
    frame = pd.DataFrame(cube.reshape(t, n * f))
    return frame.ffill(limit=limit).to_numpy().reshape(t, n, f)


def ts_zscore(cube: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    """沿时间轴滚动 z-score（每个 行业×因子 独立）。"""
    shape = cube.shape
    frame = pd.DataFrame(cube.reshape(shape[0], -1))
    min_periods = min_periods or window
    mean = frame.rolling(window, min_periods=min_periods).mean()
    std = frame.rolling(window, min_periods=min_periods).std()
    z = (frame - mean) / (std + _EPS)
    z = z.where(std > _EPS)
    return z.to_numpy().reshape(shape)


def cs_winsor_zscore(cube: np.ndarray, mad_k: float) -> np.ndarray:
    """当日截面：中位数 ± k·1.4826·MAD 缩尾，再做 z-score。"""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(cube, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(cube - median), axis=1, keepdims=True) * 1.4826
        radius = mad_k * mad
        clipped = np.where(radius > 0, np.clip(cube, median - radius, median + radius), cube)
        mean = np.nanmean(clipped, axis=1, keepdims=True)
        std = np.nanstd(clipped, axis=1, keepdims=True)
        z = (clipped - mean) / (std + _EPS)
        z = np.where(std > _EPS, z, np.where(np.isnan(clipped), np.nan, 0.0))
    return z


def cs_rank(cube: np.ndarray) -> np.ndarray:
    """当日截面排名映射到 [-1, 1]（缺失保持缺失）。"""
    from scipy.stats import rankdata

    ranks = rankdata(cube, axis=1, nan_policy="omit")
    count = np.sum(np.isfinite(cube), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = 2.0 * (ranks - 1.0) / np.maximum(count - 1.0, 1.0) - 1.0
    return np.where(np.isfinite(cube), scaled, np.nan)


def preprocess_cube(
    cube: np.ndarray,
    normalize: str,
    ts_window: int,
    winsor_mad: float,
    clip: float,
    ffill_limit: int,
    min_valid_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (X float32, valid[T,N])。"""
    x = ffill_cube(cube, ffill_limit) if ffill_limit > 0 else cube.copy()
    finite_before = np.isfinite(x)
    if normalize in ("ts", "ts+cs"):
        x = ts_zscore(x, ts_window)
    if normalize in ("cs", "ts+cs"):
        x = cs_winsor_zscore(x, winsor_mad)
    elif normalize == "cs_rank":
        x = cs_rank(x)
    elif normalize == "none":
        pass
    elif normalize != "ts":
        raise ValueError(f"未知标准化方式：{normalize}")
    finite_after = np.isfinite(x)
    valid_ratio = (finite_before & finite_after).mean(axis=2)
    valid = valid_ratio >= min_valid_ratio
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -clip, clip)
    return x.astype(np.float32), valid


def preprocess_market(
    frames: list[pd.Series], dates: pd.DatetimeIndex, window: int, clip: float
) -> np.ndarray:
    """市场特征：时序滚动 z-score（至少 window/2 个观测）。"""
    if not frames:
        return np.zeros((len(dates), 0), dtype=np.float32)
    matrix = pd.concat([f.reindex(dates).astype(float) for f in frames], axis=1)
    matrix = matrix.replace([np.inf, -np.inf], np.nan).ffill(limit=5)
    mean = matrix.rolling(window, min_periods=max(window // 2, 20)).mean()
    std = matrix.rolling(window, min_periods=max(window // 2, 20)).std()
    z = ((matrix - mean) / (std + _EPS)).where(std > _EPS)
    return np.clip(np.nan_to_num(z.to_numpy(), nan=0.0), -clip, clip).astype(np.float32)
