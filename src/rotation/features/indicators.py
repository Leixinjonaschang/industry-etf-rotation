"""技术指标纯函数。

输入输出均为 日期 × 行业 的 DataFrame；所有计算只使用当日及以前的数据（因果）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_EPS = 1e-12


def pct_change(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return frame / frame.shift(periods) - 1.0


def sma(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=window).mean()


def ema(frame: pd.DataFrame, span: int) -> pd.DataFrame:
    return frame.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Wilder 平滑（RSI/ATR/ADX 的标准做法）。"""
    return frame.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _rolling_apply_weights(frame: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """窗口内加权和（窗口内任一缺失则为 NaN），向量化实现。"""
    window = len(weights)
    values = frame.to_numpy(dtype=float)
    out = np.full_like(values, np.nan)
    if len(values) >= window:
        view = sliding_window_view(values, window, axis=0)  # [T-w+1, N, w]
        out[window - 1 :] = view @ weights
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def wma(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    weights = np.arange(1, window + 1, dtype=float)
    return _rolling_apply_weights(frame, weights / weights.sum())


def hma(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Hull 移动平均：wma(2·wma(n/2) − wma(n), √n)。"""
    half = max(window // 2, 1)
    root = max(int(np.sqrt(window)), 1)
    return wma(2.0 * wma(frame, half) - wma(frame, window), root)


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    hl = (high - low).to_numpy()
    hc = (high - prev).abs().to_numpy()
    lc = (low - prev).abs().to_numpy()
    tr = np.fmax(np.fmax(hl, hc), lc)
    return pd.DataFrame(tr, index=close.index, columns=close.columns)


def atr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    return wilder(true_range(high, low, close), window)


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), window)
    loss = wilder((-delta).clip(lower=0), window)
    return 100.0 - 100.0 / (1.0 + gain / (loss + _EPS))


def dmi(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """趋向指标：返回 (+DI, −DI, ADX)。"""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    atr_ = atr(high, low, close, window)
    plus_di = 100.0 * wilder(plus_dm, window) / (atr_ + _EPS)
    minus_di = 100.0 * wilder(minus_dm, window) / (atr_ + _EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + _EPS)
    return plus_di, minus_di, wilder(dx, window)


def adx(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    return dmi(high, low, close, window)[2]


def trix(close: pd.DataFrame, span: int = 15) -> pd.DataFrame:
    triple = ema(ema(ema(close, span), span), span)
    return pct_change(triple, 1)


def kdj(
    close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, window: int = 9
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lowest = low.rolling(window, min_periods=window).min()
    highest = high.rolling(window, min_periods=window).max()
    rsv = 100.0 * (close - lowest) / (highest - lowest + _EPS)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d, 3.0 * k - 2.0 * d


def williams_r(
    close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    highest = high.rolling(window, min_periods=window).max()
    lowest = low.rolling(window, min_periods=window).min()
    return -100.0 * (highest - close) / (highest - lowest + _EPS)


def cci(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    typical = (high + low + close) / 3.0
    mean = typical.rolling(window, min_periods=window).mean()
    values = typical.to_numpy(dtype=float)
    mad = np.full_like(values, np.nan)
    if len(values) >= window:
        view = sliding_window_view(values, window, axis=0)  # [T-w+1, N, w]
        mad[window - 1 :] = np.abs(view - view.mean(axis=-1, keepdims=True)).mean(axis=-1)
    mad_frame = pd.DataFrame(mad, index=close.index, columns=close.columns)
    return (typical - mean) / (0.015 * mad_frame + _EPS)


def cmo(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(window, min_periods=window).sum()
    losses = (-delta).clip(lower=0).rolling(window, min_periods=window).sum()
    return 100.0 * (gains - losses) / (gains + losses + _EPS)


def money_flow_multiplier(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame
) -> pd.DataFrame:
    return ((2.0 * close - high - low) / (high - low + _EPS)).clip(-1.0, 1.0)


def cmf(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    flow = money_flow_multiplier(high, low, close) * volume
    return flow.rolling(window, min_periods=window).sum() / (
        volume.rolling(window, min_periods=window).sum() + _EPS
    )


def obv_change(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """OBV 的 window 日变化 / window 日总成交量（把累积量转换为平稳量）。"""
    signed = np.sign(close.diff()) * volume
    return signed.rolling(window, min_periods=window).sum() / (
        volume.rolling(window, min_periods=window).sum() + _EPS
    )


def bollinger(
    close: pd.DataFrame, window: int = 20, k: float = 2.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (%B 位置, 带宽)。"""
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper, lower = mid + k * std, mid - k * std
    position = (close - lower) / (upper - lower + _EPS)
    width = (upper - lower) / (mid + _EPS)
    return position, width


def rolling_zscore(
    frame: pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.DataFrame:
    min_periods = min_periods or window
    mean = frame.rolling(window, min_periods=min_periods).mean()
    std = frame.rolling(window, min_periods=min_periods).std()
    return (frame - mean) / (std + _EPS)


def rolling_beta_resid_vol(
    ret: pd.DataFrame, market: pd.Series, window: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """滚动 β 与特质波动率（行业日收益对市场日收益回归的残差标准差）。"""
    m = pd.DataFrame(
        np.repeat(market.to_numpy()[:, None], ret.shape[1], axis=1),
        index=ret.index,
        columns=ret.columns,
    )
    m = m.where(ret.notna())
    r = ret.where(m.notna())
    mean_r = r.rolling(window, min_periods=window).mean()
    mean_m = m.rolling(window, min_periods=window).mean()
    cov = (r * m).rolling(window, min_periods=window).mean() - mean_r * mean_m
    var_m = (m * m).rolling(window, min_periods=window).mean() - mean_m**2
    var_r = (r * r).rolling(window, min_periods=window).mean() - mean_r**2
    beta = cov / (var_m + _EPS)
    resid_var = (var_r - beta * cov).clip(lower=0.0)
    return beta, np.sqrt(resid_var)
