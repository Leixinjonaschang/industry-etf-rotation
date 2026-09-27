"""统计检验：Newey-West(HAC) t 值、Diebold-Mariano 检验、配对检验。"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def newey_west_se(values: np.ndarray, lags: int) -> float:
    """均值的 HAC 标准误（Bartlett 核）。"""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    centered = x - x.mean()
    variance = centered @ centered / n
    for lag in range(1, min(max(lags, 0), n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * (centered[lag:] @ centered[:-lag]) / n
    return math.sqrt(max(variance, 0.0) / n)


def newey_west_t(values: np.ndarray, lags: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    se = newey_west_se(x, lags)
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    return float(x.mean() / se)


def two_sided_p(t_value: float) -> float:
    if not np.isfinite(t_value):
        return float("nan")
    return float(2.0 * (1.0 - stats.norm.cdf(abs(t_value))))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1) -> dict[str, float]:
    """DM 检验：d = loss_a − loss_b；统计量 < 0 表示 A 的损失更小。含 HLN 小样本修正。"""
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    n = len(d)
    if n < 10:
        return {"dm": float("nan"), "p_value": float("nan"), "n": n}
    lags = max(horizon - 1, 0)
    se = newey_west_se(d, lags)
    if not np.isfinite(se) or se <= 0:
        return {"dm": float("nan"), "p_value": float("nan"), "n": n}
    dm = d.mean() / se
    correction = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_hln = dm * correction
    p = float(2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1)))
    return {"dm": float(dm_hln), "p_value": p, "n": n}


def paired_hac_test(a: np.ndarray, b: np.ndarray, lags: int) -> dict[str, float]:
    """两组逐期指标（如 RankIC）之差的 HAC t 检验。"""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    diff = x[mask] - y[mask]
    t_value = newey_west_t(diff, lags)
    return {
        "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
        "t": t_value,
        "p_value": two_sided_p(t_value),
        "n": int(len(diff)),
    }
