"""调仓信号：每 h 个交易日取一次预测，按分数排序选前 N 个行业（可选缓冲规则）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_dates(eval_dates: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """评价区间第一天起每 horizon 个交易日一次（与预测评价的"调仓日口径"一致）。"""
    return pd.DatetimeIndex(eval_dates)[::horizon]


def build_signals(
    predictions: pd.DataFrame,
    industries: list[str],
    horizon: int,
    top_n: int,
    buffer: int = 0,
) -> pd.DataFrame:
    """返回长表：signal_date, industry, score, rank, priority, selected, realized（原始标签，仅供分析）。

    priority 是映射阶段使用的顺序：缓冲规则保留的已持有行业在前，其余按 rank。
    """
    scores = predictions.pivot(index="date", columns="industry", values="pred").reindex(
        columns=industries
    )
    realized = predictions.pivot(index="date", columns="industry", values="y_raw").reindex(
        columns=industries
    )
    dates = rebalance_dates(scores.index, horizon)
    rows = []
    previous: list[str] = []
    for date in dates:
        s = scores.loc[date]
        order = s.sort_values(ascending=False, na_position="last", kind="stable")
        rank = {ind: i + 1 for i, ind in enumerate(order.index)}
        keep = [
            ind
            for ind in previous
            if rank.get(ind, np.inf) <= top_n + buffer and np.isfinite(s[ind])
        ]
        keep = sorted(keep, key=lambda ind: rank[ind])[:top_n]
        others = [ind for ind in order.index if ind not in keep]
        priority = keep + others
        selected = set(priority[:top_n])
        previous = priority[:top_n]
        for p, ind in enumerate(priority, start=1):
            rows.append(
                {
                    "signal_date": date,
                    "industry": ind,
                    "score": float(s[ind]) if np.isfinite(s[ind]) else np.nan,
                    "rank": rank[ind],
                    "priority": p,
                    "selected": ind in selected,
                    "realized": float(realized.at[date, ind]),
                }
            )
    return pd.DataFrame(rows)


def target_weights(ranks: list[int], weighting: str) -> np.ndarray:
    """等权，或按排名线性加权（第 1 名权重最大）。"""
    n = len(ranks)
    if n == 0:
        return np.zeros(0)
    if weighting == "equal":
        return np.full(n, 1.0 / n)
    if weighting == "rank":
        raw = np.arange(n, 0, -1, dtype=float)
        return raw / raw.sum()
    raise ValueError(f"未知加权方式：{weighting}")
