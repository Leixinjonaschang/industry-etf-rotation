"""绩效指标（日频净值）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance(
    nav: pd.Series, risk_free: float = 0.0, benchmark: pd.Series | None = None
) -> dict[str, float]:
    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    n = len(ret)
    if n < 2:
        return {}
    rf_daily = (1.0 + risk_free) ** (1.0 / TRADING_DAYS) - 1.0
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    annual = (1.0 + total) ** (TRADING_DAYS / n) - 1.0
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    excess = ret - rf_daily
    sharpe = (
        excess.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS) if ret.std(ddof=1) > 0 else np.nan
    )
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * np.sqrt(TRADING_DAYS)
    sortino = excess.mean() * TRADING_DAYS / downside if downside > 0 else np.nan
    drawdown = nav / nav.cummax() - 1.0
    mdd = drawdown.min()
    out = {
        "total_return": total,
        "annual_return": annual,
        "annual_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": annual / abs(mdd) if mdd < 0 else np.nan,
        "daily_win_rate": float((ret > 0).mean()),
        "n_days": n,
    }
    if benchmark is not None:
        bench = benchmark.reindex(nav.index).ffill().dropna()
        b_ret = bench.pct_change().reindex(ret.index)
        active = (ret - b_ret).dropna()
        b_total = bench.iloc[-1] / bench.iloc[0] - 1.0
        b_annual = (1.0 + b_total) ** (TRADING_DAYS / max(len(active), 1)) - 1.0
        te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
        out.update(
            {
                "excess_annual": annual - b_annual,
                "tracking_error": te,
                "information_ratio": active.mean() * TRADING_DAYS / te if te > 0 else np.nan,
                "active_win_rate": float((active > 0).mean()),
            }
        )
    return out


def yearly_returns(navs: pd.DataFrame) -> pd.DataFrame:
    """各列净值的分年度收益（首年从起点算）。"""
    year_end = navs.groupby(navs.index.year).last()
    start = navs.iloc[0]
    prev = pd.concat([start.to_frame().T, year_end.iloc[:-1]])
    prev.index = year_end.index
    return year_end / prev.to_numpy() - 1.0


def monthly_returns(nav: pd.Series) -> pd.DataFrame:
    month_end = nav.groupby(nav.index.to_period("M")).last()
    prev = month_end.shift(1)
    prev.iloc[0] = nav.iloc[0]
    ret = month_end / prev - 1.0
    frame = pd.DataFrame(
        {"year": month_end.index.year, "month": month_end.index.month, "ret": ret.to_numpy()}
    )
    return frame.pivot(index="year", columns="month", values="ret")


def period_returns(nav: pd.Series, boundaries: pd.DatetimeIndex) -> pd.Series:
    """按调仓边界切分的区间收益（边界取收盘净值）。"""
    points = nav.reindex(boundaries.union([nav.index[-1]])).ffill().dropna()
    return (points / points.shift(1) - 1.0).dropna()
