"""逐日回测引擎：调仓日按开盘价成交，每日按收盘价估值。

- 目标权重相对于调仓时的组合总市值；权重之和 < 1 的部分为现金；
- 调仓日没有开盘价（停牌）的目标资产：该仓位本期持现金；
- 已持有但当日无法交易的资产：继续持有（按最近收盘价估值），不参与再平衡；
- 交易成本 = 成交名义金额 × 单边费率（佣金 + 滑点；ETF 免印花税）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    nav: pd.Series
    trades: pd.DataFrame
    holdings: pd.DataFrame  # 每个调仓日成交后的权重


def simulate(
    targets: dict[pd.Timestamp, dict[str, float]],
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
    name: str = "portfolio",
) -> SimResult:
    """start 当日收盘净值为 1（空仓），之后按 targets 在各交易日开盘调仓。"""
    dates = close_px.index[(close_px.index >= start) & (close_px.index <= end)]
    if len(dates) == 0:
        raise ValueError("回测区间内没有交易日")
    rate = cost_bps / 1e4
    open_arr = open_px.to_numpy(dtype=float)
    close_filled = close_px.ffill().to_numpy(dtype=float)
    col = {c: j for j, c in enumerate(close_px.columns)}
    row = {d: i for i, d in enumerate(close_px.index)}

    shares: dict[str, float] = {}
    cash = 1.0
    nav_values, trade_rows, holding_rows = [], [], []
    for date in dates:
        i = row[date]
        target = targets.get(date)
        if target is not None and date != dates[0]:
            prices_open = {}
            stuck_value = 0.0
            for asset, qty in shares.items():
                px = open_arr[i, col[asset]]
                if np.isfinite(px):
                    prices_open[asset] = px
                else:
                    stuck_value += qty * close_filled[i - 1, col[asset]] if i > 0 else 0.0
            current = {a: shares[a] * p for a, p in prices_open.items()}
            total = cash + sum(current.values()) + stuck_value
            desired = {}
            for asset, weight in target.items():
                if weight <= 0 or asset not in col:
                    continue
                px = open_arr[i, col[asset]]
                if np.isfinite(px) and asset not in (set(shares) - set(prices_open)):
                    desired[asset] = desired.get(asset, 0.0) + weight * total
            tradable_value = total - stuck_value
            wanted = sum(desired.values())
            if wanted > tradable_value and wanted > 0:
                desired = {a: v * tradable_value / wanted for a, v in desired.items()}
            assets = set(desired) | set(current)
            turnover = sum(abs(desired.get(a, 0.0) - current.get(a, 0.0)) for a in assets)
            cost = turnover * rate
            wanted = sum(desired.values())
            spare = tradable_value - wanted - cost
            if spare < 0 and wanted > 0:
                scale = max(tradable_value - cost, 0.0) / wanted
                desired = {a: v * scale for a, v in desired.items()}
                wanted = sum(desired.values())
            cash = tradable_value - wanted - cost
            stuck = {a: q for a, q in shares.items() if a not in prices_open}
            shares = {**stuck, **{a: v / open_arr[i, col[a]] for a, v in desired.items() if v > 0}}
            trade_rows.append(
                {
                    "trade_date": date,
                    "value_before": total,
                    "turnover": turnover / total if total > 0 else 0.0,
                    "cost": cost,
                    "n_holdings": len(shares),
                    "cash_weight": cash / total if total > 0 else 1.0,
                    "stuck_assets": ",".join(sorted(stuck)),
                }
            )
            holding_rows.append({"trade_date": date, **{a: v / total for a, v in desired.items()}})
        value = cash + sum(q * close_filled[i, col[a]] for a, q in shares.items())
        nav_values.append(value)
    nav = pd.Series(nav_values, index=dates, name=name)
    holdings = (
        pd.DataFrame(holding_rows).set_index("trade_date") if holding_rows else pd.DataFrame()
    )
    return SimResult(nav=nav, trades=pd.DataFrame(trade_rows), holdings=holdings)
