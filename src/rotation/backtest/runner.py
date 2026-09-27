"""组合回测：策略（ETF / 行业指数）+ 基准 + 随机选行业检验 + 映射损耗分解。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rotation.backtest.engine import SimResult, simulate
from rotation.backtest.metrics import performance, period_returns, yearly_returns
from rotation.config import Config
from rotation.data.etf import ETFData
from rotation.data.industry import IndustryData
from rotation.mapping.mapper import MappingResult
from rotation.strategy.signals import target_weights

SERIES_CN = {
    "strategy_etf": "策略（ETF 落地）",
    "strategy_index": "策略（行业指数，理论）",
    "strategy_index_held": "策略（实际持有行业的指数）",
    "ew_industry": "30 行业等权",
    "hs300_etf": "沪深300ETF",
    "momentum_index": "动量前 N（行业指数）",
}


@dataclass
class BacktestOutput:
    navs: pd.DataFrame
    metrics: pd.DataFrame
    metrics_vs_hs300: pd.DataFrame
    yearly: pd.DataFrame
    trades: dict[str, pd.DataFrame] = field(default_factory=dict)
    random_test: dict = field(default_factory=dict)
    random_distribution: np.ndarray | None = None
    loss_decomposition: pd.DataFrame | None = None
    periods: pd.DataFrame | None = None


def _next_day(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    loc = calendar.searchsorted(date, side="right")
    return calendar[loc] if loc < len(calendar) else None


def _index_targets(signals: pd.DataFrame, calendar: pd.DatetimeIndex, weighting: str) -> dict:
    targets = {}
    for date, group in signals[signals["selected"]].groupby("signal_date"):
        trade = _next_day(calendar, date)
        if trade is None:
            continue
        group = group.sort_values("priority")
        weights = target_weights(group["rank"].tolist(), weighting)
        targets[trade] = dict(zip(group["industry"], weights))
    return targets


def _slot_targets(slots: pd.DataFrame, top_n: int, weighting: str, use: str) -> dict:
    """use='etf'：按映射的 ETF；use='industry'：按实际持有的行业（现金/宽基仓位记为现金）。"""
    targets = {}
    weights = target_weights(list(range(1, top_n + 1)), weighting)
    for trade, group in slots.groupby("trade_date"):
        group = group.sort_values("slot")
        target: dict[str, float] = {}
        for k, row in enumerate(group.itertuples(index=False)):
            w = weights[k] if k < len(weights) else 0.0
            if use == "etf":
                code = row.etf_code
                if isinstance(code, str) and code:
                    target[code] = target.get(code, 0.0) + w
            elif row.status in ("mapped", "fallback_next") and isinstance(row.industry, str):
                target[row.industry] = target.get(row.industry, 0.0) + w
        targets[trade] = target
    return targets


def _momentum_targets(
    industry: IndustryData, signal_dates: pd.DatetimeIndex, lookback: int, top_n: int
) -> dict:
    close = industry["close"]
    score = close / close.shift(lookback) - 1.0
    targets = {}
    for date in signal_dates:
        trade = _next_day(close.index, date)
        if trade is None:
            continue
        top = score.loc[date].dropna().sort_values(ascending=False).index[:top_n]
        targets[trade] = {ind: 1.0 / len(top) for ind in top}
    return targets


def random_topn_test(
    industry: IndustryData,
    signals: pd.DataFrame,
    top_n: int,
    n_random: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    """随机选 N 个行业（每期独立抽取）的年化收益分布 vs 策略（均不含成本，开盘到开盘）。"""
    calendar = industry.dates
    open_ = industry["open"]
    close = industry["close"]
    sig_dates = pd.DatetimeIndex(sorted(signals["signal_date"].unique()))
    trades = [d for d in (_next_day(calendar, s) for s in sig_dates) if d is not None]
    if len(trades) < 2:
        return {}, np.array([])
    ends = trades[1:]
    rows = []
    for k, start in enumerate(trades):
        start_px = open_.loc[start]
        end_px = open_.loc[ends[k]] if k < len(ends) else close.iloc[-1]
        rows.append((end_px / start_px - 1.0).to_numpy(dtype=float))
    period = np.nan_to_num(np.vstack(rows), nan=0.0)  # [P, N]
    industries = list(open_.columns)
    chosen = signals[signals["selected"]]
    strat = []
    for date in sig_dates[: len(trades)]:
        sel = chosen[chosen["signal_date"] == date]["industry"]
        idx = [industries.index(i) for i in sel]
        strat.append(idx)
    strat_period = np.array([period[p, idx].mean() if idx else 0.0 for p, idx in enumerate(strat)])
    n_days = len(calendar[(calendar >= trades[0])])
    years = max(n_days / 252.0, 1e-9)

    def annualize(period_ret: np.ndarray) -> np.ndarray:
        return np.prod(1.0 + period_ret, axis=-1) ** (1.0 / years) - 1.0

    rng = np.random.default_rng(seed)
    n = period.shape[1]
    picks = np.argsort(rng.random((n_random, period.shape[0], n)), axis=2)[:, :, :top_n]
    random_period = np.take_along_axis(
        np.broadcast_to(period, (n_random, *period.shape)), picks, axis=2
    ).mean(axis=2)
    random_annual = annualize(random_period)
    strategy_annual = float(annualize(strat_period))
    p_value = float((np.sum(random_annual >= strategy_annual) + 1) / (n_random + 1))
    return (
        {
            "strategy_annual_gross": strategy_annual,
            "random_mean": float(random_annual.mean()),
            "random_p05": float(np.percentile(random_annual, 5)),
            "random_p50": float(np.percentile(random_annual, 50)),
            "random_p95": float(np.percentile(random_annual, 95)),
            "percentile_rank": float((random_annual < strategy_annual).mean()),
            "p_value": p_value,
            "n_random": n_random,
            "n_periods": int(period.shape[0]),
        },
        random_annual,
    )


def run_backtests(
    cfg: Config,
    signals: pd.DataFrame,
    industry: IndustryData,
    etf: ETFData | None,
    mapping: MappingResult | None,
) -> BacktestOutput:
    calendar = industry.dates
    top_n = cfg.strategy.top_n
    cost = cfg.backtest.cost_bps
    sig_dates = pd.DatetimeIndex(sorted(signals["signal_date"].unique()))
    start, end = sig_dates[0], calendar[-1]
    ind_open, ind_close = industry["open"], industry["close"]

    sims: dict[str, SimResult] = {}
    sims["strategy_index"] = simulate(
        _index_targets(signals, calendar, cfg.strategy.weighting),
        ind_open,
        ind_close,
        start,
        end,
        cost,
        "strategy_index",
    )
    if mapping is not None and not mapping.slots.empty and etf is not None:
        sims["strategy_etf"] = simulate(
            _slot_targets(mapping.slots, top_n, cfg.strategy.weighting, "etf"),
            etf.open,
            etf.close,
            start,
            end,
            cost,
            "strategy_etf",
        )
        sims["strategy_index_held"] = simulate(
            _slot_targets(mapping.slots, top_n, cfg.strategy.weighting, "industry"),
            ind_open,
            ind_close,
            start,
            end,
            cost,
            "strategy_index_held",
        )
    ew_targets = {}
    for date in sig_dates:
        trade = _next_day(calendar, date)
        if trade is not None:
            ew_targets[trade] = {ind: 1.0 / len(industry.industries) for ind in industry.industries}
    sims["ew_industry"] = simulate(ew_targets, ind_open, ind_close, start, end, 0.0, "ew_industry")
    sims["momentum_index"] = simulate(
        _momentum_targets(industry, sig_dates, cfg.backtest.momentum_lookback, top_n),
        ind_open,
        ind_close,
        start,
        end,
        cost,
        "momentum_index",
    )
    bench_code = str(cfg.data.benchmark_codes[0]) if cfg.data.benchmark_codes else None
    if etf is not None and bench_code in etf.close.columns:
        first_trade = _next_day(calendar, start)
        sims["hs300_etf"] = simulate(
            {first_trade: {bench_code: 1.0}}, etf.open, etf.close, start, end, cost, "hs300_etf"
        )

    order = [k for k in SERIES_CN if k in sims]
    navs = pd.DataFrame({k: sims[k].nav for k in order})
    rf = cfg.backtest.risk_free
    metrics = pd.DataFrame({k: performance(navs[k], rf, navs["ew_industry"]) for k in order}).T
    if "hs300_etf" in navs:
        metrics_hs = pd.DataFrame({k: performance(navs[k], rf, navs["hs300_etf"]) for k in order}).T
    else:
        metrics_hs = pd.DataFrame()
    for k in order:
        trades = sims[k].trades
        if len(trades):
            metrics.loc[k, "avg_turnover"] = trades["turnover"].mean()
            years = len(navs) / 252.0
            metrics.loc[k, "cost_drag_annual"] = (
                trades["cost"] / trades["value_before"]
            ).sum() / max(years, 1e-9)
            metrics.loc[k, "avg_cash_weight"] = trades["cash_weight"].mean()

    random_stats, random_dist = random_topn_test(
        industry, signals, top_n, cfg.backtest.n_random, cfg.experiment.seed
    )

    trade_dates = pd.DatetimeIndex(
        [_next_day(calendar, d) for d in sig_dates if _next_day(calendar, d) is not None]
    )
    boundaries = trade_dates.union(pd.DatetimeIndex([start]))
    periods = pd.DataFrame({k: period_returns(navs[k], boundaries) for k in order})
    loss = None
    if "strategy_etf" in navs:
        ideal, held, etf_p = (
            periods["strategy_index"],
            periods["strategy_index_held"],
            periods["strategy_etf"],
        )
        decomposition = pd.DataFrame(
            {
                "选择偏离（顺延/现金）": held - ideal,
                "跟踪偏离（ETF vs 行业指数）": etf_p - held,
                "合计映射损耗": etf_p - ideal,
            }
        )
        years = len(navs) / 252.0
        loss = pd.DataFrame(
            {
                "平均每期": decomposition.mean(),
                "年化（每期之和/年数）": decomposition.sum() / max(years, 1e-9),
                "每期标准差": decomposition.std(),
            }
        )
    return BacktestOutput(
        navs=navs,
        metrics=metrics,
        metrics_vs_hs300=metrics_hs,
        yearly=yearly_returns(navs),
        trades={k: sims[k].trades for k in order},
        random_test=random_stats,
        random_distribution=random_dist,
        loss_decomposition=loss,
        periods=periods,
    )
