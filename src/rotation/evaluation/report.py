"""预测评价：主口径 = 不重叠的调仓日截面；辅助口径 = 全部交易日（HAC 修正）。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rotation.evaluation.ranking import (
    daily_ic,
    daily_rank_ic,
    ic_summary,
    ndcg_at_k,
    quantile_returns,
    topk_excess,
    topk_hit_rate,
)
from rotation.evaluation.regression import regression_metrics
from rotation.features.selection import hac_lags


@dataclass
class WidePredictions:
    dates: pd.DatetimeIndex
    industries: list[str]
    pred: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    valid: np.ndarray
    bench: np.ndarray

    @classmethod
    def from_long(cls, df: pd.DataFrame, industries: list[str]) -> WidePredictions:
        def wide(col: str) -> np.ndarray:
            table = df.pivot(index="date", columns="industry", values=col).reindex(
                columns=industries
            )
            return table.to_numpy(dtype=float, copy=True)

        dates = pd.DatetimeIndex(sorted(df["date"].unique()))
        valid = df.pivot(index="date", columns="industry", values="valid").reindex(
            columns=industries
        )
        return cls(
            dates=dates,
            industries=industries,
            pred=wide("pred"),
            y=wide("y"),
            y_raw=wide("y_raw"),
            valid=valid.eq(True).to_numpy(dtype=bool),
            bench=wide("bench_mean"),
        )

    def take(self, rows: np.ndarray) -> WidePredictions:
        return WidePredictions(
            self.dates[rows],
            self.industries,
            self.pred[rows],
            self.y[rows],
            self.y_raw[rows],
            self.valid[rows],
            self.bench[rows],
        )


def _nanmean(values: np.ndarray, axis: int | None = None):
    """全为 NaN 时安静地返回 NaN（不发 RuntimeWarning）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def rebalance_rows(n_dates: int, horizon: int) -> np.ndarray:
    return np.arange(0, n_dates, horizon)


def _block_metrics(
    w: WidePredictions, top_n: int, lags: int, produces_returns: bool
) -> dict[str, float]:
    mask = w.valid & np.isfinite(w.y)
    rank_ic = daily_rank_ic(w.pred, w.y, mask)
    ic = daily_ic(w.pred, w.y, mask)
    hit = topk_hit_rate(w.pred, w.y, top_n, mask)
    excess = topk_excess(w.pred, w.y_raw, top_n, mask)
    groups = quantile_returns(w.pred, w.y_raw, 5, mask)
    long_short = groups[:, 0] - groups[:, -1]
    ric = ic_summary(rank_ic, lags)
    pic = ic_summary(ic, lags)
    exc = ic_summary(excess, lags)
    ls = ic_summary(long_short, lags)
    group_means = _nanmean(groups, axis=0)
    mono = (
        spearmanr(np.arange(5), group_means).statistic if np.isfinite(group_means).all() else np.nan
    )
    out = {
        "n_dates": int(np.isfinite(rank_ic).sum()),
        "rank_ic": ric["mean"],
        "rank_ic_std": ric["std"],
        "rank_icir": ric["icir"],
        "rank_ic_win": ric["win_rate"],
        "rank_ic_t": ric["t"],
        "rank_ic_p": ric["p"],
        "ic": pic["mean"],
        "ic_t": pic["t"],
        f"hit@{top_n}": float(_nanmean(hit)),
        f"hit@{top_n}_random": top_n / w.pred.shape[1],
        f"ndcg@{top_n}": float(_nanmean(ndcg_at_k(w.pred, w.y, top_n, mask))),
        f"top{top_n}_excess": exc["mean"],
        f"top{top_n}_excess_t": exc["t"],
        "long_short": ls["mean"],
        "long_short_t": ls["t"],
        "group_monotonicity": float(-mono) if np.isfinite(mono) else np.nan,
    }
    for g, value in enumerate(group_means, start=1):
        out[f"group{g}"] = float(value)
    if produces_returns:
        reg = regression_metrics(w.pred[mask], w.y[mask], w.bench[mask])
        out.update({f"reg_{k}": v for k, v in reg.items()})
    return out


def evaluate_predictions(
    df: pd.DataFrame, industries: list[str], horizon: int, top_n: int, produces_returns: bool
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """返回 (汇总表[口径 × 指标], 明细表字典)。"""
    w = WidePredictions.from_long(df, industries)
    rows = rebalance_rows(len(w.dates), horizon)
    reb = w.take(rows)
    summary = pd.DataFrame(
        {
            "rebalance": _block_metrics(reb, top_n, hac_lags(len(rows), 1), produces_returns),
            "all_days": _block_metrics(w, top_n, hac_lags(len(w.dates), horizon), produces_returns),
        }
    ).T

    mask = w.valid & np.isfinite(w.y)
    series = pd.DataFrame(
        {
            "rank_ic": daily_rank_ic(w.pred, w.y, mask),
            "ic": daily_ic(w.pred, w.y, mask),
            f"hit@{top_n}": topk_hit_rate(w.pred, w.y, top_n, mask),
            f"top{top_n}_excess": topk_excess(w.pred, w.y_raw, top_n, mask),
        },
        index=w.dates,
    )
    groups = quantile_returns(w.pred, w.y_raw, 5, mask)
    series["long_short"] = groups[:, 0] - groups[:, -1]
    series["is_rebalance"] = False
    series.iloc[rows, series.columns.get_loc("is_rebalance")] = True
    series.index.name = "date"

    yearly_rows = {}
    for year in sorted(set(reb.dates.year)):
        sel = np.flatnonzero(reb.dates.year == year)
        yearly_rows[year] = _block_metrics(
            reb.take(sel), top_n, hac_lags(len(sel), 1), produces_returns
        )
    yearly = pd.DataFrame(yearly_rows).T
    yearly.index.name = "year"

    group_table = pd.DataFrame(
        {"rebalance": _nanmean(groups[rows], axis=0), "all_days": _nanmean(groups, axis=0)},
        index=[f"G{g}" for g in range(1, 6)],
    )
    return summary, {"series": series, "yearly": yearly, "groups": group_table}
