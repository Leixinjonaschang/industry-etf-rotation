"""数据质量报告（Markdown）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rotation.config import Config
from rotation.data.etf import ETFData
from rotation.data.industry import IndustryData, daily_return
from rotation.data.schema import TIER_CN
from rotation.utils.io import markdown_table


def industry_checks(data: IndustryData) -> dict[str, pd.DataFrame | str]:
    close, high, low, open_ = data["close"], data["high"], data["low"], data["open"]
    missing = (
        pd.DataFrame({name: frame.isna().mean() for name, frame in data.fields.items()})
        .T.mean(axis=1)
        .rename("缺失率")
        .to_frame()
    )

    tol = 1e-6
    bad_high = (high + tol < np.fmax(open_, close)).sum().sum()
    bad_low = (low - tol > np.fmin(open_, close)).sum().sum()
    ohlc = pd.DataFrame(
        {
            "检查": ["最高价 < max(开,收)", "最低价 > min(开,收)"],
            "异常条数": [int(bad_high), int(bad_low)],
        }
    ).set_index("检查")

    ret = daily_return(close)
    stacked = ret.stack().rename("日收益")
    stacked.index = stacked.index.set_names(["日期", "行业"])
    big = stacked[stacked.abs() > 0.11].sort_values(key=np.abs, ascending=False).head(20).to_frame()
    big.index = big.index.set_levels(big.index.levels[0].strftime("%Y-%m-%d"), level=0)
    return {"missing": missing, "ohlc": ohlc, "big_moves": big}


def etf_availability(etf: ETFData, cfg: Config) -> pd.DataFrame:
    """各年末：有行情的 ETF 数、满足流动性门槛的 ETF 数（按层级）。"""
    rows = []
    amount_ma = etf.amount.rolling(cfg.mapping.liq_window, min_periods=1).mean()
    for year in range(etf.close.index.min().year, etf.close.index.max().year + 1):
        dates = etf.close.index[etf.close.index.year == year]
        if not len(dates):
            continue
        last = dates[-1]
        alive = etf.close.loc[last].notna()
        liquid = alive & (amount_ma.loc[last] >= cfg.mapping.min_amount)
        for tier, group in etf.master.groupby("tier"):
            codes = group.index
            rows.append(
                {
                    "年末": last.date(),
                    "层级": TIER_CN.get(tier, tier),
                    "有行情": int(alive[codes].sum()),
                    "满足流动性": int(liquid[codes].sum()),
                }
            )
    return pd.DataFrame(rows).set_index(["年末", "层级"])


def build_report(data: IndustryData, etf: ETFData, cfg: Config, unit_ratio: float) -> str:
    checks = industry_checks(data)
    dates = data.dates
    lines = [
        "# 数据质量报告",
        "",
        f"- 行业：{len(data.industries)} 个（剔除 {cfg.data.exclude_industries}），"
        f"{dates.min().date()} ~ {dates.max().date()}，{len(dates)} 个交易日",
        f"- ETF：{len(etf.master)} 只，其中 "
        + "，".join(
            f"{TIER_CN.get(k, k)} {v}" for k, v in etf.master["tier"].value_counts().items()
        ),
        f"- ETF 成交额单位检查：切换日前后成交额中位数之比 ≈ {unit_ratio:,.0f}"
        f"（配置乘数 {cfg.data.etf_amount_multiplier_before:g}；二者接近说明单位修正正确）",
        "",
        "## 行业字段缺失率",
        "",
        markdown_table(checks["missing"], ".4%"),
        "",
        "## 开高低收一致性",
        "",
        markdown_table(checks["ohlc"]),
        "",
        "## 单日涨跌幅绝对值 > 11% 的记录（需人工复核）",
        "",
        markdown_table(checks["big_moves"], ".2%") if len(checks["big_moves"]) else "无",
        "",
        "## 各年末可用 ETF 数量",
        "",
        markdown_table(etf_availability(etf, cfg)),
        "",
    ]
    return "\n".join(lines)
