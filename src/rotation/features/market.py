"""市场状态特征：所有行业共享（不做截面标准化），作为模型的上下文输入。"""

from __future__ import annotations

import pandas as pd

from rotation.features import indicators as ind
from rotation.features.registry import FactorContext, market_feature


@market_feature("mkt_ret_5", "30 行业等权指数 5 日收益")
def mkt_ret_5(c: FactorContext) -> pd.Series:
    return c.market_level / c.market_level.shift(5) - 1.0


@market_feature("mkt_ret_20", "30 行业等权指数 20 日收益")
def mkt_ret_20(c: FactorContext) -> pd.Series:
    return c.market_level / c.market_level.shift(20) - 1.0


@market_feature("mkt_ret_60", "30 行业等权指数 60 日收益")
def mkt_ret_60(c: FactorContext) -> pd.Series:
    return c.market_level / c.market_level.shift(60) - 1.0


@market_feature("mkt_vol_20", "市场 20 日波动率")
def mkt_vol_20(c: FactorContext) -> pd.Series:
    return c.market_ret1.rolling(20, min_periods=20).std()


@market_feature("dispersion_20", "行业 20 日收益的截面标准差（轮动强度）")
def dispersion_20(c: FactorContext) -> pd.Series:
    return ind.pct_change(c.close, 20).std(axis=1)


@market_feature("breadth_20", "20 日收益为正的行业占比")
def breadth_20(c: FactorContext) -> pd.Series:
    ret = ind.pct_change(c.close, 20)
    return (ret > 0).astype(float).where(ret.notna()).mean(axis=1)


@market_feature("mkt_amt_chg_5_60", "全市场成交额 5 日均值 / 60 日均值 − 1")
def mkt_amt_chg_5_60(c: FactorContext) -> pd.Series:
    total = c.amount.sum(axis=1, min_count=1)
    return total.rolling(5, min_periods=5).mean() / total.rolling(60, min_periods=60).mean() - 1.0


@market_feature("mkt_ma_gap_60", "等权指数相对 60 日均线的偏离")
def mkt_ma_gap_60(c: FactorContext) -> pd.Series:
    level = c.market_level
    return level / level.rolling(60, min_periods=60).mean() - 1.0
