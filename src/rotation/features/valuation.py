"""估值类因子（默认不启用，用于"加入估值是否有增量"的消融实验）。"""

from __future__ import annotations

import pandas as pd

from rotation.features.registry import FactorContext, factor


def _inverse(frame: pd.DataFrame) -> pd.DataFrame:
    return 1.0 / frame.where(frame != 0)


@factor("ep", "valuation", "盈利收益率 1/PE", requires=("pe",))
def ep(c: FactorContext) -> pd.DataFrame:
    return _inverse(c.field("pe"))


@factor("bp", "valuation", "账面市值比 1/PB", requires=("pb",))
def bp(c: FactorContext) -> pd.DataFrame:
    return _inverse(c.field("pb"))


@factor("sp", "valuation", "销售收益率 1/PS", requires=("ps",))
def sp(c: FactorContext) -> pd.DataFrame:
    return _inverse(c.field("ps"))


@factor("dy", "valuation", "股息率", requires=("dy",))
def dy(c: FactorContext) -> pd.DataFrame:
    return c.field("dy")


@factor("pe_pct_250", "valuation", "PE 在过去 250 日中的分位数", requires=("pe",))
def pe_pct_250(c: FactorContext) -> pd.DataFrame:
    return c.field("pe").rolling(250, min_periods=120).rank(pct=True)


@factor("pb_pct_250", "valuation", "PB 在过去 250 日中的分位数", requires=("pb",))
def pb_pct_250(c: FactorContext) -> pd.DataFrame:
    return c.field("pb").rolling(250, min_periods=120).rank(pct=True)
