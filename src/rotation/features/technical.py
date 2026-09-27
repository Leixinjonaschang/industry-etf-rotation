"""技术面候选因子（动量、趋势、震荡、量能、波动五类）。

约定：返回 日期 × 行业 的 DataFrame，只用当日及以前的数据；
累积型/量级型序列（OBV、成交量水平等）一律转换为比值或变化率，保证平稳。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rotation.features import indicators as ind
from rotation.features.registry import FactorContext, factor

# ------------------------------------------------------------------ 动量 / 反转


@factor("ret_5", "momentum", "过去 5 日收益（短期反转）")
def ret_5(c: FactorContext) -> pd.DataFrame:
    return ind.pct_change(c.close, 5)


@factor("ret_20", "momentum", "过去 20 日收益")
def ret_20(c: FactorContext) -> pd.DataFrame:
    return ind.pct_change(c.close, 20)


@factor("ret_60", "momentum", "过去 60 日收益")
def ret_60(c: FactorContext) -> pd.DataFrame:
    return ind.pct_change(c.close, 60)


@factor("ret_120", "momentum", "过去 120 日收益")
def ret_120(c: FactorContext) -> pd.DataFrame:
    return ind.pct_change(c.close, 120)


@factor("mom_60_skip5", "momentum", "t-60 到 t-5 的收益（跳过最近一周）")
def mom_60_skip5(c: FactorContext) -> pd.DataFrame:
    return c.close.shift(5) / c.close.shift(60) - 1.0


@factor("mom_250_skip20", "momentum", "t-250 到 t-20 的收益（12-1 月动量）")
def mom_250_skip20(c: FactorContext) -> pd.DataFrame:
    return c.close.shift(20) / c.close.shift(250) - 1.0


@factor("mom_accel", "momentum", "动量加速度：ret_20 − ret_60/3")
def mom_accel(c: FactorContext) -> pd.DataFrame:
    return ind.pct_change(c.close, 20) - ind.pct_change(c.close, 60) / 3.0


@factor("mom_quality_60", "momentum", "风险调整动量：ret_60 /(60 日波动 × √60)")
def mom_quality_60(c: FactorContext) -> pd.DataFrame:
    vol = c.ret1.rolling(60, min_periods=60).std()
    return ind.pct_change(c.close, 60) / (vol * np.sqrt(60) + 1e-12)


@factor("high_prox_250", "momentum", "距 250 日最高收盘价的距离")
def high_prox_250(c: FactorContext) -> pd.DataFrame:
    return c.close / c.close.rolling(250, min_periods=250).max() - 1.0


@factor("rs_ma_gap_20", "momentum", "相对强弱线（行业/市场）偏离其 20 日均线")
def rs_ma_gap_20(c: FactorContext) -> pd.DataFrame:
    rs = c.close.div(c.market_level, axis=0)
    return rs / ind.sma(rs, 20) - 1.0


# ------------------------------------------------------------------ 趋势


@factor("ma_gap_5", "trend", "收盘价相对 5 日均线的偏离")
def ma_gap_5(c: FactorContext) -> pd.DataFrame:
    return c.close / ind.sma(c.close, 5) - 1.0


@factor("ma_gap_20", "trend", "收盘价相对 20 日均线的偏离")
def ma_gap_20(c: FactorContext) -> pd.DataFrame:
    return c.close / ind.sma(c.close, 20) - 1.0


@factor("ma_gap_60", "trend", "收盘价相对 60 日均线的偏离")
def ma_gap_60(c: FactorContext) -> pd.DataFrame:
    return c.close / ind.sma(c.close, 60) - 1.0


@factor("ma_cross_5_20", "trend", "5 日均线 / 20 日均线 − 1")
def ma_cross_5_20(c: FactorContext) -> pd.DataFrame:
    return ind.sma(c.close, 5) / ind.sma(c.close, 20) - 1.0


@factor("ma_cross_20_60", "trend", "20 日均线 / 60 日均线 − 1")
def ma_cross_20_60(c: FactorContext) -> pd.DataFrame:
    return ind.sma(c.close, 20) / ind.sma(c.close, 60) - 1.0


@factor("ema_gap_12", "trend", "收盘价相对 EMA12 的偏离")
def ema_gap_12(c: FactorContext) -> pd.DataFrame:
    return c.close / ind.ema(c.close, 12) - 1.0


@factor("ema_gap_26", "trend", "收盘价相对 EMA26 的偏离")
def ema_gap_26(c: FactorContext) -> pd.DataFrame:
    return c.close / ind.ema(c.close, 26) - 1.0


@factor("hma_gap_20", "trend", "收盘价相对 Hull 均线(20) 的偏离")
def hma_gap_20(c: FactorContext) -> pd.DataFrame:
    return c.close / ind.hma(c.close, 20) - 1.0


@factor("macd", "trend", "MACD 快慢线差 DIF / 收盘价")
def macd(c: FactorContext) -> pd.DataFrame:
    return (ind.ema(c.close, 12) - ind.ema(c.close, 26)) / c.close


@factor("macd_hist", "trend", "MACD 柱 2×(DIF−DEA) / 收盘价")
def macd_hist(c: FactorContext) -> pd.DataFrame:
    dif = ind.ema(c.close, 12) - ind.ema(c.close, 26)
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    return 2.0 * (dif - dea) / c.close


@factor("adx_14", "trend", "ADX(14) 趋势强度")
def adx_14(c: FactorContext) -> pd.DataFrame:
    return ind.adx(c.high, c.low, c.close, 14)


@factor("di_spread_14", "trend", "+DI − −DI（趋势方向）")
def di_spread_14(c: FactorContext) -> pd.DataFrame:
    plus_di, minus_di, _ = ind.dmi(c.high, c.low, c.close, 14)
    return plus_di - minus_di


@factor("trix_15", "trend", "TRIX(15)")
def trix_15(c: FactorContext) -> pd.DataFrame:
    return ind.trix(c.close, 15)


# ------------------------------------------------------------------ 震荡


@factor("rsi_6", "oscillator", "RSI(6)")
def rsi_6(c: FactorContext) -> pd.DataFrame:
    return ind.rsi(c.close, 6)


@factor("rsi_14", "oscillator", "RSI(14)")
def rsi_14(c: FactorContext) -> pd.DataFrame:
    return ind.rsi(c.close, 14)


@factor("kdj_k", "oscillator", "KDJ(9,3,3) 的 K")
def kdj_k(c: FactorContext) -> pd.DataFrame:
    return ind.kdj(c.close, c.high, c.low, 9)[0]


@factor("kdj_d", "oscillator", "KDJ(9,3,3) 的 D")
def kdj_d(c: FactorContext) -> pd.DataFrame:
    return ind.kdj(c.close, c.high, c.low, 9)[1]


@factor("kdj_j", "oscillator", "KDJ(9,3,3) 的 J")
def kdj_j(c: FactorContext) -> pd.DataFrame:
    return ind.kdj(c.close, c.high, c.low, 9)[2]


@factor("wpr_14", "oscillator", "威廉指标 WR(14)")
def wpr_14(c: FactorContext) -> pd.DataFrame:
    return ind.williams_r(c.close, c.high, c.low, 14)


@factor("cci_14", "oscillator", "CCI(14)")
def cci_14(c: FactorContext) -> pd.DataFrame:
    return ind.cci(c.high, c.low, c.close, 14)


@factor("cmo_14", "oscillator", "钱德动量摆动 CMO(14)")
def cmo_14(c: FactorContext) -> pd.DataFrame:
    return ind.cmo(c.close, 14)


@factor("stoch_pos_60", "oscillator", "收盘价在 60 日高低区间中的位置")
def stoch_pos_60(c: FactorContext) -> pd.DataFrame:
    lowest = c.low.rolling(60, min_periods=60).min()
    highest = c.high.rolling(60, min_periods=60).max()
    return (c.close - lowest) / (highest - lowest + 1e-12)


# ------------------------------------------------------------------ 量能


@factor("vol_ratio_5", "volume", "当日成交量 / 5 日均量（量比）")
def vol_ratio_5(c: FactorContext) -> pd.DataFrame:
    return c.volume / (ind.sma(c.volume, 5) + 1e-12)


@factor("vol_trend_5_60", "volume", "5 日均量 / 60 日均量 − 1")
def vol_trend_5_60(c: FactorContext) -> pd.DataFrame:
    return ind.sma(c.volume, 5) / (ind.sma(c.volume, 60) + 1e-12) - 1.0


def _amount_share(c: FactorContext) -> pd.DataFrame:
    total = c.amount.sum(axis=1, min_count=1)
    return c.amount.div(total, axis=0)


@factor("amt_share_chg_5_60", "volume", "成交额占全行业比重：5 日均值 / 60 日均值 − 1（资金流向）")
def amt_share_chg_5_60(c: FactorContext) -> pd.DataFrame:
    share = _amount_share(c)
    return ind.sma(share, 5) / (ind.sma(share, 60) + 1e-12) - 1.0


@factor("amt_share_rel_20_250", "volume", "成交额占比：20 日均值 / 250 日均值 − 1（拥挤度）")
def amt_share_rel_20_250(c: FactorContext) -> pd.DataFrame:
    share = _amount_share(c)
    return ind.sma(share, 20) / (share.rolling(250, min_periods=120).mean() + 1e-12) - 1.0


@factor(
    "turnover_rel_20_250", "volume", "换手率：20 日均值 / 250 日均值 − 1", requires=("turnover",)
)
def turnover_rel_20_250(c: FactorContext) -> pd.DataFrame:
    turnover = c.field("turnover")
    return ind.sma(turnover, 20) / (turnover.rolling(250, min_periods=120).mean() + 1e-12) - 1.0


@factor("cmf_20", "volume", "蔡金资金流 CMF(20)")
def cmf_20(c: FactorContext) -> pd.DataFrame:
    return ind.cmf(c.high, c.low, c.close, c.volume, 20)


@factor("obv_chg_20", "volume", "OBV 20 日变化 / 20 日成交量")
def obv_chg_20(c: FactorContext) -> pd.DataFrame:
    return ind.obv_change(c.close, c.volume, 20)


@factor("pv_corr_20", "volume", "20 日量价相关：corr(日收益, 对数成交量变化)")
def pv_corr_20(c: FactorContext) -> pd.DataFrame:
    dlogv = np.log(c.volume.where(c.volume > 0)).diff()
    return c.ret1.rolling(20, min_periods=20).corr(dlogv)


@factor("amihud_20", "volume", "Amihud 非流动性：log(20 日均 |r|/成交额)")
def amihud_20(c: FactorContext) -> pd.DataFrame:
    illiq = c.ret1.abs() / (c.amount.where(c.amount > 0))
    return np.log(ind.sma(illiq, 20) * 1e9 + 1e-12)


# ------------------------------------------------------------------ 波动


@factor("atr_14", "volatility", "ATR(14) / 收盘价")
def atr_14(c: FactorContext) -> pd.DataFrame:
    return ind.atr(c.high, c.low, c.close, 14) / c.close


@factor("boll_pos_20", "volatility", "布林带 %B 位置 (20, 2)")
def boll_pos_20(c: FactorContext) -> pd.DataFrame:
    return ind.bollinger(c.close, 20, 2.0)[0]


@factor("boll_width_20", "volatility", "布林带宽度 (20, 2)")
def boll_width_20(c: FactorContext) -> pd.DataFrame:
    return ind.bollinger(c.close, 20, 2.0)[1]


@factor("rv_20", "volatility", "20 日实现波动率")
def rv_20(c: FactorContext) -> pd.DataFrame:
    return c.ret1.rolling(20, min_periods=20).std()


@factor("rv_60", "volatility", "60 日实现波动率")
def rv_60(c: FactorContext) -> pd.DataFrame:
    return c.ret1.rolling(60, min_periods=60).std()


@factor("vol_regime_20_60", "volatility", "20 日波动 / 60 日波动 − 1")
def vol_regime_20_60(c: FactorContext) -> pd.DataFrame:
    return (
        c.ret1.rolling(20, min_periods=20).std()
        / (c.ret1.rolling(60, min_periods=60).std() + 1e-12)
        - 1.0
    )


@factor("downvol_20", "volatility", "20 日下行波动")
def downvol_20(c: FactorContext) -> pd.DataFrame:
    neg = c.ret1.clip(upper=0.0)
    return np.sqrt((neg * neg).rolling(20, min_periods=20).mean())


@factor("skew_60", "volatility", "60 日收益偏度")
def skew_60(c: FactorContext) -> pd.DataFrame:
    return c.ret1.rolling(60, min_periods=60).skew()


@factor("max_ret_20", "volatility", "20 日最大单日收益（彩票效应）")
def max_ret_20(c: FactorContext) -> pd.DataFrame:
    return c.ret1.rolling(20, min_periods=20).max()


@factor("beta_60", "volatility", "60 日市场 β")
def beta_60(c: FactorContext) -> pd.DataFrame:
    return ind.rolling_beta_resid_vol(c.ret1, c.market_ret1, 60)[0]


@factor("ivol_60", "volatility", "60 日特质波动率")
def ivol_60(c: FactorContext) -> pd.DataFrame:
    return ind.rolling_beta_resid_vol(c.ret1, c.market_ret1, 60)[1]
