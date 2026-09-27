"""跟踪质量指标：窗口内日收益的 Pearson / Spearman / 超额收益相关 / 跟踪误差 / β。"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _masked_corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """逐列相关（a、b 形状 [W, K]，valid 为共同有效掩码）。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(valid, a, np.nan)
        b = np.where(valid, b, np.nan)
        am = a - np.nanmean(a, axis=0)
        bm = b - np.nanmean(b, axis=0)
        num = np.nansum(am * bm, axis=0)
        den = np.sqrt(np.nansum(am * am, axis=0) * np.nansum(bm * bm, axis=0))
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def tracking_metrics(
    industry_ret: np.ndarray, market_ret: np.ndarray, etf_ret: np.ndarray
) -> dict[str, np.ndarray]:
    """industry_ret/market_ret: [W]；etf_ret: [W, K]。返回每只 ETF 的指标数组。"""
    k = etf_ret.shape[1]
    r = np.repeat(industry_ret[:, None], k, axis=1)
    m = np.repeat(market_ret[:, None], k, axis=1)
    valid = np.isfinite(r) & np.isfinite(etf_ret) & np.isfinite(m)
    n_obs = valid.sum(axis=0)

    pearson = _masked_corr(r, etf_ret, valid)
    rank_r = rankdata(np.where(valid, r, np.nan), axis=0, nan_policy="omit")
    rank_e = rankdata(np.where(valid, etf_ret, np.nan), axis=0, nan_policy="omit")
    spearman = _masked_corr(rank_r, rank_e, valid)
    excess = _masked_corr(r - m, etf_ret - m, valid)

    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.where(valid, r - etf_ret, np.nan)
        te = np.nanstd(diff, axis=0, ddof=1) * np.sqrt(252.0)
        rv = np.where(valid, r, np.nan)
        ev = np.where(valid, etf_ret, np.nan)
        cov = np.nanmean((rv - np.nanmean(rv, axis=0)) * (ev - np.nanmean(ev, axis=0)), axis=0)
        var_r = np.nanvar(rv, axis=0)
        beta = cov / var_r
    return {
        "pearson": pearson,
        "spearman": spearman,
        "excess_corr": excess,
        "tracking_error": te,
        "beta": beta,
        "n_obs": n_obs,
    }
