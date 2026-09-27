"""截面排序指标（逐日期计算）。

输入统一为 [P, N]（P 个日期 × N 个行业），mask 为 False 的位置不参与计算。
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import rankdata

from rotation.evaluation.stats import newey_west_t, two_sided_p

MIN_CROSS_SECTION = 5


def _apply_mask(a: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if mask is None:
        mask = np.isfinite(a)
    return np.where(mask & np.isfinite(a), a, np.nan)


def _rowwise_pearson(
    a: np.ndarray, b: np.ndarray, min_count: int = MIN_CROSS_SECTION
) -> np.ndarray:
    """沿 axis=1 计算 Pearson，a/b 中 NaN 视为缺失（二者缺失位置需一致）。"""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        count = np.sum(np.isfinite(a) & np.isfinite(b), axis=1)
        am = a - np.nanmean(a, axis=1, keepdims=True)
        bm = b - np.nanmean(b, axis=1, keepdims=True)
        num = np.nansum(am * bm, axis=1)
        den = np.sqrt(np.nansum(am * am, axis=1) * np.nansum(bm * bm, axis=1))
        corr = num / den
    corr[(count < min_count) | ~np.isfinite(corr) | (den <= 1e-15)] = np.nan
    return corr


def _joint(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=float)
    y = np.asarray(target, dtype=float)
    joint = np.isfinite(p) & np.isfinite(y)
    if mask is not None:
        joint &= mask
    return np.where(joint, p, np.nan), np.where(joint, y, np.nan)


def daily_ic(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    p, y = _joint(pred, target, mask)
    return _rowwise_pearson(p, y)


def daily_rank_ic(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    p, y = _joint(pred, target, mask)
    rp = rankdata(p, axis=1, nan_policy="omit")
    ry = rankdata(y, axis=1, nan_policy="omit")
    return _rowwise_pearson(rp, ry)


def factor_rank_ic(x: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """多个因子一次计算：x [P, N, F] → RankIC [P, F]。"""
    y = np.asarray(target, dtype=float)
    joint = np.isfinite(y) if mask is None else (mask & np.isfinite(y))
    out = np.full((x.shape[0], x.shape[2]), np.nan)
    ry_all = rankdata(np.where(joint, y, np.nan), axis=1, nan_policy="omit")
    for f in range(x.shape[2]):
        joint_f = joint & np.isfinite(x[:, :, f])
        rx = rankdata(np.where(joint_f, x[:, :, f], np.nan), axis=1, nan_policy="omit")
        if (joint_f == joint).all():
            ry = ry_all
        else:  # 该因子有额外缺失时，在共同有效集合上重新排名
            ry = rankdata(np.where(joint_f, y, np.nan), axis=1, nan_policy="omit")
        out[:, f] = _rowwise_pearson(rx, ry)
    return out


def ic_summary(values: np.ndarray, lags: int) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {
            "mean": np.nan,
            "std": np.nan,
            "icir": np.nan,
            "win_rate": np.nan,
            "t": np.nan,
            "p": np.nan,
            "n": len(x),
        }
    mean, std = float(x.mean()), float(x.std(ddof=1))
    t_value = newey_west_t(x, lags)
    return {
        "mean": mean,
        "std": std,
        "icir": mean / std if std > 0 else np.nan,
        "win_rate": float((x > 0).mean()),
        "t": t_value,
        "p": two_sided_p(t_value),
        "n": int(len(x)),
    }


def _top_k_indicator(values: np.ndarray, k: int) -> np.ndarray:
    """每行取最大的 k 个（NaN 不参与），返回布尔矩阵。"""
    filled = np.where(np.isfinite(values), values, -np.inf)
    order = np.argsort(-filled, axis=1, kind="stable")
    indicator = np.zeros(values.shape, dtype=bool)
    rows = np.arange(values.shape[0])[:, None]
    indicator[rows, order[:, :k]] = True
    return indicator & np.isfinite(values)


def topk_hit_rate(
    pred: np.ndarray, target: np.ndarray, k: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """precision@k：预测前 k 与真实前 k 的交集 / k。随机水平为 k/N。"""
    p, y = _joint(pred, target, mask)
    valid_rows = np.sum(np.isfinite(p), axis=1) >= max(k * 2, MIN_CROSS_SECTION)
    hits = (_top_k_indicator(p, k) & _top_k_indicator(y, k)).sum(axis=1) / k
    return np.where(valid_rows, hits, np.nan)


def topk_excess(
    pred: np.ndarray, target: np.ndarray, k: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """预测前 k 的平均实际收益 − 截面平均实际收益。"""
    p, y = _joint(pred, target, mask)
    top = _top_k_indicator(p, k)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        top_mean = np.nansum(np.where(top, y, 0.0), axis=1) / np.maximum(top.sum(axis=1), 1)
        all_mean = np.nanmean(y, axis=1)
    out = top_mean - all_mean
    out[top.sum(axis=1) < k] = np.nan
    return out


def ndcg_at_k(
    pred: np.ndarray, target: np.ndarray, k: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """以真实收益的截面排名 (0..1) 作为相关度的 NDCG@k。"""
    p, y = _joint(pred, target, mask)
    out = np.full(p.shape[0], np.nan)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for i in range(p.shape[0]):
        ok = np.isfinite(p[i]) & np.isfinite(y[i])
        if ok.sum() < max(k, MIN_CROSS_SECTION):
            continue
        rel = (rankdata(y[i][ok]) - 1.0) / (ok.sum() - 1.0)
        order = np.argsort(-p[i][ok], kind="stable")[:k]
        ideal = np.sort(rel)[::-1][:k]
        idcg = float(ideal @ discounts)
        out[i] = float(rel[order] @ discounts) / idcg if idcg > 0 else np.nan
    return out


def quantile_returns(
    pred: np.ndarray, target: np.ndarray, n_groups: int = 5, mask: np.ndarray | None = None
) -> np.ndarray:
    """按预测值等分为 n_groups 组（第 1 组预测最高），返回每期各组平均实际收益 [P, G]。"""
    p, y = _joint(pred, target, mask)
    out = np.full((p.shape[0], n_groups), np.nan)
    for i in range(p.shape[0]):
        ok = np.isfinite(p[i]) & np.isfinite(y[i])
        m = int(ok.sum())
        if m < n_groups:
            continue
        order = np.argsort(-p[i][ok], kind="stable")
        groups = np.array_split(order, n_groups)
        yi = y[i][ok]
        out[i] = [yi[g].mean() for g in groups]
    return out
