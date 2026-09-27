"""指标与统计检验：与 scipy / 手算结果对照。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rotation.evaluation.ranking import (
    daily_ic,
    daily_rank_ic,
    factor_rank_ic,
    quantile_returns,
    topk_excess,
    topk_hit_rate,
)
from rotation.evaluation.regression import regression_metrics
from rotation.evaluation.stats import diebold_mariano, newey_west_t
from rotation.features import indicators as ind


def test_rank_ic_matches_scipy():
    rng = np.random.default_rng(0)
    pred, y = rng.normal(size=(20, 30)), rng.normal(size=(20, 30))
    y[3, :4] = np.nan
    ours = daily_rank_ic(pred, y)
    for i in range(20):
        ok = np.isfinite(y[i])
        assert np.isclose(ours[i], stats.spearmanr(pred[i, ok], y[i, ok]).statistic)
    assert np.isclose(daily_ic(pred, y)[0], stats.pearsonr(pred[0], y[0]).statistic)


def test_factor_rank_ic_matches_single():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(15, 30, 3)), rng.normal(size=(15, 30))
    multi = factor_rank_ic(x, y)
    for f in range(3):
        np.testing.assert_allclose(multi[:, f], daily_rank_ic(x[:, :, f], y))


def test_topk_metrics():
    y = np.arange(10, dtype=float)[None, :]
    assert topk_hit_rate(y, y, 3)[0] == 1.0
    assert topk_hit_rate(-y, y, 3)[0] == 0.0
    assert np.isclose(topk_excess(y, y, 2)[0], (9 + 8) / 2 - 4.5)
    groups = quantile_returns(y, y, 5)
    assert groups[0, 0] > groups[0, -1]


def test_newey_west_close_to_iid_t():
    x = np.random.default_rng(2).normal(0.1, 1.0, 2000)
    classic = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(newey_west_t(x, 0) - classic) / abs(classic) < 0.01


def test_diebold_mariano_direction():
    rng = np.random.default_rng(3)
    good, bad = rng.normal(0, 1, 500) ** 2, rng.normal(0, 2, 500) ** 2
    assert diebold_mariano(good, bad, 1)["dm"] < 0
    assert diebold_mariano(bad, good, 1)["dm"] > 0


def test_regression_metrics_r2_oos():
    y = np.array([0.1, -0.1, 0.2, -0.2])
    perfect = regression_metrics(y, y)
    assert perfect["r2_oos"] == 1.0 and perfect["direction_acc"] == 1.0
    assert regression_metrics(np.zeros(4), y)["r2_oos"] == 0.0


def test_wma_and_rsi():
    s = pd.DataFrame({"a": np.arange(1.0, 21.0)})
    w = ind.wma(s, 4)
    manual = s["a"].rolling(4).apply(lambda v: np.dot(v, [1, 2, 3, 4]) / 10, raw=True)
    pd.testing.assert_series_equal(w["a"], manual, check_names=False)
    assert ind.rsi(s, 6)["a"].iloc[-1] > 99.0  # 单调上涨 → RSI 接近 100
    tr = ind.true_range(s + 1, s - 1, s)
    assert tr["a"].iloc[0] == 2.0
