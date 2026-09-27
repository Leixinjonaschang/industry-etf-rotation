"""标签对齐与 walk-forward 划分。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rotation.config import SplitConfig
from rotation.dataset.panel import Panel
from rotation.dataset.splits import make_folds
from rotation.labels import make_labels


def _prices(n: int = 12, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.bdate_range("2021-01-04", periods=n)
    open_ = pd.DataFrame(
        np.arange(1, n + 1)[:, None] * np.arange(1, k + 1)[None, :] * 1.0, index=idx
    )
    close = open_ * 1.01
    return open_, close


def test_oo_label_values_and_known_pos():
    open_, close = _prices()
    lab = make_labels(open_, close, horizon=2, kind="raw", price="oo")
    # y_t = Open[t+3] / Open[t+1] - 1
    t = 4
    expected = open_.iloc[t + 3] / open_.iloc[t + 1] - 1
    np.testing.assert_allclose(lab.y[t], expected.to_numpy(), rtol=1e-6)
    assert lab.known_pos[t] == t + 3
    assert np.isnan(lab.y[-3:]).all()  # 末尾 h+1 天没有标签


def test_cc_label_values():
    open_, close = _prices()
    lab = make_labels(open_, close, horizon=3, kind="raw", price="cc")
    t = 2
    np.testing.assert_allclose(
        lab.y[t], (close.iloc[t + 3] / close.iloc[t] - 1).to_numpy(), rtol=1e-6
    )
    assert lab.known_pos[t] == t + 3


def test_excess_label_sums_to_zero():
    open_, close = _prices()
    lab = make_labels(open_, close, horizon=2, kind="excess", price="oo")
    rows = np.isfinite(lab.y).all(axis=1)
    np.testing.assert_allclose(np.nansum(lab.y[rows], axis=1), 0.0, atol=1e-6)
    # 截面排序与 raw 一致
    raw = make_labels(open_, close, horizon=2, kind="raw", price="oo")
    np.testing.assert_array_equal(
        np.argsort(lab.y[rows], axis=1), np.argsort(raw.y_raw[rows], axis=1)
    )


def _panel(n: int = 900, k: int = 6, horizon: int = 5) -> Panel:
    dates = pd.bdate_range("2018-01-01", periods=n).as_unit("ns")
    rng = np.random.default_rng(0)
    y = rng.normal(size=(n, k)).astype(np.float32)
    y[-(horizon + 1) :] = np.nan
    return Panel(
        dates=dates,
        industries=[f"i{j}" for j in range(k)],
        feature_names=["f"],
        market_names=[],
        X=rng.normal(size=(n, k, 1)).astype(np.float32),
        M=np.zeros((n, 0), np.float32),
        valid=np.ones((n, k), bool),
        y=y,
        y_raw=y,
        known_pos=np.arange(n) + horizon + 1,
        close=np.ones((n, k)),
        horizon=horizon,
        label_kind="raw",
    )


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_folds_respect_purge(mode):
    panel = _panel()
    cfg = SplitConfig(mode=mode, train_window=300, retrain_every=40, val_len=60, min_train=100)
    folds = make_folds(
        panel, cfg, panel.dates[700], panel.dates[-1], train_start=panel.dates[20], min_history=10
    )
    seen_test = []
    for fold in folds:
        origin = panel.positions([fold.origin])[0]
        assert fold.test_pos[0] == origin
        assert len(fold.test_pos) <= 40
        assert (panel.known_pos[fold.val_pos] <= origin).all(), "验证标签必须在起点前可观测"
        assert (panel.known_pos[fold.train_pos] <= fold.val_pos[0]).all(), (
            "训练标签不能与验证期重叠"
        )
        assert fold.train_pos.max() < fold.val_pos.min() and fold.val_pos.max() < origin
        assert fold.train_pos.min() >= 20
        if mode == "rolling":
            assert len(fold.train_pos) <= 300
        seen_test.extend(fold.test_pos.tolist())
    assert seen_test == list(range(700, len(panel.dates)))


def test_folds_raise_when_too_little_history():
    panel = _panel(n=300)
    cfg = SplitConfig(retrain_every=40, val_len=60, min_train=500)
    with pytest.raises(ValueError):
        make_folds(panel, cfg, panel.dates[200], panel.dates[-1], train_start=panel.dates[0])
