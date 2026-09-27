"""因子筛选：聚类规则、簇内决策、PCA 合成的方向与因果性、筛选期防泄露。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rotation.config import SelectionConfig, load_config
from rotation.data.cache import Cache
from rotation.features.builder import FeatureSet, build_features
from rotation.features.selection import (
    apply_selection,
    correlation_clusters,
    fit_composite,
    plan_clusters,
    select_factors,
)
from rotation.labels import Labels, make_labels
from tests.conftest import PROJECT_ROOT
from tests.test_leakage import _perturb_industry


def test_clusters_are_complete_linkage():
    # a–b、b–c 高相关，但 a–c 不够高：全链接不能把三者串成一簇
    corr = np.array(
        [
            [1.0, 0.85, 0.5, 0.0],
            [0.85, 1.0, 0.85, 0.0],
            [0.5, 0.85, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    labels = correlation_clusters(corr, 0.8)
    assert len(set(labels[:3])) == 2
    assert labels[3] not in set(labels[:3])
    # 负相关同样算高相关
    corr2 = np.array([[1.0, -0.9], [-0.9, 1.0]])
    labels2 = correlation_clusters(corr2, 0.8)
    assert labels2[0] == labels2[1]


def test_plan_clusters_rules():
    names = ["a", "b", "c", "d", "e", "f"]
    labels = np.array([0, 0, 0, 1, 1, 2])
    score = pd.Series([0.9, 0.8, 0.1, 0.7, 0.2, 0.05], index=names)
    high = score >= 0.5
    plans = {p["cluster"]: p for p in plan_clusters(names, labels, score, high)}
    assert plans["C01"]["action"] == "pca" and plans["C01"]["members"] == ["a", "b", "c"]
    assert plans["C02"]["action"] == "keep" and plans["C02"]["members"][0] == "d"
    assert plans["C03"]["action"] == "keep" and plans["C03"]["n_high"] == 0


def _synthetic(seed: int = 0, t: int = 400, n: int = 30):
    """s1、s2、s3 高度相关且都与标签负相关（应合成）；u 独立有效（应单独保留）；其余为噪声。"""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=(t, n))
    u = rng.normal(size=(t, n))
    y = -0.4 * sig + 0.4 * u + rng.normal(size=(t, n))
    feats = {
        "s1": sig + 0.1 * rng.normal(size=(t, n)),
        "s2": sig + 0.1 * rng.normal(size=(t, n)),
        "s3": sig + 0.1 * rng.normal(size=(t, n)),
        "u": u,
    }
    for k in range(8):
        feats[f"n{k}"] = rng.normal(size=(t, n))
    names = list(feats)
    x = np.stack([feats[k] for k in names], axis=2).astype(np.float32)
    dates = pd.bdate_range("2015-01-01", periods=t)
    fs = FeatureSet(
        dates=dates,
        industries=[f"i{j}" for j in range(n)],
        names=names,
        categories={k: "trend" for k in names},
        X=x,
        valid=np.ones((t, n), dtype=bool),
        market_names=[],
        market=np.zeros((t, 0), dtype=np.float32),
    )
    labels = Labels(
        y=y,
        y_raw=y,
        known_pos=np.arange(t) + 6,
        horizon=5,
        kind="excess",
        price="oo",
    )
    return fs, labels


def test_select_factors_merges_correlated_cluster():
    fs, labels = _synthetic()
    cfg = SelectionConfig(high_importance_frac=0.4, drop_bottom_frac=0.25, tree_max_samples=5000)
    sel = select_factors(fs, labels, cfg, fs.dates[0], fs.dates[-1])
    assert len(sel.composites) == 1
    comp = sel.composites[0]
    assert sorted(comp.members) == ["s1", "s2", "s3"]
    assert comp.explained_ratio > 0.95
    assert "u" in sel.selected
    assert not {"s1", "s2", "s3"} & set(sel.selected)
    # 末位剔除：25% × 12 = 3 个
    assert (sel.table["reason"].str.contains("综合分排名末")).sum() >= 1
    # 合成因子方向：与标签 RankIC 为正（原始因子与标签负相关）
    assert sel.table.at[comp.name, "ic_mean"] > 0
    assert all(w < 0 for w in comp.loadings)
    out = apply_selection(fs, sel)
    assert out.names == sel.feature_names
    assert out.X.shape[2] == len(sel.feature_names)
    np.testing.assert_allclose(
        out.X[:, :, out.names.index(comp.name)],
        comp.transform(fs.X[:, :, [fs.names.index(m) for m in comp.members]]),
        rtol=1e-5,
    )


def test_fit_composite_orientation_positive():
    fs, labels = _synthetic(seed=1)
    idx = [fs.names.index(k) for k in ("s1", "s2", "s3")]
    mask = np.ones(labels.y.shape, dtype=bool)
    w, center, scale, explained, stats = fit_composite(fs.X, labels.y, mask, idx, lags=5)
    assert stats["mean"] > 0
    comp = (fs.X[:, :, idx] - center) @ w / scale
    assert abs(comp.reshape(-1).std() - 1.0) < 0.05


def test_composite_transform_is_causal():
    fs, labels = _synthetic(seed=2)
    cfg = SelectionConfig(high_importance_frac=0.4, drop_bottom_frac=0.0, tree_max_samples=5000)
    sel = select_factors(fs, labels, cfg, fs.dates[0], fs.dates[200])
    assert sel.composites
    cut = 250
    x2 = fs.X.copy()
    x2[cut + 1 :] = np.random.default_rng(9).normal(size=x2[cut + 1 :].shape) * 10
    a = apply_selection(fs, sel).X
    b = apply_selection(replace(fs, X=x2), sel).X
    np.testing.assert_array_equal(a[: cut + 1], b[: cut + 1])


def test_selection_ignores_labels_after_end():
    fs, labels = _synthetic(seed=3)
    end_pos = 250
    cfg = SelectionConfig(high_importance_frac=0.4, tree_max_samples=5000)
    y2 = labels.y.copy()
    # 可观测日在 end 之后的标签全部打乱
    late = labels.known_pos > end_pos
    y2[late] = np.random.default_rng(4).normal(size=y2[late].shape) * 5
    a = select_factors(fs, labels, cfg, fs.dates[0], fs.dates[end_pos])
    b = select_factors(fs, replace(labels, y=y2), cfg, fs.dates[0], fs.dates[end_pos])
    assert a.feature_names == b.feature_names
    assert a.digest == b.digest


@pytest.mark.parametrize("method", ["importance", "ic_threshold"])
def test_selection_is_causal_on_market_data(tmp_path, industry, method):
    """改写 select_end 之后的行情：因子、标签重新计算后，筛选结果完全不变。"""
    cfg = load_config(
        PROJECT_ROOT / "configs" / "base.yaml",
        [
            "features.selection.enabled=true",
            f"features.selection.method={method}",
            "features.selection.min_factors=3",
            "features.selection.tree_max_samples=5000",
        ],
    )
    start, end = pd.Timestamp("2019-07-01"), pd.Timestamp("2020-12-31")
    cache = Cache(tmp_path, enabled=False)

    def run(data):
        fs = build_features(cfg, data, cache)
        labels = make_labels(data["open"], data["close"], 5, "excess", "oo")
        return select_factors(fs, labels, cfg.features.selection, start, end)

    a = run(industry)
    b = run(_perturb_industry(industry, end))
    assert a.feature_names == b.feature_names
    assert a.signs == b.signs
    for ca, cb in zip(a.composites, b.composites):
        assert ca.members == cb.members
        np.testing.assert_allclose(ca.loadings, cb.loadings, atol=1e-6)
