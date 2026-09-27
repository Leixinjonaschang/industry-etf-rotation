"""未来扰动测试：改写 t 日之后的数据，t 日及以前的结果必须完全不变。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rotation.config import MappingConfig
from rotation.data.etf import ETFData
from rotation.data.industry import IndustryData
from rotation.features.preprocess import preprocess_cube
from rotation.features.registry import FactorContext, all_factors, all_market_features
from rotation.mapping.mapper import ETFMapper
from rotation.mapping.whitelist import Whitelist
from tests.conftest import PROJECT_ROOT


def _perturb_industry(data: IndustryData, cut: pd.Timestamp, seed: int = 7) -> IndustryData:
    rng = np.random.default_rng(seed)
    fields = {}
    for name, frame in data.fields.items():
        new = frame.copy()
        after = new.index > cut
        new.loc[after] = new.loc[after].to_numpy() * rng.uniform(0.5, 1.5, new.loc[after].shape)
        fields[name] = new
    return IndustryData(
        fields=fields, industries=data.industries, codes=data.codes, fingerprint="perturbed"
    )


@pytest.mark.parametrize("name", sorted(all_factors()))
def test_factor_is_causal(industry, name):
    cut = industry.dates[500]
    spec = all_factors()[name]
    before = spec.func(FactorContext(industry))
    after = spec.func(FactorContext(_perturb_industry(industry, cut)))
    pd.testing.assert_frame_equal(
        before.loc[:cut], after.loc[:cut], check_exact=False, rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("name", sorted(all_market_features()))
def test_market_feature_is_causal(industry, name):
    cut = industry.dates[500]
    spec = all_market_features()[name]
    before = spec.func(FactorContext(industry))
    after = spec.func(FactorContext(_perturb_industry(industry, cut)))
    pd.testing.assert_series_equal(
        before.loc[:cut], after.loc[:cut], check_exact=False, rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("normalize", ["none", "ts", "cs", "ts+cs", "cs_rank"])
def test_preprocess_is_causal(normalize):
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(300, 12, 4))
    cube[rng.random(cube.shape) < 0.02] = np.nan
    cut = 200
    other = cube.copy()
    other[cut + 1 :] = rng.normal(size=other[cut + 1 :].shape) * 10
    kwargs = dict(
        normalize=normalize,
        ts_window=30,
        winsor_mad=5.0,
        clip=5.0,
        ffill_limit=3,
        min_valid_ratio=0.9,
    )
    x1, v1 = preprocess_cube(cube, **kwargs)
    x2, v2 = preprocess_cube(other, **kwargs)
    np.testing.assert_allclose(x1[: cut + 1], x2[: cut + 1], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(v1[: cut + 1], v2[: cut + 1])


def test_mapping_is_causal(industry, etf):
    cut_pos = 600
    cut = industry.dates[cut_pos]
    rng = np.random.default_rng(3)
    after = etf.close.index > cut

    def scramble(frame: pd.DataFrame) -> pd.DataFrame:
        new = frame.copy()
        new.loc[after] = new.loc[after].to_numpy() * rng.uniform(0.5, 1.5, new.loc[after].shape)
        return new

    etf2 = ETFData(
        master=etf.master,
        open=scramble(etf.open),
        close=scramble(etf.close),
        amount=scramble(etf.amount),
    )
    industry2 = _perturb_industry(industry, cut)
    whitelist = Whitelist.load(PROJECT_ROOT / "configs" / "etf_whitelist.yaml")
    cfg = MappingConfig(min_amount=1e6)
    m1 = ETFMapper(cfg, industry["close"], etf, whitelist, top_n=3)
    m2 = ETFMapper(cfg, industry2["close"], etf2, whitelist, top_n=3)
    for ind in industry.industries:
        a, reason_a = m1.options(cut_pos, ind)
        b, reason_b = m2.options(cut_pos, ind)
        assert reason_a == reason_b
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
