"""ETF 映射与回测引擎。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rotation.backtest.engine import simulate
from rotation.backtest.metrics import performance
from rotation.config import MappingConfig
from rotation.mapping.mapper import ETFMapper
from rotation.mapping.whitelist import Whitelist
from tests.conftest import PROJECT_ROOT


@pytest.fixture(scope="module")
def whitelist() -> Whitelist:
    return Whitelist.load(PROJECT_ROOT / "configs" / "etf_whitelist.yaml")


def test_whitelist_semantics(whitelist):
    assert whitelist.matches("钢铁", "x", "钢铁ETF", "中证钢铁指数")
    assert whitelist.matches("基础化工", "x", "某ETF", "中证全指原材料指数")  # include_index_names
    assert not whitelist.matches("电子", "x", "央企科技ETF", "中证央企电子指数")  # 全局排除
    assert not whitelist.matches("房地产", "x", "金融地产ETF", "中证金融地产指数")  # 行业排除
    assert not whitelist.matches("电力设备", "x", "新能源汽车ETF", "中证新能源汽车指数")


def test_mapping_picks_liquid_representative(industry, etf, whitelist):
    mapper = ETFMapper(MappingConfig(), industry["close"], etf, whitelist, top_n=3)
    t = 400
    options, reason = mapper.options(t, "有色金属")
    assert reason == ""
    assert options.iloc[0]["code"] == "510004"  # 同一跟踪指数中成交额更大的那只
    assert "510005" not in set(options["code"])  # 流动性不足被过滤
    assert options.iloc[0]["pearson"] > 0.9


def test_unique_assignment_and_fallback(industry, etf, whitelist):
    mapper = ETFMapper(MappingConfig(), industry["close"], etf, whitelist, top_n=3)
    t = 400
    # "综合"不存在 → 无候选；银行、钢铁、有色各自有 ETF
    priority = ["不存在的行业", "银行", "钢铁", "有色金属", "电子"]
    ranks = {ind: i + 1 for i, ind in enumerate(priority)}
    slots, attempts = mapper.map_date(t, priority, ranks, prev={})
    codes = [s["etf_code"] for s in slots]
    assert len(slots) == 3 and len(set(codes)) == 3
    assert [s["industry"] for s in slots] == ["银行", "钢铁", "有色金属"]
    assert slots[2]["status"] == "fallback_next"  # 第 4 名顺延进入
    assert attempts[0]["status"] == "unmappable"


def test_mapping_cash_fallback(industry, etf, whitelist):
    mapper = ETFMapper(MappingConfig(fallback="cash"), industry["close"], etf, whitelist, top_n=2)
    slots, _ = mapper.map_date(400, ["不存在的行业", "银行", "钢铁"], {}, prev={})
    assert [s["status"] for s in slots] == ["cash", "mapped"]


def test_simulate_constant_growth_and_costs():
    dates = pd.bdate_range("2022-01-03", periods=30).as_unit("ns")
    px = pd.DataFrame({"A": 100 * 1.01 ** np.arange(30)}, index=dates)
    targets = {dates[1]: {"A": 1.0}}
    res = simulate(targets, px, px, dates[0], dates[-1], cost_bps=0.0)
    # 在 dates[1] 开盘（=收盘价）买入，之后每天 +1%
    assert np.isclose(res.nav.iloc[-1], 1.01**28)
    res_cost = simulate(targets, px, px, dates[0], dates[-1], cost_bps=10.0)
    assert np.isclose(res_cost.nav.iloc[-1], res.nav.iloc[-1] * (1 - 10 / 1e4), rtol=1e-6)


def test_simulate_missing_open_goes_to_cash():
    dates = pd.bdate_range("2022-01-03", periods=10).as_unit("ns")
    open_ = pd.DataFrame({"A": 1.0, "B": 1.0}, index=dates)
    close = open_.copy()
    open_.loc[dates[1], "B"] = np.nan  # B 在调仓日停牌
    res = simulate(
        {dates[1]: {"A": 0.5, "B": 0.5}}, open_, close, dates[0], dates[-1], cost_bps=0.0
    )
    assert np.isclose(res.trades.iloc[0]["cash_weight"], 0.5)


def test_performance_basic():
    dates = pd.bdate_range("2022-01-03", periods=253).as_unit("ns")
    nav = pd.Series(1.1 ** (np.arange(253) / 252), index=dates)
    perf = performance(nav, 0.0)
    assert np.isclose(perf["annual_return"], 0.1, atol=1e-6)
    assert perf["max_drawdown"] == 0.0
