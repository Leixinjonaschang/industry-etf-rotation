"""原始数据的中文列名 → 内部字段名。"""

from __future__ import annotations

import pandas as pd

INDUSTRY_COLUMNS: dict[str, str] = {
    "日期": "date",
    "行业名称": "industry",
    "行业代码": "code",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "市盈率PE": "pe",
    "市净率PB": "pb",
    "市销率PS": "ps",
    "股息率": "dy",
}
INDUSTRY_REQUIRED = ("date", "industry", "open", "high", "low", "close", "volume", "amount")
INDUSTRY_VALUE_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "pe",
    "pb",
    "ps",
    "dy",
)
PRICE_FIELDS = ("open", "high", "low", "close")

ETF_LIST_COLUMNS: dict[str, str] = {
    "证券代码": "code_full",
    "证券名称": "name",
    "跟踪指数代码": "index_code",
    "跟踪指数名称": "index_name",
    "上市日期": "listing_date",
}
ETF_DAILY_COLUMNS: dict[str, str] = {
    "日期": "date",
    "开盘价": "open",
    "收盘价": "close",
    "成交额": "amount",
}
ETF_FIELDS = ("open", "close", "amount")

TIER_INDUSTRY = "industry"
TIER_THEME = "theme"
TIER_BENCHMARK = "benchmark"
TIER_CN = {TIER_INDUSTRY: "行业", TIER_THEME: "主题", TIER_BENCHMARK: "基准"}


def ns_index(index: pd.Index) -> pd.DatetimeIndex:
    """统一为纳秒精度、无时区、已排序去重的 DatetimeIndex（兼容 pandas 2/3 的时间精度差异）。"""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.as_unit("ns").normalize()
