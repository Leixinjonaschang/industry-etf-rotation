"""因子注册器：用装饰器登记因子（名称、类别、说明），按配置挑选。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from rotation.data.industry import IndustryData

CATEGORIES = ("momentum", "trend", "oscillator", "volume", "volatility", "valuation")
CATEGORY_CN = {
    "momentum": "动量",
    "trend": "趋势",
    "oscillator": "震荡",
    "volume": "量能",
    "volatility": "波动",
    "valuation": "估值",
    "market": "市场状态",
}


class FactorContext:
    """因子计算上下文：缓存各因子共用的中间量（日收益、市场收益等）。"""

    def __init__(self, data: IndustryData):
        self.data = data

    def field(self, name: str) -> pd.DataFrame:
        return self.data[name]

    @property
    def close(self) -> pd.DataFrame:
        return self.data["close"]

    @property
    def open(self) -> pd.DataFrame:
        return self.data["open"]

    @property
    def high(self) -> pd.DataFrame:
        return self.data["high"]

    @property
    def low(self) -> pd.DataFrame:
        return self.data["low"]

    @property
    def volume(self) -> pd.DataFrame:
        return self.data["volume"]

    @property
    def amount(self) -> pd.DataFrame:
        return self.data["amount"]

    @cached_property
    def ret1(self) -> pd.DataFrame:
        return self.close / self.close.shift(1) - 1.0

    @cached_property
    def market_ret1(self) -> pd.Series:
        """30 行业等权日收益。"""
        return self.ret1.mean(axis=1, skipna=True)

    @cached_property
    def market_level(self) -> pd.Series:
        return (1.0 + self.market_ret1.fillna(0.0)).cumprod()


@dataclass(frozen=True)
class FactorSpec:
    name: str
    category: str
    func: Callable[[FactorContext], pd.DataFrame]
    description: str = ""
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketSpec:
    name: str
    func: Callable[[FactorContext], pd.Series]
    description: str = ""


_FACTORS: dict[str, FactorSpec] = {}
_MARKET: dict[str, MarketSpec] = {}


def factor(name: str, category: str, description: str = "", requires: tuple[str, ...] = ()):
    if category not in CATEGORIES:
        raise ValueError(f"未知因子类别：{category}")

    def decorator(func: Callable[[FactorContext], pd.DataFrame]):
        if name in _FACTORS:
            raise ValueError(f"因子重复注册：{name}")
        _FACTORS[name] = FactorSpec(name, category, func, description, requires)
        return func

    return decorator


def market_feature(name: str, description: str = ""):
    def decorator(func: Callable[[FactorContext], pd.Series]):
        if name in _MARKET:
            raise ValueError(f"市场特征重复注册：{name}")
        _MARKET[name] = MarketSpec(name, func, description)
        return func

    return decorator


def _ensure_loaded() -> None:
    # 导入即完成注册
    from rotation.features import market, technical, valuation  # noqa: F401


def all_factors() -> dict[str, FactorSpec]:
    _ensure_loaded()
    return dict(_FACTORS)


def all_market_features() -> dict[str, MarketSpec]:
    _ensure_loaded()
    return dict(_MARKET)


def select_factor_specs(
    categories: list[str], include: list[str] | None = None, exclude: list[str] | None = None
) -> list[FactorSpec]:
    specs = all_factors()
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        raise ValueError(f"未知因子类别：{unknown}（可选 {CATEGORIES}）")
    names = [n for n, s in specs.items() if s.category in categories]
    for name in include or []:
        if name not in specs:
            raise ValueError(f"未注册的因子：{name}")
        if name not in names:
            names.append(name)
    names = [n for n in names if n not in set(exclude or [])]
    return [specs[n] for n in names]
