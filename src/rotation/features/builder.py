"""计算候选因子 → 因果预处理 → FeatureSet（带缓存）。"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from rotation.config import Config
from rotation.data.cache import Cache
from rotation.data.industry import IndustryData
from rotation.features.preprocess import preprocess_cube, preprocess_market, to_cube
from rotation.features.registry import FactorContext, all_market_features, select_factor_specs
from rotation.utils.hashing import stable_hash
from rotation.utils.logging import get_logger

log = get_logger("features")
_BUILDER_VERSION = 2


@dataclass
class FeatureSet:
    dates: pd.DatetimeIndex
    industries: list[str]
    names: list[str]
    categories: dict[str, str]
    X: np.ndarray  # [T, N, F] float32，已标准化，缺失填 0
    valid: np.ndarray  # [T, N] bool
    market_names: list[str]
    market: np.ndarray  # [T, M] float32

    def subset(self, names: list[str], signs: dict[str, float] | None = None) -> FeatureSet:
        index = [self.names.index(n) for n in names]
        x = self.X[:, :, index].copy()
        if signs:
            x *= np.asarray([signs.get(n, 1.0) for n in names], dtype=np.float32)
        return replace(
            self, names=list(names), X=x, categories={n: self.categories[n] for n in names}
        )

    def truncate(self, end: pd.Timestamp) -> FeatureSet:
        keep = int(self.dates.searchsorted(end, side="right"))
        return replace(
            self,
            dates=self.dates[:keep],
            X=self.X[:keep],
            valid=self.valid[:keep],
            market=self.market[:keep],
        )


def _cache_key(
    cfg: Config, data: IndustryData, factor_names: list[str], market_names: list[str]
) -> str:
    feats = cfg.features.model_dump(mode="json")
    feats.pop("selection", None)
    feats.pop("pca_components", None)
    return stable_hash(
        {
            "v": _BUILDER_VERSION,
            "data": data.fingerprint,
            "features": feats,
            "factors": factor_names,
            "market": market_names,
        }
    )


def build_features(cfg: Config, data: IndustryData, cache: Cache | None = None) -> FeatureSet:
    fc = cfg.features
    specs = [
        s
        for s in select_factor_specs(fc.categories, fc.include, fc.exclude)
        if all(r in data for r in s.requires)
    ]
    market_specs = list(all_market_features().values()) if fc.use_market else []
    names = [s.name for s in specs]
    market_names = [m.name for m in market_specs]
    if not names:
        raise ValueError("没有可用的候选因子，请检查 features.categories")
    cache = cache or Cache(cfg.cache_dir)
    key = _cache_key(cfg, data, names, market_names)

    def builder() -> tuple[dict[str, np.ndarray], dict]:
        ctx = FactorContext(data)
        started = time.perf_counter()
        with np.errstate(invalid="ignore", divide="ignore"):
            frames = [spec.func(ctx) for spec in specs]
            market_frames = [m.func(ctx) for m in market_specs]
        cube = to_cube(frames, data.dates, data.industries)
        x, valid = preprocess_cube(
            cube,
            normalize=fc.normalize,
            ts_window=fc.ts_window,
            winsor_mad=fc.winsor_mad,
            clip=fc.clip,
            ffill_limit=fc.ffill_limit,
            min_valid_ratio=fc.min_valid_ratio,
        )
        market = preprocess_market(market_frames, data.dates, fc.market_ts_window, fc.clip)
        log.info(
            "因子计算完成：%d 个候选因子 + %d 个市场特征，用时 %.1fs",
            len(names),
            len(market_names),
            time.perf_counter() - started,
        )
        arrays = {"X": x, "valid": valid, "market": market, "dates": data.dates.asi8}
        return arrays, {"names": names, "market_names": market_names}

    arrays, meta = cache.arrays("features", key, builder)
    dates = pd.DatetimeIndex(pd.to_datetime(arrays["dates"])).as_unit("ns")
    if not dates.equals(data.dates):
        raise RuntimeError("特征缓存的日期与行业数据不一致，请删除 cache/ 后重试")
    categories = {s.name: s.category for s in specs}
    return FeatureSet(
        dates=dates,
        industries=list(data.industries),
        names=list(meta["names"]),
        categories=categories,
        X=arrays["X"],
        valid=arrays["valid"].astype(bool),
        market_names=list(meta["market_names"]),
        market=arrays["market"],
    )
