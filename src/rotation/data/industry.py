"""申万一级行业指数：长表 Excel → 宽表字典 {字段: DataFrame[日期 × 行业]}。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rotation.config import Config
from rotation.data.cache import Cache
from rotation.data.schema import (
    INDUSTRY_COLUMNS,
    INDUSTRY_REQUIRED,
    INDUSTRY_VALUE_FIELDS,
    PRICE_FIELDS,
    ns_index,
)
from rotation.utils.hashing import file_fingerprint, stable_hash
from rotation.utils.logging import get_logger

log = get_logger("data.industry")
_LOADER_VERSION = 1


@dataclass
class IndustryData:
    fields: dict[str, pd.DataFrame]
    industries: list[str]
    codes: dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""  # 源文件 + 加载参数的哈希，用于下游缓存键

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fields["close"].index

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    def truncate(self, end: pd.Timestamp) -> IndustryData:
        """只保留 end 及以前的数据（用于切断未来信息）。"""
        return IndustryData(
            fields={k: v.loc[:end].copy() for k, v in self.fields.items()},
            industries=list(self.industries),
            codes=dict(self.codes),
            fingerprint=stable_hash({"base": self.fingerprint, "end": str(end)}),
        )


def _read_long_table(cfg: Config) -> pd.DataFrame:
    path = cfg.data_dir / cfg.paths.industry_file
    if not path.exists():
        raise FileNotFoundError(
            f"找不到行业数据：{path}（可用环境变量 ROTATION_DATA_DIR 指定数据目录）"
        )
    raw = pd.read_excel(path, sheet_name=0)
    missing = [
        c for c, n in INDUSTRY_COLUMNS.items() if n in INDUSTRY_REQUIRED and c not in raw.columns
    ]
    if missing:
        raise ValueError(f"行业数据缺少列：{missing}")
    frame = raw[[c for c in INDUSTRY_COLUMNS if c in raw.columns]].rename(columns=INDUSTRY_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["industry"] = (
        frame["industry"].astype(str).str.strip().str.replace(r"^(申万|SW|sw)", "", regex=True)
    )
    if "code" in frame:
        frame["code"] = frame["code"].astype(str).str.strip()
    for name in INDUSTRY_VALUE_FIELDS:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
    frame = frame.dropna(subset=["date"])
    dup = frame.duplicated(["date", "industry"], keep="last")
    if dup.any():
        log.warning("行业数据存在 %d 条重复的 日期×行业 记录，保留最后一条", int(dup.sum()))
        frame = frame.loc[~dup]
    return frame.reset_index(drop=True)


def load_industry_data(cfg: Config, cache: Cache | None = None) -> IndustryData:
    path = cfg.data_dir / cfg.paths.industry_file
    key = stable_hash({"v": _LOADER_VERSION, "file": file_fingerprint(path)})
    cache = cache or Cache(cfg.cache_dir)
    long = cache.frames("industry_long", key, lambda: {"long": _read_long_table(cfg)})["long"]

    long = long[~long["industry"].isin(cfg.data.exclude_industries)]
    start, end = pd.Timestamp(cfg.data.load_start), pd.Timestamp(cfg.data.test_end)
    long = long[(long["date"] >= start) & (long["date"] <= end)]

    # 行业顺序按申万代码排序，保证结果可复现
    if "code" in long:
        codes = long.groupby("industry")["code"].first().to_dict()
        industries = sorted(codes, key=lambda name: codes[name])
    else:
        codes = {}
        industries = sorted(long["industry"].unique())

    fields: dict[str, pd.DataFrame] = {}
    for name in INDUSTRY_VALUE_FIELDS:
        if name not in long:
            continue
        wide = long.pivot(index="date", columns="industry", values=name)
        wide.index = ns_index(wide.index)
        wide = wide.sort_index().reindex(columns=industries)
        wide.columns.name = None
        wide.index.name = "date"
        if name in PRICE_FIELDS:
            wide = wide.where(wide > 0)
        fields[name] = wide.astype("float64")

    calendar = fields["close"].index
    for name in fields:
        fields[name] = fields[name].reindex(calendar)

    empty = [ind for ind in industries if fields["close"][ind].isna().all()]
    if empty:
        raise ValueError(f"以下行业没有收盘价：{empty}")
    log.info(
        "行业数据：%s ~ %s，%d 个交易日 × %d 个行业，字段=%s",
        calendar.min().date(),
        calendar.max().date(),
        len(calendar),
        len(industries),
        list(fields),
    )
    fingerprint = stable_hash(
        {
            "file": key,
            "exclude": cfg.data.exclude_industries,
            "start": str(start.date()),
            "end": str(end.date()),
        }
    )
    return IndustryData(fields=fields, industries=industries, codes=codes, fingerprint=fingerprint)


def equal_weight_market_return(close: pd.DataFrame) -> pd.Series:
    """30 行业等权日收益（收盘到收盘），作为"市场"收益。"""
    ret = close / close.shift(1) - 1.0
    return ret.mean(axis=1, skipna=True).rename("market")


def daily_return(frame: pd.DataFrame) -> pd.DataFrame:
    return frame / frame.shift(1) - 1.0


def safe_log(frame: pd.DataFrame) -> pd.DataFrame:
    return np.log(frame.where(frame > 0))
