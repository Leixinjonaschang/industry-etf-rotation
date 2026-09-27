"""ETF 名单（行业/主题两张表）与后复权日线。

注意：
- 名单中的"基金规模（最新）""年成交量（1 年前）"是当前时点快照，属于前视信息，这里不读取；
- 名单只包含目前仍存续的 ETF（幸存者偏差），需在论文局限性中说明；
- 数据源在 `data.etf_amount_unit_change` 之前的成交额单位为千元，这里统一换算为元。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rotation.config import Config
from rotation.data.cache import Cache
from rotation.data.schema import (
    ETF_DAILY_COLUMNS,
    ETF_FIELDS,
    ETF_LIST_COLUMNS,
    TIER_BENCHMARK,
    TIER_INDUSTRY,
    TIER_THEME,
    ns_index,
)
from rotation.utils.hashing import directory_fingerprint, file_fingerprint, stable_hash
from rotation.utils.logging import get_logger

log = get_logger("data.etf")
_LOADER_VERSION = 1
_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


@dataclass
class ETFData:
    master: pd.DataFrame  # index=6 位代码；列 name/exchange/index_code/index_name/listing_date/tier
    open: pd.DataFrame  # 日期 × 代码（后复权）
    close: pd.DataFrame
    amount: pd.DataFrame  # 元

    @property
    def codes(self) -> list[str]:
        return list(self.close.columns)

    def truncate(self, end: pd.Timestamp) -> ETFData:
        return ETFData(
            master=self.master.copy(),
            open=self.open.loc[:end].copy(),
            close=self.close.loc[:end].copy(),
            amount=self.amount.loc[:end].copy(),
        )

    def reindex(self, calendar: pd.DatetimeIndex) -> ETFData:
        return ETFData(
            master=self.master,
            open=self.open.reindex(calendar),
            close=self.close.reindex(calendar),
            amount=self.amount.reindex(calendar),
        )


def normalize_code(value: object) -> str:
    match = _CODE_RE.search(str(value))
    return match.group(1) if match else ""


def _parse_yyyymmdd(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").round()
    text = numeric.astype("Int64").astype(str)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def load_etf_master(cfg: Config) -> pd.DataFrame:
    parts = []
    for tier, filename in [
        (TIER_INDUSTRY, cfg.paths.etf_industry_list),
        (TIER_THEME, cfg.paths.etf_theme_list),
    ]:
        path = cfg.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"找不到 ETF 名单：{path}")
        raw = pd.read_excel(path, sheet_name=0, dtype={"证券代码": str, "跟踪指数代码": str})
        missing = [c for c in ETF_LIST_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"{path.name} 缺少列：{missing}")
        part = raw[list(ETF_LIST_COLUMNS)].rename(columns=ETF_LIST_COLUMNS)
        part["code"] = part["code_full"].map(normalize_code)
        part["exchange"] = part["code_full"].astype(str).str.extract(r"\.(SH|SZ)$", expand=False)
        part["listing_date"] = _parse_yyyymmdd(part["listing_date"])
        part["tier"] = tier
        part["name"] = part["name"].astype(str).str.strip()
        part["index_code"] = part["index_code"].astype(str).str.strip()
        part["index_name"] = part["index_name"].astype(str).str.strip()
        part = part[(part["code"].str.len() == 6) & part["listing_date"].notna()]
        parts.append(part)
    master = pd.concat(parts, ignore_index=True)
    master["_priority"] = master["tier"].map({TIER_INDUSTRY: 0, TIER_THEME: 1})
    master = (
        master.sort_values(["code", "_priority"])
        .drop_duplicates("code", keep="first")
        .drop(columns=["_priority", "code_full"])
        .set_index("code")
        .sort_index()
    )
    return master[["name", "exchange", "index_code", "index_name", "listing_date", "tier"]]


def _file_map(daily_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(daily_dir.glob("*.csv")):
        code = normalize_code(path.name)
        if code:
            mapping.setdefault(code, path)
    return mapping


def _read_daily(path: Path) -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(path, encoding="utf-8-sig", usecols=["名称", *ETF_DAILY_COLUMNS])
    name = str(raw["名称"].iloc[0]) if len(raw) else ""
    frame = raw.drop(columns=["名称"]).rename(columns=ETF_DAILY_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date")
    frame.index = ns_index(frame.index)
    for col in ETF_FIELDS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    frame.loc[frame["open"] <= 0, "open"] = float("nan")
    frame.loc[frame["close"] <= 0, "close"] = float("nan")
    return frame.sort_index(), name


def _build_daily(cfg: Config, codes: list[str]) -> dict[str, pd.DataFrame]:
    files = _file_map(cfg.data_dir / cfg.paths.etf_daily_dir)
    series: dict[str, dict[str, pd.Series]] = {f: {} for f in ETF_FIELDS}
    names: dict[str, str] = {}
    missing = []
    for code in codes:
        path = files.get(code)
        if path is None:
            missing.append(code)
            continue
        frame, name = _read_daily(path)
        names[code] = name
        for f in ETF_FIELDS:
            series[f][code] = frame[f]
    if missing:
        log.warning("%d 只 ETF 缺少行情文件（已跳过）：%s", len(missing), missing[:10])
    frames = {}
    for f in ETF_FIELDS:
        wide = pd.DataFrame(series[f]).sort_index()
        wide.index.name = "date"
        frames[f] = wide
    frames["names"] = pd.DataFrame({"name": pd.Series(names)})
    return frames


def load_raw_daily(
    cfg: Config, master: pd.DataFrame, cache: Cache | None = None
) -> dict[str, pd.DataFrame]:
    """读取（并缓存）全部所需 ETF 的原始日线，成交额保持数据源原始单位。"""
    benchmark_codes = [normalize_code(c) for c in cfg.data.benchmark_codes]
    wanted = sorted(
        set(master.index) | set(benchmark_codes) | {normalize_code(cfg.mapping.broad_code)}
    )
    daily_dir = cfg.data_dir / cfg.paths.etf_daily_dir
    if not daily_dir.exists():
        raise FileNotFoundError(f"找不到 ETF 日线目录：{daily_dir}")
    key = stable_hash(
        {
            "v": _LOADER_VERSION,
            "dir": directory_fingerprint(daily_dir),
            "lists": [
                file_fingerprint(cfg.data_dir / cfg.paths.etf_industry_list),
                file_fingerprint(cfg.data_dir / cfg.paths.etf_theme_list),
            ],
            "codes": wanted,
        }
    )
    cache = cache or Cache(cfg.cache_dir)
    return cache.frames("etf_daily", key, lambda: _build_daily(cfg, wanted))


def amount_unit_ratio(cfg: Config, raw: dict[str, pd.DataFrame], window: int = 40) -> float:
    """单位切换日前后各 window 个交易日成交额中位数之比的截面中位数（约 1000 说明千元→元）。"""
    amount = raw["amount"].copy()
    amount.index = ns_index(amount.index)
    change = pd.Timestamp(cfg.data.etf_amount_unit_change)
    before = amount.loc[amount.index < change].tail(window).median()
    after = amount.loc[amount.index >= change].head(window).median()
    ratio = (after / before).replace([float("inf"), -float("inf")], float("nan")).dropna()
    return float(ratio.median()) if len(ratio) else float("nan")


def load_etf_data(cfg: Config, calendar: pd.DatetimeIndex, cache: Cache | None = None) -> ETFData:
    master = load_etf_master(cfg)
    frames = load_raw_daily(cfg, master, cache)
    names = frames["names"]["name"].astype(str).to_dict()
    wanted = list(names)

    # 基准（如 510300）不在行业/主题名单中，单独补一行主数据
    extra = [c for c in wanted if c not in master.index and c in names]
    if extra:
        bench = pd.DataFrame(
            {
                "name": [names[c] for c in extra],
                "exchange": [None] * len(extra),
                "index_code": [""] * len(extra),
                "index_name": [""] * len(extra),
                "listing_date": [pd.NaT] * len(extra),
                "tier": [TIER_BENCHMARK] * len(extra),
            },
            index=pd.Index(extra, name="code"),
        )
        master = pd.concat([master, bench])

    out: dict[str, pd.DataFrame] = {}
    change = pd.Timestamp(cfg.data.etf_amount_unit_change)
    for f in ETF_FIELDS:
        wide = frames[f].copy()
        wide.index = ns_index(wide.index)
        wide.columns = [str(c) for c in wide.columns]
        if f == "amount":
            wide.loc[wide.index < change] *= cfg.data.etf_amount_multiplier_before
        out[f] = wide.reindex(calendar)

    available = [c for c in master.index if c in out["close"].columns]
    master = master.loc[available]
    # 上市日期缺失（基准）时，用第一条有效收盘价日期代替
    first_valid = out["close"].apply(lambda s: s.first_valid_index())
    master["listing_date"] = master["listing_date"].fillna(first_valid.reindex(master.index))
    data = ETFData(
        master=master,
        open=out["open"][available],
        close=out["close"][available],
        amount=out["amount"][available],
    )
    n_ind = int((master["tier"] == TIER_INDUSTRY).sum())
    n_theme = int((master["tier"] == TIER_THEME).sum())
    log.info("ETF：%d 只（行业 %d / 主题 %d / 基准 %d）", len(master), n_ind, n_theme, len(extra))
    return data
