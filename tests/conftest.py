"""测试夹具：合成的行业/ETF 数据（内存对象 + 与真实数据同格式的文件）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rotation.data.etf import ETFData
from rotation.data.industry import IndustryData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDUSTRIES = ["农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "医药生物", "银行", "非银金融"]
CODES = {name: f"8010{i:02d}.SI" for i, name in enumerate(INDUSTRIES, start=10)}

# (代码, 名称, 跟踪指数代码, 跟踪指数名称, 层级, 跟踪的行业, 日成交额量级(元))
ETF_SPECS = [
    ("510001", "农业ETF测试", "930001", "中证农业主题指数", "主题", "农林牧渔", 5e7),
    ("510002", "化工ETF测试", "930002", "中证细分化工产业主题指数", "主题", "基础化工", 5e7),
    ("510003", "钢铁ETF测试", "930003", "中证钢铁指数", "行业", "钢铁", 5e7),
    ("510004", "有色ETF测试", "930004", "中证申万有色金属指数", "行业", "有色金属", 5e7),
    (
        "510005",
        "有色ETF测试二",
        "930004",
        "中证申万有色金属指数",
        "行业",
        "有色金属",
        1e7 * 0.2,
    ),  # 同指数、流动性差
    ("510006", "电子ETF测试", "930006", "中证电子指数", "主题", "电子", 5e7),
    ("510007", "医药ETF测试", "930007", "中证全指医药卫生指数", "行业", "医药生物", 5e7),
    ("510008", "银行ETF测试", "930008", "中证银行指数", "行业", "银行", 5e7),
    ("510009", "证券ETF测试", "930009", "中证全指证券公司指数", "行业", "非银金融", 5e7),
    ("510010", "原材料ETF测试", "930010", "中证全指原材料指数", "行业", "基础化工", 5e7),
]
BENCH = ("510300", "沪深300ETF测试")


def make_industry(dates: pd.DatetimeIndex, seed: int = 0) -> IndustryData:
    rng = np.random.default_rng(seed)
    n, k = len(dates), len(INDUSTRIES)
    market = rng.normal(0.0002, 0.012, n)
    ret = market[:, None] + rng.normal(0, 0.01, (n, k))
    close = 1000 * np.exp(np.cumsum(ret, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.003, (n, k)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, (n, k))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, (n, k))))
    volume = rng.lognormal(18, 0.3, (n, k))
    frame = lambda a: pd.DataFrame(a, index=dates, columns=INDUSTRIES)  # noqa: E731
    fields = {
        "open": frame(open_),
        "high": frame(high),
        "low": frame(low),
        "close": frame(close),
        "volume": frame(volume),
        "amount": frame(volume * close / 100),
        "turnover": frame(rng.uniform(0.5, 3, (n, k))),
        "pe": frame(rng.uniform(10, 40, (n, k))),
        "pb": frame(rng.uniform(1, 5, (n, k))),
        "ps": frame(rng.uniform(0.5, 3, (n, k))),
        "dy": frame(rng.uniform(0.5, 4, (n, k))),
    }
    for f in fields.values():
        f.index.name = "date"
    return IndustryData(
        fields=fields,
        industries=list(INDUSTRIES),
        codes=dict(CODES),
        fingerprint=f"synthetic{seed}",
    )


def make_etf(industry: IndustryData, seed: int = 1) -> ETFData:
    rng = np.random.default_rng(seed)
    dates = industry.dates
    opens, closes, amounts, rows = {}, {}, {}, []
    for code, name, idx_code, idx_name, tier, ind, amt in ETF_SPECS:
        base = industry["close"][ind].to_numpy()
        if ind == "基础化工" and code == "510010":  # 原材料：化工 + 钢铁 + 有色 的混合
            base = industry["close"][["基础化工", "钢铁", "有色金属"]].mean(axis=1).to_numpy()
        noise = np.exp(np.cumsum(rng.normal(0, 0.002, len(dates))))
        close = base / base[0] * noise
        closes[code] = close
        opens[code] = close * np.exp(rng.normal(0, 0.002, len(dates)))
        amounts[code] = rng.uniform(0.8, 1.2, len(dates)) * amt
        rows.append(
            {
                "code": code,
                "name": name,
                "exchange": "SH",
                "index_code": idx_code,
                "index_name": idx_name,
                "listing_date": dates[0],
                "tier": "industry" if tier == "行业" else "theme",
            }
        )
    bench = industry["close"].mean(axis=1).to_numpy()
    closes[BENCH[0]] = bench / bench[0]
    opens[BENCH[0]] = closes[BENCH[0]]
    amounts[BENCH[0]] = np.full(len(dates), 1e9)
    rows.append(
        {
            "code": BENCH[0],
            "name": BENCH[1],
            "exchange": "SH",
            "index_code": "",
            "index_name": "",
            "listing_date": dates[0],
            "tier": "benchmark",
        }
    )
    master = pd.DataFrame(rows).set_index("code")
    df = lambda d: pd.DataFrame(d, index=dates)  # noqa: E731
    return ETFData(master=master, open=df(opens), close=df(closes), amount=df(amounts))


@pytest.fixture(scope="session")
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2019-01-01", "2022-06-30").as_unit("ns")


@pytest.fixture(scope="session")
def industry(dates) -> IndustryData:
    return make_industry(dates)


@pytest.fixture(scope="session")
def etf(industry) -> ETFData:
    return make_etf(industry)


def write_raw_files(folder: Path, industry: IndustryData, etf: ETFData) -> Path:
    """按真实数据的文件格式写出：行业 Excel、两张 ETF 名单、后复权 CSV。"""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for ind in industry.industries + ["综合"]:
        src = ind if ind != "综合" else industry.industries[0]
        f = industry.fields
        rows.append(
            pd.DataFrame(
                {
                    "行业名称": ind,
                    "行业代码": CODES.get(ind, "801230.SI"),
                    "日期": industry.dates,
                    "开盘价": f["open"][src].to_numpy(),
                    "最高价": f["high"][src].to_numpy(),
                    "最低价": f["low"][src].to_numpy(),
                    "收盘价": f["close"][src].to_numpy(),
                    "成交量": f["volume"][src].to_numpy(),
                    "成交额": f["amount"][src].to_numpy(),
                    "换手率": f["turnover"][src].to_numpy(),
                    "市盈率PE": f["pe"][src].to_numpy(),
                    "市净率PB": f["pb"][src].to_numpy(),
                    "市销率PS": f["ps"][src].to_numpy(),
                    "股息率": f["dy"][src].to_numpy(),
                }
            )
        )
    pd.concat(rows).to_excel(folder / "申万一级指数_原始数据.xlsx", index=False)

    for tier, file in [("industry", "行业指数ETF.xlsx"), ("theme", "主题指数ETF.xlsx")]:
        sub = etf.master[etf.master["tier"] == tier]
        pd.DataFrame(
            {
                "证券代码": [f"{c}.SH" for c in sub.index],
                "证券名称": sub["name"].to_numpy(),
                "跟踪指数代码": sub["index_code"].to_numpy(),
                "跟踪指数名称": sub["index_name"].to_numpy(),
                "上市日期": [float(d.strftime("%Y%m%d")) for d in sub["listing_date"]],
            }
        ).to_excel(folder / file, index=False)

    daily = folder / "后复权"
    daily.mkdir(exist_ok=True)
    for code in etf.close.columns:
        pd.DataFrame(
            {
                "代码": f"sh.{code}",
                "名称": etf.master.at[code, "name"],
                "日期": etf.close.index.strftime("%Y-%m-%d"),
                "开盘价": etf.open[code].to_numpy(),
                "收盘价": etf.close[code].to_numpy(),
                "成交额": etf.amount[code].to_numpy() / 1000.0,  # 数据源单位：千元
            }
        ).to_csv(daily / f"sh.{code}_金玥数据.csv", index=False, encoding="utf-8-sig")
    return folder
