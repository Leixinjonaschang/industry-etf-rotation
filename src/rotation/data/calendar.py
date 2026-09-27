"""交易日历（以申万行业指数的交易日为准）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rotation.data.schema import ns_index


class TradingCalendar:
    def __init__(self, dates: pd.Index):
        self.dates = ns_index(dates).unique().sort_values()

    def __len__(self) -> int:
        return len(self.dates)

    def pos(self, date: pd.Timestamp | str) -> int:
        """精确位置；非交易日会报错。"""
        ts = pd.Timestamp(date)
        loc = self.dates.searchsorted(ts, side="left")
        if loc >= len(self.dates) or self.dates[loc] != ts:
            raise KeyError(f"{ts.date()} 不是交易日")
        return int(loc)

    def pos_on_or_after(self, date: pd.Timestamp | str) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(date), side="left"))

    def pos_on_or_before(self, date: pd.Timestamp | str) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(date), side="right")) - 1

    def positions(self, dates: pd.Index) -> np.ndarray:
        idx = ns_index(dates)
        locs = self.dates.get_indexer(idx)
        if (locs < 0).any():
            missing = idx[locs < 0][:3]
            raise KeyError(f"存在非交易日：{list(missing)}")
        return locs

    def between(self, start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DatetimeIndex:
        mask = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        return self.dates[mask]

    def shift(self, date: pd.Timestamp | str, k: int) -> pd.Timestamp | None:
        target = self.pos(date) + k
        if 0 <= target < len(self.dates):
            return self.dates[target]
        return None
