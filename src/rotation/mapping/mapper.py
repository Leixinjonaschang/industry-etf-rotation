"""行业 → ETF 动态映射（每个信号日 t 只使用 t 及以前的数据）。

对每个被考虑的行业：
1. 语义候选池（白名单；行业 ETF 层优先，其次主题 ETF 层）；
2. 可交易性过滤：已有行情天数、近 liq_window 日有效天数与日均成交额；
3. 同一跟踪指数只保留成交额最大的一只作为代表（"先选指数、再选产品"）；
4. 近 corr_window 日日收益的 Pearson / Spearman / 超额收益相关，门槛 τ_p、τ_ex，打分；
5. 匈牙利算法唯一分配（一只 ETF 只对应一个行业），上期持有的 ETF 有保留奖励；
6. 兜底：next_rank 顺延到后续排名的行业 / cash 持现金 / broad 买宽基 ETF。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from rotation.config import MappingConfig
from rotation.data.etf import ETFData
from rotation.data.schema import TIER_CN, TIER_INDUSTRY, TIER_THEME
from rotation.mapping.scoring import tracking_metrics
from rotation.mapping.whitelist import Whitelist

_OPTION_COLUMNS = [
    "code",
    "name",
    "tier",
    "index_code",
    "index_name",
    "avg_amount",
    "pearson",
    "spearman",
    "excess_corr",
    "tracking_error",
    "beta",
    "n_obs",
    "score",
]


@dataclass
class MappingResult:
    slots: pd.DataFrame  # 每个信号日 × 仓位
    attempts: pd.DataFrame  # 每个信号日 × 被考虑的行业（含失败原因）


class ETFMapper:
    def __init__(
        self,
        cfg: MappingConfig,
        industry_close: pd.DataFrame,
        etf: ETFData,
        whitelist: Whitelist,
        top_n: int,
    ):
        if not industry_close.index.equals(etf.close.index):
            raise ValueError("行业与 ETF 数据的交易日历不一致")
        self.cfg = cfg
        self.top_n = top_n
        self.dates = industry_close.index
        self.industries = list(industry_close.columns)
        ind_close = industry_close.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.ind_ret = ind_close[1:] / ind_close[:-1] - 1.0
        self.ind_ret = np.vstack([np.full((1, ind_close.shape[1]), np.nan), self.ind_ret])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 首行全为 NaN
            self.mkt_ret = np.nanmean(self.ind_ret, axis=1)
        close = etf.close
        self.codes = list(close.columns)
        self.code_pos = {c: j for j, c in enumerate(self.codes)}
        self.etf_ret = (close / close.shift(1) - 1.0).to_numpy(dtype=float)
        self.close_ok = close.notna().to_numpy()
        self.listing_days = close.notna().cumsum().to_numpy()
        self.valid_days = (
            close.notna().astype(float).rolling(cfg.liq_window, min_periods=1).sum().to_numpy()
        )
        self.avg_amount = (
            etf.amount.rolling(cfg.liq_window, min_periods=1).mean().to_numpy(dtype=float)
        )
        self.master = etf.master
        table = whitelist.candidate_table(self.industries, etf.master)
        table = table[table["code"].isin(self.code_pos)]
        self.candidates: dict[str, dict[str, list[str]]] = {
            ind: {
                tier: table[(table["industry"] == ind) & (table["tier"] == tier)]["code"].tolist()
                for tier in (TIER_INDUSTRY, TIER_THEME)
            }
            for ind in self.industries
        }
        self._cache: dict[tuple[int, str], tuple[pd.DataFrame, str]] = {}

    # ------------------------------------------------------------------ 单行业
    def _eligible(self, t: int, codes: list[str]) -> list[str]:
        cfg = self.cfg
        out = []
        for code in codes:
            j = self.code_pos[code]
            if (
                self.close_ok[t, j]
                and self.listing_days[t, j] >= cfg.min_listing_days
                and self.valid_days[t, j] >= cfg.min_valid_days
                and np.nan_to_num(self.avg_amount[t, j]) >= cfg.min_amount
            ):
                out.append(code)
        return out

    def _score(self, t: int, industry: str, codes: list[str]) -> pd.DataFrame:
        cfg = self.cfg
        start = max(0, t - cfg.corr_window + 1)
        i = self.industries.index(industry)
        cols = [self.code_pos[c] for c in codes]
        metrics = tracking_metrics(
            self.ind_ret[start : t + 1, i],
            self.mkt_ret[start : t + 1],
            self.etf_ret[start : t + 1][:, cols],
        )
        meta = self.master.loc[codes]
        frame = pd.DataFrame(
            {
                "code": codes,
                "name": meta["name"].to_numpy(),
                "tier": meta["tier"].to_numpy(),
                "index_code": meta["index_code"].to_numpy(),
                "index_name": meta["index_name"].to_numpy(),
                "avg_amount": self.avg_amount[t, cols],
                **metrics,
            }
        )
        w_p, w_s, w_e = cfg.weights
        frame["score"] = (
            w_p * frame["pearson"] + w_s * frame["spearman"] + w_e * frame["excess_corr"]
        )
        return frame

    def options(self, t: int, industry: str) -> tuple[pd.DataFrame, str]:
        """返回 (合格候选按分数降序, 失败原因)。"""
        key = (t, industry)
        if key in self._cache:
            return self._cache[key]
        cfg = self.cfg
        pools = self.candidates.get(industry, {})
        groups = (
            [[TIER_INDUSTRY], [TIER_THEME]] if cfg.tier_priority else [[TIER_INDUSTRY, TIER_THEME]]
        )
        reasons = []
        result = (pd.DataFrame(columns=_OPTION_COLUMNS), "")
        for tiers in groups:
            label = "+".join(TIER_CN[t_] for t_ in tiers)
            codes = [c for t_ in tiers for c in pools.get(t_, [])]
            if not codes:
                reasons.append(f"{label}层无语义候选")
                continue
            eligible = self._eligible(t, codes)
            if not eligible:
                reasons.append(f"{label}层无满足流动性/上市天数的候选")
                continue
            # 同一跟踪指数只保留成交额最大的一只
            liquid = sorted(
                eligible, key=lambda c: -np.nan_to_num(self.avg_amount[t, self.code_pos[c]])
            )
            reps, seen = [], set()
            for code in liquid:
                idx_code = str(self.master.at[code, "index_code"]).strip()
                if idx_code.lower() in ("", "nan", "none"):
                    idx_code = code  # 跟踪指数代码缺失时按产品本身处理
                if idx_code in seen:
                    continue
                seen.add(idx_code)
                reps.append(code)
            scored = self._score(t, industry, reps)
            passed = scored[
                (scored["n_obs"] >= cfg.min_corr_obs)
                & (scored["pearson"] >= cfg.tau_p)
                & (scored["excess_corr"] >= cfg.tau_ex)
            ]
            if len(passed):
                result = (passed.sort_values("score", ascending=False).reset_index(drop=True), "")
                break
            best_p = scored["pearson"].max()
            best_e = scored["excess_corr"].max()
            reasons.append(f"{label}层相关性不足（最高 ρ={best_p:.2f}，ρ_ex={best_e:.2f}）")
        if result[0].empty:
            result = (result[0], "；".join(reasons))
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------ 分配
    def _assign(
        self,
        industries: list[str],
        opts: dict[str, pd.DataFrame],
        prev: dict[str, str],
        require_all: bool,
    ) -> dict[str, str] | None:
        if not industries:
            return {}
        codes = sorted({c for ind in industries for c in opts[ind]["code"]})
        col = {c: j for j, c in enumerate(codes)}
        n = len(industries)
        score = np.full((n, len(codes) + n), -1e9)
        for r, ind in enumerate(industries):
            score[r, len(codes) + r] = 0.0  # "不分配"选项
            for row in opts[ind].itertuples(index=False):
                bonus = self.cfg.retention_bonus if prev.get(ind) == row.code else 0.0
                score[r, col[row.code]] = 1000.0 + float(row.score) + bonus
        rows, cols = linear_sum_assignment(score, maximize=True)
        result = {
            industries[r]: codes[c]
            for r, c in zip(rows, cols)
            if c < len(codes) and score[r, c] > 0
        }
        if require_all and len(result) < n:
            return None
        return result

    def map_date(
        self, t: int, priority: list[str], ranks: dict[str, int], prev: dict[str, str]
    ) -> tuple[list[dict], list[dict]]:
        cfg = self.cfg
        next_rank = cfg.fallback == "next_rank"
        pool = priority[: cfg.max_fallback_rank] if next_rank else priority[: self.top_n]
        chosen: list[str] = []
        opts: dict[str, pd.DataFrame] = {}
        attempts: list[dict] = []
        for ind in pool:
            frame, reason = self.options(t, ind)
            record = {
                "industry": ind,
                "model_rank": ranks.get(ind),
                "in_top_n": ind in priority[: self.top_n],
            }
            if frame.empty:
                attempts.append({**record, "status": "unmappable", "reason": reason})
                continue
            opts[ind] = frame
            if next_rank:
                if self._assign(chosen + [ind], opts, prev, require_all=True) is not None:
                    chosen.append(ind)
                    attempts.append({**record, "status": "chosen", "reason": ""})
                else:
                    attempts.append(
                        {
                            **record,
                            "status": "conflict",
                            "reason": "候选 ETF 已被排名更高的行业占用",
                        }
                    )
                if len(chosen) >= self.top_n:
                    break
            else:
                chosen.append(ind)
                attempts.append({**record, "status": "chosen", "reason": ""})

        assignment = self._assign(chosen, opts, prev, require_all=next_rank) or {}
        slots: list[dict] = []
        if next_rank:
            ordered = chosen
        else:
            ordered = list(priority[: self.top_n])
        for ind in ordered:
            code = assignment.get(ind)
            base = {
                "industry": ind,
                "model_rank": ranks.get(ind),
                "in_top_n": ind in priority[: self.top_n],
            }
            if code is None:
                status = "broad" if cfg.fallback == "broad" else "cash"
                slots.append(
                    {
                        **base,
                        "status": status,
                        "etf_code": cfg.broad_code if status == "broad" else None,
                    }
                )
                continue
            row = opts[ind][opts[ind]["code"] == code].iloc[0].to_dict()
            status = "mapped" if base["in_top_n"] else "fallback_next"
            slots.append(
                {
                    **base,
                    "status": status,
                    "etf_code": code,
                    "retained": prev.get(ind) == code,
                    **{
                        ("etf_name" if k == "name" else k): row[k]
                        for k in _OPTION_COLUMNS
                        if k != "code"
                    },
                }
            )
        while len(slots) < self.top_n:  # 顺延仍不足 N 个：剩余仓位持有现金
            slots.append(
                {
                    "industry": None,
                    "model_rank": None,
                    "in_top_n": False,
                    "status": "cash",
                    "etf_code": None,
                }
            )
        return slots, attempts

    # ------------------------------------------------------------------ 全样本
    def run(self, signals: pd.DataFrame) -> MappingResult:
        slot_rows, attempt_rows = [], []
        prev: dict[str, str] = {}
        for date, group in signals.sort_values(["signal_date", "priority"]).groupby(
            "signal_date", sort=True
        ):
            t = int(self.dates.get_loc(date))
            if t + 1 >= len(self.dates):
                continue  # 最后一天发出的信号没有下一个交易日可以成交
            trade_date = self.dates[t + 1]
            priority = group["industry"].tolist()
            ranks = dict(zip(group["industry"], group["rank"]))
            slots, attempts = self.map_date(t, priority, ranks, prev)
            prev = {
                s["industry"]: s["etf_code"]
                for s in slots
                if s.get("etf_code") and s["status"] in ("mapped", "fallback_next")
            }
            for k, s in enumerate(slots, start=1):
                slot_rows.append({"signal_date": date, "trade_date": trade_date, "slot": k, **s})
            for a in attempts:
                attempt_rows.append({"signal_date": date, **a})
        return MappingResult(slots=pd.DataFrame(slot_rows), attempts=pd.DataFrame(attempt_rows))

    def panorama(self, date: pd.Timestamp) -> pd.DataFrame:
        """某一日所有行业的最优映射（论文"映射全景表"）。"""
        t = int(self.dates.get_indexer([pd.Timestamp(date)], method="pad")[0])
        rows = []
        for ind in self.industries:
            frame, reason = self.options(t, ind)
            if frame.empty:
                rows.append({"industry": ind, "etf_code": None, "reason": reason})
            else:
                best = frame.iloc[0]
                rows.append(
                    {
                        "industry": ind,
                        "etf_code": best["code"],
                        "etf_name": best["name"],
                        "tier": TIER_CN.get(best["tier"], best["tier"]),
                        "index_name": best["index_name"],
                        "pearson": best["pearson"],
                        "excess_corr": best["excess_corr"],
                        "tracking_error": best["tracking_error"],
                        "avg_amount_mn": best["avg_amount"] / 1e6,
                        "n_options": len(frame),
                        "reason": "",
                    }
                )
        out = pd.DataFrame(rows).set_index("industry")
        out.attrs["date"] = str(self.dates[t].date())
        return out
