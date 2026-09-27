"""映射诊断：覆盖率、状态分布。损耗分解见 rotation.backtest.runner。"""

from __future__ import annotations

import pandas as pd

from rotation.mapping.mapper import MappingResult


def coverage_by_industry(result: MappingResult) -> pd.DataFrame:
    """模型原始前 N 名中，每个行业被选中的次数与映射成功率。"""
    attempts = result.attempts
    if attempts.empty:
        return pd.DataFrame()
    top = attempts[attempts["in_top_n"]]
    slots = result.slots[result.slots["status"] == "mapped"]
    mapped_counts = slots.groupby("industry").size()
    table = top.groupby("industry").agg(times_in_top_n=("signal_date", "size"))
    table["times_mapped"] = mapped_counts.reindex(table.index).fillna(0).astype(int)
    table["coverage"] = table["times_mapped"] / table["times_in_top_n"]
    stats = slots.groupby("industry").agg(
        avg_pearson=("pearson", "mean"),
        avg_excess_corr=("excess_corr", "mean"),
        avg_tracking_error=("tracking_error", "mean"),
        main_etf=("etf_name", lambda s: s.value_counts().index[0] if len(s) else ""),
    )
    table = table.join(stats)
    reasons = (
        top[top["status"] != "chosen"]
        .groupby("industry")["reason"]
        .agg(lambda s: s.value_counts().index[0])
    )
    table["main_failure_reason"] = reasons.reindex(table.index).fillna("")
    return table.sort_values("times_in_top_n", ascending=False)


def status_summary(result: MappingResult) -> pd.DataFrame:
    slots = result.slots
    if slots.empty:
        return pd.DataFrame()
    counts = slots["status"].value_counts().rename("count").to_frame()
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts
