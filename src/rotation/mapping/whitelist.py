"""语义候选池：按 configs/etf_whitelist.yaml 的关键词规则，为每个行业圈定候选 ETF。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rotation.data.schema import TIER_BENCHMARK
from rotation.utils.io import read_yaml


@dataclass
class IndustryRule:
    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    include_index_names: list[str] = field(default_factory=list)
    include_codes: list[str] = field(default_factory=list)
    exclude_codes: list[str] = field(default_factory=list)


@dataclass
class Whitelist:
    version: str
    frozen_at: str
    global_exclude: list[str]
    rules: dict[str, IndustryRule]

    @classmethod
    def load(cls, path: Path) -> Whitelist:
        raw = read_yaml(path) or {}
        rules = {
            name: IndustryRule(**{k: [str(x) for x in v] for k, v in (spec or {}).items()})
            for name, spec in (raw.get("industries") or {}).items()
        }
        return cls(
            version=str(raw.get("version", "")),
            frozen_at=str(raw.get("frozen_at", "")),
            global_exclude=[str(x) for x in raw.get("global_exclude", [])],
            rules=rules,
        )

    def matches(self, industry: str, code: str, name: str, index_name: str) -> bool:
        rule = self.rules.get(industry)
        if rule is None:
            return False
        text = f"{name} {index_name}"
        if code in rule.exclude_codes:
            return False
        if any(k in text for k in self.global_exclude) or any(
            k in text for k in rule.exclude_keywords
        ):
            return code in rule.include_codes
        return (
            code in rule.include_codes
            or index_name in rule.include_index_names
            or any(k in text for k in rule.keywords)
        )

    def candidate_table(self, industries: list[str], master: pd.DataFrame) -> pd.DataFrame:
        """长表：industry, code, tier（不含基准类 ETF）。"""
        rows = []
        pool = master[master["tier"] != TIER_BENCHMARK]
        for industry in industries:
            for code, row in pool.iterrows():
                if self.matches(industry, str(code), str(row["name"]), str(row["index_name"])):
                    rows.append({"industry": industry, "code": str(code), "tier": row["tier"]})
        return pd.DataFrame(rows, columns=["industry", "code", "tier"])
