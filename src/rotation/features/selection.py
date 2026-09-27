"""开发期因子筛选（只用 sample_start ~ select_end 且标签在 select_end 前可观测的样本）。

步骤：
1. 每个候选因子的逐日截面 RankIC → 均值、ICIR、胜率、Newey-West t；
2. 稳定性：前后两半段 IC 与整体同号；
3. 门槛：|t| ≥ min_abs_t 且稳定（不足 min_factors 时按 |t| 递补并标记 fallback）；
4. 按 |t| 优先级去相关（|ρ| ≥ corr_threshold 的后者剔除）；
5. 随机森林重要性作辅助打分（权重 tree_weight）；
6. 每个类别至少保留 min_per_category 个，再按总分补足到 max_factors；
7. 记录方向：IC 为负的因子在使用时乘以 −1，统一为"越大越好"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rotation.config import SelectionConfig
from rotation.evaluation.ranking import factor_rank_ic, ic_summary
from rotation.features.builder import FeatureSet
from rotation.labels import Labels
from rotation.utils.logging import get_logger

log = get_logger("features.selection")


def hac_lags(n: int, horizon: int) -> int:
    """Newey-West 滞后阶数：取经验法则与标签重叠长度中的较大者。"""
    rule = int(4 * (max(n, 1) / 100.0) ** (2.0 / 9.0))
    return max(horizon, rule)


@dataclass
class SelectionResult:
    selected: list[str]
    signs: dict[str, float]
    table: pd.DataFrame
    fallback: bool
    window: tuple[str, str]
    n_dates: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "signs": self.signs,
            "fallback": self.fallback,
            "window": list(self.window),
            "n_dates": self.n_dates,
            "notes": self.notes,
        }


def selection_positions(
    labels: Labels, dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp
) -> np.ndarray:
    """筛选期样本：start ≤ t，且标签在 end 当日或之前可观测。"""
    end_pos = int(dates.searchsorted(end, side="right")) - 1
    start_pos = int(dates.searchsorted(start, side="left"))
    positions = np.arange(start_pos, len(dates))
    positions = positions[labels.known_pos[positions] <= end_pos]
    finite = np.isfinite(labels.y[positions]).sum(axis=1) >= labels.y.shape[1] // 2
    return positions[finite]


def _tree_importance(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed)
    if len(y) > max_samples:
        idx = rng.choice(len(y), size=max_samples, replace=False)
        x, y = x[idx], y[idx]
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=100,
        max_features=0.5,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model.feature_importances_


def select_factors(
    fs: FeatureSet,
    labels: Labels,
    cfg: SelectionConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int = 42,
) -> SelectionResult:
    positions = selection_positions(labels, fs.dates, start, end)
    if len(positions) < 100:
        raise ValueError(
            f"因子筛选样本过少（{len(positions)} 个日期），请检查 sample_start/select_end"
        )
    x = fs.X[positions]
    y = labels.y[positions]
    mask = fs.valid[positions] & np.isfinite(y)
    ic = factor_rank_ic(x, y, mask)  # [P, F]
    lags = hac_lags(len(positions), labels.horizon)
    half = len(positions) // 2

    rows = []
    for j, name in enumerate(fs.names):
        overall = ic_summary(ic[:, j], lags)
        first = np.nanmean(ic[:half, j])
        second = np.nanmean(ic[half:, j])
        sign = np.sign(overall["mean"]) if np.isfinite(overall["mean"]) else 0.0
        rows.append(
            {
                "factor": name,
                "category": fs.categories.get(name, ""),
                "ic_mean": overall["mean"],
                "ic_std": overall["std"],
                "icir": overall["icir"],
                "win_rate": overall["win_rate"] if sign >= 0 else 1 - overall["win_rate"],
                "t": overall["t"],
                "ic_first_half": first,
                "ic_second_half": second,
                "sign_stable": bool(
                    sign != 0 and np.sign(first) == sign and np.sign(second) == sign
                ),
                "sign": float(sign if sign != 0 else 1.0),
            }
        )
    table = pd.DataFrame(rows).set_index("factor")
    table["abs_t"] = table["t"].abs().fillna(0.0)

    stable_ok = (
        table["sign_stable"] if cfg.require_sign_stability else pd.Series(True, index=table.index)
    )
    table["eligible"] = stable_ok & (table["abs_t"] >= cfg.min_abs_t)

    # 候选顺序：达标因子（按 |t|）→ 方向稳定但未达标 → 其余；后两者仅在去相关后不足 min_factors 时递补
    order = (
        table.assign(_stable=stable_ok)
        .sort_values(["eligible", "_stable", "abs_t"], ascending=False)
        .index.tolist()
    )

    # 去相关：在合并样本上计算因子间 Pearson 相关，按上述顺序贪心保留
    flat = x.reshape(-1, x.shape[2])
    flat_mask = mask.reshape(-1)
    idx = [fs.names.index(n) for n in order]
    corr = np.corrcoef(flat[flat_mask][:, idx], rowvar=False)
    corr = np.nan_to_num(np.abs(np.atleast_2d(corr)), nan=0.0)
    kept: list[int] = []
    table["max_corr_to_kept"] = np.nan
    table["passed_corr"] = False
    fallback = False
    for k, name in enumerate(order):
        if not table.at[name, "eligible"] and len(kept) >= cfg.min_factors:
            break
        max_corr = float(corr[k, kept].max()) if kept else 0.0
        table.at[name, "max_corr_to_kept"] = max_corr
        if max_corr < cfg.corr_threshold:
            kept.append(k)
            table.at[name, "passed_corr"] = True
            fallback |= not bool(table.at[name, "eligible"])
    pruned = [order[k] for k in kept]
    notes: list[str] = []
    if fallback:
        notes.append(f"达标且去相关后的因子不足 {cfg.min_factors} 个，已按方向稳定性与 |t| 递补")

    # 树模型重要性（辅助）
    table["tree_importance"] = np.nan
    if cfg.tree_weight > 0 and len(pruned) > 1:
        cols = [fs.names.index(n) for n in pruned]
        imp = _tree_importance(
            flat[flat_mask][:, cols], y.reshape(-1)[flat_mask], cfg.tree_max_samples, seed
        )
        table.loc[pruned, "tree_importance"] = imp
    sub = table.loc[pruned]
    score = (1 - cfg.tree_weight) * sub["abs_t"].rank(pct=True)
    if cfg.tree_weight > 0:
        score = score + cfg.tree_weight * sub["tree_importance"].rank(pct=True).fillna(0.0)
    table["score"] = np.nan
    table.loc[pruned, "score"] = score

    # 类别分散 + 按分数补足
    ranked = sub.assign(score=score).sort_values("score", ascending=False)
    chosen: list[str] = []
    for _, group in ranked.groupby("category", sort=False):
        chosen.extend(group.index[: cfg.min_per_category].tolist())
    chosen = sorted(set(chosen), key=lambda n: -ranked.at[n, "score"])[: cfg.max_factors]
    for name in ranked.index:
        if len(chosen) >= cfg.max_factors:
            break
        if name not in chosen:
            chosen.append(name)
    chosen = sorted(chosen, key=lambda n: -ranked.at[n, "score"])

    table["selected"] = table.index.isin(chosen)
    signs = {n: float(table.at[n, "sign"]) for n in chosen}
    window = (str(fs.dates[positions[0]].date()), str(fs.dates[positions[-1]].date()))
    log.info(
        "因子筛选（%s ~ %s，%d 个日期）：候选 %d → 达标 %d → 去相关 %d → 入选 %d%s",
        window[0],
        window[1],
        len(positions),
        len(fs.names),
        int(table["eligible"].sum()),
        len(pruned),
        len(chosen),
        "（fallback）" if fallback else "",
    )
    table = table.sort_values(["selected", "score", "abs_t"], ascending=False)
    return SelectionResult(
        selected=chosen,
        signs=signs,
        table=table,
        fallback=fallback,
        window=window,
        n_dates=len(positions),
        notes=notes,
    )
