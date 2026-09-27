"""开发期因子筛选（只用 sample_start ~ select_end 且标签在 select_end 前可观测的样本）。

默认方法 `importance`：
1. 重要性评分：对全部候选因子分别训练随机森林（MDI 重要性）和 XGBoost（gain 重要性），
   再算逐日截面 RankIC 均值的绝对值；三者各自排名（百分位）后取平均，得到综合分；
2. 相关性聚类：在筛选期合并样本上计算因子间 Pearson 相关，按 1−|ρ| 做全链接层次聚类，
   在 |ρ| ≥ corr_threshold 处切断，使同一簇内任意两个因子都满足 |ρ| ≥ 阈值；
   - 簇内只有一个"重要性高"（综合分位于前 high_importance_frac）的因子：保留它，去掉其余；
   - 簇内有多个重要性高的因子：对整簇做 PCA，用第一主成分合成一个新因子替代整簇。
     PCA 只在筛选期拟合，载荷方向使合成因子与标签的 RankIC 为正；之后各日只用这组
     固定载荷做线性变换（因果，不重新拟合）；
   - 簇内没有重要性高的因子：保留综合分最高的一个作为代表；
3. 剔除综合分排名最后 drop_bottom_frac 的单个因子（合成因子由高重要性因子构成，不在此列）；
4. 记录方向：RankIC 为负的单因子在使用时乘以 −1，统一为"越大越好"。

旧方法 `ic_threshold`（|t| 门槛 + 方向稳定 + 贪心去相关 + 类别分散）保留，用于对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from rotation.config import SelectionConfig
from rotation.evaluation.ranking import factor_rank_ic, ic_summary
from rotation.features.builder import FeatureSet
from rotation.labels import Labels
from rotation.utils.hashing import stable_hash
from rotation.utils.logging import get_logger

log = get_logger("features.selection")


def hac_lags(n: int, horizon: int) -> int:
    """Newey-West 滞后阶数：取经验法则与标签重叠长度中的较大者。"""
    rule = int(4 * (max(n, 1) / 100.0) ** (2.0 / 9.0))
    return max(horizon, rule)


@dataclass
class Composite:
    """PCA 合成因子：value = (X[members] − center) · loadings / scale。"""

    name: str
    cluster: str
    members: list[str]
    loadings: list[float]
    center: list[float]
    scale: float
    explained_ratio: float
    category: str

    def transform(self, x: np.ndarray) -> np.ndarray:
        """x [..., len(members)] → [...]，只用固定参数，逐日独立。"""
        w = np.asarray(self.loadings, dtype=np.float64)
        c = np.asarray(self.center, dtype=np.float64)
        return ((x.astype(np.float64) - c) @ w / self.scale).astype(np.float32)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SelectionResult:
    selected: list[str]
    signs: dict[str, float]
    table: pd.DataFrame
    fallback: bool
    window: tuple[str, str]
    n_dates: int
    notes: list[str] = field(default_factory=list)
    composites: list[Composite] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return self.selected + [c.name for c in self.composites]

    @property
    def digest(self) -> str:
        """入选因子、方向与 PCA 参数的指纹（用于逐折缓存键）。"""
        return stable_hash(
            {
                "selected": self.selected,
                "signs": self.signs,
                "composites": [
                    {**c.to_dict(), "loadings": [round(v, 8) for v in c.loadings]}
                    for c in self.composites
                ],
            }
        )

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "signs": self.signs,
            "composites": [c.to_dict() for c in self.composites],
            "feature_names": self.feature_names,
            "fallback": self.fallback,
            "window": list(self.window),
            "n_dates": self.n_dates,
            "notes": self.notes,
        }


def apply_selection(fs: FeatureSet, sel: SelectionResult) -> FeatureSet:
    """按筛选结果生成最终特征：单因子（乘以方向）+ PCA 合成因子（固定载荷变换）。"""
    out = fs.subset(sel.selected, sel.signs)
    if not sel.composites:
        return out
    extra = []
    for comp in sel.composites:
        idx = [fs.names.index(n) for n in comp.members]
        extra.append(comp.transform(fs.X[:, :, idx]))
    x = np.concatenate([out.X, np.stack(extra, axis=2)], axis=2).astype(np.float32)
    categories = dict(out.categories)
    categories.update({c.name: c.category for c in sel.composites})
    return replace(out, names=sel.feature_names, X=x, categories=categories)


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


def _subsample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int):
    if len(y) > max_samples:
        idx = np.random.default_rng(seed).choice(len(y), size=max_samples, replace=False)
        return x[idx], y[idx]
    return x, y


def _tree_importance(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    x, y = _subsample(x, y, max_samples, seed)
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


def _xgb_importance(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    from rotation.features.xgb_worker import xgb_importance

    x, y = _subsample(x, y, max_samples, seed)
    return xgb_importance(x, y, seed)


def correlation_clusters(corr: np.ndarray, threshold: float) -> np.ndarray:
    """全链接聚类：返回每个因子的簇编号，同簇内任意两因子 |ρ| ≥ threshold。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = corr.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int)
    dist = 1.0 - np.nan_to_num(np.abs(corr), nan=0.0)
    dist = np.clip((dist + dist.T) / 2, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform(dist, checks=False), method="complete")
    return fcluster(tree, t=1.0 - threshold + 1e-9, criterion="distance") - 1


def composite_score(table: pd.DataFrame) -> pd.Series:
    """RF 重要性、XGBoost 重要性、|RankIC| 各自的百分位排名取平均。"""
    parts = [
        table[c].rank(pct=True).fillna(0.0) for c in ("rf_importance", "xgb_importance", "abs_ic")
    ]
    return sum(parts) / len(parts)


def plan_clusters(
    names: list[str], labels: np.ndarray, score: pd.Series, high: pd.Series
) -> list[dict]:
    """逐簇决定：keep（保留一个，其余剔除）或 pca（整簇合成）。按簇内最高综合分排序编号。"""
    groups: dict[int, list[str]] = {}
    for name, lab in zip(names, labels):
        groups.setdefault(int(lab), []).append(name)
    ordered = sorted(groups.values(), key=lambda g: -score[g].max())
    plans = []
    for k, members in enumerate(ordered, start=1):
        members = sorted(members, key=lambda n: -score[n])
        n_high = int(high[members].sum())
        action = "pca" if n_high >= 2 else "keep"
        plans.append({"cluster": f"C{k:02d}", "members": members, "n_high": n_high, "action": action})
    return plans


def fit_composite(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, members_idx: list[int], lags: int
) -> tuple[np.ndarray, np.ndarray, float, float, dict]:
    """在筛选期合并样本上拟合 PCA 第一主成分；方向使合成因子与标签的 RankIC 为正。"""
    flat = x[:, :, members_idx].reshape(-1, len(members_idx)).astype(np.float64)[mask.reshape(-1)]
    center = flat.mean(axis=0)
    cov = np.cov(flat - center, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    w = eigvec[:, 0]
    explained = float(eigval[0] / eigval.sum()) if eigval.sum() > 0 else np.nan
    scale = float(np.sqrt(max(eigval[0], 1e-12)))
    comp = (x[:, :, members_idx].astype(np.float64) - center) @ w / scale
    stats = ic_summary(factor_rank_ic(comp[:, :, None], y, mask)[:, 0], lags)
    if np.isfinite(stats["mean"]) and stats["mean"] < 0:
        w = -w
        stats = {**stats, "mean": -stats["mean"], "t": -stats["t"]}
        stats["icir"] = -stats["icir"] if np.isfinite(stats["icir"]) else stats["icir"]
    return w, center, scale, explained, stats


def _ic_table(fs: FeatureSet, ic: np.ndarray, lags: int) -> pd.DataFrame:
    half = ic.shape[0] // 2
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
    return table


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
    table = _ic_table(fs, ic, lags)
    window = (str(fs.dates[positions[0]].date()), str(fs.dates[positions[-1]].date()))
    flat = x.reshape(-1, x.shape[2])
    flat_mask = mask.reshape(-1)
    if cfg.method == "ic_threshold":
        return _select_ic_threshold(fs, table, flat, y, flat_mask, cfg, seed, window, len(positions))

    # 1. 重要性评分
    xs, ys = flat[flat_mask], y.reshape(-1)[flat_mask]
    table["rf_importance"] = _tree_importance(xs, ys, cfg.tree_max_samples, seed)
    table["xgb_importance"] = _xgb_importance(xs, ys, cfg.tree_max_samples, seed)
    table["abs_ic"] = table["ic_mean"].abs()
    for col, rank_col in (
        ("rf_importance", "rf_rank"),
        ("xgb_importance", "xgb_rank"),
        ("abs_ic", "ic_rank"),
    ):
        table[rank_col] = table[col].rank(ascending=False, method="min").astype(int)
    table["score"] = composite_score(table)
    table["score_rank"] = table["score"].rank(ascending=False, method="first").astype(int)
    n = len(table)
    n_high = max(1, int(np.ceil(cfg.high_importance_frac * n)))
    n_drop = int(np.floor(cfg.drop_bottom_frac * n))
    table["high"] = table["score_rank"] <= n_high
    bottom = table["score_rank"] > n - n_drop

    # 2. 相关性聚类
    corr = np.corrcoef(xs, rowvar=False)
    table["cluster"] = ""
    table["cluster_size"] = 1
    table["status"] = ""
    table["reason"] = ""
    table["composite"] = ""
    table["pca_loading"] = np.nan
    plans = plan_clusters(
        fs.names, correlation_clusters(corr, cfg.corr_threshold), table["score"], table["high"]
    )
    kept: list[str] = []
    composites: list[Composite] = []
    comp_rows = []
    for plan in plans:
        members, cid = plan["members"], plan["cluster"]
        table.loc[members, "cluster"] = cid
        table.loc[members, "cluster_size"] = len(members)
        if plan["action"] == "keep":
            head = members[0]
            kept.append(head)
            if len(members) > 1:
                why = "簇内唯一高重要性因子" if plan["n_high"] == 1 else "簇内无高重要性因子，取综合分最高者"
                table.at[head, "reason"] = why
                table.loc[members[1:], "status"] = "剔除"
                table.loc[members[1:], "reason"] = f"与 {head} 高度相关（{cid}），综合分较低"
            continue
        idx = [fs.names.index(m) for m in members]
        w, center, scale, explained, stats = fit_composite(x, y, mask, idx, lags)
        cats = pd.Series([fs.categories.get(m, "") for m in members]).value_counts()
        comp = Composite(
            name=f"pca_{members[0]}",
            cluster=cid,
            members=list(members),
            loadings=[float(v) for v in w],
            center=[float(v) for v in center],
            scale=scale,
            explained_ratio=explained,
            category=str(cats.index[0]),
        )
        composites.append(comp)
        table.loc[members, "status"] = "合成"
        table.loc[members, "composite"] = comp.name
        table.loc[members, "pca_loading"] = w
        table.loc[members, "reason"] = f"{cid} 内 {plan['n_high']} 个高重要性因子，PCA 第一主成分合成"
        comp_rows.append(
            {
                "factor": comp.name,
                "category": comp.category,
                "ic_mean": stats["mean"],
                "ic_std": stats["std"],
                "icir": stats["icir"],
                "t": stats["t"],
                "abs_ic": abs(stats["mean"]),
                "cluster": cid,
                "cluster_size": len(members),
                "status": "入选",
                "reason": f"PCA 合成（{len(members)} 个因子，解释方差 {explained:.1%}）",
                "composite": comp.name,
                "explained_ratio": explained,
                "members": ", ".join(members),
                "sign": 1.0,
            }
        )

    # 3. 剔除末位
    selected: list[str] = []
    for name in kept:
        if bool(bottom[name]):
            table.at[name, "status"] = "剔除"
            table.at[name, "reason"] = f"综合分排名末 {n_drop} 位"
        else:
            table.at[name, "status"] = "入选"
            table.at[name, "reason"] = table.at[name, "reason"] or "与其他因子相关性低"
            selected.append(name)
    selected.sort(key=lambda n: table.at[n, "score_rank"])
    signs = {n: float(table.at[n, "sign"]) for n in selected}
    table["selected"] = table["status"] == "入选"
    table["max_abs_corr"] = [
        float(np.nanmax(np.where(np.eye(n, dtype=bool)[j], np.nan, np.abs(corr[j]))))
        for j in range(n)
    ]
    if comp_rows:
        comp_table = pd.DataFrame(comp_rows).set_index("factor")
        comp_table["selected"] = True
        table = pd.concat([table, comp_table])
    table["explained_ratio"] = table.get("explained_ratio", np.nan)

    notes = []
    if not selected and not composites:
        raise RuntimeError("因子筛选后没有剩余因子，请检查 drop_bottom_frac 等设置")
    log.info(
        "因子筛选（%s ~ %s，%d 个日期）：候选 %d → %d 个相关簇（%d 个做 PCA 合成）→ 入选单因子 %d + 合成因子 %d",
        window[0],
        window[1],
        len(positions),
        n,
        len(plans),
        len(composites),
        len(selected),
        len(composites),
    )
    status_order = table["status"].map({"入选": 0, "合成": 1, "剔除": 2}).fillna(3)
    table = (
        table.assign(_o=status_order, _s=-table["score"].fillna(np.inf))
        .sort_values(["_o", "_s"])
        .drop(columns=["_o", "_s"])
    )
    return SelectionResult(
        selected=selected,
        signs=signs,
        table=table,
        fallback=False,
        window=window,
        n_dates=len(positions),
        notes=notes,
        composites=composites,
    )


def _select_ic_threshold(
    fs: FeatureSet,
    table: pd.DataFrame,
    flat: np.ndarray,
    y: np.ndarray,
    flat_mask: np.ndarray,
    cfg: SelectionConfig,
    seed: int,
    window: tuple[str, str],
    n_dates: int,
) -> SelectionResult:
    """旧方法：|t| ≥ min_abs_t 且方向稳定 → 按 |t| 贪心去相关 → RF 辅助打分 → 类别分散。"""
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
    log.info(
        "因子筛选（旧方法，%s ~ %s，%d 个日期）：候选 %d → 达标 %d → 去相关 %d → 入选 %d%s",
        window[0],
        window[1],
        n_dates,
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
        n_dates=n_dates,
        notes=notes,
    )
