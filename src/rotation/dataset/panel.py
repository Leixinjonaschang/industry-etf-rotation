"""模型输入面板：特征立方体 + 标签 + 原始价格，全部按 [日期位置, 行业] 对齐。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from rotation.data.industry import IndustryData
from rotation.features.builder import FeatureSet
from rotation.labels import Labels


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    industries: list[str]
    feature_names: list[str]
    market_names: list[str]
    X: np.ndarray  # [T, N, F] float32
    M: np.ndarray  # [T, M] float32（市场状态，所有行业共享）
    valid: np.ndarray  # [T, N] bool
    y: np.ndarray  # [T, N] 训练目标
    y_raw: np.ndarray  # [T, N] 原始收益（同价格口径）
    known_pos: np.ndarray  # [T] 标签可观测位置
    close: np.ndarray  # [T, N] 原始收盘价（动量基线用）
    horizon: int
    label_kind: str

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_industries(self) -> int:
        return len(self.industries)

    def positions(self, dates: pd.Index) -> np.ndarray:
        locs = self.dates.get_indexer(pd.DatetimeIndex(dates))
        if (locs < 0).any():
            raise KeyError("日期不在面板中")
        return locs

    def target_mask(self, pos: np.ndarray) -> np.ndarray:
        return self.valid[pos] & np.isfinite(self.y[pos])

    def tabular(
        self, pos: np.ndarray, include_market: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """展开为表格：行 = (日期, 行业)。返回 (X[P·N, F(+M)], y[P·N], mask[P·N])。"""
        x = self.X[pos]  # [P, N, F]
        if include_market and self.M.shape[1] > 0:
            m = np.repeat(self.M[pos][:, None, :], self.n_industries, axis=1)
            x = np.concatenate([x, m], axis=2)
        mask = self.target_mask(pos)
        return x.reshape(-1, x.shape[2]), self.y[pos].reshape(-1), mask.reshape(-1)

    def with_features(self, X: np.ndarray, names: list[str]) -> Panel:
        return replace(self, X=X.astype(np.float32), feature_names=list(names))


def build_panel(features: FeatureSet, labels: Labels, data: IndustryData) -> Panel:
    """features 必须已截断到与 data 相同的日期范围。"""
    if not features.dates.equals(data.dates):
        raise ValueError("特征与行情日期不一致")
    if labels.y.shape != (len(data.dates), len(data.industries)):
        raise ValueError("标签形状与行情不一致")
    return Panel(
        dates=features.dates,
        industries=list(features.industries),
        feature_names=list(features.names),
        market_names=list(features.market_names),
        X=features.X,
        M=features.market,
        valid=features.valid,
        y=labels.y,
        y_raw=labels.y_raw,
        known_pos=labels.known_pos,
        close=data["close"].to_numpy(dtype=np.float64, copy=True),
        horizon=labels.horizon,
        label_kind=labels.kind,
    )


def fit_fold_pca(panel: Panel, train_pos: np.ndarray, n_components: int) -> tuple[Panel, dict]:
    """只在当前折训练样本上拟合 PCA，再变换全部日期（无前视）。"""
    from sklearn.decomposition import PCA

    x_train, _, mask = panel.tabular(train_pos, include_market=False)
    n = min(n_components, panel.X.shape[2])
    pca = PCA(n_components=n, svd_solver="full", random_state=0)
    train_scores = pca.fit_transform(x_train[mask])
    std = train_scores.std(axis=0) + 1e-8  # 用训练集主成分的标准差归一化
    t, k, f = panel.X.shape
    transformed = pca.transform(panel.X.reshape(-1, f)).reshape(t, k, n)
    transformed = np.clip(transformed / std, -5, 5)
    names = [f"pc{i + 1}" for i in range(n)]
    info = {"explained_variance": float(pca.explained_variance_ratio_.sum()), "n_components": n}
    return panel.with_features(transformed, names), info
