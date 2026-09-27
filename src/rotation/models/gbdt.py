"""梯度提升树对照组。

默认用 scikit-learn 的 HistGradientBoostingRegressor（LightGBM 同类算法，无额外依赖）；
`params.backend: lightgbm` 时改用 LightGBM（需 `uv sync --extra lightgbm`，macOS 还需 `brew install libomp`）。
迭代轮数按验证集 RankIC 选择。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rotation.dataset.panel import Panel
from rotation.models.base import BaseModel

_DEFAULTS = {
    "backend": "sklearn",
    "max_iter": 400,
    "learning_rate": 0.03,
    "max_depth": 4,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    "eval_every": 20,
}


class GBDTModel(BaseModel):
    name = "gbdt"

    def __init__(self, params: dict[str, Any], train_cfg, seed: int = 42):
        super().__init__({**_DEFAULTS, **params}, train_cfg, seed)

    def fit(self, panel: Panel, train_pos: np.ndarray, val_pos: np.ndarray) -> dict[str, Any]:
        x, y, mask = panel.tabular(train_pos)
        x, y = x[mask], y[mask]
        if self.params["backend"] == "lightgbm":
            return self._fit_lightgbm(panel, x, y, val_pos)
        return self._fit_sklearn(panel, x, y, val_pos)

    # ---------------------------------------------------------------- sklearn
    def _fit_sklearn(
        self, panel: Panel, x: np.ndarray, y: np.ndarray, val_pos: np.ndarray
    ) -> dict[str, Any]:
        from sklearn.ensemble import HistGradientBoostingRegressor

        p = self.params
        model = HistGradientBoostingRegressor(
            max_iter=int(p["max_iter"]),
            learning_rate=float(p["learning_rate"]),
            max_depth=p["max_depth"],
            max_leaf_nodes=int(p["max_leaf_nodes"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            l2_regularization=float(p["l2_regularization"]),
            early_stopping=False,
            random_state=self.seed,
        )
        model.fit(x, y)
        self.model_ = model
        self.best_iter_ = int(p["max_iter"])
        curve: dict[int, float] = {}
        if len(val_pos):
            xv, _, _ = panel.tabular(val_pos)
            every = int(p["eval_every"])
            best = (-np.inf, self.best_iter_)
            for i, pred in enumerate(model.staged_predict(xv), start=1):
                if i % every and i != self.best_iter_:
                    continue
                score = self.validation_rank_ic(
                    panel, val_pos, pred.reshape(len(val_pos), panel.n_industries)
                )
                curve[i] = score
                if np.isfinite(score) and score > best[0]:
                    best = (score, i)
            self.best_iter_ = best[1]
        return {"best_iter": self.best_iter_, "val_curve": curve}

    # ---------------------------------------------------------------- lightgbm
    def _fit_lightgbm(
        self, panel: Panel, x: np.ndarray, y: np.ndarray, val_pos: np.ndarray
    ) -> dict[str, Any]:
        import lightgbm as lgb

        p = self.params
        model = lgb.LGBMRegressor(
            n_estimators=int(p["max_iter"]),
            learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]) if p["max_depth"] else -1,
            num_leaves=int(p["max_leaf_nodes"]),
            min_child_samples=int(p["min_samples_leaf"]),
            reg_lambda=float(p["l2_regularization"]),
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x, y)
        self.model_ = model
        self.best_iter_ = int(p["max_iter"])
        curve: dict[int, float] = {}
        if len(val_pos):
            xv, _, _ = panel.tabular(val_pos)
            best = (-np.inf, self.best_iter_)
            for i in range(int(p["eval_every"]), self.best_iter_ + 1, int(p["eval_every"])):
                pred = model.predict(xv, num_iteration=i).reshape(len(val_pos), panel.n_industries)
                score = self.validation_rank_ic(panel, val_pos, pred)
                curve[i] = score
                if np.isfinite(score) and score > best[0]:
                    best = (score, i)
            self.best_iter_ = best[1]
        return {"best_iter": self.best_iter_, "val_curve": curve}

    def predict(self, panel: Panel, pos: np.ndarray) -> np.ndarray:
        x, _, _ = panel.tabular(pos)
        if self.params["backend"] == "lightgbm":
            pred = self.model_.predict(x, num_iteration=self.best_iter_)
        else:
            pred = None
            for i, staged in enumerate(self.model_.staged_predict(x), start=1):
                pred = staged
                if i >= self.best_iter_:
                    break
        return np.asarray(pred).reshape(len(pos), panel.n_industries).astype(np.float32)
