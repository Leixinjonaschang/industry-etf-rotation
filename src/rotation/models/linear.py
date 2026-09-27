"""岭回归对照组：在验证集上按 RankIC 选择正则化强度 α。"""

from __future__ import annotations

from typing import Any

import numpy as np

from rotation.dataset.panel import Panel
from rotation.models.base import BaseModel


class RidgeModel(BaseModel):
    name = "ridge"

    def fit(self, panel: Panel, train_pos: np.ndarray, val_pos: np.ndarray) -> dict[str, Any]:
        from sklearn.linear_model import Ridge

        alphas = list(self.params.get("alphas", [0.1, 1.0, 10.0, 100.0, 1000.0]))
        x, y, mask = panel.tabular(train_pos)
        x, y = x[mask], y[mask]
        best = (-np.inf, alphas[0], None)
        scores = {}
        for alpha in alphas:
            model = Ridge(alpha=alpha).fit(x, y)
            if len(val_pos):
                pred = self._predict_with(model, panel, val_pos)
                score = self.validation_rank_ic(panel, val_pos, pred)
            else:
                score = 0.0
            scores[str(alpha)] = score
            if np.isfinite(score) and score > best[0]:
                best = (score, alpha, model)
        if best[2] is None:
            best = (float("nan"), alphas[0], Ridge(alpha=alphas[0]).fit(x, y))
        self.model_ = best[2]
        return {"alpha": best[1], "val_rank_ic": best[0], "val_scores": scores}

    @staticmethod
    def _predict_with(model, panel: Panel, pos: np.ndarray) -> np.ndarray:
        x, _, _ = panel.tabular(pos)
        return model.predict(x).reshape(len(pos), panel.n_industries).astype(np.float32)

    def predict(self, panel: Panel, pos: np.ndarray) -> np.ndarray:
        return self._predict_with(self.model_, panel, pos)
