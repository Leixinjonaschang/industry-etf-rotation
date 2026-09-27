"""回归口径指标（只对输出收益率量纲的模型有意义）。"""

from __future__ import annotations

import numpy as np


def regression_metrics(
    pred: np.ndarray, target: np.ndarray, bench: np.ndarray | None = None
) -> dict[str, float]:
    """MAE、RMSE、R²_OOS（相对 bench，默认 0）、方向准确率、平均偏差。"""
    p = np.asarray(pred, dtype=float).ravel()
    y = np.asarray(target, dtype=float).ravel()
    b = np.zeros_like(y) if bench is None else np.asarray(bench, dtype=float).ravel()
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(b)
    p, y, b = p[ok], y[ok], b[ok]
    if len(y) == 0:
        return {k: float("nan") for k in ("mae", "rmse", "r2_oos", "direction_acc", "bias", "n")}
    err = p - y
    sse_model = float(err @ err)
    sse_bench = float((b - y) @ (b - y))
    nonzero = (y != 0) & (p != 0)
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2_oos": 1.0 - sse_model / sse_bench if sse_bench > 0 else float("nan"),
        "direction_acc": float((np.sign(p[nonzero]) == np.sign(y[nonzero])).mean())
        if nonzero.any()
        else float("nan"),
        "bias": float(err.mean()),
        "n": int(len(y)),
    }
