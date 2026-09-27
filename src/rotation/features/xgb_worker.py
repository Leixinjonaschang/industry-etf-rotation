"""在独立子进程里计算 XGBoost 特征重要性。

macOS 上 XGBoost（libomp）与 torch 自带的 OpenMP 同进程加载可能崩溃，所以主进程只负责
把样本写到临时 npz，再用 `python -m rotation.features.xgb_worker in.npz out.npy seed` 调用本模块。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    reg_lambda=1.0,
    importance_type="gain",
    tree_method="hist",
)


def fit_importance(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGB_PARAMS, random_state=seed, n_jobs=os.cpu_count() or 1)
    model.fit(x, y)
    imp = np.asarray(model.feature_importances_, dtype=float)
    total = imp.sum()
    return imp / total if total > 0 else imp


def xgb_importance(x: np.ndarray, y: np.ndarray, seed: int, timeout: float = 1800) -> np.ndarray:
    """在子进程中训练 XGBoost，返回归一化（和为 1）的 gain 重要性。"""
    src = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.npz", Path(tmp) / "out.npy"
        np.savez(inp, x=x.astype(np.float32), y=y.astype(np.float32))
        result = subprocess.run(
            [sys.executable, "-m", "rotation.features.xgb_worker", str(inp), str(out), str(seed)],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        if result.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"XGBoost 子进程失败（返回码 {result.returncode}）：\n{result.stderr[-2000:]}\n"
                "macOS 上请先 `brew install libomp`"
            )
        return np.load(out)


def main(argv: list[str]) -> None:
    inp, out, seed = argv
    data = np.load(inp)
    np.save(out, fit_importance(data["x"], data["y"], int(seed)))


if __name__ == "__main__":
    main(sys.argv[1:])
