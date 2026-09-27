"""逐折训练 → 预测。每折结果单独落盘，中断后可断点续跑；配置变化会换到新的缓存目录。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from rotation.config import Config
from rotation.dataset.panel import Panel, fit_fold_pca
from rotation.dataset.splits import Fold
from rotation.models.registry import build_model
from rotation.utils.hashing import stable_hash
from rotation.utils.io import write_json
from rotation.utils.logging import get_logger

log = get_logger("pipeline.walk_forward")


def fold_cache_key(cfg: Config, feature_names: list[str], data_fingerprint: str) -> str:
    train = cfg.train.model_dump(mode="json")
    for key in ("device", "num_threads", "save_checkpoints"):
        train.pop(key, None)
    return stable_hash(
        {
            "model": cfg.model.model_dump(mode="json"),
            "train": train if cfg.model.name in ("lstm", "gru", "transformer") else None,
            "label": cfg.label.model_dump(mode="json"),
            "features": cfg.features.model_dump(mode="json"),
            "split": cfg.split.model_dump(mode="json"),
            "data": cfg.data.model_dump(mode="json"),
            "phase": cfg.experiment.phase,
            "seed": cfg.experiment.seed,
            "feature_names": feature_names,
            "fingerprint": data_fingerprint,
        }
    )


def _fold_frame(panel: Panel, fold: Fold, pred: np.ndarray, bench_mean: float) -> pd.DataFrame:
    pos = fold.test_pos
    n = panel.n_industries
    return pd.DataFrame(
        {
            "date": np.repeat(panel.dates[pos].to_numpy(), n),
            "industry": np.tile(np.asarray(panel.industries, dtype=object), len(pos)),
            "pred": pred.reshape(-1).astype(np.float64),
            "y": panel.y[pos].reshape(-1).astype(np.float64),
            "y_raw": panel.y_raw[pos].reshape(-1).astype(np.float64),
            "valid": panel.valid[pos].reshape(-1),
            "bench_mean": bench_mean,
            "fold": fold.index,
        }
    )


def run_walk_forward(
    cfg: Config, panel: Panel, folds: list[Fold], fold_dir: Path, force: bool = False
) -> tuple[pd.DataFrame, bool]:
    """返回 (长表预测, 模型输出是否为收益率量纲)。"""
    fold_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    produces_returns = True
    started = time.perf_counter()
    for fold in folds:
        path = fold_dir / f"fold_{fold.index:03d}.parquet"
        summary = fold.summary(panel.dates)
        if path.exists() and not force:
            frames.append(pd.read_parquet(path))
            produces_returns = build_model(cfg).produces_returns
            log.info("折 %02d（起点 %s）已有缓存，跳过训练", fold.index, summary["origin"])
            continue
        log.info(
            "折 %02d/%02d：训练 %s~%s（%d 天）| 验证 %s~%s（%d 天）| 测试 %s~%s（%d 天）",
            fold.index + 1,
            len(folds),
            summary["train_start"],
            summary["train_end"],
            summary["train_days"],
            summary["val_start"],
            summary["val_end"],
            summary["val_days"],
            summary["test_start"],
            summary["test_end"],
            summary["test_days"],
        )
        model = build_model(cfg)
        produces_returns = model.produces_returns
        fold_panel, pca_info = panel, None
        if cfg.features.pca_components > 0:
            fold_panel, pca_info = fit_fold_pca(panel, fold.train_pos, cfg.features.pca_components)
        t0 = time.perf_counter()
        info = model.fit(fold_panel, fold.train_pos, fold.val_pos)
        pred = model.predict(fold_panel, fold.test_pos)
        train_y = panel.y[fold.train_pos][panel.target_mask(fold.train_pos)]
        bench_mean = float(train_y.mean()) if len(train_y) else 0.0
        frame = _fold_frame(panel, fold, pred, bench_mean)
        frame.to_parquet(path)
        if cfg.train.save_checkpoints and hasattr(model, "save"):
            model.save(fold_dir / f"fold_{fold.index:03d}_ckpt")
        write_json(
            fold_dir / f"fold_{fold.index:03d}.json",
            {"summary": summary, "fit": info, "pca": pca_info, "seconds": time.perf_counter() - t0},
        )
        frames.append(frame)
    log.info("walk-forward 完成：%d 折，用时 %.1fs", len(folds), time.perf_counter() - started)
    predictions = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "industry"])
        .reset_index(drop=True)
    )
    return predictions, produces_returns
