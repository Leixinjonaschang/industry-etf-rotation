"""神经网络模型封装：多随机种子训练，预测取平均（降低深度模型在金融数据上的方差）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rotation.config import TrainConfig
from rotation.dataset.panel import Panel
from rotation.models.base import BaseModel
from rotation.utils.logging import get_logger

log = get_logger("models.nn")

NN_DEFAULTS: dict[str, Any] = {
    "seq_len": 40,
    "hidden": 64,
    "layers": 2,
    "dropout": 0.2,
    "emb_dim": 8,
    "cross_attention": False,
}


class NNModel(BaseModel):
    produces_returns = True

    def __init__(
        self, params: dict[str, Any], train_cfg: TrainConfig, seed: int = 42, encoder: str = "lstm"
    ):
        super().__init__({**NN_DEFAULTS, **params}, train_cfg, seed)
        self.encoder = encoder
        self.name = encoder
        self.history = int(self.params["seq_len"])
        self.trainers: list = []
        self._bundle = None

    def fit(self, panel: Panel, train_pos: np.ndarray, val_pos: np.ndarray) -> dict[str, Any]:
        import torch

        from rotation.models.nn.trainer import NNTrainer, TensorBundle
        from rotation.utils.device import empty_cache, resolve_device

        if self.train_cfg.num_threads > 0:
            torch.set_num_threads(self.train_cfg.num_threads)
        device = resolve_device(self.train_cfg.device)
        self._bundle = TensorBundle.build(panel, train_pos, device)
        self.trainers, logs = [], []
        for k in range(max(self.train_cfg.n_seeds, 1)):
            trainer = NNTrainer(
                self.params, self.train_cfg, device, self.seed + 1000 * k, self.encoder
            )
            info = trainer.fit(panel, self._bundle, train_pos, val_pos)
            log.info(
                "  seed %d：最佳 epoch %d/%d，验证分数 %s，用时 %.1fs",
                info["seed"],
                info["best_epoch"],
                info["epochs_run"],
                f"{info['best_val_score']:.4f}" if info["best_val_score"] is not None else "—",
                info["seconds"],
            )
            self.trainers.append(trainer)
            logs.append(info)
        summary: dict[str, Any] = {
            "device": str(device),
            "y_scale": self._bundle.y_scale,
            "seeds": logs,
        }
        if len(val_pos):
            pred = self.predict(panel, val_pos)
            summary["val_rank_ic_ensemble"] = self.validation_rank_ic(panel, val_pos, pred)
        empty_cache(device)
        return summary

    def predict(self, panel: Panel, pos: np.ndarray) -> np.ndarray:
        from rotation.models.nn.trainer import TensorBundle

        if not self.trainers:
            raise RuntimeError("模型尚未训练")
        bundle = self._bundle
        if bundle is None or bundle.panel_id != id(panel):
            bundle = TensorBundle.build(panel, np.arange(0), self.trainers[0].device)
            bundle.y_scale = self._bundle.y_scale if self._bundle is not None else 1.0
        preds = [t.predict_scaled(bundle, pos) for t in self.trainers]
        return (np.mean(preds, axis=0) * bundle.y_scale).astype(np.float32)

    def save(self, folder: Path) -> None:
        import torch

        folder.mkdir(parents=True, exist_ok=True)
        for k, trainer in enumerate(self.trainers):
            torch.save(trainer.model.state_dict(), folder / f"seed{k}.pt")
