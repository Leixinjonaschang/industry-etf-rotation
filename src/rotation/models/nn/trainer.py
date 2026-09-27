"""神经网络训练器。

- 整个特征立方体一次性放到设备上（几 MB），按日期位置在设备端切出序列窗口，不用 DataLoader；
- 标签只按训练集标准差缩放（不平移），预测时乘回；
- 早停与学习率调度默认依据验证集 RankIC（与"选前 5 行业"的用法一致）。
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rotation.config import TrainConfig
from rotation.dataset.panel import Panel
from rotation.evaluation.ranking import daily_rank_ic
from rotation.models.nn.losses import build_loss
from rotation.models.nn.network import IndustryScorer
from rotation.utils.device import seed_torch, synchronize
from rotation.utils.logging import get_logger

log = get_logger("models.nn")


@dataclass
class TensorBundle:
    """设备上的共享张量（同一折内多个随机种子复用）。"""

    x: torch.Tensor  # [T, N, F+M]
    y: torch.Tensor  # [T, N] 已缩放，缺失为 0
    mask: torch.Tensor  # [T, N] bool
    y_scale: float
    panel_id: int

    @classmethod
    def build(cls, panel: Panel, train_pos: np.ndarray, device: torch.device) -> TensorBundle:
        x = panel.X
        if panel.M.shape[1] > 0:
            market = np.repeat(panel.M[:, None, :], panel.n_industries, axis=1)
            x = np.concatenate([x, market], axis=2)
        mask = panel.valid & np.isfinite(panel.y)
        train_values = panel.y[train_pos][mask[train_pos]]
        scale = float(np.std(train_values)) if len(train_values) > 1 else 1.0
        scale = scale if scale > 1e-8 else 1.0
        y = np.where(mask, panel.y / scale, 0.0).astype(np.float32)
        return cls(
            x=torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device),
            y=torch.from_numpy(y).to(device),
            mask=torch.from_numpy(mask).to(device),
            y_scale=scale,
            panel_id=id(panel),
        )


class NNTrainer:
    def __init__(
        self,
        params: dict[str, Any],
        train_cfg: TrainConfig,
        device: torch.device,
        seed: int,
        encoder: str,
    ):
        self.params = params
        self.cfg = train_cfg
        self.device = device
        self.seed = seed
        self.encoder = encoder
        self.seq_len = int(params.get("seq_len", 40))
        self.offsets = torch.arange(-self.seq_len + 1, 1, device=device)
        self.loss_fn = build_loss(
            train_cfg.loss,
            train_cfg.ic_weight,
            train_cfg.huber_delta,
            train_cfg.listnet_temperature,
        )
        self.model: IndustryScorer | None = None

    # ------------------------------------------------------------------ 工具
    def _windows(self, bundle: TensorBundle, pos: torch.Tensor) -> torch.Tensor:
        idx = (pos[:, None] + self.offsets[None, :]).clamp_min(0)  # [B, L]
        return bundle.x[idx].permute(0, 2, 1, 3).contiguous()  # [B, N, L, F]

    def _build_model(self, n_inputs: int, n_industries: int) -> IndustryScorer:
        p = self.params
        return IndustryScorer(
            n_inputs=n_inputs,
            n_industries=n_industries,
            encoder=self.encoder,
            hidden=int(p.get("hidden", 64)),
            layers=int(p.get("layers", 2)),
            dropout=float(p.get("dropout", 0.2)),
            emb_dim=int(p.get("emb_dim", 8)),
            cross_attention=bool(p.get("cross_attention", False)),
            attn_heads=int(p.get("attn_heads", 4)),
            tf_heads=int(p.get("tf_heads", 4)),
            tf_ff_mult=int(p.get("tf_ff_mult", 2)),
            max_len=max(int(p.get("max_len", 256)), self.seq_len),
            linear_skip=bool(p.get("linear_skip", False)),
        ).to(self.device)

    @torch.no_grad()
    def predict_scaled(self, bundle: TensorBundle, pos: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        out = []
        pos_t = torch.as_tensor(pos, dtype=torch.long, device=self.device)
        batch = max(self.cfg.batch_dates * 4, 64)
        for i in range(0, len(pos_t), batch):
            out.append(
                self.model(self._windows(bundle, pos_t[i : i + batch])).float().cpu().numpy()
            )
        if not out:
            return np.zeros((0, bundle.y.shape[1]), dtype=np.float32)
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def _val_loss(self, bundle: TensorBundle, pos: np.ndarray, pred_scaled: np.ndarray) -> float:
        pos_t = torch.as_tensor(pos, dtype=torch.long, device=self.device)
        pred = torch.from_numpy(pred_scaled).to(self.device)
        return float(self.loss_fn(pred, bundle.y[pos_t], bundle.mask[pos_t]).item())

    # ------------------------------------------------------------------ 训练
    def fit(
        self, panel: Panel, bundle: TensorBundle, train_pos: np.ndarray, val_pos: np.ndarray
    ) -> dict[str, Any]:
        cfg = self.cfg
        seed_torch(self.seed)
        n_inputs = int(bundle.x.shape[2])
        self.model = self._build_model(n_inputs, panel.n_industries)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        maximize = cfg.early_stop_metric == "rank_ic"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if maximize else "min",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
        )
        generator = torch.Generator().manual_seed(self.seed)
        train_t = torch.as_tensor(train_pos, dtype=torch.long)
        history: list[dict[str, float]] = []
        best_score, best_state, best_epoch, bad = -math.inf, None, 0, 0
        started = time.perf_counter()

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            order = train_t[torch.randperm(len(train_t), generator=generator)].to(self.device)
            total, batches = 0.0, 0
            for i in range(0, len(order), cfg.batch_dates):
                pos_b = order[i : i + cfg.batch_dates]
                pred = self.model(self._windows(bundle, pos_b))
                loss = self.loss_fn(pred, bundle.y[pos_b], bundle.mask[pos_b])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optimizer.step()
                total += float(loss.detach().item())
                batches += 1
            record = {
                "epoch": epoch,
                "train_loss": total / max(batches, 1),
                "lr": optimizer.param_groups[0]["lr"],
            }

            if len(val_pos):
                val_pred = self.predict_scaled(bundle, val_pos)
                ic = daily_rank_ic(val_pred, panel.y[val_pos], panel.target_mask(val_pos))
                record["val_rank_ic"] = (
                    float(np.nanmean(ic)) if np.isfinite(ic).any() else float("nan")
                )
                record["val_loss"] = self._val_loss(bundle, val_pos, val_pred)
                score = record["val_rank_ic"] if maximize else -record["val_loss"]
                score = score if np.isfinite(score) else -math.inf
                scheduler.step(score if maximize else record["val_loss"])
                if score > best_score + 1e-6:
                    best_score, best_epoch, bad = score, epoch, 0
                    best_state = copy.deepcopy(
                        {k: v.detach().to("cpu") for k, v in self.model.state_dict().items()}
                    )
                else:
                    bad += 1
            history.append(record)
            if len(val_pos) and epoch >= cfg.min_epochs and bad >= cfg.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        else:
            best_epoch = len(history)
        synchronize(self.device)
        return {
            "seed": self.seed,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "best_val_score": best_score if np.isfinite(best_score) else None,
            "seconds": round(time.perf_counter() - started, 2),
            "history": history,
        }
