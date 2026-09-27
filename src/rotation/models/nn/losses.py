"""带掩码的截面损失。pred / target / mask 形状均为 [B, N]（B 个日期）。

可用：mse、huber、ic（1 − 截面 Pearson）、listnet（截面 softmax 交叉熵），
以及组合 "mse+ic"、"huber+ic"、"mse+listnet" 等：基础项 + ic_weight × 排序项。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
_MIN_COUNT = 5


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    return ((pred - target) ** 2 * m).sum() / m.sum().clamp_min(1.0)


def masked_huber(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float
) -> torch.Tensor:
    m = mask.float()
    loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def ic_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    count = m.sum(dim=1)
    denom = count.clamp_min(1.0)
    p_mean = (pred * m).sum(dim=1) / denom
    y_mean = (target * m).sum(dim=1) / denom
    pc = (pred - p_mean[:, None]) * m
    yc = (target - y_mean[:, None]) * m
    cov = (pc * yc).sum(dim=1)
    corr = cov / (torch.sqrt((pc**2).sum(dim=1) + 1e-8) * torch.sqrt((yc**2).sum(dim=1) + 1e-8))
    valid = count >= _MIN_COUNT
    if not bool(valid.any()):
        return pred.sum() * 0.0
    return 1.0 - corr[valid].mean()


def listnet_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    valid_rows = mask.sum(dim=1) >= _MIN_COUNT
    if not bool(valid_rows.any()):
        return pred.sum() * 0.0
    p = pred[valid_rows]
    y = target[valid_rows]
    m = mask[valid_rows]
    neg = torch.finfo(p.dtype).min
    target_dist = torch.softmax((y / temperature).masked_fill(~m, neg), dim=1)
    log_prob = torch.log_softmax(p.masked_fill(~m, neg), dim=1)
    return -(target_dist * log_prob.masked_fill(~m, 0.0)).sum(dim=1).mean()


def build_loss(
    spec: str, ic_weight: float = 0.5, huber_delta: float = 1.0, temperature: float = 1.0
) -> LossFn:
    terms = [t.strip().lower() for t in spec.split("+") if t.strip()]
    known = {"mse", "huber", "ic", "listnet"}
    unknown = [t for t in terms if t not in known]
    if not terms or unknown:
        raise ValueError(f"未知损失：{spec}（可用 {sorted(known)} 及其 '+' 组合）")

    def single(term: str) -> LossFn:
        if term == "mse":
            return masked_mse
        if term == "huber":
            return lambda p, y, m: masked_huber(p, y, m, huber_delta)
        if term == "ic":
            return ic_loss
        return lambda p, y, m: listnet_loss(p, y, m, temperature)

    fns = [single(t) for t in terms]
    if len(fns) == 1:
        return fns[0]
    weights = [1.0] + [ic_weight] * (len(fns) - 1)

    def combined(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return sum(w * fn(pred, target, mask) for w, fn in zip(weights, fns))

    return combined
