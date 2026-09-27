"""设备基准：用与真实训练相同形状的随机数据，比较 CPU / MPS / CUDA 的训练速度。"""

from __future__ import annotations

import time

import torch

from rotation.models.nn.losses import build_loss
from rotation.models.nn.network import IndustryScorer
from rotation.utils.device import mps_available, synchronize


def _train_step(model, optimizer, loss_fn, x, y, mask) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y, mask)
    loss.backward()
    optimizer.step()


def available_devices() -> list[str]:
    devices = ["cpu"]
    if mps_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def benchmark(
    encoder: str = "lstm",
    n_industries: int = 30,
    seq_len: int = 40,
    n_features: int = 20,
    hidden: int = 64,
    layers: int = 2,
    batch_dates: int = 32,
    steps: int = 30,
    train_dates: int = 1500,
) -> list[dict]:
    results = []
    for name in available_devices():
        device = torch.device(name)
        torch.manual_seed(0)
        model = IndustryScorer(
            n_features, n_industries, encoder=encoder, hidden=hidden, layers=layers
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = build_loss("mse+ic")
        x = torch.randn(batch_dates, n_industries, seq_len, n_features, device=device)
        y = torch.randn(batch_dates, n_industries, device=device)
        mask = torch.ones_like(y, dtype=torch.bool)

        args = (model, optimizer, loss_fn, x, y, mask)
        for _ in range(3):  # 预热（MPS 首次调用会编译内核）
            _train_step(*args)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(steps):
            _train_step(*args)
        synchronize(device)
        per_step = (time.perf_counter() - started) / steps
        epoch_seconds = per_step * train_dates / batch_dates
        results.append(
            {"device": name, "ms_per_step": per_step * 1000, "sec_per_epoch": epoch_seconds}
        )
    return results
