"""计算设备选择（macOS 优先 MPS）。

优先级（device: auto）：CUDA → MPS（Apple Silicon）→ CPU。
也可在配置里显式指定 `train.device: mps | cpu | cuda`。
小模型在 MPS 上不一定比 CPU 快，可用 `rotation device --bench` 实测后再决定。
"""

from __future__ import annotations

import os
import platform
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def mps_available() -> bool:
    import torch

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built() and backend.is_available())


def resolve_device(preference: str = "auto") -> Any:
    """把配置中的设备字符串解析为 torch.device。"""
    import torch

    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "mps" and not mps_available():
        raise RuntimeError(
            "MPS 不可用：需要 Apple Silicon、macOS 12.3+，以及带 MPS 支持的 PyTorch（PyPI 官方 macOS arm64 版即可）。"
        )
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")
    if pref not in {"cpu", "mps"} and not pref.startswith("cuda"):
        raise ValueError(f"未知设备：{preference}（可选 auto/mps/cpu/cuda）")
    return torch.device(pref)


def seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(seed)  # 同时作用于 CPU / CUDA / MPS 默认生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: Any) -> None:
    import gc

    import torch

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def device_report() -> dict[str, Any]:
    import torch

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_available": mps_available(),
        "cpu_threads": torch.get_num_threads(),
        "auto_device": str(resolve_device("auto")),
    }
