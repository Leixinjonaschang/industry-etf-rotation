"""基于 parquet / npz 的简单缓存；键由源文件指纹与配置哈希组成，源数据变化自动失效。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from rotation.utils.io import read_json, write_json
from rotation.utils.logging import get_logger

log = get_logger("cache")


class Cache:
    def __init__(self, root: Path, enabled: bool = True):
        self.root = root
        self.enabled = enabled

    def _dir(self, name: str, key: str) -> Path:
        return self.root / f"{name}_{key}"

    def frames(
        self, name: str, key: str, builder: Callable[[], dict[str, pd.DataFrame]]
    ) -> dict[str, pd.DataFrame]:
        """一组 DataFrame，每个存为一个 parquet 文件。"""
        folder = self._dir(name, key)
        done = folder / "_done.json"
        if self.enabled and done.exists():
            names = read_json(done)["frames"]
            return {n: pd.read_parquet(folder / f"{n}.parquet") for n in names}
        frames = builder()
        if self.enabled:
            folder.mkdir(parents=True, exist_ok=True)
            for n, frame in frames.items():
                frame.to_parquet(folder / f"{n}.parquet")
            write_json(done, {"frames": list(frames)})
            log.info("已缓存 %s → %s", name, folder)
        return frames

    def arrays(
        self, name: str, key: str, builder: Callable[[], tuple[dict[str, np.ndarray], dict]]
    ) -> tuple[dict[str, np.ndarray], dict]:
        """一组 numpy 数组 + JSON 元数据。"""
        folder = self._dir(name, key)
        done = folder / "_done.json"
        if self.enabled and done.exists():
            with np.load(folder / "arrays.npz", allow_pickle=False) as npz:
                arrays = {k: npz[k] for k in npz.files}
            return arrays, read_json(folder / "meta.json")
        arrays, meta = builder()
        if self.enabled:
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "arrays.npz", **arrays)
            write_json(folder / "meta.json", meta)
            write_json(done, {"ok": True})
            log.info("已缓存 %s → %s", name, folder)
        return arrays, meta
