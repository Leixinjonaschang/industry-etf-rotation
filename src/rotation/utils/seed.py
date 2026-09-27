"""随机种子。torch 的种子在 rotation.utils.device.seed_torch 中设置（避免无谓导入 torch）。"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
