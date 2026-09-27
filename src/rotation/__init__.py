"""申万一级行业收益预测 + 行业/主题 ETF 映射轮动策略。"""

import os as _os

# MPS 上个别算子未实现时自动回退 CPU；必须在 torch 被导入之前设置。
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"
