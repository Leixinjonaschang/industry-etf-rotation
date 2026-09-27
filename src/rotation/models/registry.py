"""按名称构造模型（延迟导入：非神经网络实验不会加载 torch）。"""

from __future__ import annotations

from rotation.config import Config
from rotation.models.base import BaseModel

MODEL_NAMES = ("zero", "hist_mean", "momentum", "ridge", "gbdt", "lstm", "gru", "transformer")
NN_MODELS = ("lstm", "gru", "transformer")


def build_model(cfg: Config) -> BaseModel:
    name = cfg.model.name.lower()
    params = dict(cfg.model.params)
    seed = cfg.experiment.seed
    if name == "zero":
        from rotation.models.baselines import ZeroModel

        return ZeroModel(params, cfg.train, seed)
    if name == "hist_mean":
        from rotation.models.baselines import HistMeanModel

        return HistMeanModel(params, cfg.train, seed)
    if name == "momentum":
        from rotation.models.baselines import MomentumModel

        return MomentumModel(params, cfg.train, seed)
    if name == "ridge":
        from rotation.models.linear import RidgeModel

        return RidgeModel(params, cfg.train, seed)
    if name == "gbdt":
        from rotation.models.gbdt import GBDTModel

        return GBDTModel(params, cfg.train, seed)
    if name in NN_MODELS:
        from rotation.models.nn.model import NNModel

        return NNModel(params, cfg.train, seed, encoder=name)
    raise ValueError(f"未知模型：{cfg.model.name}（可选 {MODEL_NAMES}）")


def model_history(cfg: Config) -> int:
    """模型需要的历史窗口长度（用于确定最早可训练样本）。"""
    if cfg.model.name.lower() in NN_MODELS:
        return int(cfg.model.params.get("seq_len", 40))
    return 1
