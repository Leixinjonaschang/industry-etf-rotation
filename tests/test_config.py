"""配置加载：继承、覆盖、校验。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rotation.config import apply_override, load_config
from tests.conftest import PROJECT_ROOT


def test_layered_config_and_overrides():
    cfg = load_config(
        PROJECT_ROOT / "configs" / "experiments" / "lstm_raw_h5.yaml",
        ["model.params.hidden=128", "label.horizon=10", "features.categories=[momentum, trend]"],
    )
    assert cfg.experiment.name == "lstm_raw_h5"
    assert cfg.label.type == "raw" and cfg.label.horizon == 10
    assert cfg.model.params["hidden"] == 128 and cfg.model.params["seq_len"] == 40  # 深合并
    assert cfg.features.categories == ["momentum", "trend"]
    assert cfg.root == PROJECT_ROOT


def test_phase_controls_research_end():
    tune = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    test = load_config(PROJECT_ROOT / "configs" / "base.yaml", ["experiment.phase=test"])
    assert str(tune.research_end.date()) == "2022-12-31"
    assert str(test.research_end.date()) == "2025-12-31"
    assert tune.output_dir.name == "tune" and test.output_dir.name == "test"


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        load_config(PROJECT_ROOT / "configs" / "base.yaml", ["train.lerning_rate=0.1"])


def test_date_order_validated():
    with pytest.raises(ValidationError):
        load_config(PROJECT_ROOT / "configs" / "base.yaml", ["data.test_start=2020-01-01"])


def test_apply_override_parses_yaml():
    raw: dict = {}
    apply_override(raw, "a.b=[1, 2]")
    apply_override(raw, "a.c=true")
    assert raw == {"a": {"b": [1, 2], "c": True}}
