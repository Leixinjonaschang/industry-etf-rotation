"""端到端冒烟测试：合成数据文件 → 全部阶段（数据、特征、预测、评价、回测、报告）。"""

from __future__ import annotations

import pandas as pd
import pytest

from rotation.config import load_config
from rotation.pipeline.experiment import Experiment
from tests.conftest import PROJECT_ROOT, write_raw_files


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory, industry, etf):
    return write_raw_files(tmp_path_factory.mktemp("raw"), industry, etf)


def _config(tmp_path, raw_dir, name: str, extra: list[str]):
    overrides = [
        f"paths.data_dir={raw_dir}",
        f"paths.cache_dir={tmp_path / 'cache'}",
        f"paths.output_dir={tmp_path / 'out'}",
        f"paths.etf_whitelist={PROJECT_ROOT / 'configs' / 'etf_whitelist.yaml'}",
        "data.load_start=2019-01-01",
        "data.sample_start=2019-07-01",
        "data.select_end=2020-12-31",
        "data.tune_start=2021-01-01",
        "data.dev_end=2021-12-31",
        "data.test_start=2022-01-01",
        "data.test_end=2022-06-30",
        "split.retrain_every=120",
        "split.val_len=60",
        "split.min_train=100",
        "features.selection.min_factors=3",
        "features.selection.max_factors=6",
        "backtest.n_random=50",
        f"experiment.name={name}",
        *extra,
    ]
    return load_config(PROJECT_ROOT / "configs" / "base.yaml", overrides)


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    for var in ("ROTATION_DATA_DIR", "ROTATION_CACHE_DIR", "ROTATION_OUTPUT_DIR"):
        monkeypatch.delenv(var, raising=False)


def _check_outputs(out):
    for file in (
        "data_report.md",
        "predictions.parquet",
        "prediction_metrics.csv",
        "navs.csv",
        "mapping_slots.csv",
        "strategy_metrics.csv",
        "summary.md",
        "summary.xlsx",
    ):
        assert (out / file).exists(), file
    navs = pd.read_csv(out / "navs.csv", index_col=0)
    assert {"strategy_etf", "strategy_index", "ew_industry", "hs300_etf"} <= set(navs.columns)
    assert navs.notna().all().all()


def test_momentum_end_to_end(tmp_path, raw_dir):
    cfg = _config(
        tmp_path, raw_dir, "smoke_momentum", ["model.name=momentum", "model.params={lookback: 20}"]
    )
    exp = Experiment(cfg)
    exp.run()
    _check_outputs(exp.out)
    assert len(exp.industry.industries) == 8  # "综合"已剔除
    assert exp.industry.dates.max() <= pd.Timestamp("2021-12-31")  # tune 阶段看不到之后的数据


def test_ridge_test_phase(tmp_path, raw_dir):
    cfg = _config(tmp_path, raw_dir, "smoke_ridge", ["model.name=ridge", "experiment.phase=test"])
    exp = Experiment(cfg)
    exp.run()
    _check_outputs(exp.out)
    assert (cfg.output_root / "TEST_RUNS.md").exists()


def test_lstm_end_to_end(tmp_path, raw_dir):
    extra = [
        "model.name=lstm",
        "model.params={seq_len: 10, hidden: 8, layers: 1, emb_dim: 2, cross_attention: true}",
        "train.epochs=2",
        "train.min_epochs=1",
        "train.n_seeds=2",
        "train.device=cpu",
        "train.batch_dates=16",
    ]
    cfg = _config(tmp_path, raw_dir, "smoke_lstm", extra)
    exp = Experiment(cfg)
    exp.run()
    _check_outputs(exp.out)
    # 第二次运行应直接命中逐折缓存
    Experiment(cfg).run(["predict"])
