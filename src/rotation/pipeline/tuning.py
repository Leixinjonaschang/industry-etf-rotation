"""开发期网格调参：只在 tune 阶段（2021–2022 伪测试）运行，按调仓日 RankIC 排序。

网格文件示例（configs/tuning/lstm_grid.yaml）：
    base: ../experiments/lstm_excess_h5.yaml
    set: [train.n_seeds=1]               # 所有试验共同的覆盖
    grid:
      model.params.seq_len: [20, 40, 60]
      train.lr: [3.0e-4, 1.0e-3]
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import yaml

from rotation.config import load_config
from rotation.pipeline.experiment import Experiment
from rotation.reporting.tables import save_csv
from rotation.utils.io import markdown_table
from rotation.utils.logging import get_logger

log = get_logger("tuning")


def run_grid(grid_file: Path, extra: list[str] | None = None) -> pd.DataFrame:
    spec = yaml.safe_load(grid_file.read_text(encoding="utf-8"))
    base_path = (grid_file.parent / spec["base"]).resolve()
    common = list(spec.get("set", [])) + list(extra or [])
    grid: dict[str, list] = spec.get("grid", {})
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys])) or [()]
    base_cfg = load_config(base_path, common)
    base_name = base_cfg.experiment.name
    rows = []
    for i, values in enumerate(combos, start=1):
        overrides = common + [
            f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in zip(keys, values)
        ]
        overrides += [f"experiment.name=tune__{base_name}__t{i:03d}", "experiment.phase=tune"]
        cfg = load_config(base_path, overrides)
        log.info("试验 %d/%d：%s", i, len(combos), dict(zip(keys, values)))
        exp = Experiment(cfg)
        exp.run(["features", "predict", "evaluate"])
        metrics = pd.read_csv(exp.out / "prediction_metrics.csv", index_col=0)
        k = cfg.strategy.top_n
        rows.append(
            {
                "trial": cfg.experiment.name,
                **dict(zip(keys, values)),
                "rank_ic": metrics.at["rebalance", "rank_ic"],
                "rank_ic_t_all": metrics.at["all_days", "rank_ic_t"],
                f"hit@{k}": metrics.at["rebalance", f"hit@{k}"],
                "long_short": metrics.at["rebalance", "long_short"],
            }
        )
    trials = pd.DataFrame(rows).sort_values("rank_ic", ascending=False).set_index("trial")
    folder = base_cfg.output_root / "_tuning" / base_name
    save_csv(trials, folder / "trials.csv")
    print(markdown_table(trials))
    log.info("调参结果 → %s", folder / "trials.csv")
    return trials
