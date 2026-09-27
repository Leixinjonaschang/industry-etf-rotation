"""批量运行实验矩阵：每个实验在独立子进程中运行（隔离内存与 OpenMP 运行库），最后自动对比。

矩阵文件示例（configs/matrix/main.yaml）：
    ref: lstm_excess_h5
    experiments:
      - configs/experiments/lstm_excess_h5.yaml
      - config: configs/experiments/lstm_excess_h5.yaml
        name: lstm_excess_h10
        set: [label.horizon=10]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from rotation.config import find_project_root, load_config, load_default_config
from rotation.reporting.compare import compare_experiments
from rotation.utils.logging import get_logger

log = get_logger("matrix")


def run_matrix(
    matrix_file: Path, phase: str, extra: list[str] | None = None, skip_existing: bool = False
) -> None:
    spec = yaml.safe_load(matrix_file.read_text(encoding="utf-8"))
    root = find_project_root(matrix_file.parent)
    names = []
    for entry in spec["experiments"]:
        entry = {"config": entry} if isinstance(entry, str) else dict(entry)
        config = (root / entry["config"]).resolve()
        overrides = list(entry.get("set", [])) + list(extra or [])
        if entry.get("name"):
            overrides.append(f"experiment.name={entry['name']}")
        overrides.append(f"experiment.phase={phase}")
        cfg = load_config(config, overrides)
        names.append(cfg.experiment.name)
        if skip_existing and (cfg.output_dir / "summary.md").exists():
            log.info("跳过已完成的实验：%s", cfg.experiment.name)
            continue
        cmd = [
            sys.executable,
            "-m",
            "rotation",
            "run",
            "--config",
            str(config),
            "--phase",
            phase,
            "--yes",
        ]
        for item in overrides:
            cmd += ["--set", item]
        log.info("运行：%s", cfg.experiment.name)
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            raise RuntimeError(f"实验 {cfg.experiment.name} 失败（返回码 {result.returncode}）")
    if len(names) > 1:
        compare_experiments(load_default_config(root).output_root, names, phase, spec.get("ref"))
