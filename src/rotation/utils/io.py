"""文件读写小工具：JSON/YAML、Markdown 表格。"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, (pd.Timestamp, datetime)):
        is_midnight = (obj.hour, obj.minute, obj.second, obj.microsecond) == (0, 0, 0, 0)
        return obj.strftime("%Y-%m-%d") if is_midnight else obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_to_builtin(obj), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _format_cell(value: Any, floatfmt: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return format(float(value), floatfmt)
    return str(value)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f", index: bool = True) -> str:
    """不依赖 tabulate 的 Markdown 表格。"""
    frame = df.reset_index() if index else df
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_format_cell(v, floatfmt) for v in row) + " |")
    return "\n".join(lines)
