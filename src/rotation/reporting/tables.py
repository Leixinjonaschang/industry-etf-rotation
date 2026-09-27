"""表格输出：CSV + 汇总 Excel（论文直接取用）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_excel(tables: dict[str, pd.DataFrame | None], path: Path) -> Path | None:
    """每张表一个工作表；没有任何非空表时不生成文件。"""
    tables = {k: v for k, v in tables.items() if v is not None and len(v)}
    if not tables:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            frame = table.copy()
            if isinstance(frame.index, pd.DatetimeIndex):
                frame.index = frame.index.strftime("%Y-%m-%d")
            for col in frame.columns:
                if pd.api.types.is_datetime64_any_dtype(frame[col]):
                    frame[col] = frame[col].dt.strftime("%Y-%m-%d")
            frame.to_excel(writer, sheet_name=name[:31])
    return path


def save_csv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, encoding="utf-8-sig")
    return path
