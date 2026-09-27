"""论文图表（matplotlib，Agg 后端；macOS 上自动使用 PingFang/Heiti 等中文字体）。"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_CJK_FONTS = [
    "PingFang SC",
    "Heiti SC",
    "Hiragino Sans GB",
    "Songti SC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "SimHei",
    "Microsoft YaHei",
]


def _setup_fonts() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in _CJK_FONTS if f in available]
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    warnings.filterwarnings("ignore", message="Glyph .* missing from")


_setup_fonts()


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def nav_chart(navs: pd.DataFrame, labels: dict[str, str], path: Path, title: str) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for col in navs.columns:
        lw = 2.2 if col.startswith("strategy_etf") else 1.3
        ax1.plot(navs.index, navs[col], label=labels.get(col, col), linewidth=lw)
        dd = navs[col] / navs[col].cummax() - 1.0
        if col in ("strategy_etf", "strategy_index"):
            ax2.fill_between(dd.index, dd.to_numpy(), 0, alpha=0.3, label=labels.get(col, col))
    ax1.set_title(title)
    ax1.set_ylabel("净值")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("回撤")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8)
    return _save(fig, path)


def ic_chart(series: pd.DataFrame, path: Path, title: str) -> Path:
    reb = series[series["is_rebalance"]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.bar(
        reb.index,
        reb["rank_ic"],
        width=3,
        color=np.where(reb["rank_ic"] >= 0, "tab:red", "tab:green"),
    )
    ax1.plot(
        reb.index,
        reb["rank_ic"].rolling(12, min_periods=3).mean(),
        color="black",
        label="12 期均值",
    )
    ax1.axhline(0, color="grey", linewidth=0.8)
    ax1.set_title(title)
    ax1.set_ylabel("调仓日 RankIC")
    ax1.legend(fontsize=8)
    ax2.plot(reb.index, reb["rank_ic"].fillna(0).cumsum(), label="累计 RankIC")
    ax2.set_ylabel("累计")
    ax2.grid(alpha=0.3)
    return _save(fig, path)


def group_chart(groups: pd.DataFrame, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    values = groups["rebalance"] * 100
    ax.bar(groups.index, values, color=["tab:red" if v >= 0 else "tab:green" for v in values])
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("平均持有期收益 (%)")
    ax.set_title(title)
    return _save(fig, path)


def yearly_chart(yearly: pd.DataFrame, labels: dict[str, str], path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    (yearly * 100).rename(columns=labels).plot.bar(ax=ax)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("年度收益 (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def random_chart(distribution: np.ndarray, strategy: float, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(distribution * 100, bins=40, color="lightgrey", edgecolor="grey")
    ax.axvline(strategy * 100, color="tab:red", linewidth=2, label="策略")
    ax.set_xlabel("年化收益 (%)")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def training_curve_chart(histories: list[list[dict]], path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for hist in histories:
        epochs = [h["epoch"] for h in hist]
        ric = [h.get("val_rank_ic", np.nan) for h in hist]
        ax.plot(epochs, ric, alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("验证集 RankIC")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, path)
