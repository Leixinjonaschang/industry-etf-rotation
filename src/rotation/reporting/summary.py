"""单个实验的汇总报告：summary.md（含 G1–G4 关卡判定）+ summary.xlsx + figures/。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rotation.backtest.metrics import monthly_returns
from rotation.backtest.runner import SERIES_CN
from rotation.config import Config
from rotation.reporting import figures
from rotation.reporting.tables import to_excel
from rotation.utils.io import markdown_table, read_json
from rotation.utils.logging import get_logger

log = get_logger("reporting")

_PRED_COLS_CN = {
    "rank_ic": "RankIC 均值",
    "rank_ic_t": "RankIC t 值(HAC)",
    "rank_icir": "RankICIR",
    "rank_ic_win": "RankIC>0 占比",
    "long_short": "第1组−第5组",
    "group_monotonicity": "分组单调性",
    "reg_r2_oos": "R²_OOS",
    "reg_direction_acc": "方向准确率",
}
_STRAT_COLS_CN = {
    "annual_return": "年化收益",
    "annual_vol": "年化波动",
    "sharpe": "夏普",
    "max_drawdown": "最大回撤",
    "calmar": "卡玛",
    "excess_annual": "相对等权超额",
    "information_ratio": "信息比率",
    "avg_turnover": "单次换手",
    "cost_drag_annual": "年化成本",
}


def _read(path: Path, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(path, **kwargs) if path.exists() else None


def _gate_rows(
    cfg: Config, pred: pd.DataFrame | None, strat: pd.DataFrame | None, rnd: dict
) -> list[tuple]:
    k = cfg.strategy.top_n
    rows = []
    if pred is not None:
        reb, alld = pred.loc["rebalance"], pred.loc["all_days"]
        rows.append(
            (
                "G1",
                "RankIC>0 且 HAC t≥2",
                f"{reb['rank_ic']:.4f} / t={alld['rank_ic_t']:.2f}",
                reb["rank_ic"] > 0 and alld["rank_ic_t"] >= 2,
            )
        )
        rows.append(
            (
                "G1",
                f"前{k}命中率 > 随机",
                f"{reb[f'hit@{k}']:.1%} vs {reb[f'hit@{k}_random']:.1%}",
                reb[f"hit@{k}"] > reb[f"hit@{k}_random"],
            )
        )
        rows.append(("G1", "第1组−第5组 > 0", f"{reb['long_short']:.3%}", reb["long_short"] > 0))
    if strat is not None and "strategy_index" in strat.index:
        s = strat.loc["strategy_index"]
        rows.append(
            (
                "G3",
                "行业组合相对等权超额 > 0",
                f"{s['excess_annual']:.2%}（IR={s['information_ratio']:.2f}）",
                s["excess_annual"] > 0 and s["information_ratio"] > 0,
            )
        )
    if rnd:
        rows.append(
            ("G3", "随机选行业检验 p < 0.1", f"p={rnd['p_value']:.3f}", rnd["p_value"] < 0.1)
        )
    if strat is not None and "strategy_etf" in strat.index:
        s = strat.loc["strategy_etf"]
        rows.append(
            ("G4", "ETF 组合相对等权超额 > 0", f"{s['excess_annual']:.2%}", s["excess_annual"] > 0)
        )
    return rows


def write_report(cfg: Config, out: Path) -> Path:
    fig_dir = out / "figures"
    name = f"{cfg.experiment.name}（{cfg.experiment.phase}）"
    pred = _read(out / "prediction_metrics.csv", index_col=0)
    series = _read(out / "ic_series.csv", index_col=0, parse_dates=True)
    groups = _read(out / "group_returns.csv", index_col=0)
    yearly_pred = _read(out / "prediction_yearly.csv", index_col=0)
    navs = _read(out / "navs.csv", index_col=0, parse_dates=True)
    strat = _read(out / "strategy_metrics.csv", index_col=0)
    strat_hs = _read(out / "strategy_metrics_vs_hs300.csv", index_col=0)
    yearly = _read(out / "yearly_returns.csv", index_col=0)
    loss = _read(out / "mapping_loss_decomposition.csv", index_col=0)
    coverage = _read(out / "mapping_coverage.csv", index_col=0)
    selection = _read(out / "factor_selection_table.csv", index_col=0)
    rnd = read_json(out / "random_test.json") if (out / "random_test.json").exists() else {}
    if pred is None and navs is None:
        log.warning("%s 下没有预测或回测结果，请先运行 predict / evaluate / backtest 阶段", out)

    if series is not None:
        figures.ic_chart(series, fig_dir / "rank_ic.png", f"{name} 调仓日 RankIC")
    if groups is not None:
        figures.group_chart(
            groups, fig_dir / "group_returns.png", f"{name} 分组收益（第1组=预测最高）"
        )
    if navs is not None:
        figures.nav_chart(navs, SERIES_CN, fig_dir / "nav.png", f"{name} 净值与回撤")
    if yearly is not None:
        figures.yearly_chart(
            yearly, SERIES_CN, fig_dir / "yearly_returns.png", f"{name} 分年度收益"
        )
    dist_path = out / "random_distribution.npy"
    if rnd and dist_path.exists():
        figures.random_chart(
            np.load(dist_path),
            rnd["strategy_annual_gross"],
            fig_dir / "random_test.png",
            f"{name} 随机选 {cfg.strategy.top_n} 个行业的年化收益分布",
        )
    meta_path = out / "predict_meta.json"
    rel = read_json(meta_path).get("fold_dir") if meta_path.exists() else None
    fold_dir = out.parent.parent / rel if rel else None  # outputs/<实验>/<阶段> → outputs/
    fold_logs = sorted(fold_dir.glob("fold_*.json")) if fold_dir and fold_dir.is_dir() else []
    histories = []
    for path in fold_logs:
        fit = read_json(path).get("fit", {})
        histories.extend(seed.get("history", []) for seed in fit.get("seeds", []))
    if histories:
        figures.training_curve_chart(
            histories, fig_dir / "training_curves.png", f"{name} 各折各种子验证 RankIC"
        )

    lines = [f"# 实验报告：{name}", ""]
    gates = _gate_rows(cfg, pred, strat, rnd)
    if gates:
        gate_df = pd.DataFrame(gates, columns=["关卡", "条件", "结果", "通过"]).set_index("关卡")
        gate_df["通过"] = gate_df["通过"].map({True: "✅", False: "❌"})
        lines += ["## 关卡判定", "", markdown_table(gate_df), ""]
    if pred is not None:
        cols = [c for c in _PRED_COLS_CN if c in pred.columns]
        k = cfg.strategy.top_n
        extra = [c for c in (f"hit@{k}", f"hit@{k}_random", f"top{k}_excess") if c in pred.columns]
        lines += [
            "## 预测能力",
            "",
            markdown_table(pred[cols + extra].rename(columns=_PRED_COLS_CN)),
            "",
        ]
    if yearly_pred is not None:
        lines += [
            "### 分年度（调仓日口径）",
            "",
            markdown_table(
                yearly_pred[
                    [
                        c
                        for c in ("rank_ic", "rank_ic_t", f"hit@{cfg.strategy.top_n}", "long_short")
                        if c in yearly_pred.columns
                    ]
                ]
            ),
            "",
        ]
    if strat is not None:
        cols = [c for c in _STRAT_COLS_CN if c in strat.columns]
        table = strat[cols].rename(columns=_STRAT_COLS_CN).rename(index=SERIES_CN)
        lines += ["## 策略绩效（基准：30 行业等权）", "", markdown_table(table), ""]
    if strat_hs is not None and "excess_annual" in strat_hs.columns:
        table = (
            strat_hs[["excess_annual", "tracking_error", "information_ratio"]]
            .rename(
                columns={
                    "excess_annual": "相对沪深300超额",
                    "tracking_error": "跟踪误差",
                    "information_ratio": "信息比率",
                }
            )
            .rename(index=SERIES_CN)
        )
        lines += ["## 相对沪深300ETF", "", markdown_table(table), ""]
    if rnd:
        lines += [
            "## 随机选行业检验（不含成本）",
            "",
            markdown_table(pd.Series(rnd).to_frame("值")),
            "",
        ]
    if loss is not None:
        lines += ["## 映射损耗分解（每期收益差）", "", markdown_table(loss, ".4%"), ""]
    if coverage is not None and len(coverage):
        lines += [
            "## 映射覆盖率",
            "",
            markdown_table(coverage.drop(columns=["main_failure_reason"], errors="ignore")),
            "",
        ]
    lines += ["## 图表", ""] + [
        f"![{p.stem}](figures/{p.name})" for p in sorted(fig_dir.glob("*.png"))
    ]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tables = {
        "预测指标": pred,
        "分年度预测": yearly_pred,
        "分组收益": groups,
        "策略绩效": strat,
        "相对沪深300": strat_hs,
        "年度收益": yearly,
        "映射损耗": loss,
        "映射覆盖率": coverage,
        "因子筛选": selection,
    }
    if navs is not None and "strategy_etf" in navs.columns:
        tables["月度收益_ETF策略"] = monthly_returns(navs["strategy_etf"])
    to_excel({k: v for k, v in tables.items() if v is not None}, out / "summary.xlsx")
    log.info("报告 → %s", path)
    return path
