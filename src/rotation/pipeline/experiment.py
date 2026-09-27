"""实验编排：data → features → predict → evaluate → backtest → report。

每个阶段的产出都写到 outputs/<实验名>/<阶段 tune|test>/，后续阶段可单独重跑（从磁盘读取上游产物）。
"""

from __future__ import annotations

import platform
import subprocess
import time
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from rotation.config import Config
from rotation.data.cache import Cache
from rotation.data.etf import (
    ETFData,
    amount_unit_ratio,
    load_etf_data,
    load_etf_master,
    load_raw_daily,
)
from rotation.data.industry import IndustryData, load_industry_data
from rotation.data.validation import build_report
from rotation.dataset.panel import Panel, build_panel
from rotation.dataset.splits import make_folds
from rotation.evaluation.report import evaluate_predictions
from rotation.features.builder import FeatureSet, build_features
from rotation.features.registry import CATEGORY_CN
from rotation.features.selection import SelectionResult, apply_selection, select_factors
from rotation.labels import Labels, make_labels
from rotation.models.registry import model_history
from rotation.pipeline.walk_forward import fold_cache_key, run_walk_forward
from rotation.reporting.tables import save_csv
from rotation.utils.io import read_json, write_json, write_yaml
from rotation.utils.logging import get_logger, setup_logging
from rotation.utils.seed import set_global_seed

STAGES = ("data", "features", "predict", "evaluate", "backtest", "report")
log = get_logger("experiment")


class Experiment:
    def __init__(self, cfg: Config, force: bool = False):
        self.cfg = cfg
        self.force = force
        self.out = cfg.output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        setup_logging(self.out / "run.log")
        set_global_seed(cfg.experiment.seed)
        self.cache = Cache(cfg.cache_dir)
        self._predictions: pd.DataFrame | None = None
        self._produces_returns: bool | None = None

    # ------------------------------------------------------------------ 数据与特征（惰性计算）
    @cached_property
    def industry_full(self) -> IndustryData:
        return load_industry_data(self.cfg, self.cache)

    @cached_property
    def industry(self) -> IndustryData:
        """当前阶段可见的行情：tune 阶段截断到 dev_end，看不到 2023 年及以后的任何价格。"""
        return self.industry_full.truncate(self.cfg.research_end)

    @cached_property
    def etf(self) -> ETFData:
        full = load_etf_data(self.cfg, self.industry_full.dates, self.cache)
        return full.truncate(self.cfg.research_end)

    @cached_property
    def features_all(self) -> FeatureSet:
        # 因子只用过去数据（有测试保证），因此先在全样本上计算再截断，与先截断再计算完全等价
        return build_features(self.cfg, self.industry_full, self.cache).truncate(
            self.cfg.research_end
        )

    @cached_property
    def labels(self) -> Labels:
        lc = self.cfg.label
        return make_labels(
            self.industry["open"], self.industry["close"], lc.horizon, lc.type, lc.price
        )

    @cached_property
    def selection(self) -> SelectionResult | None:
        if not self.cfg.features.selection.enabled:
            return None
        return select_factors(
            self.features_all,
            self.labels,
            self.cfg.features.selection,
            pd.Timestamp(self.cfg.data.sample_start),
            pd.Timestamp(self.cfg.data.select_end),
            seed=self.cfg.experiment.seed,
        )

    @cached_property
    def panel(self) -> Panel:
        fs = self.features_all
        if self.selection is not None:
            fs = apply_selection(fs, self.selection)
        return build_panel(fs, self.labels, self.industry)

    # ------------------------------------------------------------------ 阶段
    def run(self, stages: list[str] | None = None) -> None:
        stages = list(stages or STAGES)
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise ValueError(f"未知阶段：{unknown}（可选 {STAGES}）")
        self._write_run_info()
        log.info(
            "实验 %s | 阶段 %s | 输出 %s",
            self.cfg.experiment.name,
            self.cfg.experiment.phase,
            self.out,
        )
        for stage in STAGES:
            if stage in stages:
                started = time.perf_counter()
                getattr(self, f"stage_{stage}")()
                log.info("阶段 [%s] 完成，用时 %.1fs", stage, time.perf_counter() - started)

    def stage_data(self) -> None:
        raw = load_raw_daily(self.cfg, load_etf_master(self.cfg), self.cache)
        report = build_report(self.industry, self.etf, self.cfg, amount_unit_ratio(self.cfg, raw))
        (self.out / "data_report.md").write_text(report, encoding="utf-8")
        log.info("数据报告 → %s", self.out / "data_report.md")

    def stage_features(self) -> None:
        sel = self.selection
        if sel is None:
            log.info("未启用因子筛选，使用全部 %d 个候选因子", len(self.features_all.names))
            return
        table = sel.table.copy()
        table.insert(0, "类别", table["category"].map(CATEGORY_CN))
        save_csv(table, self.out / "factor_selection_table.csv")
        write_json(self.out / "factor_selection.json", sel.to_dict())
        log.info(
            "最终特征（%d）：%s", len(sel.feature_names), ", ".join(sel.feature_names)
        )

    def stage_predict(self) -> None:
        cfg = self.cfg
        eval_start, eval_end = cfg.eval_window
        folds = make_folds(
            self.panel,
            cfg.split,
            eval_start,
            eval_end,
            train_start=pd.Timestamp(cfg.data.sample_start),
            min_history=model_history(cfg),
        )
        pd.DataFrame([f.summary(self.panel.dates) for f in folds]).to_csv(
            self.out / "folds.csv", index=False, encoding="utf-8-sig"
        )
        names = list(self.panel.feature_names)
        if self.selection is not None and self.selection.composites:
            names.append(f"selection:{self.selection.digest}")  # PCA 载荷变化时不复用旧缓存
        key = fold_cache_key(cfg, names, self.industry.fingerprint)
        # 逐折缓存按内容哈希共享：只改策略/映射/回测参数的实验会直接复用已训练的预测
        fold_dir = cfg.output_root / "_folds" / key
        preds, produces = run_walk_forward(cfg, self.panel, folds, fold_dir, self.force)
        preds.to_parquet(self.out / "predictions.parquet")
        write_json(
            self.out / "predict_meta.json",
            {
                "produces_returns": produces,
                "fold_key": key,
                "fold_dir": f"_folds/{key}",
                "n_folds": len(folds),
            },
        )
        self._predictions, self._produces_returns = preds, produces

    def _load_predictions(self) -> tuple[pd.DataFrame, bool]:
        if self._predictions is None:
            path = self.out / "predictions.parquet"
            if not path.exists():
                raise FileNotFoundError(f"找不到 {path}，请先运行 predict 阶段")
            self._predictions = pd.read_parquet(path)
            self._produces_returns = bool(
                read_json(self.out / "predict_meta.json")["produces_returns"]
            )
        return self._predictions, bool(self._produces_returns)

    def stage_evaluate(self) -> None:
        preds, produces = self._load_predictions()
        cfg = self.cfg
        summary, tables = evaluate_predictions(
            preds, self.panel.industries, cfg.label.horizon, cfg.strategy.top_n, produces
        )
        save_csv(summary, self.out / "prediction_metrics.csv")
        save_csv(tables["series"], self.out / "ic_series.csv")
        save_csv(tables["yearly"], self.out / "prediction_yearly.csv")
        save_csv(tables["groups"], self.out / "group_returns.csv")
        reb = summary.loc["rebalance"]
        allr = summary.loc["all_days"]
        k = cfg.strategy.top_n
        log.info(
            "预测评价（调仓日口径）：RankIC=%.4f（全日 HAC t=%.2f）| 前%d命中率=%.1f%%（随机 %.1f%%）| 前%d超额=%.3f%% | 多空=%.3f%%",
            reb["rank_ic"],
            allr["rank_ic_t"],
            k,
            100 * reb[f"hit@{k}"],
            100 * reb[f"hit@{k}_random"],
            k,
            100 * reb[f"top{k}_excess"],
            100 * reb["long_short"],
        )

    def stage_backtest(self) -> None:
        from rotation.backtest.runner import run_backtests
        from rotation.mapping.diagnostics import coverage_by_industry, status_summary
        from rotation.mapping.mapper import ETFMapper
        from rotation.mapping.whitelist import Whitelist
        from rotation.strategy.signals import build_signals

        cfg = self.cfg
        preds, _ = self._load_predictions()
        signals = build_signals(
            preds, self.panel.industries, cfg.label.horizon, cfg.strategy.top_n, cfg.strategy.buffer
        )
        save_csv(signals.set_index("signal_date"), self.out / "signals.csv")

        mapping = None
        if cfg.mapping.enabled:
            whitelist = Whitelist.load(cfg.whitelist_path)
            mapper = ETFMapper(
                cfg.mapping, self.industry["close"], self.etf, whitelist, cfg.strategy.top_n
            )
            mapping = mapper.run(signals)
            save_csv(mapping.slots.set_index("signal_date"), self.out / "mapping_slots.csv")
            save_csv(mapping.attempts.set_index("signal_date"), self.out / "mapping_attempts.csv")
            save_csv(coverage_by_industry(mapping), self.out / "mapping_coverage.csv")
            save_csv(status_summary(mapping), self.out / "mapping_status.csv")
            for date in self._panorama_dates():
                table = mapper.panorama(date)
                save_csv(table, self.out / f"mapping_panorama_{table.attrs['date']}.csv")
            write_json(
                self.out / "whitelist_version.json",
                {"version": whitelist.version, "frozen_at": whitelist.frozen_at},
            )

        bt = run_backtests(cfg, signals, self.industry, self.etf, mapping)
        save_csv(bt.navs, self.out / "navs.csv")
        save_csv(bt.metrics, self.out / "strategy_metrics.csv")
        if len(bt.metrics_vs_hs300):
            save_csv(bt.metrics_vs_hs300, self.out / "strategy_metrics_vs_hs300.csv")
        save_csv(bt.yearly, self.out / "yearly_returns.csv")
        if bt.periods is not None:
            save_csv(bt.periods, self.out / "period_returns.csv")
        if bt.loss_decomposition is not None:
            save_csv(bt.loss_decomposition, self.out / "mapping_loss_decomposition.csv")
        for name, trades in bt.trades.items():
            if len(trades):
                save_csv(trades.set_index("trade_date"), self.out / "trades" / f"{name}.csv")
        write_json(self.out / "random_test.json", bt.random_test)
        if bt.random_distribution is not None and len(bt.random_distribution):
            np.save(self.out / "random_distribution.npy", bt.random_distribution)
        m = bt.metrics
        for key in ("strategy_etf", "strategy_index", "ew_industry"):
            if key in m.index:
                log.info(
                    "%-20s 年化 %6.2f%% | 夏普 %5.2f | 最大回撤 %6.2f%% | 相对等权超额 %6.2f%%",
                    key,
                    100 * m.at[key, "annual_return"],
                    m.at[key, "sharpe"],
                    100 * m.at[key, "max_drawdown"],
                    100 * m.at[key, "excess_annual"],
                )
        if bt.random_test:
            log.info(
                "随机选行业检验：策略（毛）年化 %.2f%%，随机分布中位数 %.2f%%，p=%.3f",
                100 * bt.random_test["strategy_annual_gross"],
                100 * bt.random_test["random_p50"],
                bt.random_test["p_value"],
            )

    def _panorama_dates(self) -> list[pd.Timestamp]:
        eval_start, eval_end = self.cfg.eval_window
        dates = self.industry.dates
        out = []
        for year in range(eval_start.year - 1, eval_end.year + 1):
            in_year = dates[dates.year == year]
            if len(in_year):
                out.append(in_year[-1])
        return out

    def stage_report(self) -> None:
        from rotation.reporting.summary import write_report

        write_report(self.cfg, self.out)

    # ------------------------------------------------------------------ 元信息
    def _write_run_info(self) -> None:
        write_yaml(self.out / "config_resolved.yaml", self.cfg.dump())
        info = {
            "experiment": self.cfg.experiment.name,
            "phase": self.cfg.experiment.phase,
            "research_end": str(self.cfg.research_end.date()),
            "eval_window": [str(d.date()) for d in self.cfg.eval_window],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "git_commit": _git_commit(self.cfg.root),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(self.out / "env.json", info)
        if self.cfg.experiment.phase == "test":
            ledger = self.cfg.output_root / "TEST_RUNS.md"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            is_new = not ledger.exists()
            with ledger.open("a", encoding="utf-8") as handle:
                if is_new:
                    handle.write(
                        "# 测试期运行记录\n\n| 时间 | 实验 | git commit |\n|---|---|---|\n"
                    )
                handle.write(
                    f"| {info['started_at']} | {info['experiment']} | {info['git_commit']} |\n"
                )


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "pyarrow"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "-"
    return out


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = result.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return f"{commit}{'+dirty' if dirty else ''}" if commit else "-"
    except Exception:  # noqa: BLE001
        return "-"
