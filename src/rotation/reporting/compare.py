"""多实验对比：预测指标、策略指标，以及相对参照模型的显著性检验。

- RankIC 差：逐日 RankIC 序列之差的 HAC t 检验；
- DM 检验：逐日截面均方误差（仅当两个模型都输出收益率量纲时）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rotation.evaluation.stats import diebold_mariano, paired_hac_test
from rotation.features.selection import hac_lags
from rotation.reporting.tables import save_csv, to_excel
from rotation.utils.io import markdown_table, read_json

_PRED_KEYS = ["rank_ic", "rank_ic_t", "rank_icir", "long_short", "group_monotonicity"]
_STRAT_KEYS = ["annual_return", "sharpe", "max_drawdown", "excess_annual", "information_ratio"]


def _load(out_root: Path, name: str, phase: str) -> dict:
    folder = out_root / name / phase
    if not folder.exists():
        raise FileNotFoundError(f"找不到实验输出：{folder}")
    item: dict = {"folder": folder}
    for key, file in [
        ("pred", "prediction_metrics.csv"),
        ("strat", "strategy_metrics.csv"),
        ("series", "ic_series.csv"),
    ]:
        path = folder / file
        item[key] = (
            pd.read_csv(path, index_col=0, parse_dates=(key == "series")) if path.exists() else None
        )
    path = folder / "predictions.parquet"
    item["preds"] = pd.read_parquet(path) if path.exists() else None
    meta = folder / "predict_meta.json"
    item["produces_returns"] = bool(read_json(meta)["produces_returns"]) if meta.exists() else False
    cfg_path = folder / "config_resolved.yaml"
    if cfg_path.exists():
        import yaml

        item["horizon"] = int(
            yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["label"]["horizon"]
        )
    return item


def _daily_mse(preds: pd.DataFrame) -> pd.Series:
    ok = preds["valid"] & preds["y"].notna() & preds["pred"].notna()
    frame = preds[ok].assign(se=lambda d: (d["pred"] - d["y"]) ** 2)
    return frame.groupby("date")["se"].mean()


def compare_experiments(
    out_root: Path, names: list[str], phase: str, ref: str | None = None
) -> Path:
    items = {n: _load(out_root, n, phase) for n in names}
    ref = ref or names[0]
    rows = {}
    for name, item in items.items():
        row = {}
        pred = item["pred"]
        if pred is not None:
            hit_col = next(
                (c for c in pred.columns if c.startswith("hit@") and not c.endswith("random")), None
            )
            for key in _PRED_KEYS + ([hit_col] if hit_col else []):
                if key in pred.columns:
                    row[f"pred_{key}"] = pred.at["rebalance", key]
            row["pred_rank_ic_t_all"] = pred.at["all_days", "rank_ic_t"]
        strat = item["strat"]
        for series in ("strategy_index", "strategy_etf"):
            if strat is not None and series in strat.index:
                for key in _STRAT_KEYS:
                    row[f"{series}_{key}"] = strat.at[series, key]
        rows[name] = row
    table = pd.DataFrame(rows).T

    tests = []
    base = items[ref]
    horizon = base.get("horizon", 1)
    for name, item in items.items():
        if name == ref:
            continue
        record = {"model": name, "ref": ref}
        if base["series"] is not None and item["series"] is not None:
            joined = (
                base["series"]["rank_ic"]
                .to_frame("ref")
                .join(item["series"]["rank_ic"].to_frame("other"), how="inner")
            )
            res = paired_hac_test(
                joined["ref"].to_numpy(), joined["other"].to_numpy(), hac_lags(len(joined), horizon)
            )
            record.update(
                {
                    "rank_ic_diff_all_days(ref-other)": res["mean_diff"],
                    "rank_ic_diff_t": res["t"],
                    "rank_ic_diff_p": res["p_value"],
                }
            )
        if (
            base["produces_returns"]
            and item["produces_returns"]
            and base["preds"] is not None
            and item["preds"] is not None
        ):
            a, b = _daily_mse(base["preds"]), _daily_mse(item["preds"])
            common = a.index.intersection(b.index)
            dm = diebold_mariano(a.loc[common].to_numpy(), b.loc[common].to_numpy(), horizon)
            record.update({"dm_stat(ref-other)": dm["dm"], "dm_p": dm["p_value"]})
        tests.append(record)
    tests_df = pd.DataFrame(tests).set_index("model") if tests else pd.DataFrame()

    folder = out_root / "_compare" / phase / f"ref_{ref}"
    save_csv(table, folder / "comparison.csv")
    if len(tests_df):
        save_csv(tests_df, folder / "tests_vs_ref.csv")
    to_excel({"对比": table, "显著性检验": tests_df}, folder / "comparison.xlsx")
    text = ["# 实验对比（" + phase + "）", "", markdown_table(table.T), ""]
    if len(tests_df):
        text += [
            f"## 相对参照 {ref} 的检验（负的 DM 统计量表示参照模型误差更小）",
            "",
            markdown_table(tests_df),
            "",
        ]
    (folder / "comparison.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    return folder
