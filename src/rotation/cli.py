"""命令行入口：`uv run rotation <子命令>`（等价于 `python -m rotation`）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rotation.pipeline.experiment import STAGES


def _confirm_test_phase(args: argparse.Namespace) -> bool:
    if getattr(args, "yes", False):
        return True
    message = (
        "即将运行【测试期 2023–2025】。按研究设计，测试期只应在所有设定（因子、超参数、标签、"
        "白名单、映射门槛）冻结后运行一次，结果不得回头修改设定。确认继续？[y/N] "
    )
    if not sys.stdin.isatty():
        print("测试期运行需要确认：请加 --yes 参数。", file=sys.stderr)
        return False
    return input(message).strip().lower() in ("y", "yes")


def cmd_run(args: argparse.Namespace) -> int:
    from rotation.config import load_config
    from rotation.pipeline.experiment import Experiment

    overrides = list(args.set or [])
    if args.phase:
        overrides.append(f"experiment.phase={args.phase}")
    cfg = load_config(args.config, overrides)
    if cfg.experiment.phase == "test" and not _confirm_test_phase(args):
        return 1
    stages = args.stages.split(",") if args.stages else None
    Experiment(cfg, force=args.force).run(stages)
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    from rotation.pipeline.tuning import run_grid

    run_grid(Path(args.grid), args.set)
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    from rotation.pipeline.matrix import run_matrix

    if args.phase == "test" and not _confirm_test_phase(args):
        return 1
    run_matrix(Path(args.file), args.phase, args.set, skip_existing=args.skip_existing)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from rotation.config import load_config, load_default_config
    from rotation.reporting.compare import compare_experiments

    cfg = load_config(args.config) if args.config else load_default_config()
    compare_experiments(cfg.output_root, args.experiment, args.phase, args.ref)
    return 0


def cmd_device(args: argparse.Namespace) -> int:
    from rotation.utils.device import device_report

    report = device_report()
    for key, value in report.items():
        print(f"{key:>15}: {value}")
    if args.bench:
        from rotation.models.nn.bench import benchmark

        print("\n训练速度基准（LSTM，30 行业 × 40 天窗口，batch=32 个日期）：")
        results = benchmark(encoder=args.encoder, seq_len=args.seq_len, hidden=args.hidden)
        for r in results:
            print(
                f"  {r['device']:>4}: {r['ms_per_step']:7.1f} ms/step ≈ {r['sec_per_epoch']:6.1f} s/epoch（1500 个训练日）"
            )
        best = min(results, key=lambda r: r["ms_per_step"])
        print(f"\n建议：train.device: {best['device']}")
    return 0


def cmd_factors(args: argparse.Namespace) -> int:
    from rotation.features.registry import CATEGORY_CN, all_factors, all_market_features

    for spec in all_factors().values():
        print(f"{CATEGORY_CN[spec.category]:<4} {spec.name:<22} {spec.description}")
    for spec in all_market_features().values():
        print(f"{'市场':<4} {spec.name:<22} {spec.description}")
    return 0


def cmd_mapping_table(args: argparse.Namespace) -> int:
    import pandas as pd

    from rotation.config import load_config
    from rotation.mapping.mapper import ETFMapper
    from rotation.mapping.whitelist import Whitelist
    from rotation.pipeline.experiment import Experiment
    from rotation.utils.io import markdown_table

    overrides = list(args.set or [])
    cfg = load_config(args.config, overrides)
    date = pd.Timestamp(args.date)
    if date > cfg.research_end:
        print(
            f"{date.date()} 超出当前阶段（{cfg.experiment.phase}）可见区间 {cfg.research_end.date()}",
            file=sys.stderr,
        )
        return 1
    exp = Experiment(cfg)
    mapper = ETFMapper(
        cfg.mapping,
        exp.industry["close"],
        exp.etf,
        Whitelist.load(cfg.whitelist_path),
        cfg.strategy.top_n,
    )
    table = mapper.panorama(date)
    print(f"映射全景表（{table.attrs['date']}）\n")
    print(markdown_table(table, ".3f"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rotation", description="申万行业轮动 + ETF 映射策略")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="运行一个实验")
    run.add_argument("--config", "-c", required=True)
    run.add_argument("--phase", choices=["tune", "test"], help="覆盖 experiment.phase")
    run.add_argument("--stages", help=f"逗号分隔，可选 {','.join(STAGES)}（默认全部）")
    run.add_argument("--set", action="append", metavar="KEY=VALUE", help="覆盖任意配置项，可重复")
    run.add_argument("--force", action="store_true", help="忽略已有的逐折缓存，重新训练")
    run.add_argument("--yes", "-y", action="store_true", help="测试期运行时跳过确认")
    run.set_defaults(func=cmd_run)

    tune = sub.add_parser("tune", help="开发期网格调参（只用 tune 阶段）")
    tune.add_argument("--grid", required=True)
    tune.add_argument("--set", action="append", metavar="KEY=VALUE")
    tune.set_defaults(func=cmd_tune)

    matrix = sub.add_parser("matrix", help="批量运行实验矩阵并自动对比")
    matrix.add_argument("--file", required=True)
    matrix.add_argument("--phase", choices=["tune", "test"], default="tune")
    matrix.add_argument("--set", action="append", metavar="KEY=VALUE")
    matrix.add_argument("--skip-existing", action="store_true")
    matrix.add_argument("--yes", "-y", action="store_true")
    matrix.set_defaults(func=cmd_matrix)

    compare = sub.add_parser("compare", help="对比多个已完成的实验")
    compare.add_argument("--experiment", "-e", action="append", required=True)
    compare.add_argument("--phase", choices=["tune", "test"], default="tune")
    compare.add_argument("--ref", help="参照实验（默认第一个）")
    compare.add_argument("--config", help="用于定位 outputs 目录（默认 configs/base.yaml 的设置）")
    compare.set_defaults(func=cmd_compare)

    device = sub.add_parser("device", help="查看计算设备；--bench 实测 CPU/MPS 训练速度")
    device.add_argument("--bench", action="store_true")
    device.add_argument("--encoder", default="lstm", choices=["lstm", "gru", "transformer"])
    device.add_argument("--seq-len", type=int, default=40)
    device.add_argument("--hidden", type=int, default=64)
    device.set_defaults(func=cmd_device)

    factors = sub.add_parser("factors", help="列出全部已注册的候选因子")
    factors.set_defaults(func=cmd_factors)

    table = sub.add_parser("mapping-table", help="打印某日 30 个行业的最优 ETF 映射")
    table.add_argument("--config", "-c", required=True)
    table.add_argument("--date", required=True)
    table.add_argument("--set", action="append", metavar="KEY=VALUE")
    table.set_defaults(func=cmd_mapping_table)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
