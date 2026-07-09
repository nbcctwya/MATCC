"""
Qlib portfolio backtest for MATCC predictions: TopK-DropN (topk=30, n_drop=5) with
transaction costs (buy 5bps / sell 15bps), region-aware for CSI300 (CN) and SP500 (US).

Usage (one market per process, because qlib.init can run once per process):
    conda run -n matcc python backtest.py --universe csi300 --seeds 0,1,2,3,4 --tag 2009_2025
    conda run -n matcc python backtest.py --universe sp500  --seeds 0,1,2,3,4 --tag 2009_2025
    conda run -n matcc python backtest.py --universe csi300 --seeds 0 --smoke

For each seed it loads label_pred/{universe}/{tag}/{universe}_pred_{seed}.pkl, runs the
backtest over the prediction date range, and aggregates annualized_return / information_ratio
/ max_drawdown (excess return, with and without cost) mean +/- std across seeds into
backtest_results/{universe}_{tag}_summary.csv.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import qlib
from qlib.constant import REG_CN, REG_US
from qlib.utils.time import Freq
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy

from src.baseline_utils import REGION, ensure_parent, pred_path, summary_path, yaml_path

REGION_CONST = {"cn": REG_CN, "us": REG_US}

TOPK = 30
N_DROP = 5
OPEN_COST = 0.0005   # buy
CLOSE_COST = 0.0015  # sell
ACCOUNT = 1e8


def _extract(ra):
    """Pull (annualized_return, information_ratio, max_drawdown) from a risk_analysis df."""
    return (
        float(ra.loc["annualized_return", "risk"]),
        float(ra.loc["information_ratio", "risk"]),
        float(ra.loc["max_drawdown", "risk"]),
    )


def run_one_seed(universe, seed, tag, region_str):
    with open(pred_path(universe, tag, seed), "rb") as f:
        pred = pickle.load(f)
    if isinstance(pred, pd.Series):
        signal = pd.DataFrame(pred)  # column "score", MultiIndex(datetime, instrument)
    else:
        signal = pred.copy()

    dates = signal.index.get_level_values("datetime")
    start_time = str(pd.Timestamp(dates.min()).date())
    end_time = str(pd.Timestamp(dates.max()).date())

    reg = REGION[universe]
    exchange_kwargs = dict(
        freq="day",
        limit_threshold=reg["limit_threshold"],
        deal_price="close",
        open_cost=OPEN_COST,
        close_cost=CLOSE_COST,
        min_cost=reg["min_cost"],
    )
    if reg["trade_unit"] is not None:
        exchange_kwargs["trade_unit"] = reg["trade_unit"]

    strategy_obj = TopkDropoutStrategy(
        topk=TOPK, n_drop=N_DROP, signal=signal,
    )
    executor_obj = executor.SimulatorExecutor(
        time_per_step="day", generate_portfolio_metrics=True,
    )
    portfolio_metric_dict, _ = backtest(
        executor=executor_obj, strategy=strategy_obj,
        start_time=start_time, end_time=end_time,
        account=ACCOUNT, benchmark=reg["benchmark"],
        exchange_kwargs=exchange_kwargs,
    )
    analysis_freq = "{0}{1}".format(*Freq.parse("day"))
    report_normal, _ = portfolio_metric_dict.get(analysis_freq)

    ar_wo, ir_wo, mdd_wo = _extract(risk_analysis(
        report_normal["return"] - report_normal["bench"], freq=analysis_freq))
    ar_w, ir_w, mdd_w = _extract(risk_analysis(
        report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq))

    print(f"[backtest] {universe} seed={seed}: "
          f"excess(no cost) AR={ar_wo:.4f} IR={ir_wo:.4f} MDD={mdd_wo:.4f} | "
          f"excess(w/ cost) AR={ar_w:.4f} IR={ir_w:.4f} MDD={mdd_w:.4f}")
    return dict(seed=seed, AR_nocost=ar_wo, IR_nocost=ir_wo, MDD_nocost=mdd_wo,
                AR_cost=ar_w, IR_cost=ir_w, MDD_cost=mdd_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, choices=["csi300", "sp500"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="2009_2025")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tag = "smoke" if args.smoke else args.tag
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    with open(yaml_path(args.universe, smoke=args.smoke), "r") as f:
        import yaml
        cfg = yaml.safe_load(f)
    region_str = cfg["qlib_init"]["region"]
    provider_uri = cfg["qlib_init"]["provider_uri"]

    qlib.init(provider_uri=provider_uri, region=REGION_CONST[region_str])
    print(f"== backtest: universe={args.universe} tag={tag} region={region_str} "
          f"seeds={seeds} ==")

    rows = []
    for seed in seeds:
        try:
            rows.append(run_one_seed(args.universe, seed, tag, region_str))
        except FileNotFoundError:
            print(f"[backtest] missing predictions for seed={seed}, skipping.")

    if not rows:
        raise SystemExit("No seeds produced results; run test.py first.")

    df = pd.DataFrame(rows).set_index("seed")
    # add mean / std aggregation rows
    agg = pd.DataFrame({
        "mean": df.mean(numeric_only=True),
        "std": df.std(numeric_only=True),
    }).T
    out = pd.concat([df, agg])
    out.index.name = "seed"

    sp = summary_path(args.universe, tag)
    ensure_parent(sp)
    out.to_csv(sp)
    print(f"[backtest] summary -> {sp}")
    print(out.to_string())


if __name__ == "__main__":
    main()
