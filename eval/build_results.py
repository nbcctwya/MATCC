"""Build the protocol v1.0 results artifacts for ONE market (per process).

Runs in the ``matcc`` conda env. Qlib can be initialised once per process per region,
so this script takes a single ``--universe`` and writes per-market staging files + curve
CSVs under ``results/``; :mod:`eval.finalize_results` then merges across markets.

It reuses the project's existing prediction/label pickles and re-runs the IDENTICAL
Qlib ``TopkDropoutStrategy`` backtest (same TopK/N_Drop, costs, exchange kwargs, benchmark
as ``backtest.py``) to obtain the daily report, from which the unified portfolio metrics
are computed. The original ``backtest_results/*_summary.csv`` (excess-return based) is
left untouched.

Per seed it writes:
  results/_staging/{market}_seed_metrics.csv         (1 row per seed)
  results/_cache/{market}_daily/{market}_seed_{s}.csv (daily gross/cost/net/bench)

Per (market, model, ensemble_method) it writes:
  results/_staging/{market}_ensemble_metrics.csv      (1 row per method)
  results/curves/ensemble/{market}_MATCC[_method].csv (daily curve)
  results/_cache/{market}_scores/{market}__{method}.pkl (ensemble score, for validation)

Usage:
    conda run -n matcc python eval/build_results.py --universe csi300 --tag 2009_2025
    conda run -n matcc python eval/build_results.py --universe sp500  --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import qlib  # noqa: E402
from qlib.constant import REG_CN, REG_US  # noqa: E402
from qlib.utils.time import Freq  # noqa: E402
from qlib.backtest import backtest, executor  # noqa: E402
from qlib.contrib.strategy import TopkDropoutStrategy  # noqa: E402

# Reuse the project's canonical strategy constants -> identical backtest to backtest.py.
from backtest import (  # noqa: E402
    TOPK, N_DROP, OPEN_COST, CLOSE_COST, ACCOUNT, REGION,
)
from src.baseline_utils import pred_path, labels_path, yaml_path, ensure_parent  # noqa: E402
from eval import BASELINE_ID, MODEL_ID  # noqa: E402
from eval.metrics import prediction_metrics, portfolio_metrics, nav_from_curve  # noqa: E402

REGION_CONST = {"cn": REG_CN, "us": REG_US}
ENSEMBLE_METHODS = ["avg_none", "avg_zscore", "avg_rank"]
SEED_METRIC_COLS = [
    "IC", "ICIR", "RankIC", "RankICIR",
    "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar",
]


# --------------------------------------------------------------------------- #
# Backtest (identical strategy to backtest.run_one_seed, returns the daily report)
# --------------------------------------------------------------------------- #
def run_backtest_report(signal, universe):
    """Run the project's TopK-DropN backtest and return Qlib's daily ``report_normal``.

    ``report_normal`` columns include ``return`` (gross daily portfolio return, cost added
    back per qlib/backtest/account.py), ``cost`` (transaction-cost rate), and ``bench``
    (benchmark daily return). Net daily return = return - cost (cost deducted once).
    """
    if isinstance(signal, pd.Series):
        signal = pd.DataFrame(signal)  # column "score", MultiIndex(datetime, instrument)
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

    strategy_obj = TopkDropoutStrategy(topk=TOPK, n_drop=N_DROP, signal=signal)
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
    return report_normal


def report_to_daily(report):
    """Extract a clean, date-ascending daily DataFrame from a Qlib report."""
    need = ["return", "cost", "bench"]
    missing = [c for c in need if c not in report.columns]
    if missing:
        raise RuntimeError(
            f"report_normal missing expected columns {missing}; got {list(report.columns)}"
        )
    daily = report[need].copy()
    daily.index = pd.DatetimeIndex(daily.index)
    daily = daily.sort_index()
    daily = daily[~daily.index.duplicated(keep="first")]
    daily = daily.rename(columns={"return": "gross", "bench": "bench"})
    daily["net"] = daily["gross"] - daily["cost"]  # cost deducted EXACTLY once
    daily["datetime"] = daily.index.strftime("%Y-%m-%d")
    if (daily["cost"] < -1e-9).any():
        raise RuntimeError("negative transaction cost encountered; check qlib cost semantics")
    return daily


# --------------------------------------------------------------------------- #
# Ensemble combination
# --------------------------------------------------------------------------- #
def ensemble_combine(score_df, method):
    """Combine per-seed scores (columns = seeds) into one score Series.

    - avg_none:    direct mean of raw scores.
    - avg_zscore:  per-day cross-sectional z-score (ddof=0) of each seed, then mean.
    - avg_rank:    per-day cross-sectional rank percentile of each seed, then mean.

    The day grouping uses the ``datetime`` index level; the inner-join index of
    ``score_df`` already guarantees every seed contributes to every sample.
    """
    if method == "avg_none":
        combined = score_df.mean(axis=1)
    elif method == "avg_zscore":
        z = score_df.groupby(level="datetime").transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)
        )
        combined = z.mean(axis=1)
    elif method == "avg_rank":
        ranked = score_df.groupby(level="datetime").rank(pct=True)
        combined = ranked.mean(axis=1)
    else:
        raise ValueError(f"unknown ensemble method: {method}")
    return combined.rename("score")


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _write_curve(daily, out_path, market, method):
    nav = nav_from_curve(daily["net"].to_numpy())
    bench_nav = nav_from_curve(daily["bench"].to_numpy())
    curve = pd.DataFrame({
        "datetime": daily["datetime"].to_numpy(),
        "daily_ret_gross": daily["gross"].to_numpy(),
        "cost": daily["cost"].to_numpy(),
        "daily_ret_net": daily["net"].to_numpy(),
        "bench_ret": daily["bench"].to_numpy(),
        "nav": nav,
        "bench_nav": bench_nav,
    })
    ensure_parent(out_path)
    curve.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, choices=["csi300", "sp500"])
    ap.add_argument("--tag", default="2009_2025")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = args.out
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    market = args.universe

    staging_dir = os.path.join(out, "_staging")
    cache_daily = os.path.join(out, "_cache", f"{market}_daily")
    cache_scores = os.path.join(out, "_cache", f"{market}_scores")
    curves_dir = os.path.join(out, "curves", "ensemble")
    for d in (staging_dir, cache_daily, cache_scores, curves_dir):
        os.makedirs(d, exist_ok=True)

    # ---- qlib init (once, for this region) -------------------------------- #
    with open(yaml_path(market), "r") as f:
        import yaml
        cfg = yaml.safe_load(f)
    region_str = cfg["qlib_init"]["region"]
    provider_uri = cfg["qlib_init"]["provider_uri"]
    qlib.init(provider_uri=provider_uri, region=REGION_CONST[region_str])
    print(f"== build_results: market={market} tag={args.tag} seeds={seeds} ==")

    # ---- per-seed --------------------------------------------------------- #
    seed_rows = []
    preds, labels = {}, {}
    for s in seeds:
        pp = pred_path(market, args.tag, s)
        lp = labels_path(market, args.tag, s)
        if not os.path.exists(pp) or not os.path.exists(lp):
            raise SystemExit(
                f"missing predictions/labels for {market} seed={s}: {pp} / {lp}. "
                "Run test.py first."
            )
        with open(pp, "rb") as f:
            pred = pickle.load(f)
        with open(lp, "rb") as f:
            lab = pickle.load(f)
        preds[s] = pred
        labels[s] = lab

        rank = prediction_metrics(pred, lab)
        report = run_backtest_report(pred, market)
        daily = report_to_daily(report)
        port = portfolio_metrics(daily["net"].to_numpy())

        # cache daily report so the validator can recompute seed portfolio metrics
        daily.to_csv(os.path.join(cache_daily, f"{market}_seed_{s}.csv"), index=False)

        rel_pred = os.path.relpath(pp, ROOT)
        row = {"market": market, "model": MODEL_ID, "seed": s}
        for c in SEED_METRIC_COLS:
            row[c] = rank.get(c) if c in rank else port.get(c)
        row["num_test_days"] = port["num_test_days"]
        row["pred_path_or_ckpt_path"] = rel_pred
        seed_rows.append(row)
        print(f"[seed {s}] IC={row['IC']:.4f} RankIC={row['RankIC']:.4f} "
              f"AR={row['AR']:.4f} MDD={row['MDD']:.4f} Sharpe={row['Sharpe']:.4f} "
              f"days={row['num_test_days']}")

    seed_df = pd.DataFrame(seed_rows, columns=[
        "market", "model", "seed", *SEED_METRIC_COLS, "num_test_days",
        "pred_path_or_ckpt_path",
    ])
    seed_path = os.path.join(staging_dir, f"{market}_seed_metrics.csv")
    seed_df.to_csv(seed_path, index=False)
    print(f"[build] seed staging -> {seed_path}")

    # ---- ensemble (>=2 seeds) -------------------------------------------- #
    if len(seeds) >= 2:
        # inner join on (datetime, instrument): keep only samples present in ALL seeds
        score_df = pd.concat({s: preds[s].rename("score") for s in seeds}, axis=1)
        score_df.columns = list(seeds)
        before = len(score_df)
        score_df = score_df.dropna()
        print(f"[ensemble] inner join kept {len(score_df)}/{before} (datetime,instrument)")
        # label aligned to the shared index (identical across seeds by construction)
        label_aligned = labels[seeds[0]].reindex(score_df.index)
        # sanity: labels agree across seeds on the intersection
        for s in seeds[1:]:
            if not label_aligned.equals(labels[s].reindex(score_df.index)):
                print("[ensemble] WARNING: labels differ across seeds on intersection")

        ens_rows = []
        for method in ENSEMBLE_METHODS:
            combined = ensemble_combine(score_df, method)
            rank = prediction_metrics(combined, label_aligned)
            report = run_backtest_report(combined, market)
            daily = report_to_daily(report)
            port = portfolio_metrics(daily["net"].to_numpy())

            # cache the ensemble score for independent IC re-check by the validator
            score_cache = os.path.join(cache_scores, f"{market}__{method}.pkl")
            with open(score_cache, "wb") as f:
                pickle.dump(combined, f)
            # cache the aligned label once per market
            label_cache = os.path.join(cache_scores, f"{market}__label.pkl")
            if not os.path.exists(label_cache):
                with open(label_cache, "wb") as f:
                    pickle.dump(label_aligned, f)

            # curve: default method uses the bare name; others get a __method suffix
            if method == "avg_none":
                curve_name = f"{market}_{MODEL_ID}.csv"
            else:
                curve_name = f"{market}_{MODEL_ID}__{method}.csv"
            _write_curve(daily, os.path.join(curves_dir, curve_name), market, method)

            row = {"market": market, "model": MODEL_ID, "ensemble_method": method}
            for c in SEED_METRIC_COLS:
                row[c] = rank.get(c) if c in rank else port.get(c)
            row["num_test_days"] = port["num_test_days"]
            row["seeds"] = ",".join(str(s) for s in seeds)
            row["pred_paths"] = os.path.relpath(score_cache, ROOT)
            ens_rows.append(row)
            print(f"[ensemble {method}] IC={row['IC']:.4f} RankIC={row['RankIC']:.4f} "
                  f"AR={row['AR']:.4f} MDD={row['MDD']:.4f} Sharpe={row['Sharpe']:.4f}")

        ens_df = pd.DataFrame(ens_rows, columns=[
            "market", "model", "ensemble_method", *SEED_METRIC_COLS, "num_test_days",
            "seeds", "pred_paths",
        ])
        ens_path = os.path.join(staging_dir, f"{market}_ensemble_metrics.csv")
        ens_df.to_csv(ens_path, index=False)
        print(f"[build] ensemble staging -> {ens_path}")
    else:
        print("[build] <2 seeds -> no ensemble")

    print(f"[build] DONE market={market}")


if __name__ == "__main__":
    main()
