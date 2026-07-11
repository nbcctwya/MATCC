"""Merge per-market staging files into the canonical protocol v1.0 results/ tree.

Pure pandas + stdlib (no qlib). Reads what :mod:`eval.build_results` wrote under
``results/_staging/`` for every market and produces:

  results/metrics/seed_metrics.csv       (market,model,seed -> 1 row each)
  results/metrics/aggregate_metrics.csv  (market,model -> mean/std across seeds)
  results/metrics/ensemble_metrics.csv   (market,model,ensemble_method -> 1 row each)
  results/tables/seed_mean_std.csv       (4-dp "mean ± std" presentation)
  results/tables/ensemble.csv            (4-dp presentation of ensemble_metrics)
  results/metadata/eval_config.json      (real run口径)
  results/metadata/manifest.json         (real file registry)

Usage:
    python eval/finalize_results.py [--out results] [--tag 2009_2025]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.baseline_utils import yaml_path  # noqa: E402
from backtest import TOPK, N_DROP, OPEN_COST, CLOSE_COST, ACCOUNT  # noqa: E402
from eval import BASELINE_ID, MODEL_ID, ANNUAL, DDOF  # noqa: E402

METRIC_COLS = [
    "IC", "ICIR", "RankIC", "RankICIR",
    "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar",
]
SEED_COLS = ["market", "model", "seed", *METRIC_COLS, "num_test_days", "pred_path_or_ckpt_path"]
AGG_COLS = []
for m in METRIC_COLS:
    AGG_COLS += [f"{m}_mean", f"{m}_std"]


def _git(args):
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, *args], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def load_staging(out):
    seed_files = sorted(glob.glob(os.path.join(out, "_staging", "*_seed_metrics.csv")))
    ens_files = sorted(glob.glob(os.path.join(out, "_staging", "*_ensemble_metrics.csv")))
    if not seed_files:
        raise SystemExit(
            f"No staging seed files in {os.path.join(out, '_staging')}. "
            "Run eval/build_results.py for each market first."
        )
    seed_df = pd.concat([pd.read_csv(f) for f in seed_files], ignore_index=True)
    ens_df = (
        pd.concat([pd.read_csv(f) for f in ens_files], ignore_index=True)
        if ens_files else pd.DataFrame()
    )
    return seed_df, ens_df


def build_aggregate(seed_df):
    rows = []
    for (market, model), g in seed_df.groupby(["market", "model"], sort=True):
        row = {"market": market, "model": model}
        for m in METRIC_COLS:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{m}_std"] = float(vals.std(ddof=DDOF)) if len(vals) >= 2 else (
                0.0 if len(vals) == 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=["market", "model", *AGG_COLS])


def build_seed_mean_std(agg_df):
    rows = []
    for _, r in agg_df.iterrows():
        row = {"market": r["market"], "model": r["model"]}
        for m in METRIC_COLS:
            mean, std = r[f"{m}_mean"], r[f"{m}_std"]
            if pd.isna(mean):
                row[m] = "NaN"
            else:
                row[m] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows, columns=["market", "model", *METRIC_COLS])


def build_ensemble_table(ens_df):
    if ens_df.empty:
        return ens_df
    out = ens_df.copy()
    for m in METRIC_COLS:
        if m in out.columns:
            out[m] = pd.to_numeric(out[m], errors="coerce").map(
                lambda v: "NaN" if pd.isna(v) else f"{v:.4f}"
            )
    # num_test_days is a count -> display as an integer, not 4-decimal.
    if "num_test_days" in out.columns:
        out["num_test_days"] = pd.to_numeric(out["num_test_days"], errors="coerce").map(
            lambda v: "NaN" if pd.isna(v) else str(int(round(v)))
        )
    return out


def build_eval_config(seed_df, ens_df, tag):
    markets = sorted(seed_df["market"].unique().tolist())
    seeds = sorted(seed_df["seed"].unique().tolist())
    models = sorted(seed_df["model"].unique().tolist())

    segments, per_market = {}, {}
    for mk in markets:
        import yaml
        with open(yaml_path(mk), "r") as f:
            cfg = yaml.safe_load(f)
        seg = cfg["task"]["dataset"]["kwargs"]["segments"]
        segments = seg  # identical across markets (CN/US share split)
        per_market[mk] = {
            "benchmark": cfg.get("benchmark"),
            "provider_uri": cfg["qlib_init"]["provider_uri"],
            "region": cfg["qlib_init"]["region"],
            "instruments": cfg.get("market"),
            "label": cfg["data_handler_config"]["label"],
            "data_start_time": str(cfg["data_handler_config"]["start_time"]),
            "data_end_time": str(cfg["data_handler_config"]["end_time"]),
        }

    ensemble_enabled = (not ens_df.empty) and ens_df["market"].nunique() > 0 and len(seeds) >= 2
    ens_methods = sorted(ens_df["ensemble_method"].unique().tolist()) if ensemble_enabled else []

    return {
        "baseline_id": BASELINE_ID,
        "model_id": models,
        "description": (
            "MATCC (CIKM 2024) cross-asset stock-ranking baseline, reproduced as a paper "
            "baseline. Unified results exported under Baseline Results Protocol v1.0."
        ),
        "markets": markets,
        "seeds": seeds,
        "tag": tag,
        "split": {k: [str(a), str(b)] for k, (a, b) in segments.items()},
        "per_market": per_market,
        "backtest": {
            "engine": "qlib.contrib.backtest (TopkDropoutStrategy + SimulatorExecutor)",
            "strategy": "TopK-DropN",
            "topk": TOPK,
            "n_drop": N_DROP,
            "weight_method": "equal weight across held names (qlib TopkDropout default)",
            "account": ACCOUNT,
            "deal_price": "close",
            "buy_cost": OPEN_COST,
            "sell_cost": CLOSE_COST,
            "note": (
                "Identical strategy to backtest.py. report_normal['return'] is the GROSS "
                "daily portfolio return (qlib adds the day's cost back to the true net "
                "earning, see qlib/backtest/account.py); 'cost' is the transaction-cost rate; "
                "'bench' is the benchmark daily return. daily_ret_net = return - cost, so cost "
                "is deducted EXACTLY once. The original backtest_results/*_summary.csv reports "
                "EXCESS return (return - bench); protocol portfolio metrics here are on the "
                "ABSOLUTE net portfolio return, not excess."
            ),
        },
        "return_semantics": {
            "daily_ret_gross": "qlib report_normal['return'] (gross of cost)",
            "cost": "qlib report_normal['cost']",
            "daily_ret_net": "daily_ret_gross - cost (deducted once)",
            "bench_ret": "qlib report_normal['bench'] (benchmark daily return)",
            "already_net": False,
            "double_deduction_check": "net = gross - cost; cost column >= 0 (asserted at build time)",
        },
        "metric_convention": {
            "annualization": ANNUAL,
            "ddof": DDOF,
            "risk_free_rate": 0,
            "MAR_daily": 0,
            "return_transform": "log1p (g_t = log(1 + r_net_t))",
            "IC": "mean of per-day Pearson(prediction,label); days with undefined corr skipped",
            "ICIR": "mean(IC_t)/std(IC_t, ddof=1), NOT annualized",
            "RankIC": "mean of per-day Spearman(prediction,label)",
            "RankICIR": "mean(RankIC_t)/std(RankIC_t, ddof=1), NOT annualized",
            "AR": "exp(mean(g_t)*252) - 1",
            "STD": "std(g_t, ddof=1)*sqrt(252)",
            "MDD": "min(NAV/cummax(NAV) - 1), NAV = [1.0, exp(cumsum(g_t))]",
            "Sharpe": "sqrt(252)*mean(g_t)/std(g_t, ddof=1)  # absolute net return, NOT benchmark-relative IR",
            "Sortino": "sqrt(252)*mean(g_t)/sqrt(mean(min(g_t,0)^2))  # downside over ALL days",
            "Calmar": "AR/abs(MDD)",
            "undefined_handling": "zero denominator / <2 samples -> NaN (never 0); r_net<=-1 -> hard error",
        },
        "ensemble": {
            "enabled": bool(ensemble_enabled),
            "min_seeds_required": 2,
            "join": "inner join on (datetime, instrument) across selected seeds",
            "methods": ens_methods,
            "default_method": "avg_none",
            "score_formula": {
                "avg_none": "mean of raw scores across seeds",
                "avg_zscore": "per-day cross-sectional z-score (ddof=0) per seed, then mean",
                "avg_rank": "per-day cross-sectional rank percentile per seed, then mean",
            },
            "ranking_metrics_source": "recomputed directly from ensemble score vs aligned test label (NOT seed-mean)",
            "portfolio_metrics_source": "re-run the identical TopK-DropN backtest on the ensemble score",
        },
        "data_version": {
            "cn_provider": "~/.qlib/qlib_data/cn_data",
            "us_provider": "~/.qlib/qlib_data/us_data",
            "data_end_time": per_market.get(markets[0], {}).get("data_end_time")
            if markets else None,
        },
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
    }


def build_manifest(ensemble_enabled):
    files = {
        "seed_metrics": "metrics/seed_metrics.csv",
        "aggregate_metrics": "metrics/aggregate_metrics.csv",
        "seed_table": "tables/seed_mean_std.csv",
        "eval_config": "metadata/eval_config.json",
        "manifest": "metadata/manifest.json",
        "validation": "diagnostics/validation.json",
    }
    primary_keys = {
        "seed_metrics": ["market", "model", "seed"],
        "aggregate_metrics": ["market", "model"],
    }
    if ensemble_enabled:
        files["ensemble_metrics"] = "metrics/ensemble_metrics.csv"
        files["ensemble_table"] = "tables/ensemble.csv"
        files["ensemble_curves"] = "curves/ensemble/*.csv"
        primary_keys["ensemble_metrics"] = ["market", "model", "ensemble_method"]
    return {
        "schema_version": "1.0",
        "baseline": BASELINE_ID,
        "description": (
            "MATCC baseline unified evaluation results (Protocol v1.0). "
            "Markets: CSI300 (cn) + SP500 (us); single model MATCC; 5 seeds; "
            "Qlib TopK-DropN(30,5) backtest with buy 5bps/sell 15bps."
        ),
        "primary_keys": primary_keys,
        "files": files,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--tag", default="2009_2025")
    args = ap.parse_args()
    out = args.out

    for sub in ("metrics", "tables", "metadata", "diagnostics"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    seed_df, ens_df = load_staging(out)

    # Enforce the fixed column order for seed_metrics.csv.
    seed_df = seed_df[SEED_COLS]
    seed_df.to_csv(os.path.join(out, "metrics", "seed_metrics.csv"), index=False)

    agg_df = build_aggregate(seed_df)
    agg_df.to_csv(os.path.join(out, "metrics", "aggregate_metrics.csv"), index=False)

    ensemble_enabled = (not ens_df.empty) and len(seed_df["seed"].unique()) >= 2
    if ensemble_enabled:
        ens_cols = ["market", "model", "ensemble_method", *METRIC_COLS,
                    "num_test_days", "seeds", "pred_paths"]
        ens_df = ens_df[ens_cols]
        ens_df.to_csv(os.path.join(out, "metrics", "ensemble_metrics.csv"), index=False)
        ens_table = build_ensemble_table(ens_df)
        ens_table.to_csv(os.path.join(out, "tables", "ensemble.csv"), index=False)
    else:
        # remove stale ensemble artifacts if present
        for p in (os.path.join(out, "metrics", "ensemble_metrics.csv"),
                  os.path.join(out, "tables", "ensemble.csv")):
            if os.path.exists(p):
                os.remove(p)

    seed_mean_std = build_seed_mean_std(agg_df)
    seed_mean_std.to_csv(os.path.join(out, "tables", "seed_mean_std.csv"), index=False)

    eval_config = build_eval_config(seed_df, ens_df if ensemble_enabled else pd.DataFrame(), args.tag)
    with open(os.path.join(out, "metadata", "eval_config.json"), "w") as f:
        json.dump(eval_config, f, indent=2, default=str)

    manifest = build_manifest(ensemble_enabled)
    with open(os.path.join(out, "metadata", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[finalize] markets={eval_config['markets']} seeds={eval_config['seeds']} "
          f"models={eval_config['model_id']} ensemble={ensemble_enabled}")
    print(f"[finalize] seed_metrics rows={len(seed_df)} aggregate rows={len(agg_df)} "
          f"ensemble rows={len(ens_df) if ensemble_enabled else 0}")
    print(f"[finalize] DONE -> {out}")


if __name__ == "__main__":
    main()
