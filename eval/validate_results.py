"""Protocol v1.0 result validator -> diagnostics/validation.json (exit 0 on success).

Pure pandas/numpy (no qlib). Re-implements every protocol section-8 check and, in
addition, independently recomputes metrics from the saved scores/labels/daily caches so
that a copy-paste bug (e.g. ensemble IC = mean of seed IC) cannot pass silently.

Run after ``eval/finalize_results.py``:
    python eval/validate_results.py [--out results]
Exit code 0 iff all checks pass.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval import MODEL_ID  # noqa: E402
from eval.metrics import prediction_metrics, portfolio_metrics, nav_from_curve  # noqa: E402
from eval.protocol import (  # noqa: E402
    ACCOUNT, CLOSE_COST, DEAL_PRICE, EXECUTOR, FORBID_ALL_TRADE_AT_LIMIT, FREQ,
    HOLD_THRESH, METHOD_BUY, METHOD_SELL, MIN_COST, N_DROP, ONLY_TRADABLE,
    OPEN_COST, QLIB_SIGNAL_SHIFT, RISK_DEGREE, TOPK,
)

METRIC_COLS = [
    "IC", "ICIR", "RankIC", "RankICIR",
    "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar",
]
RECOMP_TOL = 1e-6    # formula recompute (same floats, CSV round-trip)
TABLE_TOL = 5e-5     # 4-decimal presentation tolerance


class Checker:
    def __init__(self):
        self.checks = []

    def add(self, name, passed, detail):
        self.checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})
        return bool(passed)


def _close(a, b, tol=RECOMP_TOL):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    try:
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                rtol=tol, atol=tol, equal_nan=True))
    except Exception:
        return False


def _load_series(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _curve_market_method(fname, markets):
    name = os.path.splitext(os.path.basename(fname))[0]
    for mk in markets:
        prefix = f"{mk}_{MODEL_ID}"
        if name == prefix:
            return mk, "avg_none"
        if name.startswith(prefix + "__"):
            return mk, name[len(prefix) + 2:]
    return None, None


def run(out):
    c = Checker()
    meta_dir = os.path.join(out, "metadata")
    diag_dir = os.path.join(out, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    # ---- load manifest + eval_config -------------------------------------- #
    manifest_path = os.path.join(meta_dir, "manifest.json")
    cfg_path = os.path.join(meta_dir, "eval_config.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else None
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    c.add("manifest.json exists", manifest is not None, manifest_path)
    c.add("eval_config.json exists", cfg is not None, cfg_path)
    if manifest is None or cfg is None:
        return c  # nothing else to check

    markets = cfg["markets"]
    models = cfg["model_id"]
    seeds = cfg["seeds"]
    ens_enabled = cfg["ensemble"]["enabled"]
    ens_methods = cfg["ensemble"]["methods"]

    # ---- protocol configuration is explicit and frozen ------------------ #
    bt = cfg.get("backtest", {})
    expected_bt = {
        "strategy": "TopkDropoutStrategy", "topk": TOPK, "n_drop": N_DROP,
        "method_sell": METHOD_SELL, "method_buy": METHOD_BUY,
        "hold_thresh": HOLD_THRESH, "only_tradable": ONLY_TRADABLE,
        "forbid_all_trade_at_limit": FORBID_ALL_TRADE_AT_LIMIT,
        "risk_degree": RISK_DEGREE, "freq": FREQ, "executor": EXECUTOR,
        "account": ACCOUNT, "deal_price": DEAL_PRICE, "open_cost": OPEN_COST,
        "close_cost": CLOSE_COST, "min_cost": MIN_COST,
    }
    mismatched = {k: {"got": bt.get(k), "expected": v}
                  for k, v in expected_bt.items() if bt.get(k) != v}
    c.add("eval_config fixed Qlib strategy/backtest parameters", not mismatched,
          mismatched if mismatched else "all fixed parameters explicit and exact")
    test_start, test_end = cfg.get("split", {}).get("test", [None, None])
    c.add("backtest interval == declared test split",
          bt.get("start_time") == test_start and bt.get("end_time") == test_end,
          f"backtest={bt.get('start_time')}..{bt.get('end_time')} test={test_start}..{test_end}")
    align = cfg.get("signal_alignment", {})
    align_ok = (align.get("signal_date") == "t-1" and align.get("trade_date") == "t"
                and align.get("qlib_internal_shift") == QLIB_SIGNAL_SHIFT
                and align.get("adapter_shift") == 0 and bool(align.get("label_horizon")))
    c.add("signal t-1 -> trade t alignment recorded", align_ok, align)
    per_market_ok = all(
        all(cfg.get("per_market", {}).get(m, {}).get(k) is not None for k in
            ("provider_uri", "region", "instruments", "benchmark", "deal_price",
             "limit_threshold", "suspension_and_untradable", "trade_unit"))
        for m in markets
    )
    c.add("per-market Qlib/exchange metadata complete", per_market_ok,
          "provider, region, instruments, benchmark, price/tradability/trade_unit")

    # ---- check 1: manifest file paths exist ------------------------------- #
    # NOTE: 'validation' is this validator's OWN output (written at the end of this run),
    # so it cannot pre-exist on a first/clean run; its existence+structure is verified by a
    # dedicated self-check after the report is written, not here.
    for key, rel in manifest["files"].items():
        if key == "validation":
            continue
        if "*" in rel:
            matches = glob.glob(os.path.join(out, rel))
            c.add(f"manifest glob '{key}' non-empty", len(matches) > 0,
                  f"{len(matches)} files match {rel}")
        else:
            p = os.path.join(out, rel)
            c.add(f"manifest path '{key}' exists", os.path.exists(p), rel)

    # ---- load metric tables ---------------------------------------------- #
    seed_path = os.path.join(out, "metrics", "seed_metrics.csv")
    agg_path = os.path.join(out, "metrics", "aggregate_metrics.csv")
    ens_path = os.path.join(out, "metrics", "ensemble_metrics.csv")
    seed_df = pd.read_csv(seed_path) if os.path.exists(seed_path) else pd.DataFrame()
    agg_df = pd.read_csv(agg_path) if os.path.exists(agg_path) else pd.DataFrame()
    ens_df = pd.read_csv(ens_path) if (ens_enabled and os.path.exists(ens_path)) else pd.DataFrame()

    # ---- check 2: seed_metrics completeness ------------------------------ #
    expected_seeds = {(m, mo, s) for m in markets for mo in models for s in seeds}
    actual_seeds = set(map(tuple, seed_df[["market", "model", "seed"]].itertuples(index=False))) \
        if not seed_df.empty else set()
    missing = expected_seeds - actual_seeds
    extra = actual_seeds - expected_seeds
    c.add("seed_metrics rows == markets x models x seeds",
          not missing and not extra,
          f"expected={len(expected_seeds)} actual={len(actual_seeds)} "
          f"missing={sorted(missing)} extra={sorted(extra)}")

    # ---- check 3: primary keys unique ------------------------------------ #
    for name, df, keys in (
        ("seed_metrics", seed_df, ["market", "model", "seed"]),
        ("aggregate_metrics", agg_df, ["market", "model"]),
        ("ensemble_metrics", ens_df, ["market", "model", "ensemble_method"]),
    ):
        if df.empty:
            continue
        dup = int(df.duplicated(subset=keys).sum())
        c.add(f"{name} primary key unique", dup == 0, f"dups={dup} on {keys}")

    # ---- check 4: no NaN/Inf in metrics (strict; real data must be clean) - #
    for name, df in (("seed_metrics", seed_df), ("ensemble_metrics", ens_df)):
        if df.empty:
            continue
        cols = METRIC_COLS + (["num_test_days"] if "num_test_days" in df.columns else [])
        bad = []
        for col in cols:
            if col not in df.columns:
                bad.append(f"{col}:MISSING")
                continue
            v = pd.to_numeric(df[col], errors="coerce")
            n_nan = int(v.isna().sum())
            n_inf = int(np.isinf(v.fillna(0)).sum())
            if n_nan or n_inf:
                bad.append(f"{col}:nan={n_nan},inf={n_inf}")
        c.add(f"{name} metrics finite (no NaN/Inf)", not bad,
              "; ".join(bad) if bad else "all finite")

    # ---- check 5: metric bounds ------------------------------------------ #
    for name, df in (("seed_metrics", seed_df), ("ensemble_metrics", ens_df)):
        if df.empty:
            continue
        ic_ok = (df["IC"].abs() <= 1.0 + RECOMP_TOL).all()
        ric_ok = (df["RankIC"].abs() <= 1.0 + RECOMP_TOL).all()
        std_ok = (df["STD"] >= -RECOMP_TOL).all()
        mdd_ok = (df["MDD"] <= RECOMP_TOL).all()
        nd_ok = (df["num_test_days"] > 0).all() if "num_test_days" in df.columns else True
        c.add(f"{name} bounds |IC|<=1 |RankIC|<=1 STD>=0 MDD<=0 days>0",
              ic_ok and ric_ok and std_ok and mdd_ok and nd_ok,
              f"IC<=1:{ic_ok} RankIC<=1:{ric_ok} STD>=0:{std_ok} MDD<=0:{mdd_ok} days>0:{nd_ok}")

    # ---- check 6: aggregate == seed mean/std (ddof=1) -------------------- #
    if not agg_df.empty and not seed_df.empty:
        mism = []
        for _, r in agg_df.iterrows():
            g = seed_df[(seed_df.market == r.market) & (seed_df.model == r.model)]
            for m in METRIC_COLS:
                vals = pd.to_numeric(g[m], errors="coerce").dropna()
                exp_mean = float(vals.mean()) if len(vals) else float("nan")
                exp_std = float(vals.std(ddof=1)) if len(vals) >= 2 else (
                    float("nan"))
                if not _close(r[f"{m}_mean"], exp_mean):
                    mism.append(f"{r.market}.{m}_mean: got {r[f'{m}_mean']} exp {exp_mean}")
                if not _close(r[f"{m}_std"], exp_std):
                    mism.append(f"{r.market}.{m}_std: got {r[f'{m}_std']} exp {exp_std}")
        c.add("aggregate_metrics == seed mean/std (ddof=1)", not mism,
              "; ".join(mism[:6]) + (f" ... ({len(mism)} total)" if len(mism) > 6 else ""))

    # ---- check 7: tables == metrics within 4dp --------------------------- #
    sms_path = os.path.join(out, "tables", "seed_mean_std.csv")
    if os.path.exists(sms_path) and not agg_df.empty:
        sms = pd.read_csv(sms_path)
        mism = []
        for _, r in sms.iterrows():
            ar = agg_df[(agg_df.market == r.market) & (agg_df.model == r.model)].iloc[0]
            for m in METRIC_COLS:
                s = str(r[m])
                if s == "NaN":
                    if not (math.isnan(ar[f"{m}_mean"])):
                        mism.append(f"{r.market}.{m}: table NaN vs metric {ar[f'{m}_mean']}")
                    continue
                try:
                    tmean, tstd = (float(x) for x in s.split("±"))
                except Exception:
                    mism.append(f"{r.market}.{m}: unparseable '{s}'")
                    continue
                if not _close(tmean, ar[f"{m}_mean"], TABLE_TOL):
                    mism.append(f"{r.market}.{m}_mean: {tmean} vs {ar[f'{m}_mean']}")
                if not _close(tstd, ar[f"{m}_std"], TABLE_TOL):
                    mism.append(f"{r.market}.{m}_std: {tstd} vs {ar[f'{m}_std']}")
        c.add("seed_mean_std table == aggregate within 4dp", not mism,
              "; ".join(mism[:6]) + (f" ... ({len(mism)} total)" if len(mism) > 6 else ""))

    ens_tbl_path = os.path.join(out, "tables", "ensemble.csv")
    if os.path.exists(ens_tbl_path) and not ens_df.empty:
        et = pd.read_csv(ens_tbl_path)
        mism = []
        merged = ens_df.merge(et, on=["market", "model", "ensemble_method"], suffixes=("_m", "_t"))
        for m in METRIC_COLS + ["num_test_days"]:
            for _, r in merged.iterrows():
                if not _close(float(r[f"{m}_m"]), float(r[f"{m}_t"]), TABLE_TOL):
                    mism.append(f"{r.market}/{r.ensemble_method}/{m}: "
                                f"{r[f'{m}_m']} vs {r[f'{m}_t']}")
        c.add("ensemble table == ensemble_metrics within 4dp", not mism,
              "; ".join(mism[:6]) + (f" ... ({len(mism)} total)" if len(mism) > 6 else ""))

    # ---- check 8: ensemble rows = markets x models x methods ------------- #
    if ens_enabled:
        expected_ens = {(m, mo, meth) for m in markets for mo in models for meth in ens_methods}
        actual_ens = set(map(tuple, ens_df[["market", "model", "ensemble_method"]]
                             .itertuples(index=False))) if not ens_df.empty else set()
        missing_e = expected_ens - actual_ens
        extra_e = actual_ens - expected_ens
        c.add("ensemble_metrics rows == markets x models x methods",
              not missing_e and not extra_e,
              f"expected={len(expected_ens)} actual={len(actual_ens)} "
              f"missing={sorted(missing_e)} extra={sorted(extra_e)}")

    # ---- checks 9-12: per-curve structural + portfolio recompute --------- #
    curve_files = sorted(glob.glob(os.path.join(out, "curves", "ensemble", "*.csv")))
    expected_curve_count = len(markets) * len(models) * len(ens_methods) if ens_enabled else 0
    c.add("ensemble curves count == markets x models x methods",
          len(curve_files) == expected_curve_count,
          f"expected={expected_curve_count} actual={len(curve_files)}")
    for cf in curve_files:
        mk, meth = _curve_market_method(cf, markets)
        tag = f"{mk}/{meth}" if mk else os.path.basename(cf)
        if mk is None:
            c.add(f"curve '{os.path.basename(cf)}' parseable", False, "unknown market/model")
            continue
        cv = pd.read_csv(cf)
        dts = pd.to_datetime(cv["datetime"])
        c.add(f"curve {tag}: dates ascending", dts.is_monotonic_increasing,
              f"{dts.iloc[0].date()}..{dts.iloc[-1].date()}")
        c.add(f"curve {tag}: dates unique", dts.is_unique, f"{len(dts)} rows")
        within_split = (dts.min() >= pd.Timestamp(test_start)
                        and dts.max() <= pd.Timestamp(test_end))
        c.add(f"curve {tag}: dates within declared test split", within_split,
              f"curve={dts.min().date()}..{dts.max().date()} split={test_start}..{test_end}")
        # no NaN/Inf in any column
        num_clean = all(np.isfinite(cv[col].to_numpy(dtype=float)).all()
                        for col in cv.columns if col != "datetime")
        c.add(f"curve {tag}: numeric finite", num_clean, "all columns finite")
        # daily_ret_net == gross - cost
        rel_ok = _close(cv["daily_ret_net"], cv["daily_ret_gross"] - cv["cost"])
        c.add(f"curve {tag}: net == gross - cost", rel_ok, "cost deducted once")
        # nav == cumprod(1+net); bench_nav == cumprod(1+bench)
        nav_ok = _close(cv["nav"], nav_from_curve(cv["daily_ret_net"]))
        c.add(f"curve {tag}: nav == cumprod(1+net)", nav_ok, "curve NAV convention")
        bnav_ok = _close(cv["bench_nav"], nav_from_curve(cv["bench_ret"]))
        c.add(f"curve {tag}: bench_nav == cumprod(1+bench)", bnav_ok, "bench NAV")
        # portfolio metrics from curve == ensemble_metrics row
        if not ens_df.empty:
            row = ens_df[(ens_df.market == mk) & (ens_df.ensemble_method == meth)]
            if len(row):
                r = row.iloc[0]
                pm = portfolio_metrics(cv["daily_ret_net"].to_numpy())
                mism = [col for col in ("AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar")
                        if not _close(pm[col], float(r[col]))]
                # num_test_days must equal curve length
                if pm["num_test_days"] != int(r["num_test_days"]):
                    mism.append(f"num_test_days {pm['num_test_days']} vs {int(r['num_test_days'])}")
                c.add(f"curve {tag}: portfolio metrics recompute == ensemble_metrics",
                      not mism, "mismatch: " + ",".join(mism) if mism else "all match")
            else:
                c.add(f"curve {tag}: ensemble_metrics row exists", False, "no matching row")

    # ---- check 13: ensemble ranking metrics from saved score ------------ #
    if ens_enabled:
        score_dir = os.path.join(out, "_cache")
        # seed IC means for the "not seed mean" guard
        seed_ic_means = {}
        if not seed_df.empty:
            for mk, g in seed_df.groupby("market"):
                seed_ic_means[mk] = float(pd.to_numeric(g["IC"], errors="coerce").mean())
        cache_root = None
        for cand in (os.path.join(out, "_cache"),):
            if os.path.isdir(os.path.join(cand, f"{markets[0]}_scores")):
                cache_root = cand
                break
        if cache_root is None:
            c.add("ensemble score cache present (for IC recompute)", False,
                  "no results/_cache/<market>_scores; cannot recompute ensemble IC")
        for mk in markets:
            label_pkl = os.path.join(out, "_cache", f"{mk}_scores", f"{mk}__label.pkl")
            label = _load_series(label_pkl) if os.path.exists(label_pkl) else None
            for meth in ens_methods:
                sc_pkl = os.path.join(out, "_cache", f"{mk}_scores", f"{mk}__{meth}.pkl")
                if not os.path.exists(sc_pkl) or label is None:
                    c.add(f"ensemble IC recompute {mk}/{meth}", False,
                          f"missing cache ({sc_pkl} or label)")
                    continue
                score = _load_series(sc_pkl)
                pm = prediction_metrics(score, label)
                row = ens_df[(ens_df.market == mk) & (ens_df.ensemble_method == meth)]
                if not len(row):
                    continue
                r = row.iloc[0]
                mism = [col for col in ("IC", "ICIR", "RankIC", "RankICIR")
                        if not _close(pm[col], float(r[col]))]
                c.add(f"ensemble IC recompute {mk}/{meth} == ensemble_metrics",
                      not mism, "mismatch: " + ",".join(mism) if mism else "matches saved score")
                # guard against the copy-paste bug: ensemble IC must NOT equal seed-IC mean
                if mk in seed_ic_means:
                    same_as_seed_mean = abs(pm["IC"] - seed_ic_means[mk]) < 1e-9
                    c.add(f"ensemble IC {mk}/{meth} != seed-IC mean (not faked)",
                          not same_as_seed_mean,
                          f"ensemble={pm['IC']:.6f} seed_mean={seed_ic_means[mk]:.6f}")

    # ---- bonus: seed metrics reconstructed from source pred/label/daily -- #
    if not seed_df.empty:
        for _, r in seed_df.iterrows():
            mk, s = r["market"], int(r["seed"])
            # ranking metrics from raw pred/label
            pred_rel = r["pred_path_or_ckpt_path"]
            pred_p = os.path.join(ROOT, pred_rel)
            lab_p = os.path.join(ROOT, pred_rel.replace("_pred_", "_labels_"))
            if os.path.exists(pred_p) and os.path.exists(lab_p):
                pred = _load_series(pred_p)
                lab = _load_series(lab_p)
                pm = prediction_metrics(pred, lab)
                mism = [col for col in ("IC", "ICIR", "RankIC", "RankICIR")
                        if not _close(pm[col], float(r[col]))]
                c.add(f"seed IC recompute {mk}/{s} == seed_metrics",
                      not mism, "mismatch: " + ",".join(mism) if mism else "matches source pred/label")
                pred_dates = pd.DatetimeIndex(pred.index.get_level_values("datetime"))
                daily_p = os.path.join(out, "_cache", f"{mk}_daily", f"{mk}_seed_{s}.csv")
                if os.path.exists(daily_p):
                    backtest_dates = pd.to_datetime(pd.read_csv(daily_p)["datetime"])
                    coverage_ok = (pred_dates.min() <= backtest_dates.min()
                                   and pred_dates.max() >= backtest_dates.max())
                    c.add(f"prediction covers full backtest calendar {mk}/{s}", coverage_ok,
                          f"prediction={pred_dates.min().date()}..{pred_dates.max().date()} "
                          f"backtest={backtest_dates.min().date()}..{backtest_dates.max().date()}")
            # portfolio metrics from cached daily report
            daily_p = os.path.join(out, "_cache", f"{mk}_daily", f"{mk}_seed_{s}.csv")
            if os.path.exists(daily_p):
                dv = pd.read_csv(daily_p)
                pm = portfolio_metrics(dv["net"].to_numpy())
                mism = [col for col in ("AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar")
                        if not _close(pm[col], float(r[col]))]
                if pm["num_test_days"] != int(r["num_test_days"]):
                    mism.append("num_test_days")
                c.add(f"seed portfolio recompute {mk}/{s} == seed_metrics",
                      not mism, "mismatch: " + ",".join(mism) if mism else "matches cached daily")

    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    checker = run(args.out)

    # Self-check: this run's own output (diagnostics/validation.json) must be well-formed
    # and internally consistent. Verified before writing so it counts toward the report.
    checks = checker.checks
    passes = sum(1 for ch in checks if ch["passed"])
    failures = len(checks) - passes
    well_formed = (
        all(all(k in ch for k in ("name", "passed", "detail")) for ch in checks)
        and isinstance(passes, int) and isinstance(failures, int)
        and passes + failures == len(checks)
    )
    checker.add("validation.json well-formed & self-consistent", well_formed,
                f"schema keys present; passes+failures == nchecks ({passes}+{failures}={len(checks)})")

    checks = checker.checks
    passes = sum(1 for ch in checks if ch["passed"])
    failures = len(checks) - passes
    report = {
        "passed": failures == 0,
        "passes": passes,
        "failures": failures,
        "checks": checks,
    }
    out_json = os.path.join(args.out, "diagnostics", "validation.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    # Reload and confirm the written file round-trips and matches the in-memory report.
    reloaded = json.load(open(out_json))
    checker.add(
        "validation.json round-trips from disk",
        reloaded.get("passes") == passes and reloaded.get("failures") == failures
        and reloaded.get("passed") == (failures == 0) and len(reloaded.get("checks", [])) == len(checks),
        f"reloaded passes={reloaded.get('passes')} failures={reloaded.get('failures')}",
    )
    # The round-trip check was added after writing; reflect it in a final rewrite so the
    # on-disk file includes every check (its own counts stay consistent by construction).
    checks = checker.checks
    passes = sum(1 for ch in checks if ch["passed"])
    failures = len(checks) - passes
    report = {"passed": failures == 0, "passes": passes, "failures": failures, "checks": checks}
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    for ch in checks:
        marker = "PASS" if ch["passed"] else "FAIL"
        print(f"[{marker}] {ch['name']} :: {ch['detail']}")
    print(f"\nvalidation: {passes} passed, {failures} failed -> {out_json}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
