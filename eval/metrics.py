"""Single source of truth for the Baseline Results Protocol v1.0 metric convention.

Pure functions only (numpy / pandas). No qlib / torch dependency, so the exact same
formulas are used for generation, validation and unit tests.

Convention (frozen across all baselines for fair comparison):
    ANNUAL = 252   ddof = 1   risk-free rate = 0   MAR_daily = 0

Prediction metrics (per-trading-day cross-section):
    IC_t     = Pearson(prediction, label)
    RankIC_t = Spearman(prediction, label)
    Days whose correlation is mathematically undefined (<2 valid samples, or zero
    variance in either vector) are skipped; IC is NOT imputed for them.
    IC       = mean(IC_t)
    ICIR     = mean(IC_t) / std(IC_t, ddof=1)          # NOT annualized (no sqrt(252))
    RankIC   = mean(RankIC_t)
    RankICIR = mean(RankIC_t) / std(RankIC_t, ddof=1)  # NOT annualized

Portfolio metrics -- input is the daily NET simple return
    r_net_t = daily_return_gross_t - cost_t
where the cost is deducted EXACTLY once. ``r_net_t <= -1`` is a hard error (it would
make log1p produce -inf/NaN silently). Undefined inputs (zero denominator, etc.) yield
NaN, never 0.

    g_t     = log(1 + r_net_t)
    AR      = exp(mean(g_t) * 252) - 1
    STD     = std(g_t, ddof=1) * sqrt(252)
    NAV     = [1.0, exp(cumsum(g_t))]                  # leading 1.0 -> first-day loss counts
    MDD     = min(NAV / cummax(NAV) - 1)               # <= 0
    Sharpe  = sqrt(252) * mean(g_t) / std(g_t, ddof=1)
    DD      = sqrt(mean(min(g_t - MAR_daily, 0)**2))   # downside deviation, over ALL days
    Sortino = sqrt(252) * mean(g_t - MAR_daily) / DD
    Calmar  = AR / abs(MDD)
    num_test_days = count(g_t)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import ANNUAL, DDOF

# Columns of seed_metrics.csv / ensemble_metrics.csv in fixed order (metric subset).
METRIC_COLUMNS = [
    "IC", "ICIR", "RankIC", "RankICIR",
    "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar",
]


class InvalidReturnError(ValueError):
    """Raised when a daily net return <= -1 is encountered (log1p would be -inf/NaN)."""


# --------------------------------------------------------------------------- #
# Prediction (ranking) metrics
# --------------------------------------------------------------------------- #
def prediction_metrics(pred: pd.Series, label: pd.Series) -> dict:
    """Cross-sectional IC / RankIC / ICIR / RankICIR.

    ``pred`` and ``label`` are pd.Series with a MultiIndex whose first level is named
    ``"datetime"`` (the trading day) and second level is the instrument id. They are
    aligned on their shared index first; rows missing either value are dropped.

    Returns a dict with keys IC, ICIR, RankIC, RankICIR and ``num_ic_days``. Any metric
    whose denominator is undefined (fewer than 2 usable days, or zero std) is NaN.
    """
    if pred is None or label is None or len(pred) == 0 or len(label) == 0:
        return {k: float("nan") for k in ("IC", "ICIR", "RankIC", "RankICIR")} | {
            "num_ic_days": 0
        }

    df = pd.DataFrame({"pred": pred, "label": label}).dropna(subset=["pred", "label"])
    if df.empty or "datetime" not in (df.index.names or []):
        return {k: float("nan") for k in ("IC", "ICIR", "RankIC", "RankICIR")} | {
            "num_ic_days": 0
        }

    grp = df.groupby(level="datetime")
    # pandas .corr returns NaN for <2 rows or zero variance -> those days are skipped.
    # Suppress the harmless RuntimeWarning/ConstantInputWarning those degenerate days emit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ic_t = grp.apply(lambda g: g["pred"].corr(g["label"])).dropna()
        ric_t = grp.apply(lambda g: g["pred"].corr(g["label"], method="spearman")).dropna()

    def _ratio(mean_vals: pd.Series) -> float:
        if len(mean_vals) < 2:
            return float("nan")
        s = mean_vals.std(ddof=DDOF)
        if s == 0 or not np.isfinite(s):
            return float("nan")
        return float(mean_vals.mean() / s)

    return {
        "IC": float(ic_t.mean()) if len(ic_t) else float("nan"),
        "ICIR": _ratio(ic_t),
        "RankIC": float(ric_t.mean()) if len(ric_t) else float("nan"),
        "RankICIR": _ratio(ric_t),
        "num_ic_days": int(len(ic_t)),
    }


# --------------------------------------------------------------------------- #
# Portfolio metrics
# --------------------------------------------------------------------------- #
def _to_log_returns(r_net: np.ndarray) -> np.ndarray:
    """Validate net returns (> -1) and convert to log returns g_t = log1p(r_net)."""
    r = np.asarray(r_net, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"daily returns must be 1-D, got shape {r.shape}")
    if np.isnan(r).any():
        # NaN days are not allowed: they would silently propagate through cumsum/log.
        raise InvalidReturnError("daily net return contains NaN; clean the series first")
    if (r <= -1.0).any():
        worst = float(r.min())
        raise InvalidReturnError(
            f"daily net return <= -1.0 encountered (worst={worst:.6f}); "
            "log1p would produce -inf/NaN. Check cost semantics (double deduction?)."
        )
    return np.log1p(r)


def portfolio_metrics(daily_ret_net) -> dict:
    """AR / STD / MDD / Sharpe / Sortino / Calmar / num_test_days from net daily returns.

    ``daily_ret_net`` is a 1-D array / Series of daily NET simple returns (gross - cost).
    Raises :class:`InvalidReturnError` if any value <= -1. Zero denominators -> NaN.
    """
    g = _to_log_returns(np.asarray(daily_ret_net, dtype=np.float64))
    n = g.size
    if n == 0:
        out = {k: float("nan") for k in ("AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar")}
        out["num_test_days"] = 0
        return out

    mean_g = float(g.mean())
    ar = float(np.expm1(mean_g * ANNUAL))

    if n >= 2:
        std_g = float(g.std(ddof=DDOF))
    else:
        std_g = float("nan")
    std_ann = std_g * np.sqrt(ANNUAL) if np.isfinite(std_g) else float("nan")

    # Drawdown on the log NAV with a leading 1.0 so a first-day loss is captured.
    nav = np.concatenate([[1.0], np.exp(np.cumsum(g))])
    running_max = np.maximum.accumulate(nav)
    mdd = float(np.min(nav / running_max - 1.0))  # <= 0

    sharpe = (
        float(np.sqrt(ANNUAL) * mean_g / std_g)
        if (np.isfinite(std_g) and std_g != 0)
        else float("nan")
    )

    # Downside deviation over ALL days (non-negative days contribute 0).
    downside = np.minimum(g, 0.0)  # g - MAR_daily with MAR_daily = 0
    dd = float(np.sqrt(np.mean(downside ** 2)))
    sortino = (
        float(np.sqrt(ANNUAL) * mean_g / dd) if dd != 0 else float("nan")
    )

    calmar = float(ar / abs(mdd)) if mdd != 0 else float("nan")

    return {
        "AR": ar,
        "STD": float(std_ann),
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "num_test_days": int(n),
    }


def nav_from_curve(daily_ret_net) -> np.ndarray:
    """Curve NAV convention: cumprod(1 + daily_ret_net), NOT renormalized to 1.0 on day 1.

    The first entry is therefore ``1 + r_0`` (the protocol forbids dividing by the first
    row, which would erase the first day's return).
    """
    r = np.asarray(daily_ret_net, dtype=np.float64)
    return np.cumprod(1.0 + r)
