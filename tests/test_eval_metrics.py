"""Unit tests for eval/metrics.py: protocol v1.0 convention + boundary conditions.

Pure (no qlib); runs under the matcc env. Covers the three mandatory boundary tests:
  1. a first-day -10% loss yields MDD == -0.10 (the leading 1.0 in NAV matters),
  2. two identical negative days do NOT blow up Sortino via a zero downside std,
  3. any daily net return <= -1 raises (no silent log1p NaN).
plus independent re-computation of AR/STD/Sharpe and IC semantics.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import ANNUAL  # noqa: E402
from eval.metrics import (  # noqa: E402
    InvalidReturnError,
    nav_from_curve,
    portfolio_metrics,
    prediction_metrics,
)
from eval.protocol import strategy_kwargs  # noqa: E402


def test_protocol_strategy_kwargs_are_all_explicit():
    assert strategy_kwargs() == {
        "topk": 30, "n_drop": 5, "method_sell": "bottom", "method_buy": "top",
        "hold_thresh": 1, "only_tradable": False,
        "forbid_all_trade_at_limit": True, "risk_degree": 0.95,
    }


# --------------------------------------------------------------------------- #
# Portfolio metric convention + boundaries
# --------------------------------------------------------------------------- #
def test_first_day_loss_minus_10pct_is_mdd_minus_10pct():
    # First day -10%, then flat. The drawdown must reach exactly -0.10 because the
    # metric NAV is [1.0, exp(cumsum(g))] (leading 1.0), not renormalized on day 1.
    r = np.array([-0.10, 0.0, 0.0, 0.0, 0.0])
    m = portfolio_metrics(r)
    assert math.isclose(m["MDD"], -0.10, abs_tol=1e-12), m["MDD"]


def test_two_identical_negative_days_sortino_defined():
    # Both days -5%. Downside deviation uses mean(min(g,0)^2) over all days, so even
    # though the std of the (identical) negative returns is 0, the Sortino denom is not.
    r = np.array([-0.05, -0.05])
    m = portfolio_metrics(r)
    g = np.log1p(r)
    expected_dd = math.sqrt(np.mean(np.minimum(g, 0.0) ** 2))
    assert expected_dd > 0
    assert not math.isnan(m["Sortino"]), m
    assert math.isclose(m["Sortino"], math.sqrt(ANNUAL) * g.mean() / expected_dd, rel_tol=1e-12)


@pytest.mark.parametrize("bad", [-1.0, -1.5, -2.0])
def test_return_le_minus_one_raises(bad):
    with pytest.raises(InvalidReturnError):
        portfolio_metrics(np.array([0.01, bad, 0.02]))


def test_all_zero_returns():
    m = portfolio_metrics(np.zeros(10))
    assert math.isclose(m["AR"], 0.0, abs_tol=1e-15)
    assert m["MDD"] == 0.0
    assert math.isnan(m["Sharpe"])   # std == 0 -> undefined
    assert math.isnan(m["Sortino"])  # no downside -> undefined
    assert math.isnan(m["Calmar"])   # MDD == 0 -> undefined
    assert m["num_test_days"] == 10


def test_independent_recompute_ar_std_sharpe():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.012, size=200)
    m = portfolio_metrics(r)
    g = np.log1p(r)
    assert math.isclose(m["AR"], math.expm1(g.mean() * ANNUAL), rel_tol=1e-12)
    assert math.isclose(m["STD"], g.std(ddof=1) * math.sqrt(ANNUAL), rel_tol=1e-12)
    assert math.isclose(m["Sharpe"], math.sqrt(ANNUAL) * g.mean() / g.std(ddof=1), rel_tol=1e-12)
    # bounds
    assert m["MDD"] <= 1e-12
    assert m["STD"] >= 0


def test_nav_from_curve_not_renormalized():
    r = np.array([0.10, -0.05])
    nav = nav_from_curve(r)
    # first entry is 1 + r0 = 1.10 (NOT 1.0); then * 0.95
    assert math.isclose(nav[0], 1.10, rel_tol=1e-15)
    assert math.isclose(nav[1], 1.10 * 0.95, rel_tol=1e-15)


# --------------------------------------------------------------------------- #
# Prediction (ranking) metric semantics
# --------------------------------------------------------------------------- #
def _series(scores, day):
    idx = pd.MultiIndex.from_product([[pd.Timestamp(day)], list(range(len(scores)))],
                                     names=["datetime", "instrument"])
    return pd.Series(scores, index=idx, name="score")


def test_prediction_metrics_perfect_and_bounds():
    # day 1: pred == label -> Pearson & Spearman == 1
    p1 = _series([0.1, 0.2, 0.3, 0.4], "2024-01-02")
    l1 = _series([0.1, 0.2, 0.3, 0.4], "2024-01-02")
    # day 2: pred == -label -> correlation == -1
    p2 = _series([0.4, 0.3, 0.2, 0.1], "2024-01-03")
    l2 = _series([0.1, 0.2, 0.3, 0.4], "2024-01-03")
    pred = pd.concat([p1, p2])
    label = pd.concat([l1, l2])
    m = prediction_metrics(pred, label)
    assert math.isclose(m["IC"], 0.0, abs_tol=1e-12)        # mean of (+1, -1)
    assert math.isclose(m["RankIC"], 0.0, abs_tol=1e-12)
    assert m["num_ic_days"] == 2
    # IC/ICIR/RankIC are within valid bounds for any input
    assert -1.0 - 1e-9 <= m["IC"] <= 1.0 + 1e-9
    assert -1.0 - 1e-9 <= m["RankIC"] <= 1.0 + 1e-9


def test_prediction_metrics_skips_undefined_days():
    # a day with a single sample and a constant-vector day must be skipped, not imputed
    p_ok = _series([0.1, 0.9], "2024-01-02")
    l_ok = _series([0.2, 0.8], "2024-01-02")
    p_single = _series([0.5], "2024-01-03")
    l_single = _series([0.5], "2024-01-03")
    p_const = _series([0.5, 0.5], "2024-01-04")
    l_const = _series([0.1, 0.9], "2024-01-04")
    pred = pd.concat([p_ok, p_single, p_const])
    label = pd.concat([l_ok, l_single, l_const])
    m = prediction_metrics(pred, label)
    assert m["num_ic_days"] == 1   # only the first day is computable
    assert math.isclose(m["IC"], 1.0, abs_tol=1e-12)
    assert math.isnan(m["ICIR"])   # only 1 day -> std undefined
