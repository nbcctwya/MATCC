"""Unified Baseline Results Protocol v1.0 evaluation/export layer for MATCC.

This package adapts the project's existing predictions + Qlib backtest artifacts to the
cross-baseline ``results/`` schema. It does NOT retrain models or change the backtest
strategy; it only (re)computes metrics under one frozen convention and exports them.
"""

# Convention constants shared across modules.
BASELINE_ID = "matcc"
MODEL_ID = "MATCC"
ANNUAL = 252  # trading days per year
DDOF = 1      # sample std everywhere
