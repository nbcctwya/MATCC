"""Frozen Baseline Results Protocol v1.0 backtest parameters."""

TOPK = 30
N_DROP = 5
METHOD_SELL = "bottom"
METHOD_BUY = "top"
HOLD_THRESH = 1
ONLY_TRADABLE = False
FORBID_ALL_TRADE_AT_LIMIT = True
RISK_DEGREE = 0.95
FREQ = "day"
ACCOUNT = 100_000_000
OPEN_COST = 0.0005
CLOSE_COST = 0.0015
MIN_COST = 0
DEAL_PRICE = "close"
EXECUTOR = "SimulatorExecutor"
QLIB_SIGNAL_SHIFT = 1


def strategy_kwargs(signal=None):
    """Return every fixed strategy kwarg explicitly (optionally including signal)."""
    kwargs = {
        "topk": TOPK,
        "n_drop": N_DROP,
        "method_sell": METHOD_SELL,
        "method_buy": METHOD_BUY,
        "hold_thresh": HOLD_THRESH,
        "only_tradable": ONLY_TRADABLE,
        "forbid_all_trade_at_limit": FORBID_ALL_TRADE_AT_LIMIT,
        "risk_degree": RISK_DEGREE,
    }
    if signal is not None:
        kwargs["signal"] = signal
    return kwargs
