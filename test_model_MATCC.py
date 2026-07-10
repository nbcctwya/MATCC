"""Compatibility entry point for the leak-free MATCC evaluation pipeline.

The original implementation that lived in this file used an order-dependent daily
sampler.  Keeping only this alias prevents the historical filename from silently
reintroducing cross-date attention and lookahead leakage.
"""

import warnings

from test import main


if __name__ == "__main__":
    warnings.warn(
        "test_model_MATCC.py is deprecated; use test.py with the same arguments.",
        DeprecationWarning,
    )
    main()
