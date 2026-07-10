"""Compatibility entry point for the leak-free MATCC training pipeline.

The original implementation that lived in this file used an order-dependent daily
sampler and inspected the test split during training.  Keep the historical filename
as a thin alias so old commands cannot accidentally run that unsafe path.
"""

import warnings

from train import main


if __name__ == "__main__":
    warnings.warn(
        "train_model_MATCC.py is deprecated; use train.py with the same arguments.",
        DeprecationWarning,
    )
    main()
