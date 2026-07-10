"""Regression tests for the leak fixes.

Run with:  python tests/test_sampler.py   (or pytest tests/)
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.baseline_utils import DailyBatchSamplerRandom, cszscore, drop_extreme


class _FakeDS:
    """Stand-in for a qlib TSDataSampler: only get_index() is needed by the sampler."""

    def __init__(self, index):
        self._index = index

    def get_index(self):
        return self._index


def _instrument_sorted_index():
    # The dangerous ordering: each instrument's dates are contiguous (as the prepared
    # MATCC data actually is). The old sampler bled across dates here.
    rows = [(d, inst) for inst in ["A", "B", "C"]
            for d in ["2023-01-03", "2023-01-04", "2023-01-05"]]
    return pd.MultiIndex.from_tuples(rows, names=["datetime", "instrument"])


def test_daily_batch_is_one_day_many_stocks():
    ds = _FakeDS(_instrument_sorted_index())
    sampler = DailyBatchSamplerRandom(ds, shuffle=False)
    groups = list(sampler)
    assert len(groups) == 3, f"expected 3 daily batches, got {len(groups)}"
    idx = ds.get_index()
    for g in groups:
        g = np.asarray(g)
        dates = idx.get_level_values("datetime")[g]
        insts = idx.get_level_values("instrument")[g]
        assert dates.nunique() == 1, "a daily batch spans multiple dates -> lookahead leak!"
        assert insts.nunique() == len(g), "batch is not cross-stock (all same instrument)"
    print("test_daily_batch_is_one_day_many_stocks: PASS")


def test_drop_extreme_and_cszscore():
    label = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, float("nan")])
    keep = drop_extreme(label, pct=0.2)            # drop ~1 from each end of the 5 valid
    assert keep.sum().item() == 3, f"expected 3 kept, got {keep.sum().item()}"
    assert torch.isnan(label[keep]).sum().item() == 0
    z = cszscore(torch.tensor([1.0, 2.0, 3.0]))
    assert abs(z.mean().item()) < 1e-5
    assert abs(z.std().item() - 1.0) < 1e-3
    print("test_drop_extreme_and_cszscore: PASS")


if __name__ == "__main__":
    test_daily_batch_is_one_day_many_stocks()
    test_drop_extreme_and_cszscore()
    print("all tests passed")
