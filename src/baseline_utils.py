"""
Shared helpers for the MATCC baseline reproduction pipeline
(train.py / test.py / backtest.py / scripts/prepare_data.py).

Centralises:
  * reproducible RNG seeding (random / numpy / torch / cuda / cudnn + DataLoader workers)
  * the per-day batch sampler and DataLoader factory used by train/test
  * MSE+NaN-mask loss and IC/RankIC computation
  * a single, tag-aware path scheme so smoke runs (tag=smoke) never clobber real runs
    (tag=2009_2025)
  * region-specific backtest parameters for CSI300 (CN) and SP500 (US)
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler


# Project layout: this file lives in <root>/src/baseline_utils.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTIL_DIR = os.path.join(ROOT, "util")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed):
    """Seed every RNG source so each of the 5 seeds (0-4) is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Seed each DataLoader worker process deterministically."""
    base_seed = torch.initial_seed() % 2 ** 32
    np.random.seed((base_seed + worker_id) % 2 ** 32)
    random.seed(base_seed + worker_id)


def get_device(gpu=0):
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


def limit_threads(cpu_num=4):
    """Match the original train script's thread pinning."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(cpu_num)
    torch.set_num_threads(cpu_num)


# --------------------------------------------------------------------------- #
# Data loading (one trading day = one batch of all stocks)
# --------------------------------------------------------------------------- #
class DailyBatchSamplerRandom(Sampler):
    """Yield, per iteration, the row POSITIONS of one trading day's stocks.

    Order-agnostic: groups the dataset index by its 'datetime' level and yields the
    actual positions of that day's stocks. The original MATCC/MASTER sampler assumed
    the rows were date-contiguous and sliced with cumsum ranges -- but the prepared
    data is instrument-sorted, so those ranges bled across many dates of ONE stock,
    turning each "daily" batch into ~1 year of a single stock and causing SAttention
    to attend across future dates (lookahead leak). Grouping by datetime explicitly
    fixes this regardless of the underlying row order.
    """

    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        index = data_source.get_index()
        positions = pd.Series(np.arange(len(index)), index=index)
        # one position-array per datetime, in chronological order
        self.daily_groups = (
            positions.groupby(level="datetime").apply(lambda g: g.to_numpy()).to_list()
        )

    def __iter__(self):
        order = np.arange(len(self.daily_groups))
        if self.shuffle:
            np.random.shuffle(order)
        for i in order:
            yield self.daily_groups[i]

    def __len__(self):
        return len(self.daily_groups)


def cszscore(label):
    """Cross-sectional z-score of a 1-D label tensor (NaNs ignored, left as NaN)."""
    label = label.clone().float()
    m = ~torch.isnan(label)
    if m.sum() > 1:
        vals = label[m]
        label[m] = (vals - vals.mean()) / (vals.std() + 1e-12)
    return label


def drop_extreme(label, pct=0.025):
    """Boolean mask dropping NaNs AND the top/bottom `pct` of labels.

    Mirrors MASTER's drop_extreme: applied per cross-section (per daily batch) during
    TRAINING only -- never on validation/test, so no test/future label info is used.
    """
    mask = ~torch.isnan(label)
    valid_idx = torch.where(mask)[0]
    n = valid_idx.numel()
    k = int(n * pct)
    if k > 0:
        vals = label[valid_idx]
        sorted_local = vals.argsort()
        drop_local = torch.cat([sorted_local[:k], sorted_local[n - k:]])
        mask[valid_idx[drop_local]] = False
    return mask


def make_loader(data, shuffle, drop_last, num_workers=0, pin_memory=False, worker_init_fn=None):
    sampler = DailyBatchSamplerRandom(data, shuffle)
    return DataLoader(
        data,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )


# --------------------------------------------------------------------------- #
# Loss & metrics
# --------------------------------------------------------------------------- #
def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    loss = (pred[mask] - label[mask]) ** 2
    return torch.mean(loss)


def calc_ic(pred, label):
    df = pd.DataFrame({"pred": pred, "label": label})
    ic = df["pred"].corr(df["label"])
    ric = df["pred"].corr(df["label"], method="spearman")
    return ic, ric


# --------------------------------------------------------------------------- #
# Paths (tag-aware: real -> 2009_2025, smoke -> smoke)
# --------------------------------------------------------------------------- #
def dataset_path(universe, tag, split):
    return os.path.join(ROOT, "dataset", universe, f"{universe}_dl_{split}_{tag}.pkl")


def model_dir(universe, tag):
    return os.path.join(ROOT, "model_params", universe, tag)


def model_path(universe, tag, seed):
    return os.path.join(model_dir(universe, tag), f"TEST_MATCC_{universe}_seed_{seed}.pth")


def model_epoch_path(universe, tag, seed, step):
    return os.path.join(model_dir(universe, tag), f"MATCC_{universe}_model_params_epoch_{step}_seed_{seed}.pth")


def last_ckpt_path(universe, tag, seed):
    """Per-epoch resume checkpoint, including best validation RankIC and RNG state."""
    return os.path.join(model_dir(universe, tag), f"last_MATCC_{universe}_seed_{seed}.pth")


def best_path(universe, tag, seed):
    """Maximum-validation-RankIC model state (not the final deliverable path)."""
    return os.path.join(model_dir(universe, tag), f"best_rankic_MATCC_{universe}_seed_{seed}.pth")


def pred_path(universe, tag, seed):
    return os.path.join(ROOT, "label_pred", universe, tag, f"{universe}_pred_{seed}.pkl")


def labels_path(universe, tag, seed):
    return os.path.join(ROOT, "label_pred", universe, tag, f"{universe}_labels_{seed}.pkl")


def metrics_path(universe, tag, seed):
    return os.path.join(ROOT, "metrics", universe, tag, f"MATCC_{universe}_seed_{seed}_test_result.txt")


def summary_path(universe, tag):
    return os.path.join(ROOT, "backtest_results", f"{universe}_{tag}_summary.csv")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def yaml_path(universe, smoke=False):
    name = f"{universe}_smoke.yaml" if smoke else f"{universe}.yaml"
    return os.path.join(UTIL_DIR, name)


# --------------------------------------------------------------------------- #
# Region-specific backtest parameters
# --------------------------------------------------------------------------- #
# trade_unit: None lets qlib pick from the region's default (100 for CN, 1 for US).
REGION = {
    "csi300": dict(benchmark="SH000300", codes="csi300", limit_threshold=0.095,
                   min_cost=5, trade_unit=100),
    "sp500": dict(benchmark="^gspc", codes="sp500", limit_threshold=None,
                  min_cost=0, trade_unit=None),
}
