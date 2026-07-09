"""
Test MATCC for one (universe, seed): load the trained checkpoint, predict on the test
split, and write per-seed prediction/label pickles + IC metrics.

Usage:
    conda run -n matcc python test.py --universe csi300 --seed 0 --tag 2009_2025
    conda run -n matcc python test.py --universe csi300 --seed 0 --smoke

Outputs (via the tag-aware path scheme):
    label_pred/{universe}/{tag}/{universe}_pred_{seed}.pkl    pd.Series name="score"
    label_pred/{universe}/{tag}/{universe}_labels_{seed}.pkl  pd.Series name="label"
    metrics/{universe}/{tag}/MATCC_{universe}_seed_{seed}_test_result.txt
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from src.MATCC import MATCC
from src.baseline_utils import (
    calc_ic, dataset_path, ensure_parent, get_device, make_loader, metrics_path,
    model_path, pred_path, labels_path, set_seed,
)

SEQ_LEN = 8
D_FEAT = 158
D_MODEL = 256
N_HEAD = 4
DROPOUT = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, choices=["csi300", "sp500"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="2009_2025")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tag = "smoke" if args.smoke else args.tag
    set_seed(args.seed)
    device = get_device(args.gpu)

    mp = model_path(args.universe, tag, args.seed)
    if not os.path.exists(mp):
        raise SystemExit(f"Model checkpoint not found: {mp}. Train first.")
    print(f"== test: universe={args.universe} seed={args.seed} tag={tag} device={device} ==")

    with open(dataset_path(args.universe, tag, "test"), "rb") as f:
        dl_test = pickle.load(f)
    test_loader = make_loader(dl_test, shuffle=False, drop_last=False, num_workers=0)

    model = MATCC(d_model=D_MODEL, d_feat=D_FEAT, seq_len=SEQ_LEN,
                  t_nhead=N_HEAD, S_dropout_rate=DROPOUT).to(device)
    model.load_state_dict(torch.load(mp, map_location=device))
    model.eval()

    preds, labels, ic, ric = [], [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(device)
            label = data[:, -1, -1]
            pred = model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())
            labels.append(label.ravel())
            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

    index = dl_test.get_index()
    predictions = pd.Series(np.concatenate(preds), name="score", index=index)
    label_series = pd.Series(np.concatenate(labels), name="label", index=index)

    metrics = {
        "IC": np.mean(ic),
        "ICIR": np.mean(ic) / np.std(ic),
        "RIC": np.mean(ric),
        "RICIR": np.mean(ric) / np.std(ric),
    }
    print(f"[test] metrics: {metrics}")

    pp = pred_path(args.universe, tag, args.seed)
    lp = labels_path(args.universe, tag, args.seed)
    mt = metrics_path(args.universe, tag, args.seed)
    for p in (pp, lp, mt):
        ensure_parent(p)
    with open(pp, "wb") as f:
        pickle.dump(predictions, f)
    with open(lp, "wb") as f:
        pickle.dump(label_series, f)
    with open(mt, "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")
    print(f"[test] pred -> {pp}")
    print(f"[test] labels -> {lp}")
    print(f"[test] metrics -> {mt}")


if __name__ == "__main__":
    main()
