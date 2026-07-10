"""
Test MATCC for one (universe, seed): load the best-val checkpoint, predict on the
test split with correct day-batches (so SAttention attends across same-day stocks),
and write a correctly-aligned prediction/label Series + IC metrics.

Usage:
    conda run -n matcc python test.py --universe csi300 --seed 0 --tag 2009_2025
    conda run -n matcc python test.py --universe csi300 --seed 0 --smoke

Outputs:
    label_pred/{universe}/{tag}/{universe}_pred_{seed}.pkl    pd.Series name="score", aligned
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
    DailyBatchSamplerRandom, calc_ic, dataset_path, ensure_parent, get_device,
    labels_path, metrics_path, model_path, pred_path, set_seed,
)

SEQ_LEN, D_FEAT, D_MODEL, N_HEAD, DROPOUT = 8, 158, 256, 4, 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, choices=["csi300", "sp500"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="2009_2025")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun even if all outputs exist")
    args = ap.parse_args()

    tag = "smoke" if args.smoke else args.tag
    set_seed(args.seed)
    device = get_device(args.gpu)

    pp = pred_path(args.universe, tag, args.seed)
    lp = labels_path(args.universe, tag, args.seed)
    mt = metrics_path(args.universe, tag, args.seed)
    if not args.force and all(os.path.exists(p) for p in (pp, lp, mt)):
        print(f"[test] all outputs exist, skipping seed={args.seed}.")
        return

    mp = model_path(args.universe, tag, args.seed)
    if not os.path.exists(mp):
        raise SystemExit(f"Model checkpoint not found: {mp}. Train first.")
    print(f"== test: universe={args.universe} seed={args.seed} tag={tag} device={device} ==")

    with open(dataset_path(args.universe, tag, "test"), "rb") as f:
        dl_test = pickle.load(f)

    model = MATCC(d_model=D_MODEL, d_feat=D_FEAT, seq_len=SEQ_LEN,
                  t_nhead=N_HEAD, S_dropout_rate=DROPOUT).to(device)
    model.load_state_dict(torch.load(mp, map_location=device, weights_only=False))
    model.eval()

    # Predict day-by-day (each batch = one trading day's stocks, so SAttention is a
    # genuine same-day cross-stock attention). Assign predictions to their row
    # positions so the Series aligns with dl_test.get_index() (whatever its sort).
    n = len(dl_test)
    all_pred = np.full(n, np.nan, dtype=np.float64)
    all_label = np.full(n, np.nan, dtype=np.float64)
    ic_list, ric_list = [], []
    with torch.no_grad():
        for positions in DailyBatchSamplerRandom(dl_test, shuffle=False):
            positions = np.asarray(positions)
            batch = torch.as_tensor(dl_test[positions])             # [N, 8, 222]
            feature = batch[:, :, :-1].to(device).float()
            label = batch[:, -1, -1]
            pred = model(feature)                                   # [N]
            cpred, clab = pred.detach().cpu().numpy(), label.cpu().numpy()
            all_pred[positions] = cpred
            all_label[positions] = clab
            m = ~np.isnan(clab)
            if m.sum() > 1:
                di, dr = calc_ic(cpred[m], clab[m])
                ic_list.append(di)
                ric_list.append(dr)

    index = dl_test.get_index()
    predictions = pd.Series(all_pred, name="score", index=index)
    label_series = pd.Series(all_label, name="label", index=index)

    metrics = {
        "IC": float(np.mean(ic_list)),
        "ICIR": float(np.mean(ic_list) / np.std(ic_list)),
        "RIC": float(np.mean(ric_list)),
        "RICIR": float(np.mean(ric_list) / np.std(ric_list)),
    }
    print(f"[test] metrics: {metrics}")

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
