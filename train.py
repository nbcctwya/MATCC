"""
Train MATCC (default hyperparameters) for one (universe, seed), with resume.

Usage:
    conda run -n matcc python train.py --universe csi300 --seed 0 --tag 2009_2025
    conda run -n matcc python train.py --universe csi300 --seed 0 --smoke   # 1 epoch, CPU

Resume behaviour:
  * If the final checkpoint TEST_MATCC_{u}_seed_{s}.pth exists  -> skip (already done),
    unless --force.
  * Else if last_MATCC_{u}_seed_{s}.pth exists -> resume from the next epoch
    (restores model + optimizer + RNG, and fast-forwards the LR scheduler).
  * Else -> train from epoch 0.
  --restart drops the resume checkpoint and starts over; --force retrains even if done.

Reproduces the reference train_model_MATCC.py loop (DailyBatchSamplerRandom day-batches,
MSE+NaN-mask loss, Adam + ChainedScheduler warmup-cosine, grad-value clip 3.0), fully
RNG-seeded, with tag-aware paths.
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.optim as optim

from src.MATCC import MATCC
from my_lr_scheduler import ChainedScheduler
from src.baseline_utils import (
    calc_ic, dataset_path, ensure_parent, get_device, last_ckpt_path,
    limit_threads, loss_fn, make_loader, model_path, set_seed, worker_init_fn,
)

# ---- Default training hyperparameters (from the original TrainConfig) --------
N_EPOCH = 75
LR = 3e-4
WEIGHT_DECAY = 0.001
GAMMA = 1.0
COEF = 1.0
COSINE_PERIOD = 4
T_0 = 15
T_MULT = 1
WARMUP_EPOCH = 10
ETA_MIN = 2e-5
SEQ_LEN = 8
D_FEAT = 158
D_MODEL = 256
N_HEAD = 4
DROPOUT = 0.5
GRAD_CLIP = 3.0
NUM_WORKERS = 2


def build_scheduler(optimizer):
    return ChainedScheduler(
        optimizer, T_0=T_0, T_mul=T_MULT, eta_min=ETA_MIN, last_epoch=-1,
        max_lr=LR, warmup_steps=WARMUP_EPOCH, gamma=GAMMA, coef=COEF,
        step_size=3, cosine_period=COSINE_PERIOD)


def save_resume(path, epoch, model, optimizer):
    """Atomically save a per-epoch resume checkpoint (tmp + rename)."""
    ensure_parent(path)
    tmp = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_np": np.random.get_state(),
        "rng_py": random.getstate(),
    }, tmp)
    os.replace(tmp, path)


def restore_rng(ckpt):
    torch.set_rng_state(ckpt["rng_torch"])
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    np.random.set_state(ckpt["rng_np"])
    random.setstate(ckpt["rng_py"])


def train_epoch(data_loader, optimizer, lr_scheduler, model, device):
    model.train()
    losses = []
    for data in data_loader:
        data = torch.squeeze(data, dim=0)
        # data: [N, T=8, F=222] = 158 stock + 63 market + 1 label
        feature = data[:, :, 0:-1].to(device)
        label = data[:, -1, -1].to(device)
        pred = model(feature.float())
        loss = loss_fn(pred, label)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), GRAD_CLIP)
        optimizer.step()
    lr_scheduler.step()
    return float(np.mean(losses))


def valid_epoch(data_loader, model, device):
    model.eval()
    losses, ic, ric = [], [], []
    with torch.no_grad():
        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(device)
            label = data[:, -1, -1].to(device)
            pred = model(feature.float())
            losses.append(loss_fn(pred, label).item())
            daily_ic, daily_ric = calc_ic(
                pred.detach().cpu().numpy(), label.detach().cpu().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)
    metrics = {
        "IC": np.mean(ic),
        "ICIR": np.mean(ic) / np.std(ic),
        "RIC": np.mean(ric),
        "RICIR": np.mean(ric) / np.std(ric),
    }
    return float(np.mean(losses)), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, choices=["csi300", "sp500"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="2009_2025")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if the final checkpoint already exists")
    ap.add_argument("--restart", action="store_true",
                    help="ignore any resume checkpoint and start from epoch 0")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override n_epoch (e.g. to test resume on smoke data)")
    args = ap.parse_args()

    tag = "smoke" if args.smoke else args.tag
    n_epoch = args.epochs if args.epochs is not None else (1 if args.smoke else N_EPOCH)
    num_workers = 0 if args.smoke else NUM_WORKERS

    final_path = model_path(args.universe, tag, args.seed)
    resume_path = last_ckpt_path(args.universe, tag, args.seed)

    # Skip if already fully trained.
    if os.path.exists(final_path) and not args.force:
        print(f"[train] final checkpoint exists, skipping: {final_path}")
        return

    if args.restart and os.path.exists(resume_path):
        print(f"[train] --restart: dropping resume checkpoint {resume_path}")
        os.remove(resume_path)

    limit_threads(4)
    set_seed(args.seed)  # sets cudnn flags + base seed (also the DataLoader worker base)
    device = get_device(args.gpu)
    pin_memory = device.type == "cuda"

    print(f"== train: universe={args.universe} seed={args.seed} tag={tag} "
          f"device={device} epochs={n_epoch} ==")

    # Data
    loaders = {}
    for split in ("train", "valid", "test"):
        with open(dataset_path(args.universe, tag, split), "rb") as f:
            ds = pickle.load(f)
        loaders[split] = make_loader(
            ds, shuffle=(split == "train"), drop_last=(split == "train"),
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn if num_workers > 0 else None)
    print("[train] data loaded.")

    model = MATCC(d_model=D_MODEL, d_feat=D_FEAT, seq_len=SEQ_LEN,
                  t_nhead=N_HEAD, S_dropout_rate=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999),
                           weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        # ChainedScheduler is not a torch _LRScheduler; rebuild and fast-forward to the
        # resume epoch (verified to reproduce the exact per-epoch LR).
        lr_scheduler = build_scheduler(optimizer)
        for _ in range(start_epoch):
            lr_scheduler.step()
        restore_rng(ckpt)
        print(f"[train] resumed from epoch {start_epoch} (checkpoint epoch {ckpt['epoch']}).")
    else:
        lr_scheduler = build_scheduler(optimizer)
        print("[train] starting fresh from epoch 0.")

    if start_epoch >= n_epoch:
        print(f"[train] checkpoint already at epoch {start_epoch} >= {n_epoch}; "
              f"writing final model.")

    ensure_parent(final_path)
    for step in range(start_epoch, n_epoch):
        train_loss = train_epoch(loaders["train"], optimizer, lr_scheduler, model, device)
        val_loss, valid_metrics = valid_epoch(loaders["valid"], model, device)
        test_loss, test_metrics = valid_epoch(loaders["test"], model, device)
        print(f"[train] epoch {step}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"test_loss={test_loss:.6f} | val_IC={valid_metrics['IC']:.4f} "
              f"test_IC={test_metrics['IC']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")
        save_resume(resume_path, step, model, optimizer)

    torch.save(model.state_dict(), final_path)
    print(f"[train] saved final model -> {final_path}")


if __name__ == "__main__":
    main()
