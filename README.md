# MATCC: Market-Aware Stock Prediction

This repository contains a corrected and reproducible pipeline for **MATCC: A Novel
Approach for Robust Stock Price Prediction Incorporating Market Trends and Cross-time
Correlations** (CIKM 2024). It supports CSI300 and S&P500 experiments with local Qlib
data.

![MATCC architecture](fig/MATCC_structure.png)

## What is fixed

The maintained pipeline addresses two lookahead problems that can otherwise produce
unrealistic IC values:

- Daily batches are built from the actual `datetime` index. Spatial attention sees
  stocks from one trading day, never future dates of the same stock.
- Labels remain raw in prepared datasets. Future labels are used only by the training
  loss mask and evaluation metrics, never by inference-time processors or model input
  selection.

The model still minimizes MSE during optimization. The final checkpoint is selected
by the **highest mean daily validation RankIC**, matching the downstream TopK ranking
objective. Test data is never used for checkpoint selection.

## Environment

The pipeline was validated with Python 3.9 and Qlib 0.9.7.

```bash
pip install pyqlib==0.9.7
pip install -r requirements.txt
```

Local Qlib data paths are configured in:

- `util/csi300.yaml`: `~/.qlib/qlib_data/cn_data`
- `util/sp500.yaml`: `~/.qlib/qlib_data/us_data`

The local providers must contain the target universes, benchmarks, and configured
market indices.

## Data splits and preprocessing

The default full experiment uses:

| Split | Period |
|---|---|
| Train | 2009-01-01 to 2020-12-31 |
| Validation | 2021-01-01 to 2022-12-31 |
| Test | 2023-01-01 to 2025-12-31 |

Each sample has shape `[8, 222]`:

```text
158 Alpha158 stock features + 63 market features + 1 label
```

Features are normalized using training-period robust statistics and missing values
are filled by Qlib. During training, labels are cross-sectionally normalized and the
top/bottom 2.5% are excluded from the loss only. All same-day stocks still participate
in spatial attention.

Prepared datasets include a manifest containing a hash of the complete normalized
YAML configuration. Changing dates, processors, labels, indices, or dataset parameters
automatically invalidates stale dataset caches.

## Running the pipeline

### Smoke test

Run a one-epoch CPU validation for both markets:

```bash
bash run_smoke.sh
```

This executes data preparation, training, testing, and backtesting on short date
windows without touching full-run artifacts.

### Full five-seed baseline

```bash
bash run_baseline.sh
```

The script runs CSI300 and S&P500 sequentially for seeds `0,1,2,3,4`, followed by
testing and TopK-DropN backtesting.

Useful options:

```bash
GPU=1 bash run_baseline.sh
ENV=matcc TAG=2009_2025 GPU=0 bash run_baseline.sh
FORCE=1 bash run_baseline.sh  # rebuild data and restart every seed from epoch 0
```

Normal runs are resumable. Re-run `bash run_baseline.sh` after an interruption to
continue from the last completed epoch. Do not use `FORCE=1` when resuming.

### Individual commands

```bash
python scripts/prepare_data.py --universe csi300 --tag 2009_2025
python train.py --universe csi300 --seed 0 --tag 2009_2025 --gpu 0
python test.py --universe csi300 --seed 0 --tag 2009_2025 --gpu 0
python backtest.py --universe csi300 --seeds 0,1,2,3,4 --tag 2009_2025
```

## Output layout

```text
dataset/<market>/                 prepared train/valid/test samplers + manifest
model_params/<market>/<tag>/     resume, best-RankIC, and final checkpoints
label_pred/<market>/<tag>/       indexed predictions and labels
metrics/<market>/<tag>/          per-seed IC/RankIC metrics
backtest_results/                five-seed portfolio summaries
logs/                            optional run logs
```

Generated artifacts are ignored by Git. A fitted `util/handler_*.pkl` accelerates data
rebuilding but is not required after prepared datasets exist; it can be deleted to
save disk space.

## Reproduced 2009–2025 results

All values below are mean ± sample standard deviation across seeds 0–4. Models are
selected by maximum validation RankIC.

### Prediction metrics

| Market | IC | ICIR | RankIC | RankICIR |
|---|---:|---:|---:|---:|
| CSI300 | 0.03306 ± 0.00668 | 0.19952 ± 0.04897 | 0.04538 ± 0.00379 | 0.26955 ± 0.03068 |
| S&P500 | 0.00056 ± 0.00513 | 0.00370 ± 0.04184 | 0.00039 ± 0.00274 | 0.00310 ± 0.02244 |

### Excess-return backtest

| Market | Annualized return, no cost | IR, no cost | Annualized return, with cost | IR, with cost | MDD, with cost |
|---|---:|---:|---:|---:|---:|
| CSI300 | 13.26% ± 4.55% | 1.394 ± 0.436 | 5.45% ± 4.60% | 0.567 ± 0.427 | -13.28% ± 3.31% |
| S&P500 | 0.52% ± 4.01% | 0.011 ± 0.385 | -7.34% ± 3.99% | -0.672 ± 0.474 | -36.47% ± 7.83% |

The corrected CSI300 results are positive and stable across seeds. The current MATCC
configuration does not produce a stable out-of-sample signal on S&P500.

## Tests

```bash
pytest -q tests
```

Regression tests cover daily batch construction, label masking, dataset cache
fingerprints, safe restarts, RNG restoration, atomic checkpoint publication, and
RankIC checkpoint selection.

## Project structure

```text
src/MATCC.py               MATCC model
src/DLinear.py             temporal decomposition block
src/RWKV.py                temporal correlation block
src/baseline_utils.py      shared sampler, metrics, paths, and reproducibility helpers
scripts/prepare_data.py    Qlib dataset builder
train.py                   resumable training and RankIC checkpoint selection
test.py                    aligned day-by-day inference
backtest.py                region-aware TopK-DropN backtest
util/*.yaml                full and smoke data configurations
tests/                     regression tests
```

## Acknowledgements

- [MASTER](https://github.com/SJTU-Quant/MASTER)
- [DLinear](https://github.com/cure-lab/LTSF-Linear)
- [RWKV](https://github.com/BlinkDL/RWKV-LM)
- [Qlib](https://github.com/microsoft/qlib)

## License

See [LICENSE](LICENSE).
