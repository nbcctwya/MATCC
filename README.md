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
backtest_results/                five-seed portfolio summaries (excess return)
results/                         unified Protocol v1.0 evaluation tree (see below)
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

## Unified evaluation results (Protocol v1.0)

`results/` is a cross-baseline-comparable evaluation tree produced by the `eval/`
package. It reuses the existing predictions and runs the protocol's additional,
fully explicit Qlib TopK-DropN backtest; it does not retrain or overwrite the native
backtest. The metrics above
(excess return) are the project's native report; the metrics in `results/` use the
shared Protocol v1.0 convention, whose portfolio metrics are on the **absolute net
portfolio return** (`daily_return_gross - cost`), not excess return.

```bash
bash run_eval.sh                       # build both markets + finalize + validate
MARKETS="csi300" bash run_eval.sh      # one market
```

Layout:

```text
results/
├── metrics/
│   ├── seed_metrics.csv        one row per (market, model, seed): IC/ICIR/RankIC/RankICIR/
│   │                           AR/STD/MDD/Sharpe/Sortino/Calmar + num_test_days + pred path
│   ├── aggregate_metrics.csv   mean/std (ddof=1) across seeds, per (market, model)
│   └── ensemble_metrics.csv    per (market, model, ensemble_method): avg_none/avg_zscore/avg_rank
├── tables/
│   ├── seed_mean_std.csv       4-dp "mean ± std" presentation of aggregate_metrics
│   └── ensemble.csv            4-dp presentation of ensemble_metrics
├── curves/ensemble/*.csv       daily_ret_gross,cost,daily_ret_net,bench_ret,nav,bench_nav
├── metadata/
│   ├── eval_config.json        actual run口径: splits, costs, benchmark, metric convention, git
│   └── manifest.json           file registry + primary keys
└── diagnostics/
    └── validation.json         machine-readable check report (exit 0 iff all pass)
```

Metric convention (frozen): 252 trading days, `ddof=1`, risk-free 0, `MAR_daily=0`,
log returns. `IC`/`RankIC` are means of per-day cross-sectional Pearson/Spearman;
`ICIR`/`RankICIR` use `ddof=1` and are **not** annualized. `AR=exp(mean(g)*252)-1`,
`STD=std(g,ddof=1)*sqrt(252)`, `MDD=min(NAV/cummax(NAV)-1)` with `NAV=[1.0,exp(cumsum(g))]`,
`Sharpe=sqrt(252)*mean(g)/std(g,ddof=1)` (absolute net, not benchmark-relative IR),
`Sortino=sqrt(252)*mean(g)/sqrt(mean(min(g,0)^2))`, `Calmar=AR/|MDD|`, where
`g=log1p(daily_ret_net)`. Undefined inputs (zero denominator, `<2` samples) are NaN,
never 0; `daily_ret_net <= -1` is a hard error. Ensemble ranking metrics are recomputed
from the averaged score (never the seed-IC mean), and the ensemble re-runs the same
backtest. `results/_staging/` and `results/_cache/` are regenerable intermediates and
are git-ignored.

The protocol backtest explicitly sets `topk=30`, `n_drop=5`, `method_sell=bottom`,
`method_buy=top`, `hold_thresh=1`, `only_tradable=false`,
`forbid_all_trade_at_limit=true`, `risk_degree=0.95`, daily frequency, account
`100000000`, buy/sell costs `0.0005/0.0015`, and `min_cost=0`. Qlib's region config
owns `trade_unit`; the adapter does not override it. Predictions must cover the full
declared test calendar. Qlib reads the `t-1` signal for trading on `t` (`shift=1`), and
the adapter applies no additional date shift. The label horizon remains the native
`Ref($close,-5)/Ref($close,-1)-1` (forward close return from `t+1` through `t+5`).

## Tests

```bash
pytest -q tests
```

Regression tests cover daily batch construction, label masking, dataset cache
fingerprints, safe restarts, RNG restoration, atomic checkpoint publication, and
RankIC checkpoint selection. `tests/test_eval_metrics.py` pins the Protocol v1.0 metric
convention and its boundary conditions (first-day -10% drawdown, Sortino with repeated
negative days, `r <= -1` raises).

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
eval/                      unified Protocol v1.0 results export (metrics, build, finalize, validate)
run_eval.sh                build the comparable results/ tree from existing predictions
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
