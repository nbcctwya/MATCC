# CIKM2024-MATCC: A Novel Approach for Robust Stock Price Prediction Incorporating Market Trends and Cross-time Correlations

This repository is the official implementation of **MATCC: A Novel Approach for Robust Stock Price Prediction Incorporating Market Trends and Cross-time Correlations**.

MATCC is a novel framework for robust stock price prediction, which explicitly extracts market trends as guiding information, decomposes stock data into trend and fluctuation components, and employs a carefully designed structure for mining cross-time correlation.

![MATCC framework](fig/MATCC_structure.png)

## Reproduction pipeline

The maintained pipeline supports CSI300 and S&P500 and avoids two lookahead traps in
the historical scripts: daily batches are grouped by the actual `datetime` index, and
future labels are never used by inference-time processors.

### Requirements

1. Create a Python 3.9 environment and install Qlib 0.9.7 plus the pinned project
   dependencies:

```bash
pip install pyqlib==0.9.7
pip install -r requirements.txt
```

2. Put local Qlib data at the paths configured in `util/csi300.yaml` and
   `util/sp500.yaml` (by default `~/.qlib/qlib_data/cn_data` and
   `~/.qlib/qlib_data/us_data`). The data must include the configured universes,
   benchmarks, and three market indices.

3. Adjust split dates, processors, labels, or market indices in the corresponding
   YAML file when needed. Prepared datasets carry a hash of the complete normalized
   configuration, so a YAML change automatically rebuilds stale split files.

### Quick validation

Run the complete CPU smoke chain for both markets:

```bash
bash run_smoke.sh
```

This recreates smoke artifacts and runs data preparation, one-epoch training,
evaluation, and backtesting.

### Full baseline

Run the resumable five-seed 2009–2025 pipeline:

```bash
bash run_baseline.sh
```

Useful variants:

```bash
GPU=1 bash run_baseline.sh
FORCE=1 bash run_baseline.sh   # discard prior checkpoints and retrain from epoch 0
```

An interrupted normal run can be resumed by running the same command again. During
an explicit restart, old resume, best, and final checkpoints are removed together so
an obsolete final model cannot be mistaken for a completed new run.

Training minimizes MSE, but checkpoint selection follows the portfolio objective:
the epoch with the highest mean daily validation RankIC is published as the final
`TEST_*.pth` model. Test data is never used for checkpoint selection.

The individual maintained entry points are:

```bash
python scripts/prepare_data.py --universe csi300 --tag 2009_2025
python train.py --universe csi300 --seed 0 --tag 2009_2025 --gpu 0
python test.py --universe csi300 --seed 0 --tag 2009_2025 --gpu 0
python backtest.py --universe csi300 --seeds 0,1,2,3,4 --tag 2009_2025
```

`train_model_MATCC.py` and `test_model_MATCC.py` are deprecated compatibility aliases
that forward to `train.py` and `test.py`; they no longer contain independent legacy
samplers. The obsolete `util/generate_dataset.py`, `util/2023.yaml`, and `backTest.py`
entry points were removed because they used the old label-aware inference pipeline.

### Preprocessing

Features use training-period robust Z-score normalization and missing-value filling.
Labels remain raw in prepared datasets. During training, cross-sectional label
normalization and extreme-label handling are applied inside the training loop; test
labels are used only for metric computation.

## Results

### Overall performance

Our model achieves the following performance on CSI300, CSI800, S&P500 (**2020.07 - 2023.12.31**) and NK225 (**2022.07 - 2024.07**):

![Performance comparison](fig/all_datasets_performance.png)

<!-- | Model name  | IC              | ICIR            | RankIC         | RankICIR        | AR              | IR              |
| ----------- | --------------- | --------------- | -------------- | --------------- | --------------- | --------------- |
| Ours        | **0.117242833** | **1.024316702** | **0.08575864** | **0.870096253** | **0.803259517** | **8.466878624** |
| MASTER      | 0.053792434     | 0.396283874     | 0.054635386    | 0.39024223      | 0.195310604     | 1.930453898     |
| DTML        | 0.051088842     | 0.35034742      | 0.051845797    | 0.351184897     | 0.154697641     | 1.537347888     |
| Transformer | 0.048012156     | 0.323552332     | 0.046412452    | 0.324412535     | 0.113665438     | 1.036304061     |
| ALSTM       | 0.043422655     | 0.304667419     | 0.041286052    | 0.303555486     | 0.110967154     | 1.092517725     |
| LSTM        | 0.048424638     | 0.336684621     | 0.050138655    | 0.340297238     | 0.132869265     | 1.336330359     |
| GAT         | 0.05297247      | 0.388511199     | 0.053731579    | 0.388697025     | 0.187204099     | 1.914184519     |
| GRU         | 0.04625093      | 0.323989797     | 0.046433043    | 0.326022902     | 0.107347975     | 1.048304919     | -->

![Performance comparison](fig/radia.png)

### Portfolio Results

Extreme Market environment

![extreme market environment](fig/cumulative_return_202401_202403.png)

Normal Market environment

![normal market environment](fig/cumulative_return_202301_202303.png)

## Acknowledgement and Reference Repositories

We appreciate the following github repos a lot for their valuable code base or datasets:

- MASTER: https://github.com/SJTU-Quant/MASTER
- AutoFormer: https://github.com/thuml/Autoformer
- DLinear: https://github.com/cure-lab/LTSF-Linear
- RWKV: https://github.com/BlinkDL/RWKV-LM
- Pytorch_linear_WarmUp_CosineAnnealing: https://github.com/saadnaeem-dev/pytorch-linear-warmup-cosine-annealing-warm-restarts-weight-decay
