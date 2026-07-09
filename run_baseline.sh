#!/usr/bin/env bash
# Full MATCC baseline: CSI300 + SP500, 5 seeds, Qlib TopK-DropN(30,5) backtest.
# Each `conda run` is a fresh process, so qlib.init-once-per-process / per-region holds.
#
#   bash run_baseline.sh
#   ENV=matcc TAG=2009_2025 GPU=0 bash run_baseline.sh
set -euo pipefail

ENV=${ENV:-matcc}
TAG=${TAG:-2009_2025}
GPU=${GPU:-0}
SEEDS="0 1 2 3 4"

for U in csi300 sp500; do
  echo "############### prepare_data: $U ($TAG) ###############"
  conda run -n "$ENV" python scripts/prepare_data.py --universe "$U" --tag "$TAG"

  for S in $SEEDS; do
    echo "############### train: $U seed=$S ###############"
    conda run -n "$ENV" python train.py --universe "$U" --seed "$S" --tag "$TAG" --gpu "$GPU"
    echo "############### test: $U seed=$S ###############"
    conda run -n "$ENV" python test.py --universe "$U" --seed "$S" --tag "$TAG" --gpu "$GPU"
  done

  echo "############### backtest: $U seeds=$SEEDS ###############"
  conda run -n "$ENV" python backtest.py --universe "$U" --seeds 0,1,2,3,4 --tag "$TAG"
done

echo "############### DONE. See backtest_results/*_${TAG}_summary.csv ###############"
