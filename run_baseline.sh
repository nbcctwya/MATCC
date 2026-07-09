#!/usr/bin/env bash
# Full MATCC baseline: CSI300 + SP500, 5 seeds, Qlib TopK-DropN(30,5) backtest.
# Resumable: each step self-skips if its output already exists, and train.py resumes
# a partially-trained seed from the last completed epoch. Just re-run this script
# after an interruption to continue.
#
#   bash run_baseline.sh                 # resume / continue (skips done work)
#   FORCE=1 bash run_baseline.sh         # redo everything from scratch
#   ENV=matcc TAG=2009_2025 GPU=0 bash run_baseline.sh
#
# Each `conda run` is a fresh process (qlib.init once per process / per region).
set -euo pipefail

ENV=${ENV:-matcc}
TAG=${TAG:-2009_2025}
GPU=${GPU:-0}
FORCE=${FORCE:-0}
SEEDS="0 1 2 3 4"

EXTRA=""; TRAIN_EXTRA=""
if [ "$FORCE" = "1" ]; then EXTRA="--force"; TRAIN_EXTRA="--force --restart"; fi

for U in csi300 sp500; do
  echo "############### prepare_data: $U ($TAG) [FORCE=$FORCE] ###############"
  conda run -n "$ENV" python scripts/prepare_data.py --universe "$U" --tag "$TAG" $EXTRA

  for S in $SEEDS; do
    echo "############### train: $U seed=$S (skips if done / resumes if partial) ###############"
    conda run -n "$ENV" python train.py --universe "$U" --seed "$S" --tag "$TAG" --gpu "$GPU" $TRAIN_EXTRA
    echo "############### test: $U seed=$S (skips if pred exists) ###############"
    conda run -n "$ENV" python test.py --universe "$U" --seed "$S" --tag "$TAG" --gpu "$GPU" $EXTRA
  done

  echo "############### backtest: $U seeds=$SEEDS (always runs; cheap aggregation) ###############"
  conda run -n "$ENV" python backtest.py --universe "$U" --seeds 0,1,2,3,4 --tag "$TAG"
done

echo "############### DONE. See backtest_results/*_${TAG}_summary.csv ###############"
