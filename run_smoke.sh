#!/usr/bin/env bash
# Lightweight CPU smoke test: tiny date windows, 1 epoch, single seed, both markets.
# Validates the whole pipeline (vendored dataset, both qlib regions, train fwd/bwd,
# checkpoint round-trip, pred pickle, TopkDropout(30,5) backtest with costs) in a few
# minutes on CPU, without disturbing GPU jobs.
#
#   bash run_smoke.sh
set -euo pipefail

ENV=${ENV:-matcc}
export CUDA_VISIBLE_DEVICES=""

for U in csi300 sp500; do
  echo "######### SMOKE prepare: $U #########"
  conda run -n "$ENV" python scripts/prepare_data.py --universe "$U" --smoke
  echo "######### SMOKE train: $U #########"
  conda run -n "$ENV" python train.py --universe "$U" --seed 0 --smoke
  echo "######### SMOKE test: $U #########"
  conda run -n "$ENV" python test.py --universe "$U" --seed 0 --smoke
  echo "######### SMOKE backtest: $U #########"
  conda run -n "$ENV" python backtest.py --universe "$U" --seeds 0 --smoke
done

echo "######### SMOKE DONE. See backtest_results/*_smoke_summary.csv #########"
