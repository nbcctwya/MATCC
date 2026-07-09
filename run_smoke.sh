#!/usr/bin/env bash
# Lightweight CPU smoke test: tiny date windows, 1 epoch, single seed, both markets.
# Clears previous smoke outputs first so each run is a genuine fresh validation of the
# full chain (prepare -> train -> test -> backtest). CPU-only -> won't disturb GPU jobs.
#
#   bash run_smoke.sh
set -euo pipefail

ENV=${ENV:-matcc}
export CUDA_VISIBLE_DEVICES=""

# Drop any previous smoke artifacts (tag=smoke) for a clean re-run.
rm -rf dataset/csi300/*smoke* dataset/sp500/*smoke* \
       model_params/csi300/smoke model_params/sp500/smoke \
       label_pred/csi300/smoke label_pred/sp500/smoke \
       metrics/csi300/smoke metrics/sp500/smoke \
       backtest_results/*smoke* util/handler_*smoke* 2>/dev/null || true

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
