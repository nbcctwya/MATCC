#!/usr/bin/env bash
# Build the unified Baseline Results Protocol v1.0 results/ tree from existing
# predictions + the project's Qlib backtest. No model training; no strategy change.
#
#   bash run_eval.sh                     # build both markets + finalize + validate
#   MARKETS="csi300" bash run_eval.sh    # one market
#   TAG=2009_2025 ENV=matcc GPU=0 bash run_eval.sh
#
# Prerequisite: test.py has produced label_pred/{universe}/{tag}/*.pkl for all seeds
# (i.e. the train -> test pipeline has run via run_baseline.sh).
set -euo pipefail

ENV=${ENV:-matcc}
TAG=${TAG:-2009_2025}
MARKETS=${MARKETS:-"csi300 sp500"}
SEEDS=${SEEDS:-"0,1,2,3,4"}
OUT=${OUT:-results}

for U in $MARKETS; do
  echo "############### build_results: $U ($TAG) seeds=$SEEDS ###############"
  conda run -n "$ENV" python eval/build_results.py --universe "$U" --tag "$TAG" --seeds "$SEEDS" --out "$OUT"
done

echo "############### finalize_results: merge + aggregate + metadata ###############"
conda run -n "$ENV" python eval/finalize_results.py --out "$OUT" --tag "$TAG"

echo "############### validate_results: protocol checks -> diagnostics/validation.json ###############"
conda run -n "$ENV" python eval/validate_results.py --out "$OUT"

echo "############### DONE. See $OUT/diagnostics/validation.json ###############"
