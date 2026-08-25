#!/bin/sh
# Pool the cross-fitted folds of one cell line.
#
#   sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/crossfit.sh <cell_line> [arm]
#
# Reads the per-perturbation tables of the main run and of the two
# complementary-fold runs and pools each model's held-out rows, so every
# perturbation in the screen is scored by a model that never saw it. No model
# is loaded and nothing is refitted: the tables already carry one row per
# perturbation. It is split out of jobs/rescore.sh so it does not have to wait
# behind the re-reads.
set -eu
LINE="$1"
ARM="${2:-own_trainable}"
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_gpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
STAGED="$SILICO_INPUT_ARTIFACTS_DIR/experiments/$SILICO_EXPERIMENT_ID"
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/rescore/$LINE"
POOL="$PWD/_pool"
mkdir -p "$OUT" "$POOL"

cp "$STAGED/scores/per_pert_${LINE}__${ARM}.parquet" "$POOL"/
cp "$STAGED/scores/summary_${LINE}__${ARM}.json" "$POOL"/
cp "$STAGED/rescore/$LINE"/per_pert_*.parquet "$POOL"/
cp "$STAGED/rescore/$LINE"/summary_*.json "$POOL"/
echo "=== pool ==="
ls "$POOL"

python3 -m scfm_ft.followup crossfit \
    --scores "$POOL" --checkpoints "$STAGED/checkpoints" \
    --cell-line "$LINE" --runs "${ARM},${ARM}__f23,${ARM}__f4" \
    --arm "$ARM" --out "$OUT" --device cuda

tar -czf "$SILICO_EXPERIMENT_ARTIFACTS_DIR/crossfit_pack_${LINE}.tar.gz" \
    -C "$OUT" crossfit_${LINE}__${ARM}.json crossfit_${LINE}__${ARM}.parquet \
    crossfit_per_pert_${LINE}__${ARM}.parquet

echo "=== outputs ==="
ls -la "$OUT"
echo "crossfit ok"
