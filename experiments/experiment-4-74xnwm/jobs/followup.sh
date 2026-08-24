#!/bin/sh
# Rescore the models' own predictions against the published baseline, and pool
# the cross-fitted arms so every perturbation is scored by a model that never
# saw it.
#
#   sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/followup.sh <crossfit runs jurkat> <hepg2>
#
# The bundle and the saved predictions arrive as staged input artifacts. The
# arm directories are read from the input artifact tree, not from a volume, so
# this job can run long after the training jobs' volumes have expired.
set -eu
JURKAT_RUNS="$1"
HEPG2_RUNS="$2"
ARM="${3:-own_trainable}"
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_gpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
STAGED="$SILICO_INPUT_ARTIFACTS_DIR/experiments/$SILICO_EXPERIMENT_ID"
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/followup"
mkdir -p "$OUT"

echo "=== staged inputs ==="
ls "$STAGED" || true
ls "$STAGED/checkpoints" | head -20 || true

echo "=== the model's own prediction, against the measured control profile ==="
python3 -m scfm_ft.followup direct \
    --checkpoints "$STAGED/checkpoints" --out "$OUT" --device cuda

for LINE in jurkat hepg2; do
    RUNS="$JURKAT_RUNS"
    [ "$LINE" = "hepg2" ] && RUNS="$HEPG2_RUNS"
    echo "=== $LINE: pooling $RUNS ==="
    python3 -m scfm_ft.followup crossfit \
        --scores "$STAGED/scores" --cell-line "$LINE" --runs "$RUNS" \
        --arm "$ARM" --out "$OUT" --device cuda
done

echo "=== outputs ==="
ls -la "$OUT"
