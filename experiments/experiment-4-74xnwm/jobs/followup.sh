#!/bin/sh
# Rescore the models' own predictions against the published baseline, pool the
# cross-fitted arms so every perturbation is scored by a model that never saw
# it, and merge every arm's scores into the summary tables the report reads.
#
#   sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/followup.sh <crossfit runs jurkat> <hepg2> [arm]
#
# The bundle, the saved predictions and every training job's score tables
# arrive as staged input artifacts. Nothing is read from a job volume, so this
# can run long after the training jobs' volumes have expired.
set -eu
JURKAT_RUNS="$1"
HEPG2_RUNS="$2"
ARM="${3:-own_trainable}"
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_gpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
STAGED="$SILICO_INPUT_ARTIFACTS_DIR/experiments/$SILICO_EXPERIMENT_ID"
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/followup"
SUMMARY="$SILICO_EXPERIMENT_ARTIFACTS_DIR/summary"
MERGED="$PWD/scratch/merged_scores"
mkdir -p "$OUT" "$SUMMARY" "$MERGED"

echo "=== staged inputs ==="
ls "$STAGED"
echo "--- score tables ---"
ls "$STAGED/scores" | wc -l
echo "--- checkpoint dirs ---"
ls "$STAGED/checkpoints"

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

# ------------------------------------------------------------------ #
# merge and summarise
# ------------------------------------------------------------------ #
# The training jobs and this job both write direct_* tables. Same filename for
# the same (line, arm, baseline), so copying the follow-up over the staged copy
# replaces rather than duplicates.
echo "=== merging score tables ==="
cp "$STAGED/scores"/*.parquet "$MERGED"/
cp "$OUT"/direct_*.parquet "$MERGED"/ 2>/dev/null || true
ls "$MERGED" | wc -l

echo "=== summary tables ==="
python3 -m scfm_ft.analysis \
    --scores "$MERGED" --checkpoints "$STAGED/checkpoints" --out "$SUMMARY"

# the cross-fit tables are not part of the arm grid, so they ride along whole
cp "$OUT"/crossfit_*.parquet "$OUT"/crossfit_*.json "$SUMMARY"/ 2>/dev/null || true
cp "$STAGED/scores"/summary_*.json "$SUMMARY"/ 2>/dev/null || true

echo "=== one archive so the summary needs one download ==="
tar -czf "$SILICO_EXPERIMENT_ARTIFACTS_DIR/summary.tar.gz" -C "$SUMMARY" .

echo "=== outputs ==="
ls -la "$OUT"
ls -la "$SUMMARY"
