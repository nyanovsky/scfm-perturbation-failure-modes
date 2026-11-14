#!/bin/sh
# Re-score the tagged runs of one cell line, then pool the cross-fitted folds.
#
#   sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/rescore.sh <cell_line> [arm]
#
# Why this job exists: run 3's bundle reader separates a feature file's model
# key from its anchor by splitting on the FIRST "__", so a run whose name
# carried a tag ("own_trainable__f23") wrote a block the reader filed under the
# wrong anchor. Its gene-table arm still resolved; its response arms did not,
# and were dropped from the scored set. The model keys now carry the tag behind
# "-", and the three tagged runs plus the last-epoch reading are produced again
# here from the saved weights -- no retraining, the same extraction and the same
# scorer as the main runs.
set -eu
LINE="$1"
ARM="${2:-own_trainable}"
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_gpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SCFM_FT_CKPT="$PWD/_ckpt/scgpt"
STAGED="$SILICO_INPUT_ARTIFACTS_DIR/experiments/$SILICO_EXPERIMENT_ID"
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/rescore/$LINE"
POOL="$PWD/_pool"
mkdir -p "$OUT" "$POOL"

echo "=== staged inputs ==="
ls "$STAGED/checkpoints"
ls "$STAGED/scores"

FAILED=""
for RUN in "${ARM}__f23" "${ARM}__f4" "${ARM}__upstream"; do
    echo
    echo "=== $LINE / $RUN : selected epoch, re-read ==="
    if python3 -m scfm_ft.followup rescore \
        --checkpoints "$STAGED/checkpoints" --cell-line "$LINE" --run "$RUN" \
        --scores-out "$OUT" --work "$PWD/_work/$RUN" --device cuda
    then
        echo "rescore $RUN ok"
    else
        echo "WARNING: rescore $RUN failed" >&2
        FAILED="$FAILED $RUN"
    fi
done

echo
echo "=== $LINE / $ARM : the LAST epoch, which the selection rule rejected ==="
if python3 -m scfm_ft.followup lastepoch \
    --checkpoints "$STAGED/checkpoints" --cell-line "$LINE" --run "$ARM" \
    --scores-out "$OUT" --work "$PWD/_work/lastepoch" --device cuda
then
    echo "lastepoch ok"
else
    echo "WARNING: lastepoch failed" >&2
    FAILED="$FAILED lastepoch"
fi

echo
echo "=== pooling the cross-fitted folds ==="
# the pool needs the main run's per-perturbation table beside the two
# cross-fitted ones, and each run's summary so the fold assignment is read
# rather than assumed
cp "$STAGED/scores/per_pert_${LINE}__${ARM}.parquet" "$POOL"/
cp "$STAGED/scores/summary_${LINE}__${ARM}.json" "$POOL"/
cp "$OUT"/per_pert_*.parquet "$POOL"/ 2>/dev/null || true
cp "$OUT"/summary_*.json "$POOL"/ 2>/dev/null || true
ls "$POOL"
if python3 -m scfm_ft.followup crossfit \
    --scores "$POOL" --checkpoints "$STAGED/checkpoints" \
    --cell-line "$LINE" --runs "${ARM},${ARM}__f23,${ARM}__f4" \
    --arm "$ARM" --out "$OUT" --device cuda
then
    echo "crossfit ok"
else
    echo "WARNING: crossfit failed" >&2
    FAILED="$FAILED crossfit"
fi

echo
echo "=== packing for the summary job ==="
tar -czf "$SILICO_EXPERIMENT_ARTIFACTS_DIR/rescore_pack_${LINE}.tar.gz" -C "$OUT" .

echo
echo "=== outputs ==="
ls -la "$OUT"
if [ -n "$FAILED" ]; then
    echo "FAILED STEPS:$FAILED" >&2
    exit 1
fi
echo "all steps ok"
