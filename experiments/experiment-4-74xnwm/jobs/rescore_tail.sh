#!/bin/sh
# The part of jobs/rescore.sh that the first Jurkat attempt did not reach.
#
#   sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/rescore_tail.sh <cell_line> [arm]
#
# The first attempt re-scored the two cross-fitted runs and was killed by the
# wall clock partway through the upstream-sampling run. Jurkat carries 705
# perturbations against 3,247 read-out genes, so one re-read plus its scoring
# costs about 35 minutes here against about 17 in HepG2, and three of them plus
# the pooling did not fit in two hours. This job picks up the two remaining
# re-reads; the pooling runs beside it in its own job, since it only needs the
# per-perturbation tables the first attempt already wrote.
set -eu
LINE="$1"
ARM="${2:-own_trainable}"
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_gpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SCFM_FT_CKPT="$PWD/_ckpt/scgpt"
STAGED="$SILICO_INPUT_ARTIFACTS_DIR/experiments/$SILICO_EXPERIMENT_ID"
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/rescore/$LINE"
mkdir -p "$OUT"

echo "=== staged inputs ==="
ls "$STAGED/checkpoints"

FAILED=""
echo
echo "=== $LINE / ${ARM}__upstream : selected epoch, re-read ==="
if python3 -m scfm_ft.followup rescore \
    --checkpoints "$STAGED/checkpoints" --cell-line "$LINE" --run "${ARM}__upstream" \
    --scores-out "$OUT" --work "$PWD/_work/upstream" --device cuda
then
    echo "rescore ${ARM}__upstream ok"
else
    echo "WARNING: rescore ${ARM}__upstream failed" >&2
    FAILED="$FAILED upstream"
fi

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
echo "=== packing ==="
tar -czf "$SILICO_EXPERIMENT_ARTIFACTS_DIR/rescore_tail_${LINE}.tar.gz" -C "$OUT" .

echo
echo "=== outputs ==="
ls -la "$OUT"
if [ -n "$FAILED" ]; then
    echo "FAILED STEPS:$FAILED" >&2
    exit 1
fi
echo "all steps ok"
