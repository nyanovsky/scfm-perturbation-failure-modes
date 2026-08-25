#!/bin/sh
# One fine-tuning arm end to end: train, query, score.
#
# Run from the delivered worktree root:
#   sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/train_arm.sh <cell_line> <arm> [--coexp-shuffle]
#
# The bundle and the ESM2 table arrive as staged input artifacts; the per-cell
# matrices arrive on the ingest job's mounted volume. Durable outputs -- the
# weights, the tables, the histories and the scores -- go to the artifact
# directory; the raw checkpoint download stays on this job's own volume.
set -eu
LINE="$1"
ARM="$2"
shift 2
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_gpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SCFM_FT_CKPT="$PWD/_ckpt/scgpt"

echo "=== $LINE / $ARM ==="
echo "run suffix      : ${SCFM_FT_RUN_SUFFIX:-none}"
echo "input artifacts : ${SILICO_INPUT_ARTIFACTS_DIR:-unset}"
echo "reused volume   : ${SILICO_REUSED_VOLUME_DIR:-unset}"
echo "artifacts       : $SILICO_EXPERIMENT_ARTIFACTS_DIR"
find "${SILICO_REUSED_VOLUME_DIR:-/nonexistent}" -maxdepth 3 -name '*.npz' 2>/dev/null | head -10 || true

python3 -m scfm_ft.run_arm --cell-line "$LINE" --arm "$ARM" --work "$PWD/_work" "$@"

SUF="${SCFM_FT_RUN_SUFFIX:-}"
echo "=== outputs ==="
ls -la "$SILICO_EXPERIMENT_ARTIFACTS_DIR/checkpoints$SUF" || true
ls -la "$SILICO_EXPERIMENT_ARTIFACTS_DIR/scores$SUF" || true
ls -la "$SILICO_EXPERIMENT_ARTIFACTS_DIR/packs$SUF" || true
