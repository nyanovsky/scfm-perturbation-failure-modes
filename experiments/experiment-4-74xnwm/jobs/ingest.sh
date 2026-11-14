#!/bin/sh
# Build one cell line's fine-tuning matrices from the raw GEO screen.
#
# Run from the delivered worktree root:  sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/ingest.sh <line>
# The raw h5ad is cached in the worktree copy (_raw/), which stays on the job's
# own volume and is never uploaded; only the compact matrices go to the
# artifact directory.
set -eu
LINE="$1"
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_cpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "=== ingest $LINE ==="
echo "input artifacts: ${SILICO_INPUT_ARTIFACTS_DIR:-unset}"
ls -R "${SILICO_INPUT_ARTIFACTS_DIR:-/nonexistent}" 2>/dev/null | head -20 || true
python3 -m scfm_ft.ingest --cell-line "$LINE" --raw ./_raw --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/cells"
echo "=== outputs ==="
ls -la "$SILICO_EXPERIMENT_ARTIFACTS_DIR/cells/$LINE"
echo "=== volume layout (for a later job that mounts it) ==="
echo "worktree root: $PWD"
echo "raw cache:     $PWD/_raw"
du -sh ./_raw 2>/dev/null || true
