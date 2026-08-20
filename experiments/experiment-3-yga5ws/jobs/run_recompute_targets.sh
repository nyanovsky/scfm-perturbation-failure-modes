#!/usr/bin/env bash
# Recompute the effect target against one half of the control cells, so that the
# co-expression baseline and the thing it predicts share no cell.
#
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/jobs/run_recompute_targets.sh" --cell-lines hepg2,jurkat
#
# The raw matrices are already in the ingest job's volume; this job mounts it
# read-only and streams from there instead of downloading 15 GB again.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

RAW_DIR="_work/_raw"
MOUNT="${SILICO_REUSED_VOLUME_DIR:-}"
if [ -n "$MOUNT" ]; then
  found=$(find "$MOUNT" -maxdepth 8 -name 'GSE264667_*_raw_singlecell_01.h5ad' -print -quit 2>/dev/null || true)
  if [ -n "$found" ]; then
    RAW_DIR=$(dirname "$found")
  fi
fi
mkdir -p "$RAW_DIR" 2>/dev/null || true
echo "[env] raw matrices: $RAW_DIR"
ls -la "$RAW_DIR" || true
echo "[env] bundle: $(uv run python -c 'from scfm_pfm import paths; print(paths.bundle_root())')"

uv run python -m scfm_pfm.ingest.recompute_effects --work "$(dirname "$RAW_DIR")" "$@"
