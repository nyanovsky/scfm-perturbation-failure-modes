#!/usr/bin/env bash
# Recompute the effect target against one half of the control cells, so that the
# co-expression baseline and the thing it predicts share no cell.
#
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/jobs/run_recompute_targets.sh" --cell-lines hepg2,jurkat
#
# The image has no h5py (it is not part of the GPU stack), so it is installed
# into a separate --target directory APPENDED to PYTHONPATH: the image's own
# numpy keeps priority over anything the install pulls in.
#
# The raw matrices may still be in the ingest job's volume, mounted read-only
# here; if they are not, the ingest code downloads them from GEO into the job's
# own volume, which costs 15 GB of transfer but nothing else.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

H5_DIR=/tmp/h5py_pkgs
uv pip install --target "$H5_DIR" h5py 2>&1 | tail -3
export PYTHONPATH="${PYTHONPATH}:${H5_DIR}"
uv run python -c "import h5py, numpy; print(f'[env] h5py={h5py.__version__} numpy={numpy.__version__}')"

RAW_DIR="_work/_raw"
MOUNT="${SILICO_REUSED_VOLUME_DIR:-}"
if [ -n "$MOUNT" ]; then
  echo "[env] reused volume top level:"
  find "$MOUNT" -maxdepth 2 -type d 2>/dev/null | head -20 || true
  found=$(find "$MOUNT" -maxdepth 12 -name '*_raw_singlecell_01.h5ad' -print -quit 2>/dev/null || true)
  if [ -n "$found" ]; then
    RAW_DIR=$(dirname "$found")
    echo "[env] reusing raw matrices from the ingest volume"
  else
    echo "[env] no raw matrices in the ingest volume; they will be downloaded from GEO"
  fi
fi
mkdir -p "$RAW_DIR" 2>/dev/null || true
echo "[env] raw matrices: $RAW_DIR"
ls -la "$RAW_DIR" || true
echo "[env] bundle: $(uv run python -c 'from scfm_pfm import paths; print(paths.bundle_root())')"

uv run python -m scfm_pfm.ingest.recompute_effects --work "$(dirname "$RAW_DIR")" "$@"
