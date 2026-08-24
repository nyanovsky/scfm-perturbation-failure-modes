#!/usr/bin/env bash
# Stage 1, second pass: the two numeric comparisons the first pass left open.
#
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/jobs/run_provenance_tables.sh"
#
# CPU only, and it re-reads the first pass's checkpoints from that job's volume
# instead of downloading 6 GB again.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

H5_DIR=/tmp/h5py_pkgs
uv pip install --target "$H5_DIR" --no-deps h5py >/tmp/h5py_install.log 2>&1 || tail -5 /tmp/h5py_install.log
export PYTHONPATH="${PYTHONPATH}:${H5_DIR}"
uv run python -c "import h5py; print(f'[env] h5py={h5py.__version__}')"

OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/provenance"
mkdir -p "$OUT"
uv run python -m scfm_pfm.protein.audit_tables \
  --mount "$SILICO_REUSED_VOLUME_DIR" --out "$OUT"

echo "[done]"
ls -la "$OUT"
