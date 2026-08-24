#!/usr/bin/env bash
# Stage 1 of the protein-model control: was each model's gene-identity table
# updated during pretraining, or is it still the vectors it started from?
#
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/jobs/run_provenance.sh"
#
# CPU only. It downloads the four released checkpoints (~6 GB) into the job's
# own worktree copy, which lives on the per-job volume, and writes only the
# audit record to the artifacts directory. h5py goes into a separate --target
# appended to PYTHONPATH, because one of TranscriptFormer's vocabulary files
# may be HDF5 and the image does not carry a reader for it.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

H5_DIR=/tmp/h5py_pkgs
if uv pip install --target "$H5_DIR" --no-deps h5py >/tmp/h5py_install.log 2>&1; then
  export PYTHONPATH="${PYTHONPATH}:${H5_DIR}"
  echo "[env] h5py installed into ${H5_DIR}"
else
  echo "[env] h5py install FAILED; an HDF5 vocabulary file would be reported unread"
  tail -n 5 /tmp/h5py_install.log || true
fi

uv run python - <<'PY'
import sys, numpy, torch
print(f"[env] python={sys.version.split()[0]} numpy={numpy.__version__} torch={torch.__version__}",
      flush=True)
PY

OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/provenance"
mkdir -p "$OUT"
uv run python -m scfm_pfm.protein.audit --cache ./_ckpt --out "$OUT"

echo "[done] provenance audit in $OUT"
ls -la "$OUT"
du -sh ./_ckpt || true
