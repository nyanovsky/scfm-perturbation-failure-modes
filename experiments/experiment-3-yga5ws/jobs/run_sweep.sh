#!/usr/bin/env bash
# The one entrypoint every training job uses, so the sweep, the null re-check
# and the transfer run are the same command with different arguments.
#
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/jobs/run_sweep.sh" --lines hepg2 ...
#
# cell-eval is installed into a separate --target directory that is APPENDED to
# PYTHONPATH, so the image's own numpy/torch keep priority; if the install
# fails the job still runs and records the failure instead of dying.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

CE_DIR=/tmp/cell_eval_pkgs
if uv pip install --target "$CE_DIR" "cell-eval==0.8.2" >/tmp/cell_eval_install.log 2>&1; then
  export PYTHONPATH="${PYTHONPATH}:${CE_DIR}"
  echo "[env] cell-eval installed into ${CE_DIR}"
else
  echo "[env] cell-eval install FAILED; continuing with our own discrimination only"
  tail -n 15 /tmp/cell_eval_install.log || true
fi

uv run python - <<'PY'
import torch, numpy, sys
print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} "
      f"numpy={numpy.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
      flush=True)
PY

uv run python -m scfm_pfm.translate.runner "$@"
