#!/usr/bin/env bash
# Reproduce the source paper's scGPT embedding baseline on its own BEELINE
# benchmark, so a disagreement about response features rests on having
# reproduced its setting first. CPU only; downloads ~210 MB.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
uv run python -m scfm_pfm.beeline \
  --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/beeline" --work ./_beeline_work
ls -la "$SILICO_EXPERIMENT_ARTIFACTS_DIR/beeline"
