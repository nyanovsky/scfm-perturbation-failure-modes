#!/usr/bin/env bash
# C1-C3 over real model output, 8 perturbed genes, first and last layer only.
# Minutes per model. Never enough to read a number from --- enough to prove the
# intervention lands, that layer 0 is exactly zero off-target, and that a no-op
# is exactly zero.
set -euo pipefail
cd "$(dirname "$0")/.."
export SCFM_PFM5_DATA_ROOT=${SCFM_PFM5_DATA_ROOT:-/biodata/nyanovsky/projects/scfm-pert-fail-modes/experiment-5-hidprb}

run() {  # run <env> <model> <cell_line>
  echo "=== smoke $2 / $3 (env $1)"
  PYTHONPATH=src "$HOME/miniconda3/envs/$1/bin/python" -m scfm_probe.probe.run \
      --model "$2" --cell-line "$3" --anchor mean_control --device cuda --smoke
}

run scgpt              scgpt            hepg2
run scprint_env        scprint          hepg2
run transcriptformer_env transcriptformer hepg2
