#!/usr/bin/env bash
# The secondary condition: 24 real non-targeting control cells kept UNAVERAGED,
# which is what gives the trained probe a held-out axis. Run 2's shipped
# individual-cells anchor uses four cells and averages their responses together
# --- right for a response feature, and it destroys exactly the variation a
# probe trains on.
#
# Roughly 24x the mean-control cost, so scGPT and scPRINT first. Whether
# TranscriptFormer gets this condition is decided by the mean-control result,
# not by this script.
#
# N_CELLS overrides how many control cells are drawn. It is a command-line
# argument and NOT an edit to `config.N_ANCHOR_CELLS`, which is part of the
# pre-registration: the recorded design is 24, and a run at fewer cells is a
# cheaper instance of that design rather than a new one. Each block records the
# count it was actually built from, so the two cannot be confused later.
#
# The floor is `config.N_HELDOUT_CELLS + 1` = 9; below that `analysis.py` emits
# no `heldout_cell` rows at all and the run measures nothing. At N_CELLS=16 the
# split is 8 train / 8 test, so the centroid every held-out cell is scored
# against is built from 8 cells rather than 16 --- a noisier centroid, which
# makes S3 harder to pass, not easier.
set -euo pipefail
cd "$(dirname "$0")/.."
export SCFM_PFM5_DATA_ROOT=${SCFM_PFM5_DATA_ROOT:-/biodata/nyanovsky/projects/scfm-pert-fail-modes/experiment-5-hidprb}
N_CELLS=${N_CELLS:-24}

run() {  # run <env> <model> <cell_line>
  echo "=== cells $2 / $3 (env $1, n_cells $N_CELLS)  $(date -Is)"
  PYTHONPATH=src "$HOME/miniconda3/envs/$1/bin/python" -m scfm_probe.probe.run \
      --model "$2" --cell-line "$3" --anchor individual_cells \
      --n-cells "$N_CELLS" --device cuda
  echo "=== ok $2 / $3  $(date -Is)"
}

for line in hepg2 jurkat; do
  run scgpt       scgpt   "$line"
  run scprint_env scprint "$line"
done
