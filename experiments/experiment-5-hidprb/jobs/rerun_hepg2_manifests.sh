#!/usr/bin/env bash
# Re-run the two HepG2 blocks written before the C3 and no-op fixes, so all six
# blocks in the grid are produced by one version of the driver.
#
# Their ARRAYS are unaffected --- neither block has a no-op query nor an undefined
# self position, which is all those fixes touch. What differs is the manifest: C3
# was recorded as a passing control on a comparison that never ran, and the Jurkat
# blocks written later correctly record it as an absence. Two blocks in one grid
# disagreeing about what C3 means is worse than eleven minutes of GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
export SCFM_PFM5_DATA_ROOT=${SCFM_PFM5_DATA_ROOT:-/biodata/nyanovsky/projects/scfm-pert-fail-modes/experiment-5-hidprb}

for pair in "scgpt:scgpt" "scprint_env:scprint"; do
  env=${pair%%:*}; model=${pair##*:}
  echo "=== rerun $model / hepg2  $(date -Is)"
  if PYTHONPATH=src "$HOME/miniconda3/envs/$env/bin/python" -m scfm_probe.probe.run \
      --model "$model" --cell-line hepg2 --anchor mean_control --device cuda; then
    echo "=== ok $model / hepg2  $(date -Is)"
  else
    echo "=== FAILED $model / hepg2  $(date -Is)"
  fi
done
echo "=== rerun finished $(date -Is)"
