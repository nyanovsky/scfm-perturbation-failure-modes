#!/usr/bin/env bash
# Resume the mean-control sweep after the TranscriptFormer/HepG2 abort.
#
# scGPT and scPRINT already have full-coverage HepG2 blocks on disk (441 s and
# 210 s), so re-running them would only overwrite good output with identical
# output. This picks up at the block that died and finishes the grid.
set -uo pipefail
cd "$(dirname "$0")/.."
export SCFM_PFM5_DATA_ROOT=${SCFM_PFM5_DATA_ROOT:-/biodata/nyanovsky/projects/scfm-pert-fail-modes/experiment-5-hidprb}
FAILED=()

run() {  # run <env> <model> <cell_line>
  echo "=== sweep $2 / $3 (env $1)  $(date -Is)"
  if PYTHONPATH=src "$HOME/miniconda3/envs/$1/bin/python" -m scfm_probe.probe.run \
      --model "$2" --cell-line "$3" --anchor mean_control --device cuda; then
    echo "=== ok $2 / $3  $(date -Is)"
  else
    echo "=== FAILED $2 / $3  $(date -Is)  (continuing)"
    FAILED+=("$2/$3")
  fi
}

run transcriptformer_env transcriptformer hepg2
run scgpt                scgpt            jurkat
run scprint_env          scprint          jurkat
run transcriptformer_env transcriptformer jurkat

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== resume finished with ${#FAILED[@]} failed block(s): ${FAILED[*]}"
  exit 1
fi
echo "=== resume finished, all blocks ok  $(date -Is)"
