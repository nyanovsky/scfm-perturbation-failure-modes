#!/usr/bin/env bash
# The primary condition: full in-scope coverage (524 HepG2, 705 Jurkat) at the
# mean-control anchor, over each model's layer sweep.
#
# mean_control is the anchor every scored arm in the paper is read at, so a null
# here speaks to the published result and a null anywhere else does not.
#
# Sequential by design: one 3090. Run 2's measured single-H100 per-anchor costs
# are the scaling baseline (docs/INDEX.md section 8) --- scGPT and scPRINT are
# minutes, TranscriptFormer is the long pole.
# NOT `set -e`. A crash in one (model, cell line) used to kill the whole script and
# take every later block with it --- a TranscriptFormer abort at gene 300 of 524 cost
# the three Jurkat blocks that had not started yet. Each combination is independent,
# so a failure is recorded and the sweep moves on; the exit status still reports it.
set -uo pipefail
FAILED=()
cd "$(dirname "$0")/.."
export SCFM_PFM5_DATA_ROOT=${SCFM_PFM5_DATA_ROOT:-/biodata/nyanovsky/projects/scfm-pert-fail-modes/experiment-5-hidprb}

run() {  # run <env> <model> <cell_line>
  echo "=== sweep $2 / $3 (env $1)  $(date -Is)"
  if PYTHONPATH=src "$HOME/miniconda3/envs/$1/bin/python" -m scfm_probe.probe.run \
      --model "$2" --cell-line "$3" --anchor mean_control --device cuda; then
    echo "=== ok $2 / $3  $(date -Is)"
  else
    echo "=== FAILED $2 / $3  $(date -Is)  (continuing; see failure.txt in its block)"
    FAILED+=("$2/$3")
  fi
}

for line in hepg2 jurkat; do
  run scgpt                scgpt            "$line"
  run scprint_env          scprint          "$line"
  run transcriptformer_env transcriptformer "$line"
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== sweep finished with ${#FAILED[@]} failed block(s): ${FAILED[*]}"
  exit 1
fi
echo "=== sweep finished, all blocks ok  $(date -Is)"
