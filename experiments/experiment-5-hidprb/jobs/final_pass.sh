#!/usr/bin/env bash
# Everything downstream of the probe blocks, in the only order it can run.
#
#   analysis -> figures -> verify -> claims
#
# The order is not a preference. `figures` refuses to draw over a missing row, so
# it must follow `analysis`; `verify` checks the summary table against the block
# manifests, so it must follow both; and `claims` generates the manifest by
# querying the summary, so it must come last. Running them out of order does not
# produce a wrong answer --- each one fails loudly instead --- but it wastes the
# three hours `analysis` costs.
#
# NOT `set -e`: a failure in one stage should still let the later stages report
# what they can, and the exit status collects them.
set -uo pipefail
cd "$(dirname "$0")/.."
export SCFM_PFM5_DATA_ROOT=${SCFM_PFM5_DATA_ROOT:-/biodata/nyanovsky/projects/scfm-pert-fail-modes/experiment-5-hidprb}

PY="$HOME/miniconda3/envs/scgpt/bin/python"
# silico_figures is the local reconstruction under tools/, not an installed
# package; see envs/README.md before trusting a rebuilt figure.
FIGPATH="src:../../tools/silico-figures-local/src"
FAILED=()

stage() {  # stage <name> <pythonpath> <module> [args...]
  local name=$1 pp=$2; shift 2
  echo "=== $name  $(date -Is)"
  if PYTHONPATH="$pp" "$PY" -m "$@"; then
    echo "=== ok $name  $(date -Is)"
  else
    echo "=== FAILED $name  $(date -Is)"
    FAILED+=("$name")
  fi
}

stage analysis src        scfm_probe.analysis --all
stage figures  "$FIGPATH" scfm_probe.report.figures
stage verify   src        scfm_probe.verify --all
stage claims   src        scfm_probe.claims --write

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== final pass finished with ${#FAILED[@]} failed stage(s): ${FAILED[*]}"
  exit 1
fi
echo "=== final pass finished, all stages ok  $(date -Is)"
