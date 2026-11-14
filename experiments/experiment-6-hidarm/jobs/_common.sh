# Shared by every run-6 job. Sourced, never executed.
#
# NOT `set -e`. A crash in one (model, cell line) must not take the later ones
# with it: run 5 lost three Jurkat blocks to a TranscriptFormer abort at gene 300
# before this was fixed. Each combination is independent, so a failure is
# recorded and the sweep moves on; the exit status still reports it.
set -uo pipefail

cd "$(dirname "$0")/.."

# Heavy bytes go to /biodata through paths.py's env ladder, never to the repo
# and never to run 5's root --- run 5's analysis.discover() globs anchor=* under
# probe/<model>/<line> and would ingest a run-6 tree written beneath it.
export SCFM_PFM6_DATA_ROOT=${SCFM_PFM6_DATA_ROOT:-/biodata/nyanovsky/projects/scfm-pert-fail-modes/experiment-6-hidarm}
# run 4's published scGPT release, in the layout run 4's loader hardcodes
export SCFM_FT_CKPT=${SCFM_FT_CKPT:-/biodata/nyanovsky/projects/scfm-pert-fail-modes/ckpt_cache/scgpt}
# the validation-Pearson execution, which is the one run 4 reports. Unset would
# silently read the earlier masked-MSE tree.
export SCFM_FT_RUN_SUFFIX=${SCFM_FT_RUN_SUFFIX:-_pearson}

# One process scores dozens of bundles in sequence and each holds 3.6-5.9 GB of
# response block on a 24 GB card, so the allocator sees a long series of large
# allocations of slightly different shapes. Expandable segments let it grow and
# reuse one region instead of stranding the gaps between them; the 2026-09-01
# wave died on fragmentation with 1.10 GiB free and 22.45 GiB in use.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ENVS_ROOT=${ENVS_ROOT:-$HOME/miniconda3/envs}
FAILED=()

# env_for <model> --- the conda env each checkpoint's pinned stack lives in.
# Mirrors config.MODEL_ENV; the two fine-tuned keys run in run 5's scgpt env,
# verified on all four checkpoints before any extraction (scratch/probe_ft_checkpoint.py).
env_for() {
  case "$1" in
    scgpt|scgpt_ft|scgpt_ft_frozen) echo scgpt ;;
    scprint)                        echo scprint_env ;;
    transcriptformer)               echo transcriptformer_env ;;
    *) echo "unknown model $1" >&2; return 1 ;;
  esac
}

py() {  # py <model> -- the interpreter that model's stack is pinned in
  echo "$ENVS_ROOT/$(env_for "$1")/bin/python"
}

record() {  # record <label> <status>
  if [ "$2" -eq 0 ]; then
    echo "=== ok $1  $(date -Is)"
  else
    echo "=== FAILED $1  $(date -Is)  (continuing; see failure.txt in its output dir)"
    FAILED+=("$1")
  fi
}

finish() {  # finish <what>
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "=== $1 finished with ${#FAILED[@]} failure(s): ${FAILED[*]}"
    exit 1
  fi
  echo "=== $1 finished, all ok  $(date -Is)"
}
