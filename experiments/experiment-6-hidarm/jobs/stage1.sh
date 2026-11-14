#!/usr/bin/env bash
# Stage 1: the hidden-state VECTOR arm --- 10 projected slots per pair, scored
# through run 3's unmodified rule. No forward passes: the blocks are already on
# disk from the 2026-08-31 extraction wave, so this is translator fits only.
#
# One unit is one (cell line, model, layer). Each writes its own table before
# the next begins, because run 3's score_bundle persists NOTHING until every
# requested split finishes --- which is how the stage-0 pilot lost its per-arm
# scores. A unit's staged blocks (~14-22 GB) are deleted once its table is on
# disk unless --keep-blocks.
#
# TranscriptFormer is deliberately excluded: it is the deferred model, and its
# hidperm null is unscoreable as written (README s7 --- the permutation relocates
# its undefined self entry off the masked self column).
#
#   jobs/stage1.sh                 both lines, four models, every sweep layer
#   jobs/stage1.sh hepg2           one line
#   MODELS=scgpt jobs/stage1.sh    one model
#   PILOT=1 jobs/stage1.sh         the pre-registered primary layer only
source "$(dirname "$0")/_common.sh"

LINES=${*:-hepg2 jurkat}
MODELS=${MODELS:-scgpt scgpt_ft scgpt_ft_frozen scprint}
PILOT_FLAG=""
[ "${PILOT:-0}" = "1" ] && PILOT_FLAG="--pilot"
LOGDIR="$SCFM_PFM6_DATA_ROOT/logs"
mkdir -p "$LOGDIR"

for line in $LINES; do
  # run 3's reference arms once per line: they are how run 6's absolute scores
  # are readable at all, and they are never merged into the run table
  echo "=== stage1 references $line  $(date -Is)"
  PYTHONPATH=src "$(py scgpt)" -m scfm_hidarm.stage1 \
      --cell-line "$line" --references --device cuda \
      >> "$LOGDIR/stage1_${line}_references.log" 2>&1
  record "stage1 references $line" $?

  echo "=== stage1 $line  ${MODELS// /,}  $(date -Is)"
  PYTHONPATH=src "$(py scgpt)" -m scfm_hidarm.stage1 \
      --cell-line "$line" --models "${MODELS// /,}" $PILOT_FLAG --device cuda \
      --skip-existing \
      >> "$LOGDIR/stage1_${line}.log" 2>&1
  record "stage1 $line" $?
done
finish "stage1"
