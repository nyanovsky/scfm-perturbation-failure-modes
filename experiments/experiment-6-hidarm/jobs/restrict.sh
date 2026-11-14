#!/usr/bin/env bash
# Re-read the fine-tuned arms --- and frozen scGPT --- on folds 0 and 1.
#
# Run 4's checkpoints trained on folds 2, 3 and 4, so the five-fold stage-1 rows
# for scgpt_ft and scgpt_ft_frozen are training numbers for three folds in five
# and MAY NOT BE QUOTED. This produces the rows that may. Frozen scGPT is
# restricted at the same time and at every layer, because a two-fold fine-tuned
# score beside a five-fold frozen score is not a comparison.
#
# No fits: run 6 keeps the held-out prediction for every arm and every cell, so
# this is a re-READING of predictions that already exist, through run 4's own
# restrict_scores and restricted_ceilings.
#
#   jobs/restrict.sh              both lines
#   jobs/restrict.sh hepg2        one line
source "$(dirname "$0")/_common.sh"

LINES=${*:-hepg2 jurkat}
LOGDIR="$SCFM_PFM6_DATA_ROOT/logs"
mkdir -p "$LOGDIR"

for line in $LINES; do
  echo "=== restrict $line  $(date -Is)"
  PYTHONPATH=src "$(py scgpt)" -m scfm_hidarm.restrict_stage1 \
      --cell-line "$line" --sweep stage1 --device cuda \
      >> "$LOGDIR/restrict_${line}.log" 2>&1
  record "restrict $line" $?
done
finish "restrict"
