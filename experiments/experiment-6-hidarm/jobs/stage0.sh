#!/usr/bin/env bash
# Stage 0: the per-pair hidden-MAGNITUDE arm, off run 5's stored norms.
# No GPU forward passes and no fetch --- only translator fits. It returns a real
# scored result and exercises install -> stage -> score -> analyse end to end
# before any extraction code has to work.
#
#   jobs/stage0.sh pilot   the last block only, both lines
#   jobs/stage0.sh         every sweep layer, both lines
source "$(dirname "$0")/_common.sh"

MODE=${1:-full}
PILOT=""
NAME="stage0"
if [ "$MODE" = "pilot" ]; then PILOT="--pilot"; NAME="stage0_pilot"; fi

for line in hepg2 jurkat; do
  echo "=== $NAME $line  $(date -Is)"
  PYTHONPATH=src "$(py scgpt)" -m scfm_hidarm.stage0 \
      --cell-line "$line" $PILOT --device cuda
  record "$NAME $line" $?
done
finish "$NAME"
