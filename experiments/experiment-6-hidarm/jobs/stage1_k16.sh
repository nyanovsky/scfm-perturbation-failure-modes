#!/usr/bin/env bash
# The k = 16 rung at each model's primary layer, both lines.
#
# `K_PRIMARY` was re-registered from 16 to 64 on 2026-08-31 --- deliberately,
# after the captured-energy diagnostic was read and BEFORE any arm was scored,
# so no answer influenced it. It is still weaker than the original
# pre-registration, and the claims discipline is explicit: **quote the k = 16
# rung beside the k = 64 verdict**, so a reader can see that the primary rung
# moved and what the other rung said.
#
# This is the cheapest reading that discharges that: the primary layer only, in
# the pre-registered cell, for each of the four models on both lines. The
# k = 16 arm is a column PREFIX of the stored k = 64 block and is taken by
# slicing it --- never by re-projecting, which differs by 1-3 float32 ulps
# (README s1).
#
#   jobs/stage1_k16.sh              both lines
#   jobs/stage1_k16.sh hepg2        one line
source "$(dirname "$0")/_common.sh"

LINES=${*:-hepg2 jurkat}
MODELS=${MODELS:-scgpt scgpt_ft scgpt_ft_frozen scprint}
LOGDIR="$SCFM_PFM6_DATA_ROOT/logs"
mkdir -p "$LOGDIR"

for line in $LINES; do
  echo "=== stage1 k=16 $line  ${MODELS// /,}  $(date -Is)"
  PYTHONPATH=src "$(py scgpt)" -m scfm_hidarm.stage1 \
      --cell-line "$line" --models "${MODELS// /,}" --pilot --k 16 \
      --device cuda --skip-existing \
      >> "$LOGDIR/stage1_k16_${line}.log" 2>&1
  record "stage1 k=16 $line" $?
done
finish "stage1 k=16"
