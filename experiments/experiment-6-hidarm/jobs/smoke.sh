#!/usr/bin/env bash
# 8 genes, layers 0 / first / last, HepG2, every model key: C1-C3 over REAL
# output in minutes. Never enough to read a number from --- enough to find out
# that a hook is mis-attributed, a slot is built from the wrong pair of rows, or
# a block is unreadable by run 3, before nine GPU hours are spent on it.
source "$(dirname "$0")/_common.sh"

LINE=${1:-hepg2}
for model in scgpt scprint transcriptformer scgpt_ft scgpt_ft_frozen; do
  echo "=== smoke basis $model / $LINE  $(date -Is)"
  PYTHONPATH=src "$(py "$model")" -m scfm_hidarm.extract.run \
      --model "$model" --cell-line "$LINE" --stage basis --smoke --device cuda
  record "basis $model/$LINE" $?
  echo "=== smoke main $model / $LINE  $(date -Is)"
  PYTHONPATH=src "$(py "$model")" -m scfm_hidarm.extract.run \
      --model "$model" --cell-line "$LINE" --stage main --smoke --device cuda
  record "main $model/$LINE" $?
done
finish "smoke"
