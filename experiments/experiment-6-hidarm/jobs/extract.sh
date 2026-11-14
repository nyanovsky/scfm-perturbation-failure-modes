#!/usr/bin/env bash
# The main pass: every scored perturbation, projected as it goes, all sweep
# layers, both the fitted-PCA block and the random-basis bracket.
#
# Sequential by design: one 3090. TranscriptFormer is the long pole --- run 2
# measured 4,621 / 6,232 s per model-line on an H100 against scGPT's 372 / 643,
# and run 6 runs 16 input rows per gene against run 5's 6. Order is cheapest
# first so a failure late in the queue costs the least already-usable output.
#
# Requires jobs/fit_basis.sh to have completed for the same (model, line):
# extract_line loads the basis and refuses a shape it did not expect.
source "$(dirname "$0")/_common.sh"

MODELS=${MODELS:-"scgpt scgpt_ft scgpt_ft_frozen scprint transcriptformer"}
LINES=${LINES:-"hepg2 jurkat"}

for line in $LINES; do
  for model in $MODELS; do
    echo "=== extract $model / $line  $(date -Is)"
    PYTHONPATH=src "$(py "$model")" -m scfm_hidarm.extract.run \
        --model "$model" --cell-line "$line" --stage main --device cuda
    record "extract $model/$line" $?
  done
done
finish "extract"
