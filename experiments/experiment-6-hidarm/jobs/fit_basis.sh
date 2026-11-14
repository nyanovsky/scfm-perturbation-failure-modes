#!/usr/bin/env bash
# The PCA basis, one per (model, cell line, layer), from 64 screened
# perturbations that are DISJOINT from every scored one.
#
# This must complete before jobs/extract.sh: the main pass projects as it goes
# and never writes raw delta_h, which is what keeps 190 GB from becoming 460.
# The leakage check runs inside the pass and RAISES rather than warning --- a
# basis gene that is also a scored perturbation would let a scored perturbation
# shape the basis it is scored through.
source "$(dirname "$0")/_common.sh"

MODELS=${MODELS:-"scgpt scprint transcriptformer scgpt_ft scgpt_ft_frozen"}
LINES=${LINES:-"hepg2 jurkat"}

for line in $LINES; do
  for model in $MODELS; do
    echo "=== basis $model / $line  $(date -Is)"
    PYTHONPATH=src "$(py "$model")" -m scfm_hidarm.extract.run \
        --model "$model" --cell-line "$line" --stage basis --device cuda
    record "basis $model/$line" $?
  done
done
finish "fit_basis"
