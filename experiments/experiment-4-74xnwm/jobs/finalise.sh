#!/bin/sh
# Merge one execution of the sixteen training runs into the tables the report
# reads, and pool the cross-fitted folds.
#
#   sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/finalise.sh
#
# Inputs are the sixteen per-run archives the training jobs left behind plus run
# 3's bundle. Each archive carries that run's score tables, its training
# metadata and its prediction-spread row, which is what keeps this job's staged
# file count at seventeen rather than a hundred and fifty. The weights and the
# prediction matrices are not needed here.
#
# The cross-fit pooling needs the bundle -- the fold assignment, the measured
# effects and the ceilings -- so this runs on a GPU node even though nothing is
# trained.
set -eu
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_gpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
SUF="${SCFM_FT_RUN_SUFFIX:-}"
STAGED="$SILICO_INPUT_ARTIFACTS_DIR/experiments/$SILICO_EXPERIMENT_ID"
SUMMARY="$SILICO_EXPERIMENT_ARTIFACTS_DIR/summary$SUF"
WORK="$PWD/_merge"
ARM="${1:-own_trainable}"
rm -rf "$WORK"
mkdir -p "$SUMMARY" "$WORK"

echo "=== staged inputs ==="
ls "$STAGED"
echo "--- archives: $(ls "$STAGED/packs$SUF"/pack_*.tar.gz 2>/dev/null | wc -l) ---"

for A in "$STAGED/packs$SUF"/pack_*.tar.gz; do
    tar -xzf "$A" -C "$WORK"
    echo "unpacked $(basename "$A")"
done
echo "score tables: $(ls "$WORK/scores" | wc -l);  run directories: $(ls "$WORK/meta" | wc -l)"

echo
echo "=== the pre-registered shuffled-label band, from run 3's bundle alone ==="
for LINE in hepg2 jurkat; do
    python3 -m scfm_ft.coexp_band --cell-line "$LINE" --out "$WORK/scores" \
        --work "$WORK/band" --device cuda
done

echo
echo "=== every perturbation scored by a model that never saw it ==="
for LINE in hepg2 jurkat; do
    echo "--- $LINE ---"
    python3 -m scfm_ft.followup crossfit \
        --scores "$WORK/scores" --checkpoints "$WORK/meta" \
        --cell-line "$LINE" --runs "${ARM},${ARM}__f23,${ARM}__f4" \
        --arm "$ARM" --out "$SUMMARY" --device cuda
done

echo
echo "=== how perturbation-specific each model's own prediction is ==="
python3 -m scfm_ft.followup spread \
    --checkpoints "$WORK/meta" --out "$SUMMARY/prediction_spread.parquet"

echo
echo "=== summary tables ==="
python3 -m scfm_ft.analysis --scores "$WORK/scores" --checkpoints "$WORK/meta" --out "$SUMMARY"
cp "$WORK/scores"/summary_*.json "$SUMMARY"/ 2>/dev/null || true

echo
echo "=== one archive so the report needs one download ==="
tar -czf "$SILICO_EXPERIMENT_ARTIFACTS_DIR/summary$SUF.tar.gz" -C "$SUMMARY" .

echo
echo "=== outputs ==="
ls -la "$SUMMARY"
