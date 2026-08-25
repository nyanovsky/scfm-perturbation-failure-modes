#!/bin/sh
# Everything after the sixteen training runs that still needs a GPU and the
# bundle:
#   1. rescore each model's own predicted response against the measured control
#      profile, the baseline the published evaluation uses
#   2. pool the cross-fitted arms so every perturbation is scored by a model
#      that never saw it
#   3. read the response features off the LAST epoch instead of the selected one
#
#   sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/followup.sh <jurkat runs> <hepg2 runs> [arm]
#
# Staging is capped at 128 files, so this job takes only what these three steps
# need: the bundle, the saved predictions, the per-run metadata, the two final
# state dicts, and the per-perturbation tables of the cross-fitted runs. The
# merge and the summary tables are a second, CPU-only job; this one leaves a
# single archive for it to pick up.
set -eu
JURKAT_RUNS="$1"
HEPG2_RUNS="$2"
ARM="${3:-own_trainable}"
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_gpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
STAGED="$SILICO_INPUT_ARTIFACTS_DIR/experiments/$SILICO_EXPERIMENT_ID"
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/followup"
mkdir -p "$OUT"

echo "=== staged inputs ==="
ls "$STAGED"
echo "--- score tables: $(ls "$STAGED/scores" 2>/dev/null | wc -l) files ---"
echo "--- checkpoint dirs ---"
ls "$STAGED/checkpoints"

echo
echo "=== 1. the model's own prediction, against the measured control profile ==="
python3 -m scfm_ft.followup direct \
    --checkpoints "$STAGED/checkpoints" --out "$OUT" --device cuda

echo
echo "=== 2. pooling the cross-fitted arms ==="
for LINE in jurkat hepg2; do
    RUNS="$JURKAT_RUNS"
    [ "$LINE" = "hepg2" ] && RUNS="$HEPG2_RUNS"
    echo "--- $LINE: $RUNS ---"
    python3 -m scfm_ft.followup crossfit \
        --scores "$STAGED/scores" --checkpoints "$STAGED/checkpoints" \
        --cell-line "$LINE" --runs "$RUNS" --arm "$ARM" --out "$OUT" --device cuda
done

echo
echo "=== 3. the last epoch, which the selection rule rejected ==="
# A control on a choice the recipe makes, not on the result. If it fails the
# rest of the run is still valid, so it must not take the job down with it.
for LINE in hepg2 jurkat; do
    echo "--- $LINE ---"
    if python3 -m scfm_ft.followup lastepoch \
        --checkpoints "$STAGED/checkpoints" --cell-line "$LINE" --arm "$ARM" \
        --scores-out "$OUT" --work "$PWD/scratch/lastepoch_$LINE" --device cuda
    then
        echo "lastepoch $LINE ok"
    else
        echo "WARNING: lastepoch $LINE failed; continuing" >&2
    fi
done

echo
echo "=== packing one archive for the summary job ==="
# The per-run training metadata rides along so the summary job does not have to
# stage thirty-two more files against its own cap.
PACK="$PWD/scratch/pack"
rm -rf "$PACK"
mkdir -p "$PACK/followup" "$PACK/checkpoint_meta"
cp "$OUT"/* "$PACK/followup"/ 2>/dev/null || true
for D in "$STAGED/checkpoints"/*; do
    R=$(basename "$D")
    mkdir -p "$PACK/checkpoint_meta/$R"
    for F in train_config.json loss_history.json; do
        [ -f "$D/$F" ] && cp "$D/$F" "$PACK/checkpoint_meta/$R/$F"
    done
done
tar -czf "$SILICO_EXPERIMENT_ARTIFACTS_DIR/followup_pack.tar.gz" -C "$PACK" .
echo "packed $(find "$PACK" -type f | wc -l) files"

echo
echo "=== outputs ==="
ls -la "$OUT"
ls -la "$SILICO_EXPERIMENT_ARTIFACTS_DIR/followup_pack.tar.gz"
