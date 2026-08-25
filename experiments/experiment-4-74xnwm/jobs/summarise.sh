#!/bin/sh
# Merge every score table and build the summary the report reads. CPU only:
# nothing here touches a model or the bundle.
#
#   sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/summarise.sh
#
# Inputs are the four summary-level score tables from each training run
# (restricted, contrasts, runs, direct) plus the single archive the follow-up
# job left behind, which carries its own score tables and every run's training
# metadata. The per-perturbation tables are deliberately not staged: they are
# large, they are not read here, and staging is capped at 128 files.
set -eu
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_cpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
STAGED="$SILICO_INPUT_ARTIFACTS_DIR/experiments/$SILICO_EXPERIMENT_ID"
SUMMARY="$SILICO_EXPERIMENT_ARTIFACTS_DIR/summary"
MERGED="$PWD/scratch/merged_scores"
PACK="$PWD/scratch/unpacked"
mkdir -p "$SUMMARY" "$MERGED" "$PACK"

echo "=== staged inputs ==="
ls "$STAGED"
echo "--- score tables: $(ls "$STAGED/scores" | wc -l) files ---"

echo "=== unpacking the follow-up archive ==="
tar -xzf "$STAGED/followup_pack.tar.gz" -C "$PACK"
find "$PACK" -maxdepth 2 -type d | sort | head -20
echo "unpacked $(find "$PACK" -type f | wc -l) files"

# The tagged control runs were scored twice: once inside the training job,
# where a name collision dropped their response arms, and again from their
# saved weights. The second reading is unpacked last so it replaces the first,
# file for file, under the same names.
#
# HepG2 arrives as one archive. Jurkat needed three jobs -- the first ran out
# of wall clock after the two cross-fitted re-reads -- so its pieces arrive as
# two archives plus the loose files the first job left in the artifact tree.
RESC="$PWD/scratch/rescored"
mkdir -p "$RESC"
for A in "$STAGED"/rescore_pack_*.tar.gz "$STAGED"/rescore_tail_*.tar.gz \
         "$STAGED"/crossfit_pack_*.tar.gz; do
    [ -f "$A" ] || continue
    tar -xzf "$A" -C "$RESC"
    echo "unpacked $A"
done
for LINE in hepg2 jurkat; do
    if [ -d "$STAGED/rescore/$LINE" ]; then
        cp "$STAGED/rescore/$LINE"/* "$RESC"/ 2>/dev/null || true
        echo "copied loose files from rescore/$LINE"
    fi
done
ls "$RESC"
for LINE in hepg2 jurkat; do
    for TAG in f23 f4 upstream lastepoch; do
        F="$RESC/restricted_${LINE}__own_trainable__${TAG}.parquet"
        [ -f "$F" ] || echo "WARNING: missing $(basename "$F")" >&2
    done
    [ -f "$RESC/crossfit_${LINE}__own_trainable.json" ] \
        || echo "WARNING: no cross-fit pooling for $LINE" >&2
done

# The training jobs and the follow-up job both write direct_* tables. Same
# filename for the same (line, arm, baseline), so copying the follow-up over
# the staged copy replaces rather than duplicates.
echo "=== merging ==="
cp "$STAGED/scores"/*.parquet "$MERGED"/
cp "$PACK/followup"/*.parquet "$MERGED"/ 2>/dev/null || true
for F in "$RESC"/restricted_*.parquet "$RESC"/contrasts_*.parquet "$RESC"/runs_*.parquet; do
    [ -f "$F" ] && cp "$F" "$MERGED"/
done
# The direct-prediction tables were re-scored after the fold-mean correction,
# so they are unpacked LAST and replace, file for file, every earlier direct_*
# table for the same (line, arm, baseline). Anything left from before the
# correction would otherwise still be in the merge.
FIX="$PWD/scratch/direct_fix"
if [ -f "$STAGED/direct_fix.tar.gz" ]; then
    mkdir -p "$FIX"
    tar -xzf "$STAGED/direct_fix.tar.gz" -C "$FIX"
    echo "--- corrected direct tables: $(ls "$FIX"/direct_*.parquet 2>/dev/null | wc -l) ---"
    rm -f "$MERGED"/direct_*.parquet
    cp "$FIX"/direct_*.parquet "$MERGED"/
    cp "$FIX"/direct_rescored.json "$SUMMARY"/ 2>/dev/null || true
else
    echo "WARNING: no direct_fix.tar.gz staged; direct tables are the uncorrected ones" >&2
fi
echo "merged: $(ls "$MERGED" | wc -l) files"

echo "=== how perturbation-specific each model's own prediction is ==="
python3 -m scfm_ft.followup spread \
    --checkpoints "$STAGED/checkpoints" --out "$SUMMARY/prediction_spread.parquet"

echo "=== summary tables ==="
python3 -m scfm_ft.analysis \
    --scores "$MERGED" --checkpoints "$PACK/checkpoint_meta" --out "$SUMMARY"

# the cross-fit tables are not part of the arm grid, so they ride along whole
cp "$STAGED/scores"/summary_*.json "$SUMMARY"/ 2>/dev/null || true
cp "$PACK/followup"/summary_*.json "$SUMMARY"/ 2>/dev/null || true
cp "$RESC"/crossfit_*.parquet "$RESC"/crossfit_*.json "$SUMMARY"/ 2>/dev/null || true
cp "$RESC"/summary_*.json "$SUMMARY"/ 2>/dev/null || true

echo "=== one archive so the summary needs one download ==="
tar -czf "$SILICO_EXPERIMENT_ARTIFACTS_DIR/summary.tar.gz" -C "$SUMMARY" .

echo
echo "=== outputs ==="
ls -la "$SUMMARY"
