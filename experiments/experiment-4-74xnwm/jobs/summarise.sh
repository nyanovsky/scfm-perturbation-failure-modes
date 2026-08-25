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

# The training jobs and the follow-up job both write direct_* tables. Same
# filename for the same (line, arm, baseline), so copying the follow-up over
# the staged copy replaces rather than duplicates.
echo "=== merging ==="
cp "$STAGED/scores"/*.parquet "$MERGED"/
cp "$PACK/followup"/*.parquet "$MERGED"/ 2>/dev/null || true
echo "merged: $(ls "$MERGED" | wc -l) files"

echo "=== summary tables ==="
python3 -m scfm_ft.analysis \
    --scores "$MERGED" --checkpoints "$PACK/checkpoint_meta" --out "$SUMMARY"

# the cross-fit tables are not part of the arm grid, so they ride along whole
cp "$PACK/followup"/crossfit_*.parquet "$PACK/followup"/crossfit_*.json \
   "$SUMMARY"/ 2>/dev/null || true
cp "$PACK/followup"/summary_*.json "$SUMMARY"/ 2>/dev/null || true

echo "=== one archive so the summary needs one download ==="
tar -czf "$SILICO_EXPERIMENT_ARTIFACTS_DIR/summary.tar.gz" -C "$SUMMARY" .

echo
echo "=== outputs ==="
ls -la "$SUMMARY"
