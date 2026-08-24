#!/usr/bin/env bash
# The two-table arm: scGPT's learned gene table concatenated with the raw
# ESM2-3B protein table, through the identical translator, folds, scoring and
# nulls that every other arm went through.
#
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/jobs/run_concat.sh"
#
# Inputs arrive through --input-artifacts: the bundle (targets, control-cell
# statistics, scGPT's table) and the ESM2 tables the protein job built. The
# producer volumes are long expired, so nothing is mounted. Only the FULL scope
# is run: the common scope is defined by the intersection of the four models'
# response blocks, and those blocks are not staged here.
#
# Three runner invocations, each into its own directory:
#   concat_smoke/     one cell, ridge, fail-fast before the grid is paid for
#   concat_sweep/     the new arms over both lines, both splits, both translators
#   concat_reference/ three arms the main sweep already scored, re-run here so
#                     any difference is the job and the staged bundle, not the
#                     feature. Kept out of the merge.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

ART="$SILICO_EXPERIMENT_ARTIFACTS_DIR"
NEW_ARMS="scgpt_esm2_3b_emb,null_randgene_d6144"
REFERENCE="scgpt_emb,esm2_3b_emb,coexp"

STAGED=$(find "$SILICO_INPUT_ARTIFACTS_DIR" -maxdepth 8 -name MANIFEST.json -path '*bundle*' -print -quit)
TABLES=$(find "$SILICO_INPUT_ARTIFACTS_DIR" -maxdepth 8 -name 'esm2_3b.npz' -print -quit)
if [ -z "$STAGED" ] || [ -z "$TABLES" ]; then
  echo "[env] missing staged inputs under $SILICO_INPUT_ARTIFACTS_DIR"
  find "$SILICO_INPUT_ARTIFACTS_DIR" -maxdepth 5 | head -40
  exit 1
fi
BUNDLE_SRC=$(dirname "$STAGED")
TABLE_DIR=$(dirname "$TABLES")
mkdir -p _root
cp -r "$BUNDLE_SRC" _root/bundle
chmod -R u+w _root/bundle
echo "[env] bundle from $BUNDLE_SRC ($(du -sh _root/bundle | cut -f1)), tables from $TABLE_DIR"

uv run python - <<'PY'
import torch
print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
PY

CE_DIR=/tmp/cell_eval_pkgs
if uv pip install --target "$CE_DIR" "cell-eval==0.8.2" >/tmp/cell_eval_install.log 2>&1; then
  export PYTHONPATH="${PYTHONPATH}:${CE_DIR}"
  echo "[env] cell-eval installed into ${CE_DIR}"
else
  echo "[env] cell-eval install FAILED; continuing with our own discrimination only"
  tail -n 15 /tmp/cell_eval_install.log || true
fi

echo "=== install the ESM2 tables into the bundle copy ==="
uv run python -m scfm_pfm.protein.install_arm \
  --tables "$TABLE_DIR" --bundle _root/bundle --out "$ART/concat/install_report.json"
ls -la _root/bundle/hepg2/emb__*.npz

echo "=== smoke: one cell of the real sweep over the new arms ==="
uv run python -m scfm_pfm.translate.runner --bundle _root/bundle \
  --lines hepg2 --scopes full --splits grouped --translators ridge \
  --arms "$NEW_ARMS" --no-shuffle-null --no-cell-eval \
  --out "$ART/concat_smoke"

echo "=== the two-table arm and its width-matched null over the grid ==="
uv run python -m scfm_pfm.translate.runner --bundle _root/bundle \
  --lines hepg2,jurkat --scopes full --splits grouped,random --translators mlp,ridge \
  --arms "$NEW_ARMS" --no-shuffle-null \
  --out "$ART/concat_sweep"

echo "=== arms the main sweep already scored, re-run as this job's reference ==="
uv run python -m scfm_pfm.translate.runner --bundle _root/bundle \
  --lines hepg2,jurkat --scopes full --splits grouped --translators mlp \
  --arms "$REFERENCE" --no-shuffle-null \
  --out "$ART/concat_reference"

echo "[done]"
ls -la "$ART/concat_sweep" "$ART/concat_reference"
