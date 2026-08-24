#!/usr/bin/env bash
# Install the ESM2 tables into a bundle, measure how close every gene table is
# to them, and run one cell of the real sweep over the new arms as the smoke.
#
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/jobs/run_protein_install.sh"
#
# The tables come from the embedding job through the artifact store, so this
# does not re-run ESM2. This job's volume is what the per-line sweep jobs and
# the transfer job mount, so the bundle they read is this one, installed once.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

STAGED=$(find "$SILICO_INPUT_ARTIFACTS_DIR" -maxdepth 8 -name MANIFEST.json -path '*bundle*' -print -quit)
TABLES=$(find "$SILICO_INPUT_ARTIFACTS_DIR" -maxdepth 8 -name 'esm2_3b.npz' -print -quit)
if [ -z "$STAGED" ] || [ -z "$TABLES" ]; then
  echo "[env] bundle=$STAGED tables=$TABLES"; find "$SILICO_INPUT_ARTIFACTS_DIR" -maxdepth 4 | head -40; exit 1
fi
TABLE_DIR=$(dirname "$TABLES")
mkdir -p _root
cp -r "$(dirname "$STAGED")" _root/bundle
chmod -R u+w _root/bundle
echo "[env] bundle from $(dirname "$STAGED"), tables from $TABLE_DIR"

ART="$SILICO_EXPERIMENT_ARTIFACTS_DIR/protein"
mkdir -p "$ART"

echo "=== install the arms into the bundle copy ==="
uv run python -m scfm_pfm.protein.install_arm \
  --tables "$TABLE_DIR" --bundle _root/bundle --out "$ART/install_report.json"

echo "=== how close each model's table is to the raw protein table ==="
uv run python -m scfm_pfm.protein.identity \
  --tables "$TABLE_DIR" --bundle _root/bundle --out "$ART/identity.json"

echo "=== smoke: one cell of the real sweep over the new arms ==="
uv run python -m scfm_pfm.translate.runner \
  --lines hepg2 --scopes full --splits grouped --translators ridge \
  --arms esm2_650m_emb,esm2_3b_emb,transcriptformer_emb,coexp \
  --no-shuffle-null --no-cell-eval \
  --bundle _root/bundle --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/protein_smoke"

echo "[done]"
ls -la "$ART" "$SILICO_EXPERIMENT_ARTIFACTS_DIR/protein_smoke"
