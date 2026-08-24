#!/usr/bin/env bash
# Stage 2a of the protein-model control: build the raw ESM2 gene tables and
# install them into a copy of the bundle, then run one cell of the real sweep
# over them as the end-to-end smoke.
#
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/jobs/run_protein_embed.sh"
#
# The bundle arrives through --input-artifacts (the producer volumes from the
# first wave are long expired). It is copied because the arms are installed by
# writing one more emb__<name>.npz into each cell line's directory, and the
# staged copy is read-only. This job's own volume is what the sweep and the
# transfer jobs then mount, so neither of them re-fetches or re-embeds anything.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
# HuggingFace caches into the worktree, which lives on the per-job volume; the
# container's root disk is a fixed quota and an 11 GB checkpoint does not fit.
export HF_HOME="${PWD}/_hf"
mkdir -p "$HF_HOME"

ART="$SILICO_EXPERIMENT_ARTIFACTS_DIR/protein"
mkdir -p "$ART"

STAGED=$(find "$SILICO_INPUT_ARTIFACTS_DIR" -maxdepth 8 -name MANIFEST.json -path '*bundle*' -print -quit)
if [ -z "$STAGED" ]; then
  echo "[env] no staged bundle under $SILICO_INPUT_ARTIFACTS_DIR"; find "$SILICO_INPUT_ARTIFACTS_DIR" -maxdepth 4 | head -40; exit 1
fi
BUNDLE_SRC=$(dirname "$STAGED")
mkdir -p _root
cp -r "$BUNDLE_SRC" _root/bundle
chmod -R u+w _root/bundle
echo "[env] bundle copied from $BUNDLE_SRC ($(du -sh _root/bundle | cut -f1))"

uv run python - <<'PY'
import sys, torch, transformers
print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} "
      f"transformers={transformers.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
PY

echo "=== gene -> canonical protein sequence ==="
uv run python -m scfm_pfm.protein.uniprot --out "$ART"

echo "=== ESM2 embeddings ==="
uv run python -m scfm_pfm.protein.esm_embed --genes "$ART" --out "$ART"

echo "=== install the arms into the bundle copy ==="
uv run python -m scfm_pfm.protein.install_arm \
  --tables "$ART" --bundle _root/bundle --out "$ART/install_report.json"

echo "=== how close each model's table is to the raw protein table ==="
uv run python -m scfm_pfm.protein.identity \
  --tables "$ART" --bundle _root/bundle --out "$ART/identity.json" || \
  echo "[warn] identity cross-check failed; the sweep does not depend on it"

echo "=== smoke: one cell of the real sweep over the new arms ==="
uv run python -m scfm_pfm.translate.runner \
  --lines hepg2 --scopes full --splits grouped --translators ridge \
  --arms esm2_650m_emb,esm2_3b_emb,coexp --no-shuffle-null --no-cell-eval \
  --bundle _root/bundle --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/protein_smoke"

echo "[done] protein tables in $ART"
ls -la "$ART"
du -sh _root/bundle _hf || true
