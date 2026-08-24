#!/usr/bin/env bash
# The protein arm across cell lines, including the unseen-gene split.
#
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/jobs/run_protein_transfer.sh"
#
# Both directions, all three transfer scopes and all three transfer modes, the
# same grid the model arms were transferred on. The unseen-gene scope is the one
# that matters most here: it scores only destination perturbations whose gene was
# never perturbed in the training line, where a fixed random code per gene has
# nothing to exploit.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

CE_DIR=/tmp/cell_eval_pkgs
if uv pip install --target "$CE_DIR" "cell-eval==0.8.2" >/tmp/cell_eval_install.log 2>&1; then
  export PYTHONPATH="${PYTHONPATH}:${CE_DIR}"
  echo "[env] cell-eval installed into ${CE_DIR}"
fi

BUNDLE=$(uv run python - <<'PY'
import os, pathlib
from scfm_pfm import paths
found = paths.find_bundle(pathlib.Path(os.environ.get("SILICO_REUSED_VOLUME_DIR", "")))
if found is None:
    raise SystemExit("no bundle on the mounted volume")
print(found)
PY
)
ART="$SILICO_EXPERIMENT_ARTIFACTS_DIR"
echo "[env] bundle=$BUNDLE"

echo "=== the protein arms, both directions, three scopes ==="
uv run python -m scfm_pfm.translate.runner --bundle "$BUNDLE" --transfer \
  --lines hepg2,jurkat --translators mlp,ridge --transfer-scopes full,common,unseen \
  --arms esm2_650m_emb,esm2_3b_emb,null_randgene_d2560 \
  --out "$ART/protein_transfer"

echo "=== arms the main transfer already scored, re-run as this job's reference ==="
uv run python -m scfm_pfm.translate.runner --bundle "$BUNDLE" --transfer \
  --lines hepg2,jurkat --translators mlp --transfer-scopes full,unseen \
  --arms coexp,scgpt_emb,transcriptformer_emb \
  --out "$ART/protein_transfer_reference"

echo "[done]"
ls -la "$ART/protein_transfer" "$ART/protein_transfer_reference"
