#!/usr/bin/env bash
# One cell line's protein-arm sweep: the same grid every other arm was scored on.
#
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/jobs/run_protein_line.sh" hepg2
#
# Three runner invocations, each into its own directory:
#   protein_sweep/<line>      the new arms, over both scopes, both splits, both
#                             translators. New arm names, so these merge into the
#                             main sweep without touching a scored row.
#   protein_shuffle/<line>    shuffled-label draws for the protein arm, the same
#                             control the winning arms got, in the one cell the
#                             null band is read from.
#   protein_reference/<line>  four arms the main sweep already scored, re-run
#                             here. Kept separate: merging them would overwrite
#                             the scored rows with re-run ones.
set -euo pipefail
LINE="${1:?usage: run_protein_line.sh <cell line>}"
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

CE_DIR=/tmp/cell_eval_pkgs
if uv pip install --target "$CE_DIR" "cell-eval==0.8.2" >/tmp/cell_eval_install.log 2>&1; then
  export PYTHONPATH="${PYTHONPATH}:${CE_DIR}"
  echo "[env] cell-eval installed into ${CE_DIR}"
else
  echo "[env] cell-eval install FAILED; continuing with our own discrimination only"
  tail -n 15 /tmp/cell_eval_install.log || true
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
echo "[env] line=$LINE bundle=$BUNDLE"
ls -la "$BUNDLE/$LINE"/emb__*.npz

uv run python - <<'PY'
import torch
print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
PY

NEW_ARMS="esm2_650m_emb,esm2_3b_emb,null_randgene_d2560"
SHUFFLED="esm2_3b_emb+shuffled_labels_d0,esm2_3b_emb+shuffled_labels_d1"
SHUFFLED="${SHUFFLED},esm2_3b_emb+shuffled_labels_d2,esm2_3b_emb+shuffled_labels_d3"
SHUFFLED="${SHUFFLED},esm2_3b_emb+shuffled_labels_d4"
REFERENCE="coexp,scgpt_emb,scprint_emb,transcriptformer_emb"

echo "=== [$LINE] the protein arms over the full grid ==="
uv run python -m scfm_pfm.translate.runner --bundle "$BUNDLE" \
  --lines "$LINE" --scopes full,common --splits grouped,random --translators mlp,ridge \
  --arms "$NEW_ARMS" --no-shuffle-null --out "$ART/protein_sweep/$LINE"

echo "=== [$LINE] shuffled-label draws for the protein arm ==="
uv run python -m scfm_pfm.translate.runner --bundle "$BUNDLE" \
  --lines "$LINE" --scopes full --splits grouped --translators mlp \
  --extra-shuffle-arms esm2_3b_emb --arms "$SHUFFLED" \
  --out "$ART/protein_shuffle/$LINE"

echo "=== [$LINE] arms the main sweep already scored, re-run as this job's reference ==="
uv run python -m scfm_pfm.translate.runner --bundle "$BUNDLE" \
  --lines "$LINE" --scopes full --splits grouped --translators mlp \
  --arms "$REFERENCE" --no-shuffle-null --out "$ART/protein_reference/$LINE"

echo "[done] $LINE"
ls -la "$ART/protein_sweep/$LINE" "$ART/protein_shuffle/$LINE" "$ART/protein_reference/$LINE"
