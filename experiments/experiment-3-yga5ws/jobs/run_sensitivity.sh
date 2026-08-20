#!/usr/bin/env bash
# Anchor and form sensitivity, the pre-registration's secondary question:
# does the uniform-virtual anchor beat the on-distribution one (UGRN's mechanism
# claim), and do masked features differ from visible ones? HepG2 only, full
# scope, grouped split, MLP — exactly the cell config.py gates the step to.
#
# The first sweep only ever fetched anchor=mean_control, so the sensitivity arms
# resolved to nothing and were silently skipped. The first sensitivity job
# fetched the three anchors and scored them, but excluded no-op queries by
# deleting the whole dose, which also deleted every other perturbation's real
# response at that value — the 0 CP10K value for all three models, and 64 CP10K
# as well for TranscriptFormer. This job re-derives the features from the same
# bytes with the exclusion done per (perturbed gene, value), and rescores.
#
# It mounts the earlier sensitivity job's volume for two things: the bundle it
# built (co-expression, node statistics, embeddings, targets) and the artifact
# mirror it downloaded, so nothing is re-fetched and the features are rebuilt
# from the same files.
set -euo pipefail
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
SENS_ROOT="${PWD}/_sensroot"

CE_DIR=/tmp/cell_eval_pkgs
if uv pip install --target "$CE_DIR" "cell-eval==0.8.2" >/tmp/cell_eval_install.log 2>&1; then
  export PYTHONPATH="${PYTHONPATH}:${CE_DIR}"
  echo "[env] cell-eval installed into ${CE_DIR}"
else
  echo "[env] cell-eval install FAILED; continuing with our own discrimination only"
  tail -n 15 /tmp/cell_eval_install.log || true
fi

uv run python - <<'PY'
import os, pathlib, shutil
from scfm_pfm import config as C, paths

mount = pathlib.Path(os.environ.get("SILICO_REUSED_VOLUME_DIR", ""))
src = paths.find_bundle(mount)
mirror = paths.find_mirror(mount, C.RUN2)
if src is None or mirror is None:
    for p in sorted(mount.glob("*/*/*")):
        print("  ", p, flush=True)
    raise SystemExit(f"bundle={src} mirror={mirror} under {mount}")
dst = pathlib.Path.cwd() / "_sensroot" / "bundle" / "hepg2"
dst.parent.mkdir(parents=True, exist_ok=True)
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src / "hepg2", dst)
total = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
print(f"[bundle] copied {src / 'hepg2'} -> {dst} ({total / 1e9:.2f} GB)", flush=True)
print(f"[mirror] {mirror}", flush=True)
pathlib.Path("_mirror_path").write_text(str(mirror))
PY

# Rebuild HepG2's targets and every anchor in the mirror from the mounted bytes.
# --skip-control-cells leaves the copied co-expression features alone.
SCFM_PFM3_DATA_ROOT="$SENS_ROOT" uv run python -m scfm_pfm.ingest.build \
  --mirror "$(cat _mirror_path)" --work ./_work --cell-lines hepg2 --skip-control-cells

ARMS="coexp,scgpt_resp,scprint_resp,transcriptformer_resp"
ARMS="${ARMS},scgpt_resp@uniform_virtual,scgpt_resp@individual_cells,scgpt_resp@mean_control_masked"
ARMS="${ARMS},scprint_resp@uniform_virtual,scprint_resp@individual_cells,scprint_resp@mean_control_masked"
ARMS="${ARMS},transcriptformer_resp@uniform_virtual,transcriptformer_resp@individual_cells"

uv run python -m scfm_pfm.translate.runner \
  --lines hepg2 --scopes full --splits grouped --translators mlp \
  --bundle "${SENS_ROOT}/bundle" --no-shuffle-null --arms "$ARMS" \
  --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/sensitivity"

cp "${SENS_ROOT}/bundle/MANIFEST.json" \
  "$SILICO_EXPERIMENT_ARTIFACTS_DIR/sensitivity/ingest_manifest.json"

echo "[done] sensitivity runs in $SILICO_EXPERIMENT_ARTIFACTS_DIR/sensitivity"
ls -la "$SILICO_EXPERIMENT_ARTIFACTS_DIR/sensitivity"
