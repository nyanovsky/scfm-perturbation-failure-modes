#!/usr/bin/env bash
# Anchor and form sensitivity, the pre-registration's secondary question:
# does the uniform-virtual anchor beat the on-distribution one (UGRN's mechanism
# claim), and do masked features differ from visible ones? HepG2 only, full
# scope, grouped split, MLP — exactly the cell config.py gates the step to.
#
# The first sweep only ever fetched anchor=mean_control, so the sensitivity arms
# resolved to nothing and were silently skipped. This job mounts the ingest
# job's volume for everything that does not change (co-expression, node
# statistics, embeddings), re-ingests HepG2 with the three sensitivity anchors
# beside them, and scores the sensitivity arms against their own mean-control
# arms re-run in the same job.
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
import pathlib, shutil, os
from scfm_pfm import paths

mount = os.environ.get("SILICO_REUSED_VOLUME_DIR", "")
src = paths.find_bundle(pathlib.Path(mount)) if mount else None
if src is None:
    raise SystemExit(f"no bundle found under SILICO_REUSED_VOLUME_DIR={mount!r}")
dst = pathlib.Path.cwd() / "_sensroot" / "bundle" / "hepg2"
dst.parent.mkdir(parents=True, exist_ok=True)
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src / "hepg2", dst)
total = 0
for p in sorted(dst.rglob("*")):
    if p.is_file():
        total += p.stat().st_size
        print(f"  {p.relative_to(dst)} {p.stat().st_size / 1e6:.1f} MB", flush=True)
print(f"[bundle] copied {src / 'hepg2'} -> {dst} ({total / 1e9:.2f} GB)", flush=True)
PY

# Re-ingest HepG2 only. --skip-control-cells leaves the copied coexpr.npz alone;
# targets and every anchor present in the new link manifest are rebuilt.
SCFM_PFM3_DATA_ROOT="$SENS_ROOT" uv run python -m scfm_pfm.ingest.build \
  --urls results/handoff/urls/artifact_urls_sens.json \
  --work ./_work --cell-lines hepg2 --skip-control-cells

ARMS="coexp,scgpt_resp,scprint_resp,transcriptformer_resp"
ARMS="${ARMS},scgpt_resp@uniform_virtual,scgpt_resp@individual_cells,scgpt_resp@mean_control_masked"
ARMS="${ARMS},scprint_resp@uniform_virtual,scprint_resp@individual_cells,scprint_resp@mean_control_masked"
ARMS="${ARMS},transcriptformer_resp@uniform_virtual,transcriptformer_resp@individual_cells"

uv run python -m scfm_pfm.translate.runner \
  --lines hepg2 --scopes full --splits grouped --translators mlp \
  --bundle "${SENS_ROOT}/bundle" --no-shuffle-null --arms "$ARMS" \
  --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/sensitivity"

echo "[done] sensitivity runs in $SILICO_EXPERIMENT_ARTIFACTS_DIR/sensitivity"
ls -la "$SILICO_EXPERIMENT_ARTIFACTS_DIR/sensitivity"
