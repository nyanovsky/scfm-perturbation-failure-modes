#!/bin/sh
# Run 2's pinned model stack, trimmed to what run 4 imports.
#
# The versions are run 2's, not re-derived: the frozen scGPT numbers this run is
# read against were produced on this stack, and the two attention substitutions
# recorded there follow from it. The base image (docker.io/library/python:3.11-slim)
# ships no scientific packages, so nothing image-owned is shadowed or replaced.
#
# scgpt goes in with --no-deps because its own pin is torch<2.1, which cannot
# build against a current CUDA; only its model source is used, imported by file
# path so scgpt/__init__ (scvi-tools, scanpy, torchtext) never executes.
set -eu

pip install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1+cu121'

pip install -q --no-cache-dir \
    'numpy==1.26.4' 'pandas==2.2.3' 'pyarrow==17.0.0' 'h5py==3.12.1' 'scipy==1.15.3' \
    'tqdm==4.67.1' 'requests==2.32.3'

pip install -q --no-cache-dir --no-deps 'scgpt==0.2.4'

python3 - <<'PY'
import numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("numpy", numpy.__version__, "pandas", pandas.__version__, "scipy", scipy.__version__)
from importlib.metadata import version
print("scgpt", version("scgpt"))
PY
