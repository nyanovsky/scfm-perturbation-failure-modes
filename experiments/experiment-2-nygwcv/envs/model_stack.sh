#!/bin/sh
# The one pinned stack that runs all four model response interfaces.
#
# The four packages declare mutually incompatible dependency ranges, so none of
# them is installed with its own resolver: torch and its CUDA wheels are pinned
# first, everything the adapters actually import is pinned next, and the four
# model packages themselves go in with --no-deps so their pins cannot pull a
# different torch in behind us. Reproduced exactly by every model job; the
# resolved versions are recorded again per model in environment.json.
#
# Base image: docker.io/library/python:3.11-slim
set -eu

pip install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121'

pip install -q --no-cache-dir \
    'numpy==1.26.4' 'pandas==2.2.3' 'pyarrow==17.0.0' 'h5py==3.12.1' 'scipy==1.15.3' \
    'transformers==4.44.2' 'tokenizers==0.19.1' 'safetensors==0.4.5' \
    'huggingface-hub==0.25.2' 'lightning==2.4.0' 'torchmetrics==1.6.0' \
    'einops==0.8.0' 'omegaconf==2.3.0' 'hydra-core==1.3.2' 'anndata==0.10.9' \
    'matplotlib==3.9.3' 'requests==2.32.3' 'tqdm==4.70.0' 'natsort==8.4.0' \
    'session_info==1.0.1' 'stdlib-list==0.12.0' 'PyYAML==6.0.3'

# --no-deps: each of these would otherwise reinstall torch at a version it was
# built against (scgpt wants torch<2.1, scprint pulls a torch-2.2 stack).
pip install -q --no-cache-dir --no-deps \
    'simpler_flash==1.0.7' 'scgpt==0.2.4' 'scprint==2.3.8' 'transcriptformer==0.6.1' \
    'torchdata==0.7.1' 'torchtext==0.17.0'

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
