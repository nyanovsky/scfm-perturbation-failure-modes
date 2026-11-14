#!/bin/sh
# CPU half of run 2's pinned stack: everything the ingest pass imports and
# nothing else. The base image (docker.io/library/python:3.11-slim) ships no
# scientific packages, so there is no image-owned build to shadow.
set -eu
pip install -q --no-cache-dir \
    'numpy==1.26.4' 'pandas==2.2.3' 'pyarrow==17.0.0' 'h5py==3.12.1' 'scipy==1.15.3'
python3 -c "import numpy, pandas, h5py, scipy; print('cpu stack', numpy.__version__, pandas.__version__, h5py.__version__, scipy.__version__)"
