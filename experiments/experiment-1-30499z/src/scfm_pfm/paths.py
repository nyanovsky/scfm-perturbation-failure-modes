"""Artifact layout. One place that resolves where outputs live.

Resolution order for the durable output root:

1. ``$SCFM_PFM_DATA_ROOT`` — explicit override (used by the pod-side mirror).
2. ``$SILICO_EXPERIMENT_ARTIFACTS_DIR`` — inside a job, this is the directory
   whose contents flow into the durable artifact store.
3. ``<repo>/results/artifacts_mirror`` — pod-side mirror of the store, so that
   ``uv run python -m scfm_pfm.verify --all`` works on the serving pod where
   the on-demand fabric keeps no persistent copy of job outputs.
"""

from __future__ import annotations

import os
from pathlib import Path

CELL_LINES = ("hepg2", "jurkat")
MODELS = ("scgpt", "aido_cell", "scprint", "transcriptformer")


def repo_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    for env in ("SCFM_PFM_DATA_ROOT", "SILICO_EXPERIMENT_ARTIFACTS_DIR"):
        val = os.environ.get(env)
        if val:
            return Path(val)
    return repo_dir() / "results" / "artifacts_mirror"


def input_root() -> Path:
    """Where to READ artifacts a previous job produced.

    On the on-demand fabric a later job mounts the producer's retained volume
    read-only, so reads and writes have different roots. ``$SCFM_PFM_INPUT_ROOT``
    points at the mounted copy; without it, reads and writes share one root.
    """
    val = os.environ.get("SCFM_PFM_INPUT_ROOT")
    return Path(val) if val else data_root()


def raw_dir() -> Path:
    """Scratch location for the ~15 GB of raw h5ads: job volume, never uploaded."""
    val = os.environ.get("SCFM_PFM_RAW_DIR")
    if val:
        return Path(val)
    return Path.cwd() / "_raw"


def effects_dir(cell_line: str) -> Path:
    return data_root() / "effects" / cell_line


def effects_input_dir(cell_line: str) -> Path:
    return input_root() / "effects" / cell_line


def universe_input_dir() -> Path:
    return input_root() / "gene_universe"


def crosscheck_dir() -> Path:
    return data_root() / "crosscheck"


def universe_dir() -> Path:
    return data_root() / "gene_universe"


def model_dir(model: str) -> Path:
    return data_root() / "models" / model


def provenance_dir() -> Path:
    return data_root() / "provenance"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
