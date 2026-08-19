"""Artifact layout for run 2. One place that resolves where things live.

Three roots, deliberately separate:

``data_root``
    where this run WRITES durable outputs. Inside a job that is
    ``$SILICO_EXPERIMENT_ARTIFACTS_DIR``, whose contents flow into the durable
    artifact store; on the pod it falls back to a mirror under ``results/`` so
    the verifier and the report run without a cluster mount.

``handoff_root``
    small tables that travel WITH the worktree (gene universe, per-model
    vocabularies, the two effect-table columns run 2 reads, the demonstration
    control cells). A fresh on-demand job volume has no copy of run 1's
    artifacts, so anything a job must read at start-up lives here.

``checkpoint_cache``
    the four model checkpoints. One prefetch job downloads them onto its own
    volume; every later GPU job mounts that volume read-only at
    ``$SILICO_REUSED_VOLUME_DIR`` and finds the same layout underneath.
"""

from __future__ import annotations

import os
from pathlib import Path

CELL_LINES = ("hepg2", "jurkat")
MODELS = ("scgpt", "aido_cell", "scprint", "transcriptformer")

#: directory name the prefetch job writes into, inside its worktree copy
CKPT_DIRNAME = "_ckpt"


def repo_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    for env in ("SCFM_PFM_DATA_ROOT", "SILICO_EXPERIMENT_ARTIFACTS_DIR"):
        val = os.environ.get(env)
        if val:
            return Path(val)
    return repo_dir() / "results" / "artifacts_mirror"


def handoff_root() -> Path:
    val = os.environ.get("SCFM_PFM_HANDOFF_ROOT")
    return Path(val) if val else repo_dir() / "results" / "handoff"


def universe_dir() -> Path:
    return handoff_root() / "gene_universe"


def effects_dir(cell_line: str) -> Path:
    return handoff_root() / "effects" / cell_line


def demo_cells_dir() -> Path:
    return handoff_root() / "demo_cells"


def checkpoint_cache() -> Path:
    """Where the four checkpoints are readable in this process.

    Order: an explicit override, then the read-only mount of the prefetch job's
    retained volume, then a local directory (used by the prefetch job itself and
    by interactive sessions that download their own copy).
    """
    val = os.environ.get("SCFM_PFM_CKPT_DIR")
    if val:
        return Path(val)
    mount = os.environ.get("SILICO_REUSED_VOLUME_DIR")
    if mount:
        found = find_checkpoint_cache(Path(mount))
        if found is not None:
            return found
    return repo_dir() / CKPT_DIRNAME


def find_checkpoint_cache(mount: Path, max_depth: int = 6) -> Path | None:
    """Locate ``_ckpt`` inside a mounted volume without walking the whole tree."""
    if not mount.exists():
        return None
    frontier = [(mount, 0)]
    while frontier:
        here, depth = frontier.pop(0)
        candidate = here / CKPT_DIRNAME
        if candidate.is_dir():
            return candidate
        if depth >= max_depth:
            continue
        try:
            for child in sorted(here.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    frontier.append((child, depth + 1))
        except PermissionError:  # pragma: no cover - defensive on a mount
            continue
    return None


# ---- run-2 output tree ----------------------------------------------------
def floor_dir() -> Path:
    return data_root() / "floor_sweep"


def features_dir(model: str, cell_line: str) -> Path:
    return data_root() / "features" / model / cell_line


def features_full_dir(model: str, cell_line: str) -> Path:
    """Full-coverage blocks: the same layout, a separate tree.

    Run 2's first wave wrote 250 perturbed genes per block under ``features/``.
    The extension covers every in-scope perturbation for the three cheap models,
    and a full block is not a drop-in replacement for a subset one (different row
    count, different nested subsets), so it is written beside the original rather
    than over it. ``feature_index/blocks.json`` names the canonical block per
    (model, cell line).
    """
    return data_root() / "features_full_scope" / model / cell_line


def hidden_readout_dir(model: str, cell_line: str) -> Path:
    """The visible / masked / median-substituted comparison for one block.

    It belongs to the block whose masked variant it interprets, so it sits inside
    that block's directory but outside the ``anchor=`` namespace: it is a
    diagnostic over one anchor, not a fourth anchor.
    """
    return features_dir(model, cell_line) / "hidden_readout"


def index_dir() -> Path:
    return data_root() / "feature_index"


def report_dir() -> Path:
    return data_root() / "report"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
