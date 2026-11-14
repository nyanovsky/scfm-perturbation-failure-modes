"""Where run 4 reads and writes.

``data_root``    durable outputs. Inside a job that is
                 ``$SILICO_EXPERIMENT_ARTIFACTS_DIR``, which the serverless
                 flow-back uploads to the artifact store; pod-side it falls back
                 to ``results/artifacts_mirror``.
``bundle_root``  run 3's consolidated inputs (targets, co-expression, node
                 features, the frozen models' features and gene tables), staged
                 read-only into a job or mirrored pod-side.
``cells_root``   the fine-tuning matrices the ingest job writes.

Run 1's and run 2's small handoff tables travel with the worktree and are read
straight out of the sibling experiment directories.
"""

from __future__ import annotations

import os
from pathlib import Path

from .sibling import run2_dir


def repo_dir() -> Path:
    """This experiment's directory (``experiments/experiment-4-74xnwm``)."""
    return Path(__file__).resolve().parents[2]


def ensure(p: Path) -> Path:
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def results_dir() -> Path:
    """Small reviewable outputs that live in the worktree, not in artifacts."""
    return ensure(repo_dir() / "results")


def data_root() -> Path:
    for env in ("SCFM_FT_DATA_ROOT", "SILICO_EXPERIMENT_ARTIFACTS_DIR"):
        val = os.environ.get(env)
        if val:
            return Path(val)
    return repo_dir() / "results" / "artifacts_mirror"


def readout_table(cell_line: str) -> Path:
    return run2_dir() / "results" / "handoff" / "readout" / f"readout_genes_{cell_line}.parquet"


def perturbations_table(cell_line: str) -> Path:
    return run2_dir() / "results" / "handoff" / "effects" / cell_line / "perturbations.parquet"


def _first_existing(*candidates: Path | None) -> Path | None:
    for c in candidates:
        if c is not None and Path(c).exists():
            return Path(c)
    return None


def bundle_root() -> Path:
    """Run 3's bundle, wherever this process can see it.

    In a job it arrives either as a staged input artifact (mirroring the store
    layout) or on a mounted producer volume; pod-side it is whatever
    ``SCFM_FT_BUNDLE`` names.
    """
    env = os.environ.get("SCFM_FT_BUNDLE")
    if env:
        return Path(env)
    staged = os.environ.get("SILICO_INPUT_ARTIFACTS_DIR")
    reused = os.environ.get("SILICO_REUSED_VOLUME_DIR")
    found = _first_existing(
        Path(staged) / "experiments" / "exp_01m0emwktvfe892x6ebqyga5ws" / "bundle" if staged else None,
        Path(reused) / "outputs" / "bundle" if reused else None,
        data_root() / "bundle",
        repo_dir() / "results" / "artifacts_mirror" / "bundle",
    )
    if found is None:
        raise RuntimeError("run 3's bundle is not visible to this process")
    return found


def esm2_table_file() -> Path:
    staged = os.environ.get("SILICO_INPUT_ARTIFACTS_DIR")
    reused = os.environ.get("SILICO_REUSED_VOLUME_DIR")
    found = _first_existing(
        Path(os.environ["SCFM_FT_ESM2"]) if os.environ.get("SCFM_FT_ESM2") else None,
        Path(staged) / "experiments" / "exp_01m0emwktvfe892x6ebqyga5ws" / "protein" / "esm2_3b.npz"
        if staged
        else None,
        Path(reused) / "outputs" / "protein" / "esm2_3b.npz" if reused else None,
        data_root() / "protein" / "esm2_3b.npz",
    )
    if found is None:
        raise RuntimeError("the ESM2-3B gene table is not visible to this process")
    return found


def cells_root() -> Path:
    """The fine-tuning cell matrices written by the ingest job.

    They arrive either on the ingest job's mounted volume or as staged input
    artifacts. The volume's internal layout is not part of any contract, so the
    likely places are tried and then a bounded search is done rather than
    assuming one.
    """
    env = os.environ.get("SCFM_FT_CELLS")
    if env:
        return Path(env)
    reused = os.environ.get("SILICO_REUSED_VOLUME_DIR")
    staged = os.environ.get("SILICO_INPUT_ARTIFACTS_DIR")
    eid = os.environ.get("SILICO_EXPERIMENT_ID", "")
    found = _first_existing(
        Path(reused) / "outputs" / "cells" if reused else None,
        Path(reused) / "cells" if reused else None,
        Path(staged) / "experiments" / eid / "cells" if (staged and eid) else None,
        data_root() / "cells",
        repo_dir() / "results" / "artifacts_mirror" / "cells",
    )
    if found is None and reused:
        # the mounted volume's internal layout is not a documented contract, so
        # look for the directory rather than guess at it, two levels deep only
        for depth in ("*/cells", "*/*/cells"):
            hits = sorted(Path(reused).glob(depth))
            if hits:
                found = hits[0]
                break
    if found is None:
        raise RuntimeError(
            "the fine-tuning cell matrices are not visible to this process "
            f"(reused volume {reused!r}, staged inputs {staged!r})"
        )
    return found


def checkpoint_cache() -> Path:
    """Where the published scGPT release is downloaded inside a job."""
    env = os.environ.get("SCFM_FT_CKPT")
    if env:
        return ensure(Path(env))
    return ensure(repo_dir() / "scratch" / "_ckpt" / "scgpt")


def run_suffix() -> str:
    """Suffix separating one execution's durable outputs from another's.

    The first execution of the sixteen arms selected its checkpoint on held-out
    masked MSE; the published recipe selects on validation Pearson. Rather than
    overwrite the first execution's weights and tables, the second writes
    alongside them under ``checkpoints<suffix>`` and ``scores<suffix>``, so both
    selection rules stay readable and comparable.
    """
    return os.environ.get("SCFM_FT_RUN_SUFFIX", "")


def arm_dir(arm_name: str, cell_line: str) -> Path:
    """One training run's durable output directory."""
    return ensure(data_root() / f"checkpoints{run_suffix()}" / f"{cell_line}__{arm_name}")


def scores_dir() -> Path:
    return ensure(data_root() / f"scores{run_suffix()}")


def packs_dir() -> Path:
    """One small archive per training run, so the merge job stages 16 files.

    Staging is capped at 128 files and the per-run small outputs are ten times
    that; each training job therefore packs its own tables and metadata.
    """
    return ensure(data_root() / f"packs{run_suffix()}")
