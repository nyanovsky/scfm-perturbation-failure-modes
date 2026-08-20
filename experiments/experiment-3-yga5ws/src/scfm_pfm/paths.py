"""Where run 3 reads and writes.

Three roots, same discipline as run 2:

``data_root``   durable outputs. Inside a job that is
                ``$SILICO_EXPERIMENT_ARTIFACTS_DIR`` (uploaded to the store and
                also readable by a later job that mounts this job's volume);
                pod-side it falls back to ``results/artifacts_mirror``.
``handoff_root`` small tables that travel WITH the worktree. Run 1's and run 2's
                handoff tables are already in the shared worktree, so run 3
                reads them from the sibling experiment directories.
``bundle_root`` the consolidated inputs the ingest job writes and every
                training job reads, either from this job's own artifacts dir or
                from ``$SILICO_REUSED_VOLUME_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

RUN1_DIRNAME = "experiment-1-30499z"
RUN2_DIRNAME = "experiment-2-nygwcv"


def repo_dir() -> Path:
    """This experiment's directory (``experiments/experiment-3-yga5ws``)."""
    return Path(__file__).resolve().parents[2]


def experiments_dir() -> Path:
    return repo_dir().parent


def run1_handoff() -> Path:
    return experiments_dir() / RUN1_DIRNAME / "results" / "handoff"


def run2_handoff() -> Path:
    return experiments_dir() / RUN2_DIRNAME / "results" / "handoff"


def readout_table(cell_line: str) -> Path:
    return run2_handoff() / "readout" / f"readout_genes_{cell_line}.parquet"


def perturbations_table(cell_line: str) -> Path:
    return run2_handoff() / "effects" / cell_line / "perturbations.parquet"


def genes_table(cell_line: str) -> Path:
    return run2_handoff() / "effects" / cell_line / "genes.parquet"


def data_root() -> Path:
    for env in ("SCFM_PFM3_DATA_ROOT", "SILICO_EXPERIMENT_ARTIFACTS_DIR"):
        val = os.environ.get(env)
        if val:
            return Path(val)
    return repo_dir() / "results" / "artifacts_mirror"


def bundle_root() -> Path:
    """The consolidated inputs. Explicit override, then a mounted producer
    volume, then this process's own data root."""
    val = os.environ.get("SCFM_PFM3_BUNDLE")
    if val:
        return Path(val)
    mount = os.environ.get("SILICO_REUSED_VOLUME_DIR")
    if mount:
        found = find_bundle(Path(mount))
        if found is not None:
            return found
    return data_root() / "bundle"


def find_bundle(mount: Path, max_depth: int = 6) -> Path | None:
    if not mount.exists():
        return None
    frontier = [(mount, 0)]
    while frontier:
        here, depth = frontier.pop(0)
        candidate = here / "bundle"
        if (candidate / "MANIFEST.json").is_file():
            return candidate
        if depth >= max_depth:
            continue
        try:
            for child in sorted(here.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    frontier.append((child, depth + 1))
        except PermissionError:  # pragma: no cover
            continue
    return None


def find_mirror(mount: Path, run: str, max_depth: int = 8) -> Path | None:
    """A previous job's artifact mirror on a mounted volume.

    The ingest downloads run 1 and run 2 objects into ``<work>/mirror/<eid>/...``
    through presigned links that live an hour. A rerun that mounts the volume of
    the job that already fetched them reads that tree instead of minting new
    links, which is also the only way to re-derive a feature from bytes another
    job verified rather than from a fresh download.
    """
    if not mount.exists():
        return None
    frontier = [(mount, 0)]
    while frontier:
        here, depth = frontier.pop(0)
        candidate = here / "mirror"
        if (candidate / run).is_dir():
            return candidate
        if depth >= max_depth:
            continue
        try:
            for child in sorted(here.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    frontier.append((child, depth + 1))
        except PermissionError:  # pragma: no cover
            continue
    return None


def line_dir(cell_line: str, root: Path | None = None) -> Path:
    return (root or bundle_root()) / cell_line


def results_dir() -> Path:
    return data_root() / "results"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
