"""Where run 6 reads and writes.

The same three-rung ladder every earlier run uses --- explicit override, then
the platform variable, then a local default --- read here and nowhere else. A
caller that reaches for ``os.environ`` has already broken the ladder.

Two path decisions are load-bearing rather than cosmetic:

**Run 6's heavy tree is NOT under run 5's root.** Run 5's ``analysis.discover()``
globs ``anchor=*`` under ``probe/<model>/<line>``, so a run-6 tree written
beneath it would be ingested by run 5's analysis as if it were probe output.
Run 6 gets its own sibling directory under ``/biodata`` instead.

**Nothing here writes into a sibling experiment.** Run 5's probe blocks, run 4's
checkpoints, run 3's bundle and run 2's handoff tables are all read-only to run
6: their numbers are cited, and a later run that edits them makes the citation
false. Run 6 stages a *copy* of run 3's bundle and adds its own blocks to the
copy.
"""

from __future__ import annotations

import os
from pathlib import Path

#: sibling runs read through files. Directory names are load-bearing across the
#: whole tree (``docs/INDEX.md`` s2) --- do not rename or move any of them.
RUN1_DIRNAME = "experiment-1-30499z"
RUN2_DIRNAME = "experiment-2-nygwcv"
RUN3_DIRNAME = "experiment-3-yga5ws"
RUN4_DIRNAME = "experiment-4-74xnwm"
RUN5_DIRNAME = "experiment-5-hidprb"

#: the project's designated heavy-output root on this machine
PROJECT_ROOT = Path("/biodata/nyanovsky/projects/scfm-pert-fail-modes")


def repo_dir() -> Path:
    """This experiment's directory (``experiments/experiment-6-hidarm``)."""
    return Path(__file__).resolve().parents[2]


def experiments_dir() -> Path:
    return repo_dir().parent


# --------------------------------------------------------------------------- #
# inputs: sibling runs, read-only
# --------------------------------------------------------------------------- #
def run2_handoff() -> Path:
    return experiments_dir() / RUN2_DIRNAME / "results" / "handoff"


def readout_table(cell_line: str) -> Path:
    return run2_handoff() / "readout" / f"readout_genes_{cell_line}.parquet"


def perturbations_table(cell_line: str) -> Path:
    """Run 2's per-perturbation table.

    Run 6 reads it for two things run 1's effect matrices would otherwise be
    needed for: ``effect_stratum``, which defines both the scored set and the
    basis pool, and ``effect_rms_lfc`` / ``target_control_cp10k``, the two axes
    the basis draw is stratified on.
    """
    return run2_handoff() / "effects" / cell_line / "perturbations.parquet"


def run3_bundle_root() -> Path:
    """Run 3's consolidated bundle: targets, co-expression, the gene tables.

    Read-only. ``stage.py`` copies from here into a writable staging tree and
    adds run 6's own feature blocks to the copy, because installing a block
    redefines ``scope_rows("common")`` for every arm that shares the bundle.
    """
    for env in ("SCFM_PFM6_BUNDLE", "SCFM_PFM3_BUNDLE"):
        val = os.environ.get(env)
        if val:
            return Path(val)
    return PROJECT_ROOT / "run4" / "bundle"


def run3_targets(cell_line: str) -> Path:
    return run3_bundle_root() / cell_line / "targets.npz"


def run5_probe_root() -> Path:
    """Run 5's probe output tree, which stage 0 reads ``norms.npz`` out of."""
    val = os.environ.get("SCFM_PFM6_RUN5_PROBE")
    if val:
        return Path(val)
    return PROJECT_ROOT / RUN5_DIRNAME / "probe"


def run5_layer_dir(model: str, cell_line: str, layer: int, anchor: str = "mean_control") -> Path:
    return run5_probe_root() / model / cell_line / f"anchor={anchor}" / f"layer={int(layer)}"


def run5_norms(model: str, cell_line: str, layer: int, anchor: str = "mean_control") -> Path:
    return run5_layer_dir(model, cell_line, layer, anchor) / "norms.npz"


def ft_checkpoint_root() -> Path:
    """Run 4's fine-tuned checkpoints, fetched here 2026-08-25 and 2026-08-31.

    NOT under the project directory: they live beside the other model weights.
    See that directory's ``PROVENANCE.md`` for sizes, md5s and the verification.
    """
    val = os.environ.get("SCFM_PFM6_FT_CKPT")
    if val:
        return Path(val)
    return Path("/biodata/nyanovsky/models/checkpoints_pearson")


def ft_checkpoint_dir(cell_line: str, arm: str) -> Path:
    return ft_checkpoint_root() / f"{cell_line}__{arm}"


def ft_checkpoint(cell_line: str, arm: str) -> Path:
    return ft_checkpoint_dir(cell_line, arm) / "model_best.pt"


def ft_train_config(cell_line: str, arm: str) -> Path:
    return ft_checkpoint_dir(cell_line, arm) / "train_config.json"


def ft_gene_table(cell_line: str, arm: str) -> Path:
    return ft_checkpoint_dir(cell_line, arm) / "gene_table_final.npz"


def checkpoint_cache() -> Path:
    """The frozen checkpoints, in the layout run 2's adapters expect.

    Run 2's loaders hardcode ``<cache>/scgpt/vocab.json`` and friends, and the
    model tree on this machine is arranged differently, so the default is run
    5's directory of symlinks in run 2's layout. Adapting the filesystem to the
    code is reversible; adapting the code to the filesystem is a fork.
    """
    val = os.environ.get("SCFM_PFM_CKPT_DIR")
    if val:
        return Path(val)
    return PROJECT_ROOT / "ckpt_cache"


# --------------------------------------------------------------------------- #
# outputs: run 6's own heavy tree
# --------------------------------------------------------------------------- #
def data_root() -> Path:
    """Durable heavy output. Never the repository, and never run 5's root."""
    for env in ("SCFM_PFM6_DATA_ROOT", "SILICO_EXPERIMENT_ARTIFACTS_DIR"):
        val = os.environ.get(env)
        if val:
            return Path(val)
    return PROJECT_ROOT / "experiment-6-hidarm"


def basis_dir(model: str, cell_line: str) -> Path:
    """Where a fitted PCA basis lives.

    Deliberately NOT under a bundle's ``features/``: run 3's reader globs every
    ``*.npz`` there and splits its name on ``__``, so a file with no ``__`` in
    it raises at bundle construction and kills every arm for that line.
    """
    return data_root() / "basis" / model / cell_line


def basis_file(model: str, cell_line: str, layer: int) -> Path:
    return basis_dir(model, cell_line) / f"layer={int(layer)}.npz"


def blocks_dir(model: str, cell_line: str) -> Path:
    """The projected feature blocks, one directory per (model, cell line)."""
    return data_root() / "blocks" / model / cell_line


def block_file(key: str, cell_line: str, model: str) -> Path:
    return blocks_dir(model, cell_line) / f"{key}.npz"


def store_scratch_dir(model: str, cell_line: str) -> Path:
    """The memory-mapped ``(n_pert, n_values, n_readout, k)`` accumulators.

    On ``/biodata`` beside the blocks rather than in ``/tmp``: the set is 34 GB
    (HepG2) to 55 GB (Jurkat) per (model, cell line) and this machine's root
    filesystem has 63 GB free in total, so a second concurrent pass would fill
    it. ``config.STORE_MEMMAPPED`` argues why the accumulator is mapped at all.

    Its own directory rather than ``blocks/``: a ``.npy`` sitting beside the
    ``.npz`` blocks is one careless glob away from being read as a feature file.
    Removed on a successful write, and deliberately LEFT BEHIND on a failure, so
    a crashed pass can be inspected rather than only re-run.
    """
    return data_root() / "store_scratch" / model / cell_line


def stage_root() -> Path:
    """Writable bundle copies, one per staged (line, block set)."""
    return data_root() / "staged"


def sweep_dir(name: str) -> Path:
    """One sweep's run tables. Sharding is a running counter rather than a hash,
    so adding an arm reshuffles every task --- each stage gets its OWN output
    directory and merges only run 6's own dirs."""
    return data_root() / "sweep" / name


def predictions_dir(name: str) -> Path:
    """Held-out predictions, saved for every scored arm.

    Run 3 keeps predictions only for a short list of arms in one cell of the
    sweep. Run 6 needs them for EVERY arm and cell, because the weighted-cosine
    and discrimination scores are computed from them after the fact rather than
    inside the runner --- see ``score.py`` for how that policy is put in force
    without editing run 3.
    """
    return sweep_dir(name) / "predictions"


# --------------------------------------------------------------------------- #
# outputs: small tables that travel with the worktree
# --------------------------------------------------------------------------- #
def results_dir() -> Path:
    return repo_dir() / "results"


def summary_dir() -> Path:
    return results_dir() / "summary"


def verification_dir() -> Path:
    return results_dir() / "verification"


def figures_dir() -> Path:
    return repo_dir() / "figures"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
