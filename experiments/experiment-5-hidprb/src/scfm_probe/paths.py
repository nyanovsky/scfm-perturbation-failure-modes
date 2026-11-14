"""Where run 5 reads and writes.

Same three-rung ladder as runs 2 and 3 --- explicit override, then the platform
variable, then a local fallback --- so a run started here and a run started on
the Silico fabric land their heavy output in the right place without either
caller knowing which it is. The override is read here and nowhere else; a caller
that reaches for ``os.environ`` has already broken the ladder.

Run 5 owns no input tables of its own. The read-out set, the perturbation table
and the effect matrices are run 1's and run 2's handoff files, read straight out
of the sibling experiment directories, and the raw screens are on local disk.
Nothing here copies them, because a copy is a second thing that can drift.
"""

from __future__ import annotations

import os
from pathlib import Path

#: the sibling runs this one reads through files. Directory names are
#: load-bearing across the whole tree (docs/INDEX.md) --- do not rename.
RUN1_DIRNAME = "experiment-1-30499z"
RUN2_DIRNAME = "experiment-2-nygwcv"

#: the two cell lines, in the order every table and figure orders them
CELL_LINES = ("hepg2", "jurkat")

#: the models this run probes. AIDO.Cell is deliberately absent: it is not on
#: local disk, it scored at zero in run 3, and it costs ~40x scGPT per forward.
MODELS = ("scgpt", "scprint", "transcriptformer")

#: the raw GSE264667 screens, needed only for the individual-cell anchor, which
#: draws more control cells than run 2's four-cell handoff table carries
SCREENS_DIR = Path("/biodata/nyanovsky/datasets/Nadig")


def repo_dir() -> Path:
    """This experiment's directory (``experiments/experiment-5-hidprb``)."""
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


def screen_h5ad(cell_line: str) -> Path:
    return SCREENS_DIR / f"GSE264667_{cell_line}_raw_singlecell_01.h5ad"


#: run 2's shipped feature blocks, on this machine. The analysis reads one array
#: out of them --- ``gene_embeddings.npz``, the model's static gene table --- for
#: control C4. Nothing here writes into that tree: run 2's blocks are cited by a
#: paper and are read-only to every later run.
RUN2_FEATURES_EID = "exp_01m0c7ecnnf3fvxskxj5nygwcv"


def run2_features_root() -> Path:
    """Run 2's ``features_full_scope`` tree. Explicit override, then local disk.

    Same ladder as :func:`data_root`, one rung shorter: there is no platform
    variable for another experiment's artifacts, only the mirror that was
    fetched here.
    """
    val = os.environ.get("SCFM_PFM5_R2_FEATURES")
    if val:
        return Path(val)
    return (
        Path("/biodata/nyanovsky/projects/scfm-pert-fail-modes/artifacts")
        / RUN2_FEATURES_EID
        / "features_full_scope"
    )


def gene_embedding_table(model: str, cell_line: str) -> Path:
    """The model's static gene table, as run 2 shipped it.

    Control C4 asks whether the pooled hidden signature is just this table
    echoed down the residual stream, so the table has to be the very one run 2
    scored --- not a fresh read of the checkpoint, which could differ in row
    order, in dtype, or in which of several embedding tensors was taken.
    """
    return run2_features_root() / model / cell_line / "gene_embeddings.npz"


def run3_bundle_root() -> Path:
    """Run 3's consolidated inputs, which carry the MEASURED effect matrix.

    Control C7 collapses a retrieval over response groups, and a response group
    is defined by run 3 from measured effects --- not from anything a model
    says. Reading run 3's own bundle rather than re-deriving the matrix is what
    makes "same response group" mean the same thing in both runs.
    """
    for env in ("SCFM_PFM5_BUNDLE", "SCFM_PFM3_BUNDLE"):
        val = os.environ.get(env)
        if val:
            return Path(val)
    return Path("/biodata/nyanovsky/projects/scfm-pert-fail-modes/run4/bundle")


def run3_targets(cell_line: str) -> Path:
    return run3_bundle_root() / cell_line / "targets.npz"


def data_root() -> Path:
    """Durable heavy output. Never the repository."""
    for env in ("SCFM_PFM5_DATA_ROOT", "SILICO_EXPERIMENT_ARTIFACTS_DIR"):
        val = os.environ.get(env)
        if val:
            return Path(val)
    return repo_dir() / "results" / "artifacts_mirror"


def checkpoint_cache() -> Path:
    """Where the checkpoints live, in the layout run 2's adapters expect.

    Run 2's loaders hardcode ``<cache>/scgpt/vocab.json``,
    ``<cache>/scprint/medium-v1.5.ckpt`` and
    ``<cache>/transcriptformer/tf_sapiens/``. The models on this machine are
    laid out differently --- ``models/scGPT/scGPT_human/``,
    ``models/scprint/medium-v1.5/medium-v1.5.ckpt`` --- so the default here is a
    directory of symlinks in run 2's layout rather than the model directory
    itself.

    The symlinks, and not a patch to run 2's loaders, because run 2's shipped
    numbers are cited by a paper: the adapter that produces run 5's hidden
    states has to be the same object that produced run 2's responses, down to
    the path it reads the weights from. Adapting the filesystem to the code is
    reversible; adapting the code to the filesystem is a fork.
    """
    val = os.environ.get("SCFM_PFM_CKPT_DIR")
    if val:
        return Path(val)
    return Path("/biodata/nyanovsky/projects/scfm-pert-fail-modes/ckpt_cache")


# ---- run-5 output tree ---------------------------------------------------- #
# One directory per (model, cell line, anchor), with the layer sweep nested
# inside it rather than beside it: a layer is a read of the same forward pass,
# not a separate measurement, and nesting keeps a block's layers from being
# mixed across two runs of the driver.


def probe_dir(model: str, cell_line: str) -> Path:
    return data_root() / "probe" / model / cell_line


def anchor_dir(model: str, cell_line: str, anchor: str) -> Path:
    return probe_dir(model, cell_line) / f"anchor={anchor}"


def layer_dir(model: str, cell_line: str, anchor: str, layer: int) -> Path:
    return anchor_dir(model, cell_line, anchor) / f"layer={layer}"


def summary_dir() -> Path:
    """Small tables that travel with the worktree, not with the artifact store."""
    return repo_dir() / "results" / "probe"


def summary_table() -> Path:
    """The one table the analysis writes, and the one the figures read."""
    return summary_dir() / "summary.parquet"


def verification_dir() -> Path:
    return repo_dir() / "results" / "verification"


def figures_dir() -> Path:
    return repo_dir() / "figures"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
