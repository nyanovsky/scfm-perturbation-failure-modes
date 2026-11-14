"""The ``d -> k`` reduction: which perturbations the basis is fitted on, and how.

Invocation (the selection and the leakage check only --- no GPU, no model)::

    PYTHONPATH=src python -m scfm_hidarm.basis.fit --model scgpt --cell-line hepg2 --dry-run

Run 6's arm is ``concat_s (W' dh_s)`` over the ten slots (five ``vvp`` doses,
five ``gdt`` doses), so exactly one linear map stands between a hidden state and
run 3's unmodified translator. This module owns that map: where its training
data comes from, how it is estimated without ever holding raw ``dh``, how it is
ordered and sign-fixed so a refit is bit-comparable, what it is written as, and
the one diagnostic that says how much of the scored perturbations' response it
fails to span.

**This module is torch-free and holds no raw ``dh``.** The GPU streaming pass is
``extract/run.py``'s: it feeds ``SecondMoment`` one ``(n_readout, d)`` array at a
time and never lets a full ``dh`` block reach disk or RAM. That division is the
reason run 6 is affordable at all (README s2), so ``main`` here does the
selection and the checks and refuses to pretend it can do the pass.

Four choices are settled in ``config.py`` and merely enforced here. Each was
made against a named alternative:

**One basis per (model, cell line, layer), shared by all ten slots**
(``BASIS_SHARED_ACROSS_SLOTS``). A per-slot basis would put each dose in its own
coordinate frame, and "how the response changes with dose" --- the quantity the
run exists to recover --- would stop being a comparison of like coordinates.

**Fitted on raw ``dh`` pooled over slots** (``BASIS_SLOT_RESCALED = False``), so
the high-dose slots dominate the leading components. Deliberate: RMS-normalising
per slot before the fit removes exactly the dose-magnitude axis run 2's
per-query standardisation already removed once, and which run 6 exists to keep.

**Uncentred** (``BASIS_CENTRED = False``), i.e. second-moment rather than
covariance PCA, is primary. The centred variant is a sensitivity arm and is
exactly run 5's missing generic-per-gene-response control: centring subtracts
the mean response direction, so a centred basis asks what is left after every
gene's generic reaction to being perturbed is removed.

**No whitening** (``BASIS_WHITENED = False``). Run 3's ``Standardizer`` z-scores
every column in-fold anyway, so whitening would be absorbed by the translator
and would only destroy the eigenvalue ordering that makes ``K_LADDER`` a free
column prefix of the stored ``K_STORE`` block.

The genuinely contestable choice, and why it is made the way it is
------------------------------------------------------------------

Run 3 scores only the perturbations in ``SCORED_STRATA`` whose target is itself
a read-out gene: 524 in HepG2, 705 in Jurkat. The basis is fitted on
``N_BASIS_GENES`` genes drawn from a **disjoint** pool --- perturbations that
*were* screened and whose target *is* a read-out gene, but whose effect stratum
is in ``BASIS_STRATA`` and therefore excludes them from run 3's scoring set
entirely (365 in HepG2, 512 in Jurkat, measured by ``--dry-run``).

Alternatives, both rejected in the design record and recorded here so the choice
is auditable rather than assumed:

- *In-fold PCA* is **weaker**, not stronger: it still lets a perturbation in
  fold 2 shape the basis that scores fold 1. Here no scored perturbation shapes
  any basis at all.
- *Strict per-fold PCA* would need either a second GPU pass or ~460 GB of stored
  raw ``dh``, and then either a fork of run 3's ``arms.py`` (which has no fold
  argument) or five sweeps re-aggregated. That would put run 6's scoring outside
  run 3's unmodified rule, which is the one thing this run exists to preserve.

The exclusion is **not random with respect to biology**: the scored strata are
the strong-effect ones and are enriched for core-essential genes, while the
basis pool is by construction weaker-effect. So the top-``k`` components may
*under-span* the subspace the scored perturbations occupy. ``captured_energy``
measures that rather than arguing about it, and its docstring states why the
resulting error can only ever run one way.

The never-screened read-out genes are the documented **fallback** pool, second
choice only: they are mostly not in the CRISPRi library at all, so nothing
matches their perturbation regime and they carry no measured knockdown or effect
size to stratify on. If they are used at all, it is printed loudly and recorded
per gene in the provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import paths

#: The expression axis the draw is stratified on, in resolution order.
#: ``config.BASIS_STRATIFY_ON`` names the *quantity* --- control-condition
#: expression of the perturbed gene --- and run 2's read-out table calls it
#: ``pooled_control_cp10k`` while run 2's perturbation table calls the same
#: quantity ``target_control_cp10k``. This is a rename map between two tables,
#: not a second place to choose a constant: the concept is fixed in ``config``.
EXPRESSION_COLUMNS = (C.BASIS_STRATIFY_ON[0], "target_control_cp10k")

#: the effect-size axis, named in ``config.BASIS_STRATIFY_ON``
EFFECT_COLUMN = C.BASIS_STRATIFY_ON[1]

#: the two pools a basis gene can come from, in preference order. ``pool`` is
#: carried per gene into the npz and the provenance, because a basis fitted
#: partly on never-screened genes is a different object from one fitted wholly
#: on real knockdowns, and the difference must survive to the limitations.
POOL_PRIMARY = "excluded_stratum"
POOL_FALLBACK = "never_screened"

#: what an ``effect_bin`` of -1 means: the row carries no measured effect size,
#: so it could only be stratified on the expression axis. Only fallback-pool
#: rows ever carry it.
NO_EFFECT_BIN = -1

#: the columns every pool frame carries, so that the basis pool and the scored
#: pool are the same shape and a leakage check can compare them column-wise
POOL_COLUMNS = (
    "gene_id",
    "target",
    "readout_col",
    "effect_stratum",
    "target_control_cp10k",
    "effect_rms_lfc",
    "pool",
    "expr_bin",
    "effect_bin",
)


# --------------------------------------------------------------------------- #
# reading the pools
# --------------------------------------------------------------------------- #
def _readout_index(cell_line: str) -> pd.DataFrame:
    """Read-out genes with their column position, which is row order in the table.

    Row order is a contract here, not a convenience: ``readout_col`` is the
    index run 3's blocks are gathered on, so it is read off the same table run 2
    handed over and asserted against it rather than recomputed from a sort.
    """
    path = paths.readout_table(cell_line)
    if not path.exists():
        raise RuntimeError(
            f"run 2's read-out table for {cell_line} is missing at {path}; the basis "
            "draw cannot be made without it, because 'the target is itself a read-out "
            "gene' is the condition that defines both pools"
        )
    tab = pd.read_parquet(path)
    if "gene_id" not in tab.columns:
        raise RuntimeError(f"{path} carries no 'gene_id' column; columns are {list(tab.columns)}")
    out = tab.loc[:, ["gene_id"]].copy()
    out["readout_col"] = np.arange(len(tab), dtype=int)
    for name in ("gene_name", "hgnc_symbol"):
        if name in tab.columns:
            out[name] = tab[name].to_numpy()
    if "pooled_control_cp10k" in tab.columns:
        out["pooled_control_cp10k"] = tab["pooled_control_cp10k"].to_numpy()
    if out["gene_id"].duplicated().any():
        raise RuntimeError(
            f"{path} repeats a gene_id; read-out column position would be ambiguous"
        )
    return out


def expression_column(frame: pd.DataFrame) -> str:
    """Which of ``EXPRESSION_COLUMNS`` this table carries, resolved and recorded.

    Recorded rather than resolved silently: run 2's read-out table and its
    perturbation table name the same quantity differently, and a draw stratified
    on a column nobody can name afterwards is not a reproducible draw.
    """
    for name in EXPRESSION_COLUMNS:
        if name in frame.columns:
            return name
    raise RuntimeError(
        f"no expression column among {EXPRESSION_COLUMNS} in {list(frame.columns)}; "
        "the basis draw is stratified on expression and cannot fall back to unstratified"
    )


def _expression_of(frame: pd.DataFrame) -> np.ndarray:
    """The control-condition expression values, under whichever name they carry."""
    return frame[expression_column(frame)].to_numpy(dtype=float)


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bins, ``-1`` where the value is missing.

    ``duplicates="drop"`` rather than an error: a tied quantile edge means the
    pool genuinely has fewer distinct levels than requested, which is a fact
    about the pool to record (the achieved bin count goes into the provenance),
    not a reason to refuse to draw.
    """
    out = np.full(values.shape, -1, dtype=int)
    ok = np.isfinite(values)
    if ok.sum() == 0:
        return out
    binned = pd.qcut(values[ok], q=n_bins, labels=False, duplicates="drop")
    out[ok] = np.asarray(binned, dtype=int)
    return out


def _screened_pool(cell_line: str, strata: tuple[str, ...], pool: str) -> pd.DataFrame:
    """Screened perturbations in ``strata`` whose target is itself a read-out gene.

    Both of run 6's screened pools --- the scored set and the basis pool --- are
    this same query at two disjoint stratum lists, which is exactly why they can
    be compared column-for-column by ``leakage_check``.
    """
    path = paths.perturbations_table(cell_line)
    if not path.exists():
        raise RuntimeError(
            f"run 2's perturbation table for {cell_line} is missing at {path}; "
            "'effect_stratum' defines both the scored set and the basis pool, so "
            "neither can be formed without it"
        )
    pert = pd.read_parquet(path)
    for col in ("target", "target_ensembl", "effect_stratum", EFFECT_COLUMN):
        if col not in pert.columns:
            raise RuntimeError(f"{path} carries no {col!r} column; columns are {list(pert.columns)}")

    readout = _readout_index(cell_line)
    merged = pert.merge(
        readout[["gene_id", "readout_col"]],
        left_on="target_ensembl",
        right_on="gene_id",
        how="inner",
    )
    sel = merged[merged["effect_stratum"].isin(strata)].copy()
    if sel.empty:
        raise RuntimeError(
            f"{cell_line}: no screened read-out-gene perturbation in strata {strata}; "
            f"the table's strata are {sorted(pert['effect_stratum'].dropna().unique())}"
        )

    out = pd.DataFrame(
        {
            "gene_id": sel["gene_id"].to_numpy(),
            "target": sel["target"].to_numpy(),
            "readout_col": sel["readout_col"].to_numpy(dtype=int),
            "effect_stratum": sel["effect_stratum"].to_numpy(),
            "target_control_cp10k": _expression_of(sel),
            "effect_rms_lfc": sel[EFFECT_COLUMN].to_numpy(dtype=float),
            "pool": pool,
        }
    )
    # sorted by read-out column so the draw is a function of the seed alone and
    # not of whatever order the parquet happened to be written in
    out = out.sort_values("readout_col", kind="mergesort").reset_index(drop=True)
    out["expr_bin"] = _quantile_bins(
        out["target_control_cp10k"].to_numpy(), C.BASIS_N_EXPRESSION_BINS
    )
    out["effect_bin"] = _quantile_bins(
        out["effect_rms_lfc"].to_numpy(), C.BASIS_N_EFFECT_BINS
    )
    return out.loc[:, list(POOL_COLUMNS)]


def _never_screened_pool(cell_line: str) -> pd.DataFrame:
    """Read-out genes that were never screened --- the documented fallback only.

    These are second choice for a reason that is not squeamishness: they are
    mostly not in the CRISPRi library, so no perturbation regime matches them,
    and they carry neither a measured knockdown efficiency nor an effect size.
    The effect axis of the stratification therefore does not exist for them and
    ``effect_bin`` is ``NO_EFFECT_BIN`` --- a degraded draw, recorded as such.
    """
    readout = _readout_index(cell_line)
    pert = pd.read_parquet(paths.perturbations_table(cell_line))
    screened = set(pert["target_ensembl"].dropna())
    rest = readout[~readout["gene_id"].isin(screened)].copy()
    name_col = next((c for c in ("gene_name", "hgnc_symbol") if c in rest.columns), None)
    out = pd.DataFrame(
        {
            "gene_id": rest["gene_id"].to_numpy(),
            "target": (
                rest[name_col].to_numpy() if name_col else rest["gene_id"].to_numpy()
            ),
            "readout_col": rest["readout_col"].to_numpy(dtype=int),
            "effect_stratum": "",
            "target_control_cp10k": _expression_of(rest),
            "effect_rms_lfc": np.nan,
            "pool": POOL_FALLBACK,
        }
    )
    out = out.sort_values("readout_col", kind="mergesort").reset_index(drop=True)
    out["expr_bin"] = _quantile_bins(
        out["target_control_cp10k"].to_numpy(), C.BASIS_N_EXPRESSION_BINS
    )
    out["effect_bin"] = NO_EFFECT_BIN
    return out.loc[:, list(POOL_COLUMNS)]


def scored_perturbations(cell_line: str) -> pd.DataFrame:
    """The perturbations run 3 actually scores: 524 in HepG2, 705 in Jurkat.

    Returned in the same columns as the basis pool for two uses: the leakage
    check compares the two frames directly, and ``extract/run.py`` streams these
    rows into the *second* accumulator whose eigendecomposition is
    ``captured_energy``'s ``W_all``. Their ``expr_bin`` / ``effect_bin`` are
    computed over this frame and are informational only --- nothing is drawn
    from the scored pool, ever.
    """
    return _screened_pool(cell_line, C.SCORED_STRATA, "scored")


def basis_pool(cell_line: str) -> pd.DataFrame:
    """The pool the basis genes are drawn from: 365 in HepG2, 512 in Jurkat.

    Screened, target is a read-out gene, and stratum in ``BASIS_STRATA`` --- so
    every row is a real CRISPRi knockdown with a measured knockdown efficiency,
    in the same experimental regime as the scored set, and excluded from every
    fold of either split because run 3's stratum filter removes it.
    """
    return _screened_pool(cell_line, C.BASIS_STRATA, POOL_PRIMARY)


# --------------------------------------------------------------------------- #
# the seeded, stratified draw
# --------------------------------------------------------------------------- #
def _round_robin_draw(
    frame: pd.DataFrame, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, int]]:
    """Draw ``n`` rows as evenly across (expr_bin, effect_bin) cells as the pool allows.

    Round-robin rather than proportional allocation, and this is the point of
    stratifying at all: a proportional draw from a pool that is itself
    skewed in expression reproduces the skew, and the basis would then be fitted
    on one tail of the range the scored perturbations span. Taking one gene per
    cell per round spreads the draw over the whole grid and degrades gracefully
    when a cell runs out, which is why the achieved per-cell counts are recorded
    rather than assumed to be ``n / n_cells``.

    The cell visit order is permuted afresh each round from the same generator,
    so an ``n`` that is not a multiple of the cell count does not systematically
    favour the low-numbered cells.
    """
    cells: list[tuple[int, int]] = sorted(
        {(int(a), int(b)) for a, b in zip(frame["expr_bin"], frame["effect_bin"])}
    )
    queues: dict[tuple[int, int], list[int]] = {}
    for cell in cells:
        idx = np.flatnonzero(
            (frame["expr_bin"].to_numpy() == cell[0])
            & (frame["effect_bin"].to_numpy() == cell[1])
        )
        queues[cell] = [int(i) for i in rng.permutation(idx)]

    picked: list[int] = []
    counts = {f"{a},{b}": 0 for a, b in cells}
    while len(picked) < n:
        progressed = False
        for j in rng.permutation(len(cells)):
            cell = cells[int(j)]
            if queues[cell]:
                picked.append(queues[cell].pop())
                counts[f"{cell[0]},{cell[1]}"] += 1
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
    return np.asarray(picked, dtype=int), counts


def select_basis_genes(
    cell_line: str,
    n: int = C.N_BASIS_GENES,
    seed: int = C.BASIS_SEED,
    smoke: bool = False,
) -> pd.DataFrame:
    """The ``n`` genes one cell line's PCA basis is fitted on, and how each was chosen.

    Deterministic given ``seed``: the pool is sorted by read-out column before
    any random number is drawn, so the draw is a function of the seed and the
    handoff tables alone and not of parquet row order. Calling this twice
    returns identical frames, which ``tests/test_basis.py`` asserts rather than
    trusts.

    The scientific reason for the pool, restated where it is enforced: these are
    screened perturbations whose target is a read-out gene but whose effect
    stratum keeps them out of run 3's scoring set, so **no scored perturbation
    shapes the basis that scores it** --- strictly stronger than the in-fold PCA
    it replaces, since in-fold PCA still lets fold 2 shape fold 1's basis.

    The frame carries the full draw record in ``.attrs["provenance"]``, which
    ``write_basis`` reads; the columns themselves are per gene.
    """
    if smoke:
        n = min(n, C.SMOKE_N_BASIS_GENES)
    rng = np.random.default_rng(seed)

    primary = basis_pool(cell_line)
    take, counts = _round_robin_draw(primary, n, rng)
    chosen = [primary.iloc[take]]
    n_from_primary = len(take)

    fallback_counts: dict[str, int] = {}
    fallback = _never_screened_pool(cell_line)
    n_short = n - n_from_primary
    if n_short > 0:
        print(
            f"[basis] FALLBACK FIRED for {cell_line}: the excluded-stratum pool holds "
            f"{len(primary)} genes and yielded only {n_from_primary} of {n} stratified "
            f"draws, so {n_short} come from the {len(fallback)} never-screened read-out "
            "genes. Those are mostly not in the CRISPRi library, carry no measured "
            "knockdown or effect size, and are stratified on expression only. This "
            "changes what the basis is fitted on and belongs in the limitations.",
            flush=True,
        )
        take_fb, fallback_counts = _round_robin_draw(fallback, n_short, rng)
        chosen.append(fallback.iloc[take_fb])

    out = pd.concat(chosen, ignore_index=True)
    if len(out) != n:
        raise RuntimeError(
            f"{cell_line}: asked for {n} basis genes and both pools together yielded "
            f"{len(out)} ({n_from_primary} from {POOL_PRIMARY} of {len(primary)}, "
            f"{len(out) - n_from_primary} from {POOL_FALLBACK} of {len(fallback)}); "
            "a basis of the wrong width would silently change every projected block"
        )
    if out["gene_id"].duplicated().any():
        raise RuntimeError(f"{cell_line}: the basis draw repeated a gene_id")

    out.attrs["provenance"] = {
        "cell_line": cell_line,
        "n_requested": int(n),
        "seed": int(seed),
        "smoke": bool(smoke),
        "pool_sizes": {
            POOL_PRIMARY: int(len(primary)),
            POOL_FALLBACK: int(len(fallback)),
            "scored": int(len(scored_perturbations(cell_line))),
        },
        "n_from_pool": {
            POOL_PRIMARY: int(n_from_primary),
            POOL_FALLBACK: int(len(out) - n_from_primary),
        },
        "fallback_fired": bool(n_short > 0),
        "cell_counts": counts,
        "fallback_cell_counts": fallback_counts,
        "n_cells_available": int(len(counts)),
        "achieved_expression_bins": int(primary["expr_bin"].nunique()),
        "achieved_effect_bins": int(primary["effect_bin"].nunique()),
        "stratify_on": list(C.BASIS_STRATIFY_ON),
        "expression_column_used": expression_column(
            pd.read_parquet(paths.perturbations_table(cell_line))
        ),
    }
    return out


# --------------------------------------------------------------------------- #
# the leakage check
# --------------------------------------------------------------------------- #
def leakage_check(cell_line: str, basis_genes: pd.DataFrame) -> dict:
    """``basis genes ∩ in-scope perturbations = ∅``, on BOTH identifiers.

    Two identifiers because they can disagree: a gene symbol is not stable and
    two Ensembl ids can carry the same symbol, so an overlap invisible in one
    column can be real in the other. Any overlap at all means a scored
    perturbation helped build the basis it is scored through, and the run's
    central claim --- that no scored perturbation shapes any basis --- is false.

    Returned as a dict rather than an assertion so ``verify.py`` can print the
    number it checked (``0 of 64``) instead of a bare PASS.
    """
    scored = scored_perturbations(cell_line)
    by_id = set(basis_genes["gene_id"]) & set(scored["gene_id"])
    by_target = set(basis_genes["target"]) & set(scored["target"])
    overlap = sorted(by_id | by_target)
    return {
        "cell_line": cell_line,
        "n_basis": int(len(basis_genes)),
        "n_scored": int(len(scored)),
        "n_overlap": int(len(overlap)),
        "overlap_gene_id": sorted(by_id),
        "overlap_target": sorted(by_target),
        "detail": (
            f"{len(overlap)} of {len(basis_genes)} basis genes are among the "
            f"{len(scored)} in-scope perturbations of {cell_line} "
            f"(gene_id: {len(by_id)}, target symbol: {len(by_target)})"
        ),
    }


# --------------------------------------------------------------------------- #
# the streaming estimator
# --------------------------------------------------------------------------- #
class SecondMoment:
    """A ``d x d`` second-moment accumulator fed one ``(n, d)`` block at a time.

    This class exists so that raw ``dh`` never has to be held. The extractor
    produces one ``(n_readout, d)`` array per (perturbation, slot) and hands it
    straight here; at ``d = 2048`` the accumulator is 33 MB while the raw block
    it summarises would be 15-94 GB per layer. The eigenvectors of ``X'X`` are
    exactly the ones a batch PCA of ``X`` would return, so nothing is
    approximated --- only the storage is.

    Accumulation is in float64 regardless of the input dtype, deliberately: the
    features are float32 differences of two forward passes agreeing to ~1e-4,
    and summing 10^6 outer products of those in float32 loses low-order bits in
    exactly the trailing components ``K_LADDER``'s upper rungs read.

    ``matrix()`` returns the **un-normalised** scatter matrix. Dividing by ``n``
    or ``n - 1`` rescales every eigenvalue by the same factor and moves no
    eigenvector, and ``explained_variance_ratio`` is a ratio, so the
    normalisation cannot reach a feature or a score.
    """

    def __init__(self, d: int, centred: bool = C.BASIS_CENTRED) -> None:
        if int(d) <= 0:
            raise ValueError(f"d must be positive, got {d}")
        self.d = int(d)
        self.centred = bool(centred)
        self.n = 0
        self._xtx = np.zeros((self.d, self.d), dtype=np.float64)
        self._sum = np.zeros(self.d, dtype=np.float64) if self.centred else None

    def update(self, X: np.ndarray) -> "SecondMoment":
        """Accumulate one ``(n, d)`` block. Rows are observations, columns model dims."""
        A = np.asarray(X, dtype=np.float64)
        if A.ndim != 2 or A.shape[1] != self.d:
            raise ValueError(f"expected an (n, {self.d}) block, got shape {A.shape}")
        if not np.isfinite(A).all():
            raise RuntimeError(
                "a non-finite value reached the basis accumulator; TranscriptFormer "
                "writes its self position as nan and those rows must be dropped by the "
                "caller, not averaged into the second moment"
            )
        self._xtx += A.T @ A
        if self._sum is not None:
            self._sum += A.sum(axis=0)
        self.n += A.shape[0]
        return self

    def matrix(self) -> np.ndarray:
        """The symmetric PSD scatter matrix: ``X'X``, or the centred scatter."""
        if self.n == 0:
            raise RuntimeError(
                "the basis accumulator saw no rows; an empty second moment has no "
                "eigenvectors and a basis silently fitted on nothing would project "
                "every feature to zero"
            )
        M = self._xtx.copy()
        if self._sum is not None:
            mu = self._sum / self.n
            M -= self.n * np.outer(mu, mu)
        # symmetrise: the two triangles differ only by round-off, and eigh reads
        # one of them, so this is free and removes a source of refit drift
        return 0.5 * (M + M.T)

    def mean(self) -> np.ndarray:
        """The accumulated mean, defined only for the centred variant."""
        if self._sum is None:
            raise RuntimeError("this accumulator is uncentred and holds no mean")
        return self._sum / self.n


# --------------------------------------------------------------------------- #
# the eigendecomposition
# --------------------------------------------------------------------------- #
def eigendecompose(M: np.ndarray, k: int = C.K_STORE) -> dict:
    """Top-``k`` eigenvectors of a symmetric PSD matrix, ordered and sign-fixed.

    ``np.linalg.eigh`` rather than ``svd`` or ``eig``: ``M`` is symmetric by
    construction, ``eigh`` exploits that, and it returns real eigenvalues in a
    defined (ascending) order which is then reversed here. Components come out
    **eigenvalue-ordered**, which is what makes ``K_LADDER`` a column prefix of
    the stored ``K_STORE`` block --- every rung below 64 is free and none of them
    can be a choice made after seeing a score.

    ``W`` is returned float32, matching what is stored and what the extractor
    projects with; the decomposition itself is done in float64. The consequence
    is recorded in ``captured_energy``: float32 columns are orthonormal only to
    ~1e-7, which puts a ~0.05-degree floor under the principal-angle diagnostic.

    Signs are fixed deterministically: in each column the entry of largest
    magnitude is made positive. An eigenvector is only defined up to sign, and
    LAPACK's choice is not stable across builds or across a re-ordered input, so
    without this a refit would produce a bit-different ``W`` and a bit-different
    feature block for arithmetically identical input. With it, a refit is
    bit-comparable and ``tests/test_basis.py`` asserts exactly that.

    Negative eigenvalues are clipped to zero. ``M`` is PSD by construction, so a
    negative value is round-off; letting one into the denominator would make the
    explained-variance ratios not sum to one and would make a diagnostic look
    like a defect.
    """
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {A.shape}")
    d = A.shape[0]
    k = int(k)
    if not 1 <= k <= d:
        raise ValueError(f"k must be in [1, {d}], got {k}")

    vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    vecs = vecs[:, order]

    # deterministic sign: the largest-|component| entry of each column positive
    pivot = np.argmax(np.abs(vecs), axis=0)
    signs = np.sign(vecs[pivot, np.arange(d)])
    signs[signs == 0.0] = 1.0
    vecs = vecs * signs

    total = float(vals.sum())
    if total <= 0.0:
        raise RuntimeError(
            "the second-moment matrix has zero total energy; every delta-h fed to the "
            "accumulator was exactly zero, which is a broken extraction rather than a "
            "basis with no structure"
        )
    return {
        "W": np.ascontiguousarray(vecs[:, :k], dtype=np.float32),
        "eigenvalues": vals,
        "explained": vals[:k] / total,
    }


# --------------------------------------------------------------------------- #
# the captured-energy diagnostic
# --------------------------------------------------------------------------- #
def captured_energy(
    M_scored: np.ndarray,
    W_basis: np.ndarray,
    W_all: np.ndarray,
    ks: tuple[int, ...] = C.K_LADDER,
) -> list[dict]:
    """How much of the SCORED perturbations' energy the basis spans. **Diagnostic only.**

    **This number is never used to score anything.** It does not select ``k``,
    it does not reweight a feature, it does not gate an arm, and it is computed
    after ``W_basis`` is already fixed. It is read once, before any score is
    read, and written into the limitations.

    What it measures. ``M_scored`` is the second moment of the **scored**
    perturbations' ``dh`` --- accumulated in the same streaming pass, 33 MB at
    ``d = 2048``, free. ``W_all`` is its own top-``k`` eigenbasis, which is by
    construction the ``k``-dimensional subspace capturing the most of that
    energy: no basis can beat it. So per ``k`` this reports the fraction
    ``W_basis``'s top-``k`` captures, the maximum ``W_all``'s top-``k``
    captures, their ratio, and the principal angles between the two subspaces.

    **The direction of the error, which is the whole reason the diagnostic is
    readable at all.** The basis never sees the targets, and the basis
    perturbations are disjoint from the scored ones, so no channel exists by
    which ``W_basis`` could encode which perturbation is which. Under-spanning
    therefore **loses** signal and cannot manufacture it: the failure mode is a
    **false null, never a false positive**. That is the safe side for a run whose
    headline may be negative --- but a limitation to quantify rather than assert,
    which is why the ratio is reported per (model, line, layer, k) rather than
    argued about. ``config.BASIS_COVERAGE_TARGET`` is the threshold at which the
    concern is closed; below it the arm is known to be downward-biased, and by
    how much.

    Principal angles are the arccos of the singular values of
    ``W_basis[:, :k]' W_all[:, :k]``, in degrees, ascending. Zero throughout
    means the same subspace; the largest angle is the direction of the scored
    response that the basis misses most.

    ⚠️ **The angles have a resolution floor of ~0.05 degrees**, and it is a
    property of the stored basis rather than of this arithmetic. ``W`` is float32
    on disk and in memory, so its columns are orthonormal only to ~1e-7; arccos
    near 1 amplifies that, since ``arccos(1 - eps) ~ sqrt(2 eps)``, and a basis
    compared against *itself* comes out at ~0.01 degrees rather than 0. Read any
    angle below ~0.05 degrees as zero. The captured-energy fractions carry no
    such amplification --- they are traces, computed in float64, and a basis
    against itself gives a ratio that is bit-exactly 1.0.
    """
    M = np.asarray(M_scored, dtype=np.float64)
    Wb = np.asarray(W_basis, dtype=np.float64)
    Wa = np.asarray(W_all, dtype=np.float64)
    if M.shape[0] != M.shape[1] or Wb.shape[0] != M.shape[0] or Wa.shape[0] != M.shape[0]:
        raise ValueError(
            f"shape mismatch: M {M.shape}, W_basis {Wb.shape}, W_all {Wa.shape}"
        )
    total = float(np.trace(M))
    if total <= 0.0:
        raise RuntimeError(
            "the scored perturbations' second moment has zero trace; the captured-energy "
            "diagnostic would be 0/0 and would be reported as a coverage failure"
        )

    rows: list[dict] = []
    for k in ks:
        k = int(k)
        if k > min(Wb.shape[1], Wa.shape[1]):
            continue
        Bk, Ak = Wb[:, :k], Wa[:, :k]
        cap_b = float(np.sum(Bk * (M @ Bk))) / total
        cap_a = float(np.sum(Ak * (M @ Ak))) / total
        ratio = cap_b / cap_a if cap_a > 0.0 else float("nan")
        sv = np.linalg.svd(Bk.T @ Ak, compute_uv=False)
        angles = np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))
        angles = np.sort(angles)
        rows.append(
            {
                "k": k,
                "captured_basis": cap_b,
                "captured_all": cap_a,
                "ratio": ratio,
                "principal_angles_deg": [float(a) for a in angles],
                "max_principal_angle_deg": float(angles.max()),
                "meets_target": bool(ratio >= C.BASIS_COVERAGE_TARGET),
                "coverage_target": float(C.BASIS_COVERAGE_TARGET),
                "diagnostic_only": True,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# on disk
# --------------------------------------------------------------------------- #
def _labels(col: pd.Series) -> np.ndarray:
    """A string column as a fixed-width unicode array, never an object array.

    ``np.savez`` stores an object array by pickling it, and ``np.load`` then
    refuses it under ``allow_pickle=False``. Arrays carry their own labels in
    this project, so a label array that cannot be read back without enabling
    pickle would break the join every projected block depends on.
    """
    return np.asarray([str(v) for v in col], dtype=np.str_)


def write_basis(
    model: str,
    cell_line: str,
    layer: int,
    *,
    W: np.ndarray,
    eigenvalues: np.ndarray,
    explained: np.ndarray,
    genes: pd.DataFrame,
    captured: list[dict] | None = None,
    draw: dict | None = None,
    centred: bool = C.BASIS_CENTRED,
    provenance: dict | None = None,
) -> Path:
    """Write one ``(model, cell line, layer)`` basis and its provenance.

    Two files, on purpose. The ``.npz`` holds what the projection needs and the
    labels that identify it --- arrays carry their own labels, so a block can be
    joined back to the genes it was fitted on rather than trusted to be in the
    right order. The sibling ``.json`` holds the argument: the per-cell counts of
    the stratified draw, the pool sizes, the seed, every reduction constant
    echoed verbatim, and the captured-energy table. Anything a reader would need
    to say what this basis is without re-running the fit is in the json.

    The basis is written under ``paths.basis_dir``, never inside a bundle's
    ``features/``: run 3's reader globs every ``*.npz`` there and splits its name
    on ``__``, so a basis file dropped in that directory would raise at bundle
    construction and kill every arm for that line.

    ``genes`` is the frame ``select_basis_genes`` returned, carrying its draw
    record in ``.attrs`` (or passed explicitly as ``draw=``); ``provenance`` is
    merged into the json for whatever the caller knows and this module does not
    --- the anchor, the layout, the gdt step, the wall clock of the pass.
    """
    W = np.ascontiguousarray(np.asarray(W, dtype=np.float32))
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    explained = np.asarray(explained, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"W must be 2-d, got shape {W.shape}")
    if explained.shape[0] != W.shape[1]:
        raise RuntimeError(
            f"explained has {explained.shape[0]} entries for a basis of width "
            f"{W.shape[1]}; one of them is stale"
        )
    if len(genes) == 0:
        raise RuntimeError("refusing to write a basis with no recorded basis genes")

    draw = draw if draw is not None else genes.attrs.get("provenance")
    if draw is None:
        raise RuntimeError(
            "no draw record: pass draw=, or pass the frame select_basis_genes returned "
            "with its .attrs intact. A basis whose gene selection is not recorded cannot "
            "be audited against the leakage check that licenses it"
        )

    out = paths.basis_file(model, cell_line, layer)
    paths.ensure(out.parent)
    np.savez(
        out,
        W=W,
        eigenvalues=eigenvalues,
        explained_variance_ratio=explained,
        # cast through numpy's fixed-width unicode dtype, not pandas' object
        # dtype: an object array is only readable with allow_pickle=True, and a
        # basis file that has to be unpickled to be read is a basis file no
        # reader should trust
        basis_gene_id=_labels(genes["gene_id"]),
        basis_gene_target=_labels(genes["target"]),
        basis_pool=_labels(genes["pool"]),
        basis_stratum=_labels(genes["effect_stratum"]),
        basis_readout_col=np.asarray(genes["readout_col"].to_numpy(dtype=int)),
        d=np.asarray(int(W.shape[0])),
        k_store=np.asarray(int(W.shape[1])),
        centred=np.asarray(bool(centred)),
        layer=np.asarray(int(layer)),
        model=np.asarray(str(model)),
        cell_line=np.asarray(str(cell_line)),
    )

    prov = {
        "model": model,
        "cell_line": cell_line,
        "layer": int(layer),
        "d": int(W.shape[0]),
        "k_store": int(W.shape[1]),
        "draw": draw,
        "leakage_check": leakage_check(cell_line, genes),
        "captured_energy": captured if captured is not None else [],
        "basis_genes": genes.loc[:, list(POOL_COLUMNS)].to_dict(orient="records"),
        "config": {
            "SEED": C.SEED,
            "BASIS_SEED": C.BASIS_SEED,
            "N_BASIS_GENES": C.N_BASIS_GENES,
            "SCORED_STRATA": list(C.SCORED_STRATA),
            "BASIS_STRATA": list(C.BASIS_STRATA),
            "BASIS_STRATIFY_ON": list(C.BASIS_STRATIFY_ON),
            "BASIS_N_EXPRESSION_BINS": C.BASIS_N_EXPRESSION_BINS,
            "BASIS_N_EFFECT_BINS": C.BASIS_N_EFFECT_BINS,
            "BASIS_SHARED_ACROSS_SLOTS": C.BASIS_SHARED_ACROSS_SLOTS,
            "BASIS_SLOT_RESCALED": C.BASIS_SLOT_RESCALED,
            "BASIS_CENTRED": C.BASIS_CENTRED,
            "BASIS_WHITENED": C.BASIS_WHITENED,
            "BASIS_COVERAGE_TARGET": C.BASIS_COVERAGE_TARGET,
            "K_STORE": C.K_STORE,
            "K_PRIMARY": C.K_PRIMARY,
            "K_LADDER": list(C.K_LADDER),
            "SLOT_FORMS": list(C.SLOT_FORMS),
            "N_SLOTS": C.N_SLOTS,
            "VALUE_GRID_CP10K": list(C.VALUE_GRID_CP10K),
            "ANCHOR": C.ANCHOR,
        },
        "centred_this_fit": bool(centred),
    }
    if provenance:
        prov.update(provenance)
    out.with_suffix(".json").write_text(json.dumps(prov, indent=2, sort_keys=False))
    print(
        f"[basis] wrote {out} (d={W.shape[0]}, k={W.shape[1]}, centred={bool(centred)}) "
        f"and its provenance",
        flush=True,
    )
    return out


def load_basis(model: str, cell_line: str, layer: int) -> dict:
    """Read one basis back, arrays and provenance together.

    Fails loudly on a missing sibling json rather than returning the arrays
    alone: a ``W`` whose draw record is absent cannot be checked against the
    leakage check that licenses it, and projecting through an unaudited basis is
    the one thing this module exists to prevent.
    """
    npz_path = paths.basis_file(model, cell_line, layer)
    if not npz_path.exists():
        raise RuntimeError(f"no basis at {npz_path}; fit it before projecting through it")
    with np.load(npz_path, allow_pickle=False) as z:
        out = {name: z[name] for name in z.files}
    json_path = npz_path.with_suffix(".json")
    if not json_path.exists():
        raise RuntimeError(
            f"{npz_path} has no sibling provenance at {json_path}; refusing to load a "
            "basis whose gene draw and leakage check are unrecorded"
        )
    out["provenance"] = json.loads(json_path.read_text())
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _report(cell_line: str, n: int, seed: int, smoke: bool) -> int:
    genes = select_basis_genes(cell_line, n=n, seed=seed, smoke=smoke)
    prov = genes.attrs["provenance"]
    leak = leakage_check(cell_line, genes)

    print(f"[basis] {cell_line}", flush=True)
    print(
        f"[basis]   pools: {POOL_PRIMARY}={prov['pool_sizes'][POOL_PRIMARY]}  "
        f"{POOL_FALLBACK}={prov['pool_sizes'][POOL_FALLBACK]}  "
        f"scored={prov['pool_sizes']['scored']}",
        flush=True,
    )
    print(
        f"[basis]   drawn: {prov['n_from_pool'][POOL_PRIMARY]} from {POOL_PRIMARY}, "
        f"{prov['n_from_pool'][POOL_FALLBACK]} from {POOL_FALLBACK} "
        f"(seed {prov['seed']}, {prov['n_cells_available']} stratum cells)",
        flush=True,
    )
    for cell, count in sorted(prov["cell_counts"].items()):
        print(f"[basis]     cell (expr,effect)=({cell}): {count}", flush=True)
    if prov["fallback_cell_counts"]:
        for cell, count in sorted(prov["fallback_cell_counts"].items()):
            print(f"[basis]     fallback cell (expr,-1)=({cell}): {count}", flush=True)
    print(f"[basis]   knockdown strata drawn: "
          f"{dict(genes['effect_stratum'].value_counts())}", flush=True)
    tag = "PASS" if leak["n_overlap"] == 0 else "FAIL"
    print(f"[{tag}] basis/leakage: {leak['detail']}", flush=True)
    return 0 if leak["n_overlap"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, choices=list(C.MODELS))
    parser.add_argument("--cell-line", default="all", choices=["all", *C.CELL_LINES])
    parser.add_argument(
        "--layers",
        default="",
        help="comma-separated layers, echoed only: the draw does not depend on layer",
    )
    parser.add_argument("--n", type=int, default=C.N_BASIS_GENES)
    parser.add_argument("--seed", type=int, default=C.BASIS_SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="gene selection and the leakage check only; no GPU, no model, no fit",
    )
    args = parser.parse_args(argv)

    lines = C.CELL_LINES if args.cell_line == "all" else (args.cell_line,)
    if not args.dry_run:
        raise SystemExit(
            "scfm_hidarm.basis.fit is torch-free and holds no raw delta-h by design: the "
            "streaming pass that fills SecondMoment runs on the GPU inside "
            "`python -m scfm_hidarm.extract.run`, which imports SecondMoment, "
            "eigendecompose, captured_energy and write_basis from here. Re-run with "
            "--dry-run for the gene selection and the leakage check."
        )

    if args.layers:
        print(f"[basis] layers requested (not used by the draw): {args.layers}", flush=True)
    if args.model:
        print(f"[basis] model: {args.model} (the draw is per cell line, not per model)",
              flush=True)
    status = 0
    for line in lines:
        status |= _report(line, n=args.n, seed=args.seed, smoke=args.smoke)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
