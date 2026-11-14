"""The half-split over read-out positions, and the pooling estimator on it.

Why a split over positions and not over anything else
-----------------------------------------------------
At `mean_control` there is exactly one delta-h per (perturbation, value, layer):
one anchor profile, one baseline pass, one perturbed pass. There is therefore no
repeat, no fold and no held-out cell to score a retrieval against --- match a
vector to itself and every method scores 1.0. The one axis that *is* replicated
is the read-out position: the same intervention writes into 2,721 (HepG2) or
3,247 (Jurkat) token states, and asking whether the half of them the estimate was
not built from agrees with the half it was is a real generalisation question with
a real chance level.

So the partition is over positions, drawn once per block at
:data:`config.HALF_SPLIT_SEED` and written beside the arrays as
``half_assignment.npz``. It is global rather than per-perturbation on purpose: a
per-perturbation redraw would let two perturbations' A-halves overlap differently
and make the comparison between them depend on the draw rather than on the model.

What is excluded, and why that is the whole point
-------------------------------------------------
The perturbed gene's own position is removed from both halves before pooling.
Decoding "which gene did I perturb" from the token whose input value was just
changed is arithmetic, not propagation, and a probe that keeps it reports ~1.0 on
a model that does nothing at all. It is kept separately as the positive control
(C1), which is the only thing it is evidence for.

Rejected alternatives
---------------------
*Pooling by sum rather than mean.* The two halves differ in size by one whenever
the self position is excluded from a block of odd width, and by more once a
degenerate position is dropped; a sum would make ``v^A`` and ``v^B`` differ in
scale for a reason that has nothing to do with the model. Cosine is scale-free,
but the norm-only control (C6) is not, so the mean is used throughout.

*Normalising delta-h per position before pooling, only.* Whether the identity is
carried by the direction of the response or by its magnitude is exactly control
C6, so both pooled forms are written --- ``pool`` over raw delta-h and
``pool_unit`` over per-position unit-normalised delta-h --- and neither is
allowed to stand in for the other.

Torch-free by construction: this is the estimator, and an estimator that can only
be checked on a GPU machine does not get checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config as C


@dataclass(frozen=True)
class QuerySummary:
    """Everything one probe query (one perturbation, one value, one layer) yields.

    Deliberately small: a query's full ``(n_readout, d)`` delta-h is never kept.
    At TranscriptFormer's width that is 22 MB per query, and the run has ~3,500
    of them per block. Pooling inside the loop is what keeps peak memory bounded
    by one query rather than by the block.
    """

    #: (2, d) mean delta-h over half A and half B, self position excluded
    pool: np.ndarray
    #: (2, d) the same over per-position unit-normalised delta-h --- control C6
    pool_unit: np.ndarray
    #: (n_readout,) per-position ||delta-h||, self position INCLUDED, so S1's
    #: denominator and the C1 positive control both read out of one array
    norm: np.ndarray
    #: (d,) delta-h at the perturbed gene's own position --- control C1
    self_delta: np.ndarray
    #: how many positions each half actually pooled over
    n_pooled: tuple[int, int]
    #: off-target positions whose delta-h was exactly zero. They contribute zero
    #: to ``pool_unit`` rather than a nan, and the count says how many did.
    n_zero_positions: int


def half_assignment(n_readout: int, seed: int = C.HALF_SPLIT_SEED) -> np.ndarray:
    """A balanced ``{0, 1}`` label per read-out position, fixed for the block.

    The split is balanced rather than independent-coin because an unbalanced draw
    would give the two halves different variances, and the retrieval score
    compares a vector pooled over one against vectors pooled over the other.
    """
    if n_readout < 2 * C.N_POSITION_HALVES:
        raise RuntimeError(
            f"a half-split over read-out positions needs at least "
            f"{2 * C.N_POSITION_HALVES} of them; this block has {n_readout}, so the "
            "retrieval score would be comparing single positions rather than pooled "
            "signatures"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_readout)
    out = np.ones(n_readout, dtype=np.int8)
    out[order[: n_readout // 2]] = 0
    return out


def half_indices(
    assignment: np.ndarray, self_col: int, exclude_self: bool = C.EXCLUDE_SELF_POSITION
) -> tuple[np.ndarray, np.ndarray]:
    """The positions each half pools over for one perturbation.

    The self position is dropped from whichever half it landed in, which is why
    the two halves are not exactly equal in size and why ``n_pooled`` is recorded
    per query rather than assumed from ``n_readout``.
    """
    assignment = np.asarray(assignment)
    keep = np.ones(assignment.size, dtype=bool)
    if exclude_self:
        if not 0 <= int(self_col) < assignment.size:
            raise RuntimeError(
                f"the perturbed gene's read-out position {self_col} is outside the "
                f"{assignment.size} read-out positions; the perturbed table and the "
                "read-out table are not the pair this block was built from"
            )
        keep[int(self_col)] = False
    return (
        np.flatnonzero(keep & (assignment == 0)),
        np.flatnonzero(keep & (assignment == 1)),
    )


def unit_rows(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise L2-normalised ``delta``, its norms, and which rows were zero.

    A zero row is mapped to the zero vector, not to nan. That is the honest
    encoding of "this position did not move": it contributes nothing to the
    direction-only pool, it is counted, and it never silently poisons a mean.
    """
    delta = np.asarray(delta, dtype=np.float64)
    norm = np.linalg.norm(delta, axis=-1)
    zero = norm == 0.0
    scale = np.where(zero, 1.0, norm)
    return delta / scale[..., None], norm, zero


def pool_query(
    delta: np.ndarray,
    assignment: np.ndarray,
    self_col: int,
    exclude_self: bool = C.EXCLUDE_SELF_POSITION,
) -> QuerySummary:
    """Reduce one query's ``(n_readout, d)`` delta-h to its stored summary.

    ``delta`` must already be in read-out order and must be perturbed-minus-
    baseline, the same sign convention run 2 wrote into every block: a positive
    entry means raising the perturbed gene's input moved the read-out gene's
    hidden state in that direction.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2:
        raise RuntimeError(
            f"one query's delta-h must be (n_readout, d); got shape {delta.shape}, which "
            "means the caller is pooling a batch and the halves would mix queries"
        )
    if delta.shape[0] != np.asarray(assignment).size:
        raise RuntimeError(
            f"delta-h has {delta.shape[0]} read-out positions but the half assignment "
            f"has {np.asarray(assignment).size}; the two were built from different "
            "read-out tables and the pooled halves would not be disjoint"
        )
    unit, norm, zero = unit_rows(delta)
    idx_a, idx_b = half_indices(assignment, self_col, exclude_self=exclude_self)
    d = delta.shape[1]
    pool = np.zeros((C.N_POSITION_HALVES, d), dtype=np.float64)
    pool_unit = np.zeros((C.N_POSITION_HALVES, d), dtype=np.float64)
    for h, idx in enumerate((idx_a, idx_b)):
        if idx.size:
            pool[h] = delta[idx].mean(axis=0)
            pool_unit[h] = unit[idx].mean(axis=0)
    off = np.zeros(delta.shape[0], dtype=bool)
    off[idx_a] = True
    off[idx_b] = True
    return QuerySummary(
        pool=pool,
        pool_unit=pool_unit,
        norm=norm,
        self_delta=delta[int(self_col)].copy(),
        n_pooled=(int(idx_a.size), int(idx_b.size)),
        n_zero_positions=int(zero[off].sum()),
    )


def is_no_op(
    delta: np.ndarray, atol: float = 0.0, self_col: int | None = None
) -> bool:
    """Is this query's delta-h identically zero everywhere it is defined?

    ``atol = 0.0`` is deliberate. A no-op query --- one whose perturbed value
    equals the anchor's own value at that gene --- runs two bit-identical inputs
    through a frozen model, so its delta is exactly zero and not approximately
    zero. Anything else means the two passes differed in something other than the
    one value, which is a wiring bug and must not be absorbed by a tolerance.

    ``self_col`` excludes the perturbed gene's own position, which is required for
    a model that never reads a gene's count there and writes ``nan`` instead
    (TranscriptFormer). Without it every genuine no-op on that model fails this
    test --- ``nan`` is not ``<= 0`` --- and aborts the block with a message
    claiming a wiring bug about a query behaving exactly as designed. That would
    bite hardest at the ``individual_cells`` anchor, where no-ops are common and
    the whole held-out-cell secondary lives.
    """
    d = np.abs(np.asarray(delta))
    if self_col is not None:
        # axis 0: `delta` is (n_readout, d_model), so the perturbed gene's own
        # position indexes the READ-OUT axis. Deleting along the last axis instead
        # would drop a hidden dimension and leave the nan in place, which reads as
        # "not a no-op" and reproduces the very abort this argument exists to stop.
        if d.ndim != 2:
            raise RuntimeError(
                f"a query's delta-h must be (n_readout, d_model) for the self position "
                f"to be excluded by read-out index; got shape {d.shape}"
            )
        d = np.delete(d, int(self_col), axis=0)
    return bool(np.all(d <= atol))
