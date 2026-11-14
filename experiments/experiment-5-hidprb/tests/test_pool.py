"""The half-split partition and the pooling estimator, checked against arithmetic.

Failure class this guards against: a split that drops a read-out position, lands
one in both halves, or leaves the perturbed gene's own position in the pool. None
of those raises, and every one of them turns S2 into a statistic about something
other than propagation --- the self position most severely, because decoding *i*
from the token whose input you just changed is trivial and would read as a
perfect signature at every layer.

The second failure class is degenerate input. A block whose delta-h is exactly
zero must pool to exactly zero and stay finite; a nan leaking out of the
direction-only pool would propagate into the retrieval score as a rank, and a
rank computed from nans is not a measurement.

Written from the run 5 plan's frozen interface, not from ``pool.py``. Where a
value is asserted, the expected number is computed here by hand rather than by
calling the code under test.

Invocation (from the run directory):  PYTHONPATH=src pytest tests/test_pool.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scfm_probe import config as C
from scfm_probe.probe import pool as P

#: block sizes worth exercising: the two real ones (HepG2, Jurkat), the toy's,
#: an odd count, and the smallest block the estimator accepts --- where removing
#: the self position leaves a one-position half.
BLOCK_SIZES = (4, 5, 24, 524, 705)


def _halves(n_readout: int, self_col: int, seed: int = C.HALF_SPLIT_SEED):
    assignment = P.half_assignment(n_readout, seed=seed)
    return P.half_indices(assignment, self_col)


# --------------------------------------------------------------------------- #
# the partition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_readout", BLOCK_SIZES)
def test_halves_partition_the_off_target_positions(n_readout: int) -> None:
    """Disjoint, and together exactly the off-target set --- for every self position.

    Asserted for every perturbation in the block, not a sampled few: a split that
    is right for 523 genes and drops a position for the 524th produces a number
    that is wrong for one row and looks right in aggregate.
    """
    expected_total = n_readout - (1 if C.EXCLUDE_SELF_POSITION else 0)
    for self_col in range(n_readout):
        a, b = _halves(n_readout, self_col)
        assert np.intersect1d(a, b).size == 0, f"self_col={self_col}: halves overlap"
        assert a.size == np.unique(a).size and b.size == np.unique(b).size
        union = np.sort(np.concatenate([a, b]))
        assert union.size == expected_total, (
            f"self_col={self_col}: halves cover {union.size} positions, "
            f"expected {expected_total}"
        )
        assert np.array_equal(union, np.array([j for j in range(n_readout) if j != self_col]))


@pytest.mark.parametrize("n_readout", BLOCK_SIZES)
def test_the_perturbed_genes_own_position_is_in_neither_half(n_readout: int) -> None:
    """C.EXCLUDE_SELF_POSITION is the design, so it is asserted as the design."""
    assert C.EXCLUDE_SELF_POSITION is True
    for self_col in range(n_readout):
        a, b = _halves(n_readout, self_col)
        assert self_col not in a.tolist()
        assert self_col not in b.tolist()


def test_a_four_position_block_yields_a_one_position_half() -> None:
    """The smallest block: one half pools a single position, and must still pool it.

    This is the edge the estimator is most likely to drop on the floor, and it is
    reachable --- the smoke path runs eight perturbations, and any block whose
    self position lands in the smaller half hits the same code.
    """
    for self_col in range(4):
        a, b = _halves(4, self_col)
        assert sorted([a.size, b.size]) == [1, 2]
        assert a.size + b.size == 3


def test_the_split_is_balanced_within_one_position() -> None:
    """A "half" that is not half makes the two pooled vectors incomparable.

    They are compared to each other by cosine distance, so a systematic size
    difference is a systematic variance difference between query and database.
    """
    for n in BLOCK_SIZES:
        assignment = P.half_assignment(n)
        counts = np.bincount(np.asarray(assignment).astype(int), minlength=2)
        assert abs(int(counts[0]) - int(counts[1])) <= 1, f"n={n}: sizes {counts}"
        assert int(counts.sum()) == n


def test_the_split_is_reproducible_at_the_configured_seed() -> None:
    """Same seed, same split --- and the default really is HALF_SPLIT_SEED."""
    first = np.asarray(P.half_assignment(524))
    second = np.asarray(P.half_assignment(524, seed=C.HALF_SPLIT_SEED))
    assert np.array_equal(first, second)
    assert C.HALF_SPLIT_SEED == C.SEED + 101


def test_a_different_seed_gives_a_different_split() -> None:
    """Otherwise the seed is decorative and the split is not a random split."""
    a = np.asarray(P.half_assignment(524, seed=C.HALF_SPLIT_SEED))
    b = np.asarray(P.half_assignment(524, seed=C.HALF_SPLIT_SEED + 1))
    assert not np.array_equal(a, b)


def test_a_self_position_outside_the_block_fails_loudly() -> None:
    """A perturbed table joined against the wrong read-out table must raise."""
    assignment = P.half_assignment(24)
    with pytest.raises(RuntimeError):
        P.half_indices(assignment, 24)
    with pytest.raises(RuntimeError):
        P.half_indices(assignment, -1)


# --------------------------------------------------------------------------- #
# the pooling estimator
# --------------------------------------------------------------------------- #


def _delta(n_readout: int = 24, d: int = 6, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_readout, d))


def test_pool_is_the_mean_of_the_half_it_pools_over() -> None:
    """Hand-computed against the same rows, so the estimator is arithmetic, not lore."""
    delta = _delta()
    assignment = P.half_assignment(delta.shape[0])
    self_col = 3
    idx_a, idx_b = P.half_indices(assignment, self_col)
    out = P.pool_query(delta, assignment, self_col)
    np.testing.assert_allclose(out.pool[0], delta[idx_a].mean(axis=0), rtol=1e-12, atol=0)
    np.testing.assert_allclose(out.pool[1], delta[idx_b].mean(axis=0), rtol=1e-12, atol=0)
    assert out.pool.shape == (C.N_POSITION_HALVES, delta.shape[1])
    assert out.n_pooled == (int(idx_a.size), int(idx_b.size))


def test_the_self_row_cannot_reach_the_pooled_vector() -> None:
    """The strongest form of the exclusion test: make the self row enormous.

    If the self position leaks into either half, a value 1e6 times the others
    dominates the mean and the assertion cannot pass by luck.
    """
    delta = _delta()
    self_col = 11
    assignment = P.half_assignment(delta.shape[0])
    clean = P.pool_query(delta, assignment, self_col)
    loud = delta.copy()
    loud[self_col] = 1e6
    contaminated = P.pool_query(loud, assignment, self_col)
    np.testing.assert_allclose(contaminated.pool, clean.pool, rtol=0, atol=0)
    np.testing.assert_allclose(contaminated.pool_unit, clean.pool_unit, rtol=0, atol=0)


def test_the_per_position_norm_is_the_norm_of_that_position() -> None:
    """S1's numerator and denominator both read out of this array, self included."""
    delta = _delta()
    self_col = 5
    out = P.pool_query(delta, P.half_assignment(delta.shape[0]), self_col)
    np.testing.assert_allclose(
        out.norm, np.linalg.norm(delta, axis=1), rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(out.self_delta, delta[self_col], rtol=0, atol=0)


def test_a_zero_delta_pools_to_exactly_zero_and_stays_finite() -> None:
    """The degenerate block: nothing moved, so nothing is pooled, and no nan appears.

    A nan here would reach the retrieval score as a distance, and a rank computed
    from nans is decided by sort order rather than by the data --- the exact way
    a dead layer comes back looking like a signal.
    """
    n_readout, d = 24, 6
    delta = np.zeros((n_readout, d))
    self_col = 2
    out = P.pool_query(delta, P.half_assignment(n_readout), self_col)
    assert np.all(out.pool == 0.0)
    assert np.all(out.pool_unit == 0.0)
    assert np.isfinite(out.pool).all() and np.isfinite(out.pool_unit).all()
    assert out.n_zero_positions == n_readout - 1, (
        f"every off-target position was zero; {out.n_zero_positions} were counted, "
        f"expected {n_readout - 1}"
    )


def test_a_zero_position_is_counted_rather_than_averaged_as_a_direction() -> None:
    """Three dead positions among live ones: counted, and contributing no direction."""
    n_readout, d = 24, 6
    delta = _delta(n_readout, d)
    dead = [0, 7, 19]
    self_col = 4
    assert self_col not in dead
    delta[dead] = 0.0
    out = P.pool_query(delta, P.half_assignment(n_readout), self_col)
    assert out.n_zero_positions == len(dead)
    assert np.isfinite(out.pool_unit).all()


def test_unit_rows_maps_a_dead_position_to_zero_not_to_nan() -> None:
    unit, norm, zero = P.unit_rows(np.array([[0.0, 0.0], [3.0, 4.0]]))
    np.testing.assert_allclose(unit[0], [0.0, 0.0], rtol=0, atol=0)
    np.testing.assert_allclose(unit[1], [0.6, 0.8], rtol=1e-12, atol=0)
    np.testing.assert_allclose(norm, [0.0, 5.0], rtol=1e-12, atol=0)
    assert zero.tolist() == [True, False]


def test_a_batched_delta_is_refused_rather_than_pooled() -> None:
    """Pooling a batch would mix two perturbations into one signature."""
    with pytest.raises(RuntimeError):
        P.pool_query(np.zeros((2, 24, 6)), P.half_assignment(24), 0)


def test_a_mismatched_half_assignment_is_refused() -> None:
    """Delta-h and the assignment built from different read-out tables must not pool."""
    with pytest.raises(RuntimeError):
        P.pool_query(_delta(24, 6), P.half_assignment(23), 0)


# --------------------------------------------------------------------------- #
# the no-op control (C3), at the estimator level
# --------------------------------------------------------------------------- #


def test_a_no_op_is_recognised_at_exactly_zero_and_not_within_a_tolerance() -> None:
    """C3 is an exact claim: two bit-identical inputs through a frozen model.

    Absorbing 1e-18 into a tolerance would hide a pass that differed in something
    other than the one edited value --- which is a wiring bug, not noise.
    """
    assert P.is_no_op(np.zeros((24, 6))) is True
    almost = np.zeros((24, 6))
    almost[3, 1] = 1e-18
    assert P.is_no_op(almost) is False


def test_a_no_op_is_recognised_when_the_self_position_is_undefined():
    """A genuine no-op must not read as a wiring bug on a model with a nan self.

    TranscriptFormer never reads a gene's count at its own position, so the driver
    writes `nan` there. `is_no_op` compares `|delta| <= 0`, and `nan` fails that,
    so without excluding the self position every genuine no-op on that model
    aborts the block claiming two bit-identical inputs disagreed.

    This is not hypothetical: it killed a 524-gene TranscriptFormer sweep at the
    `mean_control` anchor, where the model's own count clip at 30 CP10K makes the
    64 CP10K query a no-op for any gene already above the clip.

    The exclusion is by READ-OUT index (axis 0 of an `(n_readout, d_model)`
    delta). Deleting along the last axis instead drops a hidden dimension, leaves
    the nan in place, and reproduces the abort — which is how it was first written.
    """
    n_readout, d_model, self_col = 10, 4, 3
    delta = np.zeros((n_readout, d_model))
    delta[self_col] = np.nan

    assert P.is_no_op(delta, self_col=self_col)
    assert not P.is_no_op(delta)

    # a real disagreement anywhere else must still be caught, nan or no nan
    leaked = delta.copy()
    leaked[7, 1] = 1e-9
    assert not P.is_no_op(leaked, self_col=self_col)

    # and the ordinary case is unaffected
    assert P.is_no_op(np.zeros((n_readout, d_model)), self_col=self_col)

    # a wrongly-shaped delta must raise rather than silently delete a hidden dim
    with pytest.raises(RuntimeError, match="n_readout, d_model"):
        P.is_no_op(np.zeros((2, n_readout, d_model)), self_col=self_col)
