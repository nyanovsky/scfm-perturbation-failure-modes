"""S2 (split-half retrieval) and S1 (propagation), checked against known answers.

Failure class this guards against is the one that would make this run publish a
result it did not measure: a ranking rule that reports a hit on a block carrying
no signal. Two ways in. A dead layer makes every pooled vector the zero vector
and every cosine distance identical, so the tie rule alone decides the number ---
break ties by index and the diagonal wins every one of them, and a model that
does nothing reports top-1 = 1.0. And a self position that never moved makes S1's
denominator zero, so a ratio of zero over zero must be undefined and counted, not
averaged in as a small number or reported as infinity.

Every assertion here is an arithmetic answer worked out in the test --- a
constructed rank-2 database gives MRR exactly 0.5, a deranged database gives
top-1 exactly 0.0, a sign-flipped database puts the true match last --- rather
than a number read back from the estimator.

Written from the run 5 plan's frozen interface.

Invocation (from the run directory):  PYTHONPATH=src pytest tests/test_retrieval.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scfm_probe import config as C
from scfm_probe.probe import retrieval as R


def _rng(seed: int = 3) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# S2 --- the cases where the right answer is known in advance
# --------------------------------------------------------------------------- #


def test_a_perfect_match_scores_top1_of_one() -> None:
    """The positive end of the estimator's range, and the only one it can reach."""
    v = _rng().normal(size=(32, 8))
    out = R.retrieval_scores(v, v)
    assert out["top1"] == 1.0
    assert out["top10"] == 1.0
    assert out["mrr"] == 1.0
    assert np.all(out["ranks"] == 1)
    assert out["n_degenerate"] == 0
    assert out["degenerate"] is False


def test_a_rescaled_database_still_matches_because_the_distance_is_cosine() -> None:
    """C6's premise: identity must be carried by direction, not by magnitude.

    If a per-row rescaling broke the match, the score would be reading the
    perturbed gene's own expression level rather than where the edit went.
    """
    v = _rng().normal(size=(32, 8))
    scale = _rng(11).uniform(0.01, 100.0, size=(32, 1))
    out = R.retrieval_scores(v, v * scale)
    assert out["top1"] == 1.0


def test_a_sign_flipped_database_puts_the_true_match_last() -> None:
    """A direction and its negation are not the same signature, and must not rank as one."""
    v = _rng().normal(size=(16, 8))
    out = R.retrieval_scores(v, -v)
    assert out["top1"] == 0.0
    assert np.all(out["ranks"] == 16), f"ranks were {out['ranks']}"


def test_a_deranged_database_scores_exactly_zero_top1() -> None:
    """Every row's signature belongs to a different perturbation: no hit is possible."""
    v = _rng(5).normal(size=(40, 8))
    out = R.retrieval_scores(v, np.roll(v, 1, axis=0))
    assert out["top1"] == 0.0


def test_an_unrelated_database_sits_at_its_chance_level() -> None:
    """The null that has to be able to fail: independent draws must not retrieve."""
    n = 200
    q = _rng(1).normal(size=(n, 16))
    d = _rng(2).normal(size=(n, 16))
    out = R.retrieval_scores(q, d)
    assert out["chance_top1"] == 1.0 / n
    assert out["top1"] <= 10.0 / n, f"top1 {out['top1']} is far above chance {1.0 / n}"
    assert out["mrr"] <= 5.0 * out["chance_mrr"]


def test_a_constructed_rank_two_database_gives_mrr_of_exactly_one_half() -> None:
    """Hand-built so the true match is second for every row, at every row.

    ``database`` is the identity, so row *j* is the unit vector e_j. Query *i* is
    ``0.9 e_i + 1.0 e_{i+1}``, whose cosine is larger to e_{i+1} than to e_i ---
    the true match is therefore rank 2, giving MRR 1/2, top-1 0 and top-10 1
    without a single number coming back from the estimator.
    """
    n = 20
    database = np.eye(n)
    query = np.zeros((n, n))
    for i in range(n):
        query[i, i] = 0.9
        query[i, (i + 1) % n] = 1.0
    out = R.retrieval_scores(query, database)
    assert np.all(out["ranks"] == 2)
    assert out["top1"] == 0.0
    assert out["top10"] == 1.0
    assert out["mrr"] == pytest.approx(0.5, abs=1e-12)
    assert out["median_rank"] == 2.0
    assert out["normalized_rank"] == pytest.approx(1.0 - 1.0 / (n - 1), abs=1e-12)


def test_the_chance_levels_are_the_pre_registered_ones() -> None:
    """1/n for top-1, k/n for top-k --- the levels the plan reads scores against."""
    v = _rng(9).normal(size=(50, 8))
    out = R.retrieval_scores(v, v)
    assert C.TOPK == (1, 10)
    assert out["chance_top1"] == 1.0 / 50
    assert out["chance_top10"] == pytest.approx(10.0 / 50)
    assert out["n_perturbations"] == 50


# --------------------------------------------------------------------------- #
# ties, and the degenerate block that makes them load-bearing
# --------------------------------------------------------------------------- #


def test_an_exact_tie_is_not_won_by_the_diagonal() -> None:
    """Sixty-four pairs of identical database rows: the true match ties with its twin.

    An index tie-break hands every one of those ties to the diagonal and reports
    top-1 = 1.0 on data that cannot distinguish the pair. The probability of a
    fair tie-break producing 1.0 here is 2**-64, so this cannot pass by luck.
    """
    n = 64
    base = _rng(4).normal(size=(n // 2, 8))
    database = np.repeat(base, 2, axis=0)
    out = R.retrieval_scores(database.copy(), database)
    assert out["top1"] < 1.0, "an exact tie was resolved in the true match's favour"
    assert 0.25 <= out["top1"] <= 0.75, f"tied top1 {out['top1']} is not near 1/2"


def test_the_score_is_reproducible_at_the_configured_tie_seed() -> None:
    """A random tie-break still has to give the same number twice."""
    n = 64
    database = np.repeat(_rng(4).normal(size=(n // 2, 8)), 2, axis=0)
    first = R.retrieval_scores(database.copy(), database)
    second = R.retrieval_scores(database.copy(), database, seed=C.RETRIEVAL_TIE_SEED)
    assert first["top1"] == second["top1"]
    assert np.array_equal(first["ranks"], second["ranks"])


def test_a_dead_block_is_counted_and_cannot_read_as_a_signal() -> None:
    """The single most important test here: nothing moved, so nothing was decoded.

    Every pooled vector is exactly zero, so every distance is identical and the
    block carries no information at all. The result must say so --- the count of
    degenerate rows is the whole block --- and the score must not come back as a
    hit. A silent 1.0 here is the failure that would turn a null into a finding.
    """
    n = 64
    zeros = np.zeros((n, 8))
    out = R.retrieval_scores(zeros, zeros)
    assert out["n_degenerate"] == n, "a block with no signature was not counted"
    assert out["degenerate"] is True
    assert out["top1"] != 1.0
    assert out["top1"] <= 8.0 / n, (
        f"a block whose delta-h is identically zero reported top1 {out['top1']}, "
        f"which is above its chance level {1.0 / n}"
    )


def test_dead_rows_among_live_ones_are_counted_and_are_never_free_hits() -> None:
    """Five perturbations that did not move, fifteen that did.

    The live rows must still resolve exactly; the dead ones must be counted, and
    must not all be handed a rank of 1. Both conventions the plan leaves open ---
    keeping the dead rows in the comparison or dropping them from it --- satisfy
    that, and the test asserts the substance rather than picking between them.
    """
    n, d = 20, 8
    v = _rng(6).normal(size=(n, d))
    dead = np.array([0, 3, 9, 14, 19])
    live = np.setdiff1d(np.arange(n), dead)
    v[dead] = 0.0
    out = R.retrieval_scores(v, v.copy())
    assert out["n_degenerate"] == dead.size
    ranks = np.asarray(out["ranks"])
    if ranks.size == n:
        assert np.all(ranks[live] == 1), "a live perturbation failed to retrieve itself"
        assert not np.all(ranks[dead] == 1), "dead rows were all counted as top-1 hits"
    else:
        assert ranks.size == n - dead.size
        assert np.all(ranks == 1)


def test_a_single_perturbation_block_is_refused_rather_than_scored() -> None:
    """With one class, top-1 is 1.0 by construction and means nothing.

    Reporting it would be a spurious perfect score; the block has no chance level
    to be read against, so it must fail loudly instead.
    """
    with pytest.raises(RuntimeError):
        R.retrieval_scores(np.ones((1, 8)), np.ones((1, 8)))


def test_misaligned_labels_are_refused_rather_than_assumed() -> None:
    """The diagonal is only the true match if the two halves are in the same order."""
    v = _rng(8).normal(size=(10, 4))
    with pytest.raises(RuntimeError):
        R.retrieval_scores(v, v, labels=np.array([f"g{i}" for i in range(9)]))
    with pytest.raises(RuntimeError):
        R.cosine_distance(v, _rng(8).normal(size=(10, 5)))


def test_group_collapsed_is_reported_beside_the_strict_score_never_instead(
) -> None:
    """C7: paralogs counted as hits only in the control, never in the headline.

    Each query is its own partner's signature, so the strict score is 0 by
    construction; the partner shares a response group, so the collapsed score is
    1. Both keys must be present, with the strict one unchanged.
    """
    n = 16
    base = _rng(12).normal(size=(n // 2, 8))
    database = np.repeat(base, 2, axis=0) + 1e-3 * _rng(13).normal(size=(n, 8))
    partner = np.array([i + 1 if i % 2 == 0 else i - 1 for i in range(n)])
    query = database[partner].copy()
    groups = np.repeat(np.arange(n // 2), 2)
    out = R.retrieval_scores(query, database, groups=groups)
    assert out["top1"] == 0.0
    assert out["group_collapsed_top1"] == 1.0
    assert out["group_collapsed_top10"] == 1.0


def test_a_wrong_length_group_vector_is_refused() -> None:
    v = _rng(14).normal(size=(10, 4))
    with pytest.raises(RuntimeError):
        R.retrieval_scores(v, v, groups=np.arange(9))


# --------------------------------------------------------------------------- #
# S1 --- propagation
# --------------------------------------------------------------------------- #


def _norms(rows: list[list[float]]) -> np.ndarray:
    """(n_perturbed, 1 value, n_readout) per-position norms."""
    return np.asarray(rows, dtype=np.float64)[:, None, :]


def test_the_propagation_ratio_is_the_off_target_length_over_the_self_length() -> None:
    """Norms 4 at the self position and 3, 4, 0 elsewhere: 5 / 4 = 1.25, exactly."""
    norm = _norms([[4.0, 3.0, 4.0, 0.0]])
    out = R.propagation_ratio(norm, np.array([0]))
    assert out["ratio"].shape == (1, 1)
    assert out["ratio"][0, 0] == pytest.approx(1.25, abs=1e-12)
    assert out["off_target_norm"][0, 0] == pytest.approx(5.0, abs=1e-12)
    assert out["self_norm"][0, 0] == 4.0
    assert out["n_defined"] == 1
    assert out["n_undefined"] == 0


def test_a_layer_that_did_not_propagate_gives_a_ratio_of_exactly_zero() -> None:
    """Control C2: the self token moved, nothing else did, and the ratio says 0.0.

    Not "approximately 0.0" --- at layer 0 the off-target delta is zero by
    construction, and a tolerance here would let a mis-attributed hook through.
    """
    norm = _norms([[4.0, 0.0, 0.0, 0.0]])
    out = R.propagation_ratio(norm, np.array([0]))
    assert out["ratio"][0, 0] == 0.0
    assert out["max_off_target_position_norm"] == 0.0
    assert out["median_ratio"] == 0.0


def test_a_self_position_that_never_moved_is_undefined_not_infinite() -> None:
    """Control C1 failing on one perturbation: counted and excluded, never averaged.

    Zero over zero is a dead intervention, which is a bug to be reported, not a
    ratio to be reported as very large or very small.
    """
    norm = _norms([[0.0, 2.0, 2.0, 2.0], [4.0, 0.0, 0.0, 0.0]])
    out = R.propagation_ratio(norm, np.array([0, 0]))
    assert np.isnan(out["ratio"][0, 0])
    assert np.isfinite(out["ratio"][1, 0])
    assert out["n_undefined"] == 1
    assert out["n_defined"] == 1
    assert not np.isinf(out["ratio"]).any()


def test_the_ratio_does_not_move_when_every_norm_is_rescaled() -> None:
    """A dimensionless ratio is what makes layers of different width comparable."""
    norm = _norms([[4.0, 3.0, 4.0, 1.0], [2.0, 1.0, 0.5, 0.25]])
    a = R.propagation_ratio(norm, np.array([0, 0]))["ratio"]
    b = R.propagation_ratio(norm * 7.0, np.array([0, 0]))["ratio"]
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


def test_the_self_position_is_masked_out_of_the_off_target_norms() -> None:
    """Otherwise C2's maximum would pick up C1's deliberately large self entry."""
    norm = _norms([[9.0, 1.0, 2.0, 0.0], [1.0, 9.0, 0.0, 3.0]])
    masked = R.off_target_norms(norm, np.array([0, 1]))
    assert np.isnan(masked[0, 0, 0]) and np.isnan(masked[1, 0, 1])
    assert float(np.nanmax(masked)) == 3.0


def test_a_self_column_outside_the_block_is_refused() -> None:
    norm = _norms([[1.0, 2.0, 3.0, 4.0]])
    with pytest.raises(RuntimeError):
        R.propagation_ratio(norm, np.array([4]))
    with pytest.raises(RuntimeError):
        R.propagation_ratio(norm, np.array([0, 0]))
    with pytest.raises(RuntimeError):
        R.propagation_ratio(norm[0], np.array([0]))


def test_group_collapsed_cannot_be_won_by_array_order_on_a_dead_block():
    """C7 on a dead layer must not report row 0's group as everyone's answer.

    Every cosine distance is exactly 1.0 when nothing propagated, so a stable
    argsort puts perturbation 0 first in every row and the group-collapsed
    control silently becomes "what fraction of perturbations share row 0's
    response group" --- a number set by the arbitrary order of the perturbation
    table, on a layer that carries no information. The strict score reads chance
    there, so the two would disagree while both looked like results.

    The guard is that the answer must not depend on which perturbation happens to
    be first, so the same dead block is scored twice with its rows rotated.
    """
    n, d = 60, 8
    dead = np.zeros((n, d))
    labels = np.array([f"G{i:03d}" for i in range(n)])
    # one big group in front, the rest singletons: the shape that makes row 0's
    # group membership swing the statistic hardest
    groups = np.array([0] * 20 + list(range(1, n - 19)))

    a = R.retrieval_scores(dead, dead, labels=labels, groups=groups)
    roll = np.roll(np.arange(n), 17)
    b = R.retrieval_scores(
        dead, dead, labels=labels[roll], groups=groups[roll]
    )

    assert a["degenerate"] and b["degenerate"]
    assert a["top1"] == b["top1"]  # the strict score was always order-blind
    # the control must be too: a 20-member group in 60 perturbations cannot make
    # a third of a dead block "correct" just by sitting at index 0
    assert abs(a["group_collapsed_top1"] - b["group_collapsed_top1"]) < 0.15
    assert a["group_collapsed_top1"] < 0.5
