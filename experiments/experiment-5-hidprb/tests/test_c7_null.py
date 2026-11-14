"""The C7 null is a group-label permutation, however it happens to be written.

`analysis._null_max_retrieval` and `analysis.heldout_cell_layer` both build C7's
shuffled-label band by permuting the DATABASE and leaving the group labels alone.
Read literally that looks like an indexing bug --- column ``j`` of the permuted
distance matrix holds perturbation ``perm[j]``, so its group is
``groups[perm][j]`` and not ``groups[j]``. It is not a bug, and this file is the
reason anyone can believe that.

Permuting the database does not break C7. It breaks the strict score, because
`rank_of_own` takes the diagonal as truth and a permutation moves a different
perturbation onto the diagonal. C7's truth is the response group of whatever came
back nearest, and the permutation leaves the geometry --- hence what comes back
nearest --- untouched. So relabelling the database consistently returns the
query's own group and the null is 1.000 by construction, which measures nothing.

Leaving the labels unpermuted instead makes the retrieved column index an
effectively uniform perturbation, which is exactly the null C7 wants: what does
group-collapsed top-1 score when group membership carries no information. That
quantity has a closed form --- the probability two independent draws land in one
group, ``sum(p_g^2)`` --- and these tests pin the implementation to it.

Torch-free; runs anywhere.
"""

from __future__ import annotations

import numpy as np

from scfm_probe.analysis import _group_collapsed_hits


def _toy(n: int = 240, n_groups: int = 12, d: int = 16, seed: int = 7):
    """A geometry where every query's nearest database column is its own.

    Built rather than mocked: the equivalence being tested is a statement about
    what a retrieval does, so the retrieval has to actually happen. Well-separated
    random directions give an exact identity retrieval, which is the regime the
    real blocks are in (top-1 = 1.000) and the regime where a vacuous null is
    hardest to notice.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    groups = np.array([f"g{i % n_groups}" for i in range(n)])
    return x, groups


def _collision_rate(groups: np.ndarray) -> float:
    _, inv = np.unique(groups, return_inverse=True)
    p = np.bincount(inv) / groups.size
    return float((p**2).sum())


def test_identity_retrieval_scores_one_on_both_strict_and_c7():
    """The premise: without any permutation this geometry retrieves perfectly."""
    x, groups = _toy()
    dist = 1.0 - x @ x.T
    assert _group_collapsed_hits(dist, groups, 1).mean() == 1.0


def test_permuting_database_and_labels_together_is_vacuous():
    """The combination that must never be used: the null collapses to 1.000.

    This is the test that would have caught the 2026-08-31 mistake, which passed
    `verify` (every row had *a* null) while the null itself measured nothing.
    """
    x, groups = _toy()
    perm = np.random.default_rng(1).permutation(x.shape[0])
    dist = 1.0 - x @ x[perm].T
    both = _group_collapsed_hits(dist, groups, 1, database_groups=groups[perm]).mean()
    assert both == 1.0, both


def test_database_permutation_matches_label_permutation():
    """The two ways of writing the null agree, and agree with the closed form.

    Averaged over draws rather than asserted per draw: each is one sample of the
    same distribution, so a single draw would make this test a coin flip.
    """
    x, groups = _toy()
    n = x.shape[0]
    dist0 = 1.0 - x @ x.T
    as_db, as_labels = [], []
    for s in range(40):
        perm = np.random.default_rng(1000 + s).permutation(n)
        as_db.append(_group_collapsed_hits(1.0 - x @ x[perm].T, groups, 1).mean())
        as_labels.append(
            _group_collapsed_hits(dist0, groups, 1, database_groups=groups[perm]).mean()
        )
    base = _collision_rate(groups)
    assert abs(float(np.mean(as_db)) - base) < 0.02, (np.mean(as_db), base)
    assert abs(float(np.mean(as_labels)) - base) < 0.02, (np.mean(as_labels), base)
    assert abs(float(np.mean(as_db)) - float(np.mean(as_labels))) < 0.02


def test_database_groups_length_is_checked():
    """A wrong-length label array is a raise, not a silently misaligned control."""
    x, groups = _toy(n=40, n_groups=4)
    dist = 1.0 - x @ x.T
    try:
        _group_collapsed_hits(dist, groups, 1, database_groups=groups[:10])
    except RuntimeError as exc:
        assert "database group labels" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a short database_groups must raise")
