"""The estimator against two toy models whose answer is known before it runs.

This is the acceptance gate the plan puts before any GPU hour. The failure it
guards against is the one that does not raise: a probe that reports "identity is
decodable" whatever it is fed. Feeding it only a signal cannot detect that, so
the gate is a pair --- a model that propagates, where the probe MUST return
top-1 = 1.0, and the same arithmetic with the cross-gene route deleted, where the
probe MUST NOT.

The controls the plan pre-registers are checked here in the same idiom, because
each is a statement about a number that is known in advance:

C1  the intervention lands --- the perturbed gene's own token moves, in BOTH
    toys, so ``IsolatedToy``'s zero off-target delta is a real absence of
    propagation and not a dead intervention.
C2  layer 0 is the post-embedding state, so its off-target delta is exactly 0.0
    in both toys and the retrieval read there must not be a hit.
C3  a no-op intervention --- setting a gene to the value it already has --- gives
    exactly zero delta everywhere, all layers, and is recognised as a no-op
    rather than averaged in.

Invocation (from the run directory):  PYTHONPATH=src pytest tests/test_toys.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scfm_probe import config as C
from scfm_probe.probe import pool as P
from scfm_probe.probe import retrieval as R
from toy import IsolatedToy, MuteToy, PropagatingToy

#: an anchor every read-out gene sits well above zero in, so the zero-out query
#: is a real edit at every gene --- the property `mean_control` has and
#: `individual_cells` does not
BASE_CP10K = np.random.default_rng(0).uniform(1.0, 50.0, size=PropagatingToy.n_readout)


def _delta(model, layer: int, base: np.ndarray, self_col: int, value: float) -> np.ndarray:
    """One probe query's ``(n_readout, d)`` delta-h, perturbed minus baseline.

    Both passes go through ``hidden_batch`` with the same ``perturbed_col``, which
    is the discipline the real adapters need for their tile and chunk composition
    to match between the two passes.
    """
    perturbed = base.copy()
    perturbed[self_col] = value
    x = np.stack([model.to_native(perturbed), model.to_native(base)])
    h = model.hidden_batch(x, [layer], perturbed_col=self_col)[int(layer)]
    return h[0].astype(np.float64) - h[1].astype(np.float64)


def _query_matrices(model, layer: int, value: float = C.PRIMARY_VALUE_CP10K):
    """Every perturbation's half-A and half-B signature, plus its norm row."""
    n = model.n_readout
    assignment = P.half_assignment(n)
    query = np.zeros((n, model.d_model))
    database = np.zeros((n, model.d_model))
    norm = np.zeros((n, 1, n))
    for i in range(n):
        summary = P.pool_query(_delta(model, layer, BASE_CP10K, i, value), assignment, i)
        query[i] = summary.pool[0]
        database[i] = summary.pool[1]
        norm[i, 0] = summary.norm
    return query, database, norm, np.arange(n)


def _off_target(delta: np.ndarray, self_col: int) -> np.ndarray:
    return np.delete(delta, self_col, axis=0)


# --------------------------------------------------------------------------- #
# the planted-signal toy: the probe must recover identity
# --------------------------------------------------------------------------- #


def test_the_propagating_toy_moves_the_other_genes_tokens() -> None:
    """Before asking whether identity is decodable, check anything arrived at all."""
    toy = PropagatingToy()
    delta = _delta(toy, toy.n_layers, BASE_CP10K, 5, C.PRIMARY_VALUE_CP10K)
    assert np.abs(_off_target(delta, 5)).max() > 0.0


def test_the_propagating_toy_is_decoded_at_top1_of_one() -> None:
    """The known answer: an edit at gene i writes a direction that identifies i.

    The toy plants a distinct direction per perturbed gene, so a probe that
    pools, normalises and ranks correctly retrieves every one of them. Anything
    below 1.0 here is the estimator losing a signal that is present by
    construction --- not a property of any model.
    """
    toy = PropagatingToy()
    query, database, _, _ = _query_matrices(toy, toy.n_layers)
    out = R.retrieval_scores(query, database)
    assert out["top1"] == 1.0, f"planted identity was not recovered: {out['top1']}"
    assert out["mrr"] == 1.0
    assert out["n_degenerate"] == 0


def test_the_probe_reads_the_layer_argument() -> None:
    """The toy's cross-gene term grows with depth, so the depth profile must too.

    A probe whose number is identical at every layer is not hooking the layer it
    says it is, and a flat profile is exactly the shape this run would read as
    "the intervention does not propagate".
    """
    toy = PropagatingToy()
    ratios = []
    for layer in (C.EMBEDDING_LAYER, 1, toy.n_layers):
        _, _, norm, self_cols = _query_matrices(toy, layer)
        ratios.append(R.propagation_ratio(norm, self_cols)["median_ratio"])
    assert ratios[0] == 0.0, "layer 0 propagated something, so the hook is misplaced"
    assert ratios[1] < ratios[2], f"the depth profile is flat: {ratios}"


# --------------------------------------------------------------------------- #
# the no-propagation toy: the null control that can fail
# --------------------------------------------------------------------------- #


def test_the_isolated_toy_off_target_delta_is_exactly_zero_at_every_layer() -> None:
    """The cross-gene route is deleted, so nothing can arrive --- at any depth.

    Exact zero, not a tolerance: the two passes differ in one input value and the
    arithmetic that carries it to another gene is absent, so any nonzero entry
    means the delta picked up something other than the intervention.
    """
    toy = IsolatedToy()
    for layer in [C.EMBEDDING_LAYER, *range(1, toy.n_layers + 1)]:
        for self_col in (0, 7, toy.n_readout - 1):
            delta = _delta(toy, layer, BASE_CP10K, self_col, C.PRIMARY_VALUE_CP10K)
            assert np.abs(_off_target(delta, self_col)).max() == 0.0, (
                f"layer {layer}, gene {self_col}: an edit reached a gene it has no "
                "route to"
            )


def test_the_isolated_toy_intervention_still_lands_on_its_own_token() -> None:
    """C1 on the null toy: the zero above is absent propagation, not a dead edit.

    Without this, ``IsolatedToy`` would pass its own test by doing nothing at all,
    and the pair would no longer be an acceptance gate.
    """
    toy = IsolatedToy()
    for layer in [C.EMBEDDING_LAYER, toy.n_layers]:
        delta = _delta(toy, layer, BASE_CP10K, 5, C.PRIMARY_VALUE_CP10K)
        assert abs(delta[5]).max() > 0.0, f"layer {layer}: the perturbed token did not move"


def test_the_isolated_toy_is_not_reported_as_decodable() -> None:
    """The test the whole run rests on: no propagation must not read as a hit.

    Every pooled signature is the zero vector, so the block is degenerate --- it
    must be counted as such and its score must sit at the chance level of 1/n,
    never above it.
    """
    toy = IsolatedToy()
    for layer in [C.EMBEDDING_LAYER, 1, toy.n_layers]:
        query, database, _, _ = _query_matrices(toy, layer)
        assert np.abs(query).max() == 0.0 and np.abs(database).max() == 0.0
        out = R.retrieval_scores(query, database)
        assert out["n_degenerate"] == toy.n_readout
        assert out["degenerate"] is True
        assert out["top1"] <= 8.0 * out["chance_top1"], (
            f"layer {layer}: a model with no cross-gene route reported top1 "
            f"{out['top1']} against chance {out['chance_top1']}"
        )


# --------------------------------------------------------------------------- #
# C2 --- layer 0, in both toys
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("toy_cls", [PropagatingToy, IsolatedToy])
def test_layer_zero_has_no_off_target_delta_and_no_retrieval(toy_cls) -> None:
    """The post-embedding state depends on a gene's own value and nothing else."""
    toy = toy_cls()
    query, database, norm, self_cols = _query_matrices(toy, C.EMBEDDING_LAYER)
    masked = R.off_target_norms(norm, self_cols)
    assert float(np.nanmax(masked)) == 0.0, "layer 0 carried an off-target delta"
    out = R.retrieval_scores(query, database)
    assert out["degenerate"] is True
    assert out["top1"] <= 8.0 * out["chance_top1"], (
        f"layer 0 reported top1 {out['top1']}, which cannot come from the data"
    )


# --------------------------------------------------------------------------- #
# C3 --- the no-op intervention
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("toy_cls", [PropagatingToy, IsolatedToy])
def test_a_no_op_intervention_gives_exactly_zero_delta_everywhere(toy_cls) -> None:
    """Setting a gene to the value it already has: two bit-identical passes.

    Zero at every position including the perturbed gene's own, at every layer.
    Anything else means the two passes differed in something other than the one
    value --- a leak the rest of the run would silently inherit.
    """
    toy = toy_cls()
    for self_col in (0, 11, toy.n_readout - 1):
        for layer in [C.EMBEDDING_LAYER, *range(1, toy.n_layers + 1)]:
            delta = _delta(toy, layer, BASE_CP10K, self_col, float(BASE_CP10K[self_col]))
            assert np.abs(delta).max() == 0.0, f"no-op leaked at layer {layer}"
            assert P.is_no_op(delta) is True


def test_a_no_op_query_is_counted_rather_than_scored_as_propagation() -> None:
    """A no-op has an undefined ratio --- zero over zero --- and must be counted."""
    toy = PropagatingToy()
    n = toy.n_readout
    assignment = P.half_assignment(n)
    norm = np.zeros((n, 1, n))
    for i in range(n):
        delta = _delta(toy, toy.n_layers, BASE_CP10K, i, float(BASE_CP10K[i]))
        norm[i, 0] = P.pool_query(delta, assignment, i).norm
    out = R.propagation_ratio(norm, np.arange(n))
    assert out["n_undefined"] == n
    assert out["n_defined"] == 0
    assert np.isnan(out["ratio"]).all()


# --------------------------------------------------------------------------- #
# a measured absence, never a silent one
# --------------------------------------------------------------------------- #


def test_a_model_that_cannot_supply_hidden_states_says_so_verbatim() -> None:
    """``None`` plus a reason, mirroring run 2's ``exact_jvp`` / ``_jvp_error``.

    The alternative --- an array of zeros --- is indistinguishable downstream from
    a model that propagates nothing, which is exactly this run's headline finding.
    A missing measurement must never be able to become one.
    """
    toy = MuteToy()
    assert not hasattr(toy, "_hidden_error")
    out = toy.hidden_batch(np.zeros((1, toy.n_readout)), [1], perturbed_col=0)
    assert out is None
    assert isinstance(toy._hidden_error, str) and toy._hidden_error.strip()


# --------------------------------------------------------------------------- #
# arrays are joined on their labels, not on row order
# --------------------------------------------------------------------------- #


def test_permuting_the_perturbation_order_does_not_move_the_scores() -> None:
    """Row order is a contract, so re-ordering the rows must be a no-op on the score.

    A score that changes under permutation is reading position rather than
    identity, which is how a label-alignment bug reaches a published number.
    """
    toy = PropagatingToy()
    query, database, _, _ = _query_matrices(toy, toy.n_layers)
    perm = np.random.default_rng(21).permutation(toy.n_readout)
    straight = R.retrieval_scores(query, database)
    shuffled = R.retrieval_scores(query[perm], database[perm])
    assert shuffled["top1"] == straight["top1"]
    assert shuffled["mrr"] == straight["mrr"]
    assert np.array_equal(np.asarray(shuffled["ranks"]), np.asarray(straight["ranks"])[perm])


def test_permuting_only_one_half_destroys_the_match() -> None:
    """The mirror of the test above: alignment is load-bearing, so breaking it must show.

    If a permuted database still scored at 1.0, the previous test would be
    passing for a reason other than the one it claims.
    """
    toy = PropagatingToy()
    query, database, _, _ = _query_matrices(toy, toy.n_layers)
    perm = np.roll(np.arange(toy.n_readout), 1)
    out = R.retrieval_scores(query, database[perm])
    assert out["top1"] == 0.0
