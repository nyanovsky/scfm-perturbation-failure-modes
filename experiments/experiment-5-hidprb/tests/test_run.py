"""The driver, end to end, on the toys --- the acceptance gate the plan names.

``probe_anchor`` is model-agnostic (it touches only ``to_native``, ``n_readout``
and ``hidden_batch``), which is what makes the whole estimator path runnable
against ``tests/toy.py`` with no GPU and no checkpoint. So it is run here, and
the two toys must split: the model that propagates a planted per-gene direction
must come back at top-1 = 1.0, and the same arithmetic with the cross-gene route
deleted must not. The pair is the gate. Without the second half, a driver that
reports "decodable" for everything passes.

The other failure this file guards against is a measured absence turning into a
measurement. An adapter that cannot supply hidden states returns ``None`` with a
reason; if the driver wrote a block of zeros instead, that block would be
indistinguishable downstream from a model whose intervention propagates nothing
--- which is this run's headline finding. A missing measurement must never be
able to become one.

Invocation (from the run directory):  PYTHONPATH=src pytest tests/test_run.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scfm_probe import config as C
from scfm_probe.probe import retrieval as R
from scfm_probe.probe import run as D
from toy import IsolatedToy, MuteToy, PropagatingToy

#: an anchor every read-out gene sits well above the grid on, so no grid value
#: is accidentally a no-op --- the property `mean_control` was chosen for
ANCHOR_CP10K = np.random.default_rng(0).uniform(100.0, 200.0, size=PropagatingToy.n_readout)

#: the whole read-out set is perturbed, one gene at a time, which is the layout
#: the real blocks use (524 of 524 in HepG2, 705 of 705 in Jurkat)
PERTURBED = pd.DataFrame(
    {
        "gene_id": [f"TOY{i:03d}" for i in range(PropagatingToy.n_readout)],
        "readout_col": np.arange(PropagatingToy.n_readout, dtype=np.int64),
    }
)

LAYERS = (C.EMBEDDING_LAYER, 1, PropagatingToy.n_layers)


def _probe(model, anchor: np.ndarray = ANCHOR_CP10K) -> dict:
    return D.probe_anchor(model, anchor, PERTURBED, LAYERS, log_every=0, tag="test")


def _halves_at(result: dict, layer: int, value_cp10k: float):
    """The two pooled signatures per perturbation, at one layer and one dose."""
    vi = list(C.VALUE_GRID_CP10K).index(value_cp10k)
    pool = result["arrays"][int(layer)]["pool"]
    return pool[:, vi, 0, :], pool[:, vi, 1, :]


# --------------------------------------------------------------------------- #
# the input rows
# --------------------------------------------------------------------------- #


def test_a_query_batch_changes_exactly_one_input_column() -> None:
    """Row 0 is the baseline; every other row differs from it at the perturbed gene only.

    A second changed column would make delta-h a sum of two interventions with no
    way to separate them, and nothing downstream could see it.
    """
    x0 = np.asarray(ANCHOR_CP10K, dtype=np.float32)
    col = 7
    batch = D.query_batch(x0, col, np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64))
    assert batch.shape == (1 + len(C.VALUE_GRID_CP10K), x0.size)
    np.testing.assert_allclose(batch[0], x0, rtol=0, atol=0)
    for vi, value in enumerate(C.VALUE_GRID_CP10K):
        row = batch[1 + vi]
        differs = np.flatnonzero(row != batch[0])
        assert differs.tolist() == [col], f"value {value}: rows differ at {differs.tolist()}"
        assert row[col] == np.float32(max(float(value), C.NATIVE_FLOOR))


def test_the_layer_plan_leads_with_the_control_layer() -> None:
    """Layer 0 is not a sweep point; it is the cheapest check on hook attribution."""
    assert D.layer_plan("scgpt") == (0, 1, 4, 7, 10, 12)
    assert D.layer_plan("scprint") == (0, 1, 4, 7, 8)
    assert D.layer_plan("transcriptformer")[0] == C.EMBEDDING_LAYER


# --------------------------------------------------------------------------- #
# the gate: the two toys must split
# --------------------------------------------------------------------------- #


def test_the_driver_recovers_planted_identity_at_top1_of_one() -> None:
    """Known answer: a distinct direction per perturbed gene, decodable by construction."""
    out = _probe(PropagatingToy())
    assert out["available"] is True
    a, b = _halves_at(out, PropagatingToy.n_layers, C.PRIMARY_VALUE_CP10K)
    scores = R.retrieval_scores(a, b, labels=out["perturbed_gene_id"])
    assert scores["top1"] == 1.0, f"planted identity was lost by the driver: {scores['top1']}"
    assert scores["mrr"] == 1.0
    assert scores["n_degenerate"] == 0


def test_the_driver_leaves_the_isolated_toy_at_chance() -> None:
    """The half of the gate that can fail: no route, so no identity, at every layer."""
    out = _probe(IsolatedToy())
    assert out["available"] is True
    for layer in LAYERS:
        a, b = _halves_at(out, layer, C.PRIMARY_VALUE_CP10K)
        assert np.abs(a).max() == 0.0 and np.abs(b).max() == 0.0
        scores = R.retrieval_scores(a, b)
        assert scores["degenerate"] is True
        assert scores["top1"] <= 8.0 * scores["chance_top1"], (
            f"layer {layer}: a model with no cross-gene route reported top1 "
            f"{scores['top1']} against chance {scores['chance_top1']}"
        )


def test_layer_zero_carries_no_off_target_delta_through_the_driver() -> None:
    """Control C2, read from the block the driver actually writes."""
    for toy_cls in (PropagatingToy, IsolatedToy):
        out = _probe(toy_cls())
        norm = out["arrays"][C.EMBEDDING_LAYER]["norm"].astype(np.float64)
        masked = R.off_target_norms(norm, out["readout_col"])
        assert float(np.nanmax(masked)) == 0.0, f"{toy_cls.__name__}: layer 0 propagated"


def test_the_self_position_moves_in_both_toys() -> None:
    """Control C1: the intervention lands, so a zero off-target delta is a real absence."""
    for toy_cls in (PropagatingToy, IsolatedToy):
        out = _probe(toy_cls())
        self_delta = out["arrays"][toy_cls.n_layers]["self_delta"]
        vi = list(C.VALUE_GRID_CP10K).index(C.PRIMARY_VALUE_CP10K)
        assert np.abs(self_delta[:, vi, :]).max() > 0.0, toy_cls.__name__


# --------------------------------------------------------------------------- #
# a measured absence stays an absence
# --------------------------------------------------------------------------- #


def test_a_model_with_no_hidden_states_is_recorded_not_zero_filled() -> None:
    """``None`` becomes an unavailable block carrying the adapter's verbatim reason.

    Zeros would be indistinguishable from a model that propagates nothing, which
    is exactly the result this run exists to report.
    """
    toy = MuteToy()
    out = _probe(toy)
    assert out["available"] is False
    assert "arrays" not in out
    assert out["reason"] == toy._hidden_error
    assert out["reason"].strip()


# --------------------------------------------------------------------------- #
# the no-op control, through the driver
# --------------------------------------------------------------------------- #


def test_a_no_op_query_is_counted_and_named_rather_than_averaged_in() -> None:
    """An anchor sitting exactly on a grid value makes that query two identical passes.

    The driver must recognise it, record which (gene, value) it was, and assert
    the delta is exactly zero --- not absorb it as a very small response.
    """
    anchor = ANCHOR_CP10K.copy()
    anchor[4] = 16.0
    assert 16.0 in C.VALUE_GRID_CP10K
    out = _probe(PropagatingToy(), anchor)
    diagnostics = out["diagnostics"]
    assert diagnostics["n_no_op_queries"] == 1
    assert diagnostics["no_op_queries"] == [
        {"perturbed_gene_id": "TOY004", "value_cp10k": 16.0}
    ]
    vi = list(C.VALUE_GRID_CP10K).index(16.0)
    for layer in LAYERS:
        arrays = out["arrays"][int(layer)]
        assert np.abs(arrays["pool"][4, vi]).max() == 0.0
        assert np.abs(arrays["norm"][4, vi]).max() == 0.0
        assert np.abs(arrays["self_delta"][4, vi]).max() == 0.0


# --------------------------------------------------------------------------- #
# the block describes itself
# --------------------------------------------------------------------------- #


def test_the_block_carries_the_labels_its_arrays_are_joined_on() -> None:
    """Row order is a contract that is asserted, not assumed."""
    out = _probe(PropagatingToy())
    assert out["perturbed_gene_id"].tolist() == PERTURBED["gene_id"].tolist()
    assert out["readout_col"].tolist() == PERTURBED["readout_col"].tolist()
    np.testing.assert_allclose(out["value_grid_cp10k"], C.VALUE_GRID_CP10K, rtol=0, atol=0)
    half = out["diagnostics"]["half_split"]
    assert half["seed"] == C.HALF_SPLIT_SEED
    assert half["self_position_excluded"] is True
    assert half["n_half_a"] + half["n_half_b"] == PropagatingToy.n_readout


def test_a_perturbed_column_outside_the_read_out_set_is_refused() -> None:
    """A perturbed table joined against the wrong read-out table must not run."""
    bad = PERTURBED.copy()
    bad.loc[0, "readout_col"] = PropagatingToy.n_readout
    with pytest.raises(RuntimeError):
        D.probe_anchor(PropagatingToy(), ANCHOR_CP10K, bad, LAYERS, log_every=0)


def test_an_anchor_of_the_wrong_width_is_refused() -> None:
    with pytest.raises(RuntimeError):
        D.probe_anchor(
            PropagatingToy(), ANCHOR_CP10K[:-1], PERTURBED, LAYERS, log_every=0
        )
