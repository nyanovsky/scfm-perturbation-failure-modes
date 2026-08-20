"""The controls have to be controls: what each one is allowed to represent."""

from __future__ import annotations

import numpy as np
import torch

from scfm_pfm import config as C
from scfm_pfm.ingest import recompute_effects as rec
from scfm_pfm.translate import runner, trainers


def _fit(model, x, y, steps=400):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return float(((model(x) - y) ** 2).mean())


def test_additive_translator_cannot_learn_a_pair_interaction():
    """The node-only control must not be able to score on the pair.

    The target here is a pure interaction (the product of the two genes'
    properties), so an additive model is at its ceiling when it predicts a
    constant, while the unrestricted MLP over the same concatenated features
    fits it. That difference is the whole reason the control is restricted.
    """
    torch.manual_seed(0)
    x = torch.randn(4096, 4)
    y = (x[:, 0] * x[:, 2] + x[:, 1] * x[:, 3]).detach()
    var = float(y.var())
    additive = trainers.AdditiveMLP(4, 2, C.MLP.hidden)
    free = trainers.MLP(4, C.MLP.hidden)
    mse_add = _fit(additive, x, y)
    mse_free = _fit(free, x, y)
    assert mse_add > 0.7 * var, "an additive model should not fit a pure interaction"
    assert mse_free < 0.3 * var, "the unrestricted MLP should fit it"


def test_dest_reference_transfer_never_scores_a_perturbation_it_calibrated_on():
    rng = np.random.default_rng(0)
    eff = rng.normal(size=(40, 6))
    h0, h1 = eff + rng.normal(scale=0.1, size=eff.shape), eff + rng.normal(scale=0.1, size=eff.shape)
    q = runner._transfer_quantities(
        "dest_reference", eff, h0, h1, eff.mean(axis=0), np.ones(6), C.SEED
    )
    assert set(q["eval_rows"]).isdisjoint(set(q["mean_rows"]))
    assert q["eval_rows"].size + q["mean_rows"].size == 40
    # and the strict mode uses no destination row at all to build its mean
    q2 = runner._transfer_quantities(
        "source_only", eff, h0, h1, np.full(6, 3.0), np.ones(6), C.SEED
    )
    assert np.allclose(q2["means"], 3.0)


def test_recomputed_effect_is_a_within_batch_log_ratio():
    """One perturbation, two batches, known means: the assembled effect is the
    cell-count-weighted average of the two batches' log ratios."""
    n_g = 3
    stream = {
        "n_targets": 1,
        "n_genes": n_g,
        "sums": {
            (0, 7, 0): np.full(n_g, 40.0),   # 4 cells at 10
            (0, 7, 1): np.full(n_g, 40.0),
            (0, 9, 0): np.full(n_g, 120.0),  # 4 cells at 30
            (0, 9, 1): np.full(n_g, 120.0),
        },
        "counts": {(0, 7, 0): 4, (0, 7, 1): 4, (0, 9, 0): 4, (0, 9, 1): 4},
    }
    ctrl_log = {7: np.log2(np.full(n_g, 5.0 + rec.EPS_CP10K)), 9: np.log2(np.full(n_g, 5.0 + rec.EPS_CP10K))}
    out = rec.assemble(stream, ctrl_log)
    expected = 0.5 * (
        np.log2(10 + rec.EPS_CP10K) - np.log2(6.0) + np.log2(30 + rec.EPS_CP10K) - np.log2(6.0)
    )
    assert np.allclose(out["effect"][0], expected, atol=1e-5)
    assert np.allclose(out["half0"][0], expected, atol=1e-5)
    assert out["effect_cells"][0] == 16


def test_a_batch_the_control_half_does_not_cover_is_dropped():
    stream = {
        "n_targets": 1,
        "n_genes": 2,
        "sums": {(0, 1, 0): np.full(2, 50.0), (0, 2, 0): np.full(2, 50.0)},
        "counts": {(0, 1, 0): 5, (0, 2, 0): 5},
    }
    out = rec.assemble(stream, {1: np.log2(np.full(2, 11.0))})
    assert out["effect_cells"][0] == 5
    assert np.allclose(out["effect"][0], np.log2(11.0) - np.log2(11.0))
