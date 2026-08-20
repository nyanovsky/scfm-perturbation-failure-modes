"""The scoring grid, on data where the right answer is known by construction.

The case that matters: a prediction that is the SAME vector for every held-out
perturbation. It matches no perturbation in particular, but it still correlates
with the residual target, because held-out perturbations share the offset
between their own mean and the training mean. The centered cells exist to give
that prediction a zero.
"""

from __future__ import annotations

import numpy as np

from scfm_pfm import metrics
from scfm_pfm.translate import runner


def _reference(n=40, g=60, n_folds=2, seed=0):
    rng = np.random.default_rng(seed)
    effect = rng.normal(size=(n, g)).astype(np.float32)
    # a shared response direction, so the folds have real structure to share
    effect += np.linspace(-1.0, 1.0, g, dtype=np.float32)[None, :]
    fold = np.arange(n) % n_folds
    # each fold is a response family with its own shared direction, which is
    # what makes the held-out offset a real vector rather than noise
    for f in range(n_folds):
        effect[fold == f] += (f + 1) * rng.normal(size=g).astype(np.float32)
    self_cols = rng.integers(0, g, size=n)
    folds = []
    for f in range(n_folds):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        folds.append(
            {
                "train": tr,
                "test": te,
                "weights": np.ones(g),
                "mean": effect[tr].mean(axis=0),
                "rel": {},
            }
        )
    ref = {
        "folds": folds,
        "ceilings": dict.fromkeys(
            (
                "residual_weighted",
                "residual_unweighted",
                "raw_weighted",
                "raw_unweighted",
                "centered_weighted",
                "centered_unweighted",
            ),
            1.0,
        ),
        "effect": effect,
        "self_cols": self_cols,
    }
    return ref, fold


def test_constant_prediction_scores_zero_only_after_centering():
    """The prediction is one fixed vector: the held-out fold's own mean
    response, which matches no perturbation in particular."""
    ref, fold = _reference()
    n, _ = ref["effect"].shape
    info = ref["folds"][0]
    te = info["test"]
    offset = (ref["effect"][te] - info["mean"][None, :]).mean(axis=0)
    pred = np.repeat(offset[None, :], n, axis=0).astype(np.float32)
    summary, per_pert, _, _ = score_grid(pred, ref, fold)
    scored = per_pert.loc[te]
    assert scored.residual_weighted.median() > 0.2
    assert scored.centered_weighted.isna().all()
    assert summary["centered__weighted__n_scored"] == 0
    assert set(per_pert.columns) >= {"residual_weighted", "centered_weighted"}


def test_perfect_prediction_scores_one_in_every_cell():
    ref, fold = _reference()
    effect = ref["effect"]
    pred = np.zeros_like(effect)
    for f, info in enumerate(ref["folds"]):
        te = info["test"]
        pred[te] = effect[te] - info["mean"][None, :]
    summary, _, _, _ = score_grid(pred, ref, fold)
    for cell in ("residual__weighted", "centered__weighted", "raw__weighted"):
        assert summary[f"{cell}__attenuated"] > 0.999


def test_centered_target_matches_a_hand_computed_case():
    ref, fold = _reference(n=6, g=5, n_folds=2, seed=3)
    rng = np.random.default_rng(2)
    pred = rng.normal(size=ref["effect"].shape).astype(np.float32)
    summary, per_pert, _, _ = score_grid(pred, ref, fold)
    info = ref["folds"][0]
    te = info["test"]
    true_r = (ref["effect"][te] - info["mean"][None, :]).astype(np.float64)
    true_c = true_r - true_r.mean(axis=0, keepdims=True)
    pred_c = pred[te].astype(np.float64)
    pred_c = pred_c - pred_c.mean(axis=0, keepdims=True)
    w = np.ones_like(true_c)
    w[np.arange(te.size), ref["self_cols"][te]] = 0.0
    expected = metrics.weighted_pearson_rows(pred_c, true_c, w)
    got = per_pert.loc[te, "centered_weighted"].to_numpy()
    assert np.allclose(expected, got, atol=1e-10, equal_nan=True)


def score_grid(pred, ref, fold):
    return runner.score_run(pred, ref, fold, len(ref["folds"]))
