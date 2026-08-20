"""The scoring rule, checked against constructions with a known answer.

Most bugs in a scoring path do not raise; they return a plausible number. These
tests pin the pieces that decide the verdict: the weighting, the leave-one-out
residual, the Spearman-Brown correction, and the reliability estimate.
"""

from __future__ import annotations

import numpy as np

from scfm_pfm import metrics


def test_weighted_pearson_matches_numpy_when_weights_are_uniform():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 40))
    y = 0.4 * x + rng.normal(size=(7, 40))
    got = metrics.weighted_pearson_rows(x, y, np.ones(40))
    want = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in range(7)])
    assert np.allclose(got, want, atol=1e-10)


def test_weighted_pearson_zero_weight_removes_a_gene():
    x = np.array([[0.0, 0.0, 0.0, 1.0]])
    y = np.array([[1.0, 2.0, 3.0, -50.0]])
    w = np.array([1.0, 1.0, 1.0, 0.0])
    # with the outlier removed, x is constant and the correlation is undefined
    assert np.isnan(metrics.weighted_pearson_rows(x, y, w)[0])
    # with it included, the pair is perfectly (negatively) determined by 2 points
    assert np.isfinite(metrics.weighted_pearson_rows(x, y, np.ones(4))[0])


def test_weighted_pearson_weights_shift_the_answer():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = np.array([[1.0, 2.0, 3.0, -4.0]])
    hi = metrics.weighted_pearson_rows(x, y, np.array([1.0, 1.0, 1.0, 0.01]))[0]
    lo = metrics.weighted_pearson_rows(x, y, np.ones(4))[0]
    assert hi > lo


def test_residualize_leaves_no_training_mean():
    rng = np.random.default_rng(1)
    shared = rng.normal(size=50)
    effect = shared[None, :] + rng.normal(size=(20, 50)) * 0.1
    train = np.arange(15)
    resid, means = metrics.residualize(effect, train)
    # a held-out row uses the plain training mean
    assert np.allclose(means[17], effect[train].mean(axis=0))
    # a training row uses the leave-one-out mean, which excludes itself
    loo = effect[[i for i in train if i != 3]].mean(axis=0)
    assert np.allclose(means[3], loo)
    assert np.abs(resid[train].mean()) < np.abs(effect[train].mean())


def test_spearman_brown_raises_a_half_split_value():
    assert abs(metrics.spearman_brown(0.5) - 2 / 3) < 1e-12
    assert abs(metrics.spearman_brown(0.0)) < 1e-12
    assert abs(metrics.spearman_brown(1.0) - 1.0) < 1e-6


def test_per_gene_reliability_separates_signal_from_noise():
    rng = np.random.default_rng(2)
    n_pert, n_gene = 400, 60
    signal = rng.normal(size=(n_pert, n_gene))
    # first half of the genes are measured cleanly, second half is noise only
    noise_sd = np.concatenate([np.full(30, 0.2), np.full(30, 8.0)])
    h0 = signal + rng.normal(size=(n_pert, n_gene)) * noise_sd
    h1 = signal + rng.normal(size=(n_pert, n_gene)) * noise_sd
    out = metrics.per_gene_reliability(h0, h1, np.arange(n_pert))
    w = out["weights"]
    assert w[:30].mean() > 0.9
    assert w[30:].mean() < 0.2
    assert (w >= 0).all() and (w <= 1).all()


def test_reliability_shrinks_when_there_is_no_between_gene_signal():
    rng = np.random.default_rng(3)
    h0 = rng.normal(size=(100, 40))
    h1 = rng.normal(size=(100, 40))
    out = metrics.per_gene_reliability(h0, h1, np.arange(100))
    assert out["shrinkage_lambda"] < 0.5
    assert out["weights"].max() < 0.35


def test_residual_ceiling_is_below_the_raw_ceiling_when_a_shared_response_exists():
    rng = np.random.default_rng(4)
    n_pert, n_gene = 60, 200
    shared = rng.normal(size=n_gene) * 3.0
    specific = rng.normal(size=(n_pert, n_gene))
    h0 = shared[None, :] + specific + rng.normal(size=(n_pert, n_gene))
    h1 = shared[None, :] + specific + rng.normal(size=(n_pert, n_gene))
    rows = np.arange(n_pert)
    w = np.ones(n_gene)
    raw = np.nanmedian(metrics.weighted_pearson_rows(h0, h1, w))
    resid = np.nanmedian(metrics.residual_ceiling(h0, h1, rows, w))
    assert resid < raw


def test_bootstrap_difference_brackets_a_known_shift():
    rng = np.random.default_rng(5)
    a = rng.normal(0.10, 0.2, size=400)
    b = rng.normal(0.00, 0.2, size=400)
    out = metrics.bootstrap_median_difference(a, b, n_resamples=500)
    # the interval brackets the observed difference and excludes no difference
    assert out["lo"] < out["difference"] < out["hi"]
    assert out["lo"] > 0.0
    assert out["p_two_sided"] < 0.05
    assert abs(out["difference"] - 0.10) < 0.05


def test_min_resolvable_difference_shrinks_with_n_and_with_pairing():
    small = metrics.min_resolvable_difference(0.2, 100, 0.5)
    large = metrics.min_resolvable_difference(0.2, 400, 0.5)
    paired = metrics.min_resolvable_difference(0.2, 400, 0.9)
    assert large < small
    assert paired < large
