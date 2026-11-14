"""Tests for run 6's extra scoring cells, on cases whose answer is known first.

Two kinds, both here. The ordinary software tests pin the edge cases a weighted
similarity has --- zero weights, degenerate variance, a negated vector. The
experiment tests are the ones that matter: this is biology code and its failures
are silent, so every claim the module makes about itself is checked against
arithmetic that was worked out before the code ran.

Four of them are the acceptance gate for this module:

- ``test_weighted_cosine_matches_pearson_when_centred`` --- the cosine cell must
  BE the Pearson cell minus the centring, or the two are not comparable.
- ``test_pds_self_column_exclusion_really_excludes`` and
  ``test_pds_nan_in_own_column_is_removed_exactly`` --- the self-column
  correction must remove the knockdown column algebraically, not approximately,
  including when the value there is TranscriptFormer's ``nan``. A test that only
  showed the score "barely moved" would pass on a correction that was wrong.
- ``test_pds_matches_run3_own_discrimination`` --- ``denominator="n_minus_1"``
  must reproduce run 3's published column exactly, because run 6's whole claim
  is that it scores through an unchanged rule.

And one negative result is asserted rather than avoided: a CONSTANT prediction
scores ~0.5 on discrimination, not 0. Everything downstream has to read a
discrimination score against fresh shuffled-label draws in its own cell.

Torch is optional: every core path is numpy, and the two tests that need torch
skip themselves.

    pytest tests/test_metrics_extra.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scfm_hidarm import config as C
from scfm_hidarm import metrics_extra as mx
from scfm_hidarm.sibling import run3

metrics = run3("metrics")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _weighted_centre(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Remove the per-row weighted mean, so cosine and Pearson must coincide."""
    w = np.broadcast_to(np.asarray(w, dtype=np.float64), x.shape)
    m = (w * x).sum(axis=1) / w.sum(axis=1)
    return x - m[:, None]


def _fake_ref(n: int, g: int, n_folds: int, seed: int) -> tuple[dict, np.ndarray]:
    """A ``fold_reference``-shaped dict over synthetic data.

    Shaped exactly as ``runner.fold_reference`` returns --- ``folds`` entries
    with ``train``/``test``/``weights``/``mean``, plus ``effect`` and
    ``self_cols`` --- so ``score_extra`` is exercised on the real contract rather
    than on a convenience object invented for the test.
    """
    rng = np.random.default_rng(seed)
    effect = rng.normal(size=(n, g)).astype(np.float32)
    self_cols = rng.integers(0, g, size=n)
    fold = np.arange(n) % n_folds
    folds = []
    for f in range(n_folds):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        folds.append(
            {
                "train": tr,
                "test": te,
                "weights": rng.uniform(0.2, 1.0, size=g),
                "mean": effect[tr].mean(axis=0),
            }
        )
    ref = {"folds": folds, "effect": effect, "self_cols": self_cols, "ceilings": {}}
    return ref, fold


# --------------------------------------------------------------------------- #
# (a) weighted cosine
# --------------------------------------------------------------------------- #
def test_cosine_of_a_vector_with_itself_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 40))
    w = rng.uniform(0.1, 1.0, size=40)
    assert np.allclose(mx.weighted_cosine_rows(x, x, w), 1.0)


def test_cosine_of_a_vector_with_its_negation_is_minus_one():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 40))
    w = rng.uniform(0.1, 1.0, size=40)
    assert np.allclose(mx.weighted_cosine_rows(x, -x, w), -1.0)


def test_a_zero_weight_really_removes_a_gene():
    """Not "moves little": a zeroed gene may take any value at all."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 30))
    y = rng.normal(size=(5, 30))
    w = rng.uniform(0.1, 1.0, size=30)
    w[7] = 0.0
    before = mx.weighted_cosine_rows(x, y, w)
    x2, y2 = x.copy(), y.copy()
    x2[:, 7] = 1e6
    y2[:, 7] = -3.5e5
    after = mx.weighted_cosine_rows(x2, y2, w)
    assert np.allclose(before, after, atol=0.0, rtol=0.0)


def test_degenerate_row_returns_nan_not_zero():
    """A row with no weighted norm has no answer; 0.0 would be an answer."""
    x = np.zeros((2, 10))
    x[1] = np.arange(10.0)
    y = np.ones((2, 10))
    w = np.ones(10)
    out = mx.weighted_cosine_rows(x, y, w)
    assert np.isnan(out[0])
    assert np.isfinite(out[1])


def test_nonfinite_entries_are_dropped_by_weight():
    """TranscriptFormer's undefined (p, p) pair must not poison a whole row."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 20))
    y = rng.normal(size=(4, 20))
    w = np.ones(20)
    clean = mx.weighted_cosine_rows(np.delete(x, 5, axis=1), np.delete(y, 5, axis=1), np.ones(19))
    x_nan = x.copy()
    x_nan[:, 5] = np.nan
    got = mx.weighted_cosine_rows(x_nan, y, w)
    assert np.allclose(got, clean)


def test_weighted_cosine_matches_pearson_when_centred():
    """The cosine cell IS the Pearson cell without the mean removal."""
    rng = np.random.default_rng(4)
    w = rng.uniform(0.1, 1.0, size=25)
    x = _weighted_centre(rng.normal(size=(8, 25)), w)
    y = _weighted_centre(rng.normal(size=(8, 25)), w)
    assert np.allclose(
        mx.weighted_cosine_rows(x, y, w), metrics.weighted_pearson_rows(x, y, w), atol=1e-12
    )


def test_cosine_differs_from_pearson_on_an_offset():
    """And it must NOT coincide once an offset is present, or it measures nothing new."""
    rng = np.random.default_rng(5)
    w = np.ones(25)
    x = rng.normal(size=(8, 25)) + 3.0
    y = rng.normal(size=(8, 25)) + 3.0
    cos = mx.weighted_cosine_rows(x, y, w)
    pear = metrics.weighted_pearson_rows(x, y, w)
    assert not np.allclose(cos, pear)


# --------------------------------------------------------------------------- #
# (b) L1 perturbation discrimination
# --------------------------------------------------------------------------- #
def test_perfect_prediction_scores_exactly_one():
    rng = np.random.default_rng(6)
    true = rng.normal(size=(20, 15))
    out = mx.pds_l1(true.copy(), true, denominator="n")
    assert np.array_equal(out["per_perturbation"], np.ones(20))
    assert out["mean"] == 1.0


def test_permuted_prediction_scores_at_chance():
    rng = np.random.default_rng(7)
    n = 200
    true = rng.normal(size=(n, 50))
    perm = rng.permutation(n)
    out = mx.pds_l1(true[perm], true, denominator="n")
    # chance is 1 - E[rank]/n = 1 - (n-1)/(2n); 200 rows put the s.e. near 0.02
    assert abs(out["mean"] - (1.0 - (n - 1) / (2.0 * n))) < 0.05


def test_constant_prediction_does_not_score_zero():
    """A constant prediction scores ~0.5, which is why an arm is read against
    fresh shuffled-label draws in its own cell and never against 0."""
    rng = np.random.default_rng(8)
    n = 40
    true = rng.normal(size=(n, 25))
    pred = np.broadcast_to(true.mean(axis=0), (n, 25)).copy()
    out = mx.pds_l1(pred, true, denominator="n")
    # every row of pred is identical, so d[p, q] depends on q alone and the ranks
    # are a permutation of 0..n-1: the mean score is exactly 1 - (n-1)/(2n)
    assert out["mean"] == pytest.approx(1.0 - (n - 1) / (2.0 * n), abs=1e-12)
    assert out["mean"] > 0.4


def test_ties_are_not_counted():
    """An exact tie must not inflate a rank: identical rows all score 1.0."""
    true = np.tile(np.arange(6.0), (5, 1))
    out = mx.pds_l1(true.copy(), true, denominator="n")
    assert np.array_equal(out["per_perturbation"], np.ones(5))


def test_denominators_differ_by_the_documented_amount():
    rng = np.random.default_rng(9)
    n = 30
    true = rng.normal(size=(n, 20))
    pred = true + rng.normal(scale=2.0, size=(n, 20))
    a = mx.pds_l1(pred, true, denominator="n")["per_perturbation"]
    b = mx.pds_l1(pred, true, denominator="n_minus_1")["per_perturbation"]
    rank = (1.0 - a) * n
    assert np.allclose(a - b, rank / (n * (n - 1.0)), atol=1e-12)


def test_bad_denominator_raises():
    with pytest.raises(ValueError):
        mx.pds_l1(np.zeros((4, 3)), np.zeros((4, 3)), denominator="n_minus_2")


def test_fewer_than_three_rows_is_nan_not_one():
    out = mx.pds_l1(np.zeros((2, 3)), np.zeros((2, 3)))
    assert np.isnan(out["per_perturbation"]).all()
    assert np.isnan(out["median"])


# --- self-column exclusion -------------------------------------------------- #
def _exclusion_fixture(seed: int, n: int = 24, g: int = 18):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=(n, g))
    pred = true + rng.normal(scale=1.5, size=(n, g))
    self_cols = rng.permutation(g)[:n] if n <= g else rng.integers(0, g, size=n)
    return pred, true, np.asarray(self_cols)


def test_pds_self_column_exclusion_really_excludes():
    """Plant a huge value in each row's own column; the score must not move.

    The excluded set for the pair (p, q) is {s_p, s_q}, so ``pred[p, s_p]`` and
    ``true[q, s_q]`` only ever appear in entries that are excluded. Changing
    them arbitrarily must therefore change nothing at all --- bit-identically,
    not approximately.
    """
    pred, true, s = _exclusion_fixture(10)
    n = pred.shape[0]
    base = mx.pds_l1(pred, true, self_cols=s)["per_perturbation"]

    planted_pred = pred.copy()
    planted_true = true.copy()
    planted_pred[np.arange(n), s] = -1e6
    planted_true[np.arange(n), s] = 5e5
    planted = mx.pds_l1(planted_pred, planted_true, self_cols=s)["per_perturbation"]
    assert np.array_equal(base, planted)

    # and the exclusion must not be a no-op: without it, the plant moves the score
    without = mx.pds_l1(planted_pred, planted_true, self_cols=None)["per_perturbation"]
    assert not np.array_equal(base, without)


def test_pds_nan_in_own_column_is_removed_exactly():
    """TranscriptFormer's undefined (p, p) pair scores bit-identically to any
    other value in that cell, because the correction removes it algebraically."""
    pred, true, s = _exclusion_fixture(11)
    n = pred.shape[0]
    finite = mx.pds_l1(pred, true, self_cols=s)
    nan_pred = pred.copy()
    nan_pred[np.arange(n), s] = np.nan
    got = mx.pds_l1(nan_pred, true, self_cols=s)
    assert np.array_equal(finite["per_perturbation"], got["per_perturbation"])
    assert got["n_nonfinite_sanitised"] == n
    # every one of them sat in a column the correction removes: none reached a distance
    assert got["n_nonfinite_outside_self_columns"] == 0
    assert "TranscriptFormer" in got["sanitised_note"]


def test_pds_counts_nan_the_correction_does_not_remove():
    """A nan away from the self column IS summed in as 0, and must be counted."""
    pred, true, s = _exclusion_fixture(17)
    off = (s + 1) % pred.shape[1]
    pred = pred.copy()
    pred[np.arange(pred.shape[0]), off] = np.nan
    out = mx.pds_l1(pred, true, self_cols=s)
    assert out["n_nonfinite_outside_self_columns"] == pred.shape[0]


def test_pds_refuses_nan_without_self_cols():
    """Sanitising to 0 with no exclusion would score a fabricated value."""
    pred, true, _ = _exclusion_fixture(12)
    pred = pred.copy()
    pred[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="self_cols"):
        mx.pds_l1(pred, true, self_cols=None)


def test_pds_shared_self_column_is_subtracted_once():
    """When s_p == s_q the column was subtracted twice and must be added back.

    Checked against a masked brute force rather than against itself.
    """
    rng = np.random.default_rng(13)
    n, g = 12, 9
    pred = rng.normal(size=(n, g))
    true = rng.normal(size=(n, g))
    s = rng.integers(0, 3, size=n)  # deliberately many collisions
    got = mx.pds_l1(pred, true, self_cols=s, denominator="n")["per_perturbation"]

    d = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            keep = np.ones(g, dtype=bool)
            keep[s[p]] = False
            keep[s[q]] = False
            d[p, q] = np.abs(pred[p, keep] - true[q, keep]).sum()
    own = np.diag(d)
    rank = (d < own[:, None]).sum(axis=1)
    assert np.allclose(got, 1.0 - rank / n, atol=1e-12)


def test_chunked_paths_agree():
    """chunk_rows is an execution knob and must not touch the arithmetic."""
    pred, true, s = _exclusion_fixture(14, n=17, g=13)
    a = mx.pds_l1(pred, true, self_cols=s, chunk_rows=1)["per_perturbation"]
    b = mx.pds_l1(pred, true, self_cols=s, chunk_rows=128)["per_perturbation"]
    assert np.array_equal(a, b)


def test_numpy_and_torch_paths_agree():
    pytest.importorskip("torch")
    pred, true, s = _exclusion_fixture(15, n=31, g=23)
    ref = mx.pds_l1(pred, true, self_cols=s, chunk_rows=4)["per_perturbation"]
    got = mx.pds_l1(pred, true, self_cols=s, device="cpu")["per_perturbation"]
    assert np.allclose(ref, got, atol=1e-6)


def test_pds_matches_run3_own_discrimination():
    """``n_minus_1`` must reproduce run 3's published column exactly."""
    pytest.importorskip("torch")
    disc = run3("discrimination")
    rng = np.random.default_rng(16)
    n, g = 40, 30
    true = rng.normal(size=(n, g))
    pred = true + rng.normal(scale=1.2, size=(n, g))
    theirs = disc.own_discrimination(true, pred, metric="l1", device="cpu")["per_perturbation"]
    ours = mx.pds_l1(pred, true, self_cols=None, denominator="n_minus_1")["per_perturbation"]
    assert np.allclose(theirs, ours, atol=1e-9)


# --------------------------------------------------------------------------- #
# (c) ceilings
# --------------------------------------------------------------------------- #
def test_cosine_ceiling_of_identical_halves_is_one():
    ref, _ = _fake_ref(n=40, g=24, n_folds=4, seed=20)
    half = np.asarray(ref["effect"], dtype=np.float64)
    out = mx.cosine_ceiling(half, half.copy(), ref["folds"], ref["self_cols"], "residual",
                            "weighted")
    # spearman_brown clips its input at 0.999999, so a perfect half-split lands
    # one part in a million below 1.0 --- run 3's own behaviour, not a bug here
    assert out["ceiling"] == pytest.approx(1.0, abs=1e-5)
    assert out["spearman_brown_applied"] is True


def test_pds_ceiling_of_identical_halves_is_one_and_uncorrected():
    ref, _ = _fake_ref(n=40, g=24, n_folds=4, seed=21)
    half = np.asarray(ref["effect"], dtype=np.float64)
    out = mx.pds_ceiling(half, half.copy(), ref["folds"], ref["self_cols"], "residual",
                         "per_fold")
    assert out["ceiling"] == pytest.approx(1.0, abs=0.0)
    assert out["spearman_brown_applied"] is False


def test_pds_ceiling_is_not_spearman_brown_corrected():
    """A chance-level half-split must stay at chance, not be lifted to 2r/(1+r).

    Two independent halves discriminate at chance; Spearman-Brown would map
    ~0.5 to ~0.667 and manufacture a ceiling out of the correction.
    """
    rng = np.random.default_rng(22)
    ref, _ = _fake_ref(n=120, g=30, n_folds=4, seed=22)
    half0 = rng.normal(size=(120, 30))
    half1 = rng.normal(size=(120, 30))
    out = mx.pds_ceiling(half0, half1, ref["folds"], ref["self_cols"], "residual", "per_fold")
    assert out["ceiling"] < 0.62
    assert out["ceiling"] != pytest.approx(
        float(metrics.spearman_brown(out["raw_half_split_median"])), abs=1e-6
    )


def test_extra_ceilings_covers_the_declared_grid():
    ref, _ = _fake_ref(n=40, g=24, n_folds=4, seed=23)
    half = np.asarray(ref["effect"], dtype=np.float64)
    out = mx.extra_ceilings(half, half + 0.1, ref)
    expected = {f"cosine_{t}_{w}" for t in C.EXTRA_METRIC_TARGETS
                for w in C.EXTRA_METRIC_WEIGHTINGS}
    expected |= {f"pds_{t}_{p}" for t in C.EXTRA_METRIC_TARGETS for p in C.PDS_POOLS}
    assert set(out["ceilings"]) == expected
    assert out["spearman_brown_applied"] == {"cosine": True, "pds": False}
    assert out["spearman_brown_applied"]["cosine"] is C.DISATTENUATE["cosine"]
    assert out["spearman_brown_applied"]["pds"] is C.DISATTENUATE["pds"]


# --------------------------------------------------------------------------- #
# (d) score_extra
# --------------------------------------------------------------------------- #
def test_score_extra_emits_every_declared_cell():
    ref, fold = _fake_ref(n=60, g=28, n_folds=5, seed=30)
    rng = np.random.default_rng(31)
    pred = rng.normal(size=ref["effect"].shape).astype(np.float32)
    half = np.asarray(ref["effect"], dtype=np.float64)
    ceilings = mx.extra_ceilings(half, half + rng.normal(scale=0.3, size=half.shape), ref)
    summary, per_pert = mx.score_extra(pred, ref, fold, 5, ceilings=ceilings)

    for t in C.EXTRA_METRIC_TARGETS:
        for w in C.EXTRA_METRIC_WEIGHTINGS:
            k = f"cosine__{t}__{w}"
            for suffix in ("attenuated", "attenuated_lo", "attenuated_hi",
                           "disattenuated", "ceiling", "n_scored"):
                assert f"{k}__{suffix}" in summary
            assert summary[f"{k}__disattenuated"] == pytest.approx(
                summary[f"{k}__attenuated"] / np.sqrt(summary[f"{k}__ceiling"])
            )
        for pool in C.PDS_POOLS:
            k = f"pds__{t}__{pool}"
            assert f"{k}__ratio_to_ceiling" in summary
            assert f"{k}__disattenuated" not in summary
            assert summary[f"{k}__ratio_to_ceiling"] == pytest.approx(
                summary[f"{k}__attenuated"] / summary[f"{k}__ceiling"]
            )
            assert summary[f"{k}__denominator"] == C.PDS_DENOMINATOR
        # the two pools are different readings, not two names for one number
        assert summary[f"pds__{t}__pooled__pool_size_median"] != (
            summary[f"pds__{t}__per_fold__pool_size_median"]
        )
    assert summary["cosine__spearman_brown_applied"] is True
    assert summary["pds__spearman_brown_applied"] is False

    assert len(per_pert) == ref["effect"].shape[0]
    assert list(per_pert["fold"]) == list(fold)
    assert summary["rows_subset_size"] == ref["effect"].shape[0]
    assert summary["bootstrap"] == "row"
    for t in C.EXTRA_METRIC_TARGETS:
        for w in C.EXTRA_METRIC_WEIGHTINGS:
            assert f"cosine_{t}_{w}" in per_pert.columns
        for pool in C.PDS_POOLS:
            assert f"pds_{t}_{pool}" in per_pert.columns


def test_score_extra_rebuilds_the_same_targets_as_run3():
    """The cosine cell must be computed on the very matrices ``score_run`` uses.

    Checked by asking run 3's own ``score_run`` for its Pearson grid on the same
    inputs and asserting that the two agree once the cosine's inputs are
    weighted-mean-centred. If the target construction ever drifted --- a mean
    added back on the wrong side, a missing float64 promotion --- the two would
    stop matching and a cosine cell and a Pearson cell would be describing
    different quantities under one name.
    """
    runner = run3("translate.runner")
    ref, fold = _fake_ref(n=50, g=26, n_folds=5, seed=40)
    # score_run reads ref["ceilings"], so give it the six keys it looks up
    ref["ceilings"] = {f"{t}_{w}": 0.5 for t in ("residual", "raw", "centered")
                       for w in ("weighted", "unweighted")}
    rng = np.random.default_rng(41)
    pred = rng.normal(size=ref["effect"].shape).astype(np.float32)
    their_summary, their_per_pert, _, their_true_resid = runner.score_run(pred, ref, fold, 5)

    n = ref["effect"].shape[0]
    weights = np.zeros_like(ref["effect"])
    for f in range(5):
        info = ref["folds"][f]
        weights[info["test"]] = info["weights"][None, :]
    weights[np.arange(n), ref["self_cols"]] = 0.0
    ours = mx.weighted_cosine_rows(
        _weighted_centre(pred.astype(np.float64), np.where(weights > 0, weights, 1e-30)),
        _weighted_centre(their_true_resid.astype(np.float64),
                         np.where(weights > 0, weights, 1e-30)),
        weights,
    )
    assert np.allclose(ours, their_per_pert["residual_weighted"].to_numpy(), atol=1e-8)


def test_score_extra_needs_a_ceiling():
    ref, fold = _fake_ref(n=30, g=20, n_folds=3, seed=50)
    pred = np.zeros_like(ref["effect"])
    with pytest.raises(RuntimeError, match="ceiling"):
        mx.score_extra(pred, ref, fold, 3)


def test_score_extra_computes_ceilings_from_halves_when_asked():
    ref, fold = _fake_ref(n=30, g=20, n_folds=3, seed=51)
    half = np.asarray(ref["effect"], dtype=np.float64)
    rng = np.random.default_rng(52)
    pred = rng.normal(size=ref["effect"].shape).astype(np.float32)
    summary, _ = mx.score_extra(pred, ref, fold, 3, half0=half, half1=half.copy())
    assert summary["cosine__residual__weighted__ceiling"] == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# (d2) rows_subset and the grouped bootstrap
# --------------------------------------------------------------------------- #
def test_rows_subset_of_everything_reproduces_the_unrestricted_result():
    """The restriction machinery must be inert when it restricts nothing."""
    ref, fold = _fake_ref(n=45, g=22, n_folds=3, seed=60)
    rng = np.random.default_rng(61)
    pred = rng.normal(size=ref["effect"].shape).astype(np.float32)
    half = np.asarray(ref["effect"], dtype=np.float64)
    half1 = half + rng.normal(scale=0.4, size=half.shape)

    a_sum, a_pp = mx.score_extra(pred, ref, fold, 3, half0=half, half1=half1)
    b_sum, b_pp = mx.score_extra(
        pred, ref, fold, 3, rows_subset=np.arange(45), half0=half, half1=half1
    )
    assert a_sum == b_sum
    assert a_pp.equals(b_pp)


def test_pooled_pds_is_recomputed_not_subset():
    """Restricting the pool changes every rank in it.

    This is the whole reason ``rows_subset`` recomputes the discrimination
    cells instead of filtering them: a subset of the 45-way numbers would be
    reported as if it were an 18-way ranking, and run 4's fine-tuned arms are
    read on exactly such a subset.
    """
    ref, fold = _fake_ref(n=45, g=22, n_folds=3, seed=62)
    rng = np.random.default_rng(63)
    pred = rng.normal(size=ref["effect"].shape).astype(np.float32)
    half = np.asarray(ref["effect"], dtype=np.float64)
    half1 = half + rng.normal(scale=0.4, size=half.shape)
    sub = np.flatnonzero(np.isin(fold, (0, 1)))

    full_sum, full_pp = mx.score_extra(pred, ref, fold, 3, half0=half, half1=half1)
    sub_ceil = mx.extra_ceilings(half, half1, ref, rows_subset=sub)
    sub_sum, sub_pp = mx.score_extra(pred, ref, fold, 3, rows_subset=sub, ceilings=sub_ceil)

    col = "pds_residual_pooled"
    assert not np.allclose(
        sub_pp[col].to_numpy()[sub], full_pp[col].to_numpy()[sub], atol=1e-12
    )
    assert sub_sum["pds__residual__pooled__pool_size_median"] == float(sub.size)
    assert full_sum["pds__residual__pooled__pool_size_median"] == 45.0
    # rows outside the subset were not scored by this call, and say so
    outside = np.setdiff1d(np.arange(45), sub)
    assert np.isnan(sub_pp[col].to_numpy()[outside]).all()
    assert np.isnan(sub_pp["cosine_residual_weighted"].to_numpy()[outside]).all()
    assert sub_sum["rows_subset_size"] == int(sub.size)


def test_the_two_pools_disagree():
    """``pooled`` and ``per_fold`` are different questions, not two spellings.

    ``per_fold`` ranks against a third of the pool, so it is the easier of the
    two; only ``pooled`` is comparable with run 3's own column.
    """
    ref, fold = _fake_ref(n=60, g=25, n_folds=3, seed=64)
    rng = np.random.default_rng(65)
    pred = ref["effect"] + rng.normal(scale=1.5, size=ref["effect"].shape).astype(np.float32)
    half = np.asarray(ref["effect"], dtype=np.float64)
    summary, per_pert = mx.score_extra(pred, ref, fold, 3, half0=half, half1=half.copy())
    a = per_pert["pds_residual_pooled"].to_numpy()
    b = per_pert["pds_residual_per_fold"].to_numpy()
    assert not np.allclose(a, b)
    assert summary["pds__residual__per_fold__attenuated"] >= (
        summary["pds__residual__pooled__attenuated"]
    )


def test_grouped_bootstrap_with_singleton_groups_matches_the_row_bootstrap():
    """Singleton groups reduce the cluster bootstrap to the row bootstrap.

    Not bit-for-bit: the two consume the RNG differently (a multinomial count
    matrix versus an index matrix), so the point estimate is identical and the
    interval agrees to Monte-Carlo error. What must be exact is the median and
    the recorded ``bootstrap`` label.
    """
    ref, fold = _fake_ref(n=90, g=25, n_folds=3, seed=66)
    rng = np.random.default_rng(67)
    pred = rng.normal(size=ref["effect"].shape).astype(np.float32)
    half = np.asarray(ref["effect"], dtype=np.float64)
    ceilings = mx.extra_ceilings(half, half + 0.2, ref)
    row_sum, _ = mx.score_extra(pred, ref, fold, 3, ceilings=ceilings)
    grp_sum, _ = mx.score_extra(
        pred, ref, fold, 3, groups=np.arange(90), ceilings=ceilings
    )
    assert row_sum["bootstrap"] == "row"
    assert grp_sum["bootstrap"] == "grouped"
    k = "cosine__residual__weighted"
    assert grp_sum[f"{k}__attenuated"] == row_sum[f"{k}__attenuated"]
    assert grp_sum[f"{k}__n_scored"] == row_sum[f"{k}__n_scored"]
    assert grp_sum[f"{k}__attenuated_lo"] == pytest.approx(
        row_sum[f"{k}__attenuated_lo"], abs=0.05
    )
    assert grp_sum[f"{k}__attenuated_hi"] == pytest.approx(
        row_sum[f"{k}__attenuated_hi"], abs=0.05
    )


def test_grouped_bootstrap_widens_a_real_grouping():
    """And with real groups it must actually widen: that is why it is used."""
    ref, fold = _fake_ref(n=90, g=25, n_folds=3, seed=68)
    rng = np.random.default_rng(69)
    pred = rng.normal(size=ref["effect"].shape).astype(np.float32)
    half = np.asarray(ref["effect"], dtype=np.float64)
    ceilings = mx.extra_ceilings(half, half + 0.2, ref)
    groups = np.repeat(np.arange(9), 10)  # nine groups of ten, not ninety draws
    row_sum, _ = mx.score_extra(pred, ref, fold, 3, ceilings=ceilings)
    grp_sum, _ = mx.score_extra(pred, ref, fold, 3, groups=groups, ceilings=ceilings)
    k = "cosine__residual__weighted"
    row_w = row_sum[f"{k}__attenuated_hi"] - row_sum[f"{k}__attenuated_lo"]
    grp_w = grp_sum[f"{k}__attenuated_hi"] - grp_sum[f"{k}__attenuated_lo"]
    assert grp_w > row_w


def test_score_extra_refuses_a_ceiling_from_another_pool():
    """A full-pool ceiling under a restricted reading is a wrong denominator."""
    ref, fold = _fake_ref(n=45, g=22, n_folds=3, seed=70)
    rng = np.random.default_rng(71)
    pred = rng.normal(size=ref["effect"].shape).astype(np.float32)
    half = np.asarray(ref["effect"], dtype=np.float64)
    full_ceilings = mx.extra_ceilings(half, half + 0.2, ref)
    sub = np.flatnonzero(np.isin(fold, (0, 1)))
    with pytest.raises(RuntimeError, match="pool-size dependent"):
        mx.score_extra(pred, ref, fold, 3, rows_subset=sub, ceilings=full_ceilings)


def test_rows_subset_is_validated():
    ref, fold = _fake_ref(n=20, g=10, n_folds=2, seed=72)
    half = np.asarray(ref["effect"], dtype=np.float64)
    with pytest.raises(RuntimeError, match="outside"):
        mx.extra_ceilings(half, half.copy(), ref, rows_subset=np.array([0, 25]))


def test_unknown_pool_is_refused():
    ref, _ = _fake_ref(n=20, g=10, n_folds=2, seed=73)
    half = np.asarray(ref["effect"], dtype=np.float64)
    with pytest.raises(ValueError, match="pool"):
        mx.pds_ceiling(half, half.copy(), ref["folds"], ref["self_cols"], "residual", "whole")


def test_restricted_ceiling_differs_from_the_full_one():
    """Run 4's restricted_ceilings pattern: the subset has its own ceiling.

    The half-split is deliberately noisy enough that the discrimination ceiling
    sits well below 1.0; on a measurement that reproduces perfectly both pools
    saturate at 1.0 and the restriction has nothing to move, which is correct
    behaviour but tests nothing.

    Only the POOLED ceiling moves under a fold restriction, and that is the
    point: it is the pool-size-dependent one. The per-fold ceiling is aggregated
    over the folds present, and each of those folds' pools is kept whole.
    """
    ref, fold = _fake_ref(n=60, g=25, n_folds=3, seed=74)
    rng = np.random.default_rng(75)
    half = np.asarray(ref["effect"], dtype=np.float64)
    half1 = half + rng.normal(scale=2.0, size=half.shape)
    sub = np.flatnonzero(np.isin(fold, (0, 1)))
    full = mx.extra_ceilings(half, half1, ref)
    part = mx.extra_ceilings(half, half1, ref, rows_subset=sub)
    assert part["rows_subset_size"] == int(sub.size)
    assert full["ceilings"]["pds_residual_pooled"] < 0.95
    assert part["ceilings"]["pds_residual_pooled"] != full["ceilings"]["pds_residual_pooled"]


def test_pds_ceiling_residual_uses_each_folds_own_training_mean():
    """The pooled ceiling spans folds; each row's residual must not.

    ``_assemble_halves`` writes one whole-scope matrix so the pooled
    discrimination ceiling can rank across folds, and the trap it has to avoid
    is residualizing that matrix against a single pooled mean. That would let
    every held-out perturbation's own outcome into the mean it is measured
    against --- a leak the score itself does not have, which would silently
    raise the denominator every arm is divided by. Checked against a hand-built
    per-fold construction using run 3's own ``residualize``, and against the
    leaked construction, which gives a different number.
    """
    ref, fold = _fake_ref(n=60, g=25, n_folds=3, seed=80)
    rng = np.random.default_rng(81)
    h0 = np.asarray(ref["effect"], dtype=np.float64)
    h1 = h0 + rng.normal(scale=2.0, size=h0.shape)

    a = np.zeros_like(h0)
    b = np.zeros_like(h0)
    for info in ref["folds"]:
        r0, _ = metrics.residualize(h0, info["train"])
        r1, _ = metrics.residualize(h1, info["train"])
        a[info["test"]] = r0[info["test"]]
        b[info["test"]] = r1[info["test"]]
    expected = float(np.nanmedian(
        mx.pds_l1(a, b, self_cols=ref["self_cols"],
                  denominator=C.PDS_DENOMINATOR)["per_perturbation"]
    ))
    got = mx.pds_ceiling(h0, h1, ref["folds"], ref["self_cols"], "residual", "pooled")
    assert got["raw_half_split_median"] == pytest.approx(expected, abs=1e-12)

    leaked0, _ = metrics.residualize(h0, np.arange(60))
    leaked1, _ = metrics.residualize(h1, np.arange(60))
    leaked = float(np.nanmedian(
        mx.pds_l1(leaked0, leaked1, self_cols=ref["self_cols"],
                  denominator=C.PDS_DENOMINATOR)["per_perturbation"]
    ))
    assert leaked != pytest.approx(expected, abs=1e-12)
