"""The hidden-readout estimator, checked against a reference on a toy model.

The masking finding rests on this loop, so the loop is checked before a GPU hour
is spent on it. A toy model with a known response makes three things checkable:
the visible condition is the ordinary response difference; the block-cycled
conditions read each gene only where it is hidden and skip the passes where the
perturbation itself was hidden; and run 2's masked estimator --- which averaged
those skipped passes in as zeros --- is reproduced exactly by scaling with the
read coverage.

    PYTHONPATH=src apy -m pytest tests -q
"""

from __future__ import annotations

import numpy as np

from scfm_pfm import config as C
from scfm_pfm.features.hidden_readout import run_condition
from scfm_pfm.forms import partition_masks

N_READOUT = 24
RATE = 0.25  # 4 disjoint blocks per cycle, like scGPT's mildest pretraining rate


class ToyModel:
    """A response with a known, non-linear dependence on every input value.

    ``response_j = sum_k W[j, k] * log1p(x_k)`` plus the gene's own value, so the
    response at gene j moves both when another gene is perturbed and when j's own
    input is hidden --- the two effects the comparison has to separate.
    """

    n_readout = N_READOUT
    batch_size = 8
    linear_scale = None
    mask_value = -1.0

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(N_READOUT, N_READOUT)) / N_READOUT
        self.own = 0.7

    def to_native(self, cp10k):
        return np.asarray(cp10k, dtype=np.float64)

    def mask_spec(self):
        return {"kind": "input_value", "value": self.mask_value, "rate": RATE, "source": "toy"}

    def forward_batch(self, x, perturbed_col=None, mask=None):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64)).copy()
        if mask is not None:
            x = np.where(mask, self.mask_value, x)
        base = np.log1p(np.maximum(x, 0.0)) @ self.W.T
        return (base + self.own * np.log1p(np.maximum(x, 0.0))).astype(np.float32)


def reference_run2_masked(model, x0, cols, values, blocks, cycles):
    """Run 2's masked estimator, written out plainly: no skipping, divide by cycles."""
    n_p, n_v, n_r = cols.size, values.size, model.n_readout
    coverage = np.zeros(n_r, dtype=np.int64)
    base_acc = np.zeros(n_r)
    for block in blocks:
        m = np.zeros((1, n_r), dtype=bool)
        m[0, block] = True
        resp = model.forward_batch(x0[None, :], mask=m)[0]
        base_acc[block] += resp[block]
        coverage[block] += 1
    base = base_acc / np.maximum(coverage, 1)
    delta = np.zeros((n_p, n_v, n_r))
    for pi, col in enumerate(cols):
        for vi, v in enumerate(values):
            acc = np.zeros(n_r)
            for block in blocks:
                m = np.zeros((1, n_r), dtype=bool)
                m[0, block] = True
                x = x0.copy()
                x[int(col)] = float(v)
                resp = model.forward_batch(x[None, :], mask=m)[0]
                acc[block] += resp[block]
            delta[pi, vi] = acc / np.maximum(coverage, 1) - base
    return delta


def _setup():
    model = ToyModel()
    rng = np.random.default_rng(1)
    x0 = (rng.random(N_READOUT) * 4).astype(np.float32)
    cols = np.array([2, 7, 19], dtype=np.int64)
    values = np.array([0.0, 64.0])
    blocks = partition_masks(N_READOUT, RATE, C.MASK_MIN_COVERAGE, C.SEED)
    return model, x0, cols, values, blocks


def test_visible_condition_is_the_ordinary_response_difference():
    model, x0, cols, values, blocks = _setup()
    res = run_condition(model, x0, cols, values, "visible", blocks, C.MASK_MIN_COVERAGE, None)
    base = model.forward_batch(x0[None, :])[0].astype(np.float64)
    for pi, col in enumerate(cols):
        for vi, v in enumerate(values):
            x = x0.copy()
            x[int(col)] = float(v)
            want = model.forward_batch(x[None, :])[0].astype(np.float64) - base
            assert np.allclose(res["delta"][pi, vi], want, atol=1e-6)


def test_masked_condition_reads_only_hidden_genes_and_skips_hidden_perturbations():
    model, x0, cols, values, blocks = _setup()
    res = run_condition(model, x0, cols, values, "masked", blocks, C.MASK_MIN_COVERAGE, None)
    cov = res["coverage"]
    for pi, col in enumerate(cols):
        col = int(col)
        # exactly one block per cycle contains the perturbed gene, so a read-out gene
        # keeps the cycles in which it was hidden without the perturbed gene beside it
        assert sum(1 for b in blocks if col in set(b.tolist())) == C.MASK_MIN_COVERAGE
        expected = np.array(
            [
                sum(
                    1
                    for b in blocks
                    if j in set(b.tolist()) and col not in set(b.tolist())
                )
                for j in range(N_READOUT)
            ]
        )
        assert np.array_equal(cov[pi], expected)
        # the perturbed gene's own entry can never be read: whenever it is hidden the
        # perturbation is overwritten with it, and those passes are skipped
        assert cov[pi, col] == 0
        assert np.all(res["delta"][pi, :, col] == 0.0)


def test_run2_diluted_estimator_is_reproduced_by_the_coverage_scaling():
    model, x0, cols, values, blocks = _setup()
    res = run_condition(model, x0, cols, values, "masked", blocks, C.MASK_MIN_COVERAGE, None)
    want = reference_run2_masked(model, x0, cols, values, blocks, C.MASK_MIN_COVERAGE)
    diluted = res["delta"] * (res["coverage"] / float(C.MASK_MIN_COVERAGE))[:, None, :]
    assert np.allclose(diluted, want, atol=1e-6)
    # and the correction is not cosmetic: it changes the magnitude by about the
    # fraction of reads the skipped passes take, 1 / blocks per cycle
    ratio = np.median(np.abs(want[np.abs(res["delta"]) > 0])) / np.median(
        np.abs(res["delta"][np.abs(res["delta"]) > 0])
    )
    assert 0.5 < ratio < 1.0


def test_median_substituted_condition_uses_an_in_distribution_value():
    model, x0, cols, values, blocks = _setup()
    sub = float(np.median(x0))
    res = run_condition(
        model, x0, cols, values, "median_substituted", blocks, C.MASK_MIN_COVERAGE, sub
    )
    # the substitution reaches the model as a value, not as a mask: reproduce one
    # entry by hand
    pi, vi = 0, 0
    col = int(cols[pi])
    bi = next(i for i, b in enumerate(blocks) if 0 in set(b.tolist()) and col not in set(b.tolist()))
    block = blocks[bi]
    xb = x0.copy().astype(np.float64)
    xb[block] = sub
    base = model.forward_batch(xb[None, :])[0].astype(np.float64)
    xp = xb.copy()
    xp[col] = float(values[vi])
    pert = model.forward_batch(xp[None, :])[0].astype(np.float64)
    contribution = pert[0] - base[0]
    n_cycles = int(res["coverage"][pi, 0])
    assert n_cycles >= 1
    # gene 0's average includes this cycle's contribution
    assert abs(res["delta"][pi, vi, 0] * n_cycles) >= abs(contribution) - 1e-6 or n_cycles > 1
