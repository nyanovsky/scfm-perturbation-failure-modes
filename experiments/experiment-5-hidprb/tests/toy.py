"""Toy models with a hand-computable hidden state, and a known right answer.

This run's whole finding is a number produced by a loop, so the loop is checked
against arithmetic before a GPU hour is spent on it. A probe that reports
"decodable" for every input would pass any test that only ever feeds it a signal,
which is why there are two toys here and not one:

``PropagatingToy``   an edit at gene *i* reaches every other gene's hidden state,
                     carrying a direction that identifies *i*. The probe MUST
                     recover identity at top-1 = 1.0.
``IsolatedToy``      the same arithmetic with the cross-gene term removed, so an
                     edit at *i* cannot reach *j != i* at all. The probe MUST sit
                     at chance, and the off-target delta MUST be exactly 0.0.

The second is the one that can fail, and it is the reason the pair is the
acceptance gate rather than the first alone.

Both implement the run-2 ``QueryModel`` surface the driver touches plus run 5's
``hidden_batch``, in numpy, so the estimator is testable on a machine with no
torch and no checkpoint.

Layer convention, identical to the real adapters: layer 0 is the post-embedding
state, which depends on a gene's OWN input value and nothing else --- so the
off-target delta there is exactly zero in both toys, and that is control C2.
Layers 1..n mix in the cross-gene term, scaled by ``depth_gain`` so a depth
profile exists to be plotted.
"""

from __future__ import annotations

import numpy as np

N_READOUT = 24
D_MODEL = 16
N_LAYERS = 4


class _ToyBase:
    """The shared arithmetic. Subclasses set ``cross`` and nothing else."""

    n_readout = N_READOUT
    d_model = D_MODEL
    n_layers = N_LAYERS
    batch_size = 8
    linear_scale = None
    self_response_defined = True

    #: does an edit at gene i reach gene j != i? The one bit the two toys differ in.
    cross = True

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        #: per-gene identity block: the toy's analog of a static gene table. It is
        #: constant in the input, so it cancels out of every delta --- which is
        #: what makes the C4 control (is the signature just the gene table?)
        #: answerable on the toy at all.
        self.gene_table = rng.normal(size=(N_READOUT, D_MODEL))
        #: the planted per-gene direction: column i is what an edit at gene i
        #: writes into every other gene's hidden state
        self.M = rng.normal(size=(D_MODEL, N_READOUT))
        #: the own-value encoder, present in both toys and at every layer
        self.V = rng.normal(size=(D_MODEL,))
        #: per-read-out-gene gain on the cross term, so the delta's magnitude
        #: varies across positions while its direction does not
        self.w = rng.random(N_READOUT) + 0.5
        self.W = rng.normal(size=(N_READOUT, N_READOUT)) / N_READOUT
        self.own = 0.7

    # ---- the run-2 surface the driver uses ------------------------------- #

    def to_native(self, cp10k) -> np.ndarray:
        return np.asarray(cp10k, dtype=np.float64)

    def mask_spec(self) -> dict | None:
        return None

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        return self.gene_table.astype(np.float32), "toy gene_table"

    def forward_batch(self, x, perturbed_col=None, mask=None) -> np.ndarray:
        """One scalar per read-out gene, the analog of the real decoder head."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        lx = np.log1p(np.maximum(x, 0.0))
        return (lx @ self.W.T + self.own * lx).astype(np.float32)

    # ---- run 5's addition ------------------------------------------------ #

    def depth_gain(self, layer: int) -> float:
        """How much of the cross-gene term has arrived by layer ``layer``.

        Zero at layer 0 by construction, then rising: the toy's stand-in for
        information accumulating through a stack. A probe whose depth profile is
        flat on this toy is not reading the layer argument.
        """
        return 0.0 if layer <= 0 else float(layer) / self.n_layers

    def hidden_batch(self, x_native, layers, perturbed_col=None):
        """``{layer: (B, n_readout, d)}``, the frozen run-5 interface.

        ``h[b, j] = gene_table[j] + V * log1p(x[b, j]) + gain(l) * w[j] * (M @ log1p(x[b]))``

        The middle term is the gene's own value and carries no information about
        any other gene. The last term is the only route from gene *i* to gene
        *j*, and ``IsolatedToy`` deletes it --- so the difference between the two
        toys is exactly the presence of propagation, and nothing else.
        """
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float64))
        lx = np.log1p(np.maximum(x, 0.0))
        b = lx.shape[0]
        own = lx[:, :, None] * self.V[None, None, :]           # (B, n_readout, d)
        base = self.gene_table[None, :, :] + own
        out = {}
        for layer in layers:
            h = base.copy()
            if self.cross:
                mixed = lx @ self.M.T                           # (B, d)
                h = h + self.depth_gain(layer) * self.w[None, :, None] * mixed[:, None, :]
            out[int(layer)] = h.astype(np.float32)
        return out


class PropagatingToy(_ToyBase):
    """An edit at gene i writes ``M[:, i]`` into every other gene's state."""

    cross = True


class IsolatedToy(_ToyBase):
    """The cross-gene route deleted: an edit at i cannot reach j != i."""

    cross = False


class MuteToy(_ToyBase):
    """A model that cannot supply hidden states, and says so.

    The ``exact_jvp`` contract this mirrors requires a measured absence to carry
    a verbatim reason rather than a silent ``None``; this toy is what checks the
    driver records that reason instead of writing an empty block.
    """

    def hidden_batch(self, x_native, layers, perturbed_col=None):
        self._hidden_error = (
            "MuteToy exposes no per-gene hidden state; the toy exists to check "
            "that a measured absence is recorded rather than written as zeros"
        )
        return None
