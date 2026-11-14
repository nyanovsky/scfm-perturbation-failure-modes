"""Toy models whose hidden state is a known function of dose, exactly.

Run 6's feature is a *derivative* of a hidden state with respect to one gene's
input value, estimated by a central difference over a 16-row batch. Nothing about
that is checkable by looking at it: a ``gdt`` slot built from the wrong pair of
rows, or divided by the wrong span, is a well-shaped float32 array of the wrong
quantity, and every score downstream would still be produced. So the estimator is
checked against arithmetic before a GPU hour is spent on it.

Why run 6 needs its own toys rather than run 5's
------------------------------------------------
Run 5's toys expose a **6-row** ``hidden_batch`` --- one baseline plus one row per
dose --- because run 5 computes no derivative. Run 6 supplies **16** rows,
``(v - h, v, v + h)`` at each of five doses, and its whole new content is what is
done with the two flanking rows. A toy that cannot tell a central difference from
a substitution response cannot check run 6's estimator, so run 5's file is left
alone and these are new.

The four toys, and which claim each one can falsify
--------------------------------------------------
``LinearDoseToy``     ``h`` is affine in the perturbed value, so ``vvp`` at dose
                      *v* is exactly ``m_g (v - anchor) V`` and ``gdt`` is exactly
                      ``m_g V`` at *every* dose --- including the clamped one,
                      because an affine function's derivative does not depend on
                      the span it is measured over. It checks the row indexing.
``QuadraticDoseToy``  ``h`` depends on the *square* of the perturbed value, so
                      ``gdt`` is ``2 m_g v V`` and the span the difference is
                      divided by now matters. This is the toy that can falsify
                      ``gdt_denominator``: get the clamp wrong at dose 0 and the
                      answer is out by a factor of two.
``IsolatedToy``       no off-target dependence at all: an edit at gene *i* cannot
                      reach gene *j != i* at any depth. Every off-target delta is
                      then exactly ``0.0``, which is control **C2** and the one
                      check that fails if the hooks are mis-attributed.
``MuteToy``           supplies no hidden state and says why, so a measured
                      absence is checked to be *recorded* rather than written out
                      as a block of zeros.

``FlagToy`` is a fifth, and a different kind of object: run 4's fine-tuned model
takes a perturbation-flag tensor as a second input, so the toy that checks run
6's flag-constant estimator has to be a stand-in for the *torch module*, not for
the numpy adapter. It is built by ``make_flag_toy`` below, which imports torch
lazily so that this module stays importable --- and the four numpy toys stay
usable --- on a machine with no torch.

Exact arithmetic, on purpose
----------------------------
Every coefficient here is a **dyadic rational** (a small integer over 8) and the
toys return **float64**, not the float32 the real adapters return. Both are
deliberate. With dyadic coefficients and a dyadic step, ``((v+h)^2 - (v-h)^2) /
(2h)`` is exactly ``2v`` in binary floating point --- the subtraction falls under
Sterbenz's lemma and the division is by a power of two --- so the derivative tests
assert ``max abs diff == 0.0`` rather than a tolerance. A tolerance would pass a
denominator that was wrong by 1e-9 and, more to the point, would leave the reader
guessing which differences are arithmetic and which are noise. Callers must
therefore pass a dyadic ``step`` (``STEP`` below); the production steps (0.02,
1.0) are not dyadic and the tests say so where they use one.
"""

from __future__ import annotations

import numpy as np

#: read-out genes in a toy. Small, and not a multiple of the batch size, so a
#: chunk boundary falls inside the batch rather than tidily at its end.
N_READOUT = 12
#: hidden width. Deliberately not a power of two either, so a reshape that
#: assumed one would be visible.
D_MODEL = 6
N_LAYERS = 4
#: rows per forward chunk. 16 input rows at 6 per chunk is 6 + 6 + 4, so the
#: baseline row and the three rows of a dose land in *different* chunks --- which
#: is where a per-chunk tensor that ought to be per-query would go wrong.
BATCH_SIZE = 6

#: the finite-difference step the toys are exercised at: 2^-5, so ``v +- h`` and
#: its square are exactly representable. The production steps in
#: ``config.GDT_STEP`` are 0.02 and 1.0; 0.02 is not dyadic, which is why the
#: exactness claims here are made at this step and not at that one.
STEP = 0.03125


def dyadic(rng: np.random.Generator, size: int) -> np.ndarray:
    """Non-zero multiples of 1/8 in ``[-1, -1/8] u [1/8, 1]``.

    Non-zero matters: a coefficient of zero would make that gene's ``vvp`` slot
    zero for a reason that has nothing to do with the estimator, and the no-op
    test --- which asserts an *exact* zero --- would pass on a gene that could
    never have moved.
    """
    magnitude = rng.integers(1, 9, size=size).astype(np.float64) / 8.0
    sign = rng.choice(np.array([-1.0, 1.0]), size=size)
    return magnitude * sign


class _ToyBase:
    """The run-2 ``QueryModel`` surface run 6's driver actually touches.

    Only the members ``extract/run.py`` reads are present. Nothing here is a
    stub that returns a plausible value: a toy that quietly answered a question
    the real adapter answers differently would make the driver test pass for the
    wrong reason.
    """

    n_readout = N_READOUT
    d_model = D_MODEL
    n_layers = N_LAYERS
    batch_size = BATCH_SIZE
    #: scGPT and scPRINT read a gene's own value at its own position; the toys
    #: model that architecture, so C1 is a real positive control on them
    hidden_self_position_defined = True
    self_response_defined = True
    substitutions: list[str] = []
    build_report = None

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        #: per-gene constant. Cancels out of every delta, which is exactly why it
        #: is here: a slot that did *not* cancel it would be a hidden state
        #: rather than a response, and the tests would see the offset.
        self.a = dyadic(rng, N_READOUT)
        #: per-gene sensitivity to the driving value
        self.m = dyadic(rng, N_READOUT)
        #: the single hidden direction the dose is written along
        self.V = dyadic(rng, D_MODEL)

    # ---- the run-2 surface ------------------------------------------------ #
    def to_native(self, cp10k) -> np.ndarray:
        """Identity, so a native value *is* a CP10K value.

        The real adapters bin (scGPT), clip (TranscriptFormer) or log (run 4);
        keeping the toy's map the identity means a test's expected value is the
        dose it names, and a failure is a failure of the estimator rather than of
        the test's own arithmetic.
        """
        return np.asarray(cp10k, dtype=np.float64)

    def _batches(self, x: np.ndarray):
        """``(start, chunk)`` per chunk, run 2's ``QueryModel`` contract."""
        x = np.atleast_2d(np.asarray(x))
        for start in range(0, x.shape[0], self.batch_size):
            yield start, x[start : start + self.batch_size]

    def layer_gain(self, layer: int) -> float:
        """``2^-layer``: dyadic, and different at every depth.

        A per-layer gain is what makes a layer mix-up visible. A per-layer
        *offset* would not: constants cancel out of every delta, so a driver that
        read layer 7's states as layer 4's would produce identical slots.
        """
        return float(2.0 ** -int(layer))

    def hidden_batch(self, x_native, layers, perturbed_col=None):  # pragma: no cover
        raise NotImplementedError


class _DoseToyBase(_ToyBase):
    """``h[b, g] = gain(l) * (a_g + m_g * f(x[b, p])) * V``.

    Every gene responds to the perturbed gene's value through ``f``, so the
    off-target response is non-zero at every depth --- the opposite of
    ``IsolatedToy``, and the case where the derivative slots have something to
    recover. Subclasses supply ``f`` and nothing else.
    """

    def _f(self, x_perturbed: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def expected(self, x_perturbed: float, layer: int) -> np.ndarray:
        """The toy's own ``(n_readout, d)`` state, for a test to compare against.

        Written once, here, rather than re-derived in each test: two independent
        transcriptions of the same formula agreeing proves nothing about the
        estimator, and disagreeing wastes a debugging session on the copy.
        """
        drive = float(np.asarray(self._f(np.asarray([x_perturbed], dtype=np.float64)))[0])
        return self.layer_gain(layer) * np.outer(self.a + self.m * drive, self.V)

    def hidden_batch(self, x_native, layers, perturbed_col=None):
        if perturbed_col is None:
            raise RuntimeError(
                "a dose toy is defined by which column drives it; with no perturbed_col "
                "there is no dose axis and nothing for gdt to differentiate"
            )
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float64))
        if x.shape[1] != self.n_readout:
            raise RuntimeError(
                f"the toy was given {x.shape[1]} input columns, not {self.n_readout}"
            )
        drive = self._f(x[:, int(perturbed_col)])
        # (B, n_readout) coefficient, then one outer product with V per row
        coeff = self.a[None, :] + self.m[None, :] * drive[:, None]
        base = coeff[:, :, None] * self.V[None, None, :]
        return {int(l): self.layer_gain(int(l)) * base for l in layers}


class LinearDoseToy(_DoseToyBase):
    """``f(x) = x``. ``gdt`` is ``m_g V`` at every dose, clamp or no clamp."""

    def _f(self, x_perturbed: np.ndarray) -> np.ndarray:
        return np.asarray(x_perturbed, dtype=np.float64)


class QuadraticDoseToy(_DoseToyBase):
    """``f(x) = x^2``. ``gdt`` is ``2 m_g v V``, so the span matters.

    The only toy that can falsify ``gdt_denominator``: at the bottom of the grid
    the low row is clamped at ``NATIVE_FLOOR`` and the difference is one-sided
    over a span of ``h`` rather than ``2h``. Divide by the wrong one and the
    recovered derivative is out by a factor of two --- on a quadratic. On the
    affine toy it is out by nothing at all, which is why one toy is not enough.
    """

    def _f(self, x_perturbed: np.ndarray) -> np.ndarray:
        x = np.asarray(x_perturbed, dtype=np.float64)
        return x * x


class IsolatedToy(_ToyBase):
    """``h[b, g]`` depends on ``x[b, g]`` and on nothing else.

    An edit at gene *i* cannot reach gene *j != i* at any depth, so every
    off-target delta is exactly ``0.0`` --- at layer 0, which is control C2, and
    at every layer above it, which no real model satisfies and which is precisely
    why this toy is the one that can fail. The perturbed gene's *own* column does
    move, so C1 stays a positive control and a driver that zeroed everything
    would not pass.
    """

    def hidden_batch(self, x_native, layers, perturbed_col=None):
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float64))
        coeff = self.a[None, :] + self.m[None, :] * x
        base = coeff[:, :, None] * self.V[None, None, :]
        return {int(l): self.layer_gain(int(l)) * base for l in layers}


class MuteToy(_ToyBase):
    """Supplies no hidden state, and says why.

    Mirrors the ``exact_jvp`` / ``_jvp_error`` contract runs 2 and 5 use: an
    absence carries a verbatim reason and is recorded as an absence. The failure
    this guards against is not a crash --- it is a block of zeros installed as if
    it were a measurement.
    """

    def hidden_batch(self, x_native, layers, perturbed_col=None):
        self._hidden_error = (
            "MuteToy exposes no per-gene hidden state; the toy exists to check that a "
            "measured absence is recorded rather than written out as a block of zeros"
        )
        return None


# --------------------------------------------------------------------------- #
# the flag toy: a stand-in for run 4's torch module, not for run 6's adapter
# --------------------------------------------------------------------------- #
def make_flag_toy(
    respond_to: str = "value",
    n_readout: int = N_READOUT,
    d_model: int = 512,
    n_layers: int = 12,
    seed: int = 0,
):
    """A ``TransformerGenerator``-shaped module whose response is a known function.

    Run 6's fine-tuned estimator holds the perturbation flag ON in *both* passes
    and moves only the value, and the claim that makes is about a tensor built
    inside ``models/hidden_ft.py``. Checking it needs the real
    ``ScGPTPertHiddenQuery`` --- its ``_flags``, its ``_hidden_pass``, run 5's
    hooks --- driven by something whose answer is known. So the toy stands in for
    the *checkpoint*, not for the adapter, and the code under test is the real
    code rather than a transcription of it.

    ``respond_to`` selects which input the hidden state depends on:

    ``"flag"``   the flag tensor alone. Because run 6 sets that tensor
                 identically in both passes, every one of the 16 rows must come
                 back bit-identical and every ``Delta h`` must be exactly
                 ``0.0``. An implementation whose baseline pass had the flag off
                 would give a non-zero answer here, which is the failure
                 ``README.md`` s6 says would break C2 and conflate flag onset
                 with dose.
    ``"value"``  the value tensor alone, summed across genes so that *every*
                 read-out gene responds to the perturbed column --- the same
                 all-to-all shape as the dose toys, so the response is
                 recoverable in full rather than only on the diagonal.

    ``d_model`` and ``n_layers`` default to scGPT's 512 and 12 because run 5's
    ``_take`` checks a captured state against ``C.D_MODEL`` / ``C.N_LAYERS`` for
    the model key the adapter reports, and the fine-tuned adapter reports
    ``"scgpt"``. A narrower toy would be refused by the width check, correctly.

    Torch is imported here rather than at module scope so the four numpy toys
    above stay importable where torch is not installed.
    """
    import torch

    if respond_to not in ("flag", "value"):
        raise ValueError(f"respond_to must be 'flag' or 'value', not {respond_to!r}")

    rng = np.random.default_rng(seed)
    a = dyadic(rng, n_readout)
    m = dyadic(rng, n_readout)
    V = dyadic(rng, d_model)

    class _Block(torch.nn.Module):
        """One transformer block, standing in for a stack position.

        Multiplies by a dyadic gain, so the state at block ``l`` differs from the
        state at block ``l'`` by a known factor and a hook attributed to the
        wrong block is visible in the number rather than only in the module path.
        """

        def __init__(self, gain: float):
            super().__init__()
            self.gain = float(gain)

        def forward(self, x):  # noqa: ANN001
            return x * self.gain

    class _Encoder(torch.nn.Module):
        """``transformer_encoder``: a ``.layers`` ModuleList, hooked by run 5.

        The path and the block count are what ``_resolve_stack`` matches on
        (``_stack_candidates = ("transformer_encoder.layers",)``) and what
        ``_final_capture`` hooks for the last layer. Getting either wrong here
        would make the toy exercise the fallback resolution rather than the path
        the real checkpoint pins.
        """

        def __init__(self, gains):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Block(g) for g in gains])

        def forward(self, x, src_key_padding_mask=None):  # noqa: ANN001
            for block in self.layers:
                x = block(x)
            return x

    class FlagToy(torch.nn.Module):
        """``forward(src, values, flags, src_key_padding_mask=...)``, run 4's call.

        The signature is positional and matches run 4's ``_forward`` exactly, so
        the toy is driven through byte-identical adapter code to the real
        checkpoint.
        """

        respond_to_input = respond_to

        def __init__(self):
            super().__init__()
            # gain 1/2 per block: dyadic, so the state at layer l is exactly
            # 2^-l times the post-embedding state and every expected value below
            # is exact in binary floating point
            self.transformer_encoder = _Encoder([0.5] * n_layers)
            self.register_buffer("a", torch.as_tensor(a, dtype=torch.float32))
            self.register_buffer("m", torch.as_tensor(m, dtype=torch.float32))
            self.register_buffer("V", torch.as_tensor(V, dtype=torch.float32))

        def drive(self, values, flags):  # noqa: ANN001
            """The scalar per row that the hidden state is a function of."""
            source = flags if self.respond_to_input == "flag" else values
            return source.to(torch.float32).sum(dim=-1)

        def forward(self, src, values, flags, src_key_padding_mask=None):  # noqa: ANN001
            d = self.drive(values, flags)                      # (B,)
            coeff = self.a[None, :] + self.m[None, :] * d[:, None]
            h0 = coeff[:, :, None] * self.V[None, None, :]     # (B, n_readout, d)
            h = self.transformer_encoder(h0, src_key_padding_mask=src_key_padding_mask)
            # never read by run 6 --- it emits hidden states only --- but
            # ``_forward`` indexes it, so it has to be a real tensor
            return {"mlm_output": h.mean(dim=-1)}

    toy = FlagToy()
    toy.coefficients = {"a": a, "m": m, "V": V, "block_gain": 0.5, "n_layers": n_layers}
    return toy


def flag_toy_expected(toy, drive: float, layer: int) -> np.ndarray:
    """``(n_readout, d)`` state a ``make_flag_toy`` module produces, in float64.

    ``drive`` is the summed value (or flag) the row carries. Kept beside the toy
    so the expected value and the toy cannot drift apart in separate files.
    """
    c = toy.coefficients
    gain = float(c["block_gain"]) ** int(layer)
    return gain * np.outer(c["a"] + c["m"] * float(drive), c["V"])
