"""Turn one perturbed gene's 16 forward rows into the ten hidden-state slots.

Torch-free and model-free on purpose. The whole scientific content of run 6's
feature --- what a ``vvp`` slot is, what a ``gdt`` slot is, and which of the 16
captured rows each is built from --- lives here, so it can be checked against
arithmetic on a toy whose hidden response is a known function of dose, on a
machine with no GPU and no checkpoint. ``extract/run.py`` supplies the rows and
does nothing to them that is not in this file.

The 16 rows are run 2's, not run 5's 6. Run 5 needs one baseline plus one row
per dose because it computes no derivative; run 6 needs ``(v - h, v, v + h)`` at
every dose because ``gdt`` is a central difference, and reusing run 5's cheaper
batch would silently turn the derivative slots into differences of the wrong
pair of rows.

Two things this module refuses to paper over:

**A no-op is not a small response.** Where the perturbed value already equals
the anchor's own value at that gene, the two passes are bit-identical and the
``vvp`` delta is exactly zero. That is asserted rather than assumed, counted,
and named by (gene, dose) --- never averaged in as if it were a measurement.
TranscriptFormer has 20 real ones in HepG2, all at 64 CP10K, because its
``to_native`` clips at the model's training-time ``COUNT_CLIP = 30``.

**An undefined self position is an absence, not a zero.** TranscriptFormer never
reads a gene's count at its own position, so its ``(p, p)`` hidden delta is
``nan`` and stays ``nan``. Every scoring cell excludes that column by
construction; writing a zero there would put a fabricated number into a feature
matrix and nothing downstream could tell.
"""

from __future__ import annotations

import numpy as np

from .. import config as C


def query_rows(x0: np.ndarray, col: int, values_native: np.ndarray, step: float) -> np.ndarray:
    """Run 2's 16-row query batch: the anchor, then ``(v-h, v, v+h)`` per dose.

    Reproduces ``run2/features/extract.py:query_batch`` exactly, including the
    ``NATIVE_FLOOR`` clamp on the low side, and then asserts what run 2 leaves
    implicit: that every perturbed row differs from the baseline row at the
    perturbed column and at no other. Run 5 added that assertion after finding
    that a re-tiled baseline would mix an intervention with a composition
    change; it costs nothing and it is the only thing standing between a delta
    and a delta of something else.
    """
    x0 = np.asarray(x0, dtype=np.float32)
    rows = [x0]
    for v in np.asarray(values_native, dtype=np.float64):
        lo = max(float(v) - float(step), C.NATIVE_FLOOR)
        for w in (lo, float(v), float(v) + float(step)):
            x = x0.copy()
            x[int(col)] = np.float32(w)
            rows.append(x)
    batch = np.stack(rows)
    # 1 + 3 per dose, which at the production grid is config.INPUT_ROWS_PER_GENE
    # (16). Checked against the grid actually supplied rather than against the
    # constant, so a smoke run on a shorter grid is still checked and does not
    # have to disable the check to pass it.
    expected = 1 + 3 * int(np.asarray(values_native).size)
    if batch.shape[0] != expected:
        raise RuntimeError(
            f"the query batch has {batch.shape[0]} rows, not the {expected} the gdt "
            f"estimator needs (1 baseline + 3 per dose over {np.asarray(values_native).size} "
            "doses)"
        )
    differs = np.flatnonzero((batch[1:] != batch[0]).any(axis=0))
    if differs.size > 1 or (differs.size == 1 and int(differs[0]) != int(col)):
        raise RuntimeError(
            f"the perturbed rows differ from the baseline row at input columns "
            f"{differs.tolist()} rather than only at the perturbed column {col}; every "
            "delta built from this batch would mix the intervention with a second "
            "input change"
        )
    return batch


def mid_row_values(values_native: np.ndarray, step: float) -> np.ndarray:
    """The float32 values the ``v`` rows actually carry, for the no-op test.

    The comparison is made on what the batch carries and not on the float64
    grid, because a no-op is defined by the two input rows being BIT-identical
    --- which is what makes the delta exactly zero rather than merely small.
    """
    return np.array([np.float32(float(v)) for v in np.asarray(values_native)], dtype=np.float32)


def gdt_denominator(v: float, step: float) -> float:
    """The span the central difference is divided by, run 2's expression.

    One-sided at the bottom of the grid: ``v - h`` is clamped at
    ``NATIVE_FLOOR``, so at ``v = 0`` the estimator is a forward difference over
    a shorter span and the denominator has to match. Run 2 recorded the same
    ``one_sided`` flag per dose; run 6 records it in the block manifest.
    """
    lo = max(float(v) - float(step), C.NATIVE_FLOOR)
    return max(float(v) + float(step) - lo, 1e-12)


def build_slots(
    hidden: np.ndarray,
    values_native: np.ndarray,
    step: float,
) -> dict[str, np.ndarray]:
    """The ten slots for one gene at one layer, from its 16 captured rows.

    ``hidden`` is ``(16, n_readout, d)``: row 0 is the baseline, row ``1 + 3v``
    is ``v - h``, ``2 + 3v`` is ``v``, ``3 + 3v`` is ``v + h``. Returns
    ``{"vvp": (n_values, n_readout, d), "gdt": (n_values, n_readout, d)}``.

    ``vvp`` is the response to a substitution --- perturbed minus baseline, run
    2's sign convention unchanged, so the two runs' deltas have the same sign.
    ``gdt`` is a derivative and is therefore a difference of the two PERTURBED
    rows, never against the baseline: mixing the two would make one slot a
    substitution response scaled by a step size.
    """
    h = np.asarray(hidden)
    n_v = int(np.asarray(values_native).size)
    if h.shape[0] != 1 + 3 * n_v:
        raise RuntimeError(
            f"build_slots got {h.shape[0]} rows for {n_v} doses; it expects "
            f"{1 + 3 * n_v}, one baseline then (v-h, v, v+h) per dose, and any other "
            "layout would build a slot from the wrong pair of rows"
        )
    base = h[0]
    vvp = np.empty((n_v,) + h.shape[1:], dtype=h.dtype)
    gdt = np.empty((n_v,) + h.shape[1:], dtype=h.dtype)
    for vi in range(n_v):
        lo, mid, hi = h[1 + 3 * vi], h[2 + 3 * vi], h[3 + 3 * vi]
        vvp[vi] = mid - base
        gdt[vi] = (hi - lo) / gdt_denominator(float(np.asarray(values_native)[vi]), step)
    return {"vvp": vvp, "gdt": gdt}


def no_op_mask(x0: np.ndarray, cols: np.ndarray, values_native: np.ndarray) -> np.ndarray:
    """``(n_perturbed, n_values)``: which ``vvp`` queries are structural zeros.

    Only ``vvp`` can be a no-op. A gradient at the same base value is a
    derivative rather than a substitution and is unaffected --- run 3's
    ``NO_OP_AFFECTED_FORMS`` records the same distinction for the head features.
    """
    x0 = np.asarray(x0, dtype=np.float32)
    row_values = mid_row_values(values_native, 0.0)
    cols = np.asarray(cols, dtype=np.int64)
    out = np.zeros((cols.size, row_values.size), dtype=bool)
    for pi, col in enumerate(cols):
        out[pi] = row_values == x0[int(col)]
    return out


def assert_no_op_zero(vvp: np.ndarray, vi: int, self_col: int, gene_id: str, dose: float) -> None:
    """Two bit-identical inputs through a frozen model must give exactly zero.

    The perturbed gene's own position is excluded from the comparison: a model
    that does not read a gene's count at its own position writes ``nan`` there,
    and ``nan <= 0`` is False, which aborted a 524-gene sweep at gene ~300 in
    run 5 before the exclusion was added.
    """
    d = np.delete(np.asarray(vvp[vi]), int(self_col), axis=0)
    worst = float(np.abs(d).max()) if d.size else 0.0
    if worst != 0.0:
        raise RuntimeError(
            f"gene {gene_id} at {dose} CP10K is a no-op (the anchor already sits there) "
            f"but the hidden state moved by {worst:.6g}; two bit-identical inputs through "
            "a frozen model must give exactly zero, so either the batch is not what this "
            "module thinks it is or the forward pass is not deterministic"
        )


def project(slot: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``(n_readout, d) @ (d, k)`` in float32, one slot at a time.

    Projection happens on the device the forward pass ran on and the raw
    ``delta_h`` is discarded immediately: storing it would be 15-94 GB per layer
    against the 3.65-5.86 GB the projected block costs, and it is the reason a
    fold-independent basis was chosen in the first place.
    """
    return np.asarray(slot, dtype=np.float32) @ np.asarray(w, dtype=np.float32)


def pack_block(
    slots_by_gene: dict[str, np.ndarray],
    k: int,
) -> np.ndarray:
    """Lay one form's ``(n_pert, n_values, n_readout, k)`` down as run 3 reads it.

    Run 3's ``ArmFeatures`` gathers ``e[form][m, :, cols]``, so axis 1 is the
    feature axis and axis 2 is the read-out gene. The declared order within axis
    1 is C-order over ``(dose, pc)`` --- ``index = dose_i * k + pc`` --- which is
    what keeps the dose axis a comparison of like coordinates rather than a
    flattening of it (``config.SLOT_AXIS_ORDER``).
    """
    a = np.asarray(slots_by_gene)
    n_p, n_v, n_r, kk = a.shape
    if kk != int(k):
        raise RuntimeError(f"packed block has width {kk}, expected k={k}")
    return np.ascontiguousarray(a.transpose(0, 1, 3, 2).reshape(n_p, n_v * kk, n_r))
