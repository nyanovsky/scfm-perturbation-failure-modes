"""The estimator and the driver, checked against arithmetic before a GPU hour.

``tests/test_slots.py`` checks where a slot *lands*; this file checks what a slot
*is*. Run 6's feature is a derivative of a hidden state with respect to one
gene's input value, and every way of getting it wrong produces a well-shaped
float32 array of some other quantity: a ``gdt`` built from the baseline row
instead of the two flanking rows is a substitution response divided by a step
size, a ``gdt`` divided by ``2h`` where the grid floor clamped the low row is out
by a factor of two at dose 0, a ``vvp`` whose batch changed a second input column
is a sum of two interventions, and a projection whose k-ladder is not a column
prefix makes every rung below ``K_STORE`` a decision taken after seeing a score.
None of those raise, none of them are visible in a shape, and all of them would
still be scored. So they are checked here against numbers a reader can compute by
hand.

Exactness is the point, not the style. ``tests/toy.py`` returns float64 with
dyadic coefficients and is driven at a dyadic ``STEP``, so the recovered
derivative is the analytic one *in binary floating point* --- these tests assert
``max abs diff == 0.0``. A tolerance would accept a denominator wrong by 1e-9 and
would leave a reader unable to tell arithmetic from noise. Where a tolerance does
appear (the two ``project`` tests) it is because the quantity really is a float32
matmul rather than exact arithmetic, and the docstring says so.

What a reader must NOT conclude from a green run here. These toys are not
models. ``LinearDoseToy`` and ``QuadraticDoseToy`` respond all-to-all at every
depth, which no transformer does and which makes their layer-0 off-target delta
deliberately non-zero; ``IsolatedToy`` responds nowhere off-target, which no
transformer does either. Passing says the estimator recovers a known derivative
and that the driver's own loop carries it through unchanged. It says nothing
whatever about whether a real checkpoint's hidden state carries a response --- and
it cannot, because the toys were written to have a knowable answer, not a
realistic one.

Invocation (from the run directory):  PYTHONPATH=src pytest tests/test_extract.py -q
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from scfm_hidarm import config as C
from scfm_hidarm import sibling
from scfm_hidarm.basis import fit as basisfit
from scfm_hidarm.extract import run as D
from scfm_hidarm.extract import slots as S
from scfm_hidarm.nulls import random_orthonormal
from toy import (
    D_MODEL,
    N_READOUT,
    STEP,
    IsolatedToy,
    LinearDoseToy,
    MuteToy,
    QuadraticDoseToy,
)

#: the perturbed input column every slot test drives. Not 0 and not the last
#: column, so an off-by-one in ``np.delete(..., axis=0)`` on the self position
#: cannot be masked by falling off an edge of the array.
PERT_COL = 5

#: a second column, used to construct the one violation ``query_rows`` exists to
#: catch. Deliberately not adjacent to ``PERT_COL``.
OTHER_COL = 9

#: the anchor the slot tests read at: dyadic (multiples of 1/8, exact in float32
#: and float64), far above every grid dose so no query is accidentally a no-op,
#: and distinct per gene so a transposed read-out axis would be visible.
ANCHOR = 128.0 + np.arange(N_READOUT, dtype=np.float64) / 8.0

#: the layers the slot tests read. Layer 0 is included because ``layer_gain`` is
#: ``2^-l`` and only a gain of 1 would hide a missing gain factor.
SLOT_LAYERS = (0, 1, 3)

#: the value grid in the toys' native units --- the identity, because
#: ``_ToyBase.to_native`` is the identity, which is what lets a test's expected
#: value be the dose it names.
VALUES = np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64)

#: the model key the driver tests run under. A *real* key, because
#: ``config.model_stem`` refuses an unknown model and the driver builds its
#: output file names through it; the toy's width is installed against this key
#: rather than a fake one added to ``config.MODELS``.
DRIVER_MODEL = "scprint"
DRIVER_LINE = "hepg2"

#: the stored width the driver tests use. It has to be at most the toy's hidden
#: width (6) for an orthonormal ``d -> k`` basis to exist at all, and it is a rung
#: of ``config.K_LADDER`` so the captured-energy diagnostic reports on it.
DRIVER_K = 4


def _hidden(toy, col: int = PERT_COL, step: float = STEP, x0: np.ndarray | None = None):
    """The 16 captured rows for one gene, through ``query_rows`` and the toy.

    The batch is built by the production function rather than by hand: a test
    that assembled its own 16 rows would check the estimator against the test's
    own idea of the row order, which is the very thing at issue.
    """
    x0 = np.asarray(ANCHOR if x0 is None else x0, dtype=np.float32)
    batch = S.query_rows(x0, col, VALUES, step)
    return batch, toy.hidden_batch(batch, SLOT_LAYERS, perturbed_col=col)


# --------------------------------------------------------------------------- #
# 1. the query batch
# --------------------------------------------------------------------------- #
def test_the_query_batch_is_the_anchor_then_three_rows_per_dose() -> None:
    """Row ``1 + 3v / 2 + 3v / 3 + 3v`` must be ``(v - h, v, v + h)`` at that dose.

    This is the row contract ``build_slots`` indexes against, and it is the one
    place it can be checked without a model: get the order wrong and ``gdt``
    becomes a difference of two different doses' rows, which is a well-formed
    array of a quantity nobody asked for.
    """
    x0 = np.asarray(ANCHOR, dtype=np.float32)
    batch = S.query_rows(x0, PERT_COL, VALUES, STEP)

    assert batch.shape == (C.INPUT_ROWS_PER_GENE, N_READOUT) == (16, 12)
    assert np.abs(batch[0] - x0).max() == 0.0
    for vi, v in enumerate(VALUES):
        lo = max(float(v) - STEP, C.NATIVE_FLOOR)
        for offset, want in zip((1, 2, 3), (lo, float(v), float(v) + STEP)):
            row = batch[offset + 3 * vi]
            assert row[PERT_COL] == np.float32(want), f"dose {v}, row offset {offset}"
            # every other column is the anchor's, bit for bit
            assert np.abs(np.delete(row, PERT_COL) - np.delete(x0, PERT_COL)).max() == 0.0


def test_the_low_side_is_clamped_at_the_grid_floor() -> None:
    """At dose 0 the low row would be negative, and a negative count is not an input.

    The clamp is why the estimator is one-sided at the bottom of the grid, so
    this test and ``gdt_denominator``'s are two halves of one claim: clamp the
    row and forget the denominator and every dose-0 derivative is halved.
    """
    batch = S.query_rows(np.asarray(ANCHOR, dtype=np.float32), PERT_COL, VALUES, STEP)
    vi = list(C.VALUE_GRID_CP10K).index(0.0)
    # the clamp has to actually bind, or the assertion below is vacuous
    assert 0.0 - STEP < C.NATIVE_FLOOR
    assert batch[1 + 3 * vi][PERT_COL] == np.float32(C.NATIVE_FLOOR)
    assert batch[3 + 3 * vi][PERT_COL] == np.float32(STEP)
    # and nowhere else on the grid, where v - h is above the floor
    for other in (1.0, 4.0, 16.0, 64.0):
        oi = list(C.VALUE_GRID_CP10K).index(other)
        assert batch[1 + 3 * oi][PERT_COL] == np.float32(other - STEP)


def test_a_second_changed_input_column_is_refused(monkeypatch) -> None:
    """A batch that moved two inputs makes delta-h a sum of two interventions.

    The violation has to be injected, because ``query_rows`` cannot produce it on
    its own --- which is the point: the assertion guards the *callers* that
    re-tile or re-order a baseline, the failure run 5 found. A second edit is
    spliced in at the one seam between building the rows and checking them, so
    the check under test is the real one.
    """
    real_stack = np.stack

    def stack_with_a_second_edit(rows, *args, **kwargs):
        edited = [np.array(r, copy=True) for r in rows]
        edited[4][OTHER_COL] = np.float32(edited[4][OTHER_COL] + 1.0)
        return real_stack(edited, *args, **kwargs)

    monkeypatch.setattr(S.np, "stack", stack_with_a_second_edit)
    with pytest.raises(RuntimeError, match="rather than only at the perturbed column"):
        S.query_rows(np.asarray(ANCHOR, dtype=np.float32), PERT_COL, VALUES, STEP)


def test_a_non_finite_anchor_column_is_refused() -> None:
    """A ``nan`` in the anchor trips the same check, and must.

    ``nan != nan``, so a non-finite anchor column reports as differing between
    the baseline and every perturbed row --- and it really would poison every
    delta built from the batch. The check catching it is not a coincidence worth
    tolerating silently, so it is asserted here.
    """
    x0 = np.asarray(ANCHOR, dtype=np.float32).copy()
    x0[OTHER_COL] = np.float32("nan")
    with pytest.raises(RuntimeError, match="rather than only at the perturbed column"):
        S.query_rows(x0, PERT_COL, VALUES, STEP)


# --------------------------------------------------------------------------- #
# 2. the affine toy: the row indexing
# --------------------------------------------------------------------------- #
def test_the_affine_toy_recovers_vvp_and_gdt_exactly() -> None:
    """``vvp = gain m_g (v - anchor) V`` and ``gdt = gain m_g V``, bit for bit.

    On an affine hidden state the derivative does not depend on the span it is
    measured over, so this toy cannot see a wrong denominator (that is
    ``QuadraticDoseToy``'s job) --- what it can see is a slot built from the
    wrong pair of rows, at every layer and every dose at once.

    The comparison is ``== 0.0`` rather than a tolerance because every quantity
    involved is a dyadic rational with fewer than 30 significant bits: the toy's
    coefficients are multiples of 1/8, the doses and the anchor are exact, the
    step is 2^-5, and the layer gain is 2^-l. Both routes to the answer are
    therefore exact, so they agree exactly or the estimator is wrong.
    """
    toy = LinearDoseToy(seed=C.SEED)
    _, hidden = _hidden(toy)
    base = float(ANCHOR[PERT_COL])

    for layer in SLOT_LAYERS:
        built = S.build_slots(hidden[int(layer)], VALUES, STEP)
        gain = toy.layer_gain(int(layer))
        assert sorted(built) == sorted(C.SLOT_FORMS)
        assert built["vvp"].shape == (VALUES.size, N_READOUT, D_MODEL)
        for vi, v in enumerate(VALUES):
            want_vvp = gain * np.outer(toy.m * (float(v) - base), toy.V)
            want_gdt = gain * np.outer(toy.m, toy.V)
            assert np.abs(built["vvp"][vi] - want_vvp).max() == 0.0, f"l{layer} vvp v={v}"
            # including v = 0, where the low row is clamped: an affine
            # function's slope is the same over a span of h as over 2h
            assert np.abs(built["gdt"][vi] - want_gdt).max() == 0.0, f"l{layer} gdt v={v}"


def test_the_affine_toys_vvp_is_the_toys_own_state_difference() -> None:
    """``vvp`` is a substitution response, so it is one state minus the baseline state.

    Checked against ``toy.expected`` --- the toy's own formula, written once
    beside the toy --- to pin the *sign convention*: run 2's ``vvp`` is perturbed
    minus baseline, and a flipped sign would leave every magnitude test above
    passing while making run 6's deltas point the opposite way from the published
    head features they are compared against.
    """
    toy = LinearDoseToy(seed=C.SEED + 1)
    _, hidden = _hidden(toy)
    layer = SLOT_LAYERS[-1]
    built = S.build_slots(hidden[int(layer)], VALUES, STEP)
    base_state = toy.expected(float(ANCHOR[PERT_COL]), int(layer))
    for vi, v in enumerate(VALUES):
        want = toy.expected(float(v), int(layer)) - base_state
        assert np.abs(built["vvp"][vi] - want).max() == 0.0, f"v={v}"


def test_a_row_count_that_is_not_one_baseline_plus_three_per_dose_is_refused() -> None:
    """Run 5's 6-row batch through run 6's estimator must not silently build a slot.

    Reusing run 5's cheaper batch is the realistic mistake here --- it is the same
    anchor, the same grid and the same adapter --- and it would turn ``gdt`` into a
    difference of two different doses' rows.
    """
    toy = LinearDoseToy(seed=C.SEED)
    _, hidden = _hidden(toy)
    six = hidden[SLOT_LAYERS[-1]][:6]
    with pytest.raises(RuntimeError, match="one baseline then"):
        S.build_slots(six, VALUES, STEP)


# --------------------------------------------------------------------------- #
# 3. the quadratic toy: the denominator
# --------------------------------------------------------------------------- #
def test_the_quadratic_toy_recovers_the_analytic_derivative_above_the_floor() -> None:
    """``gdt = gain 2 m_g v V`` at the four doses whose low row is not clamped.

    **This is the test that can falsify ``gdt_denominator``.** On the affine toy
    a wrong denominator is invisible, because the numerator scales with the span
    by exactly the same factor; on a quadratic it is not, and a central
    difference divided by ``h`` instead of ``2h`` comes back at twice the true
    slope.

    Exact, again by construction: ``((v+h)^2 - (v-h)^2) / (2h)`` is exactly
    ``2v`` in binary floating point when ``v`` and ``h`` are dyadic --- the two
    squares are exactly representable, their difference is exact, and the
    division is by a power of two.
    """
    toy = QuadraticDoseToy(seed=C.SEED)
    _, hidden = _hidden(toy)
    for layer in SLOT_LAYERS:
        built = S.build_slots(hidden[int(layer)], VALUES, STEP)
        gain = toy.layer_gain(int(layer))
        for v in (1.0, 4.0, 16.0, 64.0):
            vi = list(C.VALUE_GRID_CP10K).index(v)
            want = gain * np.outer(2.0 * toy.m * v, toy.V)
            assert np.abs(built["gdt"][vi] - want).max() == 0.0, f"l{layer} v={v}"


def test_the_quadratic_toys_clamped_dose_is_a_one_sided_difference() -> None:
    """At dose 0 the estimator is a forward difference, and it is NOT the derivative.

    Asserted separately and stated plainly, because the honest answer at the
    bottom of the grid is not ``2 m_g v V = 0`` but ``m_g h V``: the low row is
    clamped at ``NATIVE_FLOOR``, so what is estimated is the slope of the secant
    from 0 to ``h``, which on a quadratic is ``h`` rather than 0. Writing the test
    the other way round --- expecting zero --- would be asserting that the dose-0
    ``gdt`` slot means something it does not, and every dose-0 column of every
    real block would then be read as an analytic derivative it is not.
    """
    toy = QuadraticDoseToy(seed=C.SEED)
    _, hidden = _hidden(toy)
    vi = list(C.VALUE_GRID_CP10K).index(0.0)
    for layer in SLOT_LAYERS:
        built = S.build_slots(hidden[int(layer)], VALUES, STEP)
        gain = toy.layer_gain(int(layer))
        want = gain * np.outer(toy.m * STEP, toy.V)
        assert np.abs(built["gdt"][vi] - want).max() == 0.0, f"l{layer}"
        # and it is not zero, so the one-sided case is a real measurement rather
        # than a hole in the block
        assert np.abs(built["gdt"][vi]).max() > 0.0


def test_the_denominator_is_one_sided_at_the_floor_and_two_sided_above_it() -> None:
    """``h`` at dose 0, ``2h`` everywhere else --- the span the rows actually spread over.

    The driver records the same fact per dose as ``gdt_one_sided_by_dose``, so
    the two must agree: a block whose manifest says a dose is two-sided while the
    estimator divided by ``h`` is a block whose provenance is wrong about its own
    numbers.
    """
    assert S.gdt_denominator(0.0, STEP) == STEP
    for v in (1.0, 4.0, 16.0, 64.0):
        assert S.gdt_denominator(v, STEP) == 2.0 * STEP
    # the production steps are not dyadic (config.GDT_STEP is 0.02 / 1.0), so the
    # denominator is asserted there only as a ratio, not as an exact value
    assert S.gdt_denominator(0.0, 0.02) < S.gdt_denominator(1.0, 0.02)


# --------------------------------------------------------------------------- #
# 4. a no-op is exactly zero, and a moved no-op is a failure
# --------------------------------------------------------------------------- #
def test_a_no_op_query_is_found_by_name_and_is_exactly_zero() -> None:
    """Two bit-identical input rows through a frozen model must give exactly 0.0.

    Not "small": the perturbed row and the baseline row are the same 12 float32
    values, so the two forward passes are the same computation. This is the only
    cell of a real block whose value is known in advance, which makes it the one
    place the whole path can be checked against a certainty rather than against
    another computation --- and TranscriptFormer really has 20 of them in HepG2,
    because its ``to_native`` clips at 30.

    Averaging such a cell in as a very small response would understate every
    magnitude summary of the block by however many no-ops it contains, which is
    why the driver names them rather than tolerating them.
    """
    x0 = np.asarray(ANCHOR, dtype=np.float32).copy()
    x0[PERT_COL] = np.float32(16.0)
    assert 16.0 in C.VALUE_GRID_CP10K
    vi = list(C.VALUE_GRID_CP10K).index(16.0)
    cols = np.arange(N_READOUT, dtype=np.int64)

    mask = S.no_op_mask(x0, cols, VALUES)
    assert mask.shape == (N_READOUT, VALUES.size)
    assert np.argwhere(mask).tolist() == [[PERT_COL, vi]]

    toy = LinearDoseToy(seed=C.SEED)
    _, hidden = _hidden(toy, x0=x0)
    for layer in SLOT_LAYERS:
        built = S.build_slots(hidden[int(layer)], VALUES, STEP)
        assert np.abs(built["vvp"][vi]).max() == 0.0, f"l{layer}"
        # the production assertion agrees, at the gene and dose the mask named
        S.assert_no_op_zero(built["vvp"], vi, PERT_COL, f"TOY{PERT_COL:03d}", 16.0)
        # a gradient at the same base value is a derivative, not a substitution,
        # so it is unaffected and must NOT be zero
        assert np.abs(built["gdt"][vi]).max() > 0.0, f"l{layer}"


def test_a_no_op_slot_that_moved_raises() -> None:
    """The no-op assertion has to be able to fail, or it is decoration.

    A response of 1e-12 at an off-target gene means either the batch is not what
    the module thinks it is or the forward pass is not deterministic; both are
    unrecoverable at analysis time, so they abort the pass rather than being
    recorded. The self column is excluded, and that exclusion is asserted here
    too: TranscriptFormer writes ``nan`` at a gene's own position, and ``nan <=
    0`` is False, which aborted a 524-gene sweep in run 5 before the exclusion.
    """
    toy = LinearDoseToy(seed=C.SEED)
    x0 = np.asarray(ANCHOR, dtype=np.float32).copy()
    x0[PERT_COL] = np.float32(16.0)
    vi = list(C.VALUE_GRID_CP10K).index(16.0)
    _, hidden = _hidden(toy, x0=x0)
    built = S.build_slots(hidden[SLOT_LAYERS[-1]], VALUES, STEP)

    moved = built["vvp"].copy()
    moved[vi, OTHER_COL, 0] = 1e-12
    with pytest.raises(RuntimeError, match="must give exactly zero"):
        S.assert_no_op_zero(moved, vi, PERT_COL, f"TOY{PERT_COL:03d}", 16.0)

    # an undefined self position is an absence, not a violation
    nan_self = built["vvp"].copy()
    nan_self[vi, PERT_COL, :] = np.nan
    S.assert_no_op_zero(nan_self, vi, PERT_COL, f"TOY{PERT_COL:03d}", 16.0)


# --------------------------------------------------------------------------- #
# 5-6. the driver's own loop
# --------------------------------------------------------------------------- #
def _install_driver_stubs(monkeypatch, toy, k: int = DRIVER_K) -> None:
    """Everything ``extract_line`` reaches the real data through, and nothing more.

    Only the six seams that need a checkpoint, run 2's handoff tables or run 3's
    bundle are replaced: ``build_model``, ``anchor_profile``,
    ``scope_perturbations``, ``assert_bundle_order``, ``basisfit.load_basis`` and
    run 5's baseline-pairing probe. Everything the test is about ---
    ``query_rows``, ``build_slots``, ``no_op_mask``, ``assert_no_op_zero``,
    ``project``, ``pack_block``, the layer loop, the C2 branch and the
    second-moment accumulation --- is the production code.

    Two constants are installed against a **real** model key rather than a fake
    one. ``config.model_stem`` refuses an unknown model and the driver names
    every output file through it, so a fake key could not be written; and
    ``config.D_MODEL[key]`` must equal the toy's hidden width, which cannot be
    changed on the toy side (``tests/toy.py`` fixes it at 6 and is not edited
    here). Patching one dict entry and ``K_STORE`` is the smaller intervention:
    ``capture_layers`` is bypassed by passing ``layers`` explicitly, so
    ``N_LAYERS`` never has to move.
    """
    readout = pd.DataFrame(
        {
            "gene_id": [f"TOY{i:03d}" for i in range(N_READOUT)],
            "pooled_control_cp10k": ANCHOR,
        }
    )
    perturbed = pd.DataFrame(
        {"gene_id": readout["gene_id"], "readout_col": np.arange(N_READOUT, dtype=np.int64)}
    )
    monkeypatch.setitem(C.D_MODEL, DRIVER_MODEL, int(toy.d_model))
    monkeypatch.setattr(C, "K_STORE", int(k))
    monkeypatch.setattr(
        D, "build_model", lambda model, cell_line, device, cache=None: (toy, readout, {})
    )
    monkeypatch.setattr(
        D, "anchor_profile", lambda r: np.asarray(ANCHOR, dtype=np.float64)
    )
    monkeypatch.setattr(
        D,
        "scope_perturbations",
        lambda model, cell_line, r, scope: (perturbed, {"mode": "toy"}),
    )
    monkeypatch.setattr(
        D,
        "assert_bundle_order",
        lambda cell_line, g, r: {"detail": "toy rows, in the order the stub supplied"},
    )
    monkeypatch.setattr(
        basisfit,
        "load_basis",
        lambda model, cell_line, layer: {
            "W": random_orthonormal(int(toy.d_model), int(k), C.SEED + 100 * int(layer))
        },
    )
    monkeypatch.setattr(
        sibling,
        "run5",
        lambda module=None: types.SimpleNamespace(
            _baseline_pairing_probe=lambda q, x0, cols, layers, hidden: {"stub": True}
        ),
    )


def test_layer_zero_carries_no_off_target_delta_through_the_driver(
    monkeypatch, tmp_path
) -> None:
    """Control C2, read out of the driver rather than out of a toy: exactly 0.0.

    An edit at gene *i* cannot have reached gene *j*'s token before the first
    transformer block, so the layer-0 off-target delta is zero by construction
    and anything else means the hooks are attributed to the wrong module. The
    pair is the check: ``IsolatedToy`` must come back at exactly 0.0 and the
    all-to-all ``LinearDoseToy``, whose layer-0 state does depend on the
    perturbed column, must not --- otherwise a driver that zeroed everything would
    pass the first half.

    Only layer 0 is requested here, deliberately. ``IsolatedToy``'s off-target
    delta is exactly zero at *every* depth, so a sweep layer's second moment
    would be the zero matrix and ``basis.fit.eigendecompose`` would refuse it as
    a broken extraction (see the test below, which records that). Requesting the
    control layer alone keeps this test about the C2 branch of the driver's loop.
    """
    for toy_cls, isolated in ((IsolatedToy, True), (LinearDoseToy, False)):
        toy = toy_cls(seed=C.SEED)
        _install_driver_stubs(monkeypatch, toy)
        out = D.extract_line(
            DRIVER_MODEL,
            DRIVER_LINE,
            device="cpu",
            layers=(C.EMBEDDING_LAYER,),
            out_root=tmp_path / toy_cls.__name__,
            log_every=0,
        )
        assert out["available"] is True
        c2 = out["diagnostics"]["c2_layer0_max_off_target"]
        assert set(c2) == {C.EMBEDDING_LAYER}
        worst = float(c2[C.EMBEDDING_LAYER])
        if isolated:
            assert worst == 0.0, f"layer 0 propagated: {worst}"
        else:
            assert worst > 0.0, "the all-to-all toy's layer 0 was reported as C2-clean"


def test_the_driver_writes_a_block_per_variant_and_sweep_layer(
    monkeypatch, tmp_path
) -> None:
    """The positive control for the two tests around it, and the axis-1 contract.

    Two blocks come out of every sweep layer --- ``hid`` through the fitted basis
    and ``hidrp`` through a frozen random one of the same width --- and layer 0
    gets none, because its block would be a zero matrix with one non-zero column.
    Axis 1 is ``n_doses * k`` wide and its ``dose_index`` / ``pc_index`` labels
    travel with it, so run 3 (which gathers the axis whole and has no idea what
    the columns mean) can be joined against rather than trusted.
    """
    toy = LinearDoseToy(seed=C.SEED)
    _install_driver_stubs(monkeypatch, toy)
    root = tmp_path / "blocks"
    root.mkdir()
    layers = (C.EMBEDDING_LAYER, 1, 3)
    out = D.extract_line(
        DRIVER_MODEL, DRIVER_LINE, device="cpu", layers=layers, out_root=root, log_every=0
    )
    assert out["available"] is True

    names = sorted(p.name for p in root.iterdir())
    for layer in (1, 3):
        for variant in ("hid", "hidrp"):
            key = C.model_stem(DRIVER_MODEL, variant, layer, DRIVER_K)
            assert f"{key}.npz" in names and f"{key}.json" in names
        assert f"scored_moment_l{layer}.npz" in names
    assert not any(n.startswith(f"{DRIVER_MODEL}-hid-l{C.EMBEDDING_LAYER}") for n in names)

    with np.load(root / f"{C.model_stem(DRIVER_MODEL, 'hid', 1, DRIVER_K)}.npz") as z:
        n_v = len(C.VALUE_GRID_CP10K)
        assert z["vvp"].shape == (N_READOUT, n_v * DRIVER_K, N_READOUT)
        assert z["gdt"].shape == z["vvp"].shape
        assert z["dose_index"].tolist() == np.repeat(np.arange(n_v), DRIVER_K).tolist()
        assert z["pc_index"].tolist() == np.tile(np.arange(DRIVER_K), n_v).tolist()
        assert z["perturbed_gene_id"].tolist() == [f"TOY{i:03d}" for i in range(N_READOUT)]
    # the one-sidedness of the dose-0 estimator is recorded, not left implicit
    assert out["diagnostics"]["gdt_one_sided_by_dose"][0] is True
    assert out["diagnostics"]["gdt_one_sided_by_dose"][1:] == [False] * 4


def test_a_model_with_no_hidden_states_is_recorded_and_writes_nothing(
    monkeypatch, tmp_path
) -> None:
    """An absence carries its reason verbatim; a block of zeros would not.

    This is the failure that cannot be caught downstream. A zero-filled block is
    indistinguishable from a model whose intervention propagates nothing, which
    is precisely the finding run 6 exists to test --- so a missing measurement
    must never be able to become one. The output directory is created first and
    then asserted empty, so "nothing was written" is a listing rather than an
    absent path, and the test above shows the same call *does* write when the
    adapter supplies states.
    """
    toy = MuteToy(seed=C.SEED)
    _install_driver_stubs(monkeypatch, toy)
    root = tmp_path / "blocks"
    root.mkdir()
    out = D.extract_line(
        DRIVER_MODEL,
        DRIVER_LINE,
        device="cpu",
        layers=(C.EMBEDDING_LAYER, 1, 3),
        out_root=root,
        log_every=0,
    )
    assert out["available"] is False
    assert out["reason"] == toy._hidden_error
    assert "block of zeros" in out["reason"]
    assert list(root.iterdir()) == []


def test_a_model_with_no_off_target_response_at_all_aborts_the_main_pass(
    monkeypatch, tmp_path
) -> None:
    """Recorded behaviour, not an endorsement: ``IsolatedToy`` cannot be written out.

    With no off-target response at any depth, every vector fed to the *scored*
    second moment is exactly zero, so the captured-energy diagnostic's
    eigendecomposition has zero total energy and raises --- a diagnostic
    documented as "never used to score" therefore aborts the pass. No real
    checkpoint gives exactly zero, so this cannot fire in production; it is
    asserted here so that the interaction is on the record and so that a change
    to it is visible rather than silent.
    """
    toy = IsolatedToy(seed=C.SEED)
    _install_driver_stubs(monkeypatch, toy)
    with pytest.raises(RuntimeError, match="zero total energy"):
        D.extract_line(
            DRIVER_MODEL,
            DRIVER_LINE,
            device="cpu",
            layers=(C.EMBEDDING_LAYER, 1),
            out_root=tmp_path / "isolated",
            log_every=0,
        )


# --------------------------------------------------------------------------- #
# 7. the packing
# --------------------------------------------------------------------------- #
def _planted(n_p: int, n_v: int, n_r: int, k: int) -> np.ndarray:
    """A ``(n_pert, n_dose, n_readout, k)`` array whose every entry names its own index.

    Distinct small integers, all exactly representable in float32, so a
    misplaced column is an exact mismatch rather than a tolerance question.
    """
    p = np.arange(n_p)[:, None, None, None] * 100000
    v = np.arange(n_v)[None, :, None, None] * 10000
    r = np.arange(n_r)[None, None, :, None] * 100
    c = np.arange(k)[None, None, None, :]
    return (p + v + r + c).astype(np.float32)


def test_pack_block_lays_axis_one_down_in_dose_pc_c_order() -> None:
    """``index = dose_i * k + pc``, which is ``config.SLOT_AXIS_ORDER``.

    Run 3 gathers ``e[form][m, :, cols]`` whole and would score any ordering, so
    this is a contract only run 6 can assert. Get it wrong and the dose axis --- the
    axis this run exists to recover --- is scrambled inside the feature vector
    while every number downstream is still produced.
    """
    n_p, n_v, n_r, k = 3, len(C.VALUE_GRID_CP10K), 4, 6
    arr = _planted(n_p, n_v, n_r, k)
    out = S.pack_block(arr, k)
    assert out.shape == (n_p, n_v * k, n_r)
    for dose_i in range(n_v):
        for pc in range(k):
            col = dose_i * k + pc
            assert np.array_equal(out[:, col, :], arr[:, dose_i, :, pc]), col
    assert C.SLOT_AXIS_ORDER.startswith("(dose, pc)")


def test_pack_block_refuses_a_width_that_is_not_the_stored_k() -> None:
    """A block one column narrower than its labels would install and score anyway."""
    arr = _planted(2, len(C.VALUE_GRID_CP10K), 3, 4)
    with pytest.raises(RuntimeError, match="expected k="):
        S.pack_block(arr, 5)


# --------------------------------------------------------------------------- #
# 8-9. the projection and the k-ladder
# --------------------------------------------------------------------------- #
def test_projection_through_an_orthonormal_basis_preserves_row_norms() -> None:
    """An orthonormal basis rotates the slot; it does not rescale it.

    This is why ``hidrp`` is drawn orthonormal rather than as an arbitrary
    Gaussian projection: if the random control changed the geometry as well as
    the directions, ``hid`` vs ``hidrp`` would confound "the basis directions
    matter" with "one projection is not an isometry on its image".

    A tolerance here, not an exact zero, and for a stated reason: ``project``
    casts to float32 and a ``(12, 32) @ (32, 32)`` matmul accumulates in float32,
    so the isometry holds to ~1e-6 relative and no better. The claim is
    norm preservation, not bit-exactness.
    """
    rng = np.random.default_rng(C.SEED + 11)
    d = 32
    slot = rng.standard_normal((N_READOUT, d)).astype(np.float32)
    W = random_orthonormal(d, d, C.SEED + 12)
    out = S.project(slot, W)
    assert out.shape == (N_READOUT, d)
    assert out.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(out.astype(np.float64), axis=1),
        np.linalg.norm(slot.astype(np.float64), axis=1),
        rtol=1e-5,
        atol=0.0,
    )


def test_projection_through_a_rank_deficient_basis_loses_energy_and_never_gains_it() -> None:
    """A basis that does not span ``d`` can only lose signal, which is the safe side.

    The direction of this error is load-bearing for the whole run: the basis is
    fitted on perturbations disjoint from every scored one, so if it under-spans
    the scored response it *loses* signal and cannot manufacture it --- the failure
    mode is a false null, never a false positive. That argument is only sound if
    a projection cannot amplify, so it is checked rather than asserted in prose.
    """
    rng = np.random.default_rng(C.SEED + 13)
    d = 32
    slot = rng.standard_normal((N_READOUT, d)).astype(np.float32)
    W = random_orthonormal(d, d, C.SEED + 14)
    deficient = W.copy()
    deficient[:, 3] = 0.0  # rank d - 1 in d columns

    full = np.linalg.norm(slot.astype(np.float64), axis=1)
    lost = np.linalg.norm(S.project(slot, deficient).astype(np.float64), axis=1)
    # never gains: the 1e-6 slack is the float32 matmul, not a real increase
    assert (lost <= full * (1.0 + 1e-6)).all()
    # and really loses: one direction of a 32-dimensional slot is gone
    assert (lost < full).all()


def test_the_k_ladder_is_a_column_prefix_of_the_stored_block() -> None:
    """Every rung below ``K_STORE`` is a slice of the stored block, not a second fit.

    ``config.K_LADDER`` is only pre-registered if reading a lower rung costs
    nothing: the components are eigenvalue-ordered, so the first ``k`` columns of
    the width-64 basis ARE the width-``k`` basis, and no rung can be a choice made
    after seeing a score. One basis of width 64 is built here and sliced, which is
    the operation the ladder claims --- not two fits compared.

    Two claims, at the two precisions each is true at.

    **The basis prefix is exact.** ``W[:, :k]`` is bit-identical to the first
    ``k`` columns of the stored ``W`` (``max abs diff == 0.0``), which is the
    substantive claim: a rung needs no refit, so it cannot encode a decision.

    **The projected prefix is exact only up to float32 reassociation.** ``project``
    casts to float32, and a ``(12, 128) @ (128, 1)`` product is not summed in the
    same order as column 0 of a ``(128, 64)`` product, so the two differ by a
    couple of float32 ulps --- measured here at <= 3 ulp, ~3e-7 relative, and
    exactly 0.0 at k = 64 where it is the same call. The tolerance is not a
    loosened claim: the same comparison in float64 comes back at ~1e-16, which is
    the signature of a reassociated sum rather than of a different quantity, and
    the assertion below checks that it shrinks with the precision. A reader
    reading k = 16 out of a stored k = 64 block *slices* it and never re-projects,
    so no production number depends on the float32 gap at all.
    """
    rng = np.random.default_rng(C.SEED + 15)
    d = 128
    slot = rng.standard_normal((N_READOUT, d)).astype(np.float32)
    W = random_orthonormal(d, C.K_STORE, C.SEED + 16)
    stored = S.project(slot, W)
    stored64 = slot.astype(np.float64) @ W.astype(np.float64)

    for k in C.K_LADDER:
        # the basis prefix itself: exact, and the claim the ladder rests on
        assert np.abs(W[:, :k] - W[:, :C.K_STORE][:, :k]).max() == 0.0, f"k={k}"
        rung = S.project(slot, W[:, :k])
        scale = float(np.abs(stored[:, :k]).max())
        gap32 = float(np.abs(stored[:, :k] - rung).max())
        gap64 = float(
            np.abs(stored64[:, :k] - slot.astype(np.float64) @ W[:, :k].astype(np.float64)).max()
        )
        assert gap32 <= 1e-6 * scale, f"k={k}: {gap32:.3g} against a scale of {scale:.3g}"
        assert gap64 <= 1e-12 * scale, f"k={k}: float64 gap {gap64:.3g} is not reassociation"

    # and the prefix relation is not vacuous: the WRONG columns disagree by
    # orders of magnitude more than the reassociation gap
    wrong = S.project(slot, W[:, 4:8])
    assert np.abs(stored[:, :4] - wrong).max() > 1e-3 * float(np.abs(stored[:, :4]).max())
    assert C.K_PRIMARY in C.K_LADDER and max(C.K_LADDER) == C.K_STORE
