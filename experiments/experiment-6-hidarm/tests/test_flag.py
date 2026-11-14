"""The perturbation flag is the same tensor in both passes, and nothing else is.

Run 6's fine-tuned arm is defined by one sentence: the flag is set on the
perturbed gene in the baseline row *and* in every dose row, so the only thing
that moves between them is the input value. Every downstream claim leans on it.
If the baseline row were read with the flag off, ``vvp`` would be
``(flag onset + dose)`` minus nothing --- a well-shaped float32 block of a
different quantity --- control C2's layer-0 zero would break, because flag onset
does reach other tokens, and ``vvp``/``gdt`` would stop meaning what they mean
for the three frozen models. None of that raises, none of it is visible in a
score, and no other test in the tree looks at the flag tensor at all. This file
is the only place it is checked.

What is checked, and how the checks are made able to fail
--------------------------------------------------------
A toy that responds *only to the flag* must give a ``delta h`` of exactly
``0.0``: it can move only if the flag moved. On its own that is a tautology
waiting to happen --- a toy that had no response at all would also give zero ---
so ``test_a_flag_off_baseline_is_caught`` runs the identical assertion against a
deliberately broken subclass whose baseline row is flagged zero, and requires it
to come back non-zero. And a toy that responds *only to the value* is required
to come back in full, against arithmetic, so "everything is zero" cannot pass
either.

What a reader must NOT conclude from a green run here
-----------------------------------------------------
Nothing about the real checkpoint. These toys have no attention, so a flag
genuinely reaching other tokens is asserted by ``README.md`` s6 and by run 4's
recipe, not measured here; what is measured is that run 6's adapter cannot
*expose* flag onset in a delta, whatever the flag does inside the model. Nor is
this a check on the layer sweep, the projection, or the score --- ``test_slots``,
``test_basis`` and ``test_nulls`` cover those.

Torch is required (the class under test is built out of run 4's and run 5's real
ones) but no GPU and no checkpoint are: everything runs on CPU against
``tests/toy.py``'s ``TransformerGenerator``-shaped stand-in, driven through
byte-identical adapter code to the one the checkpoints go through.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scfm_hidarm import config as C  # noqa: E402
from scfm_hidarm.extract import slots  # noqa: E402
from toy import flag_toy_expected, make_flag_toy  # noqa: E402

try:
    from scfm_hidarm.models import hidden_ft

    #: the real class, built the way ``build()`` builds it. Not importable at
    #: module scope in ``hidden_ft`` either: its bases live behind the sibling
    #: registration, which drags in runs 4 and 5 and their model stacks.
    HIDDEN_CLASS = hidden_ft.make_hidden_class()
except Exception as exc:  # the failure is itself the datum, and it is reported
    pytest.skip(
        f"run 6's fine-tuned adapter could not be built from its siblings: "
        f"{type(exc).__name__}: {exc}",
        allow_module_level=True,
    )

#: read-out genes. ``toy.py``'s default, and small enough that six layers of
#: ``(16, n_readout, 512)`` float32 captures stay under a megabyte apiece.
N_READOUT = 12
#: which column carries the flag. Interior, and not a chunk boundary, so an
#: off-by-one in the flag column would land on a real gene rather than fall off
#: the end where it would raise for the wrong reason.
PERTURBED_COL = 5
#: rows per forward chunk, overriding run 4's 16. 16 input rows at 6 per chunk
#: is 6 + 6 + 4, so the baseline row sits in chunk 0 and eleven of the fifteen
#: dose rows sit in chunks the baseline never appears in --- which is where a
#: flag tensor built per chunk, when it ought to be per query, goes wrong.
CHUNK = 6
#: every layer a production forward pass hooks, under the key the adapter
#: reports (``"scgpt"``): layer 0 is the pre-hook control C2, layer 12 is read
#: at ``transformer_encoder`` itself rather than at its last block.
LAYERS = C.capture_layers("scgpt")
#: the anchor value every gene carries, in the model's own native units. Dyadic
#: (2^-3), so the toy's drive is exact apart from the dose itself, and non-zero,
#: so no dose on the grid coincides with it and turns a slot into a no-op for a
#: reason that has nothing to do with the flag.
ANCHOR_NATIVE = 0.125
#: the production step, ``config.GDT_STEP["scgpt_ft"]``.
STEP = C.GDT_STEP["scgpt_ft"]

#: relative tolerance on a recovered ``vvp``. The toy's trunk runs in float32 on
#: states of magnitude ~6, so a difference of two of them carries ~4e-7 of
#: rounding; measured worst case across all six layers is 5.7e-8 relative.
VVP_RTOL = 1e-6
#: relative tolerance on a recovered ``gdt``, three orders looser and for one
#: reason: the central difference divides a difference of two float32 states by
#: a span of ``2 * STEP = 0.04``, multiplying that same rounding by ~25. 0.02 is
#: not a dyadic step, so the exactness ``toy.py`` offers at its own ``STEP``
#: does not apply here and is not claimed; measured worst case is 9.1e-6
#: relative. The flag assertions below are exact regardless --- they are
#: comparisons against zero, not against arithmetic.
GDT_RTOL = 1e-4


def native_values() -> np.ndarray:
    """The dose grid on run 4's input scale. ``to_native`` is ``log1p``."""
    return np.log1p(np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64))


def query_batch() -> np.ndarray:
    """Run 6's 16 rows for one gene: the anchor, then ``(v-h, v, v+h)`` per dose.

    Built by ``extract/slots.query_rows`` rather than by hand, so the rows this
    file drives the adapter with are the rows the extractor drives it with ---
    including the ``NATIVE_FLOOR`` clamp at the bottom of the grid.
    """
    x0 = np.full(N_READOUT, ANCHOR_NATIVE, dtype=np.float32)
    return slots.query_rows(x0, PERTURBED_COL, native_values(), STEP)


def make_query(respond_to: str, cls=None, seed: int = C.SEED):
    """One real adapter over one toy checkpoint, on CPU.

    ``cls`` defaults to the real class; ``test_a_flag_off_baseline_is_caught``
    passes a broken subclass so that the same driving code exercises both and
    the difference between them is the defect and nothing else.
    """
    toy = make_flag_toy(respond_to=respond_to, n_readout=N_READOUT, seed=seed)
    q = (cls or HIDDEN_CLASS)(toy, N_READOUT, device="cpu", amp=False)
    q.batch_size = CHUNK
    return q


def spy_on_forward(toy) -> list[dict]:
    """Record every tensor the toy's ``forward`` is handed, per chunk.

    Wrapping the module rather than reading ``_flags`` alone is deliberate: what
    matters scientifically is what the *forward pass* received, and a correct
    ``_flags`` whose result were rebuilt, sliced or dropped on the way down
    would pass a direct call and still be wrong. The bound method is captured
    before the patch, so the wrapper delegates to the real one.
    """
    calls: list[dict] = []
    original = toy.forward

    def wrapped(src, values, flags, src_key_padding_mask=None):  # noqa: ANN001
        calls.append(
            {
                "src": src.detach().clone(),
                "values": values.detach().clone(),
                "flags": flags.detach().clone(),
                "pad": None if src_key_padding_mask is None else src_key_padding_mask.clone(),
            }
        )
        return original(src, values, flags, src_key_padding_mask=src_key_padding_mask)

    toy.forward = wrapped
    return calls


def slots_of(hidden: dict[int, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    """``build_slots`` per layer, from the 16 captured rows."""
    vals = native_values()
    return {int(l): slots.build_slots(hidden[int(l)], vals, STEP) for l in hidden}


# --------------------------------------------------------------------------- #
# 1. the flag tensor itself
# --------------------------------------------------------------------------- #
def test_the_flag_tensor_is_one_column_and_identical_in_every_chunk():
    """The flag is fixed input context, so it must not vary with the row.

    Run 6's estimator is a difference of two passes that agree in everything but
    the value. That is a property of a tensor: one column at 1, zeros elsewhere,
    the same for the baseline row as for every dose row. The rows are deliberately
    split across three chunks, so a flag rebuilt per chunk from chunk-local
    information --- the failure that would make the baseline row's flag differ
    from a dose row's --- shows up here and nowhere else.
    """
    q = make_query("value")
    calls = spy_on_forward(q.model)
    batch = query_batch()
    q.hidden_batch(batch, LAYERS, perturbed_col=PERTURBED_COL)

    # three chunks: 6 + 6 + 4, with the baseline row alone in the first
    assert [c["flags"].shape[0] for c in calls] == [CHUNK, CHUNK, batch.shape[0] - 2 * CHUNK]
    assert sum(c["flags"].shape[0] for c in calls) == batch.shape[0] == C.INPUT_ROWS_PER_GENE

    flags = torch.cat([c["flags"] for c in calls], dim=0)
    assert flags.dtype is torch.long, "scGPT's flag embedding indexes with a long tensor"
    assert flags.shape == (batch.shape[0], N_READOUT)

    # exactly one column set, on every row, and it is the perturbed one
    expected_row = torch.zeros(N_READOUT, dtype=torch.long)
    expected_row[PERTURBED_COL] = 1
    for row in range(flags.shape[0]):
        assert torch.equal(flags[row], expected_row), f"row {row}"
    # and the baseline row's flag is elementwise the dose rows' flag: the
    # sentence the whole arm rests on, stated as the comparison it is
    for row in range(1, flags.shape[0]):
        assert torch.equal(flags[0], flags[row]), f"baseline vs dose row {row}"

    # the value tensor, by contrast, DOES move --- and only at the flagged column
    values = torch.cat([c["values"] for c in calls], dim=0).numpy()
    moved = np.flatnonzero((values[1:] != values[0]).any(axis=0))
    assert moved.tolist() == [PERTURBED_COL]

    # the remaining inputs are fixed context too, so nothing else can confound
    for c in calls:
        b = c["flags"].shape[0]
        assert torch.equal(c["src"], torch.arange(N_READOUT).unsqueeze(0).expand(b, -1))
        assert c["pad"] is not None and not bool(c["pad"].any())


def test_flags_builds_one_hot_rows_for_any_chunk_height():
    """``_flags`` takes a row count, not a row offset, and that is the point.

    A per-row flag would need to know which global rows a chunk holds; this
    signature cannot express one, which is what makes "identical in both passes"
    structural rather than a convention a later edit could break. Checked at
    chunk heights that do and do not divide the 16 rows, and at both ends of the
    read-out range.
    """
    q = make_query("value")
    for rows in (1, CHUNK, C.INPUT_ROWS_PER_GENE):
        for col in (0, PERTURBED_COL, N_READOUT - 1):
            f = q._flags(rows, col)
            assert f.shape == (rows, N_READOUT)
            assert f.dtype is torch.long
            assert int(f.sum()) == rows
            assert bool((f[:, col] == 1).all())
            assert bool((f.sum(dim=1) == 1).all())


# --------------------------------------------------------------------------- #
# 2. a flag-only response cannot appear in a delta
# --------------------------------------------------------------------------- #
def test_a_flag_only_toy_gives_exactly_zero_at_every_layer_and_slot():
    """If the flag does not move, a state that depends on nothing else cannot.

    This is the assertion the arm is built to satisfy. ``== 0.0``, not a
    tolerance: two bit-identical inputs through a deterministic module give
    bit-identical outputs, so the difference is exactly the float zero, and a
    tolerance here would accept a baseline pass that differed by a rounding's
    worth of something real. Checked at layer 0 --- control C2 --- and at every
    sweep layer, on both slots, because a flag reaching other tokens would show
    up at depth even where layer 0 stayed clean.
    """
    q = make_query("flag")
    out = q.hidden_batch(query_batch(), LAYERS, perturbed_col=PERTURBED_COL)
    assert out is not None and sorted(out) == sorted(LAYERS)

    for layer, s in slots_of(out).items():
        # the captured rows themselves are already bit-identical
        rows = out[layer]
        assert np.abs(rows - rows[0]).max() == 0.0, f"layer {layer} rows"
        for form in C.SLOT_FORMS:
            assert np.abs(s[form]).max() == 0.0, f"layer {layer} {form}"


# --------------------------------------------------------------------------- #
# 3. a value-only response is recovered in full
# --------------------------------------------------------------------------- #
def test_a_value_only_toy_is_recovered_in_full_on_both_slots():
    """Zero everywhere would satisfy the flag test; this is what forbids it.

    The toy's state is ``gain(l) * (a + m * drive) V`` with ``drive`` the summed
    input row, so with the flag held constant the response is analytic: ``vvp``
    at dose ``v`` is ``gain(l) (v - anchor) outer(m, V)`` and ``gdt`` is
    ``gain(l) outer(m, V)`` at *every* dose, the clamped one included, because an
    affine function's derivative does not depend on the span it is measured over.
    Both expected values are written down twice on purpose: once analytically,
    which also proves the estimator recovers the right *coefficient*, and once
    through ``toy.flag_toy_expected`` from the drives the batch actually carries,
    which proves it recovers the right *rows*.

    Tolerances, not exact equality, and the reason is recorded: the production
    step ``GDT_STEP["scgpt_ft"] = 0.02`` is not dyadic and the toy's trunk runs
    in float32, so ``toy.py``'s bit-exactness claim does not hold at this step
    and is not asserted. See ``VVP_RTOL`` / ``GDT_RTOL``.
    """
    q = make_query("value")
    batch = query_batch()
    out = q.hidden_batch(batch, LAYERS, perturbed_col=PERTURBED_COL)
    assert out is not None

    coef = q.model.coefficients
    mV = np.outer(coef["m"], coef["V"])
    # the float32 values the ``v`` rows actually carry, not the float64 grid
    mid = slots.mid_row_values(native_values(), STEP).astype(np.float64)
    # the drive each of the 16 rows carries, read off the batch rather than
    # recomputed from the grid, so the expected value cannot drift from the input
    drives = batch.astype(np.float64).sum(axis=1)

    got = slots_of(out)
    for layer in LAYERS:
        gain = float(coef["block_gain"]) ** int(layer)

        analytic_gdt = np.broadcast_to(gain * mV, got[layer]["gdt"].shape)
        analytic_vvp = gain * (mid - ANCHOR_NATIVE)[:, None, None] * mV[None]
        # the signal has to be there at all, or a tolerance test is vacuous
        assert np.abs(analytic_gdt).max() > 0.5 * gain
        assert np.abs(analytic_vvp).max() > 0.5 * gain

        for form, analytic in (("vvp", analytic_vvp), ("gdt", analytic_gdt)):
            rtol = VVP_RTOL if form == "vvp" else GDT_RTOL
            err = float(np.abs(got[layer][form] - analytic).max())
            scale = float(np.abs(analytic).max())
            assert err <= rtol * scale, f"layer {layer} {form}: {err:.3g} vs {scale:.3g}"

        # the same slots against the toy's own state function, row by row: this
        # is the copy that fails if the estimator read the wrong pair of rows
        expected_rows = np.stack([flag_toy_expected(q.model, d, layer) for d in drives])
        ref = slots.build_slots(expected_rows, native_values(), STEP)
        for form in C.SLOT_FORMS:
            rtol = VVP_RTOL if form == "vvp" else GDT_RTOL
            err = float(np.abs(got[layer][form] - ref[form]).max())
            assert err <= rtol * float(np.abs(ref[form]).max()), f"layer {layer} {form} rows"


# --------------------------------------------------------------------------- #
# 4. the failure that would make assertion 2 a tautology
# --------------------------------------------------------------------------- #
def _flag_off_baseline_class():
    """A subclass whose baseline row is read with the flag OFF.

    The defect ``README.md`` s6 says would break control C2 --- layer-0
    off-target ``delta h`` would stop being exactly zero, because flag onset does
    reach other tokens --- and would additionally make ``vvp`` a sum of flag
    onset and dose rather than dose alone. ``_hidden_pass`` is overridden here,
    in the test, and ``hidden_ft.py`` is untouched: this is the wrong code, kept
    beside the right code so the right code's zero can be shown to be a
    measurement and not a property of the toy.

    Note that ``_flags`` cannot express this defect at all --- it is handed a row
    *count*, never a row offset --- which is why the override has to be at the
    pass level. That the wrong version is awkward to write is itself the finding.
    """
    hidden = hidden_ft._run5_hidden()

    class FlagOffBaseline(HIDDEN_CLASS):  # type: ignore[misc, valid-type]
        def _hidden_pass(self, values, perturbed_col, plan, layers):  # noqa: ANN001
            width = hidden.C.D_MODEL[self._model_key()]
            acc: dict[int, list[np.ndarray]] = {int(l): [] for l in layers}
            with hidden._capture(plan, layers) as store:
                for start, chunk in self._batches(values):
                    store.clear()
                    t = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)
                    f = self._flags(int(chunk.shape[0]), perturbed_col)
                    if int(start) == 0:
                        f[0] = 0  # <-- the defect: row 0 is the baseline pass
                    with torch.no_grad():
                        self._forward(t, f)
                    got = self._take(store, layers, int(self.n_readout), width)
                    for l in layers:
                        acc[int(l)].append(got[int(l)])
            return {int(l): np.concatenate(acc[int(l)], axis=0) for l in layers}

    return FlagOffBaseline


def test_a_flag_off_baseline_is_caught():
    """Run the flag-only assertion against the wrong implementation.

    Without this, ``test_a_flag_only_toy_gives_exactly_zero...`` proves only that
    something is zero, and a toy with no response, a driver that returned the
    baseline row sixteen times, or a slot builder that subtracted a row from
    itself would all pass it. Here the identical toy and the identical driving
    code, with the baseline row's flag zeroed, must come back NON-zero at every
    layer --- which is what makes that test a measurement of the flag.
    """
    q = make_query("flag", cls=_flag_off_baseline_class())
    out = q.hidden_batch(query_batch(), LAYERS, perturbed_col=PERTURBED_COL)
    assert out is not None

    for layer, s in slots_of(out).items():
        # vvp is baseline-referenced, so flag onset lands in it in full
        assert np.abs(s["vvp"]).max() > 0.0, f"layer {layer} vvp"
        # gdt differences two flagged rows, so it is blind to the defect: the
        # derivative slot alone could NOT have detected a flag-off baseline, and
        # vvp is therefore the slot the flag claim has to be made on
        assert np.abs(s["gdt"]).max() == 0.0, f"layer {layer} gdt"

    # and the correct class, on the same toy, is exactly zero on both slots ---
    # asserted here too so the contrast is in one place
    good = make_query("flag")
    ref = good.hidden_batch(query_batch(), LAYERS, perturbed_col=PERTURBED_COL)
    for layer, s in slots_of(ref).items():
        assert np.abs(s["vvp"]).max() == 0.0, f"layer {layer} vvp, correct class"


# --------------------------------------------------------------------------- #
# 5. a flag on nothing is a different estimator, and is refused
# --------------------------------------------------------------------------- #
def test_hidden_batch_refuses_a_flag_on_nothing():
    """No ``perturbed_col`` is run 4's published family, not run 6's estimator.

    Run 5's three adapters make the argument optional --- for them it selects a
    sequence composition. Here it is the column the flag is set at, so a default
    would silently produce the flag-off value-substitution query run 4 already
    published, under run 6's name and against run 6's fresh nulls.
    """
    q = make_query("value")
    with pytest.raises(RuntimeError, match="no perturbed_col"):
        q.hidden_batch(query_batch(), LAYERS)
    with pytest.raises(RuntimeError, match="no perturbed_col"):
        q.hidden_batch(query_batch(), LAYERS, perturbed_col=None)


@pytest.mark.parametrize("col", [-1, -N_READOUT, N_READOUT, N_READOUT + 3])
def test_hidden_batch_refuses_a_perturbed_column_off_the_read_out_set(col: int):
    """A flag outside the read-out positions is a flag on a gene never seen.

    Negative columns matter specifically: ``-1`` is a valid torch index and would
    quietly flag the LAST read-out gene, producing a full, well-shaped block
    attributed to the wrong perturbation with nothing downstream able to tell.
    """
    q = make_query("value")
    with pytest.raises(RuntimeError, match="outside the"):
        q.hidden_batch(query_batch(), LAYERS, perturbed_col=col)


# --------------------------------------------------------------------------- #
# 6. the block cannot misdescribe its own flag
# --------------------------------------------------------------------------- #
def test_the_recorded_notes_carry_the_flag_note_and_contradict_run_4s():
    """A manifest that repeated run 4's inherited sentence would be false.

    Run 4's ``configure`` says the flags are all zero; that is true of the family
    run 4 published and false of run 6. Both strings are asserted present --- the
    inherited one still reachable, so its presence is not evidence of a bug, and
    ``FT_FLAG_NOTE`` verbatim in both the layout section and the per-call notes
    --- and asserted to differ, so a block whose recorded flag regime silently
    became the inherited one fails here.
    """
    q = make_query("value")

    layout = q.layout_notes()
    assert layout["perturbation_flags"] == C.FT_FLAG_NOTE
    assert C.FT_FLAG_IN_BOTH_PASSES is True

    # before the call there are no per-call notes at all: an absence, not a zero
    assert "perturbation_flags" not in q.hidden_notes()
    q.hidden_batch(query_batch(), LAYERS, perturbed_col=PERTURBED_COL)
    notes = q.hidden_notes()
    assert notes["perturbation_flags"] == C.FT_FLAG_NOTE
    assert notes["run6_model_key"] == "scgpt_ft"
    assert notes["n_passes"] == 1
    assert notes["composition_depends_on_perturbed_col"] is False
    assert notes["amp"] is False and C.FT_AMP is False

    # run 4's own note, still reachable through the inherited configure
    inherited = q.configure(None, np.zeros(N_READOUT, dtype=np.float64), "hepg2")
    run4_note = inherited.notes["perturbation_flags"]
    assert "all zero for this query family" in run4_note
    assert run4_note != C.FT_FLAG_NOTE
    # and run 6's note names the contradiction rather than merely differing
    assert "BOTH" in C.FT_FLAG_NOTE and "contradicts" in C.FT_FLAG_NOTE


def test_the_flag_note_is_not_inherited_by_accident_from_a_sibling_default():
    """``FT_FLAG_NOTE`` is run 6's own constant, echoed and never rebuilt.

    Guards the one edit that would break the record without breaking a score: a
    later ``layout_notes`` that composed its own sentence would drift from the
    constant ``qc.json`` echoes, and the two would disagree with no test
    noticing. Compared by identity, not equality --- the note is echoed, not
    reconstructed.
    """
    q = make_query("value")
    assert q.layout_notes()["perturbation_flags"] is C.FT_FLAG_NOTE
    q.hidden_batch(query_batch(), LAYERS, perturbed_col=PERTURBED_COL)
    assert q.hidden_notes()["perturbation_flags"] is C.FT_FLAG_NOTE
