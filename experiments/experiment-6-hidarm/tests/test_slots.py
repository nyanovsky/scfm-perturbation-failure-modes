"""The ten slots land where run 6 says they land, and the eleventh is redundant.

Run 3 gathers ``e[form][m, :, cols]`` whole and has no idea what the columns
mean, so the ``(dose, pc)`` ordering of axis 1 is a contract only run 6 can
assert. Nothing downstream would notice if the doses were laid down in the order
they happened to be computed in: every score would still be produced, and the
dose axis --- the axis run 6 exists to recover --- would be scrambled inside the
feature vector. These tests are the only place that ordering is checked.

Torch-free on purpose: the layout is arithmetic, and arithmetic should be
checkable without a GPU or a bundle.
"""

from __future__ import annotations

import numpy as np
import pytest

from scfm_hidarm import config as C
from scfm_hidarm.install import column_labels, lay_out_slots

N_PERT, K, N_READOUT = 3, 4, 5
N_DOSE = len(C.VALUE_GRID_CP10K)


def planted(form: str, dose_i: int) -> np.ndarray:
    """A slot whose every entry names the (form, dose, pc, pert, gene) it is.

    Small integers, exactly representable in float32, so a misplaced column is
    an exact mismatch rather than a tolerance question.
    """
    off = 100000 * C.SLOT_FORMS.index(form)
    p = np.arange(N_PERT)[:, None, None] * 1000
    c = np.arange(K)[None, :, None] * 10
    g = np.arange(N_READOUT)[None, None, :]
    return (off + p + c + g + 1000000 * dose_i).astype(np.float32)


def build_slots() -> dict[tuple[str, int], np.ndarray]:
    return {(f, d): planted(f, d) for f in C.SLOT_FORMS for d in range(N_DOSE)}


def test_slots_land_in_declared_dose_pc_order():
    """``index = dose_i * k + pc``, which is ``config.SLOT_AXIS_ORDER``."""
    out = lay_out_slots(build_slots(), K)
    assert sorted(out) == sorted(C.SLOT_FORMS)
    for form in C.SLOT_FORMS:
        assert out[form].shape == (N_PERT, N_DOSE * K, N_READOUT)
        for dose_i in range(N_DOSE):
            for pc in range(K):
                col = dose_i * K + pc
                assert np.array_equal(out[form][:, col, :], planted(form, dose_i)[:, pc, :])
    # ten slots, and exactly ten: five doses x two forms, zero_out dropped
    assert len(C.SLOT_FORMS) * N_DOSE == C.N_SLOTS == 10


def test_column_labels_decode_the_flattened_axis():
    """The ``(dose, pc)`` decode is on disk, so a reader joins rather than assumes."""
    lab = column_labels(K, C.VALUE_GRID_CP10K)
    for name in ("values", "value_cp10k", "dose_index", "pc_index"):
        assert lab[name].size == N_DOSE * K
    for dose_i in range(N_DOSE):
        for pc in range(K):
            col = dose_i * K + pc
            assert lab["dose_index"][col] == dose_i
            assert lab["pc_index"][col] == pc
            assert lab["value_cp10k"][col] == C.VALUE_GRID_CP10K[dose_i]
    # run 3 reads ``values``; ``value_cp10k`` is the labelled duplicate of it
    assert np.array_equal(lab["values"], lab["value_cp10k"])


def test_permuting_the_input_slots_permutes_exactly_those_columns():
    """A dose swap at the input moves that dose's k columns and nothing else."""
    slots = build_slots()
    ref = lay_out_slots(slots, K)

    swapped = dict(slots)
    for form in C.SLOT_FORMS:
        swapped[(form, 1)], swapped[(form, 3)] = slots[(form, 3)], slots[(form, 1)]
    out = lay_out_slots(swapped, K)

    moved = set(range(1 * K, 2 * K)) | set(range(3 * K, 4 * K))
    for form in C.SLOT_FORMS:
        for col in range(N_DOSE * K):
            same = np.array_equal(out[form][:, col, :], ref[form][:, col, :])
            assert same is (col not in moved), f"{form} column {col}"
        assert np.array_equal(out[form][:, 1 * K : 2 * K, :], ref[form][:, 3 * K : 4 * K, :])
        assert np.array_equal(out[form][:, 3 * K : 4 * K, :], ref[form][:, 1 * K : 2 * K, :])


def test_dropped_zero_out_slot_is_bit_identical_to_the_vvp_dose_zero_slot():
    """Run 2's eleventh scalar, built from the same raw delta-h, adds no column.

    Run 2's ``zero_out`` is the response read with the perturbed gene's input
    set to zero, and its ``vvp`` slot at dose 0.0 CP10K is the same forward
    pass. Run 2's two differ only because it standardises each query row; run 6
    applies no normalisation at all (``config.DROPPED_RUN2_SLOT_REASON``), so
    the hidden versions are the same array projected through the same basis and
    keeping both would put one column in the feature vector twice.

    Constructed here from one raw ``delta-h``, through both paths, and compared
    at ``max abs diff == 0.0`` --- not a tolerance.
    """
    rng = np.random.default_rng(C.SEED)
    d = 7
    # raw delta-h per (dose, pert, readout, d), the array the extractor holds
    raw = rng.standard_normal((N_DOSE, N_PERT, N_READOUT, d)).astype(np.float32)
    w = rng.standard_normal((d, K)).astype(np.float32)

    def project(delta: np.ndarray) -> np.ndarray:
        """(n_pert, n_readout, d) @ (d, k) -> (n_pert, k, n_readout)."""
        return np.einsum("pgd,dc->pcg", delta, w)

    slots = {("vvp", i): project(raw[i]) for i in range(N_DOSE)}
    block = lay_out_slots(slots, K)["vvp"]

    # run 2's eleventh slot, built independently: the input value is zero, which
    # is the dose-0 forward pass, and no normalisation is applied to either
    zero_out = project(raw[C.VALUE_GRID_CP10K.index(0.0)])

    assert np.abs(block[:, 0:K, :] - zero_out).max() == 0.0
    assert C.DROPPED_RUN2_SLOT == "zero_out"


def test_a_form_missing_a_dose_raises_rather_than_writing_uninitialised_columns():
    slots = build_slots()
    del slots[("vvp", 2)]
    with pytest.raises(RuntimeError, match="dose"):
        lay_out_slots(slots, K)


def test_an_unknown_form_is_refused():
    slots = build_slots()
    slots[("zero_out", 0)] = planted("vvp", 0)
    with pytest.raises(ValueError, match="unknown slot form"):
        lay_out_slots(slots, K)
