"""Reassembly of raw hidden states into read-out order, on synthetic layouts.

This is the step the plan names as the one that will silently corrupt delta-h.
Nothing here raises on its own: scGPT returns tile-ordered states with the same
gene present in several tiles and only one of them authoritative, TranscriptFormer
returns chunk-ordered states with the perturbed gene forced to position 0, and
scPRINT returns sequence-ordered states. Put a state on the wrong gene and the
block still has the right shape, the right dtype and a plausible norm --- and
every downstream number is about a gene it did not come from.

So the tests here are exact-recovery tests on layouts whose answer is written out
by hand: each read-out position carries a value encoding its own index, and the
reassembled matrix must equal the expected one at ``max abs diff 0``. And the two
ways a layout can be wrong without looking wrong --- a position no pass claims,
and a position two passes claim --- must raise, because a zero row and an
arbitrary winner are both indistinguishable from a measurement afterwards.

This is the one file in the suite that is not torch-free: the pure reassembly
functions live in ``models/reassembly.py``, deliberately apart from the adapter
subclasses that call them: ``hidden.py`` imports torch and run 2's model stack at
module scope, and these checks have to run on a machine with neither. Nothing
here builds a model.

Invocation (from the run directory):  PYTHONPATH=src pytest tests/test_reassembly.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scfm_probe.models import reassembly as H

#: a six-gene read-out set covered by two scGPT-shaped tiles: a shared
#: highest-expression prefix (genes 0, 1) plus a disjoint body, with the first
#: tile authoritative for the prefix
N_READOUT = 6
D = 3
BATCH = 2
TILES = [np.array([0, 1, 2, 3]), np.array([0, 1, 4, 5])]
OWNED = [np.array([0, 1, 2, 3]), np.array([2, 3])]


def _state(cols: np.ndarray, poison: dict[int, float] | None = None) -> np.ndarray:
    """``(BATCH, len(cols), D)`` where every row announces the gene it belongs to.

    The encoding ``1000*b + 10*gene + k`` is exact in float32 and unique per
    (batch, gene, width) cell, so a row that lands on the wrong gene cannot
    coincide with the right answer.
    """
    cols = np.asarray(cols)
    out = np.empty((BATCH, cols.size, D), dtype=np.float32)
    for b in range(BATCH):
        for p, gene in enumerate(cols):
            out[b, p] = 1000 * b + 10 * int(gene) + np.arange(D)
    for position, value in (poison or {}).items():
        out[:, position, :] = value
    return out


def _expected() -> np.ndarray:
    out = np.empty((BATCH, N_READOUT, D), dtype=np.float32)
    for b in range(BATCH):
        for gene in range(N_READOUT):
            out[b, gene] = 1000 * b + 10 * gene + np.arange(D)
    return out


# --------------------------------------------------------------------------- #
# tiles and chunks
# --------------------------------------------------------------------------- #


def test_a_tiled_layout_recovers_the_read_out_matrix_exactly() -> None:
    """Known layout, known answer, max abs diff 0 --- not "close"."""
    parts = [_state(c) for c in TILES]
    out = H.reassemble_tiles(parts, TILES, OWNED, N_READOUT)
    assert out.shape == (BATCH, N_READOUT, D)
    assert out.dtype == np.float32
    assert np.max(np.abs(out - _expected())) == 0.0


def test_a_gene_appended_to_a_tile_it_does_not_own_cannot_displace_the_owner() -> None:
    """scGPT appends the perturbed gene to any tile not already carrying it.

    That extra column is real input --- it is what makes the two passes' contexts
    match --- but it is not owned by the tile it was appended to, so its state
    must never reach the block. Here it is poisoned with a value that could not
    come from any gene, so a leak is unmissable.
    """
    cols = [TILES[0], np.array([0, 1, 4, 5, 3])]
    parts = [_state(cols[0]), _state(cols[1], poison={4: -12345.0})]
    out = H.reassemble_tiles(parts, cols, OWNED, N_READOUT)
    assert np.max(np.abs(out - _expected())) == 0.0


def test_a_read_out_position_no_pass_owns_is_refused() -> None:
    """A gene silently missing from the block would read as a row of zeros."""
    owned = [np.array([0, 1, 2]), OWNED[1]]  # nobody owns read-out gene 3
    with pytest.raises(RuntimeError):
        H.reassemble_tiles([_state(c) for c in TILES], TILES, owned, N_READOUT)


def test_a_read_out_position_two_passes_own_is_refused() -> None:
    """Two passes disagreeing about a gene: the survivor is whichever ran last."""
    owned = [OWNED[0], np.array([0, 2, 3])]  # pass 1 also claims read-out gene 0
    with pytest.raises(RuntimeError):
        H.reassemble_tiles([_state(c) for c in TILES], TILES, owned, N_READOUT)


def test_states_out_of_step_with_their_column_map_are_refused() -> None:
    """Every row of the pass would land on the wrong gene, quietly."""
    parts = [_state(TILES[0])[:, :3, :], _state(TILES[1])]
    with pytest.raises(RuntimeError):
        H.reassemble_tiles(parts, TILES, OWNED, N_READOUT)


def test_ownership_that_does_not_index_its_own_pass_is_refused() -> None:
    owned = [np.array([0, 1, 2, 9]), OWNED[1]]
    with pytest.raises(RuntimeError):
        H.reassemble_tiles([_state(c) for c in TILES], TILES, owned, N_READOUT)


def test_a_column_map_that_does_not_index_the_read_out_set_is_refused() -> None:
    cols = [np.array([0, 1, 2, 99]), TILES[1]]
    with pytest.raises(RuntimeError):
        H.reassemble_tiles([_state(c) for c in cols], cols, OWNED, N_READOUT)


def test_mismatched_map_counts_and_an_empty_block_are_refused() -> None:
    with pytest.raises(RuntimeError):
        H.reassemble_tiles([_state(TILES[0])], TILES, OWNED, N_READOUT)
    with pytest.raises(RuntimeError):
        H.reassemble_tiles([], [], [], N_READOUT)


def test_a_two_dimensional_capture_is_refused() -> None:
    """A pooled or squeezed capture has no gene axis left to reassemble."""
    with pytest.raises(RuntimeError):
        H.reassemble_tiles([_state(TILES[0])[0]], [TILES[0]], [OWNED[0]], N_READOUT)


# --------------------------------------------------------------------------- #
# sequence order
# --------------------------------------------------------------------------- #

#: read-out gene i lives at sequence slot READOUT_TO_SEQ[i], the same map run 2
#: unscatters scPRINT's predictions through
READOUT_TO_SEQ = np.array([3, 0, 4, 2, 1])


def test_a_permuted_sequence_recovers_the_read_out_matrix_exactly() -> None:
    n = READOUT_TO_SEQ.size
    seq = np.empty((BATCH, n, D), dtype=np.float32)
    for b in range(BATCH):
        for gene in range(n):
            seq[b, READOUT_TO_SEQ[gene]] = 1000 * b + 10 * gene + np.arange(D)
    out = H.reassemble_sequence(seq, READOUT_TO_SEQ, n)
    expected = np.stack(
        [np.stack([1000 * b + 10 * g + np.arange(D) for g in range(n)]) for b in range(BATCH)]
    ).astype(np.float32)
    assert np.max(np.abs(out - expected)) == 0.0


def test_the_identity_map_is_a_no_op() -> None:
    """A model already in read-out order must be passed through untouched."""
    n = READOUT_TO_SEQ.size
    seq = np.arange(BATCH * n * D, dtype=np.float32).reshape(BATCH, n, D)
    out = H.reassemble_sequence(seq, np.arange(n), n)
    assert np.max(np.abs(out - seq)) == 0.0


def test_a_sequence_map_that_is_not_a_permutation_is_refused() -> None:
    """One gene dropped and another duplicated, with nothing in the output to show it."""
    n = 5
    seq = np.zeros((BATCH, n, D), dtype=np.float32)
    with pytest.raises(RuntimeError):
        H.reassemble_sequence(seq, np.array([0, 0, 2, 3, 4]), n)


def test_a_sequence_of_the_wrong_length_is_refused() -> None:
    """A mis-sliced pad or cell-embedding prefix offsets every gene's state."""
    with pytest.raises(RuntimeError):
        H.reassemble_sequence(np.zeros((BATCH, 7, D), np.float32), READOUT_TO_SEQ, 5)
    with pytest.raises(RuntimeError):
        H.reassemble_sequence(np.zeros((BATCH, 5, D), np.float32), READOUT_TO_SEQ, 6)


# --------------------------------------------------------------------------- #
# the baseline and perturbed passes must be the same passes
# --------------------------------------------------------------------------- #


def test_identical_compositions_pass_and_a_changed_one_raises() -> None:
    """The trap the plan names first: delta-h across two different contexts.

    scGPT appends the perturbed gene to any tile not already holding it, so the
    baseline pass must be run with the perturbed pass's tile composition. If it
    is not, the difference mixes a value change with a change in which genes were
    supplied together, and no downstream statistic can separate them.
    """
    H.check_token_composition(TILES, [t.copy() for t in TILES], "baseline vs perturbed")
    with pytest.raises(RuntimeError):
        H.check_token_composition([TILES[0], np.array([0, 1, 4, 5, 3])], TILES, "tiles")
    with pytest.raises(RuntimeError):
        H.check_token_composition(TILES, [TILES[0]], "pass count")


def test_the_same_tokens_in_a_different_order_are_refused() -> None:
    """Composition must be a function of the perturbed gene alone, order included."""
    with pytest.raises(RuntimeError):
        H.check_token_composition([np.array([1, 0, 2, 3]), TILES[1]], TILES, "order")


def test_the_composition_signature_is_hashable_and_distinguishes_layouts() -> None:
    a = H.composition_signature(TILES)
    assert hash(a)
    assert a == H.composition_signature([t.copy() for t in TILES])
    assert a != H.composition_signature([np.array([1, 0, 2, 3]), TILES[1]])
