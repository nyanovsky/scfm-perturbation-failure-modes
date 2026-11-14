"""The four pure index operations that put a hidden state back in read-out order.

Split out of ``hidden.py`` for one reason: ``hidden.py`` imports torch and run 2's
model adapters at module scope, and these functions are where the silent
corruption would live. A read-out gene appears in several of scGPT's tiles and
several of TranscriptFormer's chunks with only one of them authoritative, and
scPRINT returns everything in vocabulary order rather than read-out order --- so
a reassembly that is off by one produces a perfectly well-shaped array of numbers
that belong to the wrong genes, and nothing downstream can notice.

That is exactly the failure a test has to catch on a machine with no GPU and no
checkpoint. Keeping these torch-free means the test suite can check them against
a known answer anywhere, which is why they live here and not beside the adapters
that call them.

Every one of them raises rather than repairing: a position written twice, a
position never written, a map that is not a permutation, or a token set that
changed between two passes are all conditions under which the delta being
computed is not the delta anyone asked for.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def composition_signature(cols: Sequence[np.ndarray]) -> tuple[tuple[int, ...], ...]:
    """A hashable, comparable fingerprint of what entered each pass.

    Kept as an ordered tuple rather than a set: for a bidirectional model the
    order is immaterial to the mathematics, but two passes whose orders differ
    were built by different code paths, and that is worth failing on rather than
    tolerating.
    """
    return tuple(tuple(int(c) for c in np.asarray(part).ravel()) for part in cols)


def check_token_composition(
    actual: Sequence[np.ndarray],
    expected: Sequence[np.ndarray],
    label: str,
) -> None:
    """Raise unless two passes were run over exactly the same tokens.

    The scientific reason this is an error and not a warning: the difference of
    two hidden states is only a response to the value change if everything else
    about the input was held fixed. A tile that gained or lost one gene between
    the baseline and the perturbed pass makes ``delta_h`` a sum of two effects
    with no way to separate them afterwards.
    """
    got, want = composition_signature(actual), composition_signature(expected)
    if got == want:
        return
    if len(got) != len(want):
        raise RuntimeError(
            f"{label}: the two passes used {len(got)} and {len(want)} sub-passes; a "
            "difference of hidden states taken across different pass structures is not "
            "a response to the value change"
        )
    for i, (a, b) in enumerate(zip(got, want)):
        if a == b:
            continue
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        raise RuntimeError(
            f"{label}: pass {i} was run over a different token set than its counterpart "
            f"({len(a)} vs {len(b)} tokens; only here: {only_a[:8]}; only there: "
            f"{only_b[:8]}). The delta would mix the value change with a change in the "
            "sequence composition, which is not what this run measures"
        )
    raise RuntimeError(
        f"{label}: the two passes carry the same tokens in a different order; the "
        "composition must be a function of the perturbed gene alone"
    )


def reassemble_tiles(
    parts: Sequence[np.ndarray],
    cols: Sequence[np.ndarray],
    owned: Sequence[np.ndarray],
    n_readout: int,
) -> np.ndarray:
    """Tile/chunk-ordered hidden states -> one ``(B, n_readout, d)`` in read-out order.

    ``parts[t]`` is ``(B, len(cols[t]), d)``, the states of pass ``t`` in the order
    its genes entered it; ``cols[t]`` are the read-out indices of those genes;
    ``owned[t]`` are the positions *within* pass ``t`` that pass is authoritative
    for. Ownership is the adapters' own rule (scGPT's ``tile_owned``,
    TranscriptFormer's "context genes keep the value from the first pass"), and it
    is reproduced rather than re-derived so a hidden state lands on exactly the
    read-out gene whose response run 2 recorded from the same pass.

    Every read-out position must be written exactly once. A position written zero
    times is a gene silently missing from the block; a position written twice
    means two passes disagree about who owns it and the survivor is whichever ran
    last. Both are raised on, because neither is visible in the output.
    """
    if not (len(parts) == len(cols) == len(owned)):
        raise RuntimeError(
            f"reassembly received {len(parts)} state arrays, {len(cols)} column maps and "
            f"{len(owned)} ownership maps; they describe the same passes and must agree"
        )
    if len(parts) == 0:
        raise RuntimeError("reassembly received no passes; a read-out block cannot be empty")

    first = np.asarray(parts[0])
    if first.ndim != 3:
        raise RuntimeError(
            f"a captured hidden state has shape {first.shape}; (batch, position, width) "
            "is the only shape a per-gene state can be reassembled from"
        )
    b, _, d = first.shape
    out = np.zeros((b, int(n_readout), d), dtype=np.float32)
    filled = np.zeros(int(n_readout), dtype=np.int64)

    for t, (part, col, own) in enumerate(zip(parts, cols, owned)):
        part = np.asarray(part)
        col = np.asarray(col, dtype=np.int64).ravel()
        own = np.asarray(own, dtype=np.int64).ravel()
        if part.ndim != 3 or part.shape[0] != b or part.shape[2] != d:
            raise RuntimeError(
                f"pass {t} has state shape {part.shape}, incompatible with pass 0's "
                f"(batch {b}, width {d}); the passes are not the same block"
            )
        if part.shape[1] != col.size:
            raise RuntimeError(
                f"pass {t} carries {part.shape[1]} positions but its column map names "
                f"{col.size}; the states and the gene list are out of step, so every row "
                "of this pass would land on the wrong gene"
            )
        if own.size and (own.min() < 0 or own.max() >= col.size):
            raise RuntimeError(
                f"pass {t} claims ownership of position {int(own.max())} of {col.size}; "
                "the ownership map does not index this pass"
            )
        target = col[own]
        if target.size and (target.min() < 0 or target.max() >= n_readout):
            raise RuntimeError(
                f"pass {t} owns read-out index {int(target.max())} of {n_readout}; the "
                "column map does not index the read-out set"
            )
        out[:, target, :] = part[:, own, :].astype(np.float32, copy=False)
        np.add.at(filled, target, 1)

    if not np.all(filled == 1):
        empty = int((filled == 0).sum())
        twice = int((filled > 1).sum())
        raise RuntimeError(
            f"reassembly left {empty} read-out positions unwritten and wrote {twice} more "
            "than once; the ownership maps do not partition the read-out set, so the "
            "block would carry zeros for the first and an arbitrary pass's state for the "
            f"second (first offenders: {np.flatnonzero(filled != 1)[:8].tolist()})"
        )
    return out


def reassemble_sequence(
    part: np.ndarray,
    readout_to_seq: np.ndarray,
    n_readout: int,
) -> np.ndarray:
    """Sequence-ordered hidden states -> read-out order, through the adapter's map.

    ``readout_to_seq[i]`` is the sequence slot holding read-out gene ``i``, which
    is exactly the map run 2 uses to unscatter scPRINT's predictions
    (``mean[:, self.readout_to_seq]``). Reusing it rather than re-sorting the
    positions is the point: a second sort that happened to break a tie
    differently would put gene A's state on gene B with nothing to show for it.
    """
    part = np.asarray(part)
    idx = np.asarray(readout_to_seq, dtype=np.int64).ravel()
    if part.ndim != 3:
        raise RuntimeError(
            f"a captured hidden state has shape {part.shape}; (batch, position, width) "
            "is the only shape a per-gene state can be reassembled from"
        )
    if idx.size != int(n_readout):
        raise RuntimeError(
            f"the sequence map names {idx.size} read-out genes but the block expects "
            f"{n_readout}; the adapter and the read-out table disagree"
        )
    if part.shape[1] != int(n_readout):
        raise RuntimeError(
            f"the captured sequence carries {part.shape[1]} gene positions but "
            f"{n_readout} read-out genes were supplied; the cell-embedding prefix was "
            "sliced wrongly and every gene's state is offset"
        )
    if not np.array_equal(np.sort(idx), np.arange(int(n_readout), dtype=np.int64)):
        raise RuntimeError(
            "readout_to_seq is not a permutation of the read-out positions, so it cannot "
            "be an order-preserving relabelling; some gene would be dropped and another "
            "duplicated"
        )
    return part[:, idx, :].astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Capture machinery
# --------------------------------------------------------------------------- #
