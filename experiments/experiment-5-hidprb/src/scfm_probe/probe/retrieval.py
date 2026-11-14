"""S2, split-half retrieval, and S1, the propagation statistic.

Both answer questions about the same array, in an order that matters. S1 asks
whether the intervention left the perturbed gene's own token at all; S2 asks
whether what arrived elsewhere identifies which gene was intervened on. If S1 is
at machine epsilon, S2 has nothing to decode and its value is a statement about
the tie-break rule, not about the model --- which is why every retrieval result
here carries ``n_degenerate`` beside its score and refuses to be read alone.

Why this is not run 3's ``own_discrimination``
----------------------------------------------
Run 3's ``discrimination.py:20 own_discrimination`` is the same *question* --- is
a thing closer to its own perturbation than to the others --- and it was read
before this module was written. It is not called here, for three reasons, and the
metric names are kept distinct (``retrieval_top1`` / ``retrieval_top10`` /
``retrieval_mrr`` rather than ``discrimination``) so that no reader takes a number
from this run for a number from that one:

1. **Different distance.** ``own_discrimination`` is L1 (or L2) over effect
   profiles, where every coordinate is a gene and magnitude is the measurement.
   Here the objects are hidden-state directions in R^d whose scale is dominated
   by the perturbed gene's own expression level --- exactly what control C6
   exists to separate out --- so the distance has to be cosine, and swapping the
   distance inside that function would change what the run-3 number means.
2. **Different readout.** ``own_discrimination`` returns a normalised rank,
   ``1 - rank/(n-1)``, which is a graded score with no notion of a hit. The plan
   pre-registers top-1, top-10 and MRR against a stated chance level of
   ``1/n_perturbations``; a normalised rank cannot be read against that chance.
3. **It requires torch.** It runs ``torch.cdist`` on a device. This module is the
   estimator, and the estimator is checked against the toys on a machine with no
   GPU and no torch installed.

The normalised rank *is* still reported, as ``normalized_rank``, computed from the
same ranks, so a reader who wants run 3's readout of this array can have it on
run 3's own definition without either number pretending to be the other.

Ties, and the case that makes them load-bearing
-----------------------------------------------
On a layer where the off-target delta is exactly zero --- layer 0 by
construction, and any model that does not propagate --- every pooled vector is
the zero vector and every cosine distance is exactly 1.0. The ranking is then a
pure tie among all ``n`` candidates, and the tie rule alone decides the reported
number. Breaking by array index would report top-1 = 1.0 on a model that does
nothing (the diagonal wins every tie), which is precisely the failure this run
would then publish. Ties are therefore broken at random, at
:data:`config.RETRIEVAL_TIE_SEED`, which puts a fully degenerate block at its
chance level of ``1/n`` in expectation, and ``n_degenerate`` records how much of
the score was decided that way.

Torch-free by construction.
"""

from __future__ import annotations

import numpy as np

from .. import config as C


# --------------------------------------------------------------------------- #
# S2: split-half retrieval
# --------------------------------------------------------------------------- #
def cosine_distance(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """``(n_q, n_d)`` cosine distances, with the zero vector at distance 1.0.

    The zero-vector convention is a choice and not an accident: a pooled
    signature of exactly zero carries no direction, so it is defined to be
    equally far from everything rather than dropped. Dropping it would silently
    shrink ``n`` and move the chance level the score is read against.
    """
    q = np.asarray(query, dtype=np.float64)
    d = np.asarray(database, dtype=np.float64)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1]:
        raise RuntimeError(
            f"retrieval needs two (n, d) matrices of the same width; got {q.shape} "
            f"and {d.shape}, so the two halves were pooled at different layers"
        )
    qn = np.linalg.norm(q, axis=1)
    dn = np.linalg.norm(d, axis=1)
    qs = q / np.where(qn == 0.0, 1.0, qn)[:, None]
    ds = d / np.where(dn == 0.0, 1.0, dn)[:, None]
    return 1.0 - qs @ ds.T


def rank_of_own(
    dist: np.ndarray,
    seed: int = C.RETRIEVAL_TIE_SEED,
    atol: float = C.RETRIEVAL_TIE_ATOL,
) -> np.ndarray:
    """1-based rank of the diagonal entry within its row, ties broken at random.

    Row ``i`` of ``dist`` is the distance from perturbation ``i``'s half-A vector
    to every perturbation's half-B vector, so the true match is on the diagonal;
    that pairing is a contract asserted by :func:`retrieval_scores`, never
    assumed here.
    """
    dist = np.asarray(dist, dtype=np.float64)
    n = dist.shape[0]
    if dist.shape[0] != dist.shape[1]:
        raise RuntimeError(
            f"the retrieval distance matrix must be square for the diagonal to be the "
            f"true match; got {dist.shape}"
        )
    own = np.diagonal(dist)[:, None]
    closer = dist < own - atol
    tied = np.abs(dist - own) <= atol
    np.fill_diagonal(tied, False)
    rng = np.random.default_rng(seed)
    jitter = rng.random((n, n))
    own_jitter = np.diagonal(jitter)[:, None]
    wins_tie = tied & (jitter < own_jitter)
    return 1 + closer.sum(axis=1) + wins_tie.sum(axis=1)


def retrieval_scores(
    query: np.ndarray,
    database: np.ndarray,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    topk: tuple[int, ...] = C.TOPK,
    seed: int = C.RETRIEVAL_TIE_SEED,
    atol: float = C.RETRIEVAL_TIE_ATOL,
) -> dict:
    """Score half A against half B: top-k, MRR, chance, and how degenerate it was.

    ``labels`` is the perturbed gene id of each row, in both matrices. It is
    checked for equality rather than trusted, because row order is the only thing
    that makes the diagonal the true match and a silent misalignment here would
    report chance on a model that works.

    ``groups`` supplies control C7: a hit is also counted when the retrieved
    perturbation shares a response group with the true one. It is returned beside
    the strict number, never instead of it.
    """
    q = np.asarray(query, dtype=np.float64)
    d = np.asarray(database, dtype=np.float64)
    if labels is not None:
        lab = np.asarray(labels).astype(str)
        if lab.size != q.shape[0] or lab.size != d.shape[0]:
            raise RuntimeError(
                f"{lab.size} perturbation labels for {q.shape[0]} query rows and "
                f"{d.shape[0]} database rows; the diagonal is not the true match"
            )
    n = q.shape[0]
    if n < 2:
        raise RuntimeError(
            f"retrieval over {n} perturbation(s) has no chance level to be read against; "
            "a block this small is a smoke test, not a measurement"
        )
    dist = cosine_distance(q, d)
    ranks = rank_of_own(dist, seed=seed, atol=atol)
    n_degenerate = int(
        ((np.linalg.norm(q, axis=1) == 0.0) | (np.linalg.norm(d, axis=1) == 0.0)).sum()
    )
    out: dict = {
        "n_perturbations": int(n),
        "chance_top1": 1.0 / n,
        "mrr": float(np.mean(1.0 / ranks)),
        "chance_mrr": float(np.mean(1.0 / np.arange(1, n + 1))),
        "median_rank": float(np.median(ranks)),
        # run 3's own readout of the same ranks, on run 3's definition, so the two
        # runs' numbers can be compared without either metric being renamed
        "normalized_rank": float(np.mean(1.0 - (ranks - 1) / (n - 1))),
        "n_degenerate": n_degenerate,
        "degenerate": bool(n_degenerate == n),
        "ranks": ranks,
    }
    for k in topk:
        out[f"top{int(k)}"] = float(np.mean(ranks <= int(k)))
        out[f"chance_top{int(k)}"] = min(1.0, float(int(k)) / n)
    if groups is not None:
        g = np.asarray(groups)
        if g.size != n:
            raise RuntimeError(
                f"{g.size} response-group labels for {n} perturbations; the "
                "group-collapsed control would be counting the wrong rows"
            )
        # Ties are broken by the SAME seeded jitter `rank_of_own` uses, never by
        # array index. A stable argsort would put row 0 first in every tied row, so
        # on a dead layer --- where every cosine distance is exactly 1.0 --- this
        # control would report "what fraction of perturbations share row 0's
        # response group", a number that swings with the arbitrary order of the
        # perturbation table while the strict score correctly reads chance. That is
        # the exact failure the strict path was built to avoid, and it reached this
        # one because the two were written apart.
        rng = np.random.default_rng(seed)
        keys = dist + rng.random(dist.shape) * max(atol, np.finfo(np.float64).eps)
        order = np.argsort(keys, axis=1, kind="stable")
        for k in topk:
            kk = min(int(k), n)
            hit = np.array(
                [bool(np.isin(g[order[i, :kk]], g[i]).any()) for i in range(n)]
            )
            out[f"group_collapsed_top{int(k)}"] = float(hit.mean())
    return out


# --------------------------------------------------------------------------- #
# S1: propagation
# --------------------------------------------------------------------------- #
def propagation_ratio(norm: np.ndarray, self_cols: np.ndarray) -> dict:
    """Per (perturbation, value): how much of the edit left the perturbed token.

    ``norm`` is the ``(n_perturbed, n_values, n_readout)`` array written to
    ``norms.npz``, with the perturbed gene's own position included; ``self_cols``
    is each perturbation's read-out column. The numerator is the Frobenius norm
    over the off-target positions --- the total length of what arrived elsewhere
    --- and the denominator is the length at the perturbed gene's own position,
    so the ratio is dimensionless and comparable across layers of different width.

    A perturbation whose own position did not move has an undefined ratio, not a
    ratio of zero over zero: that is a dead intervention (control C1 failing) and
    is counted and excluded rather than averaged in as a small number.
    """
    norm = np.asarray(norm, dtype=np.float64)
    if norm.ndim != 3:
        raise RuntimeError(
            f"the per-position norm array must be (n_perturbed, n_values, n_readout); "
            f"got shape {norm.shape}"
        )
    cols = np.asarray(self_cols, dtype=np.int64)
    n_p, n_v, n_r = norm.shape
    if cols.size != n_p:
        raise RuntimeError(
            f"{cols.size} perturbed read-out columns for {n_p} perturbations; the "
            "self position would be taken from the wrong gene"
        )
    if cols.min() < 0 or cols.max() >= n_r:
        raise RuntimeError(
            f"a perturbed read-out column ({cols.min()}..{cols.max()}) falls outside the "
            f"{n_r} read-out positions of this block"
        )
    # The self entry is `nan` for a model that never reads a gene's count at its
    # own position (TranscriptFormer). Summing it would poison `off_target_norm`
    # --- the one key three separate records name as that model's substitute for
    # the ratio --- so the self position is masked out of the total rather than
    # subtracted from it. Subtracting cannot work: nan - nan is nan.
    self_norm = norm[np.arange(n_p), :, cols]
    masked = norm.copy()
    masked[np.arange(n_p), :, cols] = 0.0
    if not np.isfinite(masked).all():
        raise RuntimeError(
            "a non-self read-out position carries a non-finite norm; only the "
            "perturbed gene's own position may be undefined, so this is a broken "
            "block rather than an architectural absence"
        )
    off_sq = (masked**2).sum(axis=2)
    off_norm = np.sqrt(off_sq)
    defined = self_norm > 0.0
    ratio = np.full((n_p, n_v), np.nan, dtype=np.float64)
    ratio[defined] = off_norm[defined] / self_norm[defined]
    off_rms = off_norm / np.sqrt(max(n_r - 1, 1))
    masked = off_target_norms(norm, cols)
    return {
        "ratio": ratio,
        "off_target_norm": off_norm,
        "off_target_rms": off_rms,
        "self_norm": self_norm,
        "n_defined": int(defined.sum()),
        "n_undefined": int((~defined).sum()),
        # the largest single off-target position norm anywhere in the block; this
        # is the number control C2 reads at layer 0, where it must be exactly 0.0
        "max_off_target_position_norm": float(np.nanmax(masked)),
        "median_ratio": float(np.nanmedian(ratio)) if defined.any() else float("nan"),
    }


def off_target_norms(norm: np.ndarray, self_cols: np.ndarray) -> np.ndarray:
    """``(n_perturbed, n_values, n_readout)`` per-position norms with the self
    position set to nan, so ``max`` and ``median`` over it cannot pick up C1.

    Control C2 reads this: at layer 0 the maximum over the returned array must be
    exactly 0.0, and a nan at the self position is what stops the trivially large
    self entry from masking that.
    """
    norm = np.asarray(norm, dtype=np.float64).copy()
    cols = np.asarray(self_cols, dtype=np.int64)
    norm[np.arange(norm.shape[0]), :, cols] = np.nan
    return norm
