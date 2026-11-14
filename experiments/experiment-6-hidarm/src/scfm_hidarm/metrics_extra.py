"""The two scoring families run 6 adds beside run 3's weighted Pearson.

Run 3 scores every arm with one estimator: a row-wise weighted Pearson
correlation between the predicted and the measured response, read in six
(target x weighting) cells and disattenuated by a split-half ceiling. That
estimator has two blind spots this module exists to cover, and *only* those two
--- nothing here refits anything, so no number produced here can change a fit.

**Pearson removes a per-row weighted mean.** For a response profile that mean is
not nuisance: it is the shared, offset part of what a knockdown does to the
transcriptome. An arm that gets the offset right and the shape wrong, or the
shape right and the offset wrong, is scored identically by Pearson.
``weighted_cosine_rows`` is the same cell with the mean removal deleted --- same
weights, same zeroed self column, same NaN discipline --- so the pair of numbers
separates "predicts the shape" from "predicts the whole vector".

**Pearson is a per-row statistic and asks nothing about identifiability.** A
prediction can correlate respectably with its own perturbation's profile and
still be closer to two hundred other perturbations' profiles. ``pds_l1`` is the
perturbation-discrimination score: rank a prediction against every measured
profile in the pool by Manhattan distance and report ``1 - own_rank / N``. It is
run 3's ``discrimination.own_discrimination`` definition, at the denominator
``config.PDS_DENOMINATOR`` names, with the perturbed gene's own read-out column
dropped from both members of every pair.

Both families are declared in ``config.py`` (``EXTRA_METRICS``,
``EXTRA_METRIC_TARGETS``, ``EXTRA_METRIC_WEIGHTINGS``, ``PDS_*``,
``DISATTENUATE``) *before* any arm was scored. They are pre-registered
secondaries, not metrics picked for their answer, and this module adds no
constant of its own.

**Nothing is copied out of run 3.** ``metrics`` below is run 3's actual module,
imported through ``sibling.run3``. Run 6's whole claim is that it scores a new
input through an unchanged rule, and a copied estimator that drifted by one line
would break that silently. The consequence is the one run 4 already carries: an
edit to run 3's ``metrics.py`` reaches this module.

Rejected alternatives, recorded because both are tempting:

- *Disattenuating the discrimination score.* Spearman-Brown and the
  ``r / sqrt(reliability)`` correction are theorems about a correlation under
  the classical measurement model. There is no such theorem for a rank
  statistic, so ``pds_ceiling`` is returned **uncorrected**, its
  ``spearman_brown_applied`` flag is ``False``, and the score is reported as a
  ratio to that ceiling rather than divided by its square root. This mirrors
  ``config.DISATTENUATE``.
- *A second grouped bootstrap here.* ``score_extra`` uses run 3's own
  ``bootstrap_median`` and ``bootstrap_median_grouped`` --- whichever the caller
  asks for by passing ``groups`` --- and never a grouping of its own. The
  interval that survives into a claim is the GROUP bootstrap over response
  groups, because perturbations are not independent draws; ``summary["bootstrap"]``
  records which one ran so a row cannot be read without knowing.

Two further things this module is careful about, both because a call site now
needs them:

**The discrimination score is reported at BOTH pools of ``config.PDS_POOLS``,**
never at one. ``pooled`` ranks every scored perturbation against every other ---
"can this arm pick its perturbation out of the screen" --- and is the pool run
3's ``discrimination_ours_residual_*`` column uses, so it is the only one of the
two that can be set beside a published run-3 number. ``per_fold`` ranks a
held-out perturbation against the ones it was held out with, a smaller and
easier pool, and is the reading whose pool matches the fold structure the
ceilings are aggregated over. Each carries its own ceiling at its own pool.
Reporting one would have meant either an incomparable number or an
incommensurable ceiling.

**A restricted reading recomputes rather than filters.** Run 4's fine-tuned
checkpoints trained on three of five folds, so their arms are read on folds
(0, 1) only. ``rows_subset`` is how; and while a row-wise similarity restricts
by filtering, ``1 - own_rank / N`` does not: restricting the pool changes every
rank inside it, so subsetting the 524-way numbers would report them as a
~210-way ranking. The discrimination cells and their ceilings are recomputed on
the subset, run 4's ``restricted_ceilings`` pattern, and ``score_extra`` refuses
a ceiling computed on a different number of rows rather than dividing by it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .sibling import run3

#: run 3's estimators, imported and never copied --- see the module docstring
metrics = run3("metrics")


def _bootstrap_params() -> tuple[int, float, int]:
    """Run 3's bootstrap settings, read from run 3 rather than re-declared.

    The per-run interval on a run-6 cell has to be the interval run 3 would have
    put on it, so the resample count, the coverage and the seed are read out of
    run 3's ``config`` rather than restated in run 6's. The seeds are asserted
    equal because two different seeds would make a run-6 interval and a run-3
    interval on the same values differ for no scientific reason at all.
    """
    c3 = run3("config")
    if int(c3.SEED) != int(C.SEED):
        raise RuntimeError(
            f"run 3's SEED ({c3.SEED}) and run 6's SEED ({C.SEED}) differ; the bootstrap "
            "interval on a run-6 cell would then not be the interval run 3 puts on the "
            "same values, and the two would be incomparable for a reason nothing records"
        )
    return int(c3.BOOTSTRAP_RESAMPLES), float(c3.BOOTSTRAP_CI), int(c3.SEED)


# --------------------------------------------------------------------------- #
# (a) weighted cosine
# --------------------------------------------------------------------------- #
def weighted_cosine_rows(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise weighted cosine similarity between two response profiles.

    ``sum_j w_ij x_ij y_ij / sqrt(sum_j w_ij x_ij^2 * sum_j w_ij y_ij^2)``.

    This is ``metrics.weighted_pearson_rows`` with the per-row weighted-mean
    removal deleted, and that deletion is the whole scientific point: the mean
    of a knockdown's response profile is the shared, offset part of the
    response, and a mean-centred correlation throws it away. Reading the two
    side by side separates an arm that predicts the shape of a response from one
    that predicts the response.

    Handling mirrors ``weighted_pearson_rows`` exactly, because a cosine cell and
    a Pearson cell must differ in the centring and in nothing else: ``w`` is
    ``(n_cols,)`` or ``(n_rows, n_cols)``; a weight is zeroed wherever either
    input is non-finite; a row whose weighted norm vanishes returns NaN rather
    than 0, since a degenerate row is a row with no answer and not a row that
    scored zero; everything is float64 and the guard is run 3's own ``EPS``.

    Zeroing the weight where either side is non-finite is what makes a
    TranscriptFormer arm scoreable at all. Its ``(p, p)`` hidden pair is
    undefined by construction, run 5's adapter writes ``nan`` there rather than
    inventing a zero, and run 3's translator predicts every column including the
    perturbed gene's own --- so that ``nan`` reaches ``pred[p, self_cols[p]]``.
    Every existing cell of the grid already drops that column; this one drops it
    explicitly and by the same rule, not by luck.
    """
    eps = metrics.EPS
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 1:
        w = np.broadcast_to(w, x.shape)
    finite = np.isfinite(x) & np.isfinite(y)
    w = np.where(finite, w, 0.0)
    x = np.where(finite, x, 0.0)
    y = np.where(finite, y, 0.0)
    sw = w.sum(axis=1)
    ok = sw > eps
    num = (w * x * y).sum(axis=1)
    nx = (w * x * x).sum(axis=1)
    ny = (w * y * y).sum(axis=1)
    denom = np.sqrt(nx * ny)
    out = np.full(x.shape[0], np.nan)
    good = ok & (denom > eps)
    out[good] = num[good] / denom[good]
    return out


# --------------------------------------------------------------------------- #
# (b) L1 perturbation discrimination
# --------------------------------------------------------------------------- #
def _sanitise(
    pred: np.ndarray, true: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Replace non-finite entries with 0.0, count them, and keep their positions.

    The positions are kept so the caller can report how many of them fell
    OUTSIDE the columns the self-column correction removes. Those are the only
    ones where the substituted 0.0 is actually summed into a distance, and a
    count of zero is the thing a check has to print rather than assume.

    The zero is not a value being scored --- it is a placeholder the self-column
    correction then subtracts back out exactly. ``full[p, q]`` and both
    correction terms are computed on the SAME sanitised arrays, so the
    contribution of an excluded column to ``d[p, q]`` is algebraically zero
    rather than approximately zero. Doing it the other way round --- correcting a
    ``full`` that already contains NaN --- cannot work at all, because
    ``NaN - NaN`` is NaN and every rank comparison against NaN is False, which
    would return a broken input looking like a perfect score.
    """
    bad_p = ~np.isfinite(pred)
    bad_t = ~np.isfinite(true)
    n_bad = int(bad_p.sum() + bad_t.sum())
    if n_bad == 0:
        return pred, true, 0, bad_p, bad_t
    return np.where(bad_p, 0.0, pred), np.where(bad_t, 0.0, true), n_bad, bad_p, bad_t


def _full_l1_numpy(pred: np.ndarray, true: np.ndarray, chunk_rows: int) -> np.ndarray:
    """All-pairs L1 distances in float64, chunked over rows.

    The scored pool is up to 705 perturbations over 3247 read-out genes, so the
    ``(n, n, g)`` broadcast the one-liner implies is ~13 GB. Chunking the
    prediction rows keeps the transient at ``chunk_rows`` planes and changes no
    arithmetic: the sum over genes is exact either way.
    """
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    n, m = pred.shape[0], true.shape[0]
    out = np.empty((n, m), dtype=np.float64)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        out[start:stop] = np.abs(pred[start:stop, None, :] - true[None, :, :]).sum(axis=2)
    return out


def _full_l1_torch(pred: np.ndarray, true: np.ndarray, device: str) -> np.ndarray:
    """The same matrix through ``torch.cdist``, for the large real pools.

    The numpy path above is the REFERENCE --- ``tests/test_metrics_extra.py``
    asserts the two agree --- and this one exists only because cdist on a GPU
    turns a minute into a second. float64 on both sides: run 3's
    ``own_discrimination`` casts to float32, which is fine for a rank but makes
    a bit-level comparison against the reference meaningless.
    """
    import torch

    dev = device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = torch.as_tensor(pred, dtype=torch.float64, device=dev)
    r = torch.as_tensor(true, dtype=torch.float64, device=dev)
    return torch.cdist(p, r, p=1.0).cpu().numpy().astype(np.float64)


def pds_l1(
    pred: np.ndarray,
    true: np.ndarray,
    self_cols: np.ndarray | None = None,
    denominator: str = "n",
    chunk_rows: int = 128,
    device: str | None = None,
) -> dict:
    """L1 perturbation discrimination: is a prediction closest to its own truth?

    For perturbation ``p`` the distance to every measured profile ``q`` is
    ``d[p, q] = sum_g |pred[p, g] - true[q, g]|``; ``own_rank[p]`` counts the
    ``q`` with ``d[p, q]`` **strictly less** than ``d[p, p]``; the score is
    ``1 - own_rank[p] / N``. 1.0 is perfect.

    ``denominator``: ``"n"`` uses ``N = pred.shape[0]``, which is
    ``config.PDS_DENOMINATOR`` and the run-6 cell; ``"n_minus_1"`` uses ``N - 1``,
    which is run 3's ``discrimination.own_discrimination`` convention. Both are
    reported, because the two differ by ``own_rank / (n (n-1))`` --- 0.2% at
    n = 524 --- and silently substituting one for the other is how a number stops
    being traceable to a definition.

    **Ties count as ties.** Only strictly-smaller distances are counted, so an
    exact tie does not inflate a rank. The consequence has to be stated because
    it is the failure mode this metric invites: a CONSTANT prediction, identical
    for every perturbation, has ``d[p, q]`` depending on ``q`` alone, so its
    ranks are a permutation of ``0 .. N-1`` and it scores a mean of
    ``1 - (N-1)/(2N) ~ 0.5``, not 0. A discrimination score is therefore read
    against fresh shuffled-label draws in its own cell, never against zero.

    **Self-column exclusion** (``config.PDS_EXCLUDE_SELF_COLUMNS``). When
    ``self_cols`` is given --- one read-out column index per row, the perturbed
    gene's own column --- the distance for the pair ``(p, q)`` excludes columns
    ``{self_cols[p], self_cols[q]}``. That column carries weight zero in every
    other cell of run 3's grid because it is the knockdown itself: it is large,
    it is negative, it is the same sign for every perturbation, and no arm is
    asked to predict it. Left in, the discrimination score would ride on it.

    The exclusion is a rank-1 correction on the full matrix rather than an
    ``O(N^2 G)`` masked loop::

        d[p,q] = full[p,q] - |pred[p,s_p] - true[q,s_p]| - |pred[p,s_q] - true[q,s_q]|

    plus the one term added back when ``s_p == s_q``, since it was then
    subtracted twice. Both corrections are gathers of ``N x N`` scalars.

    **Non-finite entries.** TranscriptFormer's ``(p, p)`` hidden pair is
    undefined by construction, run 5's adapter writes ``nan`` there, and run 3's
    translator predicts every column, so ``pred[p, self_cols[p]]`` can be
    ``nan``. Those entries are set to 0.0 in BOTH arrays before ``full`` is
    built, counted, and reported as ``n_nonfinite_sanitised``. The zero is never
    scored where it sits in an excluded column: because the corrections are
    computed on the same sanitised arrays, they subtract exactly what ``full``
    gained at those columns, so an excluded column contributes algebraically
    zero. Sanitising first and correcting after is the only order that works ---
    a ``full`` containing NaN cannot be corrected, since ``NaN - NaN`` is NaN.
    ``n_nonfinite_outside_self_columns`` counts the sanitised entries that do
    not sit in their own row's self column, i.e. the ones the correction does
    not remove from every pair and whose substituted 0.0 therefore reaches at
    least one distance. The design expects it to be zero, and a check should
    print the number rather than assume it.

    With ``self_cols=None`` and non-finite entries present this RAISES. A zero
    substituted for an undefined value and then actually summed into a distance
    is a fabricated number scoring an arm, which is precisely the silent failure
    this project's claims discipline exists to prevent.

    ``device``: ``None`` runs the chunked numpy reference. A string
    (``"cpu"``, ``"cuda"``, ``"auto"``) runs ``torch.cdist`` instead. There is
    deliberately no automatic size threshold --- a threshold would be a magic
    number, and this run's constants live in ``config.py`` --- so the caller
    decides and the choice is recorded in the caller's manifest.
    """
    if denominator not in ("n", "n_minus_1"):
        raise ValueError(f"denominator must be 'n' or 'n_minus_1', got {denominator!r}")
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(
            f"pred {pred.shape} and true {true.shape} must be the same pool of "
            "perturbations over the same read-out genes; own_rank is undefined otherwise"
        )
    if self_cols is None and (not np.isfinite(pred).all() or not np.isfinite(true).all()):
        raise RuntimeError(
            "pds_l1 was given non-finite values and no self_cols. The only non-finite "
            "entries this design expects are TranscriptFormer's undefined (p, p) hidden "
            "pair, which the self-column exclusion removes exactly; without the exclusion "
            "the substituted 0.0 would be summed into the distance and an arm would be "
            "scored on a fabricated value with nothing in the record saying so"
        )
    pred, true, n_bad, bad_pred, bad_true = _sanitise(pred, true)
    n = int(pred.shape[0])
    n_off = n_bad
    note = "no self-column exclusion applied"
    if n < 3:
        # Fewer than three profiles is not a ranking; run 3's own_discrimination
        # returns NaN here and so does this, rather than a score of 1.0.
        return {
            "per_perturbation": np.full(n, np.nan),
            "mean": float("nan"),
            "median": float("nan"),
            "n": n,
            "denominator": denominator,
            "n_excluded_columns_note": note,
            "n_nonfinite_sanitised": n_bad,
            "n_nonfinite_outside_self_columns": n_off,
            "sanitised_note": C.PDS_SANITISED_NOTE,
        }

    if device is None:
        d = _full_l1_numpy(pred, true, chunk_rows)
    else:
        d = _full_l1_torch(pred, true, device)

    if self_cols is not None:
        s = np.asarray(self_cols, dtype=np.int64)
        if s.shape != (n,):
            raise ValueError(f"self_cols must have shape ({n},), got {s.shape}")
        rows = np.arange(n)
        # a[p, q] = |pred[p, s_p] - true[q, s_p]|  (this row's own column)
        a = np.abs(pred[rows, s][None, :] - true[:, s]).T
        # b[p, q] = |pred[p, s_q] - true[q, s_q]|  (the other row's own column)
        b = np.abs(pred[:, s] - true[rows, s][None, :])
        same = s[:, None] == s[None, :]
        d = d - a - b + np.where(same, a, 0.0)
        # a sanitised entry away from its own row's self column is a 0.0 that
        # really is summed into at least one distance; counted here so the
        # caller can print the number instead of assuming it is zero
        n_off = n_bad - int(bad_pred[rows, s].sum() + bad_true[rows, s].sum())
        note = (
            f"columns {{self_cols[p], self_cols[q]}} excluded from every pair distance; "
            f"{int(np.unique(s).size)} distinct self columns over {n} rows"
        )

    own = np.diag(d)
    rank = (d < own[:, None]).sum(axis=1).astype(np.float64)
    denom = float(n) if denominator == "n" else float(n - 1)
    scores = 1.0 - rank / denom
    return {
        "per_perturbation": scores,
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "n": n,
        "denominator": denominator,
        "n_excluded_columns_note": note,
        "n_nonfinite_sanitised": n_bad,
        "n_nonfinite_outside_self_columns": n_off,
        "sanitised_note": C.PDS_SANITISED_NOTE,
    }


# --------------------------------------------------------------------------- #
# (c) the ceilings: what the measurement itself reproduces
# --------------------------------------------------------------------------- #
def _subset(rows_subset: np.ndarray | None, n_rows: int) -> np.ndarray:
    """The scope rows a reading is taken on, validated once.

    ``None`` means every row. Anything else is sorted and deduplicated so that
    a pool never depends on the order the caller happened to pass, and checked
    against the scope size, because an out-of-range index would silently rank a
    perturbation against the wrong profile rather than raise.
    """
    if rows_subset is None:
        return np.arange(n_rows)
    sub = np.unique(np.asarray(rows_subset, dtype=np.int64))
    if sub.size and (sub[0] < 0 or sub[-1] >= n_rows):
        raise RuntimeError(
            f"rows_subset holds indices outside [0, {n_rows}); they are scope-row indices, "
            "not perturbation ids, and an out-of-range one would score against another row"
        )
    return sub


def _half_pair(
    half0: np.ndarray, half1: np.ndarray, tr: np.ndarray, te: np.ndarray, target: str
) -> tuple[np.ndarray, np.ndarray]:
    """The two half-effect matrices on one fold's held-out rows, per target.

    Built exactly as ``runner.fold_reference`` builds its four Pearson ceilings:
    ``raw`` is the half effect itself, ``residual`` is each half residualized
    against ITS OWN training-row mean (so the number is the reproducibility of
    the perturbation-specific part alone), and ``centered`` additionally removes
    the held-out fold's own per-gene mean from both halves. Residualizing per
    half rather than once on the pooled effect is what keeps the ceiling an
    estimate of the same quantity the score is an estimate of.
    """
    if target == "raw":
        return half0[te], half1[te]
    r0, _ = metrics.residualize(half0, tr)
    r1, _ = metrics.residualize(half1, tr)
    if target == "residual":
        return r0[te], r1[te]
    if target == "centered":
        return (
            r0[te] - r0[te].mean(axis=0, keepdims=True),
            r1[te] - r1[te].mean(axis=0, keepdims=True),
        )
    raise ValueError(f"unknown target {target!r}; expected one of {C.EXTRA_METRIC_TARGETS}")


def _assemble_halves(
    half0: np.ndarray, half1: np.ndarray, folds: list[dict], target: str
) -> tuple[np.ndarray, np.ndarray]:
    """``_half_pair`` for every fold, written back into two whole-scope matrices.

    The POOLED discrimination ceiling ranks a row against rows from other folds,
    so it needs one matrix spanning the screen --- but each row must still carry
    the residual ITS OWN fold defines, against ITS OWN training rows. Building
    one residual from a single pooled mean instead would leak every held-out
    perturbation's own outcome into the mean it is measured against, and would
    also stop matching what ``score_extra`` scores.
    """
    a = np.zeros_like(half0, dtype=np.float64)
    b = np.zeros_like(half1, dtype=np.float64)
    for info in folds:
        tr, te = np.asarray(info["train"]), np.asarray(info["test"])
        if te.size == 0:
            continue
        a[te], b[te] = _half_pair(half0, half1, tr, te, target)
    return a, b


def _fold_weights(
    weights: np.ndarray, self_cols_te: np.ndarray, n_te: int, n_cols: int, weighting: str
) -> np.ndarray:
    """One fold's weight matrix, self column zeroed --- ``fold_reference``'s.

    ``unweighted`` is not "no weights": it is a matrix of ones with the same
    self column zeroed, because the knockdown column is excluded from every cell
    of the grid regardless of whether reliability weighting is on.
    """
    if weighting == "weighted":
        w = np.broadcast_to(np.asarray(weights, dtype=np.float64), (n_te, n_cols)).copy()
    elif weighting == "unweighted":
        w = np.ones((n_te, n_cols), dtype=np.float64)
    else:
        raise ValueError(
            f"unknown weighting {weighting!r}; expected one of {C.EXTRA_METRIC_WEIGHTINGS}"
        )
    w[np.arange(n_te), self_cols_te] = 0.0
    return w


def cosine_ceiling(
    half0: np.ndarray,
    half1: np.ndarray,
    folds: list[dict],
    self_cols: np.ndarray,
    target: str = "residual",
    weighting: str = "weighted",
    rows_subset: np.ndarray | None = None,
) -> dict:
    """Split-half reproducibility of the weighted cosine cell.

    ``folds`` is ``runner.fold_reference(...)["folds"]`` verbatim --- a list of
    ``{train, test, weights, mean, rel}`` --- and ``half0``/``half1`` are the
    bundle's two half-effect matrices restricted to the SAME scope rows, in the
    same order. Nothing is recomputed here that ``fold_reference`` already fixed;
    reusing its folds and its weights is what makes this ceiling the ceiling for
    the cell being scored rather than for a differently-split one.

    Aggregation is run 3's: the estimator is evaluated per fold on that fold's
    held-out rows, the per-perturbation values are pooled across folds,
    ``np.nanmedian`` reduces them, and the result is clipped into
    ``(1e-6, 1.0]`` so a downstream division cannot explode.

    ``rows_subset`` restricts the held-out rows of every fold, which is run 4's
    ``restricted_ceilings``: a fine-tuned arm read on folds (0, 1) has to be read
    against the ceiling of those folds, not of all five. A row-wise similarity
    needs no recomputation to be restricted --- each row's value does not depend
    on the other rows --- so restricting here is a filter, unlike the
    discrimination ceiling below.

    **Spearman-Brown IS applied here**, and that is a decision, not a default:
    each half is a half-data estimate, the scoring target is the full-data
    effect, and the cosine is a correlation-like agreement between two vectors
    under exactly the classical measurement model run 3 corrects under. Skipping
    it would score predictors against a cleaner target than the ceiling was
    computed for. ``config.DISATTENUATE["cosine"] is True`` is the same decision.
    """
    half0 = np.asarray(half0, dtype=np.float64)
    half1 = np.asarray(half1, dtype=np.float64)
    self_cols = np.asarray(self_cols, dtype=np.int64)
    n_cols = int(half0.shape[1])
    sub = _subset(rows_subset, int(half0.shape[0]))
    keep = np.zeros(half0.shape[0], dtype=bool)
    keep[sub] = True
    pooled: list[np.ndarray] = []
    for info in folds:
        tr, te = np.asarray(info["train"]), np.asarray(info["test"])
        te = te[keep[te]]
        if te.size == 0:
            continue
        a, b = _half_pair(half0, half1, tr, te, target)
        w = _fold_weights(info["weights"], self_cols[te], int(te.size), n_cols, weighting)
        pooled.append(weighted_cosine_rows(a, b, w))
    values = np.concatenate(pooled) if pooled else np.array([np.nan])
    raw = float(np.nanmedian(values))
    corrected = float(np.clip(metrics.spearman_brown(raw), 1e-6, 1.0))
    return {
        "ceiling": corrected,
        "raw_half_split_median": raw,
        "spearman_brown_applied": True,
        "target": target,
        "weighting": weighting,
        "n": int(np.isfinite(values).sum()),
        "rows_subset_size": int(sub.size),
    }


def pds_ceiling(
    half0: np.ndarray,
    half1: np.ndarray,
    folds: list[dict],
    self_cols: np.ndarray,
    target: str = "residual",
    pool: str = "per_fold",
    rows_subset: np.ndarray | None = None,
    chunk_rows: int = 128,
    device: str | None = None,
) -> dict:
    """Split-half reproducibility of the discrimination score, UNCORRECTED.

    One measured half's profiles are ranked against the other's with the same
    estimator, the same self-column exclusion and the same denominator the score
    uses. It answers the only question that makes a discrimination score
    readable: how identifiable is a perturbation from a *measurement* of it,
    given this much data?

    ``pool`` (``config.PDS_POOLS``) picks the pool the rank is taken against,
    and it must be the pool the score it will be divided into was taken against,
    because ``1 - own_rank / N`` depends on N:

    - ``"pooled"`` --- every row in scope (or in ``rows_subset``) at once, the
      whole-screen reading.
    - ``"per_fold"`` --- each held-out row against its own fold's held-out rows,
      then the per-perturbation values pooled across folds and reduced by
      ``np.nanmedian``, which is run 3's aggregation.

    ``rows_subset`` RECOMPUTES rather than filters. Restricting the pool changes
    every rank inside it, so a subset of full-pool numbers would report a
    524-way ranking as if it were a ~210-way one --- this is exactly why run 4's
    fine-tuned arms cannot reuse a five-fold ceiling.

    **Spearman-Brown is NOT applied**, and the returned
    ``spearman_brown_applied`` flag says so, so that a report cannot claim
    otherwise. The correction is a theorem about the correlation between a
    half-data estimate and a full-data one under the classical measurement
    model; ``1 - own_rank / N`` is a rank statistic and no such correction is
    defined for it. Applying ``2r/(1+r)`` to it would map 0.5 --- chance --- to
    0.667 and manufacture signal out of the correction. This is
    ``config.DISATTENUATE["pds"] is False``. Consequently the score is reported
    as a RATIO to this ceiling, never divided by its square root.
    """
    if pool not in C.PDS_POOLS:
        raise ValueError(f"unknown pool {pool!r}; expected one of {C.PDS_POOLS}")
    half0 = np.asarray(half0, dtype=np.float64)
    half1 = np.asarray(half1, dtype=np.float64)
    self_cols = np.asarray(self_cols, dtype=np.int64)
    sub = _subset(rows_subset, int(half0.shape[0]))
    a_all, b_all = _assemble_halves(half0, half1, folds, target)
    excl = self_cols if C.PDS_EXCLUDE_SELF_COLUMNS else None

    pooled: list[np.ndarray] = []
    sizes: list[int] = []
    if pool == "pooled":
        res = pds_l1(
            a_all[sub], b_all[sub],
            self_cols=None if excl is None else excl[sub],
            denominator=C.PDS_DENOMINATOR, chunk_rows=chunk_rows, device=device,
        )
        pooled.append(np.asarray(res["per_perturbation"], dtype=np.float64))
        sizes.append(int(sub.size))
    else:
        keep = np.zeros(half0.shape[0], dtype=bool)
        keep[sub] = True
        for info in folds:
            te = np.asarray(info["test"])
            te = te[keep[te]]
            if te.size == 0:
                continue
            res = pds_l1(
                a_all[te], b_all[te],
                self_cols=None if excl is None else excl[te],
                denominator=C.PDS_DENOMINATOR, chunk_rows=chunk_rows, device=device,
            )
            pooled.append(np.asarray(res["per_perturbation"], dtype=np.float64))
            sizes.append(int(te.size))
    values = np.concatenate(pooled) if pooled else np.array([np.nan])
    median = float(np.nanmedian(values))
    return {
        "ceiling": float(np.clip(median, 1e-6, 1.0)),
        "raw_half_split_median": median,
        "spearman_brown_applied": False,
        "target": target,
        "pool": pool,
        "denominator": C.PDS_DENOMINATOR,
        "n": int(np.isfinite(values).sum()),
        "pool_size_median": float(np.median(sizes)) if sizes else 0.0,
        "rows_subset_size": int(sub.size),
    }


def extra_ceilings(
    half0: np.ndarray,
    half1: np.ndarray,
    ref: dict,
    rows_subset: np.ndarray | None = None,
    chunk_rows: int = 128,
    device: str | None = None,
) -> dict:
    """Every ceiling ``score_extra`` needs, computed once per (line, scope, split).

    ``ref`` is ``runner.fold_reference``'s return value, so the folds, the
    weights and the self columns are the ones the scored cells use. The ceilings
    do not depend on any arm, which is why they are computed here once rather
    than inside ``score_extra`` for each of the hundreds of arms scored against
    them --- but they DO depend on ``rows_subset``, so a restricted reading needs
    its own call and ``score_extra`` refuses a mismatch rather than dividing by
    the wrong denominator.

    The ``spearman_brown_applied`` map is returned beside the numbers rather than
    left implicit: the cosine ceilings are corrected and the discrimination
    ceilings are not, and a consumer that cannot see which is which will sooner
    or later divide the wrong one.
    """
    self_cols = np.asarray(ref["self_cols"], dtype=np.int64)
    folds = ref["folds"]
    sub = _subset(rows_subset, int(np.asarray(half0).shape[0]))
    ceilings: dict[str, float] = {}
    detail: dict[str, dict] = {}
    for target in C.EXTRA_METRIC_TARGETS:
        for weighting in C.EXTRA_METRIC_WEIGHTINGS:
            out = cosine_ceiling(half0, half1, folds, self_cols, target, weighting, sub)
            key = f"cosine_{target}_{weighting}"
            ceilings[key] = out["ceiling"]
            detail[key] = out
        for pool in C.PDS_POOLS:
            out = pds_ceiling(
                half0, half1, folds, self_cols, target, pool, sub, chunk_rows, device
            )
            key = f"pds_{target}_{pool}"
            ceilings[key] = out["ceiling"]
            detail[key] = out
    return {
        "ceilings": ceilings,
        "spearman_brown_applied": {"cosine": True, "pds": False},
        "rows_subset_size": int(sub.size),
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# (d) the run-6 analogue of runner.score_run
# --------------------------------------------------------------------------- #
def score_extra(
    pred_resid: np.ndarray,
    ref: dict,
    fold: np.ndarray,
    n_folds: int,
    self_cols: np.ndarray | None = None,
    rows_subset: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    ceilings: dict | None = None,
    half0: np.ndarray | None = None,
    half1: np.ndarray | None = None,
    chunk_rows: int = 128,
    device: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """The new scoring cells only, on run 3's own construction of the targets.

    ``pred_resid``, ``ref``, ``fold`` and ``n_folds`` are exactly what
    ``runner.score_run`` takes, and the three (target, truth) pairs are rebuilt
    here line for line as ``score_run`` builds them --- the training-fold mean
    added back for ``raw``, the held-out fold's own per-gene mean removed from
    both sides for ``centered``, the float64 promotion on that centred pair, the
    per-fold reliability weights, and ``weights[arange(n), self_cols] = 0``.
    Mirroring rather than importing is forced: ``score_run`` computes the Pearson
    grid and returns it, with no seam at which a different estimator could be
    substituted. Any drift between the two constructions would make a cosine
    cell and a Pearson cell describe different quantities under the same name,
    so ``tests/test_metrics_extra.py`` pins them together.

    Ceilings come from ``extra_ceilings`` --- pass them in, since they are
    arm-independent; pass ``half0``/``half1`` instead and they are computed here.

    **``pds`` has no weighting axis.** L1 distance is unweighted by construction:
    a per-gene reliability weight inside a Manhattan distance would be a
    different distance, not a weighted version of this one, and it is not the
    distance ``config.PDS_DISTANCE`` names or the one run 3's column reports. The
    knockdown column is still excluded, by dropping it from the sum rather than
    by weighting it to zero.

    **``pds`` is reported at both pools of ``config.PDS_POOLS``, because they
    answer different questions and neither subsumes the other.** ``pooled`` ranks
    every scored row against every scored row --- "can this arm pick its
    perturbation out of the screen" --- and is the pool run 3's
    ``discrimination_ours_residual_*`` column uses, so it is the only one of the
    two comparable with a published run-3 number. ``per_fold`` ranks each
    held-out row against the rows it was held out with --- "out of the
    perturbations it was held out with" --- a smaller and therefore easier pool,
    NOT comparable with run 3's column, and the one whose pool matches the fold
    structure. Each gets its own ceiling at its own pool.

    ``rows_subset`` reads every cell on a subset of the scope rows, which is what
    run 4's ``restrict_scores`` does for the fine-tuned arms on folds (0, 1). The
    three targets and the weight matrix are still built over ALL rows, because
    they depend on the training-fold means and those must stay the means the
    prediction was made against; only the reading is restricted. The cosine cells
    are then a row filter, since a row-wise similarity does not depend on the
    other rows. **The discrimination cells are RECOMPUTED**, both pools and both
    ceilings: restricting a pool changes every rank in it, so subsetting the
    full-pool numbers would report a 524-way ranking as if it were a ~210-way
    one. Rows outside the subset come back NaN in the per-perturbation table,
    because they were not scored by this call.

    ``groups`` switches every interval to ``metrics.bootstrap_median_grouped``,
    the cluster bootstrap over response groups. Perturbations are not
    independent draws --- HepG2's 524 fall into 226 groups, one of them 149 rows
    --- so a row-level interval is too narrow for this design and any interval
    that reaches a claim must be the grouped one. With ``groups=None`` the raw
    row bootstrap ``score_run`` reports is used and the caller is expected to
    regroup later from the per-perturbation table, exactly as run 3 does. Which
    one ran is in ``summary["bootstrap"]``, so a row cannot be read without
    knowing.

    Returns ``(summary, per_pert)``. ``per_pert`` has one row per scope row in
    scope order, aligned with ``score_run``'s, so the two concatenate
    column-wise; it carries its own ``fold`` column so that ``restrict_scores``
    can read it alone, which means a column-wise concat has ``fold`` twice and
    the caller drops the duplicate.
    """
    effect = ref["effect"]
    if self_cols is None:
        self_cols = ref["self_cols"]
    self_cols = np.asarray(self_cols)
    n, _ = effect.shape
    sub = _subset(rows_subset, n)

    if ceilings is None:
        if half0 is None or half1 is None:
            raise RuntimeError(
                "score_extra needs either precomputed ceilings from extra_ceilings() or the "
                "two half-effect matrices to compute them from; scoring against no ceiling "
                "would report a bare number on an unknown scale"
            )
        ceilings = extra_ceilings(half0, half1, ref, sub, chunk_rows, device)
    if int(ceilings.get("rows_subset_size", sub.size)) != int(sub.size):
        raise RuntimeError(
            f"the ceilings were computed on {ceilings['rows_subset_size']} rows and this "
            f"reading is on {sub.size}; a discrimination ceiling is pool-size dependent, so "
            "dividing by it would compare a ranking against a differently-sized one"
        )
    ceil = ceilings["ceilings"]

    true_resid = np.zeros_like(effect)
    pred_raw = np.zeros_like(effect)
    # float64 for the centred pair, run 3's reason verbatim: a prediction that
    # barely varies across perturbations leaves float32 dust behind, and a
    # similarity of dust with anything is a number with no meaning.
    true_cen = np.zeros(effect.shape, dtype=np.float64)
    pred_cen = np.zeros(effect.shape, dtype=np.float64)
    weights = np.zeros_like(effect)
    for f in range(n_folds):
        info = ref["folds"][f]
        te = info["test"]
        true_resid[te] = effect[te] - info["mean"][None, :]
        pred_raw[te] = pred_resid[te] + info["mean"][None, :]
        t64 = true_resid[te].astype(np.float64)
        p64 = pred_resid[te].astype(np.float64)
        true_cen[te] = t64 - t64.mean(axis=0, keepdims=True)
        pred_cen[te] = p64 - p64.mean(axis=0, keepdims=True)
        weights[te] = info["weights"][None, :]
    weights[np.arange(n), self_cols] = 0.0
    ones = np.ones_like(weights)
    ones[np.arange(n), self_cols] = 0.0

    pairs = {
        "residual": (pred_resid, true_resid),
        "centered": (pred_cen, true_cen),
        "raw": (pred_raw, effect),
    }
    n_boot, ci, seed = _bootstrap_params()
    groups_sub = None if groups is None else np.asarray(groups)[sub]

    def interval(values_sub: np.ndarray) -> dict:
        """One interval, grouped when the caller supplied response groups."""
        if groups_sub is None:
            return metrics.bootstrap_median(values_sub, n_boot, ci, seed)
        return metrics.bootstrap_median_grouped(values_sub, groups_sub, n_boot, ci, seed)

    keep = np.zeros(n, dtype=bool)
    keep[sub] = True
    excl = self_cols if C.PDS_EXCLUDE_SELF_COLUMNS else None
    summary: dict = {
        "cosine__spearman_brown_applied": bool(ceilings["spearman_brown_applied"]["cosine"]),
        "pds__spearman_brown_applied": bool(ceilings["spearman_brown_applied"]["pds"]),
        "rows_subset_size": int(sub.size),
        "bootstrap": "row" if groups_sub is None else "grouped",
    }
    columns: dict[str, np.ndarray] = {"fold": np.asarray(fold)}

    for target in C.EXTRA_METRIC_TARGETS:
        p_mat, t_mat = pairs[target]
        for weighting in C.EXTRA_METRIC_WEIGHTINGS:
            w = weights if weighting == "weighted" else ones
            values = np.full(n, np.nan)
            values[sub] = weighted_cosine_rows(p_mat[sub], t_mat[sub], w[sub])
            boot = interval(values[sub])
            ceiling = ceil[f"cosine_{target}_{weighting}"]
            key = f"cosine__{target}__{weighting}"
            summary[f"{key}__attenuated"] = boot["median"]
            summary[f"{key}__attenuated_lo"] = boot["lo"]
            summary[f"{key}__attenuated_hi"] = boot["hi"]
            summary[f"{key}__disattenuated"] = boot["median"] / np.sqrt(ceiling)
            summary[f"{key}__disattenuated_lo"] = boot["lo"] / np.sqrt(ceiling)
            summary[f"{key}__disattenuated_hi"] = boot["hi"] / np.sqrt(ceiling)
            summary[f"{key}__ceiling"] = ceiling
            summary[f"{key}__n_scored"] = boot["n"]
            columns[f"cosine_{target}_{weighting}"] = values

        for pool in C.PDS_POOLS:
            values = np.full(n, np.nan)
            sizes: list[int] = []
            n_sanitised = 0
            n_off = 0
            blocks = [sub] if pool == "pooled" else [
                np.asarray(ref["folds"][f]["test"])[keep[np.asarray(ref["folds"][f]["test"])]]
                for f in range(n_folds)
            ]
            for block in blocks:
                if block.size == 0:
                    continue
                res = pds_l1(
                    p_mat[block], t_mat[block],
                    self_cols=None if excl is None else excl[block],
                    denominator=C.PDS_DENOMINATOR, chunk_rows=chunk_rows, device=device,
                )
                values[block] = res["per_perturbation"]
                sizes.append(int(block.size))
                n_sanitised += int(res["n_nonfinite_sanitised"])
                n_off += int(res["n_nonfinite_outside_self_columns"])
            boot = interval(values[sub])
            ceiling = ceil[f"pds_{target}_{pool}"]
            key = f"pds__{target}__{pool}"
            summary[f"{key}__attenuated"] = boot["median"]
            summary[f"{key}__attenuated_lo"] = boot["lo"]
            summary[f"{key}__attenuated_hi"] = boot["hi"]
            summary[f"{key}__ceiling"] = ceiling
            # NOT a disattenuation: a rank statistic has no sqrt(reliability)
            # correction, so the ceiling is the denominator of a ratio and the
            # raw score is reported beside it.
            summary[f"{key}__ratio_to_ceiling"] = boot["median"] / ceiling
            summary[f"{key}__n_scored"] = boot["n"]
            summary[f"{key}__denominator"] = C.PDS_DENOMINATOR
            summary[f"{key}__pool_size_median"] = float(np.median(sizes)) if sizes else 0.0
            # visible in the record rather than absorbed: on a TranscriptFormer
            # arm this is the count of undefined self positions the exclusion
            # removed, and the second count is how many it did not
            summary[f"{key}__n_nonfinite_sanitised"] = n_sanitised
            summary[f"{key}__n_nonfinite_outside_self_columns"] = n_off
            columns[f"pds_{target}_{pool}"] = values

    return summary, pd.DataFrame(columns)
