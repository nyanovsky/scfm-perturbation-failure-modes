"""Scoring: reliability, ceiling, weighted correlation, bootstrap, power.

Two distinct reliabilities live here and are deliberately kept apart.

``per_gene_reliability`` runs ACROSS perturbations for one gene and becomes the
weight in the score: under the classical measurement model it is
sigma_signal^2 / (sigma_signal^2 + sigma_noise^2) for that gene, so weighting by
it is the continuous form of an expression floor.

``residual_ceiling`` runs ACROSS genes for one perturbation and becomes the
denominator: it is how much perturbation-specific signal is reproducible at all,
and it bounds every predictor.

Both are estimated from half-split effect vectors and are therefore the
reliability of a HALF-data estimate. The scoring target is the full-data effect,
so both are Spearman-Brown corrected to 2r/(1+r) before use; skipping that would
score predictors against a cleaner target than the ceiling was computed for.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# correlation
# --------------------------------------------------------------------------- #
def spearman_brown(r: np.ndarray | float) -> np.ndarray | float:
    """Half-split reliability -> reliability of the full-data estimate."""
    r = np.clip(np.asarray(r, dtype=np.float64), -0.999999, 0.999999)
    return 2.0 * r / (1.0 + r)


def weighted_pearson_rows(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise weighted Pearson correlation.

    ``x``, ``y`` are (n_rows, n_cols); ``w`` is (n_cols,) or (n_rows, n_cols).
    Rows whose weighted variance vanishes return NaN rather than 0.
    """
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
    ok = sw > EPS
    mx = np.divide((w * x).sum(axis=1), sw, out=np.zeros_like(sw), where=ok)
    my = np.divide((w * y).sum(axis=1), sw, out=np.zeros_like(sw), where=ok)
    dx = x - mx[:, None]
    dy = y - my[:, None]
    cov = (w * dx * dy).sum(axis=1)
    vx = (w * dx * dx).sum(axis=1)
    vy = (w * dy * dy).sum(axis=1)
    denom = np.sqrt(vx * vy)
    out = np.full(x.shape[0], np.nan)
    good = ok & (denom > EPS)
    out[good] = cov[good] / denom[good]
    return out


def columnwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of matching columns of two (n, m) matrices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    da = a - a.mean(axis=0, keepdims=True)
    db = b - b.mean(axis=0, keepdims=True)
    num = (da * db).sum(axis=0)
    den = np.sqrt((da * da).sum(axis=0) * (db * db).sum(axis=0))
    out = np.full(a.shape[1], np.nan)
    ok = den > EPS
    out[ok] = num[ok] / den[ok]
    return out


# --------------------------------------------------------------------------- #
# reliability weights (across perturbations, per gene)
# --------------------------------------------------------------------------- #
def per_gene_reliability(
    half0: np.ndarray, half1: np.ndarray, rows: np.ndarray | None = None
) -> dict:
    """Per-gene reliability from the two half-estimates, training rows only.

    Empirical-Bayes shrinkage in Fisher-z: the sampling variance of each z is
    1/(n-3), so the between-gene variance in excess of that sets how much of a
    gene's own estimate to keep. Then Spearman-Brown to the full-data estimate,
    then clip at zero.
    """
    a = half0 if rows is None else half0[rows]
    b = half1 if rows is None else half1[rows]
    n = a.shape[0]
    raw = columnwise_pearson(a, b)
    finite = np.isfinite(raw)
    z = np.arctanh(np.clip(raw, -0.999, 0.999))
    var_sampling = 1.0 / max(n - 3, 1)
    zf = z[finite]
    var_total = float(np.var(zf)) if zf.size > 1 else 0.0
    var_between = max(var_total - var_sampling, 0.0)
    lam = var_between / (var_between + var_sampling) if (var_between + var_sampling) > 0 else 0.0
    zbar = float(np.mean(zf)) if zf.size else 0.0
    z_shrunk = zbar + lam * (z - zbar)
    r_shrunk = np.tanh(z_shrunk)
    r_full = np.asarray(spearman_brown(r_shrunk), dtype=np.float64)
    w = np.clip(np.where(np.isfinite(r_full), r_full, 0.0), 0.0, 1.0)
    return {
        "weights": w,
        "raw_half_split": raw,
        "shrinkage_lambda": float(lam),
        "mean_z": zbar,
        "n_perturbations": int(n),
        "var_between_z": float(var_between),
        "var_sampling_z": float(var_sampling),
    }


# --------------------------------------------------------------------------- #
# residual ceiling (across genes, per perturbation)
# --------------------------------------------------------------------------- #
def residualize(effect: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out residual against the mean effect of the training rows.

    Returns (residual, mean_used_per_row). Rows inside ``train_rows`` use the
    leave-one-out mean; rows outside use the plain training mean, which is what
    a held-out perturbation would see.
    """
    effect = np.asarray(effect, dtype=np.float64)
    n_train = int(train_rows.size)
    total = effect[train_rows].sum(axis=0)
    mean_all = total / max(n_train, 1)
    means = np.broadcast_to(mean_all, effect.shape).copy()
    if n_train > 1:
        loo = (total[None, :] - effect[train_rows]) / (n_train - 1)
        means[train_rows] = loo
    return effect - means, means


def residual_ceiling(
    half0: np.ndarray,
    half1: np.ndarray,
    train_rows: np.ndarray,
    weights: np.ndarray,
    self_cols: np.ndarray | None = None,
) -> np.ndarray:
    """Per-perturbation weighted split-half correlation of the RESIDUAL effect.

    Each half is residualized against that half's own training-row mean, so the
    number is the reproducibility of the perturbation-specific part only. Not
    Spearman-Brown corrected here; the caller does that once on the summary.
    """
    r0, _ = residualize(half0, train_rows)
    r1, _ = residualize(half1, train_rows)
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64), half0.shape).copy()
    if self_cols is not None:
        w[np.arange(w.shape[0]), self_cols] = 0.0
    return weighted_pearson_rows(r0, r1, w)


# --------------------------------------------------------------------------- #
# uncertainty and power
# --------------------------------------------------------------------------- #
def bootstrap_median(
    values: np.ndarray, n_resamples: int = 2000, ci: float = 0.95, seed: int = 42
) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    meds = np.median(v[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    return {
        "median": float(np.median(v)),
        "lo": float(np.quantile(meds, alpha)),
        "hi": float(np.quantile(meds, 1 - alpha)),
        "n": int(v.size),
    }


def bootstrap_median_difference(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 2000, ci: float = 0.95, seed: int = 42
) -> dict:
    """Paired bootstrap of median(a) - median(b) over shared rows."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return {"difference": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_resamples, a.size))
    diffs = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    return {
        "difference": float(np.median(a) - np.median(b)),
        "lo": float(np.quantile(diffs, alpha)),
        "hi": float(np.quantile(diffs, 1 - alpha)),
        "n": int(a.size),
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }


# --------------------------------------------------------------------------- #
# uncertainty over GROUPS rather than rows
# --------------------------------------------------------------------------- #
# Perturbed genes are not independent draws: paralogs and complex members give
# near-identical responses, which is why whole response groups are held out
# together. Resampling individual perturbations therefore understates the
# uncertainty -- HepG2's 524 perturbations are only 226 groups, one of them 149
# rows. The functions below resample GROUPS with replacement (a cluster
# bootstrap) and take the median of the whole resampled multiset, which reduces
# to the row bootstrap when every group is a singleton.


def _group_counts(groups: np.ndarray, n_resamples: int, rng: np.random.Generator) -> tuple:
    """(sorted-row order, group index per sorted row, multiplicity matrix).

    Drawing ``G`` groups with replacement is exactly a Multinomial(G, 1/G) over
    the groups, so one (n_resamples x G) count matrix stands in for the draws.
    """
    uniq, gidx = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(uniq.size)
    counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_resamples)
    return gidx, counts.astype(np.int32), n_groups


def _weighted_median(sorted_values: np.ndarray, sorted_gidx: np.ndarray, counts: np.ndarray):
    """Median of the multiset that repeats each row by its group's count.

    Matches ``np.median``: the average of the two central order statistics when
    the resampled total is even, the central one when it is odd.
    """
    w = counts[:, sorted_gidx]
    cw = np.cumsum(w, axis=1, dtype=np.int64)
    total = cw[:, -1:]
    i_lo = np.argmax(cw >= (total + 1) // 2, axis=1)
    i_hi = np.argmax(cw >= total // 2 + 1, axis=1)
    return 0.5 * (sorted_values[i_lo] + sorted_values[i_hi])


def bootstrap_median_grouped(
    values: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Cluster bootstrap of the median: groups resampled, rows carried whole."""
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if v.size == 0:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "n": 0, "n_groups": 0, "largest_group": 0}
    rng = np.random.default_rng(seed)
    gidx, counts, n_groups = _group_counts(g, n_resamples, rng)
    order = np.argsort(v, kind="stable")
    meds = _weighted_median(v[order], gidx[order], counts)
    alpha = (1.0 - ci) / 2.0
    return {
        "median": float(np.median(v)),
        "lo": float(np.quantile(meds, alpha)),
        "hi": float(np.quantile(meds, 1 - alpha)),
        "se": float(np.std(meds, ddof=1)),
        "n": int(v.size),
        "n_groups": n_groups,
        "largest_group": int(np.bincount(gidx).max()),
    }


def bootstrap_median_difference_grouped(
    a: np.ndarray,
    b: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Paired cluster bootstrap of median(a) - median(b) over shared rows."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    g = np.asarray(groups)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, g = a[ok], b[ok], g[ok]
    if a.size == 0:
        return {"difference": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "n": 0, "n_groups": 0}
    rng = np.random.default_rng(seed)
    gidx, counts, n_groups = _group_counts(g, n_resamples, rng)
    oa, ob = np.argsort(a, kind="stable"), np.argsort(b, kind="stable")
    diffs = _weighted_median(a[oa], gidx[oa], counts) - _weighted_median(b[ob], gidx[ob], counts)
    alpha = (1.0 - ci) / 2.0
    return {
        "difference": float(np.median(a) - np.median(b)),
        "lo": float(np.quantile(diffs, alpha)),
        "hi": float(np.quantile(diffs, 1 - alpha)),
        "se": float(np.std(diffs, ddof=1)),
        "n": int(a.size),
        "n_groups": n_groups,
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }


def min_resolvable_difference(
    sd_single: float, n: int, paired_rho: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest paired mean difference a two-sided test resolves at ``power``.

    ``sd_single`` is the spread of per-perturbation scores within one arm; two
    arms scored on the same perturbations are correlated, so the paired standard
    deviation is sqrt(2 (1 - rho)) times it.
    """
    from math import sqrt

    z_alpha = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(1 - alpha / 2)
    z_beta = 0.8416212335729143 if abs(power - 0.80) < 1e-9 else _z(power)
    sd_paired = sd_single * sqrt(max(2.0 * (1.0 - paired_rho), 0.0))
    return float((z_alpha + z_beta) * sd_paired / sqrt(max(n, 1)))


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    from math import log, sqrt

    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )
