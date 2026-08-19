"""Small numeric helpers shared by the data track."""

from __future__ import annotations

import numpy as np


def group_sum(block: np.ndarray, gid: np.ndarray, out: np.ndarray) -> None:
    """``out[g] += sum of block rows with gid == g`` (segment-sum, no big temporaries)."""
    if block.shape[0] == 0:
        return
    order = np.argsort(gid, kind="stable")
    gs = gid[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    sums = np.add.reduceat(block[order], starts, axis=0)
    np.add.at(out, gs[starts], sums)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = int(min(k, a.size))
    if k == 0:
        return float("nan")
    ta = set(np.argpartition(-np.abs(a), k - 1)[:k].tolist())
    tb = set(np.argpartition(-np.abs(b), k - 1)[:k].tolist())
    return len(ta & tb) / k


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency (average ranks on ties)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return pearson(_rank(a[ok]), _rank(b[ok]))


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # average ties
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values; NaN inputs come back as 1.0."""
    p = np.asarray(p, dtype=np.float64)
    q = np.ones_like(p)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    q[ok] = adj
    return q


def welch_ttest(
    m1: np.ndarray, v1: np.ndarray, n1: float, m2: np.ndarray, v2: np.ndarray, n2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sample Welch t statistic and two-sided p-value (scipy.stats.t)."""
    from scipy import stats  # local import: only the data job needs scipy

    se2 = v1 / max(n1, 1.0) + v2 / max(n2, 1.0)
    se = np.sqrt(np.where(se2 > 0, se2, np.nan))
    t = (m1 - m2) / se
    with np.errstate(divide="ignore", invalid="ignore"):
        df = se2**2 / (
            (v1 / max(n1, 1.0)) ** 2 / max(n1 - 1, 1.0)
            + (v2 / max(n2, 1.0)) ** 2 / max(n2 - 1, 1.0)
        )
    p = np.full(t.shape, np.nan)
    ok = np.isfinite(t) & np.isfinite(df) & (df > 0)
    p[ok] = 2.0 * stats.t.sf(np.abs(t[ok]), df[ok])
    return t, p
