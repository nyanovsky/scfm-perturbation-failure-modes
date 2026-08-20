"""Co-expression and per-gene properties from the non-targeting control cells.

This is the control the UGRN paper never ran: the same translator, fed plain
statistical dependence between the two genes measured in the very cells the
effects were estimated against.

Two disciplines matter here.

*Cell disjointness.* The control cells are split in half by a fixed seed. Every
feature is computed on one half; the other half exists so the same features can
be recomputed on cells the reported ones never saw, which is how the run checks
that the baseline is not inflated by sharing sampling noise with the target.

*No pair term in the node control.* ``node_features`` returns per-gene
properties only. Nothing in them depends on the relationship between the two
genes, so an arm built from them measures how far gene identity alone gets.
"""

from __future__ import annotations

import numpy as np


def split_control_cells(n_cells: int, seed: int) -> np.ndarray:
    """Balanced 0/1 assignment of control cells."""
    rng = np.random.default_rng(seed)
    half = np.zeros(n_cells, dtype=np.int8)
    perm = rng.permutation(n_cells)
    half[perm[n_cells // 2 :]] = 1
    return half


def _standardize(x: np.ndarray) -> np.ndarray:
    """Column-standardize, leaving constant columns at zero."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    out = np.zeros_like(x)
    ok = (sd > 1e-9).ravel()
    out[:, ok] = (x[:, ok] - mu[:, ok]) / sd[:, ok]
    return out


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks down each column (ties averaged)."""
    n, m = x.shape
    out = np.empty((n, m), dtype=np.float32)
    for j in range(m):
        col = x[:, j]
        order = np.argsort(col, kind="stable")
        sorted_col = col[order]
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            k = i
            while k + 1 < n and sorted_col[k + 1] == sorted_col[i]:
                k += 1
            ranks[order[i : k + 1]] = 0.5 * (i + k) + 1.0
            i = k + 1
        out[:, j] = ranks
    return out


def _quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency bin index per column, ties collapsed into one bin."""
    n, m = x.shape
    codes = np.zeros((n, m), dtype=np.int8)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for j in range(m):
        col = x[:, j]
        edges = np.unique(np.quantile(col, qs))
        if edges.size == 0:
            continue
        codes[:, j] = np.digitize(col, edges, right=False).astype(np.int8)
    return codes


def mutual_information(
    codes: np.ndarray, rows: np.ndarray, n_bins: int, chunk: int = 512
) -> np.ndarray:
    """MI in bits between each ``rows`` gene and every gene, from bin codes.

    Computed as 64 indicator matrix products rather than a per-pair histogram
    loop: for bins (a, b), the joint count is ``1[i in a]^T 1[j in b]``.
    """
    n_cells, n_genes = codes.shape
    p_row = np.zeros((rows.size, n_bins), dtype=np.float64)
    p_col = np.zeros((n_genes, n_bins), dtype=np.float64)
    ind: list[np.ndarray] = []
    for a in range(n_bins):
        ia = (codes == a).astype(np.float32)
        ind.append(ia)
        p_col[:, a] = ia.sum(axis=0) / n_cells
        p_row[:, a] = ia[:, rows].sum(axis=0) / n_cells
    mi = np.zeros((rows.size, n_genes), dtype=np.float64)
    for a in range(n_bins):
        ia_rows = np.ascontiguousarray(ind[a][:, rows].T)  # (P, N)
        for b in range(n_bins):
            joint = np.zeros((rows.size, n_genes), dtype=np.float64)
            for start in range(0, n_genes, chunk):
                stop = min(start + chunk, n_genes)
                joint[:, start:stop] = ia_rows @ ind[b][:, start:stop]
            joint /= n_cells
            denom = p_row[:, a][:, None] * p_col[:, b][None, :]
            ok = (joint > 0) & (denom > 0)
            mi[ok] += joint[ok] * np.log2(joint[ok] / denom[ok])
    return mi.astype(np.float32)


def pair_statistics(
    log_expr: np.ndarray, rows: np.ndarray, n_bins: int, shrinkage: float
) -> dict:
    """Pearson, Spearman, mutual information and partial correlation.

    ``log_expr`` is (n_cells, n_genes) log1p CP10K over the read-out set;
    ``rows`` are the gene columns that are also perturbed genes.
    """
    n_cells = log_expr.shape[0]
    z = _standardize(log_expr.astype(np.float32))
    corr = (z.T @ z) / max(n_cells - 1, 1)
    np.fill_diagonal(corr, 1.0)
    pearson = corr[rows].astype(np.float32)

    zr = _standardize(_ranks(log_expr))
    spearman = ((zr[:, rows].T @ zr) / max(n_cells - 1, 1)).astype(np.float32)

    codes = _quantile_bins(log_expr, n_bins)
    mi = mutual_information(codes, rows, n_bins)

    c = corr.astype(np.float64)
    c = (1.0 - shrinkage) * c + shrinkage * np.eye(c.shape[0])
    prec = np.linalg.inv(c)
    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    partial = (-prec[rows] / (d[rows][:, None] * d[None, :])).astype(np.float32)

    return {
        "pearson": pearson,
        "spearman": spearman,
        "mi_bits": mi,
        "partial": partial,
        "corr_full": corr.astype(np.float32),
    }


def node_features(
    cp10k: np.ndarray, log_expr: np.ndarray, corr_full: np.ndarray, degree_abs_corr: float
) -> tuple[np.ndarray, list[str]]:
    """Per-gene properties: level, spread, detection, and co-expression degree.

    Every column is a property of ONE gene. Nothing here depends on a pair, so
    an arm built from these two vectors measures gene identity alone.
    """
    mean_cp10k = cp10k.mean(axis=0)
    feats = np.stack(
        [
            np.log1p(mean_cp10k),
            log_expr.mean(axis=0),
            log_expr.std(axis=0),
            (cp10k > 0).mean(axis=0),
            (np.abs(corr_full) > degree_abs_corr).sum(axis=1).astype(np.float32)
            / corr_full.shape[1],
            np.abs(corr_full).mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    names = [
        "log1p_mean_cp10k",
        "mean_log1p",
        "sd_log1p",
        "detection_rate",
        "coexpr_degree_fraction",
        "mean_abs_corr",
    ]
    return feats, names
