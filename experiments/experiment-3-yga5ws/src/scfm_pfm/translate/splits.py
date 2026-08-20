"""Held-out perturbation splits.

The independent unit is the perturbed gene, and perturbed genes are not
independent of each other: paralogs and members of the same complex produce
near-identical responses, so a random split leaks a gene's family into training.
The primary split therefore groups perturbed genes by the similarity of their
MEASURED effect vectors and holds out whole groups. A plain random-gene split
runs beside it; the gap between the two says how much of any arm's score is
family memorization.

Grouping uses the target data, which is deliberate: it is a device to prevent
leakage, applied identically to every arm, not a model-selection step.
"""

from __future__ import annotations

import numpy as np


def effect_groups(effect: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage groups over 1 - Pearson between effect vectors."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    x = np.asarray(effect, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    d = pdist(x, metric="correlation")
    d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)
    z = linkage(d, method="average")
    return fcluster(z, t=1.0 - threshold, criterion="distance").astype(np.int64)


def balanced_group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign whole groups to folds, largest group first into the emptiest fold."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts + rng.random(counts.size) * 1e-6)
    load = np.zeros(n_folds, dtype=np.int64)
    fold_of_group: dict[int, int] = {}
    for gi in order:
        f = int(np.argmin(load))
        fold_of_group[int(uniq[gi])] = f
        load[f] += int(counts[gi])
    return np.array([fold_of_group[int(g)] for g in groups], dtype=np.int64)


def make_folds(
    effect: np.ndarray, split_type: str, n_folds: int, seed: int, threshold: float
) -> dict:
    n = effect.shape[0]
    if split_type == "random":
        rng = np.random.default_rng(seed)
        fold = np.zeros(n, dtype=np.int64)
        fold[rng.permutation(n)] = np.arange(n) % n_folds
        groups = np.arange(n, dtype=np.int64)
    elif split_type == "grouped":
        groups = effect_groups(effect, threshold)
        fold = balanced_group_folds(groups, n_folds, seed)
    else:
        raise ValueError(f"unknown split type {split_type!r}")
    sizes = [int((fold == f).sum()) for f in range(n_folds)]
    return {
        "fold": fold,
        "groups": groups,
        "n_groups": int(np.unique(groups).size),
        "fold_sizes": sizes,
        "largest_group": int(np.bincount(groups.astype(np.int64)).max()),
    }
