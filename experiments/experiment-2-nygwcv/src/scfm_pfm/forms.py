"""The two array-level operations every feature block shares, torch-free.

Both were defined next to the code that first needed them ---
``normalize_forms`` in the extraction driver, ``partition_masks`` in the model
query interface --- and both are pure numpy. They live here so the estimators
built on them can be tested on the pod, where torch is not installed, without
importing a model adapter.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C


def normalize_forms(delta: np.ndarray, y_pert: np.ndarray, y_base: np.ndarray, model) -> dict:
    """raw / standardized / recentred / log2fc for one block of differences.

    ``delta`` is (..., n_readout). The standardized form is a z-score across
    read-out genes within a query, which is what makes two models with different
    output scales comparable at all. The recentred form removes the per-query
    cross-gene mean, which for a softmax model is the part of every response that
    is redistribution rather than regulation.
    """
    out = {"raw": delta.astype(np.float32)}
    mean = delta.mean(axis=-1, keepdims=True)
    sd = delta.std(axis=-1, keepdims=True)
    out["standardized"] = np.where(sd > 0, (delta - mean) / np.where(sd > 0, sd, 1.0), 0.0).astype(
        np.float32
    )
    if model.linear_scale is not None:
        out["recentred"] = (delta - mean).astype(np.float32)
    if model.linear_scale == "identity":
        lin_p, lin_b = y_pert, y_base
    elif model.linear_scale == "expm1":
        lin_p, lin_b = np.expm1(y_pert), np.expm1(y_base)
    else:
        lin_p = lin_b = None
    if lin_p is not None:
        eps = C.LOG2FC_EPS
        out["log2fc"] = np.log2(
            (np.maximum(lin_p, 0) + eps) / (np.maximum(lin_b, 0) + eps)
        ).astype(np.float32)
    return out


def partition_masks(n: int, rate: float, cycles: int, seed: int) -> list[np.ndarray]:
    """Disjoint mask blocks: every gene is masked exactly once per cycle.

    Masking an independent random subset per pass would leave the unluckiest gene
    unmasked for ~50 passes at a 15% rate. A seeded partition into
    ``ceil(1/rate)`` blocks holds the per-pass masked fraction at the model's own
    pretraining rate and gives every gene exact, auditable coverage.
    """
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    n_blocks = max(1, int(math.ceil(1.0 / max(rate, 1e-6))))
    for _ in range(int(cycles)):
        blocks.extend(np.array_split(rng.permutation(n), n_blocks))
    return blocks
