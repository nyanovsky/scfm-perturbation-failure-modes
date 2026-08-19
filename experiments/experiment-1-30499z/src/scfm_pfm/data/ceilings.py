"""Replicate-split noise ceilings.

Two split forms, both giving two *independent* estimates of the same
perturbation's effect:

``split_half_cells``
    the perturbation's QC-passing cells are split in half (seeded), and each
    half is referenced against a **disjoint** half of the non-targeting control
    cells, so shared control noise cannot inflate the agreement.

``split_by_batch``
    the perturbation's sequencing batches are alternated into two sets
    (largest batch first), and each set is referenced against the control cells
    of its own batches. Control cells are disjoint by construction.

Three metrics per split, matching what later runs will score models with:
per-perturbation correlation across genes (on log2FC), cosine similarity to the
other split's delta vector (on the linear CP10K delta), and top-k
differentially expressed gene overlap (by |log2FC|).

``ceilings_grid`` extends the third metric to a grid of k with chance levels and
an adaptive k, reusing :func:`side_vectors` so both read the same split.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .. import config as C
from ..stats import cosine, pearson, topk_overlap

MIN_CELLS_PER_SIDE = 5

HALF_PARTS = {0: (0, 1), 1: (2, 3)}
BATCHSET_PARTS = {0: (0, 2), 1: (1, 3)}


def side_vectors(
    t_lin_t: np.ndarray,
    n_tb_side: np.ndarray,
    parts: tuple[int, int],
    ctrl_mean_per_batch: np.ndarray,
    lfc_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """One side of a split: (delta, log2FC, cells), or (None, None, n) if too small."""
    n = int(n_tb_side.sum())
    if n < MIN_CELLS_PER_SIDE:
        return None, None, n  # type: ignore[return-value]
    pert_mean = t_lin_t[list(parts)].sum(0) / n
    w = n_tb_side.astype(np.float64)
    usable = w.sum()
    if usable <= 0:
        return None, None, n  # type: ignore[return-value]
    matched = (w / usable) @ ctrl_mean_per_batch
    if not np.all(np.isfinite(matched)):
        return None, None, n  # type: ignore[return-value]
    return pert_mean - matched, lfc_fn(pert_mean, matched), n


def compute_ceilings(
    keep_t: np.ndarray,
    target_labels: np.ndarray,
    t_lin: np.ndarray,
    n_tbp: np.ndarray,
    ctrl_mean_half: np.ndarray,
    ctrl_mean_per_batch: np.ndarray,
    expressed: np.ndarray,
    lfc_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> pd.DataFrame:
    k = C.EFFECTS.topk
    rows = []
    for row_i, t in enumerate(keep_t):
        rec: dict[str, object] = {"target": target_labels[row_i]}
        for split, parts_map in (
            ("split_half_cells", HALF_PARTS),
            ("split_by_batch", BATCHSET_PARTS),
        ):
            sides = {}
            for s, parts in parts_map.items():
                n_tb_side = n_tbp[t][:, list(parts)].sum(1)
                ctrl = ctrl_mean_half[s] if split == "split_half_cells" else ctrl_mean_per_batch
                sides[s] = side_vectors(t_lin[t], n_tb_side, parts, ctrl, lfc_fn)
            (d0, l0, n0), (d1, l1, n1) = sides[0], sides[1]
            rec[f"{split}__n_side0"] = n0
            rec[f"{split}__n_side1"] = n1
            if d0 is None or d1 is None:
                for metric in C.CEILING_METRICS:
                    rec[f"{split}__{metric}"] = np.nan
                rec[f"{split}__pearson_delta"] = np.nan
                rec[f"{split}__cosine_lfc"] = np.nan
                continue
            e = expressed
            rec[f"{split}__pearson_lfc"] = pearson(l0[e], l1[e])
            rec[f"{split}__cosine_delta"] = cosine(d0[e], d1[e])
            rec[f"{split}__topk_overlap"] = topk_overlap(l0[e], l1[e], k)
            rec[f"{split}__pearson_delta"] = pearson(d0[e], d1[e])
            rec[f"{split}__cosine_lfc"] = cosine(l0[e], l1[e])
        rows.append(rec)
    return pd.DataFrame(rows)
