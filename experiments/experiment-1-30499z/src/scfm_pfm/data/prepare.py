"""Turn a raw GSE264667 h5ad into a documented per-perturbation effect matrix.

Pipeline, per cell line:

1. download the h5ad (size + md5 recorded)
2. pass 1 over X: per-cell matrix UMI total and number of detected genes
3. cell filtering funnel, in a fixed order, counts recorded at every step
4. deterministic split assignments (cell halves, batch sets) for the ceilings
5. pass 2 over X: segment-sum accumulators, in CP10K and log1p(CP10K) space,
   keyed by (target, cell-half, batch-set) and (batch, control-half)
6. effects: within-batch matched delta / log2FC (primary) and batch-pooled
   (secondary), on-target knockdown, Welch DE counts, effect-size strata
7. replicate-split noise ceilings, two split forms x three metrics

Nothing here loads the full matrix; peak memory is one row chunk plus the
accumulators (~1.2 GB for HepG2).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .. import config as C
from .. import paths, stats
from . import h5ad_io, sources

CHUNK_ROWS = 4096


# --------------------------------------------------------------------------- #
# split assignment
# --------------------------------------------------------------------------- #
def assign_cell_halves(groups: np.ndarray, keep: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic balanced 0/1 split of kept cells within each group value."""
    rng = np.random.default_rng(seed)
    half = np.full(groups.shape[0], -1, dtype=np.int8)
    idx = np.flatnonzero(keep)
    order = np.argsort(groups[idx], kind="stable")
    idx = idx[order]
    g = groups[idx]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    bounds = np.r_[starts, g.size]
    for a, b in zip(bounds[:-1], bounds[1:]):
        members = idx[a:b]
        perm = rng.permutation(members.size)
        cut = members.size // 2
        half[members[perm[:cut]]] = 0
        half[members[perm[cut:]]] = 1
    return half


def assign_batch_sets(
    target_idx: np.ndarray, batch_idx: np.ndarray, keep: np.ndarray, n_targets: int, n_batches: int
) -> np.ndarray:
    """Per target, alternate its batches (largest first) between set 0 and 1.

    Alternating by descending cell count keeps the two halves of a
    perturbation's cells comparable in size, which matters because the median
    (target, batch) cell count in this series is 1-2.
    """
    counts = np.zeros((n_targets, n_batches), dtype=np.int64)
    sel = keep & (target_idx >= 0)
    np.add.at(counts, (target_idx[sel], batch_idx[sel]), 1)
    bset_map = np.full((n_targets, n_batches), -1, dtype=np.int8)
    for t in range(n_targets):
        present = np.flatnonzero(counts[t] > 0)
        if present.size == 0:
            continue
        order = present[np.argsort(-counts[t, present], kind="stable")]
        bset_map[t, order] = (np.arange(order.size) % 2).astype(np.int8)
    out = np.full(target_idx.shape[0], -1, dtype=np.int8)
    out[sel] = bset_map[target_idx[sel], batch_idx[sel]]
    return out


# --------------------------------------------------------------------------- #
# accumulation
# --------------------------------------------------------------------------- #
class Accumulators:
    def __init__(self, n_targets: int, n_batches: int, n_genes: int):
        self.n_targets = n_targets
        self.n_batches = n_batches
        self.n_genes = n_genes
        # targets: (target, part) with part = cell_half * 2 + batch_set
        self.t_lin = np.zeros((n_targets * 4, n_genes), dtype=np.float64)
        self.t_log = np.zeros((n_targets, n_genes), dtype=np.float64)
        self.t_log2 = np.zeros((n_targets, n_genes), dtype=np.float64)
        self.t_n = np.zeros((n_targets, n_batches, 4), dtype=np.int64)
        # controls: (batch, control_half)
        self.c_lin = np.zeros((n_batches * 2, n_genes), dtype=np.float64)
        self.c_log = np.zeros((n_batches, n_genes), dtype=np.float64)
        self.c_log2 = np.zeros((n_batches, n_genes), dtype=np.float64)
        self.c_n = np.zeros((n_batches, 2), dtype=np.int64)


def accumulate(
    h: h5py.File,
    keep: np.ndarray,
    is_control: np.ndarray,
    target_idx: np.ndarray,
    batch_idx: np.ndarray,
    cell_half: np.ndarray,
    batch_set: np.ndarray,
    n_counts: np.ndarray,
    acc: Accumulators,
) -> None:
    norm = C.EFFECTS.norm_target
    t0 = time.time()
    for start, stop, block in h5ad_io.iter_row_chunks(h, CHUNK_ROWS):
        sl = slice(start, stop)
        k = keep[sl]
        if not k.any():
            continue
        block = block[k]
        scale = (norm / np.maximum(n_counts[sl][k], 1.0)).astype(np.float32)
        block *= scale[:, None]
        lin = block.astype(np.float64)
        log = np.log1p(lin)

        ctrl = is_control[sl][k]
        pert = ~ctrl
        if pert.any():
            t = target_idx[sl][k][pert]
            b = batch_idx[sl][k][pert]
            part = (cell_half[sl][k][pert] * 2 + batch_set[sl][k][pert]).astype(np.int64)
            stats.group_sum(lin[pert], t * 4 + part, acc.t_lin)
            stats.group_sum(log[pert], t, acc.t_log)
            stats.group_sum(log[pert] ** 2, t, acc.t_log2)
            np.add.at(acc.t_n, (t, b, part), 1)
        if ctrl.any():
            b = batch_idx[sl][k][ctrl]
            ch = cell_half[sl][k][ctrl].astype(np.int64)
            stats.group_sum(lin[ctrl], b * 2 + ch, acc.c_lin)
            stats.group_sum(log[ctrl], b, acc.c_log)
            stats.group_sum(log[ctrl] ** 2, b, acc.c_log2)
            np.add.at(acc.c_n, (b, ch), 1)
        if (start // CHUNK_ROWS) % 8 == 0:
            print(
                f"    accumulate {stop}/{h['X'].shape[0]} rows "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )


# --------------------------------------------------------------------------- #
# effects
# --------------------------------------------------------------------------- #
def _safe_mean(sums: np.ndarray, n: np.ndarray) -> np.ndarray:
    out = np.zeros_like(sums)
    nz = n > 0
    out[nz] = sums[nz] / n[nz, None]
    return out


def _lfc(pert: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    eps = C.EFFECTS.lfc_eps
    return np.log2((pert + eps) / (ctrl + eps))


def control_profiles(acc: Accumulators, n_batches: int, n_genes: int) -> dict:
    """Non-targeting control means: per batch, per control half, and pooled.

    ``expressed`` is the gene mask every magnitude, ceiling and cross-check
    statistic is restricted to, so it is derived here rather than at each use.
    """
    c_lin = acc.c_lin.reshape(n_batches, 2, n_genes)
    c_n = acc.c_n
    n_ctrl_total = int(c_n.sum())
    ctrl_pooled = acc.c_lin.sum(0) / max(n_ctrl_total, 1)
    ctrl_log_m = acc.c_log.sum(0) / max(n_ctrl_total, 1)
    return {
        "ctrl_mean_per_batch": _safe_mean(c_lin.sum(1), c_n.sum(1)),
        "ctrl_mean_half": np.stack(
            [_safe_mean(c_lin[:, 0], c_n[:, 0]), _safe_mean(c_lin[:, 1], c_n[:, 1])]
        ),
        "n_control_cells": n_ctrl_total,
        "ctrl_pooled": ctrl_pooled,
        "ctrl_log_mean": ctrl_log_m,
        "ctrl_log_var": np.maximum(
            (acc.c_log2.sum(0) - n_ctrl_total * ctrl_log_m**2) / max(n_ctrl_total - 1, 1),
            0.0,
        ),
        "expressed": ctrl_pooled >= C.EFFECTS.expressed_min_cp10k,
    }


def matched_control(weights: np.ndarray, ctrl_mean_per_batch: np.ndarray) -> np.ndarray:
    """Batch-matched control profile: control means weighted by the
    perturbation's own per-batch cell counts."""
    w = weights.astype(np.float64)
    total = w.sum()
    if total <= 0:
        return np.full(ctrl_mean_per_batch.shape[1], np.nan)
    return (w / total) @ ctrl_mean_per_batch


@dataclass
class Staged:
    """What the two streaming passes over one raw h5ad produce.

    Everything downstream -- effects, differential expression, the noise
    ceilings, and any later recompute of a ceiling metric -- is a function of
    these arrays alone, so a stage that only needs the ceilings runs the same
    filtering and the same seeded split assignment as the original run instead
    of a second copy of that logic.
    """

    cell_line: str
    download: dict
    n_cells: int
    n_genes: int
    funnel: list
    keep: np.ndarray
    is_control: np.ndarray
    target_label: np.ndarray
    targets: np.ndarray
    target_idx: np.ndarray
    batches: np.ndarray
    batch_idx: np.ndarray
    batches_per_target: np.ndarray
    pert_keep: np.ndarray
    cell_half: np.ndarray
    batch_set: np.ndarray
    acc: "Accumulators"
    obs: pd.DataFrame
    var: pd.DataFrame


def stage(cell_line: str, raw_dir: Path) -> Staged:
    """Download, filter, assign the splits, and accumulate the group sums."""
    url = sources.h5ad_url(cell_line)
    dest = raw_dir / sources.H5AD_FILES[cell_line]
    print(f"[{cell_line}] download {url}", flush=True)
    dl = sources.download(url, dest)
    print(f"[{cell_line}] {dl['size_bytes'] / 1e9:.2f} GB md5={dl['md5']}", flush=True)

    with h5py.File(dest, "r") as h:
        n_cells, n_genes = h5ad_io.matrix_shape(h)
        obs = h5ad_io.read_frame(h, "obs")
        var = h5ad_io.read_frame(h, "var")
        print(f"[{cell_line}] X = {n_cells} cells x {n_genes} genes", flush=True)

        # ---- pass 1: per-cell matrix totals and detected genes
        n_counts = np.zeros(n_cells, dtype=np.float64)
        n_detected = np.zeros(n_cells, dtype=np.int32)
        t0 = time.time()
        for start, stop, block in h5ad_io.iter_row_chunks(h, CHUNK_ROWS):
            n_counts[start:stop] = block.sum(axis=1)
            n_detected[start:stop] = (block > 0).sum(axis=1)
            if (start // CHUNK_ROWS) % 8 == 0:
                print(f"    pass1 {stop}/{n_cells} ({time.time() - t0:.0f}s)", flush=True)

        target_label = obs["gene"].to_numpy().astype(str)
        batch_label = obs["gem_group"].to_numpy().astype(np.int64)
        mito = obs["mitopercent"].to_numpy().astype(np.float64)
        guide = obs["sgID_AB"].to_numpy().astype(str)

        # ---- cell filtering funnel, in order
        funnel = []
        mask = np.ones(n_cells, dtype=bool)
        funnel.append({"step": "raw_cells_in_series_file", "cells": int(mask.sum())})

        single = (
            (target_label != "")
            & (target_label != "nan")
            & (guide != "")
            & (guide != "nan")
        )
        mask &= single
        funnel.append(
            {
                "step": "single_target_assignment_present",
                "criterion": "obs.gene and obs.sgID_AB both non-empty",
                "cells": int(mask.sum()),
                "removed": int((~single).sum()),
            }
        )

        f = n_counts >= C.CELL_FILTERS.min_counts
        removed = int((mask & ~f).sum())
        mask &= f
        funnel.append(
            {
                "step": "min_counts",
                "criterion": f"matrix UMI total >= {C.CELL_FILTERS.min_counts}",
                "cells": int(mask.sum()),
                "removed": removed,
            }
        )

        f = n_detected >= C.CELL_FILTERS.min_genes_detected
        removed = int((mask & ~f).sum())
        mask &= f
        funnel.append(
            {
                "step": "min_genes_detected",
                "criterion": f"genes with >=1 count >= {C.CELL_FILTERS.min_genes_detected}",
                "cells": int(mask.sum()),
                "removed": removed,
            }
        )

        f = mito <= C.CELL_FILTERS.max_mito_fraction
        removed = int((mask & ~f).sum())
        mask &= f
        funnel.append(
            {
                "step": "max_mito_fraction",
                "criterion": f"obs.mitopercent <= {C.CELL_FILTERS.max_mito_fraction}",
                "cells": int(mask.sum()),
                "removed": removed,
            }
        )

        keep = mask
        is_control = target_label == C.CONTROL_LABEL

        targets = np.array(sorted(set(target_label[keep & ~is_control].tolist())))
        t_lookup = {t: i for i, t in enumerate(targets)}
        target_idx = np.array([t_lookup.get(t, -1) for t in target_label], dtype=np.int64)
        batches = np.array(sorted(set(batch_label.tolist())))
        b_lookup = {b: i for i, b in enumerate(batches)}
        batch_idx = np.array([b_lookup[b] for b in batch_label], dtype=np.int64)
        n_targets, n_batches = targets.size, batches.size
        if n_targets == 0:
            raise RuntimeError(
                "no perturbation survived cell QC -- check the filter thresholds "
                "against the matrix (e.g. min_genes_detected vs the gene count)"
            )

        cells_per_target = np.zeros(n_targets, dtype=np.int64)
        sel = keep & ~is_control
        np.add.at(cells_per_target, target_idx[sel], 1)
        batches_per_target = np.array(
            [len(set(batch_idx[sel & (target_idx == t)].tolist())) for t in range(n_targets)]
        )
        funnel.append(
            {
                "step": "perturbations_with_any_qc_passing_cell",
                "perturbations": int(n_targets),
                "control_cells": int((keep & is_control).sum()),
            }
        )
        pert_keep = (cells_per_target >= C.PERTURBATION_FILTERS.min_cells) & (
            batches_per_target >= C.PERTURBATION_FILTERS.min_batches
        )
        funnel.append(
            {
                "step": "min_cells_per_perturbation",
                "criterion": f">= {C.PERTURBATION_FILTERS.min_cells} QC-passing cells",
                "perturbations": int(
                    (cells_per_target >= C.PERTURBATION_FILTERS.min_cells).sum()
                ),
            }
        )
        funnel.append(
            {
                "step": "min_batches_per_perturbation",
                "criterion": f">= {C.PERTURBATION_FILTERS.min_batches} sequencing batches",
                "perturbations": int(pert_keep.sum()),
                "cells": int(sel.sum()),
            }
        )

        # cells of dropped perturbations no longer contribute
        drop_cells = sel & ~pert_keep[np.clip(target_idx, 0, None)]
        keep = keep & ~drop_cells
        sel = keep & ~is_control
        funnel.append(
            {
                "step": "final_working_set",
                "cells": int(keep.sum()),
                "perturbed_cells": int(sel.sum()),
                "control_cells": int((keep & is_control).sum()),
                "perturbations": int(pert_keep.sum()),
            }
        )
        for row in funnel:
            print(f"    funnel {row}", flush=True)

        # ---- deterministic splits
        group_for_half = np.where(is_control, -1 - batch_idx, target_idx)
        cell_half = assign_cell_halves(group_for_half, keep, C.SEED)
        batch_set = assign_batch_sets(target_idx, batch_idx, keep & ~is_control, n_targets, n_batches)
        batch_set = np.where(is_control, 0, batch_set)
        cell_half = np.where(cell_half < 0, 0, cell_half)
        batch_set = np.where(batch_set < 0, 0, batch_set)

        # ---- pass 2
        acc = Accumulators(n_targets, n_batches, n_genes)
        accumulate(
            h,
            keep,
            is_control,
            np.clip(target_idx, 0, None),
            batch_idx,
            cell_half,
            batch_set,
            n_counts,
            acc,
        )

    return Staged(
        cell_line=cell_line,
        download=dl,
        n_cells=int(n_cells),
        n_genes=int(n_genes),
        funnel=funnel,
        keep=keep,
        is_control=is_control,
        target_label=target_label,
        targets=targets,
        target_idx=target_idx,
        batches=batches,
        batch_idx=batch_idx,
        batches_per_target=batches_per_target,
        pert_keep=pert_keep,
        cell_half=cell_half,
        batch_set=batch_set,
        acc=acc,
        obs=obs,
        var=var,
    )


def run(cell_line: str, out_dir: Path, raw_dir: Path) -> dict:
    started = time.time()
    st = stage(cell_line, raw_dir)
    dl, funnel, acc = st.download, st.funnel, st.acc
    obs, var = st.obs, st.var
    n_cells, n_genes = st.n_cells, st.n_genes
    targets, target_label = st.targets, st.target_label
    n_targets, n_batches = targets.size, st.batches.size
    keep = st.keep
    pert_keep, batches_per_target = st.pert_keep, st.batches_per_target

    gene_ids = var["gene_id"].to_numpy().astype(str)
    gene_names = var["gene_name"].to_numpy().astype(str)

    # ---- control profiles
    cp = control_profiles(acc, n_batches, n_genes)
    ctrl_mean_per_batch = cp["ctrl_mean_per_batch"]
    ctrl_mean_half = cp["ctrl_mean_half"]
    n_ctrl_total = cp["n_control_cells"]
    ctrl_pooled = cp["ctrl_pooled"]
    ctrl_log_m, ctrl_log_v = cp["ctrl_log_mean"], cp["ctrl_log_var"]
    expressed = cp["expressed"]

    # ---- per-perturbation effects
    keep_t = np.flatnonzero(pert_keep)
    t_lin = acc.t_lin.reshape(n_targets, 4, n_genes)
    n_tbp = acc.t_n
    n_tb = n_tbp.sum(2)
    n_t = n_tb.sum(1)

    pert_mean = np.zeros((keep_t.size, n_genes), dtype=np.float64)
    matched = np.zeros_like(pert_mean)
    for row, t in enumerate(keep_t):
        pert_mean[row] = t_lin[t].sum(0) / max(n_t[t], 1)
        matched[row] = matched_control(n_tb[t], ctrl_mean_per_batch)

    delta_within = pert_mean - matched
    lfc_within = _lfc(pert_mean, matched)
    delta_pooled = pert_mean - ctrl_pooled[None, :]
    lfc_pooled = _lfc(pert_mean, np.broadcast_to(ctrl_pooled, pert_mean.shape))

    # differential expression against pooled controls (log1p CP10K, Welch)
    m1 = acc.t_log[keep_t] / np.maximum(n_t[keep_t], 1)[:, None]
    v1 = np.maximum(
        (acc.t_log2[keep_t] - n_t[keep_t][:, None] * m1**2)
        / np.maximum(n_t[keep_t] - 1, 1)[:, None],
        0.0,
    )
    de_q = np.ones_like(m1, dtype=np.float64)
    n_de = np.zeros(keep_t.size, dtype=np.int64)
    for row in range(keep_t.size):
        _, p = stats.welch_ttest(
            m1[row], v1[row], float(n_t[keep_t[row]]), ctrl_log_m, ctrl_log_v, float(n_ctrl_total)
        )
        p = np.where(expressed, p, np.nan)
        q = stats.benjamini_hochberg(p)
        de_q[row] = q
        n_de[row] = int((q < C.EFFECTS.de_fdr).sum())

    # on-target knockdown
    tgt_ensg_map: dict[str, str] = {}
    for lab, ens in zip(target_label, obs["gene_id"].to_numpy().astype(str)):
        if lab != C.CONTROL_LABEL and lab not in tgt_ensg_map:
            tgt_ensg_map[lab] = ens
    gene_row = {g: i for i, g in enumerate(gene_ids)}
    name_row: dict[str, int] = {}
    for i, g in enumerate(gene_names):
        name_row.setdefault(g, i)

    kd = np.full(keep_t.size, np.nan)
    kd_lfc = np.full(keep_t.size, np.nan)
    tgt_ctrl_level = np.full(keep_t.size, np.nan)
    tgt_measured = np.zeros(keep_t.size, dtype=bool)
    tgt_ensg = []
    for row, t in enumerate(keep_t):
        label = targets[t]
        ens = tgt_ensg_map.get(label, "")
        tgt_ensg.append(ens)
        j = gene_row.get(ens, name_row.get(label, -1))
        if j < 0:
            continue
        tgt_measured[row] = True
        ctrl_lvl = matched[row, j]
        tgt_ctrl_level[row] = ctrl_lvl
        if ctrl_lvl > 0:
            kd[row] = 1.0 - pert_mean[row, j] / ctrl_lvl
        kd_lfc[row] = lfc_within[row, j]

    eff_l2 = np.linalg.norm(np.where(expressed, lfc_within, 0.0), axis=1)
    eff_rms = eff_l2 / np.sqrt(max(int(expressed.sum()), 1))
    eff_l2_delta = np.linalg.norm(np.where(expressed, delta_within, 0.0), axis=1)

    ranks = pd.Series(eff_l2).rank(method="first")
    strat_codes = np.floor((ranks - 1) / len(ranks) * C.EFFECTS.n_strata).astype(int)
    strat_codes = np.clip(strat_codes, 0, C.EFFECTS.n_strata - 1)
    stratum = np.array([C.EFFECTS.strata_labels[i] for i in strat_codes])

    # ---- noise ceilings
    from .ceilings import compute_ceilings  # local import keeps module graph flat

    ceilings = compute_ceilings(
        keep_t=keep_t,
        target_labels=targets[keep_t],
        t_lin=t_lin,
        n_tbp=n_tbp,
        ctrl_mean_half=ctrl_mean_half,
        ctrl_mean_per_batch=ctrl_mean_per_batch,
        expressed=expressed,
        lfc_fn=_lfc,
    )

    # ---- write outputs
    out_dir = paths.ensure(out_dir)
    np.savez_compressed(
        out_dir / "effects.npz",
        targets=targets[keep_t],
        gene_ids=gene_ids,
        gene_names=gene_names,
        delta_within_batch=delta_within.astype(np.float32),
        lfc_within_batch=lfc_within.astype(np.float32),
        delta_pooled=delta_pooled.astype(np.float32),
        lfc_pooled=lfc_pooled.astype(np.float32),
        de_qvalue=de_q.astype(np.float32),
        pert_mean_cp10k=pert_mean.astype(np.float32),
        matched_control_cp10k=matched.astype(np.float32),
        pooled_control_cp10k=ctrl_pooled.astype(np.float32),
        expressed=expressed,
    )

    pert = pd.DataFrame(
        {
            "target": targets[keep_t],
            "target_ensembl": tgt_ensg,
            "target_gene_measured": tgt_measured,
            "n_cells": n_t[keep_t],
            "n_batches": batches_per_target[keep_t],
            "n_cells_half0": n_tbp[keep_t][:, :, [0, 1]].sum(axis=(1, 2)),
            "n_cells_half1": n_tbp[keep_t][:, :, [2, 3]].sum(axis=(1, 2)),
            "n_cells_batchset0": n_tbp[keep_t][:, :, [0, 2]].sum(axis=(1, 2)),
            "n_cells_batchset1": n_tbp[keep_t][:, :, [1, 3]].sum(axis=(1, 2)),
            "knockdown": kd,
            "knockdown_log2fc": kd_lfc,
            "target_control_cp10k": tgt_ctrl_level,
            "knockdown_detected": kd >= C.EFFECTS.knockdown_detect_min,
            "effect_l2_lfc": eff_l2,
            "effect_rms_lfc": eff_rms,
            "effect_l2_delta": eff_l2_delta,
            "n_de_genes": n_de,
            "effect_stratum": stratum,
        }
    ).merge(ceilings, on="target", how="left")
    pert.to_parquet(out_dir / "perturbations.parquet", index=False)

    genes = pd.DataFrame(
        {
            "gene_index": np.arange(n_genes),
            "gene_id": gene_ids,
            "gene_name": gene_names,
            "pooled_control_cp10k": ctrl_pooled,
            "expressed": expressed,
        }
    )
    genes.to_parquet(out_dir / "genes.parquet", index=False)

    summary = ceiling_summary(pert)
    summary.to_parquet(out_dir / "noise_ceiling_summary.parquet", index=False)

    qc = {
        "cell_line": cell_line,
        "source": dl,
        "shape": {"cells": int(n_cells), "genes": int(n_genes)},
        "n_gem_groups": int(n_batches),
        "filter_funnel": funnel,
        "config": {
            "seed": C.SEED,
            "control_label": C.CONTROL_LABEL,
            "cell_filters": vars(C.CELL_FILTERS),
            "perturbation_filters": vars(C.PERTURBATION_FILTERS),
            "effects": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in vars(C.EFFECTS).items()
            },
            "normalization": "per-cell CP10K over the measured gene matrix, "
            "pseudobulk = mean of CP10K over cells",
            "primary_estimator": "within-batch matched control: control means per "
            "gem_group weighted by the perturbation's own per-batch cell counts",
            "de_test": "Welch t-test on log1p(CP10K), perturbed cells vs pooled "
            "non-targeting controls, BH across expressed genes",
        },
        "retained": {
            "perturbations": int(pert_keep.sum()),
            "cells": int(keep.sum()),
            "control_cells": int(n_ctrl_total),
            "expressed_genes": int(expressed.sum()),
        },
        "knockdown": {
            "median": float(np.nanmedian(kd)),
            "n_below_detect_threshold": int(np.sum(~(kd >= C.EFFECTS.knockdown_detect_min))),
            "n_target_gene_not_measured": int((~tgt_measured).sum()),
        },
        "runtime_seconds": round(time.time() - started, 1),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
    }
    (out_dir / "qc.json").write_text(json.dumps(qc, indent=2, default=str))
    print(f"[{cell_line}] wrote {out_dir}", flush=True)
    return qc


def ceiling_summary(pert: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in C.CEILING_SPLITS:
        for metric in C.CEILING_METRICS:
            col = f"{split}__{metric}"
            if col not in pert:
                continue
            for stratum in list(C.EFFECTS.strata_labels) + ["all"]:
                sub = pert if stratum == "all" else pert[pert["effect_stratum"] == stratum]
                vals = sub[col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                rows.append(
                    {
                        "split": split,
                        "metric": metric,
                        "stratum": stratum,
                        "n_perturbations": int(vals.size),
                        "median": float(np.median(vals)),
                        "q25": float(np.percentile(vals, 25)),
                        "q75": float(np.percentile(vals, 75)),
                        "mean": float(vals.mean()),
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-line", choices=list(paths.CELL_LINES) + ["all"], default="all")
    args = ap.parse_args(argv)
    lines = paths.CELL_LINES if args.cell_line == "all" else (args.cell_line,)
    raw = paths.ensure(paths.raw_dir())
    for cl in lines:
        run(cl, paths.effects_dir(cl), raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
