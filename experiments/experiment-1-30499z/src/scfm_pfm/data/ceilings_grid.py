"""Top-k overlap ceilings over a grid of k, with chance levels and an adaptive k.

This stage recomputes **only** the replicate-split noise ceiling for the
`topk_overlap` metric. The effect matrices, the differential-expression counts,
the effect-size strata, the cross-check and the model contract are read, never
rewritten: this module writes new files beside them and touches nothing that
already exists.

Why it needs the raw h5ads again: the split-half effect vectors were never
persisted, and the metric is a function of those vectors, not of the pooled
effect matrix. This run persists them (`split_half_vectors.npz`) so a future
question about a different k or a different metric is a file read.

What it adds over the single ``topk_overlap`` column:

``k grid``
    overlap at k in :data:`K_GRID` for every perturbation, instead of one k for
    every stratum. The expensive part is the two streaming passes that rebuild
    the split-half vectors; once those exist, another k costs microseconds.

``chance level``
    two independent sets of size k drawn from n expressed genes overlap at k/n
    by chance, which runs from 0.32% at k=10 to 16% at k=500 in HepG2. Raw
    overlap is therefore not comparable across k, so every row also carries
    ``excess_over_chance`` = (overlap - chance) / (1 - chance): the fraction of
    the *achievable* excess that was reached, 0 at chance and 1 at identity.

``adaptive k``
    k set to the perturbation's own ``n_de_genes``, clipped to
    [:data:`ADAPTIVE_K_MIN`, :data:`ADAPTIVE_K_MAX`]. The strata are rank
    quartiles cut inside a cell line, so a stratum label does not mean the same
    response size in the two lines (median 79 against 130 differentially
    expressed genes in S3, 984 against 448 in S4). A k tied to the response is
    comparable across cell lines in a way a k tied to a label is not.

Two diagnostics come with every cell, because the metric can be quietly
meaningless without them:

``tie_fraction``
    ``argpartition`` breaks ties arbitrarily. This is the fraction of the k
    selected genes sitting within numerical tolerance of the k-th value, i.e.
    the part of the selected set whose membership was effectively a coin flip.
    The k-th value ties itself, so the floor is 1/k; membership is only actually
    contested when ``n_tied_excluded`` (genes that lost the coin flip) is above
    zero, which the summary reports as ``frac_selection_contested``.

``median_control_cp10k_selected``
    selection is on |log2FC| with no expression or significance filter, so a
    small-k set can be dominated by low-expression genes whose ratios are large
    and unreliable. This is the median pooled control level of the selected
    genes, against a median of about 1 CP10K over all expressed genes.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .. import config as C
from .. import paths
from ..stats import cosine, pearson
from . import ceilings, prepare

#: fixed k values evaluated for every perturbation and both split forms
K_GRID = (10, 50, 100, 500)

#: the k the report calls out per stratum -- small k where the response is
#: small, large k where it is large. Any other cell of the grid is still there.
DIAGONAL_K = {"S1_lowest": 10, "S2": 50, "S3": 100, "S4_highest": 500}

#: clip range for the per-perturbation adaptive k
ADAPTIVE_K_MIN, ADAPTIVE_K_MAX = 10, 500

#: a selected value is "tied with the k-th" within this tolerance
TIE_RTOL, TIE_ATOL = 1e-6, 1e-9

SPLIT_PARTS = {
    "split_half_cells": ceilings.HALF_PARTS,
    "split_by_batch": ceilings.BATCHSET_PARTS,
}
QUANTITIES = ("lfc", "delta")


# --------------------------------------------------------------------------- #
# the metric
# --------------------------------------------------------------------------- #
def select_topk(magnitude: np.ndarray, k: int) -> tuple[np.ndarray, int, int]:
    """Indices of the k largest values, plus how arbitrary that selection was.

    Returns ``(indices, n_selected_at_the_boundary, n_tied_but_excluded)``. The
    boundary is the k-th largest value; anything within
    ``TIE_ATOL + TIE_RTOL * |kth|`` of it is indistinguishable from it, so which
    side of the cut those genes land on is decided by ``argpartition``'s
    arbitrary ordering rather than by the data.
    """
    k = int(min(k, magnitude.size))
    idx = np.argpartition(-magnitude, k - 1)[:k]
    kth = float(magnitude[idx].min())
    tol = TIE_ATOL + TIE_RTOL * abs(kth)
    at_boundary = int(np.count_nonzero(np.abs(magnitude[idx] - kth) <= tol))
    tied_total = int(np.count_nonzero(np.abs(magnitude - kth) <= tol))
    return idx, at_boundary, tied_total - at_boundary


def topk_metrics(
    mag0: np.ndarray, mag1: np.ndarray, k: int, control_level: np.ndarray
) -> dict:
    """Overlap of the two sides' top-k sets, with its tie and expression diagnostics."""
    i0, edge0, excl0 = select_topk(mag0, k)
    i1, edge1, excl1 = select_topk(mag1, k)
    k_eff = int(min(k, mag0.size))
    overlap = len(set(i0.tolist()) & set(i1.tolist())) / k_eff
    selected = np.concatenate([i0, i1])
    return {
        "overlap": float(overlap),
        "tie_fraction": float((edge0 + edge1) / (2 * k_eff)),
        "n_tied_excluded": float((excl0 + excl1) / 2),
        "median_control_cp10k_selected": float(np.median(control_level[selected])),
    }


def chance_level(k: int, n_expressed: int) -> float:
    """Expected overlap of two independent size-k sets drawn from n genes."""
    return float(min(k, n_expressed)) / float(n_expressed)


def excess_over_chance(overlap: float, chance: float) -> float:
    """Overlap rescaled so 0 is chance and 1 is a perfectly reproduced set."""
    if not np.isfinite(overlap) or chance >= 1.0:
        return float("nan")
    return float((overlap - chance) / (1.0 - chance))


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #
def reference_dir(cell_line: str) -> Path:
    """Where the published per-perturbation table for this cell line is readable.

    A fresh job volume has no copy of a finished job's artifacts, so the two
    small tables this stage reads travel with the worktree under
    ``results/handoff/effects`` exactly as they do for the model jobs.
    """
    for cand in (
        paths.effects_input_dir(cell_line),
        paths.repo_dir() / "results" / "handoff" / "effects" / cell_line,
    ):
        if (cand / "perturbations.parquet").exists():
            return cand
    raise FileNotFoundError(
        f"no perturbations.parquet for {cell_line} under {paths.effects_input_dir(cell_line)} "
        f"or results/handoff/effects/{cell_line}"
    )


def run(cell_line: str, out_dir: Path, raw_dir: Path) -> dict:
    started = time.time()
    ref_dir = reference_dir(cell_line)
    pert_ref = pd.read_parquet(ref_dir / "perturbations.parquet")
    genes_ref = pd.read_parquet(ref_dir / "genes.parquet")
    print(f"[{cell_line}] reference tables from {ref_dir}", flush=True)

    st = prepare.stage(cell_line, raw_dir)
    n_genes = st.n_genes
    n_targets, n_batches = st.targets.size, st.batches.size
    cp = prepare.control_profiles(st.acc, n_batches, n_genes)
    expressed = cp["expressed"]
    n_expressed = int(expressed.sum())
    keep_t = np.flatnonzero(st.pert_keep)
    targets = st.targets[keep_t]
    t_lin = st.acc.t_lin.reshape(n_targets, 4, n_genes)
    n_tbp = st.acc.t_n

    # The recompute is only worth anything if it reproduces the run that
    # produced the published matrices, so refuse to continue if the working set
    # or the gene mask has moved at all.
    if list(pert_ref["target"]) != list(targets):
        raise RuntimeError(
            f"[{cell_line}] recomputed working set does not match the published one: "
            f"{len(targets)} vs {pert_ref.shape[0]} perturbations"
        )
    published_expressed = genes_ref["expressed"].to_numpy().astype(bool)
    if not np.array_equal(published_expressed, expressed):
        raise RuntimeError(
            f"[{cell_line}] recomputed expressed-gene mask differs from the published one "
            f"({n_expressed} vs {int(published_expressed.sum())} genes)"
        )
    ctrl_published = genes_ref["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    ctrl_max_diff = float(np.max(np.abs(ctrl_published - cp["ctrl_pooled"])))
    control_level = ctrl_published[expressed]
    gene_ids = genes_ref["gene_id"].to_numpy().astype(str)[expressed]

    n_de = pert_ref["n_de_genes"].to_numpy(dtype=np.int64)
    adaptive_k = np.clip(n_de, ADAPTIVE_K_MIN, ADAPTIVE_K_MAX).astype(np.int64)
    stratum = pert_ref["effect_stratum"].to_numpy().astype(str)

    n_pert = len(targets)
    vectors = {
        f"{split}__{q}_side{s}": np.zeros((n_pert, n_expressed), dtype=np.float32)
        for split in SPLIT_PARTS
        for q in QUANTITIES
        for s in (0, 1)
    }
    scored = {split: 0 for split in SPLIT_PARTS}
    repro = {
        f"{split}__{metric}": 0.0
        for split in SPLIT_PARTS
        for metric in ("pearson_lfc", "cosine_delta", "topk_overlap")
    }
    rows: list[dict] = []
    t0 = time.time()

    for row_i, t in enumerate(keep_t):
        for split, parts_map in SPLIT_PARTS.items():
            sides = {}
            for s, parts in parts_map.items():
                n_tb_side = n_tbp[t][:, list(parts)].sum(1)
                ctrl = (
                    cp["ctrl_mean_half"][s]
                    if split == "split_half_cells"
                    else cp["ctrl_mean_per_batch"]
                )
                sides[s] = ceilings.side_vectors(t_lin[t], n_tb_side, parts, ctrl, prepare._lfc)
            (d0, l0, _), (d1, l1, _) = sides[0], sides[1]
            base = {
                "target": targets[row_i],
                "split": split,
                "effect_stratum": stratum[row_i],
                "n_de_genes": int(n_de[row_i]),
                "n_expressed_genes": n_expressed,
            }
            grid = [(str(k), int(k)) for k in K_GRID] + [
                ("adaptive", int(adaptive_k[row_i]))
            ]
            if d0 is None or d1 is None:
                for label, k in grid:
                    rows.append(
                        {
                            **base,
                            "k_label": label,
                            "k": k,
                            "overlap": np.nan,
                            "chance": chance_level(k, n_expressed),
                            "excess_over_chance": np.nan,
                            "tie_fraction": np.nan,
                            "n_tied_excluded": np.nan,
                            "median_control_cp10k_selected": np.nan,
                        }
                    )
                continue

            scored[split] += 1
            l0e, l1e = l0[expressed], l1[expressed]
            d0e, d1e = d0[expressed], d1[expressed]
            for q, (v0, v1) in (("lfc", (l0e, l1e)), ("delta", (d0e, d1e))):
                vectors[f"{split}__{q}_side0"][row_i] = v0.astype(np.float32)
                vectors[f"{split}__{q}_side1"][row_i] = v1.astype(np.float32)

            mag0, mag1 = np.abs(l0e), np.abs(l1e)
            for label, k in grid:
                m = topk_metrics(mag0, mag1, k, control_level)
                c = chance_level(k, n_expressed)
                rows.append(
                    {
                        **base,
                        "k_label": label,
                        "k": k,
                        "chance": c,
                        "excess_over_chance": excess_over_chance(m["overlap"], c),
                        **m,
                    }
                )

            # reproduction of the published columns, computed the same way
            published = pert_ref.iloc[row_i]
            for metric, value in (
                ("pearson_lfc", pearson(l0e, l1e)),
                ("cosine_delta", cosine(d0e, d1e)),
                (
                    "topk_overlap",
                    next(r["overlap"] for r in rows[-len(grid) :] if r["k"] == C.EFFECTS.topk),
                ),
            ):
                ref_val = float(published[f"{split}__{metric}"])
                if np.isfinite(ref_val):
                    repro[f"{split}__{metric}"] = max(
                        repro[f"{split}__{metric}"], abs(value - ref_val)
                    )
        if row_i % 250 == 0:
            print(
                f"    grid {row_i}/{n_pert} perturbations ({time.time() - t0:.0f}s)",
                flush=True,
            )

    grid_df = pd.DataFrame(rows)
    summary = grid_summary(grid_df)

    out_dir = paths.ensure(out_dir)
    grid_df.to_parquet(out_dir / "topk_overlap_grid.parquet", index=False)
    summary.to_parquet(out_dir / "topk_overlap_grid_summary.parquet", index=False)
    vec_path = out_dir / "split_half_vectors.npz"
    np.savez_compressed(
        vec_path,
        targets=np.asarray(targets, dtype=object).astype(str),
        gene_ids=gene_ids,
        gene_index=np.flatnonzero(expressed).astype(np.int64),
        effect_stratum=stratum,
        n_de_genes=n_de,
        **vectors,
    )
    print(
        f"[{cell_line}] split-half vectors {vec_path.stat().st_size / 1e6:.0f} MB "
        f"({len(vectors)} arrays of {n_pert} x {n_expressed} float32)",
        flush=True,
    )

    qc = {
        "cell_line": cell_line,
        "stage": "topk_overlap ceiling grid (recompute of the ceiling stage only)",
        "source": st.download,
        "n_perturbations": n_pert,
        "n_expressed_genes": n_expressed,
        "n_scored_perturbations": scored,
        "config": {
            "k_grid": list(K_GRID),
            "diagonal_k_by_stratum": DIAGONAL_K,
            "adaptive_k": {
                "definition": "clip(n_de_genes, min, max) per perturbation",
                "min": ADAPTIVE_K_MIN,
                "max": ADAPTIVE_K_MAX,
                "median_k": {
                    s: float(np.median(adaptive_k[stratum == s]))
                    for s in C.EFFECTS.strata_labels
                },
            },
            "chance_level": {
                "definition": "k / n_expressed_genes, the expected overlap of two "
                "independent size-k draws",
                "by_k": {str(k): chance_level(k, n_expressed) for k in K_GRID},
            },
            "excess_over_chance": "(overlap - chance) / (1 - chance)",
            "tie_tolerance": {"rtol": TIE_RTOL, "atol": TIE_ATOL},
            "selection": "top k by |log2FC| over expressed genes, no expression or "
            "significance filter",
            "min_cells_per_side": ceilings.MIN_CELLS_PER_SIDE,
            "expressed_min_cp10k": C.EFFECTS.expressed_min_cp10k,
            "seed": C.SEED,
        },
        "reproduction_of_published_ceilings": {
            "reference_dir": str(ref_dir),
            "max_abs_diff": repro,
            "max_abs_diff_pooled_control_cp10k": ctrl_max_diff,
            "note": "the same split assignment, control estimator and metric code as "
            "the published run; these are the largest per-perturbation differences "
            "against perturbations.parquet",
        },
        "outputs": {
            "topk_overlap_grid.parquet": int(grid_df.shape[0]),
            "topk_overlap_grid_summary.parquet": int(summary.shape[0]),
            "split_half_vectors.npz": {
                "bytes": int(vec_path.stat().st_size),
                "arrays": sorted(vectors),
                "shape": [n_pert, n_expressed],
                "dtype": "float32",
                "rows_follow": "targets (same order as perturbations.parquet)",
                "columns_follow": "gene_ids (the expressed genes, in matrix order)",
            },
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
    (out_dir / "topk_overlap_grid_qc.json").write_text(json.dumps(qc, indent=2, default=str))
    print(f"[{cell_line}] wrote {out_dir} in {qc['runtime_seconds']}s", flush=True)
    print(f"[{cell_line}] reproduction max abs diff {repro}", flush=True)
    return qc


def grid_summary(grid: pd.DataFrame) -> pd.DataFrame:
    """Per (split, k, stratum) medians and quartiles, with 'all' as a fifth stratum."""
    rows = []
    labels = [str(k) for k in K_GRID] + ["adaptive"]
    for split in C.CEILING_SPLITS:
        for label in labels:
            for stratum in list(C.EFFECTS.strata_labels) + ["all"]:
                sub = grid[(grid.split == split) & (grid.k_label == label)]
                if stratum != "all":
                    sub = sub[sub.effect_stratum == stratum]
                sub = sub[np.isfinite(sub.overlap.to_numpy(dtype=float))]
                if sub.empty:
                    continue
                o = sub.overlap.to_numpy(dtype=float)
                e = sub.excess_over_chance.to_numpy(dtype=float)
                rows.append(
                    {
                        "split": split,
                        "k_label": label,
                        "stratum": stratum,
                        "n_perturbations": int(o.size),
                        "median_k": float(np.median(sub.k.to_numpy(dtype=float))),
                        "chance": float(np.median(sub.chance.to_numpy(dtype=float))),
                        "median_overlap": float(np.median(o)),
                        "q25_overlap": float(np.percentile(o, 25)),
                        "q75_overlap": float(np.percentile(o, 75)),
                        "median_excess_over_chance": float(np.median(e)),
                        "q25_excess_over_chance": float(np.percentile(e, 25)),
                        "q75_excess_over_chance": float(np.percentile(e, 75)),
                        "median_tie_fraction": float(
                            np.median(sub.tie_fraction.to_numpy(dtype=float))
                        ),
                        "frac_selection_contested": float(
                            np.mean(sub.n_tied_excluded.to_numpy(dtype=float) > 0)
                        ),
                        "median_n_tied_excluded": float(
                            np.median(sub.n_tied_excluded.to_numpy(dtype=float))
                        ),
                        "median_control_cp10k_selected": float(
                            np.median(
                                sub.median_control_cp10k_selected.to_numpy(dtype=float)
                            )
                        ),
                        "is_diagonal": bool(
                            stratum != "all" and DIAGONAL_K.get(stratum) == int(label)
                            if label.isdigit()
                            else False
                        ),
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
