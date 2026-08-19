"""Does the expression floor depress the replicate-split ceiling, and where does it stop?

Run 1 called a gene "expressed" at a pooled control level of 0.5 CP10K and
measured every ceiling over that set. Ranking genes by |log2 fold change|
preferentially picks small denominators, so a floor that low can put genes into
the ceiling whose two replicate halves disagree mostly because the assay barely
measured them -- which would depress the ceiling for reasons that have nothing
to do with the biology or with any model.

The question is answerable without touching raw data: run 1 persisted the two
replicate halves' per-gene vectors (``split_half_vectors.npz``), and every
higher floor is a subset of the genes already in them. This stage recomputes the
ceiling at each floor in :data:`~scfm_pfm.config.FLOOR_GRID_CP10K`, for both cell
lines and both split forms, and applies one stated rule to pick run 2's read-out
floor:

    the read-out floor is the LOWEST floor whose top-stratum median Pearson
    ceiling is within :data:`~scfm_pfm.config.FLOOR_PLATEAU_TOLERANCE` of the
    best value any higher floor reaches, in both split forms.

So the read-out set stays as inclusive as it can be without the ceiling still
climbing when the weakest genes are dropped. If the ceiling never rises by more
than the tolerance, the rule returns the lowest floor on the grid and the run
records that the floor was not the limiting factor.

    python -m scfm_pfm.data.floor_sweep --vectors-dir /tmp/r2in --out results/floor_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import paths

SPLITS = ("split_half_cells", "split_by_batch")
#: k for the "which genes does |log2FC| ranking select" diagnostic
DIAG_K = 50


def row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Pearson correlation of two (n_rows, n_genes) blocks."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    out = np.full(a.shape[0], np.nan)
    ok = (na > 0) & (nb > 0)
    out[ok] = np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok])
    return out


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    out = np.full(a.shape[0], np.nan)
    ok = (na > 0) & (nb > 0)
    out[ok] = np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok])
    return out


def selected_control_level(lfc: np.ndarray, control: np.ndarray, k: int) -> np.ndarray:
    """Median control CP10K of each row's top-k genes by |log2FC|."""
    k = int(min(k, lfc.shape[1]))
    idx = np.argpartition(-np.abs(lfc), k - 1, axis=1)[:, :k]
    return np.median(control[idx], axis=1)


def _row_standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows centred and scaled to unit norm, plus the mask of usable rows."""
    z = x.astype(np.float64)
    z = z - z.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(z, axis=1)
    ok = norm > 0
    z[ok] = z[ok] / norm[ok][:, None]
    return z, ok


def shared_response_diagnostics(
    a: np.ndarray, b: np.ndarray, max_pairs: int = 400
) -> dict:
    """How much of a stratum's response is shared across perturbations?

    A higher expression floor removes noisy genes, which lifts the ceiling. It
    also removes genes, and the ones it keeps are the high-expression ribosomal,
    mitochondrial and housekeeping transcripts that move in a similar direction
    whenever an essential gene is knocked down. If that is what a high floor is
    buying, then a high-floor ceiling is easy to approach without knowing
    anything perturbation-specific, and a raw correlation there means less than
    the same number at a low floor.

    Three measurements, all built so predictor and target never share sampling
    noise -- ``a`` and ``b`` are the two independent replicate halves:

    ``loo_mean_baseline``
        predict perturbation *i*'s effect vector by the mean of every OTHER
        perturbation's vector in the same stratum (from half ``a``), and score it
        against *i*'s own vector from half ``b`` with the same Pearson the
        ceiling uses. This is the score a model reaches by emitting the stratum's
        average response and nothing else.
    ``cross_perturbation_pearson``
        median Pearson between *different* perturbations, half ``a`` against
        half ``b``, i.e. how much any two perturbations already look alike.
    ``pc1_variance_share``
        share of the total sum of squares of the effect matrix carried by its
        first singular direction (uncentred, so the common axis is not removed;
        the perturbation-centred value is reported next to it).
    """
    az, aok = _row_standardize(a)
    bz, bok = _row_standardize(b)
    ok = aok & bok
    n = int(ok.sum())
    if n < 3:
        return {"n_scored": n}
    az, bz = az[ok], bz[ok]

    ceiling = np.einsum("ij,ij->i", az, bz)

    # leave-one-out mean of the other perturbations, scored against the held-out half
    total = az.sum(axis=0)
    loo = (total[None, :] - az) / (n - 1)
    loo_z, loo_ok = _row_standardize(loo)
    baseline = np.full(n, np.nan)
    baseline[loo_ok] = np.einsum("ij,ij->i", loo_z[loo_ok], bz[loo_ok])

    # cross-perturbation agreement, subsampled when the stratum is large
    idx = np.arange(n)
    if n > max_pairs:
        idx = np.random.default_rng(C.SEED).choice(n, size=max_pairs, replace=False)
    cross = az[idx] @ bz[idx].T
    off = ~np.eye(idx.size, dtype=bool)

    effect = (a[ok] + b[ok]) / 2.0
    sv = np.linalg.svd(effect.astype(np.float64), compute_uv=False)
    centred = effect - effect.mean(axis=0, keepdims=True)
    sv_c = np.linalg.svd(centred, compute_uv=False)
    return {
        "n_scored": n,
        "median_ceiling": float(np.nanmedian(ceiling)),
        "median_loo_mean_baseline": float(np.nanmedian(baseline)),
        "median_cross_perturbation_pearson": float(np.median(cross[off])),
        "pc1_variance_share_uncentred": float(sv[0] ** 2 / np.sum(sv**2)),
        "pc1_variance_share_centred": float(sv_c[0] ** 2 / np.sum(sv_c**2)),
        "headroom_ceiling_over_baseline": float(
            np.nanmedian(ceiling) - np.nanmedian(baseline)
        ),
    }


def sweep_cell_line(
    cell_line: str, vectors_path: Path, genes_path: Path, floors: tuple[float, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    npz = np.load(vectors_path, allow_pickle=False)
    genes = pd.read_parquet(genes_path)
    gene_index = npz["gene_index"].astype(np.int64)  # rows of genes.parquet, in matrix order
    control = genes["pooled_control_cp10k"].to_numpy(dtype=np.float64)[gene_index]
    gene_ids = genes["gene_id"].to_numpy().astype(str)[gene_index]
    if not np.array_equal(gene_ids, npz["gene_ids"].astype(str)):
        raise RuntimeError(f"[{cell_line}] gene order in the vectors and genes.parquet disagree")
    stratum = npz["effect_stratum"].astype(str)
    targets = npz["targets"].astype(str)

    per_rows: list[dict] = []
    summary_rows: list[dict] = []
    shared_rows: list[dict] = []
    for floor in floors:
        mask = control >= floor
        n_genes = int(mask.sum())
        for split in SPLITS:
            l0 = npz[f"{split}__lfc_side0"][:, mask]
            l1 = npz[f"{split}__lfc_side1"][:, mask]
            d0 = npz[f"{split}__delta_side0"][:, mask]
            d1 = npz[f"{split}__delta_side1"][:, mask]
            pear = row_pearson(l0, l1)
            cos = row_cosine(d0, d1)
            sel = selected_control_level(l0, control[mask], DIAG_K)
            for i, t in enumerate(targets):
                per_rows.append(
                    {
                        "cell_line": cell_line,
                        "floor_cp10k": float(floor),
                        "split": split,
                        "target": t,
                        "effect_stratum": stratum[i],
                        "pearson_lfc": float(pear[i]),
                        "cosine_delta": float(cos[i]),
                        "median_control_cp10k_topk_lfc": float(sel[i]),
                    }
                )
            for s in C.PERTURBED_STRATA:
                rows = stratum == s
                if not rows.any():
                    continue
                diag = shared_response_diagnostics(l0[rows], l1[rows])
                shared_rows.append(
                    {
                        "cell_line": cell_line,
                        "floor_cp10k": float(floor),
                        "split": split,
                        "effect_stratum": s,
                        "n_expressed_genes": n_genes,
                        **diag,
                    }
                )
            for s in sorted(set(stratum.tolist())):
                rows = stratum == s
                summary_rows.append(
                    {
                        "cell_line": cell_line,
                        "floor_cp10k": float(floor),
                        "split": split,
                        "effect_stratum": s,
                        "n_perturbations": int(rows.sum()),
                        "n_scored": int(np.isfinite(pear[rows]).sum()),
                        "n_expressed_genes": n_genes,
                        "median_pearson_lfc": float(np.nanmedian(pear[rows])),
                        "mean_pearson_lfc": float(np.nanmean(pear[rows])),
                        "median_cosine_delta": float(np.nanmedian(cos[rows])),
                        "median_control_cp10k_topk_lfc": float(np.nanmedian(sel[rows])),
                    }
                )
    per = pd.DataFrame(per_rows)
    summary = pd.DataFrame(summary_rows)
    shared = pd.DataFrame(shared_rows)
    meta = {
        "cell_line": cell_line,
        "vectors": str(vectors_path),
        "n_perturbations": int(targets.size),
        "n_genes_at_run1_floor": int(control.size),
        "median_control_cp10k_all_expressed": float(np.median(control)),
        "genes_by_floor": {
            f"{floor:g}": int((control >= floor).sum()) for floor in floors
        },
    }
    return per, summary, meta, shared


def choose_floor(summary: pd.DataFrame, cell_line: str) -> dict:
    """Apply the stated rule to one cell line's ceiling-vs-floor curve.

    The rule looks for a plateau. If the ceiling is still climbing at the top of
    the grid there is no plateau to find, and picking the highest floor would be
    reading the rule backwards: it would keep the ~140 best-measured genes and
    throw away 96% of the read-out set to buy correlation. In that case the
    read-out set stays at run 1's inclusive floor and the floor becomes a
    scoring-time column mask -- every block stores the per-gene control level, so
    any floor on this curve can be applied later without recomputing a feature.
    """
    table = summary[
        (summary.cell_line == cell_line) & (summary.effect_stratum == C.FLOOR_RULE_STRATUM)
    ]
    picks = {}
    for split in SPLITS:
        curve = table[table.split == split].sort_values("floor_cp10k")
        floors = curve.floor_cp10k.to_numpy(dtype=float)
        ceil = curve.median_pearson_lfc.to_numpy(dtype=float)
        best = float(np.nanmax(ceil))
        ok = np.flatnonzero(ceil >= best - C.FLOOR_PLATEAU_TOLERANCE)
        pick = float(floors[ok[0]]) if ok.size else float(floors[-1])
        picks[split] = {
            "plateau_pick_floor": pick,
            "plateau_found": bool(pick < float(floors[-1])),
            "best_ceiling": best,
            "ceiling_at_pick": float(ceil[ok[0]]) if ok.size else float("nan"),
            "ceiling_at_lowest_floor": float(ceil[0]),
            "rise_from_lowest_to_best": best - float(ceil[0]),
            "curve": {f"{f:g}": float(c) for f, c in zip(floors, ceil)},
        }
    plateau_found = all(p["plateau_found"] for p in picks.values())
    limiting = any(
        p["rise_from_lowest_to_best"] > C.FLOOR_PLATEAU_TOLERANCE for p in picks.values()
    )
    chosen = (
        max(p["plateau_pick_floor"] for p in picks.values())
        if plateau_found
        else C.RUN1_FLOOR_CP10K
    )
    coverage = (
        summary[(summary.cell_line == cell_line) & (summary.split == SPLITS[0])]
        .drop_duplicates("floor_cp10k")
        .sort_values("floor_cp10k")
    )
    return {
        "cell_line": cell_line,
        "rule": (
            "lowest floor whose S4_highest median Pearson ceiling is within "
            f"{C.FLOOR_PLATEAU_TOLERANCE} of the best value any higher floor reaches, "
            "required in both split forms; if the ceiling is still rising at the top of "
            "the grid there is no such floor and the read-out set stays at run 1's "
            f"{C.RUN1_FLOOR_CP10K} CP10K floor, with the floor recorded as a "
            "scoring-time column mask"
        ),
        "per_split": picks,
        "plateau_found": bool(plateau_found),
        "chosen_floor_cp10k": float(chosen),
        "floor_was_limiting": bool(limiting),
        "ceiling_vs_coverage": {
            f"{row.floor_cp10k:g}": {
                "n_genes_above_floor": int(row.n_expressed_genes),
                "s4_median_pearson_ceiling": float(
                    picks[SPLITS[0]]["curve"][f"{row.floor_cp10k:g}"]
                ),
            }
            for row in coverage.itertuples()
        },
        "note": (
            "the ceiling rises monotonically across the whole grid, so part of run 1's "
            "low ceiling was the floor rather than the biology; but no floor stops it "
            "rising, and every step up costs most of the read-out set, so the floor is "
            "reported as a curve instead of being baked into the extraction"
            if limiting and not plateau_found
            else "the ceiling never rose by more than the tolerance, so the floor was "
            "not what held the ceiling down"
        ),
    }


def readout_genes(cell_line: str, floor: float, genes_path: Path, universe: pd.DataFrame) -> pd.DataFrame:
    """The read-out set: in the shared universe, expressed in this line at ``floor``."""
    genes = pd.read_parquet(genes_path)
    keep = genes[genes.pooled_control_cp10k >= floor]
    frame = universe.merge(
        keep[["gene_id", "pooled_control_cp10k"]], on="gene_id", how="inner"
    )
    return frame.sort_values("gene_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vectors-dir", required=True, help="holds shv_<line>.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--handoff", default=None, help="where the read-out gene tables go")
    args = ap.parse_args(argv)

    out = paths.ensure(Path(args.out) if args.out else paths.floor_dir())
    handoff = paths.ensure(Path(args.handoff) if args.handoff else paths.handoff_root() / "readout")
    universe = pd.read_parquet(paths.universe_dir() / "gene_universe.parquet")
    started = time.time()

    pers, summaries, shareds, metas, decisions = [], [], [], {}, {}
    for cell_line in C.CELL_LINES:
        vectors = Path(args.vectors_dir) / f"shv_{cell_line}.npz"
        genes_path = paths.effects_dir(cell_line) / "genes.parquet"
        per, summary, meta, shared = sweep_cell_line(
            cell_line, vectors, genes_path, C.FLOOR_GRID_CP10K
        )
        pers.append(per)
        summaries.append(summary)
        shareds.append(shared)
        metas[cell_line] = meta
        print(f"[floor] {cell_line}: {meta['n_perturbations']} perturbations", flush=True)

    summary = pd.concat(summaries, ignore_index=True)
    per = pd.concat(pers, ignore_index=True)
    shared = pd.concat(shareds, ignore_index=True)
    for cell_line in C.CELL_LINES:
        decision = choose_floor(summary, cell_line)
        frame = readout_genes(cell_line, decision["chosen_floor_cp10k"], paths.effects_dir(cell_line) / "genes.parquet", universe)
        frame.to_parquet(handoff / f"readout_genes_{cell_line}.parquet", index=False)
        decision["n_readout_genes"] = int(frame.shape[0])
        decision["n_universe_genes"] = int(universe.shape[0])
        decisions[cell_line] = decision
        print(
            f"[floor] {cell_line}: floor {decision['chosen_floor_cp10k']} CP10K -> "
            f"{decision['n_readout_genes']} read-out genes",
            flush=True,
        )

    summary.to_parquet(out / "floor_ceiling_summary.parquet", index=False)
    per.to_parquet(out / "floor_ceiling_per_perturbation.parquet", index=False)
    shared.to_parquet(out / "floor_shared_response.parquet", index=False)
    primary = shared[shared.split == SPLITS[0]]
    payload = {
        "stage": "expression-floor sensitivity of the replicate-split ceiling",
        "floors_cp10k": list(C.FLOOR_GRID_CP10K),
        "run1_floor_cp10k": C.RUN1_FLOOR_CP10K,
        "plateau_tolerance": C.FLOOR_PLATEAU_TOLERANCE,
        "rule_stratum": C.FLOOR_RULE_STRATUM,
        "metric": "per-perturbation Pearson correlation of the two replicate halves' "
        "log2 fold-change vectors over the genes surviving the floor",
        "inputs": metas,
        "decisions": decisions,
        "shared_response": {
            "question": "does a higher floor buy its higher ceiling by narrowing the "
            "read-out to the response every perturbation shares?",
            "baseline": "leave-one-out mean effect vector of the other perturbations in "
            "the same stratum, taken from one replicate half and scored against the "
            "other half with the ceiling's own Pearson",
            "by_cell_line_stratum_floor": {
                f"{row.cell_line}|{row.effect_stratum}|{row.floor_cp10k:g}": {
                    "ceiling": round(float(row.median_ceiling), 4),
                    "loo_mean_baseline": round(float(row.median_loo_mean_baseline), 4),
                    "headroom": round(float(row.headroom_ceiling_over_baseline), 4),
                    "cross_perturbation_pearson": round(
                        float(row.median_cross_perturbation_pearson), 4
                    ),
                    "pc1_variance_share_uncentred": round(
                        float(row.pc1_variance_share_uncentred), 4
                    ),
                    "n_genes": int(row.n_expressed_genes),
                }
                for row in primary.itertuples()
            },
        },
        "seconds": round(time.time() - started, 1),
    }
    (out / "floor_decision.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(decisions, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
