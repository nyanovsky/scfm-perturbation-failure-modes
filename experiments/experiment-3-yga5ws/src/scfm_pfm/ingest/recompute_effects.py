"""Recompute the effect target against ONE HALF of the control cells.

Run 1's effect is a within-batch log2 fold change of a perturbation's cells
against the non-targeting control cells, and the co-expression baseline is
estimated from those same control cells. Sampling noise in a control mean can
therefore appear in both the feature and the thing being predicted, which is a
way for the statistical baseline to be scored on something other than biology.

This job removes the overlap by recomputing the target from the raw matrices
against one half of the control cells only. The halves are the same two halves
the ingest job used (same filter, same cell order, same seeded split), so:

    target ``ctrl0``   built on control half 0 — DISJOINT from the cells the
                       ``coexp`` features come from (half 1)
    target ``ctrl1``   built on control half 1 — the shared-cell version of the
                       same comparison, which is what the difference between the
                       two isolates
    target ``ctrlall`` both halves, the closest thing to run 1's estimator; used
                       to check that this recomputation reproduces it

One streaming pass per cell line. Control rows are small enough to hold
(~5k x 3.2k), so they are kept and split after the pass; perturbed rows are
accumulated into per (perturbation, batch, half-of-cells) sums as they stream by,
which is what the split-half reliability vectors need.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import metrics, paths
from . import coexpr as coexprmod

#: pseudocount in CP10K units, added before the log on both sides
EPS_CP10K = 1.0
#: a batch enters a perturbation's average only if both sides have this many
#: cells; a one-cell pseudobulk is noise with a batch label on it
MIN_CELLS_PER_BATCH = 3
MIN_CONTROL_CELLS_PER_BATCH = 10


def _cell_side(n: int, seed: int) -> np.ndarray:
    """Which half of a perturbation's own cells a cell belongs to, for the
    split-half reliability estimate."""
    rng = np.random.default_rng(seed)
    side = np.zeros(n, dtype=np.int8)
    side[rng.permutation(n)[n // 2 :]] = 1
    return side


def stream_line(cell_line: str, raw_dir: Path, scope: pd.DataFrame, readout: pd.DataFrame) -> dict:
    """One pass: control rows kept, perturbed rows summed per (target, batch, side)."""
    import h5py  # noqa: PLC0415 - only the job needs it, not the analysis side

    from ..data import h5ad_io, sources
    from . import cells as cellmod

    matrix_cols = readout[f"{cell_line}_matrix_index"].to_numpy().astype(np.int64)
    url = sources.h5ad_url(cell_line)
    dest = Path(raw_dir) / sources.H5AD_FILES[cell_line]
    if not dest.exists():
        dl = sources.download(url, dest)
        print(f"[{cell_line}] downloaded {dl['size_bytes'] / 1e9:.2f} GB", flush=True)
    targets = scope["target"].to_numpy().astype(str)
    t_index = {t: i for i, t in enumerate(targets)}
    n_genes = matrix_cols.size

    with h5py.File(dest, "r") as h:
        n_cells, _ = h5ad_io.matrix_shape(h)
        obs = h5ad_io.read_frame(h, "obs")
        var = h5ad_io.read_frame(h, "var")
        got = (
            var["gene_id"].to_numpy().astype(str)
            if "gene_id" in var.columns
            else var.index.to_numpy().astype(str)
        )[matrix_cols]
        want = readout["gene_id"].to_numpy().astype(str)
        if not np.array_equal(got, want):
            raise RuntimeError(f"{cell_line}: read-out columns do not match the raw matrix")

        label = obs["gene"].to_numpy().astype(str)
        guide = obs["sgID_AB"].to_numpy().astype(str)
        mito = obs["mitopercent"].to_numpy().astype(np.float64)
        batch = obs["gem_group"].to_numpy().astype(np.int64)
        is_control = label == cellmod.CONTROL_LABEL
        single = (
            (label != "") & (label != "nan") & (guide != "") & (guide != "nan")
        )
        # a stable per-cell side, independent of chunk boundaries
        side_all = _cell_side(n_cells, C.SEED + 3301)

        ctrl_rows: list[np.ndarray] = []
        ctrl_batch: list[np.ndarray] = []
        sums: dict[tuple[int, int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(n_genes, dtype=np.float64)
        )
        counts: dict[tuple[int, int, int], int] = defaultdict(int)
        n_kept = 0
        t0 = time.time()
        for start, stop, block in h5ad_io.iter_row_chunks(h, cellmod.CHUNK_ROWS):
            n_counts = block.sum(axis=1)
            n_detected = (block > 0).sum(axis=1)
            keep = (
                single[start:stop]
                & (n_counts >= cellmod.MIN_COUNTS)
                & (n_detected >= cellmod.MIN_GENES_DETECTED)
                & (mito[start:stop] <= cellmod.MAX_MITO_FRACTION)
            )
            if not keep.any():
                continue
            n_kept += int(keep.sum())
            scale = (cellmod.NORM_TARGET / np.maximum(n_counts, 1.0)).astype(np.float32)
            sub = block[:, matrix_cols] * scale[:, None]
            ctrl_sel = keep & is_control[start:stop]
            if ctrl_sel.any():
                ctrl_rows.append(sub[ctrl_sel].astype(np.float32))
                ctrl_batch.append(batch[start:stop][ctrl_sel])
            pert_sel = keep & ~is_control[start:stop]
            if pert_sel.any():
                idx = np.flatnonzero(pert_sel)
                labs = label[start:stop][pert_sel]
                bs = batch[start:stop][pert_sel]
                sd = side_all[start:stop][pert_sel]
                for k, lab in enumerate(labs):
                    ti = t_index.get(str(lab))
                    if ti is None:
                        continue
                    key = (ti, int(bs[k]), int(sd[k]))
                    sums[key] += sub[idx[k]]
                    counts[key] += 1
            if (start // cellmod.CHUNK_ROWS) % 20 == 0:
                print(
                    f"[{cell_line}] {stop}/{n_cells} cells in {time.time() - t0:.0f}s",
                    flush=True,
                )
    ctrl = np.concatenate(ctrl_rows, axis=0)
    ctrl_b = np.concatenate(ctrl_batch, axis=0)
    half = coexprmod.split_control_cells(ctrl.shape[0], C.SEED)
    print(
        f"[{cell_line}] pass done in {time.time() - t0:.0f}s: {n_kept} cells kept, "
        f"{ctrl.shape[0]} controls, {len(sums)} (target,batch,side) groups",
        flush=True,
    )
    return {
        "control": ctrl,
        "control_batch": ctrl_b,
        "control_half": half,
        "sums": {k: v for k, v in sums.items()},
        "counts": dict(counts),
        "n_targets": targets.size,
        "n_genes": n_genes,
        "n_cells_kept": n_kept,
    }


def control_means(stream: dict, which: str) -> dict[int, np.ndarray]:
    """log2 mean CP10K per batch for one control half (or both)."""
    ctrl, b, half = stream["control"], stream["control_batch"], stream["control_half"]
    sel = np.ones(ctrl.shape[0], dtype=bool) if which == "all" else (half == int(which))
    out: dict[int, np.ndarray] = {}
    for batch in np.unique(b[sel]):
        rows = sel & (b == batch)
        if int(rows.sum()) < MIN_CONTROL_CELLS_PER_BATCH:
            continue
        out[int(batch)] = np.log2(ctrl[rows].mean(axis=0, dtype=np.float64) + EPS_CP10K)
    return out


def assemble(stream: dict, ctrl_log: dict[int, np.ndarray]) -> dict:
    """Within-batch log2 fold change, and the same thing from each half of a
    perturbation's own cells."""
    n_t, n_g = stream["n_targets"], stream["n_genes"]
    acc = {
        key: (np.zeros((n_t, n_g), dtype=np.float64), np.zeros(n_t, dtype=np.float64))
        for key in ("effect", "half0", "half1")
    }
    # the whole-perturbation estimate pools a batch's two sides before the log,
    # so it is the ordinary pseudobulk fold change rather than an average of two
    # noisier ones
    pooled: dict[tuple[int, int], list] = {}
    for (ti, batch, side), total in stream["sums"].items():
        n = stream["counts"][(ti, batch, side)]
        entry = pooled.setdefault((ti, batch), [np.zeros(n_g, dtype=np.float64), 0])
        entry[0] += total
        entry[1] += n
        base = ctrl_log.get(batch)
        if base is None or n < MIN_CELLS_PER_BATCH:
            continue
        lfc = np.log2(total / n + EPS_CP10K) - base
        key = f"half{side}"
        acc[key][0][ti] += n * lfc
        acc[key][1][ti] += n
    for (ti, batch), (total, n) in pooled.items():
        base = ctrl_log.get(batch)
        if base is None or n < MIN_CELLS_PER_BATCH:
            continue
        acc["effect"][0][ti] += n * (np.log2(total / n + EPS_CP10K) - base)
        acc["effect"][1][ti] += n
    out = {}
    for key, (num, den) in acc.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            out[key] = (num / np.maximum(den, 1e-12)[:, None]).astype(np.float32)
        out[key][den == 0] = np.nan
        out[f"{key}_cells"] = den
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-lines", default="hepg2,jurkat")
    ap.add_argument("--work", default="_work", help="scratch dir holding the raw h5ads")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bundle", default=None, help="bundle to check the recomputation against")
    args = ap.parse_args(argv)

    out = paths.ensure(Path(args.out) if args.out else paths.data_root() / "ctrl_split_targets")
    root = Path(args.bundle) if args.bundle else paths.bundle_root()
    report: dict = {"eps_cp10k": EPS_CP10K, "seed": C.SEED, "lines": {}}
    for line in (x.strip() for x in args.cell_lines.split(",")):
        if not line:
            continue
        from . import cells as cellmod  # noqa: PLC0415 - h5py lives behind this

        readout = pd.read_parquet(paths.readout_table(line))
        perts = pd.read_parquet(paths.perturbations_table(line))
        scope = cellmod.in_scope_perturbed(perts, readout, C.PERTURBED_STRATA)
        stream = stream_line(line, Path(args.work) / "_raw", scope, readout)
        bundle_targets = np.load(paths.line_dir(line, root) / "targets.npz", allow_pickle=False)
        if not np.array_equal(
            bundle_targets["pert_target"].astype(str), scope["target"].to_numpy().astype(str)
        ):
            raise RuntimeError(f"{line}: recomputed row order differs from the bundle's")
        rec: dict = {
            "n_cells_kept": stream["n_cells_kept"],
            "n_control_cells": int(stream["control"].shape[0]),
            "n_batches": int(np.unique(stream["control_batch"]).size),
            "targets": {},
        }
        for which, tag in (("all", "ctrlall"), ("0", "ctrl0"), ("1", "ctrl1")):
            ctrl_log = control_means(stream, which)
            built = assemble(stream, ctrl_log)
            eff = np.nan_to_num(built["effect"], nan=0.0)
            np.savez_compressed(
                out / f"{line}__{tag}.npz",
                effect=eff,
                half0=np.nan_to_num(built["half0"], nan=0.0),
                half1=np.nan_to_num(built["half1"], nan=0.0),
                cells_used=built["effect_cells"],
                pert_target=scope["target"].to_numpy().astype(str),
                readout_gene_id=readout["gene_id"].to_numpy().astype(str),
                control_half=np.asarray([which], dtype=object).astype(str),
            )
            agree = metrics.columnwise_pearson(
                eff.T.astype(np.float64),
                bundle_targets["effect"].astype(np.float64).T,
            )
            rel = metrics.per_gene_reliability(
                np.nan_to_num(built["half0"], nan=0.0), np.nan_to_num(built["half1"], nan=0.0)
            )
            rec["targets"][tag] = {
                "n_batches_used": len(ctrl_log),
                "median_agreement_with_run1_effect": float(np.nanmedian(agree)),
                "q10_agreement_with_run1_effect": float(np.nanquantile(agree, 0.10)),
                "median_reliability_weight": float(np.median(rel["weights"])),
                "perturbations_without_cells": int((built["effect_cells"] == 0).sum()),
                "median_abs_effect": float(np.nanmedian(np.abs(eff))),
            }
            print(f"[{line}] {tag}: {json.dumps(rec['targets'][tag])}", flush=True)
        report["lines"][line] = rec
    (out / "recompute_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
