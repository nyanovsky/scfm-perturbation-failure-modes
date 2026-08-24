"""Build the per-cell matrices scGPT's perturbation recipe trains on.

Runs 1 and 3 kept pseudobulk effects and pooled control means; fine-tuning
needs the cells themselves. This is one streaming pass over the raw GEO matrix
per cell line, with run 3's filter funnel applied unchanged (the same minimum
counts, minimum detected genes, mitochondrial fraction and single-guide rule),
restricted to the same read-out genes and the same in-scope perturbations the
bundle carries. Two products:

``control_cells``   every surviving non-targeting cell, CP10K over the read-out
                    genes. This is the input pool a perturbed cell is paired
                    with, and the mean control profile every query is anchored
                    at.
``pert_cells``      up to ``MAX_CELLS_PER_PERT`` cells per in-scope
                    perturbation, drawn by seeded reservoir so a heavily
                    sampled target cannot dominate the loss.

The fold assignment is computed here too, from the bundle's own effect matrix
through run 3's split code, so the perturbations withheld from fine-tuning are
exactly the ones run 3 held out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import paths
from .sibling import run3


def fold_assignment(cell_line: str, bundle_root: Path) -> dict:
    """Run 3's grouped folds over the in-scope perturbations, recomputed.

    ``make_folds`` is a pure function of the effect matrix and the seed, so this
    reproduces the fold every grouped run of run 3's sweep used rather than
    inventing a new one.
    """
    C3 = run3("config")
    splitsmod = run3("translate.splits")
    t = np.load(Path(bundle_root) / cell_line / "targets.npz", allow_pickle=False)
    effect = t["effect"]
    info = splitsmod.make_folds(
        effect, "grouped", C3.N_FOLDS, C3.SEED, C3.GROUP_CORR_THRESHOLD
    )
    fold = info["fold"]
    heldout = np.isin(fold, np.asarray(C.MODEL_HELDOUT_FOLDS))
    return {
        "pert_target": t["pert_target"].astype(str),
        "pert_gene_id": t["pert_gene_id"].astype(str),
        "fold": fold.astype(np.int64),
        "groups": info["groups"].astype(np.int64),
        "model_heldout": heldout,
        "n_groups": int(info["n_groups"]),
        "fold_sizes": [int(x) for x in info["fold_sizes"]],
        "largest_group": int(info["largest_group"]),
    }


def training_split(folds: dict, seed: int, heldout_folds=None) -> dict:
    """Split the fine-tuning perturbations into train and validation.

    Whole response groups move together, and no group that contributes a scored
    perturbation can appear in either, so early stopping never sees a held-out
    outcome.

    ``heldout_folds`` overrides which of run 3's folds are withheld from
    fine-tuning. The default is the pre-registered pair; the other pairs exist
    so the same arm can be re-fitted with a different pair withheld and every
    perturbation in the screen scored by a model that never saw it.
    """
    rng = np.random.default_rng(seed)
    fold, groups = folds["fold"], folds["groups"]
    if heldout_folds is None:
        heldout = folds["model_heldout"]
    else:
        heldout = np.isin(fold, np.asarray(heldout_folds, dtype=np.int64))
    folds = {**folds, "model_heldout": heldout}
    trainable = ~heldout
    # A response group is held out whole by run 3's grouped split, so no group
    # may straddle the fine-tuning boundary. If one did, a near-identical
    # perturbation would sit in training while its partner was scored, and the
    # held-out number would not be held out. Checked rather than assumed.
    straddling = np.intersect1d(np.unique(groups[trainable]), np.unique(groups[~trainable]))
    if straddling.size:
        raise RuntimeError(
            f"{straddling.size} response groups span the fine-tuning boundary: "
            f"{straddling[:10].tolist()}"
        )
    g_train = np.unique(groups[trainable])
    n_val = max(1, int(round(C.VAL_GROUP_FRACTION * g_train.size)))
    val_groups = set(rng.permutation(g_train)[:n_val].tolist())
    is_val = trainable & np.array([g in val_groups for g in groups])
    is_train = trainable & ~is_val
    return {
        "is_train": is_train,
        "is_val": is_val,
        "is_scored": folds["model_heldout"].copy(),
        "n_train": int(is_train.sum()),
        "n_val": int(is_val.sum()),
        "n_scored": int(folds["model_heldout"].sum()),
        "n_train_groups": int(np.unique(groups[is_train]).size),
        "n_val_groups": int(np.unique(groups[is_val]).size),
        "n_scored_groups": int(np.unique(groups[folds["model_heldout"]]).size),
    }


class Reservoir:
    """Algorithm R, one per perturbation, so the pass stays single and bounded."""

    def __init__(self, cap: int, n_genes: int, seed: int):
        self.cap = int(cap)
        self.buf = np.zeros((cap, n_genes), dtype=np.float32)
        self.batch = np.zeros(cap, dtype=np.int64)
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def offer(self, row: np.ndarray, batch: int) -> None:
        if self.seen < self.cap:
            self.buf[self.seen] = row
            self.batch[self.seen] = batch
        else:
            j = int(self.rng.integers(0, self.seen + 1))
            if j < self.cap:
                self.buf[j] = row
                self.batch[j] = batch
        self.seen += 1

    @property
    def n(self) -> int:
        return min(self.seen, self.cap)


def stream_line(cell_line: str, raw_dir: Path, readout: pd.DataFrame, targets: np.ndarray) -> dict:
    """One pass over the raw matrix: control rows kept, perturbed rows sampled."""
    import h5py

    h5ad_io = run3("data.h5ad_io")
    sources = run3("data.sources")
    cellmod = run3("ingest.cells")

    matrix_cols = readout[f"{cell_line}_matrix_index"].to_numpy().astype(np.int64)
    dest = Path(raw_dir) / sources.H5AD_FILES[cell_line]
    if not dest.exists():
        dl = sources.download(sources.h5ad_url(cell_line), dest)
        print(f"[{cell_line}] raw {dl['size_bytes'] / 1e9:.2f} GB md5={dl['md5']}", flush=True)

    t_index = {t: i for i, t in enumerate(targets.astype(str))}
    n_genes = matrix_cols.size
    reservoirs = [
        Reservoir(C.MAX_CELLS_PER_PERT, n_genes, C.SEED + 1000 + i) for i in range(targets.size)
    ]

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
        single = (label != "") & (label != "nan") & (guide != "") & (guide != "nan")

        ctrl_rows: list[np.ndarray] = []
        ctrl_batch: list[np.ndarray] = []
        funnel = {"total": int(n_cells), "kept": 0, "control_kept": 0, "pert_in_scope": 0}
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
            funnel["kept"] += int(keep.sum())
            scale = (cellmod.NORM_TARGET / np.maximum(n_counts, 1.0)).astype(np.float32)
            sub = block[:, matrix_cols] * scale[:, None]
            csel = keep & is_control[start:stop]
            if csel.any():
                ctrl_rows.append(sub[csel].astype(np.float32))
                ctrl_batch.append(batch[start:stop][csel])
                funnel["control_kept"] += int(csel.sum())
            psel = keep & ~is_control[start:stop]
            if psel.any():
                idx = np.flatnonzero(psel)
                labs = label[start:stop][psel]
                bs = batch[start:stop][psel]
                for k, lab in enumerate(labs):
                    ti = t_index.get(str(lab))
                    if ti is None:
                        continue
                    reservoirs[ti].offer(sub[idx[k]], int(bs[k]))
                    funnel["pert_in_scope"] += 1
            if (start // cellmod.CHUNK_ROWS) % 40 == 0:
                print(f"[{cell_line}] {stop}/{n_cells} cells in {time.time() - t0:.0f}s", flush=True)

    control = np.concatenate(ctrl_rows, axis=0)
    control_batch = np.concatenate(ctrl_batch, axis=0)
    counts = np.array([r.seen for r in reservoirs], dtype=np.int64)
    kept = np.array([r.n for r in reservoirs], dtype=np.int64)
    print(
        f"[{cell_line}] pass done in {time.time() - t0:.0f}s: {funnel['kept']} cells kept, "
        f"{control.shape[0]} controls, {int(kept.sum())} perturbed cells sampled "
        f"from {int(counts.sum())} in scope",
        flush=True,
    )
    return {
        "control": control,
        "control_batch": control_batch,
        "reservoirs": reservoirs,
        "cells_available": counts,
        "cells_kept": kept,
        "funnel": funnel,
    }


def build(cell_line: str, raw_dir: Path, bundle_root: Path, out_root: Path) -> dict:
    readout = pd.read_parquet(paths.readout_table(cell_line))
    folds = fold_assignment(cell_line, bundle_root)
    split = training_split(folds, C.SEED)
    stream = stream_line(cell_line, raw_dir, readout, folds["pert_target"])

    kept = stream["cells_kept"]
    enough = kept >= C.MIN_CELLS_PER_PERT
    rows, row_target = [], []
    row_batch = []
    for i, res in enumerate(stream["reservoirs"]):
        if not enough[i]:
            continue
        rows.append(res.buf[: res.n])
        row_target.append(np.full(res.n, i, dtype=np.int64))
        row_batch.append(res.batch[: res.n])
    X = np.concatenate(rows, axis=0) if rows else np.zeros((0, readout.shape[0]), dtype=np.float32)
    pert_of_row = np.concatenate(row_target) if rows else np.zeros(0, dtype=np.int64)
    batch_of_row = np.concatenate(row_batch) if rows else np.zeros(0, dtype=np.int64)

    out = paths.ensure(Path(out_root) / cell_line)
    np.savez(
        out / "control_cells.npz",
        X=stream["control"].astype(np.float32),
        gem_group=stream["control_batch"].astype(np.int64),
    )
    np.savez(
        out / "pert_cells.npz",
        X=X.astype(np.float32),
        pert_index=pert_of_row,
        gem_group=batch_of_row,
    )
    np.savez(
        out / "perturbation_index.npz",
        pert_target=folds["pert_target"],
        pert_gene_id=folds["pert_gene_id"],
        fold=folds["fold"],
        groups=folds["groups"],
        model_heldout=folds["model_heldout"],
        is_train=split["is_train"],
        is_val=split["is_val"],
        is_scored=split["is_scored"],
        cells_available=stream["cells_available"],
        cells_kept=kept,
        usable=enough,
    )
    readout.to_parquet(out / "readout_genes.parquet")

    report = {
        "cell_line": cell_line,
        "n_readout_genes": int(readout.shape[0]),
        "n_perturbations": int(folds["pert_target"].size),
        "n_groups": folds["n_groups"],
        "fold_sizes": folds["fold_sizes"],
        "largest_group": folds["largest_group"],
        "model_heldout_folds": list(C.MODEL_HELDOUT_FOLDS),
        **{k: v for k, v in split.items() if not isinstance(v, np.ndarray)},
        "n_perturbations_with_too_few_cells": int((~enough).sum()),
        "min_cells_per_pert": C.MIN_CELLS_PER_PERT,
        "max_cells_per_pert": C.MAX_CELLS_PER_PERT,
        "cells_per_pert_median_available": float(np.median(stream["cells_available"])),
        "cells_per_pert_median_kept": float(np.median(kept)),
        "n_train_cells": int(np.isin(pert_of_row, np.flatnonzero(split["is_train"])).sum()),
        "n_val_cells": int(np.isin(pert_of_row, np.flatnonzero(split["is_val"])).sum()),
        "n_scored_cells": int(np.isin(pert_of_row, np.flatnonzero(split["is_scored"])).sum()),
        "n_control_cells": int(stream["control"].shape[0]),
        "filter_funnel": stream["funnel"],
        "seed": C.SEED,
    }
    (out / "ingest_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    ap.add_argument("--raw", default=None, help="where the GEO h5ad is cached")
    ap.add_argument("--bundle", default=None, help="run 3's bundle root")
    ap.add_argument("--out", default=None, help="where the cell matrices go")
    a = ap.parse_args(argv)
    raw = Path(a.raw) if a.raw else paths.ensure(Path("_raw"))
    bundle = Path(a.bundle) if a.bundle else paths.bundle_root()
    out = Path(a.out) if a.out else paths.ensure(paths.data_root() / "cells")
    build(a.cell_line, raw, bundle, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
