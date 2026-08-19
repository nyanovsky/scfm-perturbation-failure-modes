"""Export a few real QC-passing non-targeting control cells.

The model track demonstrates each response interface on a real cell rather than
a synthetic vector. Selection is deterministic: the QC-passing non-targeting
control cells are ordered by barcode and the first ones are taken, so the same
cells are reproduced on a rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .. import config as C
from .. import paths
from . import h5ad_io, sources

N_CELLS = 4


def export(cell_line: str, raw_dir: Path, out_dir: Path) -> dict:
    path = raw_dir / sources.H5AD_FILES[cell_line]
    # the matrix is not part of the delivered worktree; fetch it if this job does
    # not already have it (download() verifies length and reuses a complete file)
    fetch = sources.download(sources.h5ad_url(cell_line), path)
    with h5py.File(path, "r") as h:
        obs = h5ad_io.read_frame(h, "obs")
        var = h5ad_io.read_frame(h, "var")
        target = obs["gene"].to_numpy().astype(str)
        mito = obs["mitopercent"].to_numpy().astype(np.float64)
        barcode = obs.index.to_numpy().astype(str)
        gem = obs["gem_group"].to_numpy().astype(np.int64)

        control = np.flatnonzero(
            (target == C.CONTROL_LABEL) & (mito <= C.CELL_FILTERS.max_mito_fraction)
        )
        order = control[np.argsort(barcode[control], kind="stable")]
        picked: list[int] = []
        counts: list[np.ndarray] = []
        for idx in order:
            row = np.asarray(h["X"][idx : idx + 1, :], dtype=np.float32)[0]
            if row.sum() < C.CELL_FILTERS.min_counts:
                continue
            if (row > 0).sum() < C.CELL_FILTERS.min_genes_detected:
                continue
            picked.append(int(idx))
            counts.append(row)
            if len(picked) >= N_CELLS:
                break

    counts_arr = np.stack(counts)
    out_dir = paths.ensure(out_dir)
    np.savez_compressed(
        out_dir / f"demo_cells_{cell_line}.npz",
        counts=counts_arr,
        gene_ids=var["gene_id"].to_numpy().astype(str),
        gene_names=var["gene_name"].to_numpy().astype(str),
        barcodes=barcode[picked],
        gem_groups=gem[picked],
        row_indices=np.array(picked, dtype=np.int64),
    )
    # Also written as parquet: the delivery snapshot honours .gitignore, which
    # excludes *.npz, so the tabular form is the one that reaches a job.
    import pandas as pd

    frame = pd.DataFrame(
        {
            "gene_id": var["gene_id"].to_numpy().astype(str),
            "gene_name": var["gene_name"].to_numpy().astype(str),
        }
    )
    for j, bc in enumerate(barcode[picked]):
        frame[f"counts__{bc}"] = counts_arr[j]
    frame.to_parquet(out_dir / f"demo_cells_{cell_line}.parquet", index=False)

    meta = {
        "cell_line": cell_line,
        "source": {k: fetch.get(k) for k in ("url", "size_bytes", "md5", "reused_existing")},
        "n_cells": int(counts_arr.shape[0]),
        "n_genes": int(counts_arr.shape[1]),
        "barcodes": barcode[picked].tolist(),
        "gem_groups": gem[picked].tolist(),
        "total_counts": [float(x) for x in counts_arr.sum(1)],
        "genes_detected": [int(x) for x in (counts_arr > 0).sum(1)],
        "selection": (
            "non-targeting control cells passing the run's cell filters, ordered by "
            "barcode, first N taken"
        ),
    }
    print(json.dumps(meta, indent=1), flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw-dir", default="_raw", help="scratch for the raw h5ad (never uploaded)")
    ap.add_argument("--cell-lines", nargs="*", default=list(paths.CELL_LINES))
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else paths.data_root() / "demo_cells"
    raw = paths.ensure(Path(args.raw_dir))
    meta = {cl: export(cl, raw, out) for cl in args.cell_lines}
    (paths.ensure(out) / f"demo_cells_{'_'.join(args.cell_lines)}.json").write_text(
        json.dumps(meta, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
