"""A stand-in demo cell for interactive adapter debugging only.

The published response vectors are computed on a real QC-passing non-targeting
control cell (``data/demo_cells.py``). This file exists so the four model
adapters can be exercised in an interactive session before the real cell's npz
has flowed back from the data job; it is never used for a recorded artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-line", default="hepg2")
    args = ap.parse_args()

    uni = pd.read_parquet(Path(args.universe) / "gene_universe.parquet")
    gene_ids = uni.gene_id.to_numpy().astype(str)
    gene_names = uni.gene_name.to_numpy().astype(str)
    rng = np.random.default_rng(42)
    # a plausible single-cell count profile: heavy-tailed, ~40% zeros
    counts = rng.lognormal(mean=0.4, sigma=1.5, size=(4, gene_ids.size))
    counts = np.where(rng.random(counts.shape) < 0.4, 0.0, counts)
    counts = np.rint(counts).astype(np.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"demo_cells_{args.cell_line}.npz",
        counts=counts,
        gene_ids=gene_ids,
        gene_names=gene_names,
        barcodes=np.array([f"SYNTHETIC-{i}" for i in range(counts.shape[0])]),
        gem_groups=np.zeros(counts.shape[0], dtype=np.int64),
        row_indices=np.arange(counts.shape[0]),
    )
    print(
        f"wrote {out}/demo_cells_{args.cell_line}.npz "
        f"cells={counts.shape[0]} genes={counts.shape[1]} "
        f"total_counts={counts.sum(1).astype(int).tolist()} "
        f"detected={(counts > 0).sum(1).tolist()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
