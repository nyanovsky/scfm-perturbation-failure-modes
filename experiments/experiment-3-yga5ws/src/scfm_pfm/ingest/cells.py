"""The non-targeting control cells, filtered exactly as run 1 filtered them.

Run 1 streamed the raw GSE264667 matrices to build the effect matrices but kept
only pooled control means, so the per-cell control matrix that a co-expression
baseline needs does not exist as an artifact. It is cheap to rebuild: the same
two passes, the same filter funnel, and only the control rows are retained
(4,916 cells in HepG2), restricted to the read-out gene set.
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..data import h5ad_io, sources

CHUNK_ROWS = 4096
CONTROL_LABEL = "non-targeting"
MIN_COUNTS = 2000
MIN_GENES_DETECTED = 500
MAX_MITO_FRACTION = 0.15
NORM_TARGET = 1.0e4


def load_control_cells(cell_line: str, raw_dir: Path, matrix_cols: np.ndarray) -> dict:
    """CP10K control-cell matrix over ``matrix_cols``, plus the filter funnel."""
    url = sources.h5ad_url(cell_line)
    dest = Path(raw_dir) / sources.H5AD_FILES[cell_line]
    dl = sources.download(url, dest)
    print(f"[{cell_line}] raw {dl['size_bytes'] / 1e9:.2f} GB md5={dl['md5']}", flush=True)

    with h5py.File(dest, "r") as h:
        n_cells, n_genes = h5ad_io.matrix_shape(h)
        obs = h5ad_io.read_frame(h, "obs")
        var = h5ad_io.read_frame(h, "var")

        target_label = obs["gene"].to_numpy().astype(str)
        guide = obs["sgID_AB"].to_numpy().astype(str)
        mito = obs["mitopercent"].to_numpy().astype(np.float64)
        batch = obs["gem_group"].to_numpy().astype(np.int64)

        is_control = target_label == CONTROL_LABEL
        single = (
            (target_label != "")
            & (target_label != "nan")
            & (guide != "")
            & (guide != "nan")
        )

        n_counts = np.zeros(n_cells, dtype=np.float64)
        n_detected = np.zeros(n_cells, dtype=np.int32)
        t0 = time.time()
        for start, stop, block in h5ad_io.iter_row_chunks(h, CHUNK_ROWS):
            n_counts[start:stop] = block.sum(axis=1)
            n_detected[start:stop] = (block > 0).sum(axis=1)
        print(f"[{cell_line}] pass 1 in {time.time() - t0:.0f}s", flush=True)

        keep = (
            single
            & (n_counts >= MIN_COUNTS)
            & (n_detected >= MIN_GENES_DETECTED)
            & (mito <= MAX_MITO_FRACTION)
        )
        want = keep & is_control
        n_want = int(want.sum())
        cp10k = np.zeros((n_want, matrix_cols.size), dtype=np.float32)
        cursor = 0
        t0 = time.time()
        for start, stop, block in h5ad_io.iter_row_chunks(h, CHUNK_ROWS):
            sel = want[start:stop]
            if not sel.any():
                continue
            rows = block[sel][:, matrix_cols]
            scale = (NORM_TARGET / np.maximum(n_counts[start:stop][sel], 1.0)).astype(np.float32)
            cp10k[cursor : cursor + rows.shape[0]] = rows * scale[:, None]
            cursor += rows.shape[0]
        print(
            f"[{cell_line}] pass 2 kept {cursor} control cells in {time.time() - t0:.0f}s",
            flush=True,
        )

    return {
        "cp10k": cp10k,
        "batch": batch[want],
        "n_counts": n_counts[want],
        "gene_ids_matrix": (
            var["gene_id"].to_numpy().astype(str)
            if "gene_id" in var.columns
            else var.index.to_numpy().astype(str)
        ),
        "download": dl,
        "funnel": {
            "raw_cells": int(n_cells),
            "raw_genes": int(n_genes),
            "control_cells_raw": int(is_control.sum()),
            "control_cells_kept": int(n_want),
            "removed_single_target": int((is_control & ~single).sum()),
            "removed_min_counts": int((is_control & single & (n_counts < MIN_COUNTS)).sum()),
            "removed_min_genes": int(
                (is_control & single & (n_detected < MIN_GENES_DETECTED)).sum()
            ),
            "removed_mito": int((is_control & single & (mito > MAX_MITO_FRACTION)).sum()),
        },
    }


def in_scope_perturbed(
    perts: pd.DataFrame, readout: pd.DataFrame, strata: tuple[str, ...]
) -> pd.DataFrame:
    """Run 2's scope rule, reproduced so full-coverage row order is recomputable.

    A perturbed gene has to be a column of the supplied set to be perturbable at
    all, so the scope is the intersection of the effect-size strata with the
    read-out set. Row order follows the perturbation table, which is the order
    run 2's full-coverage blocks were written in.
    """
    by_symbol: dict[str, int] = {}
    for col in ("gene_name", "hgnc_symbol"):
        if col in readout.columns:
            for i, sym in enumerate(readout[col].to_numpy().astype(str)):
                by_symbol.setdefault(sym, i)
    rows = []
    for rec in perts.itertuples():
        if str(rec.effect_stratum) not in strata:
            continue
        col = by_symbol.get(str(rec.target))
        if col is None:
            continue
        rows.append(
            {
                "target": str(rec.target),
                "readout_col": int(col),
                "gene_id": str(readout["gene_id"].to_numpy()[col]),
                "effect_stratum": str(rec.effect_stratum),
                "n_de_genes": int(getattr(rec, "n_de_genes", 0)),
                "knockdown": float(getattr(rec, "knockdown", float("nan"))),
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates("readout_col").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("no perturbed genes survive the scope intersection")
    return frame
