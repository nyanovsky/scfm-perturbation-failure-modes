"""Minimal h5ad readers.

The two GSE264667 matrices are *dense* float32 HDF5 datasets written by an old
anndata version (categoricals live in ``obs/__categories``). We therefore read
them with h5py directly and stream row chunks, which keeps peak memory at one
chunk instead of the 5.6 / 9.4 GB full matrix.
"""

from __future__ import annotations

from typing import Iterator

import h5py
import numpy as np
import pandas as pd


def _decode(arr) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def _categories(h: h5py.File, group: str, col: str) -> np.ndarray:
    return _decode(h[f"{group}/__categories/{col}"][:])


def read_frame(h: h5py.File, group: str) -> pd.DataFrame:
    g = h[group]
    index_name = g.attrs.get("_index", "index")
    if isinstance(index_name, bytes):
        index_name = index_name.decode()
    cats = set(g["__categories"].keys()) if "__categories" in g else set()
    data = {}
    for col in g.keys():
        if col == "__categories":
            continue
        if col in cats:
            values = _categories(h, group, col)[g[col][:]]
        else:
            raw = g[col][:]
            values = _decode(raw) if raw.dtype == object else raw
        data[col] = values
    frame = pd.DataFrame(data)
    frame = frame.set_index(index_name, drop=False)
    frame.index.name = index_name
    return frame


def matrix_shape(h: h5py.File) -> tuple[int, int]:
    return tuple(int(x) for x in h["X"].shape)  # type: ignore[return-value]


def iter_row_chunks(
    h: h5py.File, chunk_rows: int = 8192
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(start, stop, block)`` over X, block dtype float32."""
    x = h["X"]
    n = x.shape[0]
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        yield start, stop, np.asarray(x[start:stop, :], dtype=np.float32)
