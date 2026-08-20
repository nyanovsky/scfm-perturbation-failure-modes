"""No-op queries: which ones they are, and how the feature is corrected.

A no-op query is one where the value the query asks for is the value the input
already carried, so the two forward passes were the same call and the response
is an exact zero rather than a small effect. It happens at the individual-cells
anchor, where the response is averaged over 4 real control cells and a gene
that already reads zero in one of them is unchanged by being set to zero.

Run 2 ships two records of this. ``report/no_op_queries.csv`` counts them per
(model, cell line, anchor, value), and each block's ``query_summary.parquet``
has one row per (cell, perturbed gene, value), which is where that count comes
from. The aggregate is not enough to act on: a no-op belongs to one cell and
one perturbed gene, so a count per value cannot say which queries to exclude,
and excluding the whole dose would delete the real responses of every other
perturbation at that value.

This module derives the table the ingest joins on -- one row per (perturbed
gene, value) with the number of cells queried and the number that were no-ops --
and holds the correction itself. With ``k`` of ``n`` cells no-op, the stored
row is ``(n - k) / n`` times the average over the cells that were really
queried, because a no-op cell contributes an exact zero, so excluding them is
multiplication by ``n / (n - k)``. Both halves of that statement are measured
on run 2's own arrays by ``scfm_pfm.ingest.no_op_check`` rather than assumed.

Run as a CLI it writes the committed table:

    model, cell_line, anchor, perturbed_gene_id, value_cp10k, n_cells, n_no_op

A no-op query is identified exactly as run 2 defines it: the response is
identically zero, which is the row whose ``max_abs`` and ``cross_gene_sd`` are
both exactly zero. The counts this reproduces are checked against the shipped
aggregate before the table is written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def per_gene_counts(qs: pd.DataFrame) -> pd.DataFrame:
    """One row per (perturbed gene, value): cells queried, cells that were no-ops."""
    d = qs.copy()
    d["is_no_op"] = (d["max_abs"] == 0.0) & (d["cross_gene_sd"] == 0.0)
    g = (
        d.groupby(["perturbed_gene", "value_cp10k"], as_index=False)
        .agg(n_cells=("cell_index", "nunique"), n_no_op=("is_no_op", "sum"))
        .rename(columns={"perturbed_gene": "perturbed_gene_id"})
    )
    g["n_no_op"] = g["n_no_op"].astype(int)
    return g


def correction_factors(
    per_gene: pd.DataFrame,
    model: str,
    line: str,
    anchor: str,
    pert_ids: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Per (perturbed gene, value) factor that undoes the no-op dilution.

    A query that was a no-op in every cell carries no measurement at all and
    gets a factor of 0; everything else, including every perturbation that had
    no no-op at that value, is left alone.
    """
    sel = per_gene[
        (per_gene.model == model)
        & (per_gene.cell_line == line)
        & (per_gene.anchor == anchor)
    ]
    row = {g: i for i, g in enumerate(pert_ids)}
    col = {float(v): i for i, v in enumerate(values)}
    factor = np.ones((len(pert_ids), len(values)), dtype=np.float64)
    for rec in sel.itertuples():
        i = row.get(str(rec.perturbed_gene_id))
        if i is None:  # a gene this block does not cover
            continue
        j = col[float(rec.value_cp10k)]
        keep = int(rec.n_cells) - int(rec.n_no_op)
        factor[i, j] = int(rec.n_cells) / keep if keep > 0 else 0.0
    return factor


def standardize_rows(a: np.ndarray) -> np.ndarray:
    """Run 2's per-query standardization: z-score across read-out genes."""
    m = a.mean(axis=-1, keepdims=True)
    s = a.std(axis=-1, keepdims=True)
    return np.divide(a - m, s, out=np.zeros_like(a), where=s > 0)


def zero_index(values: np.ndarray) -> int:
    hit = np.flatnonzero(np.isclose(values, 0.0))
    if hit.size != 1:
        raise RuntimeError(f"expected exactly one 0 CP10K value in the grid, got {values}")
    return int(hit[0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--query-summary",
        action="append",
        required=True,
        metavar="MODEL:LINE:ANCHOR:PATH",
        help="a run 2 per-query record and the block it belongs to",
    )
    ap.add_argument("--aggregate", required=True, help="run 2 no_op_queries.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    agg = pd.read_csv(args.aggregate)
    frames: list[pd.DataFrame] = []
    for spec in args.query_summary:
        model, line, anchor, path = spec.split(":", 3)
        g = per_gene_counts(pd.read_parquet(path))
        g.insert(0, "anchor", anchor)
        g.insert(0, "cell_line", line)
        g.insert(0, "model", model)

        # the derived counts have to reproduce the shipped aggregate exactly
        want = agg[(agg.model == model) & (agg.cell_line == line) & (agg.anchor == anchor)]
        got = g.groupby("value_cp10k").n_no_op.sum()
        for rec in want.itertuples():
            mine = int(got.get(float(rec.value_cp10k), 0))
            if mine != int(rec.n_no_op_queries):
                raise RuntimeError(
                    f"{model}/{line}/{anchor} at {rec.value_cp10k} CP10K: derived "
                    f"{mine} no-op queries, run 2 recorded {rec.n_no_op_queries}"
                )
            total = int(g[g.value_cp10k == float(rec.value_cp10k)].n_cells.sum())
            if total != int(rec.n_queries):
                raise RuntimeError(
                    f"{model}/{line}/{anchor} at {rec.value_cp10k} CP10K: derived "
                    f"{total} queries, run 2 recorded {rec.n_queries}"
                )
        n_full = int((g.n_no_op == g.n_cells).sum())
        print(
            f"{model}/{line}/{anchor}: {int(g.n_no_op.sum())} no-op queries over "
            f"{int(g.n_cells.sum())}, touching {int((g.n_no_op > 0).sum())} "
            f"(gene, value) pairs, {n_full} of them in every cell",
            flush=True,
        )
        frames.append(g[g.n_no_op > 0])

    out = pd.concat(frames, ignore_index=True)
    out = out[
        ["model", "cell_line", "anchor", "perturbed_gene_id", "value_cp10k", "n_cells", "n_no_op"]
    ].sort_values(["model", "cell_line", "anchor", "value_cp10k", "perturbed_gene_id"])
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"{len(out)} rows -> {dest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
