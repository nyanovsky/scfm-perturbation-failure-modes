"""Measure what excluding no-op cells actually does to the feature.

At the individual-cells anchor a query is run in 4 real control cells and the
stored response is the average over them. When the perturbed gene already
carried the value being asked for in one of those cells, that cell's two
forward passes were the same call and it contributes an exact zero, so the
stored average is diluted. Run 2's guidance is to exclude those queries rather
than average their zeros in, and the earlier build of this run implemented that
by deleting the whole dose for every perturbation, which threw away thousands
of valid values.

The correct exclusion is per (perturbed gene, value): with ``k`` of ``n`` cells
no-op, the stored row is ``(n - k) / n`` times the average over the cells that
were really queried, so the correction is a per-query positive scale factor.
The feature every arm reads is the per-query standardized form, z-scored across
read-out genes, which is invariant to that factor -- but only if the two things
the argument rests on are true of the shipped arrays. This step measures them
instead of assuming them, on run 2's own files:

1. the stored ``standardized`` array is the per-query z-score of ``raw``;
2. the stored ``raw`` row is the plain unweighted mean over the cells, checked
   against the per-cell cross-gene means in the block's query record;
3. the only rows whose z-score is undefined are those where every cell was a
   no-op, and the shipped array is zero there;
4. applying the correction and re-standardizing reproduces the shipped feature.

Writes one record per block to ``results/verification/no_op_correction.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import no_op as no_op_mod

TOL_ZSCORE = 1e-4  # arrays are stored float32
#: the plain-mean identity is measured as a fraction of the query's own spread
#: over read-out genes, because two of the four heads end in a softmax and
#: their cross-gene mean is zero by construction, which makes a relative error
#: against it meaningless
TOL_MEAN = 1e-7


def _zscore_rows(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-query z-score across read-out genes; also returns the defined mask."""
    return no_op_mod.standardize_rows(a), a.std(axis=-1) > 0


def check_block(model: str, line: str, anchor: str, vvp: Path, qs: Path) -> dict:
    z = np.load(vvp, allow_pickle=False)
    raw = z["raw"].astype(np.float64)
    std = z["standardized"].astype(np.float64)
    pid = z["perturbed_gene_id"].astype(str)
    values = z["value_grid_cp10k"].astype(np.float64)

    d = pd.read_parquet(qs)
    g = no_op_mod.per_gene_counts(d).rename(columns={"perturbed_gene_id": "perturbed_gene"})
    g = g.merge(
        d.groupby(["perturbed_gene", "value_cp10k"], as_index=False).agg(
            cell_mean=("cross_gene_mean", "mean")
        ),
        on=["perturbed_gene", "value_cp10k"],
    )
    row = {p: i for i, p in enumerate(pid)}
    col = {float(v): i for i, v in enumerate(values)}
    gi = np.array([row[p] for p in g.perturbed_gene], dtype=np.int64)
    vi = np.array([col[float(v)] for v in g.value_cp10k], dtype=np.int64)

    n_cells = np.zeros(raw.shape[:2], dtype=np.int64)
    n_no_op = np.zeros(raw.shape[:2], dtype=np.int64)
    n_cells[gi, vi] = g.n_cells.to_numpy()
    n_no_op[gi, vi] = g.n_no_op.to_numpy().astype(np.int64)
    if (n_cells == 0).any():
        raise RuntimeError(f"{model}/{line}/{anchor}: query record misses some (gene, value)")

    # 1. the standardized array is the per-query z-score of raw
    zz, defined = _zscore_rows(raw)
    d_zscore = float(np.abs(zz - std).max())

    # 2. the raw row is the plain unweighted mean over the cells
    sub = raw[gi, vi, :]
    stored = sub.mean(axis=1)
    spread = sub.std(axis=1)
    want = g.cell_mean.to_numpy()
    d_mean = float((np.abs(stored - want) / np.maximum(spread, 1e-300)).max())

    # 3. undefined rows are exactly the everywhere-no-op ones
    all_no_op = n_no_op == n_cells
    same = bool((~defined == all_no_op).all())
    zero_there = float(np.abs(std[all_no_op]).max()) if all_no_op.any() else 0.0

    # 4. correcting the dilution and re-standardizing reproduces the feature
    keep = np.maximum(n_cells - n_no_op, 0)
    factor = np.divide(
        n_cells.astype(np.float64), keep, out=np.zeros(raw.shape[:2]), where=keep > 0
    )
    corrected = raw * factor[:, :, None]
    zc, _ = _zscore_rows(corrected)
    partial = (n_no_op > 0) & ~all_no_op
    d_partial = float(np.abs(zc - std)[partial].max()) if partial.any() else 0.0
    d_all = float(np.abs(zc - std)[~all_no_op].max())

    # what the earlier whole-dose rule would have deleted
    dropped = [float(v) for j, v in enumerate(values) if n_no_op[:, j].sum() > 0]

    checks = {
        "standardized_is_per_query_zscore": (d_zscore, TOL_ZSCORE),
        "raw_is_plain_cell_mean": (d_mean, TOL_MEAN),
        "undefined_rows_are_everywhere_no_op": (0.0 if same else 1.0, 0.5),
        "shipped_feature_zero_on_undefined_rows": (zero_there, 0.0),
        "correction_reproduces_shipped_feature": (d_all, TOL_ZSCORE),
    }
    return {
        "model": model,
        "cell_line": line,
        "anchor": anchor,
        "n_perturbed": int(raw.shape[0]),
        "n_values": int(raw.shape[1]),
        "n_readout": int(raw.shape[2]),
        "cells_per_query": int(np.median(n_cells)),
        "n_no_op_queries": int(n_no_op.sum()),
        "n_query_rows_touched": int((n_no_op > 0).sum()),
        "n_query_rows_everywhere_no_op": int(all_no_op.sum()),
        "max_correction_factor": float(factor[partial].max()) if partial.any() else 1.0,
        "median_correction_factor_where_touched": (
            float(np.median(factor[partial])) if partial.any() else 1.0
        ),
        "max_abs_feature_change_partial_rows": d_partial,
        "values_the_whole_dose_rule_deleted": dropped,
        "checks": {
            name: {"value": val, "tolerance": tol, "pass": bool(val <= tol)}
            for name, (val, tol) in checks.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--block",
        action="append",
        required=True,
        metavar="MODEL:LINE:ANCHOR:VVP:QUERY_SUMMARY",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    records = []
    for spec in args.block:
        model, line, anchor, vvp, qs = spec.split(":", 4)
        rec = check_block(model, line, anchor, Path(vvp), Path(qs))
        records.append(rec)
        bad = [k for k, v in rec["checks"].items() if not v["pass"]]
        print(
            f"{model}/{line}/{anchor}: {rec['n_no_op_queries']} no-op queries, "
            f"{rec['n_query_rows_touched']} query rows corrected, "
            f"largest feature change {rec['max_abs_feature_change_partial_rows']:.3g}, "
            f"{len(rec['checks']) - len(bad)}/{len(rec['checks'])} checks pass"
            + (f" FAILED: {bad}" if bad else ""),
            flush=True,
        )
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"blocks": records}, indent=1) + "\n")
    print(f"-> {dest}")
    return 0 if all(v["pass"] for r in records for v in r["checks"].values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
