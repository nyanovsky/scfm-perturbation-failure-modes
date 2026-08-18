"""Condense the effect matrices into the slice the cross-check needs.

The cross-check job streams a 29.8 GB reference table, so it has to run on the
cluster, but a finished job's outputs do not stay on its volume: they live in the
artifact store. Rather than shipping two 423 MB `effects.npz` files into that
job, this module builds the small thing the comparison actually reads -- the
log2 fold changes over expressed genes only -- and computes the
within-batch-vs-pooled agreement here, where both matrices are already at hand.

    uv run python -m scfm_pfm.data.crosscheck_input --root results/artifacts_mirror
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..stats import pearson, spearman

FILENAME = "crosscheck_input_{cell_line}.npz"
AGREEMENT = "own_estimator_agreement_{cell_line}.parquet"


def build(cell_line: str, effects_dir: Path, out_dir: Path) -> dict:
    npz = np.load(effects_dir / "effects.npz", allow_pickle=False)
    pert = pd.read_parquet(effects_dir / "perturbations.parquet")
    expressed = npz["expressed"].astype(bool)
    cols = np.flatnonzero(expressed).astype(np.int64)
    wb, bp = npz["lfc_within_batch"], npz["lfc_pooled"]
    dwb, dbp = npz["delta_within_batch"], npz["delta_pooled"]
    targets = np.asarray(npz["targets"])

    rows = []
    for i, target in enumerate(targets):
        rows.append(
            {
                "cell_line": cell_line,
                "target": str(target),
                "pearson_lfc_within_vs_pooled": pearson(wb[i][cols], bp[i][cols]),
                "spearman_lfc_within_vs_pooled": spearman(wb[i][cols], bp[i][cols]),
                "pearson_delta_within_vs_pooled": pearson(dwb[i][cols], dbp[i][cols]),
                "l2_within": float(np.linalg.norm(wb[i][cols])),
                "l2_pooled": float(np.linalg.norm(bp[i][cols])),
            }
        )
    agree = pd.DataFrame(rows).merge(
        pert[["target", "effect_stratum", "effect_l2_lfc", "n_cells", "knockdown"]],
        on="target",
        how="left",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    agree.to_parquet(out_dir / AGREEMENT.format(cell_line=cell_line), index=False)
    np.savez_compressed(
        out_dir / FILENAME.format(cell_line=cell_line),
        targets=targets,
        gene_index=cols,
        lfc_within_batch=wb[:, cols].astype(np.float32),
    )
    meta = {
        "cell_line": cell_line,
        "perturbations": int(targets.size),
        "expressed_genes": int(cols.size),
        "source_matrix_genes": int(expressed.size),
        "median_pearson_lfc_within_vs_pooled": float(
            agree.pearson_lfc_within_vs_pooled.median()
        ),
        "median_pearson_delta_within_vs_pooled": float(
            agree.pearson_delta_within_vs_pooled.median()
        ),
    }
    print(json.dumps(meta), flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="directory holding effects/<cell_line>/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else paths.repo_dir() / "results" / "handoff" / "crosscheck"
    meta = {
        cl: build(cl, root / "effects" / cl, out) for cl in paths.CELL_LINES
    }
    (out / "crosscheck_input.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
