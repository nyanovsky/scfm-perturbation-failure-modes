"""End-to-end smoke of the top-k ceiling grid on the synthetic h5ad.

Runs the real entrypoints -- ``prepare.run`` (which now goes through the
extracted ``prepare.stage``) and then ``ceilings_grid.run`` against the tables
that first run wrote -- and asserts the properties the grid has to have:

* the recompute reproduces the published ceiling columns exactly (same split,
  same estimator, same metric code), which is the check that says the raw h5ads
  were re-read into the same state;
* the k=50 column of the grid *is* the ``topk_overlap`` column already reported;
* chance = k/n, excess = (overlap - chance) / (1 - chance), adaptive k is the
  clipped DE count;
* the persisted split-half vectors reproduce the overlap they were scored with,
  so a later run can ask a new question of the file instead of the cluster.

    PYTHONPATH=src SCFM_PFM_DATA_ROOT=/tmp/gridsmoke apy scratch/smoke_ceilings_grid.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scfm_pfm import config as C  # noqa: E402
from scfm_pfm import paths  # noqa: E402
from scfm_pfm.data import ceilings_grid, prepare, sources  # noqa: E402
from smoke_prepare import write_fake  # noqa: E402


def main() -> int:
    root = Path(os.environ["SCFM_PFM_DATA_ROOT"])
    raw = paths.ensure(paths.raw_dir())
    dest = raw / sources.H5AD_FILES["hepg2"]
    if not dest.exists():
        write_fake(dest)
        # size_verified is what tells sources.download the file is complete; without
        # it the downloader decides the planted file is a truncated transfer and
        # pulls the real 5.6 GB h5ad over the top of it.
        dest.with_suffix(dest.suffix + ".meta.json").write_text(
            json.dumps(
                {
                    "url": "synthetic",
                    "size_bytes": dest.stat().st_size,
                    "declared_size_bytes": dest.stat().st_size,
                    "size_verified": True,
                    "md5": "synthetic",
                }
            )
        )
    object.__setattr__(C.PERTURBATION_FILTERS, "min_cells", 20)

    eff = root / "effects" / "hepg2"
    if not (eff / "perturbations.parquet").exists():
        prepare.run("hepg2", eff, raw)
    qc = ceilings_grid.run("hepg2", eff, raw)

    pert = pd.read_parquet(eff / "perturbations.parquet")
    grid = pd.read_parquet(eff / "topk_overlap_grid.parquet")
    summ = pd.read_parquet(eff / "topk_overlap_grid_summary.parquet")
    npz = np.load(eff / "split_half_vectors.npz", allow_pickle=False)
    n_expr = qc["n_expressed_genes"]
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<28}  {detail}")
        if not ok:
            fails.append(name)

    worst = max(qc["reproduction_of_published_ceilings"]["max_abs_diff"].values())
    check("reproduces_published", worst == 0.0, f"max abs diff over every ceiling column {worst:.3g}")

    for split in C.CEILING_SPLITS:
        g = grid[(grid.split == split) & (grid.k_label == str(C.EFFECTS.topk))].set_index("target")
        ref = pert.set_index("target")[f"{split}__topk_overlap"]
        d = (g.overlap - ref).abs()
        check(f"k50_matches_{split}", float(d.max()) == 0.0, f"max |grid - published| {float(d.max()):.3g}")

    ok_k = all(
        set(grid[grid.k_label == str(k)].k.unique()) == {k} for k in ceilings_grid.K_GRID
    )
    check("k_grid", ok_k and set(grid.k_label.unique()) == {*map(str, ceilings_grid.K_GRID), "adaptive"},
          f"labels {sorted(set(grid.k_label))}")

    ch = grid.assign(expect=grid.k / n_expr)
    check("chance_level", np.allclose(ch.chance, ch.expect),
          f"k/n over {n_expr} expressed genes, k=10 -> {10 / n_expr:.4f}, k=500 -> {500 / n_expr:.4f}")

    fin = grid[np.isfinite(grid.overlap)]
    exp_excess = (fin.overlap - fin.chance) / (1 - fin.chance)
    check("excess_formula", np.allclose(fin.excess_over_chance, exp_excess),
          f"max diff {float((fin.excess_over_chance - exp_excess).abs().max()):.3g}")

    ad = grid[grid.k_label == "adaptive"]
    want = np.clip(ad.n_de_genes, ceilings_grid.ADAPTIVE_K_MIN, ceilings_grid.ADAPTIVE_K_MAX)
    check("adaptive_k", bool((ad.k == want).all()),
          f"k in [{int(ad.k.min())}, {int(ad.k.max())}] from DE counts "
          f"[{int(ad.n_de_genes.min())}, {int(ad.n_de_genes.max())}]")

    check("diagnostics_range",
          bool(fin.tie_fraction.between(0, 1).all() and (fin.n_tied_excluded >= 0).all()
               and (fin.median_control_cp10k_selected > 0).all()),
          f"median tie fraction {float(fin.tie_fraction.median()):.4f}, median selected level "
          f"{float(fin.median_control_cp10k_selected.median()):.3f} CP10K")

    # the persisted vectors are the ones that were scored
    targets = list(npz["targets"])
    rng = np.random.default_rng(0)
    sample = rng.choice(len(targets), size=min(25, len(targets)), replace=False)
    worst_vec = 0.0
    for i in sample:
        for split in C.CEILING_SPLITS:
            a = np.abs(npz[f"{split}__lfc_side0"][i])
            b = np.abs(npz[f"{split}__lfc_side1"][i])
            k = C.EFFECTS.topk
            o = len(set(np.argpartition(-a, k - 1)[:k].tolist())
                    & set(np.argpartition(-b, k - 1)[:k].tolist())) / k
            ref = float(pert.set_index("target").loc[targets[i], f"{split}__topk_overlap"])
            worst_vec = max(worst_vec, abs(o - ref))
    check("persisted_vectors", worst_vec <= 0.02,
          f"overlap recomputed from the float32 vectors differs by at most {worst_vec:.3g}")

    shapes = {k: npz[k].shape for k in npz.files if npz[k].ndim == 2}
    check("vector_arrays", len(shapes) == 8 and all(s == (pert.shape[0], n_expr) for s in shapes.values()),
          f"{len(shapes)} arrays of {pert.shape[0]} x {n_expr}, "
          f"{qc['outputs']['split_half_vectors.npz']['bytes'] / 1e6:.1f} MB")

    check("summary_rows", summ.shape[0] == 2 * 5 * 5 and summ.median_overlap.between(0, 1).all(),
          f"{summ.shape[0]} rows over {summ.k_label.nunique()} k labels x "
          f"{summ.stratum.nunique()} strata x {summ.split.nunique()} splits")
    diag = summ[summ.is_diagonal]
    check("diagonal_flag", diag.shape[0] == 2 * len(ceilings_grid.DIAGONAL_K),
          f"{diag.shape[0]} diagonal cells {sorted(set(zip(diag.stratum, diag.k_label)))}")

    print("\nsummary (split_half_cells):")
    print(summ[summ.split == "split_half_cells"][
        ["k_label", "stratum", "n_perturbations", "chance", "median_overlap",
         "median_excess_over_chance", "median_tie_fraction", "frac_selection_contested",
         "median_control_cp10k_selected"]
    ].round(4).to_string(index=False))
    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'OK: every grid property holds'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
