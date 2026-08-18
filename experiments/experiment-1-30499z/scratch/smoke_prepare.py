"""End-to-end smoke of the data pipeline on a synthetic h5ad with the exact
GSE264667 layout (dense float32 X, old-anndata categoricals in __categories).

Plants a fake raw file where sources.download expects it, so the real
prepare.run() code path executes unchanged apart from the network fetch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scfm_pfm import config as C  # noqa: E402
from scfm_pfm import paths  # noqa: E402
from scfm_pfm.data import prepare, sources  # noqa: E402

RNG = np.random.default_rng(0)


def write_fake(path: Path, n_cells=6000, n_genes=1000, n_targets=40, n_batches=8):
    targets = [C.CONTROL_LABEL] + [f"TGT{i:03d}" for i in range(n_targets)]
    # controls get a big share of cells; each target ~40 cells over ~4 batches
    probs = np.r_[0.35, np.full(n_targets, 0.65 / n_targets)]
    tcode = RNG.choice(len(targets), size=n_cells, p=probs)
    gem = RNG.integers(1, n_batches + 1, size=n_cells)

    base = RNG.gamma(2.0, 1.5, size=n_genes) * 3.0
    batch_shift = RNG.normal(0, 0.25, size=(n_batches, n_genes))
    # each target knocks its own gene down and moves a small module
    eff = np.zeros((len(targets), n_genes))
    for t in range(1, len(targets)):
        eff[t, (t - 1) % n_genes] = -1.6  # on-target
        mod = RNG.choice(n_genes, size=12, replace=False)
        eff[t, mod] += RNG.normal(0, 0.9, size=12)

    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    depth = RNG.integers(4000, 30000, size=n_cells)
    for i in range(n_cells):
        rate = base * np.exp(eff[tcode[i]] + batch_shift[gem[i] - 1])
        rate = rate / rate.sum() * depth[i]
        X[i] = RNG.poisson(rate)

    umi = X.sum(1) * 1.02
    mito = RNG.uniform(0.02, 0.14, size=n_cells)
    gene_ids = np.array([f"ENSG{i:011d}" for i in range(n_genes)])
    gene_names = np.array([f"G{i}" for i in range(n_genes)])
    # target i knocks down gene (i-1); its ENSG must resolve into var
    tgt_ensg = [gene_ids[(t - 1) % n_genes] if t > 0 else "none" for t in range(len(targets))]

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("X", data=X)
        obs = h.create_group("obs")
        obs.attrs["_index"] = "cell_barcode"
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["column-order"] = "['gem_group' 'gene' 'gene_id' 'sgID_AB' 'mitopercent' 'UMI_count']"
        cats = obs.create_group("__categories")
        cats.create_dataset("gene", data=np.array(targets, dtype="S"))
        cats.create_dataset("gene_id", data=np.array(tgt_ensg, dtype="S"))
        cats.create_dataset(
            "sgID_AB", data=np.array([f"sg_{t}" for t in targets], dtype="S")
        )
        obs.create_dataset("cell_barcode", data=np.array([f"BC{i}" for i in range(n_cells)], dtype="S"))
        obs.create_dataset("gem_group", data=gem.astype(np.int64))
        obs.create_dataset("gene", data=tcode.astype(np.int16))
        obs.create_dataset("gene_id", data=tcode.astype(np.int16))
        obs.create_dataset("sgID_AB", data=tcode.astype(np.int16))
        obs.create_dataset("mitopercent", data=mito.astype(np.float32))
        obs.create_dataset("UMI_count", data=umi.astype(np.float32))
        var = h.create_group("var")
        var.attrs["_index"] = "gene_id"
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["column-order"] = "['gene_name']"
        vcats = var.create_group("__categories")
        vcats.create_dataset("gene_name", data=np.array(gene_names, dtype="S"))
        var.create_dataset("gene_id", data=np.array(gene_ids, dtype="S"))
        var.create_dataset("gene_name", data=np.arange(n_genes, dtype=np.int16))
    return path


def main() -> int:
    root = Path(os.environ["SCFM_PFM_DATA_ROOT"])
    raw = paths.raw_dir()
    dest = raw / sources.H5AD_FILES["hepg2"]
    write_fake(dest)
    dest.with_suffix(dest.suffix + ".meta.json").write_text(
        json.dumps({"url": "synthetic", "size_bytes": dest.stat().st_size, "md5": "synthetic"})
    )
    # a smaller min_cells so the synthetic set retains perturbations
    object.__setattr__(C.PERTURBATION_FILTERS, "min_cells", 20)
    qc = prepare.run("hepg2", root / "effects" / "hepg2", raw)
    print(json.dumps({k: qc[k] for k in ("shape", "retained", "knockdown")}, indent=1))

    import pandas as pd

    pert = pd.read_parquet(root / "effects/hepg2/perturbations.parquet")
    summ = pd.read_parquet(root / "effects/hepg2/noise_ceiling_summary.parquet")
    npz = np.load(root / "effects/hepg2/effects.npz", allow_pickle=False)
    print("\nperturbations:", pert.shape)
    print(pert[[
        "target", "n_cells", "n_batches", "knockdown", "knockdown_detected",
        "effect_l2_lfc", "n_de_genes", "effect_stratum",
        "split_half_cells__pearson_lfc", "split_by_batch__pearson_lfc",
        "split_half_cells__cosine_delta", "split_half_cells__topk_overlap",
    ]].head(8).to_string())
    print("\nnan check:", {c: int(pert[c].isna().sum()) for c in pert.columns if pert[c].dtype.kind == "f"})
    print("\nceiling summary:\n", summ.to_string())
    print("\nnpz arrays:", {k: (npz[k].shape, str(npz[k].dtype)) for k in npz.files})
    for k in ("delta_within_batch", "lfc_within_batch"):
        assert np.isfinite(npz[k]).all(), f"{k} has non-finite values"
    print("\nOK: primary matrices finite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
