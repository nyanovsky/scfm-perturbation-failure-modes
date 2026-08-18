"""Read the full obs tables of both Nadig h5ads over HTTP range reads and
summarise the design: batches, targets, guides, QC distributions, gene sets.

Also spot-checks whether X holds integer counts.
"""

import json

import fsspec
import h5py
import numpy as np

URLS = {
    "hepg2": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_hepg2_raw_singlecell_01.h5ad",
    "jurkat": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_jurkat_raw_singlecell_01.h5ad",
}


def cats(h, group, col):
    raw = h[f"{group}/__categories/{col}"][:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in raw])


def q(a, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    return {f"p{p}": float(np.percentile(a, p)) for p in qs}


def main():
    out = {}
    genes = {}
    for key, url in URLS.items():
        fs = fsspec.filesystem("http", block_size=4 * 1024 * 1024)
        with fs.open(url, "rb") as fh, h5py.File(fh, "r") as h:
            n_cells, n_genes = h["X"].shape
            gem = h["obs/gem_group"][:]
            tgt_codes = h["obs/gene"][:]
            tgt_cats = cats(h, "obs", "gene")
            sg_codes = h["obs/sgID_AB"][:]
            sg_cats = cats(h, "obs", "sgID_AB")
            tr_codes = h["obs/transcript"][:]
            tr_cats = cats(h, "obs", "transcript")
            gt_cats = cats(h, "obs", "gene_transcript")
            gid_cats = cats(h, "obs", "gene_id")
            mito = h["obs/mitopercent"][:]
            umi = h["obs/UMI_count"][:]

            gene_ids = np.array(
                [x.decode() if isinstance(x, bytes) else str(x) for x in h["var/gene_id"][:]]
            )
            gname = cats(h, "var", "gene_name")[h["var/gene_name"][:]]
            in_matrix = h["var/in_matrix"][:]

            targets = tgt_cats[tgt_codes]
            tcounts = {}
            uniq, cnt = np.unique(targets, return_counts=True)
            for u, c in zip(uniq, cnt):
                tcounts[str(u)] = int(c)

            # cells per (target, batch)
            pair = np.array([f"{t}|{g}" for t, g in zip(targets, gem)])
            _, pcnt = np.unique(pair, return_counts=True)
            # batches per target
            bpt = {}
            for t in uniq:
                bpt[str(t)] = int(len(np.unique(gem[targets == t])))

            # X integrality spot check on 3 chunks
            checks = []
            for start in (0, n_cells // 2, max(0, n_cells - 200)):
                blk = h["X"][start : start + 200, :]
                checks.append(
                    {
                        "start": int(start),
                        "min": float(blk.min()),
                        "max": float(blk.max()),
                        "all_integer": bool(np.all(blk == np.round(blk))),
                        "row_sum_head": [float(v) for v in blk.sum(1)[:4]],
                        "frac_zero": float((blk == 0).mean()),
                    }
                )
            umi_head = [float(v) for v in umi[:4]]

            genes[key] = {"ids": gene_ids, "names": gname}
            out[key] = {
                "n_cells": int(n_cells),
                "n_genes": int(n_genes),
                "n_gem_groups": int(len(np.unique(gem))),
                "gem_groups": sorted(int(x) for x in np.unique(gem)),
                "cells_per_gem_group": q(np.unique(gem, return_counts=True)[1]),
                "n_target_categories": int(len(tgt_cats)),
                "n_targets_observed": int(len(uniq)),
                "n_sgid_categories": int(len(sg_cats)),
                "sgid_head": [str(x) for x in sg_cats[:6]],
                "transcript_categories": [str(x) for x in tr_cats[:10]],
                "n_transcript_categories": int(len(tr_cats)),
                "gene_transcript_head": [str(x) for x in gt_cats[:6]],
                "gene_id_cat_head": [str(x) for x in gid_cats[:6]],
                "control_like_targets": {
                    str(u): int(c)
                    for u, c in zip(uniq, cnt)
                    if any(
                        s in str(u).lower()
                        for s in ("non-target", "nontarget", "non_target", "ntc", "control", "safe", "scramble")
                    )
                },
                "targets_smallest": dict(
                    sorted(tcounts.items(), key=lambda kv: kv[1])[:10]
                ),
                "targets_largest": dict(
                    sorted(tcounts.items(), key=lambda kv: -kv[1])[:10]
                ),
                "cells_per_target": q(cnt),
                "cells_per_target_batch": q(pcnt),
                "n_target_batch_pairs": int(len(pcnt)),
                "batches_per_target": q(np.array(list(bpt.values()))),
                "mitopercent": q(mito),
                "UMI_count": q(umi),
                "umi_head": umi_head,
                "var_in_matrix_all_true": bool(in_matrix.all()),
                "gene_id_versioned": bool(any("." in g for g in gene_ids[:50])),
                "n_unique_gene_ids": int(len(set(gene_ids))),
                "n_unique_gene_names": int(len(set(gname.tolist()))),
                "X_checks": checks,
                "targets_head": [str(x) for x in sorted(uniq)[:15]],
            }
    a, b = genes["hepg2"], genes["jurkat"]
    out["gene_overlap"] = {
        "hepg2_ids": len(set(a["ids"])),
        "jurkat_ids": len(set(b["ids"])),
        "shared_ids": len(set(a["ids"]) & set(b["ids"])),
        "shared_names": len(set(a["names"].tolist()) & set(b["names"].tolist())),
    }
    th = set(k for k in out["hepg2"]["targets_smallest"])  # noqa: F841
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
