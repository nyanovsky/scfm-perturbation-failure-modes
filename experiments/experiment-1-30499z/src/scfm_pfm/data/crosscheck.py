"""Cross-check our effect estimates against the tahoebio reference release.

The reference (HuggingFace ``tahoebio/replogle-nadig-de-rhaister``, ``pdex``
table) holds one row per (cell line, gem_group, target, measured gene) with
``fold_change = log2(target_mean / ref_mean)`` computed *within* batch -- the same
contrast our primary estimator uses. It is a consistency reference, never the
source of record.

The 29.8 GB parquet is streamed row group by row group over HTTP range reads,
projecting seven columns and skipping row groups whose statistics show they hold
neither of our two cell lines. Per (cell line, target, gene) we recompute the
batch-pooled fold change the same way we pool ours:

    log2( sum_b n_b * target_mean_b / sum_b n_b * ref_mean_b )

and then report both a pooled Spearman over every shared (cell line, target,
gene) entry and the distribution of per-perturbation Spearman across genes.

The same module also measures our *own* two estimators against each other
(within-batch vs batch-pooled), which is what exposes batch confounding.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import paths
from ..stats import pearson, spearman
from . import sources as sources_mod

PDEX_URL = (
    "https://huggingface.co/datasets/tahoebio/replogle-nadig-de-rhaister/"
    "resolve/main/pdex/all_pdex.parquet"
)
FEATURE_NAMES_URL = (
    "https://huggingface.co/datasets/tahoebio/replogle-nadig-de-rhaister/"
    "resolve/main/definition/feature_names.json"
)
COLUMNS = [
    "cell_line",
    "target",
    "feature",
    "fold_change",
    "target_mean",
    "ref_mean",
    "target_membership",
]


def _open_remote(url: str):
    import fsspec

    fs = fsspec.filesystem("http", block_size=16 * 1024 * 1024)
    return fs.open(url, "rb")


def stream_reference(
    cell_lines: tuple[str, ...], n_features: int, max_row_groups: int | None = None
) -> dict[str, dict]:
    """Accumulate weighted target/ref means per (cell line, target, gene index)."""
    import pyarrow.parquet as pq

    fh = _open_remote(PDEX_URL)
    pf = pq.ParquetFile(fh)
    md = pf.metadata
    names = [md.schema.column(i).name for i in range(md.num_columns)]
    cl_col = names.index("cell_line")
    wanted = set(cell_lines)

    max_targets = 2600  # the library has ~2,400 targets; leaves headroom
    acc = {
        cl: {
            "targets": {},
            "num": np.zeros((max_targets, n_features), dtype=np.float64),
            "den": np.zeros((max_targets, n_features), dtype=np.float64),
        }
        for cl in cell_lines
    }
    t0 = time.time()
    n_read = 0
    n_skipped = 0
    total_groups = md.num_row_groups
    limit = total_groups if max_row_groups is None else min(total_groups, max_row_groups)
    for i in range(limit):
        stats = md.row_group(i).column(cl_col).statistics
        if stats is not None and stats.has_min_max:
            lo, hi = str(stats.min), str(stats.max)
            if not any(lo <= cl <= hi for cl in wanted):
                n_skipped += 1
                continue
        tab = pf.read_row_group(i, columns=COLUMNS).to_pandas()
        tab = tab[tab.cell_line.isin(wanted)]
        n_read += 1
        if tab.empty:
            continue
        for cl, sub in tab.groupby("cell_line", observed=True):
            a = acc[str(cl)]
            for tgt in pd.unique(sub.target):
                if tgt not in a["targets"]:
                    a["targets"][tgt] = len(a["targets"])
            ti = sub.target.map(a["targets"]).to_numpy().astype(np.int64)
            fi = sub.feature.to_numpy().astype(np.int64)
            w = sub.target_membership.to_numpy().astype(np.float64)
            np.add.at(a["num"], (ti, fi), w * sub.target_mean.to_numpy(dtype=np.float64))
            np.add.at(a["den"], (ti, fi), w * sub.ref_mean.to_numpy(dtype=np.float64))
        if i % 20 == 0:
            print(
                f"    reference row group {i}/{limit} read={n_read} skipped={n_skipped} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    fh.close()
    return acc


def own_estimator_agreement(cell_line: str) -> pd.DataFrame:
    """Within-batch vs batch-pooled, per perturbation: the batch-confounding check."""
    d = paths.effects_input_dir(cell_line)
    npz = np.load(d / "effects.npz", allow_pickle=False)
    pert = pd.read_parquet(d / "perturbations.parquet")
    expressed = npz["expressed"]
    wb, bp = npz["lfc_within_batch"], npz["lfc_pooled"]
    dwb, dbp = npz["delta_within_batch"], npz["delta_pooled"]
    rows = []
    for i, target in enumerate(npz["targets"]):
        rows.append(
            {
                "cell_line": cell_line,
                "target": str(target),
                "pearson_lfc_within_vs_pooled": pearson(wb[i][expressed], bp[i][expressed]),
                "spearman_lfc_within_vs_pooled": spearman(wb[i][expressed], bp[i][expressed]),
                "pearson_delta_within_vs_pooled": pearson(dwb[i][expressed], dbp[i][expressed]),
                "l2_within": float(np.linalg.norm(wb[i][expressed])),
                "l2_pooled": float(np.linalg.norm(bp[i][expressed])),
            }
        )
    return pd.DataFrame(rows).merge(
        pert[["target", "effect_stratum", "effect_l2_lfc", "n_cells", "knockdown"]],
        on="target",
        how="left",
    )


def run(out_dir: Path, max_row_groups: int | None = None, sample: int = 40000) -> dict:
    started = time.time()
    import urllib.request

    with urllib.request.urlopen(FEATURE_NAMES_URL, timeout=120) as fh:
        feature_names = json.load(fh)
    feature_names = [str(x) for x in feature_names]
    print(f"[crosscheck] reference gene panel: {len(feature_names)} genes", flush=True)

    acc = stream_reference(paths.CELL_LINES, len(feature_names), max_row_groups)

    pair_rows: list[dict] = []
    scatter: list[pd.DataFrame] = []
    pooled_ours: list[np.ndarray] = []
    pooled_ref: list[np.ndarray] = []
    per_line: dict[str, dict] = {}
    rng = np.random.default_rng(C.SEED)

    for cl in paths.CELL_LINES:
        d = paths.effects_input_dir(cl)
        npz = np.load(d / "effects.npz", allow_pickle=False)
        genes = pd.read_parquet(d / "genes.parquet")
        pert = pd.read_parquet(d / "perturbations.parquet")
        our_targets = {str(t): i for i, t in enumerate(npz["targets"])}
        our_lfc = npz["lfc_within_batch"]
        expressed = npz["expressed"]

        # reference gene symbol -> our row (restricted to expressed genes)
        row_by_name: dict[str, int] = {}
        for name, idx, exp in zip(genes.gene_name, genes.gene_index, genes.expressed):
            if exp and name not in row_by_name:
                row_by_name[str(name)] = int(idx)
        shared_feat = [
            (fi, row_by_name[name]) for fi, name in enumerate(feature_names) if name in row_by_name
        ]
        fidx = np.array([a for a, _ in shared_feat], dtype=np.int64)
        gidx = np.array([b for _, b in shared_feat], dtype=np.int64)

        a = acc[cl]
        num, den = a["num"], a["den"]
        eps = C.EFFECTS.lfc_eps
        n_pairs = 0
        for target, ri in a["targets"].items():
            oi = our_targets.get(str(target))
            if oi is None:
                continue
            ref = np.log2((num[ri][fidx] + eps) / (den[ri][fidx] + eps))
            ours = our_lfc[oi][gidx]
            ok = np.isfinite(ref) & np.isfinite(ours) & (den[ri][fidx] > 0)
            if ok.sum() < 50:
                continue
            n_pairs += 1
            pair_rows.append(
                {
                    "cell_line": cl,
                    "target": str(target),
                    "n_shared_genes": int(ok.sum()),
                    "spearman": spearman(ours[ok], ref[ok]),
                    "pearson": pearson(ours[ok], ref[ok]),
                }
            )
            pooled_ours.append(ours[ok])
            pooled_ref.append(ref[ok])
            if rng.random() < 0.05:
                take = rng.choice(int(ok.sum()), size=min(40, int(ok.sum())), replace=False)
                scatter.append(
                    pd.DataFrame(
                        {
                            "cell_line": cl,
                            "target": str(target),
                            "our_log2fc": ours[ok][take],
                            "reference_log2fc": ref[ok][take],
                        }
                    )
                )
        per_line[cl] = {
            "reference_targets_seen": len(a["targets"]),
            "shared_genes": int(fidx.size),
            "expressed_genes": int(expressed.sum()),
            "matched_pairs": n_pairs,
            "our_retained_perturbations": int(pert.shape[0]),
        }
        print(f"[crosscheck] {cl}: {json.dumps(per_line[cl])}", flush=True)

    pairs = pd.DataFrame(pair_rows)
    all_ours = np.concatenate(pooled_ours) if pooled_ours else np.array([])
    all_ref = np.concatenate(pooled_ref) if pooled_ref else np.array([])
    pooled_rho = spearman(all_ours, all_ref) if all_ours.size else float("nan")

    own = pd.concat([own_estimator_agreement(cl) for cl in paths.CELL_LINES], ignore_index=True)

    out_dir = paths.ensure(out_dir)
    pairs.to_parquet(out_dir / "reference_agreement.parquet", index=False)
    own.to_parquet(out_dir / "own_estimator_agreement.parquet", index=False)
    if scatter:
        pd.concat(scatter, ignore_index=True).to_parquet(
            out_dir / "scatter_sample.parquet", index=False
        )

    by_stratum = {}
    if not pairs.empty:
        merged = pairs.merge(
            pd.concat(
                [
                    pd.read_parquet(paths.effects_input_dir(cl) / "perturbations.parquet")[
                        ["target", "effect_stratum"]
                    ].assign(cell_line=cl)
                    for cl in paths.CELL_LINES
                ],
                ignore_index=True,
            ),
            on=["cell_line", "target"],
            how="left",
        )
        by_stratum = {
            str(k): {
                "n": int(v.shape[0]),
                "median_spearman": float(v.spearman.median()),
            }
            for k, v in merged.groupby("effect_stratum")
        }

    summary = {
        "gate_metric": "pooled Spearman over every shared (cell line, target, gene) log2FC entry",
        "pooled_spearman": float(pooled_rho),
        "n_pooled_entries": int(all_ours.size),
        "n_shared_cell_line_target_pairs": int(pairs.shape[0]),
        "median_per_pair_spearman": float(pairs.spearman.median()) if not pairs.empty else float("nan"),
        "median_per_pair_spearman_top_stratum": (
            by_stratum.get("S4_highest", {}).get("median_spearman", float("nan"))
        ),
        "per_pair_spearman_by_stratum": by_stratum,
        "per_cell_line": per_line,
        "reference": {
            "repo": sources_mod.REFERENCE_REPO,
            "table": "pdex/all_pdex.parquet",
            "reference_gene_panel": len(feature_names),
            "pooling": "log2(sum_b n_b*target_mean_b / sum_b n_b*ref_mean_b)",
        },
        "own_estimator_agreement": {
            "median_pearson_within_vs_pooled_lfc": float(
                own.pearson_lfc_within_vs_pooled.median()
            ),
            "median_pearson_within_vs_pooled_delta": float(
                own.pearson_delta_within_vs_pooled.median()
            ),
            "by_stratum": {
                str(k): float(v.pearson_lfc_within_vs_pooled.median())
                for k, v in own.groupby("effect_stratum")
            },
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:4000], flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-row-groups", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else paths.crosscheck_dir()
    run(out, args.max_row_groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
