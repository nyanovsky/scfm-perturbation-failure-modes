"""Build the one bundle every later job reads.

A fresh on-demand job cannot see run 1's or run 2's artifacts, so this job pulls
them once through presigned links, rebuilds the control-cell statistics that
were never persisted, and writes a single consolidated tree:

    bundle/MANIFEST.json
    bundle/<line>/targets.npz      measured effect, its two half-estimates, keys
    bundle/<line>/coexpr.npz       co-expression pair features + node features
    bundle/<line>/emb__<model>.npz the model's gene embedding over the read-out set
    bundle/<line>/features/<model>__<anchor>.npz   zero-out / VVP / GDT
    bundle/<line>/reference.json   reliability, residual ceiling, power

Everything downstream addresses the bundle, never the store, so the training
jobs mount this job's volume and need no credential of their own.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import fetch, metrics, paths
from . import cells as cellmod
from . import no_op as no_op_mod
from . import coexpr as coexprmod

ANCHOR_FORMS = {
    "mean_control": ("zero_out", "vvp", "gdt"),
    "uniform_virtual": ("zero_out", "vvp", "gdt"),
    "individual_cells": ("zero_out", "vvp", "gdt"),
    "mean_control_masked": ("zero_out", "vvp"),
}


def _mirror_block(mirror: Path, model: str, line: str) -> Path:
    return mirror / C.RUN2 / C.BLOCK_PREFIX[(model, line)]


def _load_no_op(repo: Path) -> pd.DataFrame:
    return pd.read_csv(repo / "results" / "handoff" / "no_op_queries.csv")


def _load_no_op_per_gene(repo: Path) -> pd.DataFrame:
    """Which (perturbed gene, value) queries were no-ops, and in how many cells.

    Derived from run 2's per-query records by ``scfm_pfm.ingest.no_op`` and
    committed, because the aggregate count per value cannot say which queries
    it refers to.
    """
    return pd.read_csv(repo / "results" / "handoff" / "no_op_queries_per_gene.csv")


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def build_targets(mirror: Path, line: str) -> dict:
    readout = pd.read_parquet(paths.readout_table(line))
    perts = pd.read_parquet(paths.perturbations_table(line))
    scope = cellmod.in_scope_perturbed(perts, readout, C.PERTURBED_STRATA)

    eff = np.load(mirror / C.RUN1 / "effects" / line / "effects.npz", allow_pickle=False)
    gene_ids = eff["gene_ids"].astype(str)
    targets = eff["targets"].astype(str)
    col_of_gene = {g: i for i, g in enumerate(gene_ids)}
    row_of_target = {t: i for i, t in enumerate(targets)}

    readout_ids = readout["gene_id"].to_numpy().astype(str)
    cols = np.array([col_of_gene[g] for g in readout_ids], dtype=np.int64)
    rows = np.array([row_of_target[t] for t in scope["target"].to_numpy().astype(str)], dtype=np.int64)

    effect = eff[C.EFFECT_KEY][rows][:, cols].astype(np.float32)

    shv = np.load(
        mirror / C.RUN1 / "effects" / line / "split_half_vectors.npz", allow_pickle=False
    )
    shv_targets = shv["targets"].astype(str)
    shv_row = {t: i for i, t in enumerate(shv_targets)}
    gene_index = shv["gene_index"].astype(np.int64)  # matrix columns the vectors carry
    pos_of_col = {int(c): i for i, c in enumerate(gene_index)}
    missing = [int(c) for c in cols if int(c) not in pos_of_col]
    if missing:
        raise RuntimeError(f"{line}: {len(missing)} read-out genes absent from split-half vectors")
    shv_cols = np.array([pos_of_col[int(c)] for c in cols], dtype=np.int64)
    shv_rows = np.array([shv_row[t] for t in scope["target"].to_numpy().astype(str)], dtype=np.int64)
    half0 = shv["split_half_cells__lfc_side0"][shv_rows][:, shv_cols].astype(np.float32)
    half1 = shv["split_half_cells__lfc_side1"][shv_rows][:, shv_cols].astype(np.float32)

    return {
        "scope": scope,
        "readout": readout,
        "effect": effect,
        "half0": half0,
        "half1": half1,
        "n_perturbed": int(effect.shape[0]),
        "n_readout": int(effect.shape[1]),
    }


def reference_quantities(t: dict, line: str) -> dict:
    """Step 1: what is reproducible at all, and what difference is resolvable."""
    n_pert = t["n_perturbed"]
    all_rows = np.arange(n_pert)
    rel = metrics.per_gene_reliability(t["half0"], t["half1"], all_rows)
    w = rel["weights"]
    self_cols = t["scope"]["readout_col"].to_numpy().astype(np.int64)
    ceil_half = metrics.residual_ceiling(t["half0"], t["half1"], all_rows, w, self_cols)
    ceil_full = np.asarray(metrics.spearman_brown(ceil_half), dtype=np.float64)
    ceil_full = np.clip(ceil_full, 0.0, 1.0)

    raw_half = metrics.weighted_pearson_rows(
        t["half0"],
        t["half1"],
        _zero_self(np.broadcast_to(w, t["half0"].shape), self_cols),
    )
    raw_full = np.clip(np.asarray(metrics.spearman_brown(raw_half)), 0.0, 1.0)

    resid, _ = metrics.residualize(t["effect"], all_rows)
    mean_effect = t["effect"].mean(axis=0)

    sd_single = float(np.nanstd(ceil_full))
    power = {
        f"paired_rho_{rho}": metrics.min_resolvable_difference(
            sd_single, n_pert, rho, C.POWER_ALPHA, C.POWER_TARGET
        )
        for rho in C.POWER_PAIRED_RHO
    }
    return {
        "cell_line": line,
        "n_perturbations": n_pert,
        "n_readout_genes": t["n_readout"],
        "reliability": {
            "shrinkage_lambda": rel["shrinkage_lambda"],
            "median_weight": float(np.median(w)),
            "mean_weight": float(np.mean(w)),
            "fraction_weight_below_0.1": float((w < 0.1).mean()),
            "fraction_weight_above_0.5": float((w > 0.5).mean()),
            "var_between_z": rel["var_between_z"],
            "var_sampling_z": rel["var_sampling_z"],
        },
        "residual_ceiling": {
            "median_half_split": float(np.nanmedian(ceil_half)),
            "median_spearman_brown": float(np.nanmedian(ceil_full)),
            "q25": float(np.nanquantile(ceil_full, 0.25)),
            "q75": float(np.nanquantile(ceil_full, 0.75)),
            "sd": sd_single,
        },
        "raw_effect_ceiling": {
            "median_half_split": float(np.nanmedian(raw_half)),
            "median_spearman_brown": float(np.nanmedian(raw_full)),
        },
        "shared_response": {
            "mean_effect_l2": float(np.linalg.norm(mean_effect)),
            "median_residual_fraction_of_variance": float(
                np.median(
                    np.nanvar(resid, axis=1) / np.maximum(np.nanvar(t["effect"], axis=1), 1e-12)
                )
            ),
        },
        "power": {
            "n_held_out_perturbations": n_pert,
            "sd_per_perturbation_used": sd_single,
            "alpha": C.POWER_ALPHA,
            "target_power": C.POWER_TARGET,
            "min_resolvable_difference": power,
            "note": (
                "sd is the spread of the per-perturbation residual ceiling, used as "
                "the pre-run estimate of per-perturbation heterogeneity; the achieved "
                "resolution is recomputed from the actual paired differences"
            ),
        },
    }


def _zero_self(w: np.ndarray, self_cols: np.ndarray) -> np.ndarray:
    w = np.array(w, dtype=np.float64, copy=True)
    w[np.arange(w.shape[0]), self_cols] = 0.0
    return w


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def build_features(
    mirror: Path,
    line: str,
    t: dict,
    no_op: pd.DataFrame,
    no_op_per_gene: pd.DataFrame,
    out: Path,
) -> list[dict]:
    readout_ids = t["readout"]["gene_id"].to_numpy().astype(str)
    gene_to_scope = {
        g: i for i, g in enumerate(t["scope"]["gene_id"].to_numpy().astype(str))
    }
    records: list[dict] = []
    fdir = paths.ensure(out / "features")
    for model in C.MODELS:
        block = _mirror_block(mirror, model, line)
        emb_path = block / "gene_embeddings.npz"
        if emb_path.exists():
            e = np.load(emb_path, allow_pickle=False)
            emb = e["embedding"].astype(np.float32)
            emb_ids = e["gene_id"].astype(str)
            if not np.array_equal(emb_ids, readout_ids):
                order = {g: i for i, g in enumerate(emb_ids)}
                emb = emb[[order[g] for g in readout_ids]]
            np.savez(out / f"emb__{model}.npz", embedding=emb)
            records.append(
                {"model": model, "kind": "embedding", "shape": list(emb.shape)}
            )
        for anchor, forms in ANCHOR_FORMS.items():
            adir = block / f"anchor={anchor}"
            present = {f: adir / f"{f}.npz" for f in forms if (adir / f"{f}.npz").exists()}
            if "vvp" not in present:
                continue
            # No-op queries -- the value asked for was the value the input
            # already carried, so the two passes were the same call -- are
            # excluded per (perturbed gene, value) rather than by deleting the
            # dose, whose other perturbations are real measurements.
            sel = no_op[
                (no_op.model == model) & (no_op.cell_line == line) & (no_op.anchor == anchor)
            ]
            n_no_op = int(sel.n_no_op_queries.sum()) if len(sel) else 0
            arrays: dict[str, np.ndarray] = {}
            raws: dict[str, np.ndarray] = {}
            pert_ids = None
            values = None
            for form, path in present.items():
                z = np.load(path, allow_pickle=False)
                arr = z[C.FEATURE_FORM].astype(np.float32)
                if n_no_op and form in C.NO_OP_AFFECTED_FORMS:
                    raws[form] = z["raw"].astype(np.float64)
                ids = z["perturbed_gene_id"].astype(str)
                if pert_ids is None:
                    pert_ids = ids
                elif not np.array_equal(ids, pert_ids):
                    raise RuntimeError(f"{model}/{line}/{anchor}: {form} row keys disagree")
                if "readout_gene_id" in z.files:
                    rid = z["readout_gene_id"].astype(str)
                    if not np.array_equal(rid, readout_ids):
                        raise RuntimeError(f"{model}/{line}/{anchor}: {form} column order differs")
                if "value_grid_cp10k" in z.files:
                    values = z["value_grid_cp10k"].astype(np.float64)
                arrays[form] = arr
            pert_index = np.array([gene_to_scope[g] for g in pert_ids], dtype=np.int64)
            if values is None:
                values = np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64)
            correction: dict = {"n_no_op_queries": n_no_op}
            if n_no_op:
                if C.FEATURE_FORM != "standardized":
                    raise RuntimeError(
                        f"{model}/{line}/{anchor}: the no-op correction is defined for the "
                        f"per-query standardized form, not {C.FEATURE_FORM!r}"
                    )
                factor = no_op_mod.correction_factors(
                    no_op_per_gene, model, line, anchor, pert_ids, values
                )
                if np.allclose(factor, 1.0):
                    raise RuntimeError(
                        f"{model}/{line}/{anchor}: {n_no_op} no-op queries recorded but the "
                        "per-(gene, value) record covers none of this block's genes"
                    )
                dev = 0.0
                for form in sorted(raws):
                    at_zero = raws[form].ndim == 2
                    f = factor[:, no_op_mod.zero_index(values)] if at_zero else factor
                    fixed = no_op_mod.standardize_rows(raws[form] * f[..., None])
                    dev = max(dev, float(np.abs(fixed - arrays[form]).max()))
                    arrays[form] = fixed.astype(np.float32)
                undefined = factor == 0.0
                correction.update(
                    {
                        "forms_corrected": sorted(raws),
                        "n_query_rows_corrected": int(((factor > 0) & (factor != 1.0)).sum()),
                        "n_query_rows_everywhere_no_op": int(undefined.sum()),
                        "max_factor": float(factor.max()),
                        "max_abs_change_vs_shipped_feature": dev,
                    }
                )
            payload = {
                "pert_index": pert_index,
                "values": values,
                # every value is kept: the field stays for the reader's contract
                "value_keep": np.ones(values.size, dtype=bool),
            }
            for form, arr in arrays.items():
                payload[form] = arr
            np.savez(fdir / f"{model}__{anchor}.npz", **payload)
            records.append(
                {
                    "model": model,
                    "anchor": anchor,
                    "forms": sorted(arrays),
                    "n_perturbed": int(pert_index.size),
                    "values": [float(v) for v in values],
                    "no_op": correction,
                    "shapes": {f: list(a.shape) for f, a in arrays.items()},
                }
            )
            print(f"  [{line}] {model}/{anchor}: {records[-1]['shapes']}", flush=True)
    return records


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urls", help="presigned link manifest (JSON)")
    ap.add_argument(
        "--mirror",
        help="an existing artifact mirror to read instead of downloading one",
    )
    ap.add_argument("--work", default="_work", help="scratch dir for the mirror and raw h5ads")
    ap.add_argument("--cell-lines", default="hepg2,jurkat")
    ap.add_argument("--skip-control-cells", action="store_true")
    args = ap.parse_args(argv)

    t_start = time.time()
    work = Path(args.work)
    out_root = paths.ensure(paths.data_root() / "bundle")
    if bool(args.urls) == bool(args.mirror):
        raise SystemExit("pass exactly one of --urls (download) or --mirror (already fetched)")
    if args.mirror:
        mirror = Path(args.mirror)
        if not (mirror / C.RUN2).is_dir():
            raise SystemExit(f"{mirror} does not look like an artifact mirror")
        objects = sorted(p for p in mirror.rglob("*") if p.is_file())
        receipt = {
            "mode": "mounted_mirror",
            "mirror": str(mirror),
            "n_objects": len(objects),
            "total_bytes": int(sum(p.stat().st_size for p in objects)),
        }
        print(
            f"reading {receipt['n_objects']} objects already fetched into {mirror} "
            f"({receipt['total_bytes'] / 1e9:.2f} GB)",
            flush=True,
        )
    else:
        mirror = work / "mirror"
        receipt = fetch.download_manifest(Path(args.urls), mirror)
        print(
            f"fetched {receipt['n_objects']} objects, {receipt['total_bytes'] / 1e9:.2f} GB",
            flush=True,
        )

    no_op = _load_no_op(paths.repo_dir())
    no_op_per_gene = _load_no_op_per_gene(paths.repo_dir())
    manifest = {
        "written_by": "scfm_pfm.ingest.build",
        "run": 3,
        "seed": C.SEED,
        "effect_key": C.EFFECT_KEY,
        "feature_form": C.FEATURE_FORM,
        "primary_anchor": C.PRIMARY_ANCHOR,
        "perturbed_strata": list(C.PERTURBED_STRATA),
        "coexpr": {
            "half_used": C.COEXPR_HALF,
            "half_for_leakage_check": C.COEXPR_CHECK_HALF,
            "mi_bins": C.MI_BINS,
            "partial_shrinkage": C.PARTIAL_CORR_SHRINKAGE,
            "degree_abs_corr": C.DEGREE_ABS_CORR,
        },
        "sources": receipt,
        "lines": {},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    for line in args.cell_lines.split(","):
        line = line.strip()
        if not line:
            continue
        print(f"=== {line} ===", flush=True)
        out = paths.ensure(out_root / line)
        t = build_targets(mirror, line)
        scope = t["scope"]
        np.savez(
            out / "targets.npz",
            effect=t["effect"],
            half0=t["half0"],
            half1=t["half1"],
            pert_target=scope["target"].to_numpy().astype(str),
            pert_gene_id=scope["gene_id"].to_numpy().astype(str),
            pert_readout_col=scope["readout_col"].to_numpy().astype(np.int64),
            pert_stratum=scope["effect_stratum"].to_numpy().astype(str),
            pert_knockdown=scope["knockdown"].to_numpy().astype(np.float32),
            pert_n_de=scope["n_de_genes"].to_numpy().astype(np.int64),
            readout_gene_id=t["readout"]["gene_id"].to_numpy().astype(str),
            readout_gene_name=t["readout"]["gene_name"].to_numpy().astype(str),
            readout_pooled_cp10k=t["readout"]["pooled_control_cp10k"].to_numpy().astype(np.float32),
        )
        print(f"[{line}] targets {t['effect'].shape}", flush=True)

        ref = reference_quantities(t, line)

        coex_record: dict = {"computed": False}
        if not args.skip_control_cells:
            matrix_cols = (
                t["readout"][f"{line}_matrix_index"].to_numpy().astype(np.int64)
            )
            cc = cellmod.load_control_cells(line, work / "_raw", matrix_cols)
            got = cc["gene_ids_matrix"][matrix_cols]
            want = t["readout"]["gene_id"].to_numpy().astype(str)
            if not np.array_equal(got, want):
                bad = int((got != want).sum())
                raise RuntimeError(
                    f"{line}: {bad} read-out columns do not match the raw matrix gene ids"
                )
            cp10k = cc["cp10k"]
            half = coexprmod.split_control_cells(cp10k.shape[0], C.SEED)
            rows = scope["readout_col"].to_numpy().astype(np.int64)
            payload: dict[str, np.ndarray] = {}
            per_half: dict[str, dict] = {}
            for h in (C.COEXPR_HALF, C.COEXPR_CHECK_HALF):
                sub = cp10k[half == h]
                log_expr = np.log1p(sub)
                stats = coexprmod.pair_statistics(
                    log_expr, rows, C.MI_BINS, C.PARTIAL_CORR_SHRINKAGE
                )
                suffix = "" if h == C.COEXPR_HALF else "_check"
                for key in ("pearson", "spearman", "mi_bits", "partial"):
                    payload[key + suffix] = stats[key]
                node, node_names = coexprmod.node_features(
                    sub, log_expr, stats["corr_full"], C.DEGREE_ABS_CORR
                )
                payload["node" + suffix] = node
                per_half[f"half{h}"] = {
                    "n_cells": int(sub.shape[0]),
                    "median_abs_pearson": float(np.median(np.abs(stats["pearson"]))),
                    "median_mi_bits": float(np.median(stats["mi_bits"])),
                    "median_abs_partial": float(np.median(np.abs(stats["partial"]))),
                }
            agree = metrics.columnwise_pearson(
                payload["pearson"].T.astype(np.float64), payload["pearson_check"].T.astype(np.float64)
            )
            payload["node_names"] = np.asarray(node_names, dtype=object).astype(str)
            np.savez(out / "coexpr.npz", **payload)
            coex_record = {
                "computed": True,
                "download": cc["download"],
                "funnel": cc["funnel"],
                "n_control_cells": int(cp10k.shape[0]),
                "per_half": per_half,
                "half_agreement_median_pearson_per_perturbed_gene": float(np.nanmedian(agree)),
                "node_features": node_names,
            }
            print(f"[{line}] co-expression {payload['pearson'].shape}", flush=True)

        feature_records = build_features(mirror, line, t, no_op, no_op_per_gene, out)
        (out / "reference.json").write_text(json.dumps(ref, indent=2))
        manifest["lines"][line] = {
            "reference": ref,
            "coexpr": coex_record,
            "features": feature_records,
            "n_perturbed": t["n_perturbed"],
            "n_readout": t["n_readout"],
        }

    manifest["seconds"] = round(time.time() - t_start, 1)
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2)[:4000])
    print(f"bundle at {out_root} in {manifest['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
