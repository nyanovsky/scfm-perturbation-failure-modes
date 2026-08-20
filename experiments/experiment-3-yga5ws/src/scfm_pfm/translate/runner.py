"""Run the sweep: one fixed translator, one varying feature.

For every (cell line, perturbation scope, split type, arm, translator) the
runner trains on the training perturbations of each fold and predicts the whole
read-out vector of the held-out ones, so each perturbation is scored exactly
once. It writes two tables: one row per run with the whole correlation grid,
and one row per (run, held-out perturbation) so that paired differences between
arms can be computed later without refitting anything.

Everything that could differ between arms is fixed here: the folds, the
leave-one-out mean, the reliability weights, the feature standardization and the
translator. Only ``ArmFeatures.batch`` changes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import config as C
from .. import discrimination as disc
from .. import metrics, paths
from . import arms as armsmod
from . import splits as splitsmod
from . import trainers
from .data import LineBundle


def fold_reference(bundle: LineBundle, rows: np.ndarray, fold: np.ndarray, n_folds: int) -> dict:
    """Per-fold weights, means and the four ceilings the grid is read against."""
    half0 = bundle.half0[rows]
    half1 = bundle.half1[rows]
    effect = bundle.effect[torch.as_tensor(rows, device=bundle.effect.device)].cpu().numpy()
    self_cols = bundle.self_col_np[rows]
    per_fold = []
    ceil = {k: [] for k in ("residual_weighted", "residual_unweighted", "raw_weighted", "raw_unweighted")}
    for f in range(n_folds):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        rel = metrics.per_gene_reliability(half0, half1, tr)
        w = rel["weights"]
        mean_train = effect[tr].mean(axis=0)
        per_fold.append({"train": tr, "test": te, "weights": w, "mean": mean_train, "rel": rel})
        wmat = np.broadcast_to(w, (te.size, half0.shape[1])).copy()
        wmat[np.arange(te.size), self_cols[te]] = 0.0
        ones = np.ones_like(wmat)
        ones[np.arange(te.size), self_cols[te]] = 0.0
        r0, _ = metrics.residualize(half0, tr)
        r1, _ = metrics.residualize(half1, tr)
        ceil["residual_weighted"].append(metrics.weighted_pearson_rows(r0[te], r1[te], wmat))
        ceil["residual_unweighted"].append(metrics.weighted_pearson_rows(r0[te], r1[te], ones))
        ceil["raw_weighted"].append(metrics.weighted_pearson_rows(half0[te], half1[te], wmat))
        ceil["raw_unweighted"].append(metrics.weighted_pearson_rows(half0[te], half1[te], ones))
    ceilings = {
        k: float(np.clip(metrics.spearman_brown(np.nanmedian(np.concatenate(v))), 1e-6, 1.0))
        for k, v in ceil.items()
    }
    return {"folds": per_fold, "ceilings": ceilings, "effect": effect, "self_cols": self_cols}


def _keep_predictions(arm: str, translator: str, scope: str, split: str) -> bool:
    """Predictions are kept for the arms the later checks and figures need, not
    for every cell of the sweep: one held-out prediction matrix is ~15 MB."""
    if translator != "mlp" or scope != "full" or split != "grouped":
        return False
    return (
        arm in {"coexp", "node", "coexp_node", "mean_only", "null_randpair"}
        or arm.endswith(("_full", "_resp", "+shuffled_labels"))
    )


def score_run(
    pred_resid: np.ndarray,
    ref: dict,
    fold: np.ndarray,
    n_folds: int,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """The eight-cell correlation grid plus the per-perturbation scores."""
    effect = ref["effect"]
    self_cols = ref["self_cols"]
    n, r = effect.shape
    true_resid = np.zeros_like(effect)
    pred_raw = np.zeros_like(effect)
    weights = np.zeros_like(effect)
    for f in range(n_folds):
        info = ref["folds"][f]
        te = info["test"]
        true_resid[te] = effect[te] - info["mean"][None, :]
        pred_raw[te] = pred_resid[te] + info["mean"][None, :]
        weights[te] = info["weights"][None, :]
    weights[np.arange(n), self_cols] = 0.0
    ones = np.ones_like(weights)
    ones[np.arange(n), self_cols] = 0.0

    cells = {
        ("residual", "weighted"): metrics.weighted_pearson_rows(pred_resid, true_resid, weights),
        ("residual", "unweighted"): metrics.weighted_pearson_rows(pred_resid, true_resid, ones),
        ("raw", "weighted"): metrics.weighted_pearson_rows(pred_raw, effect, weights),
        ("raw", "unweighted"): metrics.weighted_pearson_rows(pred_raw, effect, ones),
    }
    summary: dict = {}
    for (target, weighting), values in cells.items():
        boot = metrics.bootstrap_median(values, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_CI, C.SEED)
        ceiling = ref["ceilings"][f"{target}_{weighting}"]
        key = f"{target}__{weighting}"
        summary[f"{key}__attenuated"] = boot["median"]
        summary[f"{key}__attenuated_lo"] = boot["lo"]
        summary[f"{key}__attenuated_hi"] = boot["hi"]
        summary[f"{key}__disattenuated"] = boot["median"] / np.sqrt(ceiling)
        summary[f"{key}__disattenuated_lo"] = boot["lo"] / np.sqrt(ceiling)
        summary[f"{key}__disattenuated_hi"] = boot["hi"] / np.sqrt(ceiling)
        summary[f"{key}__ceiling"] = ceiling
        summary[f"{key}__n_scored"] = boot["n"]
    scored = ones > 0
    true_sd = float(np.std(true_resid[scored]))
    summary["calibration_sd_ratio"] = float(np.std(pred_resid[scored]) / max(true_sd, 1e-12))
    per_pert = pd.DataFrame(
        {
            "fold": fold,
            "residual_weighted": cells[("residual", "weighted")],
            "residual_unweighted": cells[("residual", "unweighted")],
            "raw_weighted": cells[("raw", "weighted")],
            "raw_unweighted": cells[("raw", "unweighted")],
        }
    )
    return summary, per_pert, pred_raw, true_resid


def run_cell(
    bundle: LineBundle,
    scope: str,
    split: str,
    task_filter,
    out_dir: Path,
    shard: int,
    n_shards: int,
    translators: tuple[str, ...],
    with_cell_eval: bool,
    shuffle_null: bool,
    only_arms: tuple[str, ...] = (),
    save_predictions: bool = False,
) -> tuple[list[dict], list[pd.DataFrame]]:
    device = str(bundle.effect.device)
    rows = bundle.scope_rows(scope, C.PRIMARY_ANCHOR)
    if rows.size < 50:
        return [], []
    models = bundle.models_for_scope(scope, C.PRIMARY_ANCHOR, rows)
    effect_rows = bundle.effect[torch.as_tensor(rows, device=device)].cpu().numpy()
    fold_info = splitsmod.make_folds(
        effect_rows, split, C.N_FOLDS, C.SEED, C.GROUP_CORR_THRESHOLD
    )
    fold = fold_info["fold"]
    ref = fold_reference(bundle, rows, fold, C.N_FOLDS)

    include_sens = (
        scope == "full" and bundle.cell_line == C.SENSITIVITY_CELL_LINE and split == "grouped"
    )
    specs = armsmod.resolve_arms(bundle, models, C.PRIMARY_ANCHOR, include_sens)
    if shuffle_null:
        for name in C.SHUFFLE_NULL_ARMS:
            base = [s for s in specs if s.name == name]
            if base:
                s = base[0]
                specs.append(
                    C.ArmSpec(
                        f"{s.name}+shuffled_labels", "null", s.blocks, s.model, s.anchor,
                        note="perturbation labels permuted",
                    )
                )

    records: list[dict] = []
    tables: list[pd.DataFrame] = []
    local_self = bundle.self_col_np[rows]
    # map scope rows onto bundle rows once, so ArmFeatures gathers correctly
    row_lookup = torch.as_tensor(rows, device=device)

    for spec in specs:
        if only_arms and spec.name not in only_arms:
            continue
        if spec.model is not None:
            entry = bundle.features.get((spec.model, spec.anchor))
            if entry is None or (entry["row_map_np"][rows] < 0).any():
                continue
        for translator in translators:
            if spec.name == "mean_only" and translator != "ridge":
                continue
            key = (bundle.cell_line, scope, split, spec.name, translator)
            if not task_filter(key):
                continue
            t0 = time.time()
            pred = np.zeros((rows.size, bundle.n_readout), dtype=np.float32)
            detail: list[dict] = []
            shuffled = spec.name.endswith("+shuffled_labels")
            for f in range(C.N_FOLDS):
                info = ref["folds"][f]
                tr_local, te_local = info["train"], info["test"]
                target = torch.as_tensor(
                    ref["effect"] - info["mean"][None, :], device=device, dtype=torch.float32
                )
                if shuffled:
                    # Permute labels WITHIN the training perturbations only. A
                    # permutation over all rows would move held-out rows'
                    # targets into the training set, and those rows share a
                    # per-gene offset (their mean minus the training mean) that
                    # a null model could then learn and score on.
                    rng = np.random.default_rng(C.SEED + f)
                    perm = np.arange(rows.size)
                    perm[tr_local] = tr_local[rng.permutation(tr_local.size)]
                    target = target[torch.as_tensor(perm, device=device)]
                w = torch.as_tensor(info["weights"], device=device, dtype=torch.float32)
                arm = _bind(spec, bundle, row_lookup)
                if spec.name == "mean_only":
                    res = trainers.fit_mean_only(target, te_local)
                elif translator == "mlp":
                    res = trainers.fit_mlp(arm, target, w, tr_local, te_local, local_self)
                else:
                    res = trainers.fit_ridge(arm, target, w, tr_local, te_local, local_self)
                pred[te_local] = res.pop("prediction")
                detail.append(res)
            summary, per_pert, pred_raw, _ = score_run(pred, ref, fold, C.N_FOLDS)
            record = {
                "cell_line": bundle.cell_line,
                "scope": scope,
                "split": split,
                "arm": spec.name,
                "family": spec.family,
                "model": spec.model or "",
                "anchor": spec.anchor or "",
                "translator": translator,
                "n_perturbations": int(rows.size),
                "n_readout": int(bundle.n_readout),
                "feature_dim": int(detail[0].get("dim", 0)) if detail else 0,
                "seconds": round(time.time() - t0, 1),
                "n_groups": fold_info["n_groups"],
                "largest_group": fold_info["largest_group"],
                **summary,
            }
            if detail and "train_loss_first" in detail[0]:
                record["train_loss_first"] = float(np.mean([d["train_loss_first"] for d in detail]))
                record["train_loss_last"] = float(np.mean([d["train_loss_last"] for d in detail]))
            if detail and "alpha" in detail[0]:
                record["ridge_alpha_median"] = float(np.median([d["alpha"] for d in detail]))
            own = disc.own_discrimination(
                np.stack([ref["effect"][i] - ref["folds"][fold[i]]["mean"] for i in range(rows.size)]),
                pred,
            )
            record["discrimination_ours_residual_mean"] = own["mean"]
            record["discrimination_ours_residual_median"] = own["median"]
            if with_cell_eval:
                try:
                    ce = disc.cell_eval_discrimination(
                        ref["effect"],
                        pred_raw,
                        bundle.pert_target[rows],
                        bundle.readout_gene_name,
                    )
                    record["discrimination_cell_eval_mean"] = ce["mean"]
                    record["discrimination_cell_eval_median"] = ce["median"]
                    record["cell_eval_version"] = ce["package_version"]
                except Exception as exc:  # noqa: BLE001 - recorded, not raised
                    record["discrimination_cell_eval_error"] = f"{type(exc).__name__}: {exc}"[:300]
            per_pert = per_pert.assign(
                cell_line=bundle.cell_line,
                scope=scope,
                split=split,
                arm=spec.name,
                translator=translator,
                perturbation=bundle.pert_target[rows],
            )
            if save_predictions and _keep_predictions(spec.name, translator, scope, split):
                pdir = paths.ensure(out_dir / "predictions")
                np.savez_compressed(
                    pdir / f"{bundle.cell_line}__{scope}__{split}__{spec.name}__{translator}.npz",
                    prediction_residual=pred.astype(np.float32),
                    prediction_effect_scale=pred_raw.astype(np.float32),
                    measured_effect=ref["effect"].astype(np.float32),
                    perturbation=bundle.pert_target[rows],
                    gene_name=bundle.readout_gene_name,
                    fold=fold,
                )
            records.append(record)
            tables.append(per_pert)
            print(
                f"  [{bundle.cell_line}/{scope}/{split}] {spec.name}/{translator} "
                f"d={record['feature_dim']} "
                f"resid_w_dis={record['residual__weighted__disattenuated']:.4f} "
                f"({record['seconds']}s)",
                flush=True,
            )
    return records, tables


def run_transfer(
    src: LineBundle,
    dst: LineBundle,
    translators: tuple[str, ...],
    only_arms: tuple[str, ...] = (),
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Train on one cell line, test on the other, over the shared read-out set.

    The translator sees nothing from the test line. The mean effect subtracted
    from the test line's measurements is still estimated from the test line's
    OTHER perturbations, because the shared growth-arrest response is a
    population property of that screen rather than something a predictor could
    know; the same leave-one-out rule is used inside a line.
    """
    shared = np.array(
        [g for g in src.readout_gene_id if g in set(dst.readout_gene_id.tolist())], dtype=object
    )
    sv, dv = src.restrict_columns(shared), dst.restrict_columns(shared)
    rows_src = np.flatnonzero(sv.self_col_np >= 0)
    rows_dst = np.flatnonzero(dv.self_col_np >= 0)
    device = str(sv.effect.device)

    eff_src = sv.effect[torch.as_tensor(rows_src, device=device)].cpu().numpy()
    eff_dst = dv.effect[torch.as_tensor(rows_dst, device=device)].cpu().numpy()
    all_src, all_dst = np.arange(rows_src.size), np.arange(rows_dst.size)
    rel_src = metrics.per_gene_reliability(sv.half0[rows_src], sv.half1[rows_src], all_src)
    rel_dst = metrics.per_gene_reliability(dv.half0[rows_dst], dv.half1[rows_dst], all_dst)
    w_src = torch.as_tensor(rel_src["weights"], device=device, dtype=torch.float32)
    resid_src, _ = metrics.residualize(eff_src, all_src)
    resid_dst, mean_dst = metrics.residualize(eff_dst, all_dst)
    y_src = torch.as_tensor(resid_src, device=device, dtype=torch.float32)

    self_src = sv.self_col_np[rows_src]
    self_dst = dv.self_col_np[rows_dst]
    w_dst = np.broadcast_to(rel_dst["weights"], resid_dst.shape).copy()
    w_dst[np.arange(rows_dst.size), self_dst] = 0.0
    ones = np.ones_like(w_dst)
    ones[np.arange(rows_dst.size), self_dst] = 0.0
    r0, _ = metrics.residualize(dv.half0[rows_dst], all_dst)
    r1, _ = metrics.residualize(dv.half1[rows_dst], all_dst)
    ceilings = {
        "residual_weighted": float(
            np.clip(metrics.spearman_brown(
                np.nanmedian(metrics.weighted_pearson_rows(r0, r1, w_dst))), 1e-6, 1.0)
        ),
        "residual_unweighted": float(
            np.clip(metrics.spearman_brown(
                np.nanmedian(metrics.weighted_pearson_rows(r0, r1, ones))), 1e-6, 1.0)
        ),
        "raw_weighted": float(
            np.clip(metrics.spearman_brown(np.nanmedian(
                metrics.weighted_pearson_rows(dv.half0[rows_dst], dv.half1[rows_dst], w_dst))),
                1e-6, 1.0)
        ),
        "raw_unweighted": float(
            np.clip(metrics.spearman_brown(np.nanmedian(
                metrics.weighted_pearson_rows(dv.half0[rows_dst], dv.half1[rows_dst], ones))),
                1e-6, 1.0)
        ),
    }

    models = [m for m in sv.models_present(C.PRIMARY_ANCHOR) if m in dv.emb]
    models = [
        m for m in models
        if (sv.features[(m, C.PRIMARY_ANCHOR)]["row_map_np"][rows_src] >= 0).all()
        and (dv.features[(m, C.PRIMARY_ANCHOR)]["row_map_np"][rows_dst] >= 0).all()
    ]
    specs = armsmod.resolve_arms(sv, models, C.PRIMARY_ANCHOR, False)
    records, tables = [], []
    lookup_src = torch.as_tensor(rows_src, device=device)
    lookup_dst = torch.as_tensor(rows_dst, device=device)
    for spec in specs:
        if only_arms and spec.name not in only_arms:
            continue
        if spec.name == "mean_only":
            continue
        for translator in translators:
            t0 = time.time()
            arm_tr = _bind(spec, sv, lookup_src)
            arm_te = _bind(spec, dv, lookup_dst)
            fit = trainers.fit_mlp if translator == "mlp" else trainers.fit_ridge
            res = fit(
                arm_tr, y_src, w_src, all_src, all_dst, self_src,
                arm_test=arm_te, n_readout_test=int(sv.n_readout),
            )
            pred = res["prediction"]
            cells = {
                ("residual", "weighted"): metrics.weighted_pearson_rows(pred, resid_dst, w_dst),
                ("residual", "unweighted"): metrics.weighted_pearson_rows(pred, resid_dst, ones),
                ("raw", "weighted"): metrics.weighted_pearson_rows(
                    pred + mean_dst, eff_dst, w_dst),
                ("raw", "unweighted"): metrics.weighted_pearson_rows(
                    pred + mean_dst, eff_dst, ones),
            }
            record = {
                "cell_line": f"{src.cell_line}->{dst.cell_line}",
                "scope": "transfer",
                "split": "cross_line",
                "arm": spec.name,
                "family": spec.family,
                "model": spec.model or "",
                "anchor": spec.anchor or "",
                "translator": translator,
                "n_perturbations": int(rows_dst.size),
                "n_train_perturbations": int(rows_src.size),
                "n_readout": int(sv.n_readout),
                "feature_dim": int(res.get("dim", 0)),
                "seconds": round(time.time() - t0, 1),
            }
            for (target, weighting), values in cells.items():
                boot = metrics.bootstrap_median(
                    values, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_CI, C.SEED)
                ceiling = ceilings[f"{target}_{weighting}"]
                key = f"{target}__{weighting}"
                record[f"{key}__attenuated"] = boot["median"]
                record[f"{key}__attenuated_lo"] = boot["lo"]
                record[f"{key}__attenuated_hi"] = boot["hi"]
                record[f"{key}__disattenuated"] = boot["median"] / np.sqrt(ceiling)
                record[f"{key}__disattenuated_lo"] = boot["lo"] / np.sqrt(ceiling)
                record[f"{key}__disattenuated_hi"] = boot["hi"] / np.sqrt(ceiling)
                record[f"{key}__ceiling"] = ceiling
                record[f"{key}__n_scored"] = boot["n"]
            own = disc.own_discrimination(resid_dst, pred)
            record["discrimination_ours_residual_mean"] = own["mean"]
            record["discrimination_ours_residual_median"] = own["median"]
            records.append(record)
            tables.append(
                pd.DataFrame(
                    {
                        "fold": np.zeros(rows_dst.size, dtype=np.int64),
                        "residual_weighted": cells[("residual", "weighted")],
                        "residual_unweighted": cells[("residual", "unweighted")],
                        "raw_weighted": cells[("raw", "weighted")],
                        "raw_unweighted": cells[("raw", "unweighted")],
                        "cell_line": record["cell_line"],
                        "scope": "transfer",
                        "split": "cross_line",
                        "arm": spec.name,
                        "translator": translator,
                        "perturbation": dv.pert_target[rows_dst],
                    }
                )
            )
            print(
                f"  [{record['cell_line']}] {spec.name}/{translator} "
                f"resid_w_dis={record['residual__weighted__disattenuated']:.4f} "
                f"({record['seconds']}s)",
                flush=True,
            )
    return records, tables


def _bind(spec: C.ArmSpec, bundle: LineBundle, row_lookup: torch.Tensor) -> armsmod.ArmFeatures:
    """ArmFeatures over scope-local row indices."""
    arm = armsmod.ArmFeatures(spec, bundle) if spec.blocks else None
    if arm is None:
        return arm
    arm._rows = row_lookup  # noqa: SLF001
    base_batch = arm.batch

    def batch(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        return base_batch(row_lookup[rows], cols)

    arm.batch = batch  # type: ignore[method-assign]
    return arm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lines", default="hepg2,jurkat")
    ap.add_argument("--scopes", default="full,common")
    ap.add_argument("--splits", default="grouped,random")
    ap.add_argument("--translators", default="mlp,ridge")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--arms", default="", help="comma-separated arm names; empty = every arm")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--no-cell-eval", action="store_true")
    ap.add_argument("--no-shuffle-null", action="store_true")
    ap.add_argument("--transfer", action="store_true", help="cross-cell-line direction only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    root = Path(args.bundle) if args.bundle else paths.bundle_root()
    out_dir = paths.ensure(Path(args.out) if args.out else paths.results_dir() / "sweep")
    counter = {"i": 0}

    def task_filter(_key) -> bool:
        take = counter["i"] % args.n_shards == args.shard
        counter["i"] += 1
        return take

    records: list[dict] = []
    tables: list[pd.DataFrame] = []
    if args.transfer:
        a, b = (x.strip() for x in args.lines.split(","))
        ba = LineBundle(a, root, args.device)
        bb = LineBundle(b, root, args.device)
        translators = tuple(x.strip() for x in args.translators.split(","))
        only = tuple(x for x in args.arms.split(",") if x)
        for src, dst in ((ba, bb), (bb, ba)):
            r, t = run_transfer(src, dst, translators, only)
            records += r
            tables += t
        if records:
            pd.DataFrame(records).to_parquet(
                out_dir / f"runs_shard{args.shard}.parquet", index=False)
            pd.concat(tables, ignore_index=True).to_parquet(
                out_dir / f"per_perturbation_shard{args.shard}.parquet", index=False)
        print(json.dumps({"n_runs": len(records), "out": str(out_dir)}))
        return 0
    for line in args.lines.split(","):
        line = line.strip()
        if not line:
            continue
        bundle = LineBundle(line, root, args.device)
        print(f"=== {line}: {bundle.n_pert} perturbations x {bundle.n_readout} genes ===", flush=True)
        for scope in args.scopes.split(","):
            for split in args.splits.split(","):
                if scope == "common" and split == "random":
                    continue  # the matched four-way ranking runs on the primary split
                r, t = run_cell(
                    bundle,
                    scope.strip(),
                    split.strip(),
                    task_filter,
                    out_dir,
                    args.shard,
                    args.n_shards,
                    tuple(x.strip() for x in args.translators.split(",")),
                    not args.no_cell_eval,
                    not args.no_shuffle_null,
                    tuple(x for x in args.arms.split(",") if x),
                    args.save_predictions,
                )
                records += r
                tables += t
        del bundle
        torch.cuda.empty_cache() if args.device == "cuda" else None

    if records:
        pd.DataFrame(records).to_parquet(out_dir / f"runs_shard{args.shard}.parquet", index=False)
        pd.concat(tables, ignore_index=True).to_parquet(
            out_dir / f"per_perturbation_shard{args.shard}.parquet", index=False
        )
    print(json.dumps({"n_runs": len(records), "out": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
