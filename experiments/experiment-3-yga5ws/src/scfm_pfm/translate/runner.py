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
    ceil = {
        k: []
        for k in (
            "residual_weighted",
            "residual_unweighted",
            "raw_weighted",
            "raw_unweighted",
            "centered_weighted",
            "centered_unweighted",
        )
    }
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
        c0 = r0[te] - r0[te].mean(axis=0, keepdims=True)
        c1 = r1[te] - r1[te].mean(axis=0, keepdims=True)
        ceil["centered_weighted"].append(metrics.weighted_pearson_rows(c0, c1, wmat))
        ceil["centered_unweighted"].append(metrics.weighted_pearson_rows(c0, c1, ones))
    ceilings = {
        k: float(np.clip(metrics.spearman_brown(np.nanmedian(np.concatenate(v))), 1e-6, 1.0))
        for k, v in ceil.items()
    }
    return {"folds": per_fold, "ceilings": ceilings, "effect": effect, "self_cols": self_cols}


def bundle_inventory(bundle: LineBundle) -> dict:
    """What this bundle actually holds, read off the loaded arrays.

    Written next to the sweep so the report describes the inputs the runs used
    rather than what a manifest written by an earlier pass claims.
    """
    feats = {}
    for (model, anchor), e in sorted(bundle.features.items()):
        feats[f"{model}__{anchor}"] = {
            "n_perturbed": int(e["pert_index"].size),
            "values": [float(v) for v in e["values"]],
            "forms": sorted([f for f in ("zero_out", "vvp", "gdt") if f in e]),
            "gdt_rank_only": bool(e.get("gdt_rank_only", False)),
        }
    return {
        "cell_line": bundle.cell_line,
        "n_perturbations": int(bundle.n_pert),
        "n_readout": int(bundle.n_readout),
        "embeddings": {m: int(v.shape[1]) for m, v in sorted(bundle.emb.items())},
        "features": feats,
        "coexpr_keys": sorted(bundle.coexpr),
        "node_features": list(bundle.node_names),
        "strata": sorted(set(bundle.pert_stratum.tolist())),
        "perturbed_genes_in_readout": int((bundle.self_col_np >= 0).sum()),
        "reference": bundle.reference,
    }


def _keep_predictions(arm: str, translator: str, scope: str, split: str) -> bool:
    """Predictions are kept for the arms the later checks and figures need, not
    for every cell of the sweep: one held-out prediction matrix is ~15 MB."""
    if translator != "mlp" or scope != "full" or split != "grouped":
        return False
    return (
        arm in {"coexp", "node", "coexp_node", "mean_only", "null_randpair"}
        or arm.endswith(("_full", "_resp"))
        or arm.endswith("+shuffled_labels_d0")
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
    # float64 for the centred pair: a prediction that barely varies across
    # perturbations leaves float32 dust behind, and a correlation of dust with
    # anything is a number with no meaning.
    true_cen = np.zeros(effect.shape, dtype=np.float64)
    pred_cen = np.zeros(effect.shape, dtype=np.float64)
    weights = np.zeros_like(effect)
    for f in range(n_folds):
        info = ref["folds"][f]
        te = info["test"]
        true_resid[te] = effect[te] - info["mean"][None, :]
        pred_raw[te] = pred_resid[te] + info["mean"][None, :]
        # Held-out perturbations share a per-gene offset: the mean subtracted
        # comes from the training fold, and a grouped fold is a whole response
        # family with a real shared response. Any prediction that is constant
        # across perturbations correlates with that offset, by luck rather than
        # by matching a perturbation. Removing the held-out fold's own mean from
        # both sides scores only what separates one perturbation from another.
        t64 = true_resid[te].astype(np.float64)
        p64 = pred_resid[te].astype(np.float64)
        true_cen[te] = t64 - t64.mean(axis=0, keepdims=True)
        pred_cen[te] = p64 - p64.mean(axis=0, keepdims=True)
        weights[te] = info["weights"][None, :]
    weights[np.arange(n), self_cols] = 0.0
    ones = np.ones_like(weights)
    ones[np.arange(n), self_cols] = 0.0

    cells = {
        ("residual", "weighted"): metrics.weighted_pearson_rows(pred_resid, true_resid, weights),
        ("residual", "unweighted"): metrics.weighted_pearson_rows(pred_resid, true_resid, ones),
        ("raw", "weighted"): metrics.weighted_pearson_rows(pred_raw, effect, weights),
        ("raw", "unweighted"): metrics.weighted_pearson_rows(pred_raw, effect, ones),
        ("centered", "weighted"): metrics.weighted_pearson_rows(pred_cen, true_cen, weights),
        ("centered", "unweighted"): metrics.weighted_pearson_rows(pred_cen, true_cen, ones),
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
            "centered_weighted": cells[("centered", "weighted")],
            "centered_unweighted": cells[("centered", "unweighted")],
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
    extra_shuffle_arms: tuple[str, ...] = (),
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
    # a model with a gene table and no response features: the protein arm
    emb_only = [m for m in sorted(bundle.emb) if m not in models]
    specs = armsmod.resolve_arms(bundle, models, C.PRIMARY_ANCHOR, include_sens, emb_only)
    if shuffle_null:
        for name in tuple(C.SHUFFLE_NULL_ARMS) + tuple(extra_shuffle_arms):
            base = [s for s in specs if s.name == name]
            if not base:
                continue
            s = base[0]
            # Several draws, not one: a single permutation is one sample from
            # the null, and the spread between draws is the thing a score has
            # to clear.
            for draw in range(C.N_SHUFFLE_DRAWS):
                specs.append(
                    C.ArmSpec(
                        f"{s.name}+shuffled_labels_d{draw}", "null", s.blocks, s.model, s.anchor,
                        note="perturbation labels permuted within the training fold",
                        rand_dim=s.rand_dim,
                        additive=s.additive,
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
            if entry is None and "resp" in spec.blocks:
                continue  # the arm needs response features this model has none of
            if entry is not None and (entry["row_map_np"][rows] < 0).any():
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
            shuffled = "+shuffled_labels" in spec.name
            draw = int(spec.name.rsplit("_d", 1)[1]) if shuffled else 0
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
                    rng = np.random.default_rng(C.SEED + 101 * (draw + 1) + f)
                    perm = np.arange(rows.size)
                    perm[tr_local] = tr_local[rng.permutation(tr_local.size)]
                    target = target[torch.as_tensor(perm, device=device)]
                    # A permuted row carries its own knockdown column with it,
                    # and that is not the column the pair index drops for this
                    # row. Those few hundred large negative values are then the
                    # one systematic thing left for a null model to fit, so set
                    # them to the training-fold mean, which is zero here.
                    target[
                        torch.arange(rows.size, device=device),
                        torch.as_tensor(local_self[perm], device=device),
                    ] = 0.0
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
                "target_source": bundle.target_source,
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


def _reference_mask(n: int, seed: int) -> np.ndarray:
    """Half the destination perturbations, drawn once.

    They are used only to estimate the destination line's mean response profile
    and its per-gene reliability weights, and are never scored; the transfer
    number is read from the other half.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    mask[rng.permutation(n)[: n // 2]] = True
    return mask


def _transfer_quantities(
    mode: str,
    eff_dst: np.ndarray,
    half0: np.ndarray,
    half1: np.ndarray,
    mean_src: np.ndarray,
    w_src: np.ndarray,
    seed: int,
) -> dict:
    """Which destination rows are scored, what is subtracted from them, and
    which per-gene weights the correlation uses.

    Predicting a held-out cell line means predicting the perturbation-specific
    part of its response, and "perturbation-specific" is defined against some
    mean. Where that mean and the reliability weights come from decides whether
    the evaluation is transductive:

    ``dest_reference``  a disjoint half of the destination perturbations, which
                        is never scored — nothing that is scored contributed to
                        the quantities used to score it.
    ``source_only``     the training line, over the shared genes: no destination
                        outcome enters at all.
    ``dest_all``        every destination perturbation, leave-one-out. This is
                        the convention used inside a cell line, kept here so the
                        cost of the other two is visible.
    """
    n = eff_dst.shape[0]
    all_rows = np.arange(n)
    if mode == "dest_all":
        _, means = metrics.residualize(eff_dst, all_rows)
        return {
            "eval_rows": all_rows,
            "means": means,
            "weights": metrics.per_gene_reliability(half0, half1, all_rows)["weights"],
            "mean_rows": all_rows,
        }
    if mode == "dest_reference":
        ref = _reference_mask(n, seed)
        mean_rows = np.flatnonzero(ref)
        return {
            "eval_rows": np.flatnonzero(~ref),
            "means": np.broadcast_to(eff_dst[mean_rows].mean(axis=0), eff_dst.shape).copy(),
            "weights": metrics.per_gene_reliability(half0, half1, mean_rows)["weights"],
            "mean_rows": mean_rows,
        }
    if mode == "source_only":
        return {
            "eval_rows": all_rows,
            "means": np.broadcast_to(mean_src, eff_dst.shape).copy(),
            "weights": np.asarray(w_src, dtype=np.float64),
            "mean_rows": all_rows,
        }
    raise ValueError(f"unknown transfer mode {mode!r}")


def _transfer_scoring(q: dict, eff_dst: np.ndarray, half0: np.ndarray, half1: np.ndarray,
                      self_dst: np.ndarray) -> dict:
    """Everything about a mode that does not depend on the arm: the weights
    matrix over the scored rows, the truth it is scored against, and the four
    split-half ceilings."""
    te = q["eval_rows"]
    means = q["means"][te]
    w = np.broadcast_to(q["weights"], (te.size, eff_dst.shape[1])).copy()
    w[np.arange(te.size), self_dst[te]] = 0.0
    ones = np.ones_like(w)
    ones[np.arange(te.size), self_dst[te]] = 0.0
    mh = q["mean_rows"]
    r0 = half0[te] - half0[mh].mean(axis=0, keepdims=True)
    r1 = half1[te] - half1[mh].mean(axis=0, keepdims=True)
    c0 = r0 - r0.mean(axis=0, keepdims=True)
    c1 = r1 - r1.mean(axis=0, keepdims=True)
    def sb(x):
        return float(np.clip(metrics.spearman_brown(np.nanmedian(x)), 1e-6, 1.0))
    ceilings = {
        "residual_weighted": sb(metrics.weighted_pearson_rows(r0, r1, w)),
        "residual_unweighted": sb(metrics.weighted_pearson_rows(r0, r1, ones)),
        "centered_weighted": sb(metrics.weighted_pearson_rows(c0, c1, w)),
        "centered_unweighted": sb(metrics.weighted_pearson_rows(c0, c1, ones)),
        "raw_weighted": sb(metrics.weighted_pearson_rows(half0[te], half1[te], w)),
        "raw_unweighted": sb(metrics.weighted_pearson_rows(half0[te], half1[te], ones)),
    }
    truth_resid = eff_dst[te] - means
    return {
        "rows": te,
        "means": means,
        "w": w,
        "ones": ones,
        "truth_resid": truth_resid,
        "truth_centered": truth_resid - truth_resid.mean(axis=0, keepdims=True),
        "ceilings": ceilings,
    }


def transfer_rows(
    bundle: LineBundle, scope: str, exclude_targets: set[str] | None = None
) -> np.ndarray:
    """Perturbations of one line that a transfer run can use.

    ``full`` is every in-scope perturbation whose own gene is a shared read-out
    column; ``common`` narrows that to the perturbations every model covers,
    which is the only scope where AIDO.Cell — whose block is a 250-gene subset —
    can be compared against the other three on the same rows; ``unseen`` keeps
    only destination perturbations whose gene was never perturbed in the
    training line.
    """
    rows = np.flatnonzero(bundle.self_col_np >= 0)
    if scope == "common":
        keep = set(bundle.scope_rows("common", C.PRIMARY_ANCHOR).tolist())
        rows = np.array([r for r in rows if int(r) in keep], dtype=np.int64)
    if exclude_targets:
        rows = np.array(
            [r for r in rows if str(bundle.pert_target[r]) not in exclude_targets], dtype=np.int64
        )
    return rows


def run_transfer(
    src: LineBundle,
    dst: LineBundle,
    translators: tuple[str, ...],
    only_arms: tuple[str, ...] = (),
    scope: str = "full",
    modes: tuple[str, ...] = C.TRANSFER_MODES,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Train on one cell line, test on the other, over the shared read-out set.

    The translator sees nothing from the test line. What the test line's
    measurements are scored against is set by ``modes``; the model is fitted
    once per arm and the same predictions are read under each of them.
    """
    shared = np.array(
        [g for g in src.readout_gene_id if g in set(dst.readout_gene_id.tolist())], dtype=object
    )
    sv, dv = src.restrict_columns(shared), dst.restrict_columns(shared)
    # ``unseen`` is the same training set as ``full``; what changes is that a
    # destination perturbation is scored only if its gene was never perturbed in
    # the training line. Without it, a fixed random code per gene scores as well
    # as a model embedding, because the two lines' responses to the SAME
    # knockdown are correlated and gene identity is enough to exploit that.
    base_scope = "full" if scope == "unseen" else scope
    rows_src = transfer_rows(sv, base_scope)
    seen = set(str(t) for t in sv.pert_target[rows_src]) if scope == "unseen" else None
    rows_dst = transfer_rows(dv, base_scope, seen)
    if rows_src.size < 50 or rows_dst.size < 50:
        return [], []
    device = str(sv.effect.device)

    eff_src = sv.effect[torch.as_tensor(rows_src, device=device)].cpu().numpy()
    eff_dst = dv.effect[torch.as_tensor(rows_dst, device=device)].cpu().numpy()
    all_src = np.arange(rows_src.size)
    rel_src = metrics.per_gene_reliability(sv.half0[rows_src], sv.half1[rows_src], all_src)
    w_src = torch.as_tensor(rel_src["weights"], device=device, dtype=torch.float32)
    resid_src, mean_src_rows = metrics.residualize(eff_src, all_src)
    mean_src = eff_src.mean(axis=0)
    y_src = torch.as_tensor(resid_src, device=device, dtype=torch.float32)

    self_src = sv.self_col_np[rows_src]
    self_dst = dv.self_col_np[rows_dst]
    half0_dst, half1_dst = dv.half0[rows_dst], dv.half1[rows_dst]
    scoring = {
        mode: _transfer_scoring(
            _transfer_quantities(
                mode, eff_dst, half0_dst, half1_dst, mean_src, rel_src["weights"], C.SEED
            ),
            eff_dst, half0_dst, half1_dst, self_dst,
        )
        for mode in modes
    }

    models = [m for m in sv.models_present(C.PRIMARY_ANCHOR) if m in dv.emb]
    models = [
        m for m in models
        if (sv.features[(m, C.PRIMARY_ANCHOR)]["row_map_np"][rows_src] >= 0).all()
        and (dv.features[(m, C.PRIMARY_ANCHOR)]["row_map_np"][rows_dst] >= 0).all()
    ]
    emb_only = [m for m in sorted(sv.emb) if m in dv.emb and m not in models]
    specs = armsmod.resolve_arms(sv, models, C.PRIMARY_ANCHOR, False, emb_only)
    records, tables = [], []
    lookup_src = torch.as_tensor(rows_src, device=device)
    lookup_dst = torch.as_tensor(rows_dst, device=device)
    print(
        f"=== transfer {src.cell_line}->{dst.cell_line} [{scope}] "
        f"{rows_src.size} train x {rows_dst.size} test perturbations, "
        f"{shared.size} shared genes, models={models}, embedding-only={emb_only} ===",
        flush=True,
    )
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
                arm_tr, y_src, w_src, all_src, np.arange(rows_dst.size), self_src,
                arm_test=arm_te, n_readout_test=int(sv.n_readout),
            )
            pred = res["prediction"]
            for mode in modes:
                s = scoring[mode]
                te = s["rows"]
                p = pred[te]
                pc = p - p.mean(axis=0, keepdims=True)
                cells = {
                    ("residual", "weighted"): metrics.weighted_pearson_rows(
                        p, s["truth_resid"], s["w"]),
                    ("residual", "unweighted"): metrics.weighted_pearson_rows(
                        p, s["truth_resid"], s["ones"]),
                    ("centered", "weighted"): metrics.weighted_pearson_rows(
                        pc, s["truth_centered"], s["w"]),
                    ("centered", "unweighted"): metrics.weighted_pearson_rows(
                        pc, s["truth_centered"], s["ones"]),
                    ("raw", "weighted"): metrics.weighted_pearson_rows(
                        p + s["means"], eff_dst[te], s["w"]),
                    ("raw", "unweighted"): metrics.weighted_pearson_rows(
                        p + s["means"], eff_dst[te], s["ones"]),
                }
                record = {
                    "cell_line": f"{src.cell_line}->{dst.cell_line}",
                    "scope": f"transfer_{scope}",
                    "split": f"cross_line__{mode}",
                    "transfer_mode": mode,
                    "target_source": dv.target_source,
                    "arm": spec.name,
                    "family": spec.family,
                    "model": spec.model or "",
                    "anchor": spec.anchor or "",
                    "translator": translator,
                    "n_perturbations": int(te.size),
                    "n_train_perturbations": int(rows_src.size),
                    "n_readout": int(sv.n_readout),
                    "feature_dim": int(res.get("dim", 0)),
                    "seconds": round(time.time() - t0, 1),
                }
                for (target, weighting), values in cells.items():
                    boot = metrics.bootstrap_median(
                        values, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_CI, C.SEED)
                    ceiling = s["ceilings"][f"{target}_{weighting}"]
                    key = f"{target}__{weighting}"
                    record[f"{key}__attenuated"] = boot["median"]
                    record[f"{key}__attenuated_lo"] = boot["lo"]
                    record[f"{key}__attenuated_hi"] = boot["hi"]
                    record[f"{key}__disattenuated"] = boot["median"] / np.sqrt(ceiling)
                    record[f"{key}__disattenuated_lo"] = boot["lo"] / np.sqrt(ceiling)
                    record[f"{key}__disattenuated_hi"] = boot["hi"] / np.sqrt(ceiling)
                    record[f"{key}__ceiling"] = ceiling
                    record[f"{key}__n_scored"] = boot["n"]
                own = disc.own_discrimination(s["truth_resid"], p)
                record["discrimination_ours_residual_mean"] = own["mean"]
                record["discrimination_ours_residual_median"] = own["median"]
                records.append(record)
                tables.append(
                    pd.DataFrame(
                        {
                            "fold": np.zeros(te.size, dtype=np.int64),
                            "residual_weighted": cells[("residual", "weighted")],
                            "residual_unweighted": cells[("residual", "unweighted")],
                            "raw_weighted": cells[("raw", "weighted")],
                            "raw_unweighted": cells[("raw", "unweighted")],
                            "centered_weighted": cells[("centered", "weighted")],
                            "centered_unweighted": cells[("centered", "unweighted")],
                            "cell_line": record["cell_line"],
                            "scope": record["scope"],
                            "split": record["split"],
                            "arm": spec.name,
                            "translator": translator,
                            "perturbation": dv.pert_target[rows_dst][te],
                        }
                    )
                )
                print(
                    f"  [{record['cell_line']}/{scope}/{mode}] {spec.name}/{translator} "
                    f"resid_w_dis={record['residual__weighted__disattenuated']:.4f} "
                    f"cen_w_dis={record['centered__weighted__disattenuated']:.4f} "
                    f"({record['seconds']}s)",
                    flush=True,
                )
    return records, tables


def _bind(spec: C.ArmSpec, bundle: LineBundle, row_lookup: torch.Tensor) -> armsmod.ArmFeatures:
    """ArmFeatures over scope-local row indices."""
    suffix = "_check" if spec.name.endswith("_check") else ""
    arm = armsmod.ArmFeatures(spec, bundle, suffix) if spec.blocks else None
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
    ap.add_argument(
        "--extra-shuffle-arms",
        default="",
        help=("arms to draw the shuffled-label null for beyond config.SHUFFLE_NULL_ARMS; "
              "for a control added after that list was fixed"),
    )
    ap.add_argument("--transfer", action="store_true", help="cross-cell-line direction only")
    ap.add_argument("--transfer-scopes", default=",".join(C.TRANSFER_SCOPES))
    ap.add_argument("--transfer-modes", default=",".join(C.TRANSFER_MODES))
    ap.add_argument(
        "--targets",
        default=None,
        help="directory of recomputed targets; <line>__<tag>.npz replaces the bundle's",
    )
    ap.add_argument("--target-tag", default="", help="which recomputed target to read")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    root = Path(args.bundle) if args.bundle else paths.bundle_root()

    def targets_file(line: str) -> Path | None:
        if not args.targets or not args.target_tag:
            return None
        p = Path(args.targets) / f"{line}__{args.target_tag}.npz"
        if not p.exists():
            raise FileNotFoundError(f"recomputed target {p} not found")
        return p
    out_dir = paths.ensure(Path(args.out) if args.out else paths.results_dir() / "sweep")
    counter = {"i": 0}

    def task_filter(_key) -> bool:
        take = counter["i"] % args.n_shards == args.shard
        counter["i"] += 1
        return take

    records: list[dict] = []
    tables: list[pd.DataFrame] = []
    inventory: dict[str, dict] = {}
    if args.transfer:
        a, b = (x.strip() for x in args.lines.split(","))
        ba = LineBundle(a, root, args.device, targets_file(a))
        bb = LineBundle(b, root, args.device, targets_file(b))
        translators = tuple(x.strip() for x in args.translators.split(","))
        only = tuple(x for x in args.arms.split(",") if x)
        modes = tuple(x.strip() for x in args.transfer_modes.split(",") if x.strip())
        for scope in (x.strip() for x in args.transfer_scopes.split(",")):
            for src, dst in ((ba, bb), (bb, ba)):
                r, t = run_transfer(src, dst, translators, only, scope, modes)
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
        bundle = LineBundle(line, root, args.device, targets_file(line))
        print(
            f"=== {line}: {bundle.n_pert} perturbations x {bundle.n_readout} genes "
            f"[target={bundle.target_source}] ===",
            flush=True,
        )
        inventory[line] = bundle_inventory(bundle)
        for scope in args.scopes.split(","):
            for split in args.splits.split(","):
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
                    tuple(x for x in args.extra_shuffle_arms.split(",") if x),
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
    if inventory:
        (out_dir / f"bundle_inventory_shard{args.shard}.json").write_text(
            json.dumps(inventory, indent=2, sort_keys=True)
        )
    print(json.dumps({"n_runs": len(records), "out": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
