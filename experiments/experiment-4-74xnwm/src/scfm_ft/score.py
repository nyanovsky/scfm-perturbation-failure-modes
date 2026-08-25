"""Score a fine-tuned checkpoint with run 3's rule, on the folds it never saw.

Everything scientific is run 3's: the folds, the leave-one-fold-out mean, the
reliability weights, the translator, the feature standardization, the arm
definitions, the correlation grid and the cluster bootstrap. Run 4 supplies
three things and nothing else.

1. The bundle a run gets: run 3's own bundle for this cell line, plus the
   fine-tuned model's gene table and response features written in run 3's own
   block format, so ``resolve_arms`` picks them up as it picks up any model.
2. The arm list, because a fine-tuned model has arms run 3 never had.
3. The restriction to the evaluation folds. A fine-tuned model saw three of the
   five folds, so only the other two carry a number. The restriction is applied
   after run 3's runner has produced its per-perturbation scores: the same
   correlations, the same weights, read on the subset the model is entitled to
   be read on, with the ceiling recomputed on that subset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .sibling import run3

#: the bundle files a run 4 scoring job needs from run 3
BUNDLE_FILES = (
    "targets.npz",
    "coexpr.npz",
    "reference.json",
    "emb__scgpt.npz",
    "features/scgpt__mean_control.npz",
)


def stage_bundle(staged_root: Path, work: Path, cell_line: str) -> Path:
    """Copy the read-only staged bundle into a writable tree we can add to."""
    src = Path(staged_root) / cell_line
    dst = Path(work) / cell_line
    (dst / "features").mkdir(parents=True, exist_ok=True)
    for rel in BUNDLE_FILES:
        s = src / rel
        if not s.exists():
            raise RuntimeError(f"bundle file missing: {s}")
        shutil.copy2(s, dst / rel)
    return Path(work)


def write_block(
    bundle_dir: Path,
    cell_line: str,
    model_key: str,
    anchor: str,
    pert_index: np.ndarray,
    embedding: np.ndarray,
    forms: dict[str, np.ndarray],
    values: np.ndarray,
    value_keep: np.ndarray,
) -> dict:
    """One model's gene table and response features, in run 3's block format."""
    base = Path(bundle_dir) / cell_line
    (base / "features").mkdir(parents=True, exist_ok=True)
    np.savez(base / f"emb__{model_key}.npz", embedding=np.asarray(embedding, dtype=np.float32))
    payload = {
        "pert_index": np.asarray(pert_index, dtype=np.int64),
        "values": np.asarray(values, dtype=np.float64),
        "value_keep": np.asarray(value_keep, dtype=bool),
    }
    for form, arr in forms.items():
        payload[form] = np.asarray(arr, dtype=np.float32)
    np.savez(base / f"features/{model_key}__{anchor}.npz", **payload)
    return {
        "model_key": model_key,
        "anchor": anchor,
        "forms": sorted(forms),
        "embedding_dim": int(np.asarray(embedding).shape[1]),
        "n_perturbed": int(np.asarray(pert_index).size),
        "values": [float(v) for v in np.asarray(values)],
    }


def arm_names(arm_name: str, with_coexp_shuffle: bool) -> dict:
    """Which arms this job scores, and which get a shuffled-label null.

    The arm name embeds the bundle model key, not the run name: a tagged run's
    key has to survive run 3's ``split("__", 1)`` on the feature file name.
    """
    stem = C.model_stem(arm_name)
    ft, ftq = f"ft_{stem}", f"ftq_{stem}"
    pre, preq = f"pre_{stem}", f"preq_{stem}"
    main = [
        # run 3's arms, re-run in this job on the same restricted folds
        "mean_only",
        "coexp",
        "node",
        "coexp_node",
        "null_randpair",
        # the matched-dimension null for every 512-wide gene-table arm here
        "null_randgene_d1024",
        "scgpt_resp",
        "scgpt_emb",
        "scgpt_full",
        # the fine-tuned checkpoint, value-substitution query (run 2's family)
        f"{ft}_resp",
        f"{ft}_emb",
        f"{ft}_full",
        # the fine-tuned checkpoint, its own perturbation query
        f"{ftq}_resp",
        f"{ftq}_full",
        # the same architecture before any gradient step
        f"{pre}_resp",
        f"{pre}_emb",
        f"{preq}_resp",
    ]
    shuffle_bases = [f"{ft}_resp", f"{ftq}_resp"]
    if with_coexp_shuffle:
        shuffle_bases.append("coexp")
    return {"main": main, "shuffle_bases": shuffle_bases, "models": [ft, ftq, pre, preq]}


def restricted_ceilings(bundle, ref: dict, rows: np.ndarray, eval_folds) -> dict:
    """Run 3's ceilings, aggregated over the evaluation folds only.

    Same estimator as ``runner.fold_reference``: the split-half agreement of the
    measured response on the held-out rows of each fold, Spearman-Brown
    corrected, median over the pooled per-perturbation values. Restricted here
    because a fine-tuned model is scored on two of the five folds and the
    disattenuation has to be read against those folds' reproducibility.
    """
    metrics = run3("metrics")
    half0, half1 = bundle.half0[rows], bundle.half1[rows]
    self_cols = bundle.self_col_np[rows]
    keys = (
        "residual_weighted",
        "residual_unweighted",
        "raw_weighted",
        "raw_unweighted",
        "centered_weighted",
        "centered_unweighted",
    )
    acc = {k: [] for k in keys}
    for f in eval_folds:
        info = ref["folds"][int(f)]
        te, tr = info["test"], info["train"]
        w = info["weights"]
        wmat = np.broadcast_to(w, (te.size, half0.shape[1])).copy()
        wmat[np.arange(te.size), self_cols[te]] = 0.0
        ones = np.ones_like(wmat)
        ones[np.arange(te.size), self_cols[te]] = 0.0
        r0, _ = metrics.residualize(half0, tr)
        r1, _ = metrics.residualize(half1, tr)
        acc["residual_weighted"].append(metrics.weighted_pearson_rows(r0[te], r1[te], wmat))
        acc["residual_unweighted"].append(metrics.weighted_pearson_rows(r0[te], r1[te], ones))
        acc["raw_weighted"].append(metrics.weighted_pearson_rows(half0[te], half1[te], wmat))
        acc["raw_unweighted"].append(metrics.weighted_pearson_rows(half0[te], half1[te], ones))
        c0 = r0[te] - r0[te].mean(axis=0, keepdims=True)
        c1 = r1[te] - r1[te].mean(axis=0, keepdims=True)
        acc["centered_weighted"].append(metrics.weighted_pearson_rows(c0, c1, wmat))
        acc["centered_unweighted"].append(metrics.weighted_pearson_rows(c0, c1, ones))
    return {
        k: float(np.clip(metrics.spearman_brown(np.nanmedian(np.concatenate(v))), 1e-6, 1.0))
        for k, v in acc.items()
    }


def restrict_scores(
    per_pert: pd.DataFrame,
    fold: np.ndarray,
    groups: np.ndarray,
    eval_folds,
    ceilings: dict,
    cell_line: str,
    arm_label: str,
) -> list[dict]:
    """The scoring grid, read on the evaluation folds, with group bootstraps."""
    C3 = run3("config")
    metrics = run3("metrics")
    keep = np.isin(per_pert["fold"].to_numpy(), np.asarray(eval_folds))
    g = np.asarray(groups)[keep]
    out = []
    for target in ("residual", "centered", "raw"):
        for weighting in ("weighted", "unweighted"):
            col = f"{target}_{weighting}"
            v = per_pert[col].to_numpy()[keep]
            boot = metrics.bootstrap_median_grouped(
                v, g, C3.BOOTSTRAP_RESAMPLES, C3.BOOTSTRAP_CI, C3.SEED
            )
            ceiling = ceilings[f"{target}_{weighting}"]
            out.append(
                {
                    "cell_line": cell_line,
                    "arm_label": arm_label,
                    "target": target,
                    "weighting": weighting,
                    "attenuated": boot["median"],
                    "attenuated_lo": boot["lo"],
                    "attenuated_hi": boot["hi"],
                    "disattenuated": boot["median"] / np.sqrt(ceiling),
                    "disattenuated_lo": boot["lo"] / np.sqrt(ceiling),
                    "disattenuated_hi": boot["hi"] / np.sqrt(ceiling),
                    "ceiling": ceiling,
                    "n_scored": boot["n"],
                    "n_groups": boot["n_groups"],
                    "se": boot.get("se", float("nan")),
                    "sd_single": float(np.nanstd(v)),
                }
            )
    return out


def paired_difference(
    per_pert: pd.DataFrame,
    fold: np.ndarray,
    groups: np.ndarray,
    eval_folds,
    ceilings: dict,
    arm: str,
    against: str,
    translator: str,
    cell_line: str,
    metric: tuple[str, str] = ("residual", "weighted"),
) -> dict | None:
    """One arm minus another on the same perturbations, bootstrapped over groups.

    The minimum resolvable difference is reported beside every comparison,
    because run 4 is scored on part of run 3's perturbation set and a
    difference smaller than that number is not a finding either way.
    """
    C3 = run3("config")
    metrics = run3("metrics")
    target, weighting = metric
    col = f"{target}_{weighting}"
    # each arm's table has one row per scope row, in scope order, so the two
    # arms line up positionally and the group vector indexes the same way
    sel = per_pert["translator"] == translator
    a = per_pert[sel & (per_pert["arm"] == arm)]
    b = per_pert[sel & (per_pert["arm"] == against)]
    if a.empty or b.empty or len(a) != len(b) or len(a) != len(groups):
        return None
    if not np.array_equal(a["perturbation"].to_numpy(), b["perturbation"].to_numpy()):
        raise RuntimeError(f"{arm} and {against} are not scored on the same perturbations")
    keep = np.isin(a["fold"].to_numpy(), np.asarray(eval_folds))
    g = np.asarray(groups)[keep]
    va, vb = a[col].to_numpy()[keep], b[col].to_numpy()[keep]
    boot = metrics.bootstrap_median_difference_grouped(
        va, vb, g, C3.BOOTSTRAP_RESAMPLES, C3.BOOTSTRAP_CI, C3.SEED
    )
    ceiling = np.sqrt(ceilings[f"{target}_{weighting}"])
    mrd = {
        f"rho_{rho}": metrics.min_resolvable_difference(
            sd_single=float(np.nanstd(va)), n=int(boot["n_groups"]), paired_rho=float(rho)
        )
        / ceiling
        for rho in C3.POWER_PAIRED_RHO
    }
    return {
        "cell_line": cell_line,
        "arm": arm,
        "against": against,
        "translator": translator,
        "target": target,
        "weighting": weighting,
        "difference": boot["difference"] / ceiling,
        "lo": boot["lo"] / ceiling,
        "hi": boot["hi"] / ceiling,
        "n": boot["n"],
        "n_groups": boot["n_groups"],
        "p_two_sided": boot.get("p_two_sided", float("nan")),
        **{f"min_resolvable_difference__{k}": v for k, v in mrd.items()},
    }


def _fold_context(bundle, device: str):
    """The scope rows, folds, groups and fold reference every scorer shares."""
    import torch

    C3 = run3("config")
    runner = run3("translate.runner")
    splitsmod = run3("translate.splits")
    rows = bundle.scope_rows(C.SCOPE, C3.PRIMARY_ANCHOR)
    effect_rows = bundle.effect[torch.as_tensor(rows, device=bundle.effect.device)].cpu().numpy()
    info = splitsmod.make_folds(
        effect_rows, C.SPLIT, C3.N_FOLDS, C3.SEED, C3.GROUP_CORR_THRESHOLD
    )
    ref = runner.fold_reference(bundle, rows, info["fold"], C3.N_FOLDS)
    return rows, info, ref


def score_direct(
    bundle_dir: Path,
    cell_line: str,
    delta: np.ndarray,
    out_dir: Path,
    arm_name: str,
    device: str = "cuda",
    eval_folds=None,
    baseline: str = "unflagged_model_output",
) -> dict:
    """Score the model's own prediction, with no translator in the path.

    ``delta`` is the model's predicted perturbation response, one row per
    perturbation over the read-out genes.

    ``score_run`` takes a prediction of the RESIDUAL: it forms the raw
    prediction as ``argument + training-fold mean`` and scores that against the
    measured effect. A translator trained on residual targets satisfies that
    contract; the model does not, because it predicts the perturbed profile
    itself. Handing the model's own effect over unchanged therefore adds the
    fold mean on top of a prediction that already contains it, and the raw cell
    it produces is not a correlation with the measured effect at all.

    So the training-fold mean is removed here, per fold, before the call, which
    makes the raw cell what it should be: the predicted profile against the
    measured one, shared response included.

    Two of the three cells are then interpretable and the third is not:

    * raw, ``corr(prediction, effect)``, is the reading the published recipe is
      evaluated on and is scale-free per perturbation.
    * centred subtracts each held-out fold's own mean from both sides, so it is
      unchanged by any constant shift and asks only whether the prediction
      separates one perturbation from another.
    * residual would subtract the measured training-fold mean from both sides.
      The model's output is not on the measured log2 fold-change scale, so
      subtracting a measured profile from it mixes two scales and the result is
      not a comparison of anything. It is left in the table for completeness and
      excluded from ``meaningful_cells``; the earlier version of this function
      got it wrong in the other direction, subtracting the mean from the
      measurement only.
    """
    C3 = run3("config")
    runner = run3("translate.runner")
    LineBundle = run3("translate.data").LineBundle

    bundle = LineBundle(cell_line, root=Path(bundle_dir), device=device)
    rows, info, ref = _fold_context(bundle, device)
    fold, groups = info["fold"], info["groups"]
    eval_folds = np.asarray(C.MODEL_HELDOUT_FOLDS if eval_folds is None else eval_folds)
    ceilings = restricted_ceilings(bundle, ref, rows, eval_folds)
    pred = np.asarray(delta, dtype=np.float32)[rows]
    pred_resid = pred.copy()
    for f in range(C3.N_FOLDS):
        info = ref["folds"][f]
        pred_resid[info["test"]] = pred[info["test"]] - info["mean"][None, :]
    _, per_pert, _, _ = runner.score_run(pred_resid, ref, fold, C3.N_FOLDS)
    per_pert = per_pert.assign(
        arm="direct_prediction", translator="none", perturbation=bundle.pert_target[rows]
    )
    restricted = restrict_scores(
        per_pert, fold, groups, eval_folds, ceilings, cell_line, f"direct_prediction/{arm_name}"
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cell_line}__{arm_name}" if baseline == "unflagged_model_output" else (
        f"{cell_line}__{arm_name}__{baseline}"
    )
    pd.DataFrame(restricted).assign(
        arm="direct_prediction", translator="none", baseline=baseline
    ).to_parquet(out / f"direct_{stem}.parquet")
    per_pert.to_parquet(out / f"direct_per_pert_{stem}.parquet")
    head = [r for r in restricted if r["target"] == "residual" and r["weighting"] == "weighted"]
    return {
        "baseline": baseline,
        "note": "no translator: the model's own predicted response against the measured one",
        "meaningful_cells": ["raw", "centered"],
        "fold_mean_removed_before_scoring": True,
        "residual_note": (
            "not interpretable for this arm: the model's output is not on the measured "
            "log2 fold-change scale, so no single profile can be subtracted from both sides"
        ),
        "residual_weighted": head[0] if head else None,
        "all_cells": restricted,
    }


def score_bundle(
    bundle_dir: Path,
    cell_line: str,
    arm_name: str,
    out_dir: Path,
    with_coexp_shuffle: bool,
    device: str = "cuda",
    eval_folds=None,
) -> dict:
    """Run every arm, then read the grid on the evaluation folds."""
    C3 = run3("config")
    runner = run3("translate.runner")
    LineBundle = run3("translate.data").LineBundle

    bundle = LineBundle(cell_line, root=Path(bundle_dir), device=device)
    rows, fold_info, ref = _fold_context(bundle, device)
    fold, groups = fold_info["fold"], fold_info["groups"]
    eval_folds = np.asarray(C.MODEL_HELDOUT_FOLDS if eval_folds is None else eval_folds)
    ceilings = restricted_ceilings(bundle, ref, rows, eval_folds)

    names = arm_names(arm_name, with_coexp_shuffle)
    main = tuple(names["main"])
    shuffled = tuple(
        f"{b}+shuffled_labels_d{d}"
        for b in names["shuffle_bases"]
        for d in range(C3.N_SHUFFLE_DRAWS)
    )
    # run 3 already builds a shuffled null for the arms in its own list, so
    # asking for those again would score the same arm twice and give the
    # aggregation two rows per perturbation
    extra = tuple(b for b in names["shuffle_bases"] if b not in tuple(C3.SHUFFLE_NULL_ARMS))

    def always(_key) -> bool:
        return True

    records, tables = runner.run_cell(
        bundle, C.SCOPE, C.SPLIT, always, Path(out_dir), 0, 1,
        translators=C.TRANSLATORS, with_cell_eval=False, shuffle_null=False,
        only_arms=main, save_predictions=False,
    )
    rec2, tab2 = runner.run_cell(
        bundle, C.SCOPE, C.SPLIT, always, Path(out_dir), 0, 1,
        translators=("mlp",), with_cell_eval=False, shuffle_null=True,
        only_arms=shuffled, save_predictions=False,
        extra_shuffle_arms=extra,
    )
    records += rec2
    tables += tab2
    per_pert = pd.concat(tables, ignore_index=True)
    per_pert["fold"] = per_pert["fold"].astype(np.int64)
    n_rows = int(rows.size)
    counts = per_pert.groupby(["arm", "translator"], sort=True).size()
    bad = counts[counts != n_rows]
    if len(bad):
        raise RuntimeError(
            f"an arm was scored more than once: {bad.to_dict()} against {n_rows} perturbations"
        )
    # an arm whose feature block the bundle reader did not pick up is dropped
    # silently, which would leave the run's headline arm missing from the table
    # rather than wrong. The arms this job exists to produce are required.
    scored = set(per_pert["arm"].unique())
    required = {f"{names['models'][0]}_resp", f"{names['models'][1]}_resp"}
    absent = sorted(required - scored)
    if absent:
        raise RuntimeError(
            f"{cell_line}/{arm_name}: the fine-tuned response arms {absent} were not scored; "
            f"the bundle reader resolved {sorted(a for a in scored if a.startswith(('ft', 'pre')))}"
        )

    restricted, contrasts = [], []
    for (arm, translator), grp in per_pert.groupby(["arm", "translator"], sort=True):
        restricted += [
            {**r, "arm": arm, "translator": translator}
            for r in restrict_scores(
                grp, fold, groups, eval_folds, ceilings, cell_line, f"{arm}/{translator}"
            )
        ]
    stem = C.model_stem(arm_name)
    ft, ftq = f"ft_{stem}", f"ftq_{stem}"
    interesting = [
        f"{ft}_resp", f"{ft}_full", f"{ft}_emb", f"{ftq}_resp", f"{ftq}_full",
        f"pre_{stem}_resp", f"preq_{stem}_resp", "scgpt_resp", "scgpt_emb", "scgpt_full",
    ]
    for translator in C.TRANSLATORS:
        for a in interesting:
            for against in ("coexp", "scgpt_resp"):
                if a == against:
                    continue
                d = paired_difference(
                    per_pert, fold, groups, eval_folds, ceilings, a, against, translator, cell_line
                )
                if d is not None:
                    contrasts.append(d)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out / f"runs_{cell_line}__{arm_name}.parquet")
    per_pert.to_parquet(out / f"per_pert_{cell_line}__{arm_name}.parquet")
    pd.DataFrame(restricted).to_parquet(out / f"restricted_{cell_line}__{arm_name}.parquet")
    pd.DataFrame(contrasts).to_parquet(out / f"contrasts_{cell_line}__{arm_name}.parquet")
    summary = {
        "cell_line": cell_line,
        "arm": arm_name,
        "n_perturbations_total": int(rows.size),
        "n_scored": int(np.isin(fold, eval_folds).sum()),
        "n_groups_total": int(fold_info["n_groups"]),
        "n_groups_scored": int(np.unique(groups[np.isin(fold, eval_folds)]).size),
        "eval_folds": [int(f) for f in eval_folds],
        "fold_sizes": [int(x) for x in fold_info["fold_sizes"]],
        "ceilings_restricted": ceilings,
        "ceilings_all_folds": ref["ceilings"],
        "n_runs": len(records),
        "bundle_inventory": runner.bundle_inventory(bundle),
    }
    (out / f"scoring_{cell_line}__{arm_name}.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
