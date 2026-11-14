"""Two things that need the bundle but not a GPU-trained model.

``direct`` rescores each fine-tuned model's own predicted response against the
published evaluation's baseline -- the measured mean control profile -- instead
of the model's unflagged forward pass. The unflagged query is never trained
(every fine-tuning row carries a perturbation flag), so it is an untrained
baseline and cannot carry a claim that the model predicts better.

``crossfit`` pools the per-perturbation scores of several models of the same
arm, each fitted with a different pair of folds withheld, so that every
perturbation in the screen is scored by a model that never saw it. The pooled
number is read against the whole-screen ceiling and is therefore comparable to
run 3's published five-fold numbers, which the two-fold restriction is not.

``lastepoch`` reads the response features off a run's LAST epoch rather than
the epoch the recipe selects. The recipe selects on held-out masked MSE, which
in one of the two screens is lowest after a single epoch; that leaves the
objection that the features might appear later, at a checkpoint the selection
rule rejects. This answers it directly from the saved ``model_final.pt``, with
no retraining.

``rescore`` does the same from ``model_best.pt``: rebuild the model, read both
query families off it and score them again. It exists because the tagged runs'
first scoring lost the response arms to a name collision in the bundle reader.

    python -m scfm_ft.followup direct --scores DIR --checkpoints DIR
    python -m scfm_ft.followup crossfit --scores DIR --runs own_trainable,own_trainable__f23,...
    python -m scfm_ft.followup lastepoch --checkpoints DIR --cell-line hepg2 --run own_trainable
    python -m scfm_ft.followup rescore --checkpoints DIR --cell-line hepg2 --run own_trainable__f23
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import paths, score
from .sibling import run3


def _bundle(bundle_root: Path, cell_line: str, device: str):
    LineBundle = run3("translate.data").LineBundle
    return LineBundle(cell_line, root=Path(bundle_root), device=device)


# --------------------------------------------------------------------------- #
# the model's own prediction, against the baseline the published evaluation uses
# --------------------------------------------------------------------------- #
def rescore_direct(
    bundle_root: Path, ckpt_root: Path, out_dir: Path, device: str = "cuda"
) -> list[dict]:
    rows = []
    for d in sorted(Path(ckpt_root).iterdir()):
        npz = d / "direct_prediction.npz"
        if not npz.exists():
            continue
        line, _, run_name = d.name.partition("__")
        if line not in C.CELL_LINES:
            continue
        z = np.load(npz, allow_pickle=False)
        if "flagged" not in z or "control_mean_log1p" not in z:
            rows.append({"run": d.name, "skipped": "no flagged profile saved"})
            continue
        cfg = json.loads((d / "train_config.json").read_text()) if (
            d / "train_config.json"
        ).exists() else {}
        heldout = cfg.get("model_heldout_folds") or list(C.MODEL_HELDOUT_FOLDS)
        # both reference points, from the same saved forward pass: the measured
        # control profile the published evaluation uses, and the model's own
        # no-flag output. Scoring them together keeps the two columns of the
        # report on one code path.
        ctrl = z["control_mean_log1p"].astype(np.float64)[None, :]
        preds = {"measured_control_mean": z["flagged"].astype(np.float64) - ctrl}
        if "unflagged" in z:
            preds["unflagged_model_output"] = (
                z["flagged"].astype(np.float64)
                - z["unflagged"].astype(np.float64)[None, :]
            )
        for baseline, pred in preds.items():
            res = score.score_direct(
                bundle_root, line, pred, Path(out_dir), run_name, device=device,
                eval_folds=heldout, baseline=baseline,
            )
            cells = {
                (r["target"], r["weighting"]): r
                for r in res.get("all_cells") or []
            }
            head = cells.get(("residual", "weighted"), {})
            raw = cells.get(("raw", "weighted"), {})
            cen = cells.get(("centered", "weighted"), {})
            rows.append(
                {
                    "run": d.name,
                    "cell_line": line,
                    "arm": run_name,
                    "heldout_folds": heldout,
                    "baseline": baseline,
                    "disattenuated": head.get("disattenuated"),
                    "lo": head.get("disattenuated_lo"),
                    "hi": head.get("disattenuated_hi"),
                    "raw": raw.get("disattenuated"),
                    "raw_lo": raw.get("disattenuated_lo"),
                    "raw_hi": raw.get("disattenuated_hi"),
                    "centered": cen.get("disattenuated"),
                    "centered_lo": cen.get("disattenuated_lo"),
                    "centered_hi": cen.get("disattenuated_hi"),
                    "n_scored": head.get("n_scored"),
                }
            )
            print(json.dumps(rows[-1], default=str), flush=True)
    return rows


# --------------------------------------------------------------------------- #
# every perturbation scored by a model that never saw it
# --------------------------------------------------------------------------- #
def crossfit(
    bundle_root: Path,
    scores_dir: Path,
    cell_line: str,
    runs: list[str],
    arm_stem: str,
    out_dir: Path,
    device: str = "cuda",
    reference_arms: tuple[str, ...] = ("coexp", "scgpt_resp", "scgpt_emb", "node"),
    translator: str = "mlp",
    ckpt_root: Path | None = None,
) -> dict:
    """Pool one arm's held-out rows across the models fitted on complementary folds."""
    C3 = run3("config")
    metrics = run3("metrics")
    runnermod = run3("translate.runner")
    splitsmod = run3("translate.splits")

    bundle = _bundle(bundle_root, cell_line, device)
    rows_idx = bundle.scope_rows(C.SCOPE, C3.PRIMARY_ANCHOR)
    import torch

    effect_rows = bundle.effect[torch.as_tensor(rows_idx, device=bundle.effect.device)].cpu().numpy()
    info = splitsmod.make_folds(
        effect_rows, C.SPLIT, C3.N_FOLDS, C3.SEED, C3.GROUP_CORR_THRESHOLD
    )
    ref = runnermod.fold_reference(bundle, rows_idx, info["fold"], C3.N_FOLDS)
    groups = info["groups"]
    all_folds = np.arange(C3.N_FOLDS)
    ceilings = score.restricted_ceilings(bundle, ref, rows_idx, all_folds)

    tables, covered, provenance = [], set(), []
    for run in runs:
        f = Path(scores_dir) / f"per_pert_{cell_line}__{run}.parquet"
        if not f.exists():
            raise RuntimeError(f"missing per-perturbation table for {cell_line}/{run}: {f}")
        t = pd.read_parquet(f)
        cfg_folds = _heldout_of(scores_dir, cell_line, run, ckpt_root)
        take = [x for x in cfg_folds if x not in covered]
        covered.update(take)
        want = f"ft_{C.model_stem(run)}_resp"
        if want not in set(t["arm"].unique()):
            raise RuntimeError(
                f"{cell_line}/{run}: no {want} rows in {f.name}; the run's response "
                "features were never scored, so it cannot contribute to the pool"
            )
        sub = t[
            (t["arm"] == want)
            & (t["translator"] == translator)
            & (t["fold"].astype(int).isin(take))
        ]
        if take and sub.empty:
            raise RuntimeError(
                f"{cell_line}/{run}: no rows in folds {take} for {want}"
            )
        tables.append(sub)
        provenance.append({"run": run, "folds_used": take, "n_rows": int(sub.shape[0])})
    pooled = pd.concat(tables, ignore_index=True)
    missing = sorted(set(all_folds.tolist()) - covered)

    # the pooled rows are a subset of the scope in fold order; index the group
    # vector by the same perturbations rather than by position
    pert_order = {p: i for i, p in enumerate(bundle.pert_target[rows_idx])}
    idx = np.array([pert_order[p] for p in pooled["perturbation"].to_numpy()])
    g = np.asarray(groups)[idx]

    out: dict = {
        "cell_line": cell_line,
        "arm": arm_stem,
        "runs": provenance,
        "folds_not_covered": missing,
        "n_scored": int(pooled.shape[0]),
        "n_groups": int(np.unique(g).size),
        "ceilings": ceilings,
        "arms": {},
    }
    records = []
    for target in ("residual", "centered"):
        for weighting in ("weighted", "unweighted"):
            col = f"{target}_{weighting}"
            boot = metrics.bootstrap_median_grouped(
                pooled[col].to_numpy(), g, C3.BOOTSTRAP_RESAMPLES, C3.BOOTSTRAP_CI, C3.SEED
            )
            ceil = ceilings[f"{target}_{weighting}"]
            records.append(
                {
                    "cell_line": cell_line, "arm": f"ft_{arm_stem}_resp", "scope": "cross_fitted",
                    "target": target, "weighting": weighting,
                    "attenuated": boot["median"],
                    "disattenuated": boot["median"] / np.sqrt(ceil),
                    "disattenuated_lo": boot["lo"] / np.sqrt(ceil),
                    "disattenuated_hi": boot["hi"] / np.sqrt(ceil),
                    "ceiling": ceil, "n_scored": boot["n"], "n_groups": boot["n_groups"],
                }
            )

    # the reference arms do not depend on which folds a model saw, so their
    # whole-screen score is read straight off one run's table
    base = pd.read_parquet(Path(scores_dir) / f"per_pert_{cell_line}__{runs[0]}.parquet")
    for arm in reference_arms:
        sub = base[(base["arm"] == arm) & (base["translator"] == translator)]
        if sub.empty:
            continue
        sidx = np.array([pert_order[p] for p in sub["perturbation"].to_numpy()])
        gs = np.asarray(groups)[sidx]
        for target in ("residual", "centered"):
            for weighting in ("weighted", "unweighted"):
                col = f"{target}_{weighting}"
                boot = metrics.bootstrap_median_grouped(
                    sub[col].to_numpy(), gs, C3.BOOTSTRAP_RESAMPLES, C3.BOOTSTRAP_CI, C3.SEED
                )
                ceil = ceilings[f"{target}_{weighting}"]
                records.append(
                    {
                        "cell_line": cell_line, "arm": arm, "scope": "whole_screen",
                        "target": target, "weighting": weighting,
                        "attenuated": boot["median"],
                        "disattenuated": boot["median"] / np.sqrt(ceil),
                        "disattenuated_lo": boot["lo"] / np.sqrt(ceil),
                        "disattenuated_hi": boot["hi"] / np.sqrt(ceil),
                        "ceiling": ceil, "n_scored": boot["n"], "n_groups": boot["n_groups"],
                    }
                )

    frame = pd.DataFrame(records)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    frame.to_parquet(Path(out_dir) / f"crossfit_{cell_line}__{arm_stem}.parquet")
    pooled.to_parquet(Path(out_dir) / f"crossfit_per_pert_{cell_line}__{arm_stem}.parquet")
    out["records"] = records
    (Path(out_dir) / f"crossfit_{cell_line}__{arm_stem}.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2, default=str),
          flush=True)
    return out


# --------------------------------------------------------------------------- #
# the last epoch, not the selected one
# --------------------------------------------------------------------------- #
def rescore(
    bundle_root: Path,
    ckpt_root: Path,
    cell_line: str,
    run_name: str,
    scores_out: Path,
    work: Path,
    esm2_path: Path | None = None,
    device: str = "cuda",
    state: str = "best",
    label: str | None = None,
) -> dict:
    """Re-read a saved run's response features and score them again.

    Everything except which state dict is loaded is the main run's code path:
    the same model build, the same two query families, the same block writer
    and the same scorer on the same held-out folds. Two uses:

    ``state="best"``   re-score a run whose scoring in the training job lost the
                       response arms, from the selected epoch's saved weights.
    ``state="final"``  read a run's LAST epoch instead of the selected one.
    """
    import torch

    from . import run_arm as ra
    from .scgpt_pert import checkpoint as ckptmod
    from .scgpt_pert import model as modelmod

    arm = C.ARMS_BY_NAME[run_name.split("__", 1)[0]]
    src = Path(ckpt_root) / f"{cell_line}__{run_name}"
    state_file = src / ("model_final.pt" if state == "final" else "model_best.pt")
    if not state_file.exists():
        raise RuntimeError(f"no saved {state} state at {state_file}")
    cfg = json.loads((src / "train_config.json").read_text())
    hist = json.loads((src / "loss_history.json").read_text())
    heldout = [int(x) for x in (cfg.get("model_heldout_folds") or C.MODEL_HELDOUT_FOLDS)]
    epochs_run = len(hist.get("history", []))
    selected = int(hist.get("best_epoch", 0))

    readout = pd.read_parquet(paths.readout_table(cell_line))
    targets = ra.load_targets(bundle_root, cell_line)
    cache = paths.checkpoint_cache()
    ckptmod.fetch(cache)
    if arm.table == "scgpt":
        table = None
    else:
        table, _ = ra.substituted_table(
            arm, Path(esm2_path) if esm2_path else paths.esm2_table_file(),
            targets["readout_gene_id"],
        )
    positions = readout["scgpt_position"].to_numpy(dtype=np.int64)
    model, _ = modelmod.build_arm_model(arm, cache, table, positions, device)
    state = torch.load(state_file, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state dict does not match the rebuilt model: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )

    out_name = label or run_name
    block = ra.extract_stage(model, "final", out_name, readout, targets, device)
    del model
    torch.cuda.empty_cache()

    all_rows = np.arange(targets["pert_target"].size)
    bdir = score.stage_bundle(bundle_root, Path(work) / "bundle", cell_line)
    installed = [
        score.write_block(
            bdir, cell_line, block["value_key"], C.ANCHOR_VALUE, all_rows, block["gene_table"],
            block["value_forms"], block["values"], block["value_keep"],
        ),
        score.write_block(
            bdir, cell_line, block["flag_key"], C.ANCHOR_VALUE, all_rows, block["gene_table"],
            block["flag_forms"], block["flag_values"], block["flag_value_keep"],
        ),
    ]
    summary = score.score_bundle(
        bdir, cell_line, out_name, Path(scores_out), False, device=device, eval_folds=heldout,
    )
    summary["installed_blocks"] = installed
    summary["heldout_folds"] = heldout
    summary["read_from"] = str(state_file)
    summary["source_run"] = run_name
    summary["epochs_run"] = epochs_run
    summary["selected_epoch"] = selected
    summary["note"] = (
        "features read from the last epoch, which the recipe's selection rule rejected"
        if state == "final"
        else "features re-read from the selected epoch's saved weights"
    )
    (Path(scores_out) / f"summary_{cell_line}__{out_name}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "bundle_inventory"},
            indent=2, default=str,
        )[:3000],
        flush=True,
    )
    return summary


# --------------------------------------------------------------------------- #
# how much of the predicted response depends on which gene was knocked down
# --------------------------------------------------------------------------- #
def spread_row(
    flagged: np.ndarray, unflagged: np.ndarray, control_mean: np.ndarray,
    cell_line: str, run_name: str,
) -> dict:
    """How perturbation-specific one model's own prediction is.

    Run 3 reports, for the measured effects, the median over perturbations of
    the fraction of a perturbation's variance across genes that survives
    subtracting the mean effect. The same statistic is computed here on the
    model's predicted response, so the two are read on the same scale: a model
    that predicts the same profile whatever gene is knocked down scores near
    zero however well it reproduces the average knockdown.
    """
    row = {"run": f"{cell_line}__{run_name}", "cell_line": cell_line, "arm": run_name}
    f = np.asarray(flagged, dtype=np.float64)
    for label, pred in (
        ("vs_measured_control_mean", f - np.asarray(control_mean, dtype=np.float64)[None, :]),
        ("vs_unflagged_forward_pass", f - np.asarray(unflagged, dtype=np.float64).reshape(1, -1)),
    ):
        resid = pred - pred.mean(axis=0, keepdims=True)
        frac = np.var(resid, axis=1) / np.maximum(np.var(pred, axis=1), 1e-12)
        # how alike two perturbations' predicted profiles are, on a bounded
        # subsample so the pairwise matrix stays small
        rng = np.random.default_rng(C.SEED)
        take = (
            rng.choice(pred.shape[0], 100, replace=False)
            if pred.shape[0] > 100
            else np.arange(pred.shape[0])
        )
        sub = pred[take]
        sub = sub - sub.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(sub, axis=1, keepdims=True)
        cor = (sub / np.maximum(norm, 1e-12)) @ (sub / np.maximum(norm, 1e-12)).T
        iu = np.triu_indices(cor.shape[0], k=1)
        row[f"{label}__median_residual_fraction_of_variance"] = float(np.median(frac))
        row[f"{label}__median_pairwise_profile_correlation"] = float(np.median(cor[iu]))
    return row


def prediction_spread(ckpt_root: Path, out_file: Path | None = None) -> list[dict]:
    """``spread_row`` for every run under ``ckpt_root``.

    Reads each run's saved prediction matrices where they are available and its
    ``prediction_spread.json`` where only the metadata was staged, so the same
    table can be built from a full checkpoint tree or from the training jobs'
    small archives.
    """
    rows = []
    for d in sorted(Path(ckpt_root).iterdir()):
        line, _, run_name = d.name.partition("__")
        if line not in C.CELL_LINES:
            continue
        npz, pre = d / "direct_prediction.npz", d / "prediction_spread.json"
        if npz.exists():
            z = np.load(npz, allow_pickle=False)
            if "flagged" not in z or "control_mean_log1p" not in z:
                continue
            row = spread_row(
                z["flagged"], z["unflagged"], z["control_mean_log1p"], line, run_name
            )
        elif pre.exists():
            row = json.loads(pre.read_text())
        else:
            continue
        rows.append(row)
        print(json.dumps(row), flush=True)
    if out_file is not None:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(Path(out_file))
    return rows


def _heldout_of(
    scores_dir: Path, cell_line: str, run: str, ckpt_root: Path | None = None
) -> list[int]:
    """Which folds a run withheld from training.

    Recorded in two places by the training job. The cross-fitted runs withhold
    different folds from the main run, so guessing the default here would
    silently score a model on perturbations it trained on.
    """
    f = Path(scores_dir) / f"summary_{cell_line}__{run}.json"
    if f.exists():
        d = json.loads(f.read_text())
        if d.get("heldout_folds"):
            return [int(x) for x in d["heldout_folds"]]
    if ckpt_root is not None:
        g = Path(ckpt_root) / f"{cell_line}__{run}" / "train_config.json"
        if g.exists():
            d = json.loads(g.read_text())
            if d.get("model_heldout_folds"):
                return [int(x) for x in d["model_heldout_folds"]]
    raise RuntimeError(
        f"cannot tell which folds {cell_line}/{run} withheld: no "
        f"summary_{cell_line}__{run}.json under {scores_dir} and no train_config.json"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("direct")
    d.add_argument("--bundle", default=None)
    d.add_argument("--checkpoints", required=True)
    d.add_argument("--out", default=None)
    d.add_argument("--device", default="cuda")

    for name in ("lastepoch", "rescore"):
        le = sub.add_parser(name)
        le.add_argument("--bundle", default=None)
        le.add_argument("--checkpoints", required=True)
        le.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
        le.add_argument("--run", default="own_trainable", help="checkpoint directory suffix")
        le.add_argument("--scores-out", default=None)
        le.add_argument("--work", default="scratch/rescore")
        le.add_argument("--esm2", default=None)
        le.add_argument("--device", default="cuda")

    sp = sub.add_parser("spread")
    sp.add_argument("--checkpoints", required=True)
    sp.add_argument("--out", default=None)

    x = sub.add_parser("crossfit")
    x.add_argument("--bundle", default=None)
    x.add_argument("--scores", required=True)
    x.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    x.add_argument("--runs", required=True, help="comma-separated run names, main run first")
    x.add_argument("--arm", required=True)
    x.add_argument("--checkpoints", default=None)
    x.add_argument("--out", default=None)
    x.add_argument("--device", default="cuda")

    a = ap.parse_args(argv)
    if a.cmd == "spread":
        prediction_spread(
            Path(a.checkpoints),
            Path(a.out) if a.out else paths.scores_dir() / "prediction_spread.parquet",
        )
        return 0
    bundle_root = Path(a.bundle) if a.bundle else paths.bundle_root()
    if a.cmd in ("lastepoch", "rescore"):
        last = a.cmd == "lastepoch"
        rescore(
            bundle_root, Path(a.checkpoints), a.cell_line, a.run,
            Path(a.scores_out) if a.scores_out else paths.scores_dir(),
            Path(a.work), Path(a.esm2) if a.esm2 else None, a.device,
            state="final" if last else "best",
            label=f"{a.run}__lastepoch" if last else None,
        )
        return 0
    out = Path(a.out) if a.out else paths.scores_dir()
    if a.cmd == "direct":
        rows = rescore_direct(bundle_root, Path(a.checkpoints), out, a.device)
        (out / "direct_rescored.json").write_text(json.dumps(rows, indent=2, default=str))
        return 0
    crossfit(
        bundle_root, Path(a.scores), a.cell_line,
        [r for r in a.runs.split(",") if r], a.arm, out, a.device,
        ckpt_root=Path(a.checkpoints) if a.checkpoints else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
