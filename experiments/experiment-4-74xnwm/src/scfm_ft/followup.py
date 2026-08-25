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

    python -m scfm_ft.followup direct --scores DIR --checkpoints DIR
    python -m scfm_ft.followup crossfit --scores DIR --runs own_trainable,own_trainable__f23,...
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
        pred = z["flagged"].astype(np.float64) - z["control_mean_log1p"].astype(np.float64)[None, :]
        res = score.score_direct(
            bundle_root, line, pred, Path(out_dir), run_name, device=device,
            eval_folds=heldout, baseline="measured_control_mean",
        )
        head = res.get("residual_weighted") or {}
        rows.append(
            {
                "run": d.name,
                "cell_line": line,
                "arm": run_name,
                "heldout_folds": heldout,
                "baseline": "measured_control_mean",
                "disattenuated": head.get("disattenuated"),
                "lo": head.get("disattenuated_lo"),
                "hi": head.get("disattenuated_hi"),
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
        cfg_folds = _heldout_of(scores_dir, cell_line, run)
        take = [x for x in cfg_folds if x not in covered]
        covered.update(take)
        sub = t[
            (t["arm"] == f"ft_{run}_resp")
            & (t["translator"] == translator)
            & (t["fold"].astype(int).isin(take))
        ]
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


def _heldout_of(scores_dir: Path, cell_line: str, run: str) -> list[int]:
    f = Path(scores_dir) / f"summary_{cell_line}__{run}.json"
    if f.exists():
        d = json.loads(f.read_text())
        if d.get("heldout_folds"):
            return [int(x) for x in d["heldout_folds"]]
    return list(C.MODEL_HELDOUT_FOLDS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("direct")
    d.add_argument("--bundle", default=None)
    d.add_argument("--checkpoints", required=True)
    d.add_argument("--out", default=None)
    d.add_argument("--device", default="cuda")

    x = sub.add_parser("crossfit")
    x.add_argument("--bundle", default=None)
    x.add_argument("--scores", required=True)
    x.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    x.add_argument("--runs", required=True, help="comma-separated run names, main run first")
    x.add_argument("--arm", required=True)
    x.add_argument("--out", default=None)
    x.add_argument("--device", default="cuda")

    a = ap.parse_args(argv)
    bundle_root = Path(a.bundle) if a.bundle else paths.bundle_root()
    out = Path(a.out) if a.out else paths.scores_dir()
    if a.cmd == "direct":
        rows = rescore_direct(bundle_root, Path(a.checkpoints), out, a.device)
        (out / "direct_rescored.json").write_text(json.dumps(rows, indent=2, default=str))
        return 0
    crossfit(
        bundle_root, Path(a.scores), a.cell_line,
        [r for r in a.runs.split(",") if r], a.arm, out, a.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
