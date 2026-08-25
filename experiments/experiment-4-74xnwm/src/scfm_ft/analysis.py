"""Merge the per-arm scoring shards into the tables the report reads.

Every number the report states comes from one of the parquet files this writes,
and every one of those is a restriction of what run 3's scoring rule produced
inside the job that trained the model.

    PYTHONPATH=src apy -m scfm_ft.analysis --scores scratch/dl/scores \
        --checkpoints scratch/dl/checkpoints --out results/summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

#: the pre-registered cell of run 3's correlation grid, and the fold-centred
#: diagnostic reported beside it everywhere
VERDICT = ("residual", "weighted")
DIAGNOSTIC = ("centered", "weighted")
#: the model's own predicted response can be read against two reference points.
#: ``measured_control_mean`` subtracts the control cells' measured profile, which
#: is the comparison the published recipe is evaluated on; ``unflagged_model_output``
#: subtracts the model's own no-flag forward pass, which is the reference the
#: response features use. The first is primary, the second is reported beside it.
DIRECT_BASELINES = ("measured_control_mean", "unflagged_model_output")


def _read_all(root: Path, stem: str) -> pd.DataFrame:
    files = sorted(Path(root).glob(f"{stem}_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def arm_of(name: str) -> tuple[str, str, str]:
    """Split a scored arm name into (stage, arm, query family).

    ``ft_<arm>_resp``  the fine-tuned model, value-substitution query
    ``ftq_<arm>_resp`` the fine-tuned model, its own perturbation query
    ``pre_``/``preq_`` the same before any gradient step
    anything else is a reference arm and keeps its own name.
    """
    for prefix, stage, family in (
        ("ftq_", "final", "pert_flag"),
        ("ft_", "final", "value_substitution"),
        ("preq_", "init", "pert_flag"),
        ("pre_", "init", "value_substitution"),
    ):
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            for suffix in ("_resp", "_emb", "_full", "_hybrid"):
                if rest.endswith(suffix):
                    return stage, rest[: -len(suffix)], f"{family}{suffix}"
            return stage, rest, family
    return "reference", "", name


#: where each training run's durable directory lives, for the summary to cite
ARTIFACT_PREFIX = (
    "artifact://fundacion-instituto-leloir-553557/experiments/"
    "exp_01m0trpxqyf0g8pq29k774xnwm/checkpoints"
)


def training_tables(ckpt_root: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What each fine-tuning run did, read from the run's own directory.

    One row per epoch in the first table, one row per training run in the
    second, each carrying the artifact reference of the directory the weights,
    the configuration and the resolved gene table were written to.
    """
    if ckpt_root is None or not Path(ckpt_root).exists():
        return pd.DataFrame(), pd.DataFrame()
    epochs, runs = [], []
    for d in sorted(Path(ckpt_root).iterdir()):
        hist_file, cfg_file = d / "loss_history.json", d / "train_config.json"
        if not hist_file.exists():
            continue
        line, _, arm = d.name.partition("__")
        hist = json.loads(hist_file.read_text())
        cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
        for rec in hist["history"]:
            row = {
                "cell_line": line,
                "arm": arm,
                "epoch": int(rec["epoch"]),
                "train_loss": float(rec["train_loss"]),
                "val_loss": float(rec.get("val_loss", np.nan)),
                "lr": float(rec["lr"]),
                "seconds": float(rec["seconds"]),
            }
            for k, v in (rec.get("val_delta_pearson") or {}).items():
                row[f"delta_{k}"] = float(v)
            epochs.append(row)
        sub = cfg.get("gene_subsampling_counters", {})
        runs.append(
            {
                "cell_line": line,
                "arm": arm,
                "selected_epoch": int(hist.get("best_epoch", 0)),
                "epochs_run": len(hist["history"]),
                "best_val_loss": float(hist.get("best_val_loss", np.nan)),
                "selection_rule": hist.get("selection_rule", ""),
                "total_seconds": float(hist.get("total_seconds", np.nan)),
                "n_train_cells": cfg.get("n_train_cells"),
                "n_train_perturbations": (cfg.get("split") or {}).get("n_train"),
                "n_val_perturbations": (cfg.get("split") or {}).get("n_val"),
                "n_scored_perturbations": (cfg.get("split") or {}).get("n_scored"),
                "table": (cfg.get("table_report") or {}).get("table"),
                "trainable_table": (cfg.get("arm_spec") or {}).get("trainable_table"),
                "projected": (cfg.get("arm_spec") or {}).get("projected"),
                "rows_losing_perturbed_gene": (
                    sub["rows_that_would_lose_the_perturbed_gene"] / sub["rows"]
                    if sub.get("rows")
                    else None
                ),
                "artifact": f"{ARTIFACT_PREFIX}/{d.name}/",
                **{
                    f"selected_delta_{k}": float(v)
                    for k, v in (hist.get("best_val_delta_pearson") or {}).items()
                },
            }
        )
    return pd.DataFrame(epochs), pd.DataFrame(runs)


def build(scores: Path, out: Path, checkpoints: Path | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    restricted = _read_all(scores, "restricted")
    contrasts = _read_all(scores, "contrasts")
    runs = _read_all(scores, "runs")
    direct = _read_all(scores, "direct")
    if restricted.empty:
        raise RuntimeError(f"no restricted_*.parquet under {scores}")

    parts = restricted["arm"].map(arm_of)
    restricted["stage"] = [p[0] for p in parts]
    restricted["train_arm"] = [p[1] for p in parts]
    restricted["family"] = [p[2] for p in parts]
    restricted["is_shuffled"] = restricted["arm"].str.contains(r"\+shuffled_labels_d")
    restricted["base_arm"] = restricted["arm"].str.replace(
        r"\+shuffled_labels_d\d+", "", regex=True
    )

    # the fine-tuned arms are scored once per training job; the reference arms
    # are re-run inside every job, so they appear once per arm. They must agree.
    ref = restricted[
        (restricted.stage == "reference")
        & (~restricted.is_shuffled)
        & (restricted.target == VERDICT[0])
        & (restricted.weighting == VERDICT[1])
    ]
    agreement = (
        ref.groupby(["cell_line", "arm", "translator"])["disattenuated"]
        .agg(["min", "max", "count"])
        .assign(spread=lambda d: d["max"] - d["min"])
        .reset_index()
    )

    band = (
        restricted[restricted.is_shuffled]
        .groupby(["cell_line", "base_arm", "translator", "target", "weighting"])["disattenuated"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "null_mean", "std": "null_sd", "min": "null_min", "max": "null_max",
                         "count": "n_draws"})
    )

    verdict = restricted[
        (restricted.target == VERDICT[0]) & (restricted.weighting == VERDICT[1])
    ].copy()
    diagnostic = restricted[
        (restricted.target == DIAGNOSTIC[0]) & (restricted.weighting == DIAGNOSTIC[1])
    ].copy()

    epochs, training = training_tables(checkpoints)

    for name, frame in (
        ("restricted", restricted),
        ("contrasts", contrasts),
        ("runs", runs),
        ("direct", direct),
        ("null_band", band),
        ("reference_agreement", agreement),
        ("training_epochs", epochs),
        ("training_runs", training),
    ):
        if not frame.empty:
            frame.to_parquet(out / f"{name}.parquet")

    head = headline(verdict, diagnostic, contrasts, band, direct, training)
    (out / "headline.json").write_text(json.dumps(head, indent=2, default=str))
    return head


def _pick(frame: pd.DataFrame, **kw) -> pd.DataFrame:
    for k, v in kw.items():
        frame = frame[frame[k] == v]
    return frame


def _row(frame: pd.DataFrame) -> dict | None:
    if frame.empty:
        return None
    r = frame.iloc[0]
    return {
        "score": float(r["disattenuated"]),
        "lo": float(r["disattenuated_lo"]),
        "hi": float(r["disattenuated_hi"]),
        "attenuated": float(r["attenuated"]),
        "ceiling": float(r["ceiling"]),
        "n": int(r["n_scored"]),
        "n_groups": int(r["n_groups"]),
    }


def headline(
    verdict: pd.DataFrame,
    diagnostic: pd.DataFrame,
    contrasts: pd.DataFrame,
    band: pd.DataFrame,
    direct: pd.DataFrame,
    training: pd.DataFrame | None = None,
) -> dict:
    """The numbers the page states, each read from the table it comes from."""
    out: dict = {
        "metric": "residual target, reliability-weighted, ceiling-corrected per-perturbation "
        "Pearson, median over the held-out perturbations of run 3's grouped split, restricted "
        "to the folds the fine-tuned model never saw",
        "diagnostic_metric": "the fold-centred version of the same correlation, reported beside "
        "the verdict and never substituted for it",
        "model_heldout_folds": list(C.MODEL_HELDOUT_FOLDS),
        "cell_lines": {},
    }
    for line in C.CELL_LINES:
        v = _pick(verdict, cell_line=line, translator="mlp")
        d = _pick(diagnostic, cell_line=line, translator="mlp")
        if v.empty:
            continue
        entry: dict = {"reference": {}, "arms": {}, "null_band": {}}
        for arm in ("coexp", "node", "coexp_node", "scgpt_resp", "scgpt_emb", "scgpt_full",
                    "null_randpair", "null_randgene_d1024"):
            r = _row(_pick(v, arm=arm, is_shuffled=False))
            if r:
                entry["reference"][arm] = r
        b = _pick(band, cell_line=line, translator="mlp", target=VERDICT[0], weighting=VERDICT[1])
        for _, row in b.iterrows():
            entry["null_band"][str(row["base_arm"])] = {
                "mean": float(row["null_mean"]),
                "sd": float(row["null_sd"]) if np.isfinite(row["null_sd"]) else None,
                "min": float(row["null_min"]),
                "max": float(row["null_max"]),
                "n_draws": int(row["n_draws"]),
            }
        for a in C.ARMS:
            per_arm: dict = {}
            for stage, prefix in (("final", "ft"), ("init", "pre")):
                for family, key in ((f"{prefix}_{a.name}", "value_substitution"),
                                    (f"{prefix}q_{a.name}", "pert_flag")):
                    for suffix in ("_resp", "_emb", "_full"):
                        r = _row(_pick(v, arm=f"{family}{suffix}", is_shuffled=False))
                        if r:
                            dd = _row(_pick(d, arm=f"{family}{suffix}", is_shuffled=False))
                            per_arm[f"{stage}__{key}{suffix}"] = {
                                **r,
                                "fold_centred": dd["score"] if dd else None,
                            }
            if training is not None and not training.empty:
                t = training[(training.cell_line == line) & (training.arm == a.name)]
                if not t.empty:
                    r = t.iloc[0]
                    per_arm["training"] = {
                        "selected_epoch": int(r["selected_epoch"]),
                        "epochs_run": int(r["epochs_run"]),
                        "best_val_loss": float(r["best_val_loss"]),
                        "n_train_cells": r["n_train_cells"],
                        "artifact": str(r["artifact"]),
                        "response_agreement_at_selected_epoch": {
                            k[len("selected_delta_") :]: float(r[k])
                            for k in t.columns
                            if k.startswith("selected_delta_") and np.isfinite(r[k])
                        },
                    }
            if not direct.empty:
                dr = direct[
                    (direct.cell_line == line)
                    & (direct.target == VERDICT[0])
                    & (direct.weighting == VERDICT[1])
                    & (direct.arm_label == f"direct_prediction/{a.name}")
                ]
                if not dr.empty:
                    by_base = {}
                    if "baseline" in dr.columns:
                        for b, g in dr.groupby("baseline"):
                            r = _row(g)
                            dg = _row(
                                _pick(
                                    direct[
                                        (direct.cell_line == line)
                                        & (direct.arm_label == f"direct_prediction/{a.name}")
                                        & (direct.baseline == b)
                                    ],
                                    target=DIAGNOSTIC[0],
                                    weighting=DIAGNOSTIC[1],
                                )
                            )
                            r["fold_centred"] = dg["score"] if dg else None
                            by_base[str(b)] = r
                    else:  # a table written before the baseline column existed
                        by_base["unflagged_model_output"] = _row(dr)
                    per_arm["direct_prediction_by_baseline"] = by_base
                    primary = next(
                        (by_base[b] for b in DIRECT_BASELINES if b in by_base),
                        next(iter(by_base.values())),
                    )
                    per_arm["direct_prediction"] = primary
                    per_arm["direct_prediction_baseline"] = next(
                        (b for b in DIRECT_BASELINES if b in by_base), None
                    )
            cs = contrasts[
                (contrasts.cell_line == line)
                & (contrasts.translator == "mlp")
                & (contrasts.target == VERDICT[0])
                & (contrasts.arm.str.contains(f"_{a.name}_", regex=False))
            ] if not contrasts.empty else pd.DataFrame()
            per_arm["paired_differences"] = [
                {
                    "arm": str(r["arm"]),
                    "against": str(r["against"]),
                    "difference": float(r["difference"]),
                    "lo": float(r["lo"]),
                    "hi": float(r["hi"]),
                    "n_groups": int(r["n_groups"]),
                    "min_resolvable_difference": float(
                        r.get("min_resolvable_difference__rho_0.5", np.nan)
                    ),
                }
                for _, r in cs.iterrows()
            ]
            entry["arms"][a.name] = per_arm
        out["cell_lines"][line] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--checkpoints", default=None)
    ap.add_argument("--out", default="results/summary")
    a = ap.parse_args(argv)
    head = build(
        Path(a.scores), Path(a.out), Path(a.checkpoints) if a.checkpoints else None
    )
    print(json.dumps(head, indent=2, default=str)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
