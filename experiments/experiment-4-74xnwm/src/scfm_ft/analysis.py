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
from . import paths

#: the pre-registered cell of run 3's correlation grid, and the fold-centred
#: diagnostic reported beside it everywhere
VERDICT = ("residual", "weighted")
DIAGNOSTIC = ("centered", "weighted")

#: the two refits that withhold different folds, and so are scored on different
#: perturbations against a different ceiling than every other reading
REFIT_TAGS = ("__f23", "__f4")
#: the model's own predicted response can be read against two reference points.
#: ``measured_control_mean`` subtracts the control cells' measured profile, which
#: is the comparison the published recipe is evaluated on; ``unflagged_model_output``
#: subtracts the model's own no-flag forward pass, which is the reference the
#: response features use. The first is primary, the second is reported beside it.
DIRECT_BASELINES = ("measured_control_mean", "unflagged_model_output")


#: the per-perturbation tables sit beside the summary ones and share their
#: prefix, so ``direct_*`` would otherwise also match ``direct_per_pert_*``.
#: Those have a different schema and belong to a different level of the run.
_PER_PERT = ("per_pert_", "direct_per_pert_", "crossfit_per_pert_")


def _read_all(
    root: Path, stem: str, defaults: dict[str, object] | None = None
) -> pd.DataFrame:
    files = [
        f
        for f in sorted(Path(root).glob(f"{stem}_*.parquet"))
        if not any(f.name.startswith(p) for p in _PER_PERT)
    ]
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        frame = pd.read_parquet(f)
        # Which reading wrote this table. The reference arms are recomputed in
        # every job, so without this the rows are indistinguishable and a
        # comparison across jobs cannot tell readings on different held-out
        # folds apart from genuine disagreement on the same folds.
        frame["source_run"] = f.name[len(stem) + 1 : -len(".parquet")]
        frames.append(frame)
    # A column added to the scorer after some tables were written is filled in
    # with what those tables meant, rather than left absent: concatenating over
    # a missing column would put NaN there, and a groupby would then drop those
    # rows without saying so.
    for name, value in (defaults or {}).items():
        for f in frames:
            if name not in f.columns:
                f[name] = value
    cols = {frozenset(f.columns) for f in frames}
    if len(cols) > 1:
        raise RuntimeError(
            f"{stem}: the {len(files)} tables do not share one schema; "
            f"{len(cols)} distinct column sets across {[f.name for f in files][:6]}"
        )
    return pd.concat(frames, ignore_index=True)


def arm_of(name: str) -> tuple[str, str, str]:
    """Split a scored arm name into (stage, arm, query family).

    ``ft_<arm>_resp``  the fine-tuned model, value-substitution query
    ``ftq_<arm>_resp`` the fine-tuned model, its own perturbation query
    ``pre_``/``preq_`` the same before any gradient step
    anything else is a reference arm and keeps its own name.

    A tagged run carries its tag in the arm name behind ``-`` rather than
    ``__`` (see :func:`config.model_stem`); the run name is restored here so
    every table downstream keys on ``own_trainable__f23`` as the run does.
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
                    return stage, C.run_of_stem(rest[: -len(suffix)]), f"{family}{suffix}"
            return stage, C.run_of_stem(rest), family
    return "reference", "", name


#: where each training run's durable directory lives, for the summary to cite
ARTIFACT_ROOT = (
    "artifact://fundacion-instituto-leloir-553557/experiments/"
    "exp_01m0trpxqyf0g8pq29k774xnwm"
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
                "val_tutorial_pearson": float(rec.get("val_tutorial_pearson", np.nan)),
            }
            for k, v in (rec.get("val_delta_pearson") or {}).items():
                row[f"delta_{k}"] = float(v)
            epochs.append(row)
        sub = cfg.get("gene_subsampling_counters", {})
        runs.append(
            {
                "cell_line": line,
                "arm": arm,
                "selected_epoch": int(hist.get("selected_epoch", hist.get("best_epoch", 0))),
                "epochs_run": len(hist["history"]),
                "selected_val_pearson": float(hist.get("selected_val_pearson", np.nan)),
                "mse_selected_epoch": hist.get("mse_selected_epoch"),
                "pearson_rule_stopped_at_epoch": hist.get("pearson_rule_stopped_at_epoch"),
                "mse_rule_stopped_at_epoch": hist.get("mse_rule_stopped_at_epoch"),
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
                "artifact": f"{ARTIFACT_ROOT}/checkpoints{paths.run_suffix()}/{d.name}/",
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
    direct = _read_all(scores, "direct", defaults={"baseline": "unflagged_model_output"})
    if restricted.empty:
        raise RuntimeError(f"no restricted_*.parquet under {scores}")

    restricted["is_shuffled"] = restricted["arm"].str.contains(r"\+shuffled_labels_d")
    restricted["base_arm"] = restricted["arm"].str.replace(
        r"\+shuffled_labels_d\d+", "", regex=True
    )
    # Parse the arm WITHOUT its shuffled-draw suffix. Parsing the raw name puts
    # the whole tail into train_arm for every shuffled row, so those rows match
    # no known arm and drop out of any filter keyed on the training arm.
    parts = restricted["base_arm"].map(arm_of)
    restricted["stage"] = [p[0] for p in parts]
    restricted["train_arm"] = [p[1] for p in parts]
    restricted["family"] = [p[2] for p in parts]

    # the fine-tuned arms are scored once per training job; the reference arms
    # are re-run inside every job, so they appear once per arm. They must agree.
    # ...on the same rows. The two refits withhold folds 2+3 and fold 4, so
    # their reference recomputations are scored on different perturbations
    # against a different ceiling; comparing those with the folds 0+1 readings
    # measures the fold difference, not the scoring path. Only readings that
    # share the evaluation folds belong in this check.
    ref = restricted[
        (restricted.stage == "reference")
        & (~restricted.is_shuffled)
        & (restricted.target == VERDICT[0])
        & (restricted.weighting == VERDICT[1])
        & (~restricted.source_run.str.endswith(REFIT_TAGS))
    ]
    agreement = (
        ref.groupby(["cell_line", "arm", "translator"])["disattenuated"]
        .agg(["min", "max", "count"])
        .assign(spread=lambda d: d["max"] - d["min"])
        .reset_index()
    )
    agreement = agreement[agreement["count"] > 0].reset_index(drop=True)

    # The band is run 3's pre-registered null, unchanged: the co-expression arm
    # re-scored against shuffled labels, recomputed inside every job. Every
    # other arm's shuffled draws could be folded in as well, but each addition
    # can only widen the interval the result is read against, and widening it is
    # the direction that flatters this experiment's conclusion. The fine-tuned
    # arms' own draws are reported separately by ``null_check`` instead, so the
    # claim that they also sit inside this band is checkable rather than assumed.
    in_preregistered = (
        restricted.is_shuffled
        & (restricted.stage == "reference")
        & (restricted.base_arm == "coexp")
    )
    band = (
        restricted[in_preregistered]
        .groupby(["cell_line", "base_arm", "translator", "target", "weighting"])["disattenuated"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "null_mean", "std": "null_sd", "min": "null_min", "max": "null_max",
                         "count": "n_draws"})
    )

    # Each fine-tuned arm also draws its own shuffled labels. Those draws do not
    # define the band, but they say whether the band the result is read against
    # is the right size for that arm: a draw outside it would mean the arm's own
    # null is wider than co-expression's, and the band would be too tight.
    own = restricted[
        restricted.is_shuffled
        & (restricted.stage == "final")
        & (restricted.target == VERDICT[0])
        & (restricted.weighting == VERDICT[1])
        & (restricted.translator == "mlp")
    ]
    null_check = (
        own.groupby(["cell_line", "base_arm", "train_arm", "family"])["disattenuated"]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={"min": "draw_min", "max": "draw_max", "count": "n_draws"})
    )
    if not null_check.empty and not band.empty:
        mlp_band = band[
            (band.translator == "mlp")
            & (band.target == VERDICT[0])
            & (band.weighting == VERDICT[1])
        ].set_index("cell_line")
        null_check["band_min"] = null_check["cell_line"].map(mlp_band["null_min"])
        null_check["band_max"] = null_check["cell_line"].map(mlp_band["null_max"])
        # two-sided: a draw below the band's floor also means the arm's own null
        # is wider than the band it is judged against
        null_check["inside_band"] = (null_check["draw_max"] <= null_check["band_max"]) & (
            null_check["draw_min"] >= null_check["band_min"]
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
        ("null_check", null_check),
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
                        "selected_val_pearson": (
                            float(r["selected_val_pearson"])
                            if "selected_val_pearson" in t.columns
                            and np.isfinite(r["selected_val_pearson"])
                            else None
                        ),
                        "mse_selected_epoch": (
                            int(r["mse_selected_epoch"])
                            if "mse_selected_epoch" in t.columns
                            and r["mse_selected_epoch"] is not None
                            and np.isfinite(r["mse_selected_epoch"])
                            else None
                        ),
                        "selection_rule": str(r["selection_rule"]),
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
                            same = direct[
                                (direct.cell_line == line)
                                & (direct.arm_label == f"direct_prediction/{a.name}")
                                & (direct.baseline == b)
                            ]
                            # The two cells that are interpretable for a
                            # prediction the model makes on its own scale: raw,
                            # the predicted profile against the measured one,
                            # and centred, which removes the held-out fold's own
                            # mean from both sides and so asks only whether the
                            # prediction separates one perturbation from another.
                            # The residual cell would have to subtract a measured
                            # profile from a prediction that is not on the
                            # measured scale, so it is carried but not headlined.
                            dg = _row(
                                _pick(same, target=DIAGNOSTIC[0], weighting=DIAGNOSTIC[1])
                            )
                            dw = _row(_pick(same, target="raw", weighting=VERDICT[1]))
                            r["residual_not_interpretable"] = True
                            for prefix, cell in (("fold_centred", dg), ("raw", dw)):
                                r[prefix] = cell["score"] if cell else None
                                r[f"{prefix}_lo"] = cell["lo"] if cell else None
                                r[f"{prefix}_hi"] = cell["hi"] if cell else None
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
