"""Turn the sweep's shards into the numbers the report states.

Three things happen here and nowhere else:

* the eight-cell correlation grid is assembled per arm, with the
  pre-registered cell (residual target, reliability-weighted, ceiling-corrected)
  marked as the one the verdict is read from;
* every arm is compared to co-expression **paired on the same perturbations**,
  which is the only comparison the design supports, with a paired bootstrap;
* the resolution the run actually achieved is recomputed from those paired
  differences, replacing the pre-run projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics, paths

#: the cell config.py pre-registered. It is reported everywhere, but it is not
#: what the verdict is read from: the shuffled-label control does not sit at
#: zero on it, because a held-out grouped fold shares a per-gene offset that any
#: near-constant prediction correlates with.
PREREG_COL = "residual__weighted__disattenuated"
PRIMARY_COL = PREREG_COL
CENTERED_COL = "centered__weighted__disattenuated"
#: the cell the verdict is read from instead: the same correlation after the
#: held-out fold's own per-gene mean is removed from prediction and truth, which
#: is the version whose shuffled-label control is at zero. Chosen after seeing
#: the null fail, so it is post hoc, and it answers the narrower question of how
#: well an arm separates one held-out perturbation from another.
VERDICT_COL = CENTERED_COL
REFERENCE_ARM = "coexp"
SHUFFLE_TAG = "+shuffled_labels_d"


def base_arm(arm: str) -> str:
    """The arm a shuffled-label draw was made from."""
    return arm.split(SHUFFLE_TAG)[0]


def null_band(runs: pd.DataFrame) -> pd.DataFrame:
    """The spread of the shuffled-label draws, which is what a score must clear.

    One permutation is one sample from the null. Several draws per arm give the
    band; a real arm inside that band has not shown anything.
    """
    d = runs[runs.arm.str.contains(SHUFFLE_TAG, regex=False)].copy()
    if d.empty:
        return pd.DataFrame()
    d["base_arm"] = d.arm.map(base_arm)
    keys = ["cell_line", "scope", "split", "translator", "target_source", "base_arm"]
    rows = []
    for key, block in d.groupby(keys, dropna=False):
        rec = {**dict(zip(keys, key)), "n_draws": int(block.shape[0])}
        for col, tag in ((PRIMARY_COL, "residual"), (CENTERED_COL, "centered")):
            v = block[col].to_numpy(dtype=float)
            rec[f"{tag}_mean"] = float(np.nanmean(v))
            rec[f"{tag}_sd"] = float(np.nanstd(v, ddof=1)) if v.size > 1 else float("nan")
            rec[f"{tag}_min"] = float(np.nanmin(v))
            rec[f"{tag}_max"] = float(np.nanmax(v))
        rows.append(rec)
    return pd.DataFrame(rows)


#: what identifies one run. A later wave that re-ran a cell of the sweep — the
#: node control after it was made additive, say — replaces the earlier one.
RUN_KEYS = ["cell_line", "scope", "split", "arm", "translator", "target_source"]


def load_sweep(*sweep_dirs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every shard of every wave, later waves winning where they overlap.

    Directories are read in the order given, so the caller states the ordering
    rather than it depending on a filename.
    """
    runs, per_pert = [], []
    for d in sweep_dirs:
        d = Path(d)
        runs += [pd.read_parquet(p) for p in sorted(d.glob("runs_shard*.parquet"))]
        per_pert += [pd.read_parquet(p) for p in sorted(d.glob("per_perturbation_shard*.parquet"))]
    if not runs:
        raise FileNotFoundError(f"no shards under {[str(d) for d in sweep_dirs]}")
    r = pd.concat(runs, ignore_index=True)
    p = pd.concat(per_pert, ignore_index=True)
    for frame in (r, p):
        if "target_source" not in frame.columns:
            frame["target_source"] = "run1_all_controls"
        frame["target_source"] = frame["target_source"].fillna("run1_all_controls")
    r = r.drop_duplicates(subset=RUN_KEYS, keep="last").reset_index(drop=True)
    p = p.drop_duplicates(subset=RUN_KEYS + ["perturbation"], keep="last").reset_index(drop=True)
    return r, p


def contrasts(
    per_pert: pd.DataFrame, reference: str = REFERENCE_ARM, metric: str = "residual_weighted"
) -> pd.DataFrame:
    """Paired difference of every arm against a reference arm.

    Arms are scored on the same held-out perturbations, so the comparison is
    paired; an unpaired interval would be wider than the design needs and would
    hide the fact that the arms rise and fall together across perturbations.
    """
    rows = []
    keys = ["cell_line", "scope", "split", "translator", "target_source"]
    for key, block in per_pert.groupby(keys, dropna=False):
        ref = block[block.arm == reference]
        if ref.empty:
            continue
        ref = ref.set_index("perturbation")[metric]
        for arm, sub in block.groupby("arm"):
            if arm == reference:
                continue
            s = sub.set_index("perturbation")[metric]
            shared = ref.index.intersection(s.index)
            a = s.loc[shared].to_numpy()
            b = ref.loc[shared].to_numpy()
            boot = metrics.bootstrap_median_difference(
                a, b, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_CI, C.SEED
            )
            diff = a - b
            ok = np.isfinite(diff)
            sd_paired = float(np.std(diff[ok])) if ok.sum() > 1 else float("nan")
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "arm": arm,
                    "reference": reference,
                    "metric": metric,
                    "median_difference": boot["difference"],
                    "lo": boot["lo"],
                    "hi": boot["hi"],
                    "p_two_sided": boot.get("p_two_sided", float("nan")),
                    "n_paired": int(ok.sum()),
                    "sd_paired": sd_paired,
                    "paired_correlation": float(
                        np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan
                    ),
                    "min_resolvable_difference_achieved": (
                        2.801583 * sd_paired / np.sqrt(max(int(ok.sum()), 1))
                    ),
                }
            )
    return pd.DataFrame(rows)


def null_ceiling(runs: pd.DataFrame, cell: dict, col: str) -> float:
    """The largest shuffled-label draw in one cell of the sweep: the number a
    score has to be above before it means anything."""
    d = runs
    for k, v in cell.items():
        d = d[d[k] == v]
    d = d[d.arm.str.contains(SHUFFLE_TAG, regex=False)]
    if d.empty:
        return float("nan")
    return float(np.nanmax(d[col].to_numpy(dtype=float)))


def headline(runs: pd.DataFrame, con: pd.DataFrame, nulls: pd.DataFrame | None = None) -> dict:
    """The verdict cell of the grid, per cell line, on the primary split."""
    out: dict = {
        "verdict_metric": VERDICT_COL,
        "preregistered_metric": PREREG_COL,
        "verdict_metric_note": (
            "the pre-registered cell is reported for every arm, but the verdict is read "
            "from the fold-centred cell, because the shuffled-label control sits at zero "
            "there and does not on the pre-registered one"
        ),
        "reference_arm": REFERENCE_ARM,
        "lines": {},
    }
    if nulls is not None and not nulls.empty:
        band = nulls[(nulls.scope == "full") & (nulls.split == "grouped")]
        out["null_band_all_cells"] = nulls.to_dict("records")
        out["null_band"] = band.to_dict("records")
    primary = runs[
        (runs.scope == "full")
        & (runs.split == "grouped")
        & (runs.translator == "mlp")
        & (runs.target_source == "run1_all_controls")
    ]
    for line, block in primary.groupby("cell_line"):
        ranked = block.sort_values(VERDICT_COL, ascending=False)
        cell = {
            "cell_line": line,
            "scope": "full",
            "split": "grouped",
            "translator": "mlp",
            "target_source": "run1_all_controls",
        }
        null_prereg = null_ceiling(runs, cell, PREREG_COL)
        null_verdict = null_ceiling(runs, cell, VERDICT_COL)
        c = con[
            (con.cell_line == line)
            & (con.scope == "full")
            & (con.split == "grouped")
            & (con.translator == "mlp")
            & (con.metric == "centered_weighted")
        ]
        out["lines"][line] = {
            "n_perturbations": int(block.n_perturbations.iloc[0]),
            "n_readout": int(block.n_readout.iloc[0]),
            "residual_ceiling": float(block["residual__weighted__ceiling"].iloc[0]),
            "centered_ceiling": float(block["centered__weighted__ceiling"].iloc[0]),
            "largest_shuffled_draw_preregistered": null_prereg,
            "largest_shuffled_draw_verdict": null_verdict,
            "arms": [
                {
                    "arm": r.arm,
                    "family": r.family,
                    "dim": int(r.feature_dim),
                    "score": float(getattr(r, VERDICT_COL)),
                    "lo": float(getattr(r, VERDICT_COL + "_lo")),
                    "hi": float(getattr(r, VERDICT_COL + "_hi")),
                    "clears_null": bool(float(getattr(r, VERDICT_COL)) > null_verdict),
                    "preregistered": float(getattr(r, PREREG_COL)),
                    "preregistered_lo": float(getattr(r, PREREG_COL + "_lo")),
                    "preregistered_hi": float(getattr(r, PREREG_COL + "_hi")),
                    "preregistered_clears_null": bool(
                        float(getattr(r, PREREG_COL)) > null_prereg
                    ),
                    "discrimination_ours": float(r.discrimination_ours_residual_mean),
                    "discrimination_cell_eval": float(r.discrimination_cell_eval_mean),
                }
                for r in ranked.itertuples()
            ],
            "vs_coexpression": [
                {
                    "arm": r.arm,
                    "median_difference": float(r.median_difference),
                    "lo": float(r.lo),
                    "hi": float(r.hi),
                    "paired_correlation": float(r.paired_correlation),
                    "achieved_min_resolvable_difference": float(
                        r.min_resolvable_difference_achieved
                    ),
                }
                for r in c.sort_values("median_difference", ascending=False).itertuples()
            ],
        }
    return out


def control_cell_overlap(runs: pd.DataFrame) -> pd.DataFrame:
    """The same arms scored against targets built on different control cells.

    ``ctrl0`` shares no control cell with the co-expression features, ``ctrl1``
    shares all of them, and ``ctrlall`` is the estimator run 1 used. If sharing
    control cells with the target inflated the statistical baseline, the
    baseline's advantage would shrink from ctrl1 to ctrl0 while the model arms,
    which touch no control cell, would not move.
    """
    d = runs[runs.target_source != "run1_all_controls"]
    if d.empty:
        return pd.DataFrame()
    keep = [
        "cell_line", "scope", "split", "translator", "target_source", "arm", "family",
        VERDICT_COL, VERDICT_COL + "_lo", VERDICT_COL + "_hi", PREREG_COL,
        "residual__weighted__ceiling", "centered__weighted__ceiling",
        "discrimination_ours_residual_mean",
    ]
    wide = d[[c for c in keep if c in d.columns]].copy()
    return wide.sort_values(["cell_line", "arm", "translator", "target_source"])


def grid_table(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, grid cell): the eight numbers, side by side."""
    rows = []
    for r in runs.itertuples():
        for target in C.GRID_TARGETS:
            for weighting in C.GRID_WEIGHTINGS:
                for correction in C.GRID_CORRECTIONS:
                    col = f"{target}__{weighting}__{correction}"
                    rows.append(
                        {
                            "cell_line": r.cell_line,
                            "scope": r.scope,
                            "split": r.split,
                            "target_source": r.target_source,
                            "arm": r.arm,
                            "family": r.family,
                            "translator": r.translator,
                            "target": target,
                            "weighting": weighting,
                            "correction": correction,
                            "value": float(getattr(r, col)),
                            "lo": float(getattr(r, col + "_lo")),
                            "hi": float(getattr(r, col + "_hi")),
                            "is_primary": col == PRIMARY_COL,
                        }
                    )
    return pd.DataFrame(rows)


def ordering_agreement(grid: pd.DataFrame) -> pd.DataFrame:
    """Does the grid agree about which arm is ahead?

    Spearman correlation between the arm ordering under the primary cell and
    under every other cell, per (cell line, scope, split, translator).
    """
    rows = []
    keys = ["cell_line", "scope", "split", "translator", "target_source"]
    for key, block in grid.groupby(keys, dropna=False):
        base = block[block.is_primary].set_index("arm")["value"]
        for (t, w, c), sub in block.groupby(["target", "weighting", "correction"]):
            s = sub.set_index("arm")["value"]
            shared = base.index.intersection(s.index)
            pair = pd.DataFrame({"a": base.loc[shared], "b": s.loc[shared]}).dropna()
            if pair.shape[0] < 3:
                continue
            a = pair["a"].rank()
            b = pair["b"].rank()
            shared = pair.index
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "cell": f"{t}__{w}__{c}",
                    "spearman_vs_primary": float(np.corrcoef(a, b)[0, 1]),
                    "n_arms": int(shared.size),
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=None, help="comma-separated shard directories, in order")
    ap.add_argument("--transfer-sweep", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    sweep_dirs = (
        [Path(x) for x in args.sweep.split(",") if x]
        if args.sweep
        else [paths.results_dir() / "sweep"]
    )
    out = paths.ensure(Path(args.out) if args.out else paths.results_dir() / "summary")

    runs, per_pert = load_sweep(*sweep_dirs)
    con = pd.concat(
        [contrasts(per_pert, metric=m) for m in ("residual_weighted", "centered_weighted")],
        ignore_index=True,
    )
    nulls = null_band(runs)
    if not nulls.empty:
        nulls.to_parquet(out / "null_band.parquet", index=False)
    grid = grid_table(runs)
    agree = ordering_agreement(grid)
    head = headline(runs, con, nulls)
    overlap = control_cell_overlap(runs)
    if not overlap.empty:
        overlap.to_parquet(out / "control_cell_overlap.parquet", index=False)
        head["control_cell_overlap"] = overlap.to_dict("records")

    if args.transfer_sweep:
        t_runs, t_pert = load_sweep(*[Path(x) for x in args.transfer_sweep.split(",") if x])
        t_runs.to_parquet(out / "transfer_runs.parquet", index=False)
        t_pert.to_parquet(out / "transfer_per_perturbation.parquet", index=False)
        t_con = pd.concat(
            [contrasts(t_pert, metric=m) for m in ("residual_weighted", "centered_weighted")],
            ignore_index=True,
        )
        t_con.to_parquet(out / "transfer_contrasts.parquet", index=False)
        head["transfer"] = {
            f"{line}|{scope}|{split}": [
                {
                    "arm": r.arm,
                    "family": r.family,
                    "score": float(getattr(r, VERDICT_COL)),
                    "lo": float(getattr(r, VERDICT_COL + "_lo")),
                    "hi": float(getattr(r, VERDICT_COL + "_hi")),
                    "preregistered": float(getattr(r, PREREG_COL)),
                    "n_perturbations": int(r.n_perturbations),
                }
                for r in block[block.translator == "mlp"]
                .sort_values(VERDICT_COL, ascending=False)
                .itertuples()
            ]
            for (line, scope, split), block in t_runs.groupby(["cell_line", "scope", "split"])
        }

    runs.to_parquet(out / "runs.parquet", index=False)
    per_pert.to_parquet(out / "per_perturbation.parquet", index=False)
    con.to_parquet(out / "contrasts.parquet", index=False)
    grid.to_parquet(out / "grid.parquet", index=False)
    agree.to_parquet(out / "ordering_agreement.parquet", index=False)
    (out / "headline.json").write_text(json.dumps(head, indent=2))
    print(json.dumps(head, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
