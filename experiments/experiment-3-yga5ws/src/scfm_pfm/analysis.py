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

PRIMARY_COL = "residual__weighted__disattenuated"
CENTERED_COL = "centered__weighted__disattenuated"
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
    keys = ["cell_line", "scope", "split", "translator", "base_arm"]
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


def load_sweep(sweep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.concat(
        [pd.read_parquet(p) for p in sorted(sweep_dir.glob("runs_shard*.parquet"))],
        ignore_index=True,
    )
    per_pert = pd.concat(
        [pd.read_parquet(p) for p in sorted(sweep_dir.glob("per_perturbation_shard*.parquet"))],
        ignore_index=True,
    )
    return runs, per_pert


def contrasts(
    per_pert: pd.DataFrame, reference: str = REFERENCE_ARM, metric: str = "residual_weighted"
) -> pd.DataFrame:
    """Paired difference of every arm against a reference arm.

    Arms are scored on the same held-out perturbations, so the comparison is
    paired; an unpaired interval would be wider than the design needs and would
    hide the fact that the arms rise and fall together across perturbations.
    """
    rows = []
    keys = ["cell_line", "scope", "split", "translator"]
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


def headline(runs: pd.DataFrame, con: pd.DataFrame, nulls: pd.DataFrame | None = None) -> dict:
    """The primary cell of the grid, per cell line, on the primary split."""
    out: dict = {
        "primary_metric": PRIMARY_COL,
        "secondary_metric": CENTERED_COL,
        "reference_arm": REFERENCE_ARM,
        "lines": {},
    }
    if nulls is not None and not nulls.empty:
        band = nulls[(nulls.scope == "full") & (nulls.split == "grouped")]
        out["null_band"] = band.to_dict("records")
    primary = runs[(runs.scope == "full") & (runs.split == "grouped") & (runs.translator == "mlp")]
    for line, block in primary.groupby("cell_line"):
        ranked = block.sort_values(PRIMARY_COL, ascending=False)
        c = con[
            (con.cell_line == line)
            & (con.scope == "full")
            & (con.split == "grouped")
            & (con.translator == "mlp")
            & (con.metric == "residual_weighted")
        ]
        out["lines"][line] = {
            "n_perturbations": int(block.n_perturbations.iloc[0]),
            "n_readout": int(block.n_readout.iloc[0]),
            "residual_ceiling": float(block["residual__weighted__ceiling"].iloc[0]),
            "arms": [
                {
                    "arm": r.arm,
                    "family": r.family,
                    "dim": int(r.feature_dim),
                    "score": float(getattr(r, PRIMARY_COL)),
                    "lo": float(getattr(r, PRIMARY_COL + "_lo")),
                    "hi": float(getattr(r, PRIMARY_COL + "_hi")),
                    "centered": float(getattr(r, CENTERED_COL)),
                    "centered_lo": float(getattr(r, CENTERED_COL + "_lo")),
                    "centered_hi": float(getattr(r, CENTERED_COL + "_hi")),
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
    keys = ["cell_line", "scope", "split", "translator"]
    for key, block in grid.groupby(keys, dropna=False):
        base = block[block.is_primary].set_index("arm")["value"]
        for (t, w, c), sub in block.groupby(["target", "weighting", "correction"]):
            s = sub.set_index("arm")["value"]
            shared = base.index.intersection(s.index)
            if shared.size < 3:
                continue
            a = base.loc[shared].rank()
            b = s.loc[shared].rank()
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
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--transfer-sweep", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    sweep_dir = Path(args.sweep) if args.sweep else paths.results_dir() / "sweep"
    out = paths.ensure(Path(args.out) if args.out else paths.results_dir() / "summary")

    runs, per_pert = load_sweep(sweep_dir)
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

    if args.transfer_sweep:
        t_runs, t_pert = load_sweep(Path(args.transfer_sweep))
        t_runs.to_parquet(out / "transfer_runs.parquet", index=False)
        t_pert.to_parquet(out / "transfer_per_perturbation.parquet", index=False)
        t_con = contrasts(t_pert)
        t_con.to_parquet(out / "transfer_contrasts.parquet", index=False)
        head["transfer"] = {
            str(line): [
                {
                    "arm": r.arm,
                    "family": r.family,
                    "score": float(getattr(r, PRIMARY_COL)),
                    "lo": float(getattr(r, PRIMARY_COL + "_lo")),
                    "hi": float(getattr(r, PRIMARY_COL + "_hi")),
                }
                for r in block[block.translator == "mlp"]
                .sort_values(PRIMARY_COL, ascending=False)
                .itertuples()
            ]
            for line, block in t_runs.groupby("cell_line")
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
