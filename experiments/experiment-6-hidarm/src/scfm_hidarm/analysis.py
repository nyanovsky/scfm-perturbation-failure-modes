"""Merge stage 1's per-unit tables into one depth profile, and write it tracked.

Run 3's own ``analysis.py`` hardcodes arm names in seven places, so a run-6 arm
is silently omitted from it. Run 6 therefore writes its own merge and imports
only run 3's ``metrics.py`` and ``groups.py`` --- never its analysis.

Three things this module is careful about, each because the alternative
produces a plausible table rather than an error:

**The reference arms are never merged into the run table.** They are re-runs of
rows run 3 already published; merging them would overwrite a published row with
a re-run of it. They go into their own table and a separate column says how far
each is from run 3's published value.

**Every row carries the null it was read against, and only the nulls its own
cell actually has.** ``hidperm`` and ``hidrand`` are drawn at every depth;
fresh shuffled-label draws are taken at the pre-registered primary layer only.
A depth without its own draws must not be quoted as though it had them, so the
``shuffle_max`` column is null there rather than filled from another depth.

**Every row carries its cell's captured-energy value.** The fitted basis
under-spans the scored perturbations' subspace and does so unevenly across arms
(README s2), so a depth profile drawn without it mixes two quantities. The bias
is downward only and can never inflate a positive, but it is not uniform, and
``scgpt_ft`` vs ``scgpt_ft_frozen`` is exactly where the spread is worst.

Both metric cells travel in every row. The verdict is the **pre-registered**
``residual__weighted__disattenuated``; arms are only *ordered* by the post-hoc
``centered__weighted__disattenuated``, and neither confirms the other.

    python -m scfm_hidarm.analysis --sweep stage1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import paths

#: the columns the tracked summary carries. Small on purpose: it is the file a
#: paper figure is allowed to read, and "if a number cannot be recovered from a
#: tracked file it does not go in the paper".
SUMMARY_COLUMNS = (
    "cell_line", "model", "variant", "layer", "k", "arm", "split", "translator",
    "feature_dim", "n_perturbations", "n_groups", "largest_group",
    C.PRIMARY_METRIC, C.ORDERING_METRIC,
    "discrimination_ours_residual_mean", "seconds",
    "captured_energy_ratio", "captured_energy_meets_target",
    "is_primary_layer", "shuffle_draws_taken",
)
#: the arms that are re-runs of run 3's published rows and never merged
REFERENCE_ARM_SET = frozenset(
    C.REFERENCE_ARMS + tuple(f"{m}_emb" for m in C.REFERENCE_EMB_MODELS)
)
#: run 3's shuffled-label arms carry this suffix; the base arm is what precedes it
SHUFFLE_MARK = "+shuffled_labels_d"


def unit_records(sweep: str) -> list[dict]:
    """Every ``unit.json`` stage 1 wrote, in a deterministic order."""
    root = paths.sweep_dir(sweep)
    if not root.is_dir():
        raise RuntimeError(
            f"no sweep directory at {root}; a missing sweep fails rather than producing an "
            "empty table, because an empty table reads as 'nothing cleared its null'"
        )
    out = []
    for f in sorted(root.glob("*/unit.json")):
        out.append(json.loads(f.read_text()) | {"unit_dir": str(f.parent)})
    if not out:
        raise RuntimeError(f"no unit.json under {root}; stage 1 has not written a unit yet")
    return out


def read_runs(sweep: str) -> pd.DataFrame:
    """Concatenate every bundle's ``runs_<line>.parquet`` under one sweep.

    The unit directory name is not parsed for the (model, layer, k) --- those
    are taken from the arm key through ``config.run_of_stem``, which is the same
    decoder the block file name was built with. Parsing a directory name would
    be a second, independent encoding of the same fact, free to disagree.
    """
    frames = []
    for rec in unit_records(sweep):
        for f in sorted(Path(rec["unit_dir"]).glob("b*/runs_*.parquet")):
            df = pd.read_parquet(f)
            frames.append(df.assign(unit=rec["unit"], source=str(f)))
    if not frames:
        raise RuntimeError(f"units exist under {paths.sweep_dir(sweep)} but no runs_*.parquet")
    return pd.concat(frames, ignore_index=True)


def decode_arm(arm: str) -> dict:
    """``<key>_resp`` -> ``{model, variant, layer, k}``; anything else -> empty.

    A reference arm (``coexp``, ``scgpt_emb``, ...) is not a run-6 key and
    decodes to empty rather than raising: the same table holds both, and the
    caller separates them on ``is_reference`` rather than on a traceback.
    """
    base = arm.split(SHUFFLE_MARK, 1)[0]
    if not base.endswith("_resp"):
        return {}
    try:
        return C.run_of_stem(base[: -len("_resp")])
    except ValueError:
        return {}


def annotate(runs: pd.DataFrame, sweep: str) -> pd.DataFrame:
    """Attach the decoded key, the captured energy and the primary-layer flag."""
    energies: dict[tuple[str, str, int, int], dict] = {}
    shuffled_at: set[tuple[str, str, int, int]] = set()
    for rec in unit_records(sweep):
        if rec.get("model") is None:
            continue
        cell = (rec["cell_line"], rec["model"], int(rec["layer"]), int(rec["k"]))
        if rec.get("captured_energy"):
            energies[cell] = rec["captured_energy"]
        if rec.get("shuffle_bases"):
            shuffled_at.add(cell)

    decoded = pd.DataFrame([decode_arm(a) for a in runs["arm"]], index=runs.index)
    out = runs.copy()
    for col in ("model", "variant", "layer", "k"):
        out[f"key_{col}"] = decoded[col] if col in decoded else np.nan
    # run 3's own `model` column is the pseudo-model, which for a run-6 block is
    # the whole key. The decoded one is the real model and is what every join uses.
    out["model"] = out["key_model"].fillna("")
    out["variant"] = out["key_variant"].fillna("")
    out["layer"] = out["key_layer"]
    out["k"] = out["key_k"]
    out["is_reference"] = out["arm"].isin(REFERENCE_ARM_SET)
    out["is_shuffle_draw"] = out["arm"].str.contains(SHUFFLE_MARK, regex=False)
    # regex=False: the marker begins with '+', which pandas would otherwise
    # compile as a quantifier with nothing to repeat
    out["shuffle_base"] = np.where(
        out["is_shuffle_draw"],
        out["arm"].str.split(SHUFFLE_MARK, regex=False).str[0],
        "",
    )

    ratio, meets, primary, drew = [], [], [], []
    for _, r in out.iterrows():
        cell = (r["cell_line"], r["model"], r["layer"], r["k"])
        e = energies.get(cell) if r["model"] else None
        ratio.append(float(e["ratio"]) if e else np.nan)
        meets.append(bool(e["meets_target"]) if e else False)
        primary.append(
            bool(r["model"]) and not np.isnan(r["layer"])
            and int(r["layer"]) == C.primary_layer(r["model"])
        )
        drew.append(cell in shuffled_at)
    out["captured_energy_ratio"] = ratio
    out["captured_energy_meets_target"] = meets
    out["is_primary_layer"] = primary
    out["shuffle_draws_taken"] = drew
    return out


def null_bands(annotated: pd.DataFrame, metric: str = C.PRIMARY_METRIC) -> pd.DataFrame:
    """The band each real arm is read against, per (line, model, layer, k, cell).

    Three separate numbers, never one combined "null":

    ``hidperm_max``  the machinery null --- the real features with the pairing
                     destroyed and everything else preserved. Drawn at every depth.
    ``hidrand_max``  width only. Projecting an isotropic Gaussian through an
                     orthonormal W gives N(0, s^2 I_k) whatever W is, so this
                     controls for the arm's 640 columns and for nothing else ---
                     which at the k = 64 rung is the relevant width control,
                     because a 640-dim arm is width-matched to no run 3 reference.
    ``shuffle_max``  fresh shuffled-label draws in the arm's OWN cell, and NaN
                     wherever they were not taken. An arm counts only if it
                     clears the largest of its own draws.

    ``hidrp`` is deliberately not folded in: it is not a null but a second real
    arm through a random basis, and it answers whether the PCA basis is doing
    the work rather than whether the pairing is.
    """
    keys = ["cell_line", "model", "layer", "k", "split", "translator"]
    rows: list[dict] = []
    real = annotated[(annotated["variant"] == "hid") & ~annotated["is_shuffle_draw"]]
    for cell, grp in real.groupby(keys, dropna=False):
        d = dict(zip(keys, cell))
        sel = annotated
        for col, val in d.items():
            sel = sel[sel[col] == val] if not pd.isna(val) else sel[sel[col].isna()]
        band = {}
        for variant in ("hidperm", "hidrand", "hidrp"):
            v = sel[(sel["variant"] == variant) & ~sel["is_shuffle_draw"]][metric]
            band[f"{variant}_max"] = float(v.max()) if len(v) else np.nan
            band[f"{variant}_n"] = int(len(v))
        base = C.arm_name(C.model_stem(d["model"], "hid", int(d["layer"]), int(d["k"])))
        sh = annotated[
            annotated["is_shuffle_draw"]
            & (annotated["shuffle_base"] == base)
            & (annotated["cell_line"] == d["cell_line"])
            & (annotated["split"] == d["split"])
        ][metric]
        band["shuffle_max"] = float(sh.max()) if len(sh) else np.nan
        band["shuffle_n"] = int(len(sh))
        rows.append(
            d | band | {
                "metric": metric,
                "arm_score": float(grp[metric].max()),
                "clears_hidperm": bool(grp[metric].max() > band["hidperm_max"])
                if not np.isnan(band["hidperm_max"]) else False,
                "clears_shuffle": bool(grp[metric].max() > band["shuffle_max"])
                if not np.isnan(band["shuffle_max"]) else None,
            }
        )
    return pd.DataFrame(rows)


def basis_check(summary: pd.DataFrame, metric: str = C.PRIMARY_METRIC) -> dict:
    """Compare each fitted-basis arm against the frozen-random-basis arm beside it.

    Why this is a summary and not a figure caption: ``hidrp`` is real ``Delta h``
    through a random orthonormal basis of the same width, so it holds the arm's
    dimension and its machinery fixed and varies only whether the projection was
    fitted. If the two agree, the fitted basis contributes nothing, and the uneven
    captured energy across cells (README s2) therefore cannot be what drives any
    arm-to-arm difference. That is a statement about a *pairing* of rows, which no
    single row of the depth table carries, so it is computed once here and written
    tracked rather than re-derived by whoever quotes it.

    It does **not** retire the declared captured-energy target. It bounds what the
    shortfall against that target can explain.
    """
    wide = summary[summary["translator"] == "mlp"].pivot_table(
        index=["cell_line", "model", "layer", "k"], columns="variant", values=metric
    )
    both = wide.dropna(subset=["hid", "hidrp"])
    if both.empty:
        raise RuntimeError(
            "no cell carries both a fitted and a random-basis arm; the comparison that "
            "bounds what the captured-energy shortfall can explain cannot be made"
        )
    delta = (both["hidrp"] - both["hid"]).astype(float)
    return {
        "metric": metric,
        "translator": "mlp",
        "n_cells": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "min_delta": float(delta.min()),
        "max_delta": float(delta.max()),
        "n_random_at_least_fitted": int((delta >= 0).sum()),
        "captured_energy_min": float(
            summary.loc[summary["variant"] == "hid", "captured_energy_ratio"].min()
        ),
        "captured_energy_max": float(
            summary.loc[summary["variant"] == "hid", "captured_energy_ratio"].max()
        ),
        "note": (
            "hidrp is real hidden-state deltas through a frozen random orthonormal basis "
            "of the same width; agreement bounds what the captured-energy shortfall can "
            "explain and does not retire the declared target"
        ),
    }


def build(sweep: str = "stage1", out_dir: Path | None = None) -> dict:
    """Write the heavy merge beside the sweep and the small summary into the repo.

    ``out_dir`` exists so a smoke can be told where to put its planted numbers.
    The default is the **tracked** ``results/summary/``, which is the point of
    this file --- a paper figure may read only tracked files --- but a default
    that a test cannot redirect is a default that writes planted numbers into
    the record, which is exactly what ``scratch/smoke_report.py`` caught on its
    first run.
    """
    runs = read_runs(sweep)
    ann = annotate(runs, sweep)

    heavy = paths.ensure(paths.sweep_dir(sweep) / "merged")
    ann.to_parquet(heavy / "runs_annotated.parquet", index=False)

    scored = ann[~ann["is_reference"]].copy()
    missing = [c for c in SUMMARY_COLUMNS if c not in scored.columns]
    if missing:
        raise RuntimeError(
            f"the tracked summary is missing {missing}; a paper figure may read only tracked "
            "files, so a column absent here is a number that cannot go in the paper"
        )
    summary = (
        scored[~scored["is_shuffle_draw"]][list(SUMMARY_COLUMNS)]
        .sort_values(["cell_line", "model", "layer", "variant", "translator"])
        .reset_index(drop=True)
    )
    bands = {
        metric: null_bands(ann, metric) for metric in (C.PRIMARY_METRIC, C.ORDERING_METRIC)
    }

    out = paths.ensure(Path(out_dir) if out_dir is not None else paths.summary_dir())
    summary.to_csv(out / f"{sweep}_depth.csv", index=False)
    pd.concat(bands.values(), ignore_index=True).to_csv(
        out / f"{sweep}_null_bands.csv", index=False
    )
    ann[ann["is_reference"]][
        ["cell_line", "arm", "split", "translator", "feature_dim",
         C.PRIMARY_METRIC, C.ORDERING_METRIC]
    ].sort_values(["cell_line", "arm", "translator"]).to_csv(
        out / f"{sweep}_references.csv", index=False
    )
    shuf = ann[ann["is_shuffle_draw"]][
        ["cell_line", "shuffle_base", "split", "translator", C.PRIMARY_METRIC,
         C.ORDERING_METRIC]
    ].sort_values(["cell_line", "shuffle_base"])
    shuf.to_csv(out / f"{sweep}_shuffle_draws.csv", index=False)

    headline = {
        "sweep": sweep,
        "primary_metric": C.PRIMARY_METRIC,
        "ordering_metric": C.ORDERING_METRIC,
        "primary_cell": C.PRIMARY_CELL,
        "k_primary": C.K_PRIMARY,
        "k_primary_note": C.K_PRIMARY_NOTE if hasattr(C, "K_PRIMARY_NOTE") else None,
        "n_runs": int(len(ann)),
        "n_scored_rows": int(len(summary)),
        "n_reference_rows": int(ann["is_reference"].sum()),
        "n_shuffle_draws": int(ann["is_shuffle_draw"].sum()),
        "cells_scored": sorted(
            {f"{r.cell_line}/{r.model}/l{int(r.layer)}"
             for r in summary.itertuples() if not pd.isna(r.layer)}
        ),
        "emits": "hidden_states_only",
        "reference_rows_excluded_from_merge": True,
    }
    (out / f"{sweep}_headline.json").write_text(json.dumps(headline, indent=2, default=str))
    (out / f"{sweep}_basis_check.json").write_text(
        json.dumps(basis_check(summary), indent=2, default=str)
    )
    print(
        f"[analysis] {len(ann)} runs -> {len(summary)} scored rows, "
        f"{headline['n_reference_rows']} reference, {headline['n_shuffle_draws']} shuffled; "
        f"tracked summary in {out}",
        flush=True,
    )
    return headline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="stage1")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the tracked summary goes (default: results/summary)")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.sweep, a.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
