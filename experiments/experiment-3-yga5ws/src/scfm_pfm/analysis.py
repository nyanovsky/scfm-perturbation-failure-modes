"""Turn the sweep's shards into the numbers the report states.

Three things happen here and nowhere else:

* the eight-cell correlation grid is assembled per arm, with the
  pre-registered cell (residual target, reliability-weighted, ceiling-corrected)
  marked as the one the verdict is read from;
* every arm is compared to co-expression **paired on the same perturbations**,
  which is the only comparison the design supports, with a paired bootstrap;
* the resolution the run actually achieved is recomputed from those paired
  differences, replacing the pre-run projection.

Every interval here resamples RESPONSE GROUPS, not perturbations. The scoring
jobs bootstrapped rows, which treats 524 HepG2 perturbations as 524 independent
draws when they are 226 correlated groups (one of them 149 rows). The groups are
recomputed from the bundle's effect matrix by ``groups.GroupTable``, which
reproduces the ``n_groups`` and ``largest_group`` the jobs recorded, and every
``_lo``/``_hi`` in the tables this module writes is replaced with the group
resampled version. Point estimates are untouched: only the uncertainty changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics, paths
from .groups import GroupTable

#: the cell config.py pre-registered, and the cell the verdict is read from.
#: Prediction and truth are both residualized against the TRAINING folds' mean
#: effect, weighted by per-gene reliability and divided by the split-half
#: ceiling.
PREREG_COL = "residual__weighted__disattenuated"
PRIMARY_COL = PREREG_COL
VERDICT_COL = PREREG_COL
CENTERED_COL = "centered__weighted__disattenuated"
#: a second cell, reported beside the pre-registered one and never in place of
#: it. It is the same correlation after the HELD-OUT fold's own per-gene mean is
#: removed from prediction and truth. Two things follow. It uses held-out
#: outcomes, so it is not a prediction a model could make in advance, and it was
#: chosen after seeing that a held-out grouped fold shares a per-gene offset
#: that a near-constant prediction correlates with, so it is post hoc. What it
#: answers is the narrower question of how well an arm separates one held-out
#: perturbation from another, and its shuffled-label draws sit at zero where the
#: pre-registered cell's sit slightly above it.
DISCRIMINATION_COL = CENTERED_COL
REFERENCE_ARM = "coexp"
SHUFFLE_TAG = "+shuffled_labels_d"


def base_arm(arm: str) -> str:
    """The arm a shuffled-label draw was made from."""
    return arm.split(SHUFFLE_TAG)[0]


#: arms that exist only because of the protein control: the raw ESM2 tables and
#: the random-vector null cut to ESM2-650M's width. They were scored in their
#: own job, months after the sweep, so roll-ups that describe *the sweep* --
#: the shuffled-label ceiling and the grid-cell ordering agreement -- are taken
#: over the arms the sweep itself scored. The protein arms get their own
#: versions of both, and every per-arm table holds all of them.
PROTEIN_ONLY_ARMS = ("esm2_650m_emb", "esm2_3b_emb", "null_randgene_d2560")


def is_protein_arm(arm: str) -> bool:
    """True for an arm the protein control introduced, shuffled draws included."""
    return base_arm(arm) in PROTEIN_ONLY_ARMS


#: the two-table arm, its two parts, and the random projection cut to its width
CONCAT_ARM = "scgpt_esm2_3b_emb"
CONCAT_PARTS = ("scgpt_emb", "esm2_3b_emb")
CONCAT_NULL = "null_randgene_d6144"
#: the nulls the two parts are read against, so the three arms in the comparison
#: each appear beside a random projection of their own width
CONCAT_PART_NULLS = ("null_randgene_d1024", "null_randgene_d5120")

#: pairs of arms whose per-perturbation scores are correlated across held-out
#: perturbations. The first is the question the two-table arm was added for; the
#: rest are the scale it has to be read against -- two sizes of one protein
#: model (one route, two tables), two model tables, and a model table against
#: co-expression, which is a different route entirely.
AGREEMENT_PAIRS = (
    ("scgpt_emb", "esm2_3b_emb"),
    ("esm2_650m_emb", "esm2_3b_emb"),
    ("scgpt_emb", "transcriptformer_emb"),
    ("scgpt_emb", "coexp"),
)

#: Fixed before the two-table arm was scored. Both are Spearman correlations
#: between two arms' per-perturbation scores over the same held-out
#: perturbations, so they say whether the two arms succeed and fail on the same
#: perturbations rather than whether their averages agree.
CONCAT_STRONG_AGREEMENT = 0.5
CONCAT_WEAK_AGREEMENT = 0.3

#: the reading, in the words it was fixed in
CONCAT_RULE = {
    "redundant": (
        "the two tables are proxies for one underlying quantity reached by different "
        "routes: the concatenation does not beat the better of its two parts by more "
        "than the paired comparison resolves, and the parts' per-perturbation scores "
        f"are correlated at Spearman {CONCAT_STRONG_AGREEMENT} or above"
    ),
    "complementary": (
        "the two tables carry different information, which is a separate finding: the "
        "concatenation beats both parts with an interval clear of zero, and the parts' "
        f"per-perturbation scores are correlated below Spearman {CONCAT_WEAK_AGREEMENT}"
    ),
    "mixed": "anything else, reported as mixed rather than resolved either way",
}


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


#: what identifies a run apart from which target matrix it was scored against
ARM_KEYS = ["cell_line", "scope", "split", "arm", "translator"]


def load_sweep(*sweep_dirs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every shard of every wave, later waves winning where they overlap.

    Directories are read in the order given, so the caller states the ordering
    rather than it depending on a filename.

    ``target_source`` is filled in per directory, not once at the end. The
    control-cell job scored the same arms against a target matrix built from one
    half of the control cells and wrote that label into its run table but not
    into its per-perturbation table. Filling the whole concatenation at the end
    labelled those per-perturbation rows as the main target and, being read
    later, they replaced the main target's rows -- which silently moved every
    paired difference onto the control-half target. A directory's
    per-perturbation rows now inherit the label from that directory's own run
    rows.
    """
    runs, per_pert = [], []
    for d in sweep_dirs:
        d = Path(d)
        r_d = [pd.read_parquet(p) for p in sorted(d.glob("runs_shard*.parquet"))]
        p_d = [pd.read_parquet(p) for p in sorted(d.glob("per_perturbation_shard*.parquet"))]
        if not r_d:
            continue
        r_one = pd.concat(r_d, ignore_index=True)
        if "target_source" not in r_one.columns:
            r_one["target_source"] = "run1_all_controls"
        r_one["target_source"] = r_one["target_source"].fillna("run1_all_controls")
        runs.append(r_one)
        if not p_d:
            continue
        p_one = pd.concat(p_d, ignore_index=True)
        if "target_source" in p_one.columns:
            p_one["target_source"] = p_one["target_source"].fillna("run1_all_controls")
        else:
            label = r_one[ARM_KEYS + ["target_source"]].drop_duplicates()
            if label.duplicated(subset=ARM_KEYS).any():
                raise RuntimeError(f"{d}: one arm scored against two targets in one directory")
            p_one = p_one.merge(label, on=ARM_KEYS, how="left")
            p_one["target_source"] = p_one["target_source"].fillna("run1_all_controls")
        per_pert.append(p_one)
    if not runs:
        raise FileNotFoundError(f"no shards under {[str(d) for d in sweep_dirs]}")
    r = pd.concat(runs, ignore_index=True)
    p = pd.concat(per_pert, ignore_index=True)
    r = r.drop_duplicates(subset=RUN_KEYS, keep="last").reset_index(drop=True)
    p = p.drop_duplicates(subset=RUN_KEYS + ["perturbation"], keep="last").reset_index(drop=True)
    return r, p


#: the cell line whose perturbations a row was scored on. Transfer rows are
#: labelled "source->destination" and the scored perturbations are the
#: destination's.
def scored_line(cell_line: str) -> str:
    return cell_line.split("->")[-1]


def contrasts(
    per_pert: pd.DataFrame,
    reference: str = REFERENCE_ARM,
    metric: str = "residual_weighted",
    group_table: GroupTable | None = None,
) -> pd.DataFrame:
    """Paired difference of every arm against a reference arm.

    Arms are scored on the same held-out perturbations, so the comparison is
    paired; an unpaired interval would be wider than the design needs and would
    hide the fact that the arms rise and fall together across perturbations.

    The interval resamples response groups, so it carries the same dependence
    the split is built around. ``min_resolvable_difference_achieved`` is the
    difference a two-sided 5% test would resolve at 80% power given that
    interval's standard error, which is the group-aware version of the
    row-count formula the pre-run projection used.
    """
    rows = []
    keys = ["cell_line", "scope", "split", "translator", "target_source"]
    for key, block in per_pert.groupby(keys, dropna=False):
        ref = block[block.arm == reference]
        if ref.empty:
            continue
        line = scored_line(str(dict(zip(keys, key))["cell_line"]))
        ref = ref.set_index("perturbation")[metric]
        for arm, sub in block.groupby("arm"):
            if arm == reference:
                continue
            s = sub.set_index("perturbation")[metric]
            shared = ref.index.intersection(s.index)
            a = s.loc[shared].to_numpy()
            b = ref.loc[shared].to_numpy()
            if group_table is None:
                raise ValueError("contrasts needs a GroupTable: row-wise intervals are too narrow")
            g = group_table.groups(line, [str(x) for x in shared])
            boot = metrics.bootstrap_median_difference_grouped(
                a, b, g, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_CI, C.SEED
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
                    "n_groups_paired": int(boot["n_groups"]),
                    "se_grouped": float(boot["se"]),
                    "sd_paired": sd_paired,
                    "paired_correlation": float(
                        np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan
                    ),
                    "min_resolvable_difference_achieved": 2.801583 * float(boot["se"]),
                }
            )
    return pd.DataFrame(rows)


#: the six correlation cells every run carries, as (column stem, per-perturbation
#: column). ``__attenuated`` and ``__disattenuated`` intervals are both derived
#: from the same resampled medians.
GRID_CELLS = tuple(
    (f"{t}__{w}", f"{t}_{w}") for t in C.GRID_TARGETS for w in C.GRID_WEIGHTINGS
)


def regroup_run_intervals(
    runs: pd.DataFrame, per_pert: pd.DataFrame, group_table: GroupTable
) -> pd.DataFrame:
    """Replace every run's row-wise interval with a group-resampled one.

    The medians the jobs wrote are re-derived here as a check and must match to
    within floating point; only ``_lo`` and ``_hi`` move. Rows with no
    per-perturbation record keep what they had and are counted in the return
    value's ``regrouped`` attribute.
    """
    runs = runs.copy()
    for col in ("n_groups_scored", "largest_group_scored"):
        runs[col] = pd.array([pd.NA] * len(runs), dtype="Int64")
    index = {tuple(v) for v in runs[RUN_KEYS].itertuples(index=False, name=None)}
    pos = {k: i for i, k in enumerate(runs[RUN_KEYS].itertuples(index=False, name=None))}
    n_done = 0
    max_median_gap = 0.0
    for key, block in per_pert.groupby(RUN_KEYS, dropna=False):
        if key not in index:
            continue
        i = pos[key]
        line = scored_line(str(key[0]))
        g = group_table.groups(line, [str(x) for x in block.perturbation])
        for stem, pcol in GRID_CELLS:
            values = block[pcol].to_numpy(dtype=float)
            boot = metrics.bootstrap_median_grouped(
                values, g, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_CI, C.SEED
            )
            if not np.isfinite(boot["median"]):
                continue
            recorded = float(runs.at[runs.index[i], f"{stem}__attenuated"])
            if np.isfinite(recorded):
                max_median_gap = max(max_median_gap, abs(recorded - boot["median"]))
            root = np.sqrt(float(runs.at[runs.index[i], f"{stem}__ceiling"]))
            runs.at[runs.index[i], f"{stem}__attenuated_lo"] = boot["lo"]
            runs.at[runs.index[i], f"{stem}__attenuated_hi"] = boot["hi"]
            runs.at[runs.index[i], f"{stem}__disattenuated_lo"] = boot["lo"] / root
            runs.at[runs.index[i], f"{stem}__disattenuated_hi"] = boot["hi"] / root
            runs.at[runs.index[i], "n_groups_scored"] = boot["n_groups"]
            runs.at[runs.index[i], "largest_group_scored"] = boot["largest_group"]
        n_done += 1
    missing = len(runs) - n_done
    if max_median_gap > 1e-9:
        raise RuntimeError(
            f"regrouped medians disagree with the recorded ones by {max_median_gap:.3g}"
        )
    if missing:
        print(f"[regroup] {missing} of {len(runs)} runs have no per-perturbation record")
    return runs


def null_ceiling(runs: pd.DataFrame, cell: dict, col: str) -> float:
    """The largest shuffled-label draw in one cell of the sweep: the number a
    score has to be above before it means anything."""
    d = runs
    for k, v in cell.items():
        d = d[d[k] == v]
    d = d[d.arm.str.contains(SHUFFLE_TAG, regex=False) & ~d.arm.map(is_protein_arm)]
    if d.empty:
        return float("nan")
    return float(np.nanmax(d[col].to_numpy(dtype=float)))


def headline(runs: pd.DataFrame, con: pd.DataFrame, nulls: pd.DataFrame | None = None) -> dict:
    """The verdict cell of the grid, per cell line, on the primary split."""
    out: dict = {
        "verdict_metric": VERDICT_COL,
        "preregistered_metric": PREREG_COL,
        "discrimination_metric": DISCRIMINATION_COL,
        "verdict_metric_note": (
            "the verdict is read from the pre-registered cell. The fold-centred cell is "
            "reported beside it as a secondary, post-hoc reading: it removes the "
            "held-out fold's own per-gene mean from both sides, so it uses held-out "
            "outcomes and answers the narrower question of how well an arm separates "
            "one held-out perturbation from another. Arms are ordered by it because that "
            "is the harder question, not because the verdict rests on it"
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
        ranked = block.sort_values(DISCRIMINATION_COL, ascending=False)
        cell = {
            "cell_line": line,
            "scope": "full",
            "split": "grouped",
            "translator": "mlp",
            "target_source": "run1_all_controls",
        }
        null_prereg = null_ceiling(runs, cell, PREREG_COL)
        null_verdict = null_ceiling(runs, cell, DISCRIMINATION_COL)
        c_all = con[
            (con.cell_line == line)
            & (con.scope == "full")
            & (con.split == "grouped")
            & (con.translator == "mlp")
            & (con.target_source == "run1_all_controls")
        ]
        c = c_all[c_all.metric == "centered_weighted"]
        c_prereg = c_all[c_all.metric == "residual_weighted"]
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
                    "score": float(getattr(r, DISCRIMINATION_COL)),
                    "lo": float(getattr(r, DISCRIMINATION_COL + "_lo")),
                    "hi": float(getattr(r, DISCRIMINATION_COL + "_hi")),
                    "clears_null": bool(float(getattr(r, DISCRIMINATION_COL)) > null_verdict),
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
            "vs_coexpression_preregistered": [
                {
                    "arm": r.arm,
                    "median_difference": float(r.median_difference),
                    "lo": float(r.lo),
                    "hi": float(r.hi),
                    "n_paired": int(r.n_paired),
                    "n_groups_paired": int(r.n_groups_paired),
                    "achieved_min_resolvable_difference": float(
                        r.min_resolvable_difference_achieved
                    ),
                }
                for r in c_prereg.sort_values("median_difference", ascending=False).itertuples()
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
        DISCRIMINATION_COL, DISCRIMINATION_COL + "_lo", DISCRIMINATION_COL + "_hi", PREREG_COL,
        "residual__weighted__ceiling", "centered__weighted__ceiling",
        "discrimination_ours_residual_mean",
    ]
    wide = d[[c for c in keep if c in d.columns]].copy()
    wide["variant"] = wide["target_source"].astype(str).str.rsplit("__", n=1).str[-1]
    return wide.sort_values(["cell_line", "arm", "translator", "variant"])


def protein_reference_check(
    reference: pd.DataFrame, primary: pd.DataFrame, job_key: str = "protein_job"
) -> list[dict]:
    """Arms a later job re-ran, against the same arms in the main sweep.

    The protein arms were scored in a job that mounted a rebuilt bundle. The
    same job re-ran four arms the main sweep already has, so any difference is
    the bundle and the job, not the feature. These rows are kept out of the
    merged table -- a merge would replace the scored primary rows with the
    re-run ones. The two-table job re-runs its own arms the same way, and names
    its column with ``job_key``.
    """
    if reference.empty or primary.empty:
        return []
    p = (
        primary[primary.target_source == "run1_all_controls"]
        if "target_source" in primary.columns
        else primary
    )
    base = {
        (r.cell_line, r.scope, r.split, r.arm, r.translator): float(getattr(r, DISCRIMINATION_COL))
        for r in p.itertuples()
    }
    rows = []
    for r in reference.itertuples():
        key = (r.cell_line, r.scope, r.split, r.arm, r.translator)
        if key not in base:
            continue
        rows.append({
            "cell_line": r.cell_line,
            "scope": r.scope,
            "split": r.split,
            "arm": r.arm,
            "translator": r.translator,
            job_key: float(getattr(r, DISCRIMINATION_COL)),
            "main_sweep": base[key],
            "difference": float(getattr(r, DISCRIMINATION_COL)) - base[key],
        })
    return rows


#: the arms the protein control is paired against, one paired bootstrap each
PROTEIN_REFERENCES = ("coexp", "transcriptformer_emb", "scprint_emb", "scgpt_emb")


def protein_summary(
    runs: pd.DataFrame,
    per_pert: pd.DataFrame,
    nulls: pd.DataFrame,
    group_table: GroupTable | None = None,
) -> dict:
    """The protein arm read against the arms it is a control for.

    One block per cell line on the cell the verdict is read from: the raw ESM2
    arms, the model embedding arms they are the control for, co-expression, the
    width-matched random-vector nulls, and the largest shuffled-label draw.

    The comparison that answers the question is paired: every arm is scored on
    the same held-out perturbations under the same folds, so the difference
    between the protein arm and a model's embedding arm is read per perturbation
    rather than between two independent intervals.
    """
    keep = (
        (runs.scope == "full") & (runs.split == "grouped") & (runs.translator == "mlp")
        & (runs.target_source == "run1_all_controls")
    )
    d = runs[keep]
    if d.empty or not d.arm.str.startswith("esm2").any():
        return {}
    out: dict = {"cell": "full scope, grouped split, MLP translator", "metric": DISCRIMINATION_COL,
                 "lines": {}}
    for line, block in d.groupby("cell_line"):
        want = block[
            block.arm.str.startswith(("esm2", "scgpt_emb", "scprint_emb",
                                      "transcriptformer_emb", "aido_cell_emb",
                                      "coexp", "null_randgene"))
            & ~block.arm.str.contains(SHUFFLE_TAG, regex=False)
        ]
        rows = [
            {
                "arm": r.arm,
                "family": r.family,
                "feature_dim": int(r.feature_dim),
                "score": float(getattr(r, DISCRIMINATION_COL)),
                "lo": float(getattr(r, DISCRIMINATION_COL + "_lo")),
                "hi": float(getattr(r, DISCRIMINATION_COL + "_hi")),
                "preregistered": float(getattr(r, PREREG_COL)),
            }
            for r in want.sort_values(DISCRIMINATION_COL, ascending=False).itertuples()
        ]
        shuffled = block[
            block.arm.str.contains(SHUFFLE_TAG, regex=False) & block.arm.map(is_protein_arm)
        ]
        pp = per_pert[
            (per_pert.cell_line == line) & (per_pert.scope == "full")
            & (per_pert.split == "grouped") & (per_pert.translator == "mlp")
            & (per_pert.target_source == "run1_all_controls")
        ]
        paired = []
        paired_prereg = []
        for ref in PROTEIN_REFERENCES:
            if ref not in set(pp.arm):
                continue
            for metric, sink in (("centered_weighted", paired),
                                 ("residual_weighted", paired_prereg)):
                c = contrasts(pp, reference=ref, metric=metric, group_table=group_table)
                for r in c[c.arm.str.startswith("esm2")
                           & ~c.arm.str.contains(SHUFFLE_TAG, regex=False)].itertuples():
                    sink.append({
                        "arm": r.arm,
                        "reference": ref,
                        "median_difference": float(r.median_difference),
                        "lo": float(r.lo),
                        "hi": float(r.hi),
                        "n_paired": int(r.n_paired),
                        "n_groups_paired": int(r.n_groups_paired),
                        "min_resolvable_difference_achieved": float(
                            r.min_resolvable_difference_achieved
                        ),
                    })
        out["lines"][line] = {
            "arms": rows,
            "largest_protein_shuffled_draw": (
                float(np.nanmax(shuffled[DISCRIMINATION_COL].to_numpy(dtype=float)))
                if not shuffled.empty else float("nan")
            ),
            "n_protein_shuffled_draws": int(len(shuffled)),
            "paired_differences": paired,
            "paired_differences_preregistered": paired_prereg,
        }
    if not nulls.empty:
        band = nulls[nulls.base_arm.str.startswith("esm2")]
        out["protein_shuffled_label_band"] = band.to_dict("records")
    return out


#: arms carried into the cross-line protein table, in the order they are read
PROTEIN_TRANSFER_ARMS = (
    "esm2_3b_emb",
    "esm2_650m_emb",
    "transcriptformer_emb",
    "scprint_emb",
    "scgpt_emb",
    "coexp",
    "null_randgene_d2560",
    "null_randgene_d5120",
)


def protein_transfer_summary(transfer: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """The cross-line table, with the protein arms alongside what they control for.

    The protein arms were scored in their own job. Where that job also re-ran a
    model embedding arm, its own number is used, so the comparison sits inside
    one job; where it did not, the main transfer sweep's number is used. Every
    row says which of the two it came from, and
    ``protein_transfer_reference_check`` reports how far the re-runs sit from
    the main sweep.
    """
    if transfer.empty or not transfer.arm.str.startswith("esm2").any():
        return {}
    keys = ["cell_line", "scope", "split", "arm"]
    own = transfer[transfer.arm.str.startswith(("esm2", "null_randgene_d2560"))]
    pieces = [own.assign(source="protein_job")]
    covered = set(map(tuple, own[keys].to_numpy().tolist()))
    if not reference.empty:
        pieces.append(reference.assign(source="protein_job"))
        covered |= set(map(tuple, reference[keys].to_numpy().tolist()))
    rest = transfer[
        transfer.arm.isin(PROTEIN_TRANSFER_ARMS)
        & ~pd.Series(
            [tuple(x) in covered for x in transfer[keys].to_numpy().tolist()],
            index=transfer.index,
        )
    ]
    pieces.append(rest.assign(source="main_sweep"))
    d = pd.concat(pieces, ignore_index=True)
    d = d[(d.translator == "mlp") & d.arm.isin(PROTEIN_TRANSFER_ARMS)]
    groups = {
        "protein_derived": ("esm2_3b_emb", "esm2_650m_emb", "transcriptformer_emb",
                            "scprint_emb"),
        "scgpt": ("scgpt_emb",),
        "coexpression_and_nulls": ("coexp", "null_randgene_d2560", "null_randgene_d5120"),
    }
    unseen = d[d.scope == "transfer_unseen"]
    # Counting the three destination-reference modes together would mix sample
    # sizes: the mode that builds a reference out of half the destination's
    # perturbations can only score the other half, so it holds 78-89 rows where
    # the other two hold 156-177. The count is therefore reported per mode, with
    # the rows each mode scored.
    clearance = {}
    for mode, mode_block in unseen.groupby("split"):
        per_mode = {}
        for name, arms in groups.items():
            sel = mode_block[mode_block.arm.isin(arms)]
            per_mode[name] = "{} of {}".format(
                int((sel[VERDICT_COL + "_lo"] > 0).sum()), int(len(sel))
            )
            per_mode[name + "__discrimination_metric"] = "{} of {}".format(
                int((sel[DISCRIMINATION_COL + "_lo"] > 0).sum()), int(len(sel))
            )
        sizes = sorted({int(x) for x in mode_block.n_perturbations})
        per_mode["n_perturbations_scored"] = (
            str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]}"
        )
        clearance[str(mode).replace("cross_line__", "")] = per_mode
    out: dict = {"metric": VERDICT_COL,
                 "discrimination_metric": DISCRIMINATION_COL,
                 "blocks": {},
                 "unseen_gene_clearance": clearance,
                 "unseen_gene_clearance_note": (
                     "per destination-reference mode, how many (direction x arm) cells of "
                     "the unseen-gene cross-line evaluation have a group-resampled interval "
                     "clear of zero, out of 8 protein-derived (2 directions x 4 tables), "
                     "2 scGPT and 6 co-expression-or-null cells. The modes are not "
                     "interchangeable: dest_reference scores half the destination's "
                     "perturbations because the other half builds its reference"
                 )}
    for (line, scope, split), block in d.groupby(["cell_line", "scope", "split"]):
        out["blocks"][f"{line}|{scope}|{split}"] = [
            {
                "arm": r.arm,
                "family": r.family,
                "source": r.source,
                "score": float(getattr(r, DISCRIMINATION_COL)),
                "lo": float(getattr(r, DISCRIMINATION_COL + "_lo")),
                "hi": float(getattr(r, DISCRIMINATION_COL + "_hi")),
                "preregistered": float(getattr(r, PREREG_COL)),
                "preregistered_lo": float(getattr(r, PREREG_COL + "_lo")),
                "preregistered_hi": float(getattr(r, PREREG_COL + "_hi")),
                "n_perturbations": int(r.n_perturbations),
            }
            for r in block.sort_values(DISCRIMINATION_COL, ascending=False).itertuples()
        ]
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, so one perturbation with a huge score cannot set it."""
    if x.size < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def arm_agreement(
    per_pert: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = AGREEMENT_PAIRS,
    metric_cols: tuple[str, ...] = ("residual_weighted", "centered_weighted"),
) -> pd.DataFrame:
    """How far two arms rise and fall together across held-out perturbations.

    Two arms can reach the same average score either by succeeding on the same
    perturbations or by succeeding on different ones, and an average cannot tell
    those apart. The correlation of the per-perturbation scores can. Spearman is
    what the reading rule uses, because the question is whether the two arms
    order the perturbations the same way, not whether the sizes of their scores
    match.
    """
    d = per_pert[(per_pert.scope == "full") & (per_pert.target_source == "run1_all_controls")]
    keys = ["cell_line", "split", "translator"]
    rows = []
    for key, block in d.groupby(keys, dropna=False):
        by_arm = {a: sub.set_index("perturbation") for a, sub in block.groupby("arm")}
        for a, b in pairs:
            if a not in by_arm or b not in by_arm:
                continue
            shared = by_arm[a].index.intersection(by_arm[b].index)
            if shared.size < 3:
                continue
            for col in metric_cols:
                x = by_arm[a].loc[shared, col].to_numpy(dtype=float)
                y = by_arm[b].loc[shared, col].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                rows.append({
                    **dict(zip(keys, key)),
                    "arm_a": a,
                    "arm_b": b,
                    "metric": col,
                    "n_perturbations": int(ok.sum()),
                    "pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                    "spearman": _spearman(x[ok], y[ok]),
                })
    return pd.DataFrame(rows)


def concat_reading(beats_better_part: bool, parts_spearman: float) -> str:
    """Apply the rule to one cell. Written before the arm was scored."""
    strong = bool(np.isfinite(parts_spearman) and parts_spearman >= CONCAT_STRONG_AGREEMENT)
    weak = bool(np.isfinite(parts_spearman) and parts_spearman < CONCAT_WEAK_AGREEMENT)
    if not beats_better_part and strong:
        return "redundant"
    if beats_better_part and weak:
        return "complementary"
    return "mixed"


def concat_summary(
    runs: pd.DataFrame,
    per_pert: pd.DataFrame,
    agree: pd.DataFrame,
    group_table: GroupTable | None = None,
) -> dict:
    """The two-table arm read against each of the two tables it is made of.

    scGPT's table and the raw protein table reach the same score while the
    protein-space similarity measure says scGPT's table carries no protein
    structure. Either they are two routes to one quantity, in which case putting
    them side by side buys nothing and they succeed on the same perturbations,
    or they carry different information, in which case the concatenation beats
    both and they succeed on different perturbations. The comparison against
    each part is paired on the same held-out perturbations, and the arm has its
    own width-matched random projection because it is wider than either part.
    """
    keep = (
        (runs.scope == "full") & (runs.translator == "mlp")
        & (runs.target_source == "run1_all_controls")
    )
    d = runs[keep]
    if d.empty or CONCAT_ARM not in set(d.arm):
        return {}
    want = (CONCAT_ARM, *CONCAT_PARTS, CONCAT_NULL, *CONCAT_PART_NULLS, "coexp")
    out: dict = {
        "cell": "full scope, MLP translator",
        "metric": PREREG_COL,
        "diagnostic_metric": DISCRIMINATION_COL,
        "rule": CONCAT_RULE,
        "agreement_thresholds": {
            "strong_at_or_above": CONCAT_STRONG_AGREEMENT,
            "weak_below": CONCAT_WEAK_AGREEMENT,
            "statistic": "Spearman correlation of per-perturbation scores",
        },
        "lines": {},
    }
    for (line, split), block in d.groupby(["cell_line", "split"]):
        sel = block[block.arm.isin(want)]
        scores = {
            r.arm: {
                "arm": r.arm,
                "feature_dim": int(r.feature_dim),
                "score": float(getattr(r, PREREG_COL)),
                "lo": float(getattr(r, PREREG_COL + "_lo")),
                "hi": float(getattr(r, PREREG_COL + "_hi")),
                "fold_centred": float(getattr(r, DISCRIMINATION_COL)),
                "fold_centred_lo": float(getattr(r, DISCRIMINATION_COL + "_lo")),
                "fold_centred_hi": float(getattr(r, DISCRIMINATION_COL + "_hi")),
                "n_perturbations": int(r.n_perturbations),
            }
            for r in sel.itertuples()
        }
        if CONCAT_ARM not in scores:
            continue
        pp = per_pert[
            (per_pert.cell_line == line) & (per_pert.split == split)
            & (per_pert.scope == "full") & (per_pert.translator == "mlp")
            & (per_pert.target_source == "run1_all_controls")
        ]
        paired: dict = {}
        for part in CONCAT_PARTS:
            two = pp[pp.arm.isin([CONCAT_ARM, part])]
            for col, tag in (("residual_weighted", "preregistered"),
                             ("centered_weighted", "fold_centred")):
                con = contrasts(two, reference=part, metric=col, group_table=group_table)
                row = con[con.arm == CONCAT_ARM]
                if row.empty:
                    continue
                r = row.iloc[0]
                paired[f"{part}__{tag}"] = {
                    "difference": float(r.median_difference),
                    "lo": float(r.lo),
                    "hi": float(r.hi),
                    "n_paired": int(r.n_paired),
                    "n_groups_paired": int(r.n_groups_paired),
                    "min_resolvable_difference": float(r.min_resolvable_difference_achieved),
                }
        a = agree[
            (agree.cell_line == line) & (agree.split == split) & (agree.translator == "mlp")
        ] if not agree.empty else agree
        agreement = {
            f"{r.arm_a}__{r.arm_b}__{r.metric}": {
                "spearman": float(r.spearman),
                "pearson": float(r.pearson),
                "n_perturbations": int(r.n_perturbations),
            }
            for r in a.itertuples()
        } if not a.empty else {}
        present = [p for p in CONCAT_PARTS if p in scores]
        better = max(present, key=lambda p: scores[p]["score"]) if present else None
        diff = paired.get(f"{better}__preregistered") if better else None
        key = f"{CONCAT_PARTS[0]}__{CONCAT_PARTS[1]}__residual_weighted"
        spearman = agreement.get(key, {}).get("spearman", float("nan"))
        beats = bool(diff and diff["lo"] > 0)
        reading = concat_reading(beats, spearman)
        out["lines"][f"{line}|{split}"] = {
            "scores": [scores[k] for k in want if k in scores],
            "paired_against_parts": paired,
            "better_part": better,
            "agreement": agreement,
            "parts_spearman": spearman,
            "beats_better_part": beats,
            "reading": reading,
        }
    grouped = [v["reading"] for k, v in out["lines"].items() if k.endswith("|grouped")]
    out["reading"] = grouped[0] if grouped and len(set(grouped)) == 1 else "mixed"
    out["reading_read_from"] = "the grouped split in both cell lines; mixed unless they agree"
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


def anchor_sensitivity(
    sens: pd.DataFrame,
    primary: pd.DataFrame,
    sens_pert: pd.DataFrame | None = None,
    group_table: GroupTable | None = None,
) -> dict:
    """Anchor and form sensitivity, against two references.

    The sensitivity job re-runs each model's mean-control response arm beside
    the other anchors, so the anchor comparison is within one job. Those re-runs
    are also compared against the same arm in the main sweep, which is a
    reproducibility check on the rebuilt bundle rather than part of the
    sensitivity question.
    """
    if sens.empty:
        return {}
    s = sens[sens.translator == "mlp"]
    base = {
        (r.cell_line, r.arm): float(getattr(r, DISCRIMINATION_COL))
        for r in primary[
            (primary.translator == "mlp")
            & (primary.scope == "full")
            & (primary.split == "grouped")
            & (primary.target_source == "run1_all_controls")
        ].itertuples()
    }
    rows = []
    for r in s.itertuples():
        model, anchor = (r.arm.split("@") + ["mean_control"])[:2]
        rows.append(
            {
                "cell_line": r.cell_line,
                "arm": r.arm,
                "model": model.replace("_resp", ""),
                "anchor": anchor,
                "score": float(getattr(r, DISCRIMINATION_COL)),
                "lo": float(getattr(r, DISCRIMINATION_COL + "_lo")),
                "hi": float(getattr(r, DISCRIMINATION_COL + "_hi")),
                "preregistered": float(getattr(r, PREREG_COL)),
                "n_perturbations": int(r.n_perturbations),
                "feature_dim": int(r.feature_dim),
            }
        )
    out = pd.DataFrame(rows)
    ref = {
        (r.cell_line, r.model): r.score
        for r in out[out.anchor == "mean_control"].itertuples()
    }
    out["minus_mean_control"] = [
        r.score - ref.get((r.cell_line, r.model), float("nan")) for r in out.itertuples()
    ]
    rerun = [
        {
            "cell_line": r.cell_line,
            "arm": r.arm,
            "sensitivity_job": r.score,
            "main_sweep": base.get((r.cell_line, r.arm), float("nan")),
            "difference": r.score - base.get((r.cell_line, r.arm), float("nan")),
        }
        for r in out[out.anchor == "mean_control"].itertuples()
        if (r.cell_line, r.arm) in base
    ]
    coexp = s[s.arm == "coexp"]
    if not coexp.empty:
        rerun.append(
            {
                "cell_line": str(coexp.cell_line.iloc[0]),
                "arm": "coexp",
                "sensitivity_job": float(coexp[DISCRIMINATION_COL].iloc[0]),
                "main_sweep": base.get((str(coexp.cell_line.iloc[0]), "coexp"), float("nan")),
                "difference": float(coexp[DISCRIMINATION_COL].iloc[0])
                - base.get((str(coexp.cell_line.iloc[0]), "coexp"), float("nan")),
            }
        )
    paired: list[dict] = []
    if sens_pert is not None and not sens_pert.empty:
        for model in sorted(set(out.model)):
            ref = f"{model}_resp"
            if ref not in set(sens_pert.arm):
                continue
            con = contrasts(sens_pert, reference=ref, metric="centered_weighted",
                            group_table=group_table)
            con = con[con.arm.str.startswith(f"{ref}@")]
            for row in con.itertuples():
                paired.append({
                    "model": model,
                    "anchor": row.arm.split("@")[1],
                    "median_difference": float(row.median_difference),
                    "lo": float(row.lo),
                    "hi": float(row.hi),
                    "min_resolvable_difference_achieved": float(
                        row.min_resolvable_difference_achieved
                    ),
                    "n_paired": int(row.n_paired),
                })
    return {
        "rows": out.to_dict("records"),
        "rerun_check": rerun,
        "max_abs_rerun_difference": (
            float(np.nanmax([abs(x["difference"]) for x in rerun])) if rerun else float("nan")
        ),
        "paired_against_mean_control": paired,
    }


def ordering_agreement(grid: pd.DataFrame, with_protein: bool = False) -> pd.DataFrame:
    """Does the grid agree about which arm is ahead?

    Spearman correlation between the arm ordering under the primary cell and
    under every other cell, per (cell line, scope, split, translator). Read over
    the sweep's own arms by default; ``with_protein`` adds the protein control's
    arms, which changes the ranking because it changes the set being ranked.
    """
    rows = []
    keys = ["cell_line", "scope", "split", "translator", "target_source"]
    if not with_protein:
        grid = grid[~grid.arm.map(is_protein_arm)]
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
    ap.add_argument(
        "--sensitivity-sweep",
        default=None,
        help=(
            "shard directories for the anchor/form sensitivity job. Kept in their own "
            "table rather than merged into runs.parquet: that job re-runs some of the "
            "primary arms as its own reference, and a merge would silently overwrite "
            "the scored primary rows with the re-run ones."
        ),
    )
    ap.add_argument(
        "--protein-reference",
        default=None,
        help=(
            "shard directories holding the arms the protein job re-ran as its own "
            "reference. Kept out of runs.parquet for the same reason the sensitivity "
            "job's re-runs are."
        ),
    )
    ap.add_argument(
        "--protein-transfer-reference",
        default=None,
        help=(
            "shard directories holding the cross-line arms the protein job re-ran as "
            "its own reference, kept out of transfer_runs.parquet for the same reason."
        ),
    )
    ap.add_argument(
        "--concat-sweep",
        default=None,
        help=(
            "shard directories for the two-table arm. Kept in their own table rather "
            "than merged into runs.parquet: the two-table arm is a supplementary test "
            "with its own reading, and merging it would change what every 'over all "
            "arms' number in the main result is computed over -- the best arm, the "
            "counts against the null, the ordering agreement between grid cells."
        ),
    )
    ap.add_argument(
        "--concat-reference",
        default=None,
        help=(
            "shard directories holding the arms the two-table job re-ran as its own "
            "reference, kept out of runs.parquet for the same reason."
        ),
    )
    ap.add_argument(
        "--bundle-root",
        default=None,
        help=(
            "the bundle whose targets.npz the response groups are recomputed from. "
            "Every interval this module writes resamples those groups, so there is no "
            "run without it."
        ),
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    gt = GroupTable(Path(args.bundle_root) if args.bundle_root else paths.bundle_root())
    sweep_dirs = (
        [Path(x) for x in args.sweep.split(",") if x]
        if args.sweep
        else [paths.results_dir() / "sweep"]
    )
    out = paths.ensure(Path(args.out) if args.out else paths.results_dir() / "summary")

    runs, per_pert = load_sweep(*sweep_dirs)
    runs = regroup_run_intervals(runs, per_pert, gt)
    con = pd.concat(
        [
            contrasts(per_pert, metric=m, group_table=gt)
            for m in ("residual_weighted", "centered_weighted")
        ],
        ignore_index=True,
    )
    nulls = null_band(runs)
    if not nulls.empty:
        nulls.to_parquet(out / "null_band.parquet", index=False)
    grid = grid_table(runs)
    agree = ordering_agreement(grid)
    agree_protein = ordering_agreement(grid, with_protein=True)
    head = headline(runs, con, nulls)
    overlap = control_cell_overlap(runs)
    if not overlap.empty:
        overlap.to_parquet(out / "control_cell_overlap.parquet", index=False)
        head["control_cell_overlap"] = overlap.to_dict("records")

    transfer_runs = pd.DataFrame()
    if args.transfer_sweep:
        t_runs, t_pert = load_sweep(*[Path(x) for x in args.transfer_sweep.split(",") if x])
        t_runs = regroup_run_intervals(t_runs, t_pert, gt)
        transfer_runs = t_runs
        t_runs.to_parquet(out / "transfer_runs.parquet", index=False)
        t_pert.to_parquet(out / "transfer_per_perturbation.parquet", index=False)
        t_con = pd.concat(
            [
                contrasts(t_pert, metric=m, group_table=gt)
                for m in ("residual_weighted", "centered_weighted")
            ],
            ignore_index=True,
        )
        t_con.to_parquet(out / "transfer_contrasts.parquet", index=False)
        head["transfer"] = {
            f"{line}|{scope}|{split}": [
                {
                    "arm": r.arm,
                    "family": r.family,
                    "score": float(getattr(r, DISCRIMINATION_COL)),
                    "lo": float(getattr(r, DISCRIMINATION_COL + "_lo")),
                    "hi": float(getattr(r, DISCRIMINATION_COL + "_hi")),
                    "preregistered": float(getattr(r, PREREG_COL)),
                    "n_perturbations": int(r.n_perturbations),
                }
                for r in block[block.translator == "mlp"]
                .sort_values(DISCRIMINATION_COL, ascending=False)
                .itertuples()
            ]
            for (line, scope, split), block in t_runs.groupby(["cell_line", "scope", "split"])
        }

    if args.sensitivity_sweep:
        s_runs, s_pert = load_sweep(*[Path(x) for x in args.sensitivity_sweep.split(",") if x])
        s_runs = regroup_run_intervals(s_runs, s_pert, gt)
        s_runs.to_parquet(out / "sensitivity.parquet", index=False)
        head["sensitivity"] = anchor_sensitivity(s_runs, runs, s_pert, group_table=gt)

    agree_arms = arm_agreement(per_pert)
    if not agree_arms.empty:
        agree_arms.to_parquet(out / "arm_agreement.parquet", index=False)
    concat_runs, concat_pert = runs, per_pert
    if args.concat_sweep:
        cs_runs, cs_pert = load_sweep(*[Path(x) for x in args.concat_sweep.split(",") if x])
        cs_runs = regroup_run_intervals(cs_runs, cs_pert, gt)
        cs_runs.to_parquet(out / "concat_runs.parquet", index=False)
        cs_pert.to_parquet(out / "concat_per_perturbation.parquet", index=False)
        # combined only for the two-table summary: the parts live in the main tables
        # and the two-table arm and its width-matched null live in their own
        concat_runs = pd.concat([runs, cs_runs], ignore_index=True)
        concat_pert = pd.concat([per_pert, cs_pert], ignore_index=True)
    concat = concat_summary(concat_runs, concat_pert, agree_arms, group_table=gt)
    if concat:
        head["concat"] = concat
    if args.concat_reference:
        c_runs, c_pert = load_sweep(*[Path(x) for x in args.concat_reference.split(",") if x])
        c_runs = regroup_run_intervals(c_runs, c_pert, gt)
        c_runs.to_parquet(out / "concat_reference.parquet", index=False)
        head["concat_reference_check"] = protein_reference_check(
            c_runs, runs, job_key="concat_job"
        )

    protein = protein_summary(runs, per_pert, nulls, group_table=gt)
    if protein:
        head["protein"] = protein
    if args.protein_reference:
        p_runs, p_pert = load_sweep(*[Path(x) for x in args.protein_reference.split(",") if x])
        p_runs = regroup_run_intervals(p_runs, p_pert, gt)
        p_runs.to_parquet(out / "protein_reference.parquet", index=False)
        head["protein_reference_check"] = protein_reference_check(p_runs, runs)

    pt_runs = pd.DataFrame()
    if args.protein_transfer_reference:
        pt_runs, pt_pert = load_sweep(
            *[Path(x) for x in args.protein_transfer_reference.split(",") if x]
        )
        pt_runs = regroup_run_intervals(pt_runs, pt_pert, gt)
        pt_runs.to_parquet(out / "protein_transfer_reference.parquet", index=False)
        head["protein_transfer_reference_check"] = protein_reference_check(
            pt_runs, transfer_runs
        )
    protein_transfer = protein_transfer_summary(transfer_runs, pt_runs)
    if protein_transfer:
        head["protein_transfer"] = protein_transfer

    runs.to_parquet(out / "runs.parquet", index=False)
    per_pert.to_parquet(out / "per_perturbation.parquet", index=False)
    con.to_parquet(out / "contrasts.parquet", index=False)
    grid.to_parquet(out / "grid.parquet", index=False)
    agree.to_parquet(out / "ordering_agreement.parquet", index=False)
    agree_protein.to_parquet(out / "ordering_agreement_with_protein.parquet", index=False)
    (out / "headline.json").write_text(json.dumps(head, indent=2))
    print(json.dumps(head, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
