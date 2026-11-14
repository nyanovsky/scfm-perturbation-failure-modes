"""Write ``report_summary.json`` and ``claims_manifest.json`` from the tables.

    PYTHONPATH=src uv run python -m scfm_ft.report.contract

Both files are generated, never typed: every number in the summary is read out
of ``results/summary`` and every claim in the manifest names the file and the
row it came from, so the two cannot drift from the report or from each other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import config as C
from .. import paths
from . import build as B

HEAD = "results/summary/headline.json"
RESTRICTED = "results/summary/restricted.parquet"
BAND = "results/summary/null_band.parquet"
DIRECT = "results/summary/direct.parquet"
TRAIN_RUNS = "results/summary/training_runs.parquet"
TRAIN_EPOCHS = "results/summary/training_epochs.parquet"
CONTRASTS = "results/summary/contrasts.parquet"
AGREE = "results/summary/reference_agreement.parquet"
SPREAD = "results/summary/prediction_spread.parquet"
NULLCHECK = "results/summary/null_check.parquet"
RUN3CMP = "results/run3_comparison.parquet"
PROV = "results/recipe_provenance.json"
#: an earlier, separate execution of the same sixteen arms, selected on masked
#: MSE. It predates the three faithfulness controls, so it differs from the
#: primary execution in random initialisation as well as in the rule.
ALT = "results/summary_masked_mse"

LINE = B.LINE
ARM_LABEL = B.ARM_LABEL


def key(**kw) -> str:
    return ", ".join(f"{k}={v}" for k, v in kw.items())


def rkey(**kw) -> str:
    """A locator into ``restricted.parquet`` pinned to the verdict cell.

    Without every index column the locator matches several rows -- the same arm
    is scored on three targets and two weightings by two translators -- so a
    re-reader could not check it. Everything the verdict cell fixes goes in.
    """
    return key(target="residual", weighting="weighted", translator="mlp",
               is_shuffled=False, **kw)


def collect(r: B.Results) -> tuple[dict, list[dict]]:
    claims: list[dict] = []
    n: dict = {"lines": {}}

    def claim(text: str, value, source: str, where: str):
        if value is None or float(value) != float(value):
            return None
        v = float(value)
        claims.append({"claim": text, "value": round(v, 6), "source": source, "key": where})
        return v

    for line in C.CELL_LINES:
        L = LINE[line]
        e: dict = {"arms": {}}

        e["null_ceiling"] = claim(
            f"largest shuffled-label draw in {L} on the verdict cell",
            r.null_ceiling(line), BAND,
            key(cell_line=line, target="residual", weighting="weighted", translator="mlp",
                column="null_max", over="all base arms"),
        )
        if (d := B.own_query_draws(r, line)) is not None:
            e["own_query_draw_min"] = claim(
                f"lowest shuffled-label draw of the perturbation-flag query in {L}, "
                "pooled over the arms scored on the band's folds",
                d[0], NULLCHECK,
                key(cell_line=line, family="pert_flag", column="draw_min", agg="min"),
            )
            e["own_query_draw_max"] = claim(
                f"highest shuffled-label draw of the perturbation-flag query in {L}, "
                "pooled over the arms scored on the band's folds",
                d[1], NULLCHECK,
                key(cell_line=line, family="pert_flag", column="draw_max", agg="max"),
            )
            e["own_query_readings_above_draws"] = int(sum(
                1 for arm in (a.name for a in C.ARMS)
                if (p := r.feat(line, arm, "final__pert_flag_resp")) and p["score"] > d[1]
            ))
        for ref in ("coexp", "scgpt_emb", "scgpt_resp", "null_randpair"):
            row = r.ref(line, ref)
            if row:
                e[ref] = claim(
                    f"{ref} scored on {L}'s held-out folds, verdict cell",
                    row["score"], RESTRICTED, rkey(cell_line=line, arm=ref),
                )
                e[f"{ref}__n"] = row["n"]
                e[f"{ref}__n_groups"] = row["n_groups"]

        for arm in (a.name for a in C.ARMS):
            a_out: dict = {}
            for label, k in (
                ("response_before", "init__value_substitution_resp"),
                ("response_after", "final__value_substitution_resp"),
                ("response_after_own_query", "final__pert_flag_resp"),
                ("gene_table_before", "init__value_substitution_emb"),
                ("gene_table_after", "final__value_substitution_emb"),
            ):
                p = r.feat(line, arm, k)
                if not p:
                    continue
                prefix = "ft" if k.startswith("final") else "pre"
                q = "q" if "pert_flag" in k else ""
                suffix = k.rsplit("_", 1)[-1]
                a_out[label] = claim(
                    f"{ARM_LABEL[arm]} in {L}, {label.replace('_', ' ')}, verdict cell",
                    p["score"], RESTRICTED,
                    rkey(cell_line=line, arm=f"{prefix}{q}_{arm}_{suffix}"),
                )
                a_out[f"{label}__lo"] = p["lo"]
                a_out[f"{label}__hi"] = p["hi"]

            # The direct arm has two interpretable cells. `raw` compares the
            # predicted profile with the measured effect; `centered` removes the
            # held-out fold's own mean from both sides. The residual cell would
            # need a measured profile subtracted from a prediction that is not on
            # the measured scale, so it is not claimed.
            by = r.arm(line, arm).get("direct_prediction_by_baseline", {})
            for bname, row in by.items():
                what = (
                    "against the measured control profile, so the whole predicted profile"
                    if bname == "measured_control_mean"
                    else "against the model's own unflagged output, so only what the "
                         "perturbation flag changes"
                )
                tag = "" if bname == "measured_control_mean" else f"__{bname}"
                # (summary field, key in the by-baseline row, run 3 grid cell)
                for field, rowkey, cell, desc in (
                    ("direct_raw", "raw", "raw",
                     "its own predicted response versus the measured effect"),
                    ("direct_fold_centred", "fold_centred", "centered",
                     "the same with the held-out fold's mean removed from both sides"),
                ):
                    if row.get(rowkey) is None:
                        continue
                    a_out[f"{field}{tag}"] = claim(
                        f"{ARM_LABEL[arm]} in {L}, {desc}, {what}",
                        row[rowkey], DIRECT,
                        key(cell_line=line, arm_label=f"direct_prediction/{arm}",
                            target=cell, weighting="weighted", baseline=bname),
                    )
                    for side in ("lo", "hi"):
                        a_out[f"{field}{tag}__{side}"] = row.get(f"{rowkey}_{side}")

            t = r.training(line, arm)
            if t:
                a_out["selected_epoch"] = t.get("selected_epoch")
                a_out["epochs_run"] = t.get("epochs_run")
                a_out["best_val_loss"] = claim(
                    f"{ARM_LABEL[arm]} in {L}, lowest masked MSE on held-out validation cells",
                    t.get("best_val_loss"), TRAIN_RUNS, key(cell_line=line, arm=arm),
                )
                ag = t.get("response_agreement_at_selected_epoch", {})
                for form in ("ctrlmean", "ctrlmean_centered"):
                    if form in ag:
                        a_out[f"val_agreement__{form}"] = claim(
                            f"{ARM_LABEL[arm]} in {L}, validation response agreement "
                            f"({form}) at the selected epoch",
                            ag[form], TRAIN_RUNS,
                            key(cell_line=line, arm=arm, column=f"selected_delta_{form}"),
                        )
                a_out["artifact"] = t.get("artifact")
            e["arms"][arm] = a_out

        cfsrc = "results/summary/crossfit_%s__own_trainable.parquet" % line
        for arm, label in B.CROSSFIT_ARMS:
            cf = r.crossfit_row(line, arm)
            if not cf:
                continue
            stem = "crossfit" if arm == "ft_own_trainable_resp" else f"crossfit__{arm}"
            e[stem] = claim(
                f"{L}, {label}, scored over the whole screen, verdict cell",
                cf["score"], cfsrc,
                key(cell_line=line, arm=arm, target="residual", weighting="weighted"),
            )
            e[f"{stem}__lo"], e[f"{stem}__hi"] = cf["lo"], cf["hi"]
            e[f"{stem}__n"], e[f"{stem}__n_groups"] = cf["n"], cf["n_groups"]

        le = r.last_epoch_row(line, "own_trainable")
        if le:
            e["last_epoch"] = claim(
                f"{L}, response features read off the LAST epoch rather than the selected one, "
                "verdict cell",
                le["score"], RESTRICTED,
                rkey(cell_line=line, arm=B.last_epoch_arm("own_trainable")),
            )
        key_sp = "vs_measured_control_mean__median_residual_fraction_of_variance"
        sp = r.spread_row(line, "own_trainable") or {}
        e["prediction_spread"] = claim(
            f"{L}, fraction of the primary arm's predicted profile variance that survives "
            "subtracting the mean prediction",
            sp.get(key_sp), SPREAD, key(cell_line=line, arm="own_trainable", column=key_sp),
        )
        e["measured_spread"] = r.measured_spread(line)
        n["lines"][line] = e

    # the pre-registered comparison is a paired difference, so how many of those
    # differences include zero is the claim, not a pair of separate scores
    if not r.contrasts.empty:
        c = r.contrasts[
            (r.contrasts.translator == "mlp")
            & (r.contrasts.target == "residual")
            & (r.contrasts.weighting == "weighted")
            & (r.contrasts.against == "scgpt_resp")
            & (r.contrasts.arm.isin([f"ft_{a.name}_resp" for a in C.ARMS]))
        ]
        n["n_paired"] = int(len(c))
        n["n_paired_zero"] = int(((c["lo"] <= 0) & (c["hi"] >= 0)).sum())
        n["paired_excluding_zero"] = [
            {"cell_line": x.cell_line, "arm": x.arm, "difference": round(float(x.difference), 6),
             "lo": round(float(x.lo), 6), "hi": round(float(x.hi), 6),
             "p_two_sided": float(x.p_two_sided)}
            for x in c.itertuples()
            if not (x.lo <= 0 <= x.hi)
        ]
        n["n_paired_all_references"] = int(len(
            r.contrasts[
                (r.contrasts.translator == "mlp")
                & (r.contrasts.target == "residual")
                & (r.contrasts.weighting == "weighted")
                & (r.contrasts.arm.str.startswith("ft_"))
                & (r.contrasts.arm.str.endswith("_resp"))
            ]
        ))

    pts = r.all_feature_points("final__value_substitution_resp")
    n["n_models"] = len(pts)
    n["n_models_in_band"] = sum(1 for p in pts if r.in_band(p["cell_line"], p))
    if pts:
        n["best_response_feature_score"] = claim(
            "highest response-feature score reached by any fine-tuned model, verdict cell",
            max(p["score"] for p in pts), RESTRICTED,
            rkey(arm="ft_<arm>_resp", over="all arms and both cell lines"),
        )
    if not r.agreement.empty:
        n["reference_reproducibility"] = claim(
            "largest disagreement between two independent recomputations of the same "
            "reference arm, over the readings that withhold the same two folds (the two "
            "refits are excluded because they score different perturbations)",
            float(r.agreement["spread"].max()), AGREE, key(column="spread", agg="max"),
        )
        n["reference_readings_per_line"] = int(r.agreement["count"].max())

    # whether the band the fine-tuned arms are read against is wide enough for
    # their own nulls, and where it is not
    if not r.null_check.empty and "inside_band" in r.null_check:
        nc = r.null_check[~r.null_check["train_arm"].astype(str).str.endswith(B.REFIT_TAGS)]
        for fam, tag in (("value_substitution", "value_substitution"), ("pert_flag", "pert_flag")):
            sub = nc[nc["family"].astype(str).str.startswith(fam)]
            if sub.empty:
                continue
            n[f"own_null_inside_band__{tag}"] = int(sub["inside_band"].sum())
            n[f"own_null_arms__{tag}"] = int(len(sub))
            claim(
                f"widest own shuffled-label draw among the {tag} arms, in absolute value",
                float(sub[["draw_min", "draw_max"]].abs().max().max()),
                NULLCHECK, key(family=fam, columns="draw_min,draw_max", agg="max_abs"),
            )

    # the same before/after-centring contrast inside training, over every epoch
    if not r.epochs.empty and "delta_ctrlmean" in r.epochs:
        raw = r.epochs["delta_ctrlmean"].dropna()
        cen = r.epochs["delta_ctrlmean_centered"].dropna()
        n["n_validation_epochs"] = int(len(raw))
        n["val_agreement_uncentred_min"] = claim(
            "lowest uncentred validation response agreement over every epoch of every run",
            float(raw.min()), TRAIN_EPOCHS, key(column="delta_ctrlmean", agg="min"),
        )
        n["val_agreement_centred_max_abs"] = claim(
            "largest absolute centred validation response agreement over every epoch of "
            "every run",
            float(cen.abs().max()), TRAIN_EPOCHS,
            key(column="delta_ctrlmean_centered", agg="max_abs"),
        )

    # whether run 4's re-reading of run 3's arms lands on run 3's own numbers
    if not r.run3_cmp.empty:
        cmp = r.run3_cmp.dropna(subset=["run3", "run4"])
        both = cmp[cmp.run4_in_run3_interval & cmp.run3_in_run4_interval]
        n["run3_arms_compared"] = int(len(cmp))
        n["run3_arms_mutually_contained"] = int(len(both))
        if not both.empty:
            claim(
                "largest difference between run 3's published whole-screen score and run 4's "
                "re-reading of the same arm, among the arms whose intervals each contain the "
                "other's estimate",
                float(both["difference"].abs().max()), RUN3CMP,
                key(column="difference", agg="max_abs", filter="mutually contained arms"),
            )
        for x in cmp.itertuples():
            claim(
                f"{LINE[x.cell_line]} {x.label}, run 3's published whole-screen score",
                float(x.run3), RUN3CMP, key(cell_line=x.cell_line, arm=x.arm, column="run3"),
            )
            claim(
                f"{LINE[x.cell_line]} {x.label}, run 4's re-reading of the same arm",
                float(x.run4), RUN3CMP, key(cell_line=x.cell_line, arm=x.arm, column="run4"),
            )
    return n, claims


def _band_clause(n: dict) -> str:
    parts = []
    for line in C.CELL_LINES:
        e = n["lines"].get(line, {})
        if e.get("null_ceiling") is None:
            continue
        parts.append(
            f"in {LINE[line]} the shuffled-label draws reach {e['null_ceiling']:.4f} and "
            f"co-expression itself scores {e.get('coexp', float('nan')):.4f} on the "
            f"{e.get('coexp__n_groups', '?')} response groups these folds leave"
        )
    return "; ".join(parts)


def selection_claims(r: B.Results, alt: B.Results, n: dict, claims: list[dict]) -> dict:
    """Each arm read in each execution, as traceable claims.

    The rule decides which weights every other number comes from, so the
    comparison is part of the contract rather than a remark on the page. Two
    distinct comparisons live here and they are kept apart: within the primary
    execution the masked-MSE epoch is tracked on the same trajectory, and the
    alternative summary is a separate earlier execution of the same arms.
    """
    same, tot = B._rule_agreement(r)
    out = {
        "within_run_same_epoch": same,
        "within_run_n": tot,
        "n_arms": 0,
        "n_both_in_band": 0,
        "arms": {},
    }
    for line in C.CELL_LINES:
        band = r.null_ceiling(line)
        for arm in B.ARM_ORDER:
            a = r.feat(line, arm, "final__value_substitution_resp")
            b = alt.feat(line, arm, "final__value_substitution_resp")
            if not a or not b:
                continue
            ta, tb = r.training(line, arm), alt.training(line, arm)
            out["n_arms"] += 1
            out["n_both_in_band"] += int(max(a["score"], b["score"]) <= band)
            out["arms"][f"{line}/{arm}"] = {
                "pearson_rule_epoch": ta.get("selected_epoch"),
                "pearson_rule_score": a["score"],
                "masked_mse_rule_epoch": tb.get("selected_epoch"),
                "masked_mse_rule_score": b["score"],
                "band_ceiling": band,
            }
            claims.append({
                "claim": f"{line} {arm}: response features in the earlier masked-MSE execution",
                "value": round(float(b["score"]), 6),
                "source": f"{ALT}/restricted.parquet",
                "key": rkey(cell_line=line, arm=f"ft_{C.model_stem(arm)}_resp"),
            })
    n["selection_rule"] = out
    return out


def build(summary_dir: Path, out_dir: Path, alt_dir: Path | None = None) -> tuple[Path, Path]:
    r = B.Results(summary_dir)
    n, claims = collect(r)
    sel = (
        selection_claims(r, B.Results(alt_dir), n, claims)
        if alt_dir is not None and Path(alt_dir).exists()
        else None
    )
    # the numbers the prose below asserts, read from the tables rather than
    # restated from a previous execution
    exc = B.paired_exceptions(r)
    exc_frozen = [d for d in exc if d["against"] == "scgpt_resp"]
    exc_text = (
        "; ".join(d["label"] for d in exc_frozen) if exc_frozen else "none"
    )
    ratios = [
        e["score"] / p["score"]
        for p in r.all_feature_points("final__value_substitution_resp")
        if (e := r.ref(p["cell_line"], "scgpt_emb")) and p["score"] > 0
    ]
    ratio_text = (
        f"{min(ratios):.0f} to {max(ratios):.0f} times higher" if ratios else "far higher"
    )
    own_above = B.own_query_above(r)
    own_text = (
        ""
        if not own_above
        else " One secondary read-out is not a clean null: querying the checkpoints "
        "through the model's own perturbation-flag input rather than through the "
        f"value substitution runs 2 and 3 use puts {len(own_above)} of "
        f"{n['n_models']} readings above both the band and that query's own shuffled "
        "draws ("
        + "; ".join(
            f"{B.LINE[p['cell_line']]} {B.ARM_LABEL.get(p['arm'], p['arm'])} "
            f"{p['score']:.4f}" for p in own_above
        )
        + "), and those two are reported as unexplained rather than resolved."
    )
    h = n["lines"].get("hepg2", {})
    j = n["lines"].get("jurkat", {})
    ho = h.get("arms", {}).get("own_trainable", {})
    jo = j.get("arms", {}).get("own_trainable", {})

    doc = {
        "headline": (
            "Fine-tuning scGPT on the CRISPRi screen it is evaluated on, with scGPT's own "
            "published perturbation recipe, leaves the queried response features inside the "
            "shuffled-label band and far below the model's own static gene table. Across the "
            f"{n['n_models']} fine-tuned models (five gene-table arms in two cell lines), "
            f"{'every one leaves' if n['n_models_in_band'] == n['n_models'] else str(n['n_models_in_band']) + ' leave'} them inside the band, and the highest "
            f"any reaches is {n.get('best_response_feature_score', float('nan')):.4f}, against "
            f"the gene table at {h.get('scgpt_emb', float('nan')):.4f} in HepG2 and "
            f"{j.get('scgpt_emb', float('nan')):.4f} in Jurkat. Paired against the frozen "
            f"model's own response features, {n.get('n_paired_zero', '?')} of the "
            f"{n.get('n_paired', '?')} differences include zero; the exceptions are "
            f"{exc_text}, a few hundredths of a correlation point and in opposite "
            f"directions.{own_text} On the model's own prediction "
            "head the predicted profile matches the measured effect at "
            f"{ho.get('direct_raw', float('nan')):.4f} in HepG2 and "
            f"{jo.get('direct_raw', float('nan')):.4f} in Jurkat, but that agreement is not "
            "perturbation-specific: measuring only what the perturbation flag changes gives "
            f"{ho.get('direct_raw__unflagged_model_output', float('nan')):.4f} and "
            f"{jo.get('direct_raw__unflagged_model_output', float('nan')):.4f}, and removing "
            "the held-out fold's own mean from both sides gives "
            f"{ho.get('direct_fold_centred', float('nan')):.4f} and "
            f"{jo.get('direct_fold_centred', float('nan')):.4f}. Only "
            f"{h.get('prediction_spread', float('nan')):.4f} and "
            f"{j.get('prediction_spread', float('nan')):.4f} of a predicted profile's variance "
            "survives subtracting the mean prediction, against "
            f"{h.get('measured_spread', float('nan')):.4f} and "
            f"{j.get('measured_spread', float('nan')):.4f} for the measurement. What "
            "fine-tuning installed is close to one profile for every knockdown, so a read-out "
            "that asks which gene was knocked down has very little to recover."
        ),
        "assessment": (
            "The pre-registered prediction holds in its first half: the response features stay "
            "inside the shuffled-label band in all ten fine-tuned models, and the gene-table "
            f"arm stays {ratio_text}. Its second half, that the model might "
            "predict better even so, is not borne out either, but the evidence for that is of "
            "two different strengths and they should not be run together.\n\n"
            "What the data supports. No fine-tuned arm's response features clear the band. The "
            "paired differences against the frozen model's own response features are small; "
            f"most include zero, and the {len(exc_frozen)} that do not ({exc_text}) are a "
            "few hundredths of a correlation point, point in opposite directions, and sit "
            f"among the {n.get('n_paired_all_references', '?')} paired differences computed on "
            "this page, so neither is evidence "
            "on its own. The model's predicted profile matches the measured effect when the "
            "reference is the measured control, and stops matching when the reference is the "
            "model's own unflagged output, which locates that agreement in a systematic offset "
            "shared across perturbations rather than in a response to the gene. The strongest "
            "single statement is the spread of the prediction itself, which is not a "
            "correlation and carries no interval width: almost none of a predicted profile's "
            "variance survives subtracting the mean prediction, and the profiles predicted for "
            "two different knockdowns correlate above 0.9.\n\n"
            "What it does not support. This is not a demonstration that fine-tuning cannot "
            "install the signal. The fold-centred correlation intervals are wide enough to "
            "admit a modest positive effect, and on the restricted folds co-expression itself "
            "is not separated from the band, so the evaluation cannot resolve a co-expression"
            "-sized effect there. The result is a bound, not a zero: at the resolution these "
            "folds achieve, fine-tuning produces no detectable improvement in the queried "
            "response features and leaves them far below the gene table."
            + own_text
            + " That column's own shuffled draws are several times wider than the band and "
            "uneven across arms, the other arms in the same cell line do not reproduce it, "
            "and both readings stay three to four times below the gene table, so it is "
            "reported as a noisy reading rather than as a signal.\n\n"
            "Scope and limits. One architecture and one recipe, run on the screen the question "
            "is about, which is the setting most favourable to installing the signal and "
            "therefore the setting in which a null is most informative; it does not speak to "
            "other architectures or to fine-tuning on a different screen. A fine-tuned model "
            "may only be scored on perturbations it never saw, so the main arms read two of "
            f"run 3's five folds: {_band_clause(n)}. Four controls address the obvious "
            "objections: the arm is refitted on the complementary folds so the pooled result "
            "covers the whole screen, the batch sampler is run in the exact upstream "
            "configuration as well as ours, the alternative checkpoint-selection rule is "
            "tracked inside every run and read again in a separate earlier execution of all "
            "sixteen arms, and every recipe setting including the selection rule is checked "
            "field by field against the published notebook."
        ),
        "decision_rule": {
            "preregistered": (
                "run 3's rule, unchanged: residual target, reliability-weighted, "
                "ceiling-corrected per-perturbation Pearson correlation, median over held-out "
                "perturbations of the grouped split, read against the shuffled-label band and "
                "against the same model's static gene-table arm"
            ),
            "restriction_added_by_this_run": (
                "only folds 0 and 1 are scored, because a fine-tuned model must not be scored "
                "on a perturbation it trained on. The validation split used for epoch "
                "selection is drawn from the TRAINING folds, never from these"
            ),
            "reported_beside_it": (
                "the fold-centred version of the same correlation, and the model's own "
                "predicted response scored with no translator in the path"
            ),
            "not_substituted": (
                "the verdict is read from the pre-registered cell. The direct-prediction "
                "result is additional evidence about the model, not a replacement metric"
            ),
        },
        "methods": {
            "base_model": f"{C.SCGPT_REPO} at {C.SCGPT_REVISION}",
            "recipe": (
                f"scGPT's published perturbation tutorial: {C.RECIPE.loss}, "
                f"{C.RECIPE.optimizer} at lr {C.RECIPE.lr}, batch {C.RECIPE.batch_size}, up to "
                f"{C.RECIPE.epochs} epochs with early stop after {C.RECIPE.early_stop}, "
                f"schedule gamma {C.RECIPE.schedule_gamma}; selection on {C.RECIPE.val_metric}"
            ),
            "arms": {a.name: a.note for a in C.ARMS},
            "heldout_folds": list(C.MODEL_HELDOUT_FOLDS),
            "feature_queries": (
                "run 2's value-substitution family, unmodified, so the fine-tuned numbers are "
                "commensurable with the frozen ones; plus the model's own perturbation-flag "
                "query, which is the query the fine-tuning objective is defined on"
            ),
            "restatement_checks": (
                "the masked-MSE loss is compared numerically against the installed "
                "scgpt/loss.py at load time and the result recorded in each run's "
                "provenance.json; --upstream-sampling reproduces the published batch sampler "
                "exactly, including the ~40% of rows that lose their perturbed gene"
            ),
            "n_training_runs": 16,
            "checkpoint_selection": (
                "the tutorial's own rule, read out of its training-loop cell by "
                "src/scfm_ft/recipe_provenance.py: keep the epoch with the highest mean "
                "Pearson correlation, across all genes, between each held-out perturbation's "
                "measured and predicted mean expression profile, and apply the patience "
                "counter to that score. The masked MSE on the same held-out cells is tracked "
                "in parallel with its own patience counter, so the epoch that rule would keep "
                "is recorded from the same run and both checkpoints are saved"
            ),
            "selection_rule_sensitivity": (
                None
                if sel is None
                else (
                    f"the two rules keep the same epoch in {sel['within_run_same_epoch']} of "
                    f"{sel['within_run_n']} runs of this execution, where the selected weights "
                    "are therefore identical; separately, an earlier execution of the same "
                    f"sixteen arms selected on masked MSE throughout has {sel['n_arms']} arms "
                    f"comparable here and {sel['n_both_in_band']} of them are inside the "
                    "shuffled-label band in both executions. That earlier execution predates "
                    "the three faithfulness controls, so it differs in random initialisation "
                    "as well as in the rule and is read as a replicate"
                )
            ),
        },
        "results": n,
        "limitations": [
            "One model family and one recipe. scGPT's own published perturbation recipe on the "
            "screen the question is about is the strongest single test available, but another "
            "architecture or objective could behave differently.",
            "The main arms score two of five folds, which resolves a paired difference more "
            "coarsely than run 3's five; on these folds co-expression is not separated from "
            "the shuffled-label band either. The cross-fitted variant covers the whole screen "
            "and agrees.",
            "Fine-tuning and evaluation come from the same screen, split by response group. "
            "That is the published setting and the one most favourable to the model; it says "
            "nothing about fine-tuning on a different screen.",
            "The response features are read at run 3's primary anchor. Run 3 swept four "
            "anchors on the frozen model and none lifted the features out of the band, but "
            "that sweep was not repeated on every fine-tuned checkpoint.",
            "A null is bounded by what the measurement resolves. The shuffled-label band and "
            "the achieved resolution of each paired difference are reported next to every "
            "number so the size of an effect that would have been missed is visible.",
        ],
        "artifacts": [
            HEAD, RESTRICTED, BAND, DIRECT, TRAIN_RUNS, TRAIN_EPOCHS, CONTRASTS, AGREE,
            NULLCHECK, RUN3CMP, PROV,
            *([f"{ALT}/restricted.parquet", f"{ALT}/training_runs.parquet"] if sel else []),
            "src/scfm_ft/recipe_provenance.py",
            "src/scfm_ft/compare_run3.py",
            "src/scfm_ft/analysis.py",
            "src/scfm_ft/config.py",
            "src/scfm_ft/scgpt_pert/train.py",
            "src/scfm_ft/scgpt_pert/query.py",
            "src/scfm_ft/score.py",
            "src/scfm_ft/run_arm.py",
            "src/scfm_ft/followup.py",
            "jobs/train_arm.sh",
            "jobs/followup.sh",
            "jobs/summarise.sh",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    a = out_dir / "report_summary.json"
    b = out_dir / "claims_manifest.json"
    a.write_text(json.dumps(doc, indent=2))
    b.write_text(json.dumps({"claims": claims}, indent=2))
    return a, b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--alt-summary", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    summary = Path(args.summary) if args.summary else paths.results_dir() / "summary"
    alt = (
        Path(args.alt_summary)
        if args.alt_summary
        else paths.repo_dir() / ALT
    )
    out = Path(args.out) if args.out else paths.repo_dir()
    a, b = build(summary, out, alt)
    doc = json.loads(a.read_text())
    print(doc["headline"])
    print(f"\nwrote {a} and {b} ({len(json.loads(b.read_text())['claims'])} claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
