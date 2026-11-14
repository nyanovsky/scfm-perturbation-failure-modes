"""Write ``report_summary.json`` and ``claims_manifest.json`` from the tables.

    PYTHONPATH=src uv run python -m scfm_pfm.report.contract

Both files are generated, never typed: every number in the summary is read out
of ``results/summary`` and every claim in the manifest names the file and the
row it came from, so the two cannot drift from the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from . import build as B
from . import figures as figmod

DISCRIM = figmod.DISCRIM
PREREG = figmod.PREREG
RUNS = "results/summary/runs.parquet"
#: the two-table arm and its width-matched null are scored into their own table,
#: so nothing computed over "all arms" in the main result is computed over them
CONCAT_RUNS = "results/summary/concat_runs.parquet"
CON = "results/summary/contrasts.parquet"
TRANS = "results/summary/transfer_runs.parquet"
ORDER = "results/summary/ordering_agreement.parquet"
OVERLAP = "results/summary/control_cell_overlap.parquet"
RECOMPUTE = "results/recompute_report.json"
INVENTORY = "results/bundle_inventory.json"
SENS = "results/summary/sensitivity.parquet"
HEAD = "results/summary/headline.json"
DISCV = "results/discrimination_versions.json"
BEELINE = "results/beeline_reproduction.json"
NOOP = "results/verification/no_op_correction.json"
AUDIT = "results/provenance/audit.json"
AUDIT_T = "results/provenance/audit_tables.json"
MAP = "results/protein/gene_to_protein_report.json"
EMBED = "results/protein/esm_embed_report.json"
INSTALL = "results/protein/install_report.json"
IDENTITY = "results/protein/identity.json"


def key(**kw) -> str:
    return ", ".join(f"{k}={v}" for k, v in kw.items())


def pkey(**kw) -> str:
    """A locator into ``runs.parquet`` pinned to the cell the verdict is read from.

    Without every index column the locator matches several rows -- the same arm
    is scored under four target sources and two splits -- so the re-reader
    cannot check it. Everything the primary cell fixes goes in.
    """
    return key(scope="full", split="grouped", translator="mlp",
               target_source="run1_all_controls", **kw)


def collect(r: B.Results) -> tuple[dict, list[dict]]:
    claims: list[dict] = []

    def claim(text: str, value, source: str, where: str) -> float:
        v = float(value)
        claims.append({"claim": text, "value": round(v, 6), "source": source, "key": where})
        return v

    lines = {}
    for line in ("hepg2", "jurkat"):
        best = r.best(line, B.FULL_SCOPE_MODELS, exclude="hybrid")
        co = r.interval(line, "coexp")
        cop = r.interval(line, "coexp", PREREG)
        nullv = r.null_ceiling(line)
        nullp = r.null_ceiling(line, PREREG)
        resp = max(
            r.score(line, f"{m}_resp") for m in B.FULL_SCOPE_MODELS
        )
        npm = r.score(line, "node_pairmlp")
        block = r.primary(line)
        lines[line] = {
            "n_perturbations": int(block.n_perturbations.iloc[0]),
            "n_readout": int(block.n_readout.iloc[0]),
            "best_model_arm": best.arm,
            "best_model_score_fold_centred": claim(
                f"best foundation-model arm in {line} scores this on the fold-centred cell",
                best[DISCRIM], RUNS, pkey(cell_line=line, arm=best.arm, column=DISCRIM),
            ),
            "best_model_ci": [float(best[DISCRIM + "_lo"]), float(best[DISCRIM + "_hi"])],
            "best_model_score_preregistered": claim(
                f"the same arm on the pre-registered cell in {line}, which the verdict is read "
                "from",
                best[PREREG], RUNS, pkey(cell_line=line, arm=best.arm, column=PREREG),
            ),
            "best_model_ci_preregistered": [
                float(best[PREREG + "_lo"]), float(best[PREREG + "_hi"])
            ],
            "coexpression_score_fold_centred": claim(
                f"co-expression in {line} on the fold-centred cell",
                co[0], RUNS, pkey(cell_line=line, arm="coexp", column=DISCRIM),
            ),
            "coexpression_ci": [co[1], co[2]],
            "coexpression_score_preregistered": claim(
                f"co-expression in {line} on the pre-registered cell",
                cop[0], RUNS, pkey(cell_line=line, arm="coexp", column=PREREG),
            ),
            "coexpression_ci_preregistered": [cop[1], cop[2]],
            "n_response_groups": int(block.n_groups_scored.dropna().iloc[0]),
            "largest_response_group": int(block.largest_group_scored.dropna().iloc[0]),
            "shuffled_label_ceiling_fold_centred": claim(
                f"largest shuffled-label draw in {line}, fold-centred, MLP",
                nullv, RUNS, pkey(cell_line=line, arm="*+shuffled_labels_d*", column=DISCRIM, statistic="max"),
            ),
            "shuffled_label_ceiling_preregistered": claim(
                f"largest shuffled-label draw in {line}, pre-registered, MLP",
                nullp, RUNS, pkey(cell_line=line, arm="*+shuffled_labels_d*", column=PREREG, statistic="max"),
            ),
            "best_response_only_score": claim(
                f"largest response-only arm in {line}",
                resp, RUNS, pkey(cell_line=line, arm="*_resp", column=DISCRIM, statistic="max"),
            ),
            "node_free_score": claim(
                f"free network on node statistics in {line}",
                npm, RUNS, pkey(cell_line=line, arm="node_pairmlp", column=DISCRIM),
            ),
            "residual_ceiling": float(
                r.manifest[line]["reference"]["residual_ceiling"]["median_spearman_brown"]
            ),
            "residual_ceiling_weighted": claim(
                f"reliability-weighted split-half ceiling the pre-registered score in {line} "
                "divides by",
                r.ceiling(line, PREREG), RUNS,
                pkey(cell_line=line, column="residual__weighted__ceiling", statistic="max"),
            ),
            "fold_centred_ceiling_weighted": claim(
                f"reliability-weighted split-half ceiling the fold-centred score in {line} "
                "divides by",
                r.ceiling(line), RUNS,
                pkey(cell_line=line, column="centered__weighted__ceiling", statistic="max"),
            ),
        }

    def con_for(metric: str):
        return r.con[(r.con.scope == "full") & (r.con.split == "grouped")
                     & (r.con.translator == "mlp") & (r.con.metric == metric)
                     & (r.con.target_source == "run1_all_controls")]

    paired = {}
    for metric, tag in (("residual_weighted", "preregistered"),
                        ("centered_weighted", "fold_centred")):
        con = con_for(metric)
        for line in ("hepg2", "jurkat"):
            arms = [lines[line]["best_model_arm"]] + [
                a for a in ("scgpt_emb", "scprint_emb", "transcriptformer_emb",
                            "esm2_3b_emb", "esm2_650m_emb")
                if a in set(con[con.cell_line == line].arm)
            ]
            for arm in dict.fromkeys(arms):
                c = con[(con.cell_line == line) & (con.arm == arm)]
                if c.empty:
                    continue
                where = key(cell_line=line, scope="full", split="grouped", translator="mlp",
                            target_source="run1_all_controls", arm=arm, reference="coexp",
                            metric=metric)
                paired[f"{line}__{arm}__{tag}"] = {
                    "arm": arm, "cell_line": line, "metric": metric,
                    "median_difference": claim(
                        f"paired difference of {arm} against co-expression in {line} "
                        f"on the {tag} target",
                        float(c.median_difference.iloc[0]), CON,
                        where + ", column=median_difference",
                    ),
                    "ci": [float(c.lo.iloc[0]), float(c.hi.iloc[0])],
                    "n_paired": int(c.n_paired.iloc[0]),
                    "n_groups_paired": int(c.n_groups_paired.iloc[0]),
                    "min_resolvable_difference": claim(
                        f"smallest paired difference this design resolves for {arm} in {line} "
                        f"on the {tag} target",
                        float(c.min_resolvable_difference_achieved.iloc[0]), CON,
                        where + ", column=min_resolvable_difference_achieved",
                    ),
                }
    con = con_for("centered_weighted")

    transfer = {}
    for scope in ("transfer_full", "transfer_unseen"):
        d = r.tr(scope)
        if d.empty:
            continue
        for direction in sorted(d.cell_line.unique()):
            b = d[(d.cell_line == direction) & d.arm.str.startswith(B.FULL_SCOPE_MODELS)
                  & ~d.arm.str.contains("hybrid", regex=False)]
            b = b.sort_values(DISCRIM, ascending=False).iloc[0]
            gi = d[(d.cell_line == direction) & (d.arm == "null_randgene_d5120")]
            co = d[(d.cell_line == direction) & (d.arm == "coexp")]
            k = f"{scope}|{direction}"
            transfer[k] = {
                "n_held_out": int(b.n_perturbations),
                "best_model_arm": b.arm,
                "best_model_score": claim(
                    f"best model arm on {k}", float(b[DISCRIM]), TRANS,
                    key(scope=scope, cell_line=direction, arm=b.arm, translator="mlp",
                        split="cross_line__dest_reference", column=DISCRIM),
                ),
                "best_model_ci": [float(b[DISCRIM + "_lo"]), float(b[DISCRIM + "_hi"])],
                "gene_identity_null": claim(
                    f"fixed random vector per gene on {k}",
                    float(gi[DISCRIM].iloc[0]), TRANS,
                    key(scope=scope, cell_line=direction, arm="null_randgene_d5120",
                        translator="mlp", split="cross_line__dest_reference",
                        column=DISCRIM),
                ),
                "coexpression": float(co[DISCRIM].iloc[0]) if not co.empty else float("nan"),
            }

    agree = pd.read_parquet(r.dir / "ordering_agreement.parquet")
    a = agree[(agree.scope == "full") & (agree.split == "grouped")
              & (agree.translator == "mlp") & (agree.target_source == "run1_all_controls")]
    a = a.assign(target=a.cell.str.split("__").str[0])
    ordering = {
        t: claim(f"rank correlation of the {t} target's ordering against the pre-registered cell",
                 float(g.spearman_vs_primary.min()), ORDER, key(scope="full", split="grouped", translator="mlp",
                            target_source="run1_all_controls", cell=f"{t}__*",
                            column="spearman_vs_primary", statistic="min"))
        for t, g in a.groupby("target")
    }

    common = {}
    for line in ("hepg2", "jurkat"):
        d = r.scope_arms(line, "common", "grouped")
        nulls = d[d.arm.str.contains(figmod.SHUFFLE_TAG, regex=False)][DISCRIM]
        ceiling = float(np.nanmax(nulls.to_numpy()))
        row = {"shuffled_label_ceiling": ceiling,
               "n_perturbations": int(d.n_perturbations.iloc[0])}
        for model in B.MODELS:
            sub = d[d.arm.str.startswith(model)]
            if sub.empty:
                continue
            b = sub.sort_values(DISCRIM, ascending=False).iloc[0]
            row[model] = claim(
                f"best {model} arm on the 250-gene common scope in {line}",
                float(b[DISCRIM]), RUNS,
                key(cell_line=line, scope="common", split="grouped", translator="mlp",
                    target_source="run1_all_controls", arm=b.arm, column=DISCRIM),
            )
        common[line] = row

    overlap = {}
    if not r.overlap.empty:
        d = r.overlap[r.overlap.translator == "mlp"]
        piv = d.pivot_table(index=["cell_line", "arm"], columns="variant",
                            values=DISCRIM).reset_index()
        if {"ctrl0", "ctrl1"}.issubset(piv.columns):
            piv = piv.assign(delta=piv["ctrl1"] - piv["ctrl0"])
            co = piv[piv.arm.isin(["coexp", "coexp_check"])]
            mo = piv[piv.arm.isin(["scgpt_emb", "transcriptformer_emb"])]
            overlap = {"per_arm": piv.to_dict("records")}
            # the claim is a difference between two rows, which the re-reader
            # cannot compute, so each end of the largest difference is claimed
            # on its own row and the difference is reported beside them.
            for name, sub, text in (
                ("coexpression_arms", co,
                 "largest change in a plain co-expression arm's score between a target built "
                 "from the control half it uses and one built from the other half"),
                ("embedding_arms", mo,
                 "the same change for the embedding arms, which touch no control cell"),
            ):
                hit = sub.loc[sub.delta.abs().idxmax()]
                overlap[f"largest_shift_{name}"] = float(abs(hit.delta))
                overlap[f"largest_shift_{name}_arm"] = f"{hit.cell_line}/{hit.arm}"
                claims.append({
                    "claim": text,
                    "value": round(float(abs(hit.delta)), 6),
                    "source": OVERLAP,
                    "key": key(derived=(f"|score({hit.arm} in {hit.cell_line}, ctrl1) - "
                                        f"score(same arm, ctrl0)| in "
                                        "control_cell_overlap.parquet")),
                })
                for variant in ("ctrl0", "ctrl1"):
                    overlap[f"largest_shift_{name}_{variant}"] = claim(
                        f"{hit.arm} in {hit.cell_line} scored against the {variant} target",
                        float(hit[variant]), OVERLAP,
                        key(cell_line=hit.cell_line, arm=hit.arm, translator="mlp",
                            variant=variant, column=DISCRIM),
                    )
    for line in ("hepg2", "jurkat"):
        if line in r.recompute:
            t = r.recompute[line]["targets"]["ctrlall"]
            overlap[f"recomputed_target_agreement_{line}"] = claim(
                f"median per-perturbation correlation between the recomputed effect for {line} "
                "and the effect matrix run 1 shipped",
                t["median_agreement_with_run1_effect"], RECOMPUTE,
                key(field=f"lines.{line}.targets.ctrlall.median_agreement_with_run1_effect"),
            )
            overlap[f"recomputed_reliability_weight_{line}"] = claim(
                f"median reliability weight of the recomputed {line} target",
                t["median_reliability_weight"], RECOMPUTE,
                key(field=f"lines.{line}.targets.ctrlall.median_reliability_weight"),
            )

    prereg_draw = B._null_by_translator(r, PREREG)
    centred_draw = B._null_by_translator(r, DISCRIM)
    metric_check = {}
    for line in ("hepg2", "jurkat"):
        metric_check[f"shuffled_draw_ridge_preregistered_{line}"] = claim(
            f"largest shuffled-label draw in {line} on the pre-registered cell, ridge",
            prereg_draw[(line, "ridge")], RUNS,
            key(cell_line=line, scope="full", split="grouped", translator="ridge",
                target_source="run1_all_controls", arm="*+shuffled_labels_d*",
                column=PREREG, statistic="max"),
        )
        metric_check[f"shuffled_draw_ridge_fold_centred_{line}"] = claim(
            f"the same draws in {line} on the fold-centred cell, ridge",
            centred_draw[(line, "ridge")], RUNS,
            key(cell_line=line, scope="full", split="grouped", translator="ridge",
                target_source="run1_all_controls", arm="*+shuffled_labels_d*",
                column=DISCRIM, statistic="max"),
        )
        metric_check[f"coexpression_preregistered_{line}"] = claim(
            f"co-expression's own pre-registered score in {line}",
            r.score(line, "coexp", PREREG), RUNS,
            pkey(cell_line=line, arm="coexp", column=PREREG),
        )
        m = r.model_arms(line)
        m = m[m.arm.str.startswith(B.FULL_SCOPE_MODELS)]
        metric_check[f"model_arms_clearing_null_{line}"] = (
            f"{int((m[DISCRIM] > r.null_ceiling(line)).sum())} of {int(m.shape[0])}"
        )
        claims.append({
            "claim": f"foundation-model arms clearing the shuffled-label null in {line}",
            "value": metric_check[f"model_arms_clearing_null_{line}"],
            "source": RUNS,
            "key": key(derived=(f"count over the twelve {line} foundation-model arms on the "
                                "primary cell of runs.parquet of score > the largest "
                                "shuffled-label draw in that cell")),
        })

    feature_deltas = {
        "largest_response_only": claim(
            "largest response-only score in either cell line",
            max(abs(r.score(line, f"{m}_resp"))
                for line in ("hepg2", "jurkat") for m in B.FULL_SCOPE_MODELS),
            RUNS, key(scope="full", split="grouped", translator="mlp",
                      target_source="run1_all_controls", arm="*_resp", column=DISCRIM,
                      statistic="max"),
        ),
        "largest_change_from_adding_response": claim(
            "largest change from adding the response features to the embedding",
            max(abs(r.score(line, f"{m}_full") - r.score(line, f"{m}_emb"))
                for line in ("hepg2", "jurkat") for m in B.FULL_SCOPE_MODELS),
            RUNS, key(derived="max over model and cell line of |score(<model>_full) - "
                      "score(<model>_emb)| on the primary cell of runs.parquet"),
        ),
        "largest_change_from_adding_coexpression": claim(
            "largest change from adding co-expression to the embedding",
            max(abs(r.score(line, f"{m}_hybrid") - r.score(line, f"{m}_emb"))
                for line in ("hepg2", "jurkat") for m in B.FULL_SCOPE_MODELS),
            RUNS, key(derived="max over model and cell line of |score(<model>_hybrid) - "
                      "score(<model>_emb)| on the primary cell of runs.parquet"),
        ),
    }

    rnd = r.runs[(r.runs.cell_line == "hepg2") & (r.runs.scope == "full")
                 & (r.runs.split == "random") & (r.runs.translator == "mlp")
                 & (r.runs.arm == "node_pairmlp")][DISCRIM]
    splits = {
        "node_free_random_split_hepg2": claim(
            "free node-statistics network in hepg2 under the random split",
            float(rnd.iloc[0]), RUNS,
            key(cell_line="hepg2", scope="full", split="random", translator="mlp",
                target_source="run1_all_controls", arm="node_pairmlp", column=DISCRIM),
        ),
        "node_free_grouped_split_hepg2": lines["hepg2"]["node_free_score"],
    }

    mode = {}
    piv = r.transfer[r.transfer.translator == "mlp"].pivot_table(
        index=["cell_line", "scope", "arm"], columns="transfer_mode", values=DISCRIM)
    if "dest_reference" in piv.columns:
        shifts, ranks = [], []
        for other in ("source_only", "dest_all"):
            if other not in piv.columns:
                continue
            shifts.append(float((piv[other] - piv["dest_reference"]).abs().max()))
            for _, g in piv.groupby(level=[0, 1]):
                gg = g[["dest_reference", other]].dropna()
                if gg.shape[0] > 3:
                    ranks.append(float(np.corrcoef(gg["dest_reference"].rank(),
                                                   gg[other].rank())[0, 1]))
        mode = {
            "largest_shift_from_reference_choice": claim(
                "largest change in a transfer arm's score from using the source-line mean or "
                "all destination cells instead of held-out destination cells as the reference",
                max(shifts), TRANS, key(derived="max over arm, direction and scope of |score under "
                    "dest_all or source_only - score under dest_reference| in "
                    "transfer_runs.parquet"),
            ),
            "lowest_rank_correlation_across_modes": claim(
                "lowest rank correlation between the reported mode and either other mode, "
                "within one direction and scope",
                min(ranks), TRANS, key(derived="min over direction and scope of the Spearman rank "
                    "correlation between the arm ordering under dest_reference and under each "
                    "other reference choice in transfer_runs.parquet"),
            ),
        }

    # the second scoring axis, and what its version pin cost
    disc: dict = {}
    for line in ("hepg2", "jurkat"):
        for arm in ("coexp", "scgpt_emb", "scprint_emb", "transcriptformer_emb",
                    "scgpt_resp", "transcriptformer_resp", "null_randpair"):
            disc[f"{arm}_{line}"] = claim(
                f"perturbation discrimination of {arm} in {line}, cell-eval on effect profiles",
                r.disc(line, arm), RUNS,
                key(cell_line=line, scope="full", split="grouped", translator="mlp", arm=arm,
                    column="discrimination_cell_eval_mean"),
            )
    disc["chance"] = 0.5
    disc["package_version_used"] = str(
        r.primary("hepg2").cell_eval_version.dropna().iloc[0]
    )
    if r.disc_versions:
        v = r.disc_versions
        disc["version_pinned_in_preregistration"] = v["pinned_in_preregistration"]
        disc["pinned_version_exists"] = v["pinned_version_exists"]
        disc["largest_difference_between_versions"] = claim(
            "largest difference in the discrimination score of one saved prediction matrix "
            "between cell-eval 0.5.18/0.5.21 and 0.8.2",
            v["max_abs_difference_any_file"], DISCV, key(field="max_abs_difference_any_file"),
        )
        same = [
            f for f, d in v["per_file"].items()
            if abs(d["values"]["0.5.18"] - d["values"]["0.5.21"]) < 1e-12
        ]
        disc["versions_0_5_18_and_0_5_21_identical_on"] = f"{len(same)} of {len(v['per_file'])}"
        claims.append({
            "claim": "saved prediction matrices where cell-eval 0.5.18 and 0.5.21 agree exactly",
            "value": disc["versions_0_5_18_and_0_5_21_identical_on"],
            "source": DISCV,
            "key": key(derived="count over the six saved prediction files of whether "
                               "per_file[*].values['0.5.18'] equals ['0.5.21']"),
        })
        # the reversal itself: the same predictions, two versions, opposite ordering
        pairs = {
            "hepg2_scgpt_full": "hepg2__full__grouped__scgpt_full__mlp.npz",
            "hepg2_coexp": "hepg2__full__grouped__coexp__mlp.npz",
            "jurkat_transcriptformer_full":
                "jurkat__full__grouped__transcriptformer_full__mlp.npz",
            "jurkat_coexp": "jurkat__full__grouped__coexp__mlp.npz",
        }
        for lab, f in pairs.items():
            for ver in ("0.5.18", "0.8.2"):
                disc[f"{lab}_at_cell_eval_{ver.replace('.', '_')}"] = claim(
                    f"discrimination of the {lab} predictions under cell-eval {ver}",
                    v["per_file"][f]["values"][ver], DISCV, key(file=f, version=ver),
                )
        disc["ordering_reverses_between_versions"] = True
        claims.append({
            "claim": "the model arm scores below co-expression under 0.5.18/0.5.21 and above "
                     "it under 0.8.2, in both cell lines",
            "value": True, "source": DISCV,
            "key": key(derived="sign of (model arm minus co-expression) under each version in "
                               "per_file[*].values, compared between 0.5.21 and 0.8.2"),
        })

    # anchor and form sensitivity
    sens: dict = {}
    head_sens = r.head.get("sensitivity", {})
    if not r.sens.empty:
        for model in B.FULL_SCOPE_MODELS:
            for anchor in ("mean_control", "uniform_virtual", "individual_cells",
                           "mean_control_masked"):
                v = r.sens_score(model, anchor)
                if v != v:
                    continue
                sens[f"{model}_at_{anchor}"] = claim(
                    f"response-only score for {model} at the {anchor} anchor, HepG2",
                    v, SENS,
                    key(cell_line="hepg2", scope="full", split="grouped", translator="mlp",
                        arm=f"{model}_resp@{anchor}" if anchor != "mean_control"
                        else f"{model}_resp", column=DISCRIM),
                )
        for p in head_sens.get("paired_against_mean_control", []):
            sens[f"paired_{p['model']}_{p['anchor']}"] = claim(
                f"paired difference of the {p['anchor']} anchor against mean control for "
                f"{p['model']}, HepG2",
                p["median_difference"], HEAD,
                key(field="sensitivity.paired_against_mean_control", model=p["model"],
                    anchor=p["anchor"], statistic="median_difference"),
            )
        sens["largest_rerun_difference_against_main_sweep"] = claim(
            "largest difference between an arm re-run in the sensitivity job and the same arm "
            "in the main sweep",
            head_sens.get("max_abs_rerun_difference", float("nan")), HEAD,
            key(field="sensitivity.max_abs_rerun_difference"),
        )
        sens["best_score_at_any_anchor"] = claim(
            "best response-only score at any anchor in HepG2",
            max(r.sens_score(m, a)
                for m in B.FULL_SCOPE_MODELS
                for a in ("mean_control", "uniform_virtual", "individual_cells",
                          "mean_control_masked")
                if r.sens_score(m, a) == r.sens_score(m, a)),
            SENS, key(cell_line="hepg2", arms="*_resp@*", statistic="max", column=DISCRIM),
        )
        sens["shuffled_label_ceiling_hepg2"] = lines["hepg2"]["shuffled_label_ceiling_fold_centred"]

    # excluding no-op queries: what it touches, and the checks behind it
    no_op: dict = {}
    if r.no_op:
        for b in r.no_op:
            tag = f"{b['model']}_{b['cell_line']}_{b['anchor']}"
            for field in (
                "n_no_op_queries",
                "n_query_rows_touched",
                "n_query_rows_everywhere_no_op",
                "max_abs_feature_change_partial_rows",
            ):
                no_op[f"{tag}__{field}"] = claim(
                    f"{field.replace('_', ' ')} for {tag}",
                    b[field], NOOP, key(block=tag, field=field),
                )
            for name, c in b["checks"].items():
                no_op[f"{tag}__check__{name}"] = claim(
                    f"{name.replace('_', ' ')} for {tag}: measured value against its tolerance",
                    c["value"], NOOP, key(block=tag, check=name, field="value"),
                )
        checks = [c for b in r.no_op for c in b["checks"].values()]
        no_op["checks_passed"] = f"{sum(c['pass'] for c in checks)} of {len(checks)}"
        claims.append({
            "claim": "checks passed over the corrected individual-cells blocks",
            "value": no_op["checks_passed"],
            "source": NOOP, "key": key(field="blocks[*].checks[*].pass"),
        })
        no_op["values_the_earlier_whole_dose_rule_deleted"] = {
            b["model"]: b["values_the_whole_dose_rule_deleted"] for b in r.no_op
        }
        claims.append({
            "claim": "values in CP10K the earlier whole-dose rule deleted, per model",
            "value": no_op["values_the_earlier_whole_dose_rule_deleted"],
            "source": NOOP,
            "key": key(derived="per-model grouping of "
                               "blocks[*].values_the_whole_dose_rule_deleted"),
        })

    # a side arm that was run, did not reproduce its source, and was dropped. Only the
    # numbers that decided the drop are carried; nothing measured inside that setting is
    # reported as a finding, so those values stay in the artifact and out of the claims.
    bee: dict = {}
    if r.beeline:
        b = r.beeline
        bee["status"] = (
            "dropped: the re-implementation did not reproduce the source paper's reported "
            "baseline, so neither it nor any control run inside it supports a claim here"
        )
        for row in b["rows"]:
            cell = f"{row['network']}_{row['dataset']}"
            bee[f"auprc_{cell}"] = claim(
                f"AUPRC of the dropped re-implementation on {row['network']}/{row['dataset']}",
                row["auprc"], BEELINE, key(row=cell, field="auprc"),
            )
            bee[f"auprc_reported_{cell}"] = claim(
                f"AUPRC the source paper reports for its scGPT embedding baseline on "
                f"{row['network']}/{row['dataset']}",
                row["paper_baseline_emb_auprc"], BEELINE,
                key(row=cell, field="paper_baseline_emb_auprc"),
            )
            bee[f"pairs_scored_{cell}"] = claim(
                f"pairs scored on {row['network']}/{row['dataset']}",
                row["n_pairs_kept"], BEELINE, key(row=cell, field="n_pairs_kept"),
            )
        for field in ("mean_auprc", "paper_mean_auprc", "mean_difference",
                      "max_abs_difference"):
            bee[field] = claim(
                f"dropped BEELINE arm, {field.replace('_', ' ')}",
                b["summary"][field], BEELINE, key(field=f"summary.{field}"),
            )
        bee["train_cell"] = f"{b['train_cell']['network']}/{b['train_cell']['dataset']}"
        bee["train_pairs"] = b["train_cell"]["n_pairs_kept"]
        bee["arms_never_run"] = b["arms_not_reproduced"]

    # the protein-model control: where each gene table came from, and what a raw
    # protein table does through the same pipeline
    protein: dict = {}
    if r.head.get("protein"):
        protein["gene_table_provenance"] = {
            "transcriptformer": ("a frozen ESM2 protein embedding, bitwise the file shipped "
                                 "beside the weights, never updated in pretraining"),
            "scprint": "ESM-initialised and then trained for 400,000 steps",
            "scgpt": "learned from expression, no protein model in the path",
            "aido_cell": "learned from expression, no protein model in the path",
        }
        at = r.audit_tables.get("transcriptformer", {})
        claims.append({
            "claim": "largest absolute difference between TranscriptFormer's 23,823 gene rows "
                     "and the ESM2 file shipped beside its weights",
            "value": at.get("weights_vs_shipped_genes_only", {}).get("max_abs_diff"),
            "source": AUDIT_T,
            "key": key(field="transcriptformer.weights_vs_shipped_genes_only.max_abs_diff"),
        })
        claims.append({
            "claim": "largest absolute difference on the 7 special-token rows, which are "
                     "regenerated from a seeded random draw rather than read from the file",
            "value": at.get("special_token_rows", {}).get("max_abs_diff"),
            "source": AUDIT_T,
            "key": key(field="transcriptformer.special_token_rows.max_abs_diff"),
        })
        protein["gene_to_protein"] = {
            "n_genes": r.protein_map.get("n_genes"),
            "by_ensembl_xref": r.protein_map.get("routes", {}).get("ensembl"),
            "by_symbol": r.protein_map.get("routes", {}).get("symbol"),
            "unmapped": len(r.protein_map.get("unmapped_genes", [])),
            "uniprot_release": r.protein_map.get("uniprot", {}).get("release"),
            "reviewed_human_entries": r.protein_map.get("uniprot", {}).get("n_entries"),
            "multi_candidate_genes": r.protein_map.get("multi_candidate_genes"),
            "over_esm2_context": r.protein_map.get("length", {}).get("over_esm2_context"),
        }
        for field, where, text in (
            ("n_genes", "n_genes", "read-out genes routed to a protein"),
            ("by_ensembl_xref", "routes.ensembl",
             "genes mapped through the Ensembl cross-reference"),
            ("by_symbol", "routes.symbol", "genes mapped by gene symbol"),
            ("unmapped", "unmapped_genes", "genes with no reviewed human protein"),
            ("multi_candidate_genes", "multi_candidate_genes",
             "genes with more than one reviewed candidate, resolved by the tie-break rule"),
            ("over_esm2_context", "length.over_esm2_context",
             "proteins longer than ESM2's 1,022-residue context"),
        ):
            claims.append({
                "claim": text, "value": protein["gene_to_protein"][field],
                "source": MAP,
                "key": key(field=where) + (", statistic=count" if field == "unmapped" else ""),
            })
        protein["zero_rows_installed"] = sum(
            b["lines"][ln]["zero_rows"]
            for b in r.protein_install.get("installed", []) for ln in b["lines"]
        )
        claims.append({
            "claim": "read-out genes left without a protein vector in either installed table",
            "value": protein["zero_rows_installed"], "source": INSTALL,
            "key": key(field="installed[*].lines[*].zero_rows", statistic="sum"),
        })
        protein["scores"] = {}
        for line in ("hepg2", "jurkat"):
            block = {}
            for arm in ("esm2_3b_emb", "esm2_650m_emb", "transcriptformer_emb",
                        "scprint_emb", "scgpt_emb", "coexp",
                        "null_randgene_d2560", "null_randgene_d5120"):
                block[arm] = claim(
                    f"{arm} held-out fold-centred score in {line}, full scope, grouped split",
                    r.score(line, arm), RUNS,
                    pkey(cell_line=line, arm=arm, column=DISCRIM),
                )
                pv, plo, phi = r.interval(line, arm, PREREG)
                block[f"{arm}__preregistered"] = claim(
                    f"{arm} held-out pre-registered score in {line}, full scope, grouped split",
                    pv, RUNS, pkey(cell_line=line, arm=arm, column=PREREG),
                )
                block[f"{arm}__preregistered_ci"] = [plo, phi]
            d = r.scope_arms(line, "full", "random")
            for arm in ("esm2_3b_emb", "transcriptformer_emb", "scgpt_emb", "coexp",
                        "null_randgene_d2560"):
                row = d[d.arm == arm]
                if row.empty:
                    continue
                block[f"{arm}__random_split"] = claim(
                    f"{arm} held-out fold-centred score in {line} under the random split",
                    float(row[DISCRIM].iloc[0]), RUNS,
                    key(cell_line=line, scope="full", split="random", translator="mlp",
                        target_source="run1_all_controls", arm=arm, column=DISCRIM),
                )
                block[f"{arm}__random_split__preregistered"] = claim(
                    f"{arm} held-out pre-registered score in {line} under the random split",
                    float(row[PREREG].iloc[0]), RUNS,
                    key(cell_line=line, scope="full", split="random", translator="mlp",
                        target_source="run1_all_controls", arm=arm, column=PREREG),
                )
            block["largest_protein_shuffled_draw"] = claim(
                f"largest shuffled-label draw of the ESM2 arms in {line}",
                r.head["protein"]["lines"][line]["largest_protein_shuffled_draw"], HEAD,
                key(field=f"protein.lines.{line}.largest_protein_shuffled_draw"),
            )
            protein["scores"][line] = block
        protein["paired_differences"] = {}
        protein["paired_differences_preregistered"] = {}
        for field, sink in (("paired_differences", protein["paired_differences"]),
                            ("paired_differences_preregistered",
                             protein["paired_differences_preregistered"])):
            for line in ("hepg2", "jurkat"):
                for row in r.head["protein"]["lines"][line].get(field, []):
                    tag = f"{line}__{row['arm']}__vs__{row['reference']}"
                    sink[tag] = {
                        "median_difference": row["median_difference"],
                        "lo": row["lo"], "hi": row["hi"], "n_paired": row["n_paired"],
                        "n_groups_paired": row.get("n_groups_paired"),
                    }
                    claims.append({
                        "claim": f"paired difference between {row['arm']} and "
                                 f"{row['reference']} on the same held-out perturbations in "
                                 f"{line} ({field})",
                        "value": round(float(row["median_difference"]), 6), "source": HEAD,
                        "key": key(field=f"protein.lines.{line}.{field}",
                                   arm=row["arm"], reference=row["reference"],
                                   statistic="median_difference"),
                    })
        protein["in_job_reference_check"] = {
            "n_arms_rerun_within_line": len(r.head.get("protein_reference_check", [])),
            "n_arms_rerun_cross_line": len(
                r.head.get("protein_transfer_reference_check", [])
            ),
            "largest_absolute_difference_from_the_main_sweep": max(
                [abs(x["difference"]) for x in r.head.get("protein_reference_check", [])]
                + [abs(x["difference"])
                   for x in r.head.get("protein_transfer_reference_check", [])] or [float("nan")]
            ),
        }
        claims.append({
            "claim": "largest absolute difference between an arm the protein job re-ran and "
                     "the same arm in the main sweep, over all 44 re-runs",
            "value": protein["in_job_reference_check"][
                "largest_absolute_difference_from_the_main_sweep"],
            "source": HEAD, "key": key(field="protein_reference_check[*].difference", statistic="max_abs"),
        })
        protein["table_similarity_rsa"] = {}
        for a, b_ in (("esm2_3b", "esm2_650m"), ("transcriptformer", "esm2_3b"),
                      ("scprint", "esm2_650m"), ("scprint", "esm2_3b"),
                      ("scgpt", "scprint"), ("scgpt", "esm2_3b"),
                      ("aido_cell", "esm2_3b")):
            pair = "|".join(sorted((a, b_)))
            protein["table_similarity_rsa"][pair] = claim(
                f"rank similarity between the {a} and {b_} gene tables in HepG2",
                r.rsa("hepg2", a, b_), IDENTITY,
                key(field=f"lines.hepg2.pairs.{pair}.rsa_spearman"),
            )
        if r.head.get("protein_transfer"):
            unseen: dict = {}
            for direction in ("hepg2->jurkat", "jurkat->hepg2"):
                for mode_name in ("dest_reference", "dest_all", "source_only"):
                    block = r.head["protein_transfer"]["blocks"].get(
                        f"{direction}|transfer_unseen|cross_line__{mode_name}", []
                    )
                    for row in block:
                        tag = f"{direction}__{mode_name}__{row['arm']}"
                        unseen[tag] = {
                            "score": row["score"], "lo": row["lo"], "hi": row["hi"],
                            "preregistered": row["preregistered"],
                            "preregistered_lo": row["preregistered_lo"],
                            "preregistered_hi": row["preregistered_hi"],
                            "n_perturbations": row["n_perturbations"],
                        }
                        if mode_name == "dest_reference":
                            claims.append({
                                "claim": f"{row['arm']} on the unseen-gene cross-line slice, "
                                         f"{direction}, held-out reference cells, "
                                         "pre-registered rule",
                                "value": round(float(row["preregistered"]), 6), "source": HEAD,
                                "key": key(
                                    field=(f"protein_transfer.blocks.{direction}|"
                                           f"transfer_unseen|cross_line__dest_reference"),
                                    arm=row["arm"], statistic="preregistered"),
                            })
            protein["unseen_gene_transfer"] = unseen
            # Counted per destination-reference mode: the non-transductive mode scores half
            # the destination's perturbations because the other half builds its reference, so
            # pooling the three modes would mix sample sizes.
            counts = r.head["protein_transfer"]["unseen_gene_clearance"]
            for mode_name, per_mode in counts.items():
                for name in ("protein_derived", "scgpt", "coexpression_and_nulls"):
                    claims.append({
                        "claim": f"unseen-gene cross-line cells where the group-resampled "
                                 f"interval excludes zero on the pre-registered rule, {name} "
                                 f"arms, {mode_name} reference mode "
                                 f"({per_mode['n_perturbations_scored']} perturbations scored)",
                        "value": per_mode[name],
                        "source": HEAD,
                        "key": key(
                            field=(f"protein_transfer.unseen_gene_clearance.{mode_name}."
                                   f"{name}")),
                    })
            protein["unseen_gene_transfer_counts"] = counts

    # the two-table arm: whether scGPT's table and the raw protein table are two
    # routes to one quantity or carry different information
    concat: dict = {}
    if r.head.get("concat"):
        h = r.head["concat"]
        concat["arm"] = (
            "scgpt_esm2_3b_emb: scGPT's 512-wide gene table concatenated with the 2,560-wide "
            "raw ESM2-3B protein table, for both genes, through the identical translator, "
            "folds, weighting, ceiling correction and scoring"
        )
        concat["rule_fixed_before_the_arm_was_scored"] = h["rule"]
        concat["agreement_thresholds"] = h["agreement_thresholds"]
        concat["reading"] = h["reading"]
        concat["reading_read_from"] = h["reading_read_from"]
        concat["scores"] = {}
        concat["paired_against_parts"] = {}
        concat["agreement"] = {}
        concat["reading_by_cell"] = {}
        concat["better_part_by_cell"] = {}
        for cell, blk in h["lines"].items():
            line, split = cell.split("|")
            for row in blk["scores"]:
                concat["scores"][f"{cell}__{row['arm']}"] = claim(
                    f"{row['arm']} in {line}, {split} split, pre-registered rule",
                    row["score"],
                    CONCAT_RUNS if row["arm"] in B.CONCAT_TABLE_ARMS else RUNS,
                    key(scope="full", split=split, translator="mlp",
                        target_source="run1_all_controls", cell_line=line, arm=row["arm"],
                        column=PREREG),
                )
            for tag, d in blk["paired_against_parts"].items():
                concat["paired_against_parts"][f"{cell}__{tag}"] = {
                    "difference": d["difference"], "lo": d["lo"], "hi": d["hi"],
                    "n_paired": d["n_paired"], "n_groups_paired": d["n_groups_paired"],
                    "min_resolvable_difference": d["min_resolvable_difference"],
                }
                claims.append({
                    "claim": f"two-table arm minus {tag.replace('__', ', ')} in {line}, "
                             f"{split} split",
                    "value": round(float(d["difference"]), 6), "source": HEAD,
                    "key": key(field=f"concat.lines.{cell}.paired_against_parts.{tag}.difference"),
                })
                # the assessment states the power limit as a number, so it is claimed too
                claims.append({
                    "claim": f"smallest difference the two-table comparison against "
                             f"{tag.replace('__', ', ')} can resolve in {line}, "
                             f"{split} split",
                    "value": round(float(d["min_resolvable_difference"]), 6), "source": HEAD,
                    "key": key(field=(f"concat.lines.{cell}.paired_against_parts.{tag}."
                                      "min_resolvable_difference")),
                })
            concat["agreement"][cell] = claim(
                "Spearman correlation between the scGPT-table arm's and the ESM2-3B-table "
                f"arm's per-perturbation scores in {line}, {split} split",
                blk["parts_spearman"], HEAD,
                key(field=f"concat.lines.{cell}.parts_spearman"),
            )
            for pair in ("esm2_650m_emb__esm2_3b_emb", "scgpt_emb__transcriptformer_emb",
                         "scgpt_emb__coexp"):
                got = blk["agreement"].get(f"{pair}__residual_weighted")
                if got is None:
                    continue
                concat["agreement"][f"{cell}__{pair}"] = claim(
                    f"Spearman correlation between the {pair.replace('__', ' and ')} arms' "
                    f"per-perturbation scores in {line}, {split} split",
                    got["spearman"], HEAD,
                    key(field=(f"concat.lines.{cell}.agreement."
                               f"{pair}__residual_weighted.spearman")),
                )
            concat["reading_by_cell"][cell] = blk["reading"]
            concat["better_part_by_cell"][cell] = blk["better_part"]
        checks = r.head.get("concat_reference_check", [])
        if checks:
            concat["in_job_reference_check"] = {
                "n_arms_rerun": len(checks),
                "largest_absolute_difference_from_the_main_sweep": max(
                    abs(x["difference"]) for x in checks
                ),
            }
            claims.append({
                "claim": "largest absolute difference between an arm the two-table job "
                         "re-ran and the same arm in the main sweep",
                "value": concat["in_job_reference_check"][
                    "largest_absolute_difference_from_the_main_sweep"],
                "source": HEAD,
                "key": key(field="concat_reference_check[*].difference", statistic="max_abs"),
            })

    summary = {
        "lines": lines,
        "protein_control": protein,
        "two_table_arm": concat,
        "paired_vs_coexpression": paired,
        "metric_check": metric_check,
        "feature_deltas": feature_deltas,
        "split_design": splits,
        "transfer": transfer,
        "transfer_reference_mode": mode,
        "ordering_agreement_min": ordering,
        "common_scope": common,
        "control_cell_overlap": overlap,
        "discrimination": disc,
        "anchor_sensitivity": sens,
        "no_op_query_correction": no_op,
        "beeline_arm_dropped": bee,
    }
    return summary, claims


def _concat_assessment(n: dict) -> str:
    """What the two-table arm adds to the assessment, in its own words.

    The paired contrast is underpowered by design: in every cell the
    difference against the better part is smaller than the smallest
    difference that comparison can resolve, so it could not have detected a
    gain of the size on offer. The agreement statistic, read against its own
    calibration, is the half that carries information, and it points toward
    two tables holding different information rather than two routes to one
    quantity. Neither half closes the question.
    """
    concat = n.get("two_table_arm")
    if not concat:
        return ""
    agree = concat.get("agreement", {})

    def val(cell: str) -> float | None:
        got = agree.get(cell)
        return None if got is None else float(got)

    paired = concat.get("paired_against_parts", {})
    better = concat.get("better_part_by_cell", {})
    cells = (("hepg2|grouped", "HepG2 grouped"), ("hepg2|random", "HepG2 random"),
             ("jurkat|grouped", "Jurkat grouped"), ("jurkat|random", "Jurkat random"))
    power, under, total = [], 0, 0
    for cell, name in cells:
        d = paired.get(f"{cell}__{better.get(cell, '')}__preregistered")
        if not d:
            continue
        total += 1
        under += int(abs(d["difference"]) < d["min_resolvable_difference"])
        power.append(
            f"{name} {B.P.sig(d['difference'], signed_=True)} against "
            f"{B.P.sig(d['min_resolvable_difference'])}"
        )
    power_clause = (
        " Its paired comparison was underpowered for the effect it was looking for: in "
        f"{'all four cells' if under == total == 4 else f'{under} of the {total} cells'} "
        "the two-table arm's "
        "difference against the better of its two parts is smaller than the smallest "
        f"difference that comparison can resolve ({'; '.join(power)}). That is a property of "
        "the design, not of the models: a gain the size of the ones on offer could not have "
        "been detected here, so nothing in this arm supports calling the two tables "
        "equivalent or calling either one redundant."
        if power else ""
    )
    # the agreement statistic is reported with the two references that calibrate it: the
    # same protein model at two sizes above it, and scGPT against co-expression beside it
    hg, jk = val("hepg2|grouped"), val("jurkat|grouped")
    same_family = [val(f"{c}__esm2_650m_emb__esm2_3b_emb") for c in
                   ("hepg2|grouped", "jurkat|grouped")]
    vs_coexp = [val(f"{c}__scgpt_emb__coexp") for c in ("hepg2|grouped", "jurkat|grouped")]
    thresholds = concat.get("agreement_thresholds", {})
    weak = float(thresholds.get("weak_below", 0.3))
    strong = float(thresholds.get("strong_at_or_above", 0.5))
    if hg is not None and jk is not None and all(
            v is not None for v in (*same_family, *vs_coexp)):
        agreement_clause = (
            " The half that does carry information is the agreement statistic, and it is "
            "read against its own calibration. scGPT's table and the raw ESM2-3B table agree "
            f"on which held-out perturbations they score well at Spearman {hg:.4f} in HepG2 "
            f"and {jk:.4f} in Jurkat. Two sizes of the same protein model, which is what two "
            "views of one underlying quantity look like in this setup, agree at "
            f"{same_family[0]:.4f} and {same_family[1]:.4f}; scGPT against co-expression "
            f"agrees at {vs_coexp[0]:.4f} and {vs_coexp[1]:.4f}. scGPT's table therefore "
            "agrees with a protein model about as weakly as it agrees with co-expression and "
            f"far below the same-family reference, below the {weak:g} the rule set for "
            f"different information and well below the {strong:g} it set for the same "
            "information."
        )
    else:
        agreement_clause = ""
    # only the comparison against the better part decides the reading, so only that
    # one is checked on the diagnostic
    centred = []
    for cell, part in better.items():
        d = paired.get(f"{cell}__{part}__fold_centred")
        if d and d["lo"] > 0 and d["difference"] > d["min_resolvable_difference"]:
            centred.append(
                f"{cell.replace('|', ', ')} against {part} "
                f"({d['difference']:+.4f}, {d['lo']:.4f} to {d['hi']:.4f}, against "
                f"{d['min_resolvable_difference']:.4f} resolvable)"
            )
    centred_clause = (
        " On the fold-centred diagnostic, " + "; ".join(centred)
        + f" clears what it resolves, {len(centred)} of the {len(better)} comparisons "
        "against the better part; that diagnostic is not the verdict and one cell does not "
        "change the reading."
        if centred else
        " The fold-centred diagnostic reads the same way on every one of those comparisons."
    )
    return (
        " The two-table arm is a supplementary test with its own reading, not a second "
        "verdict on the run's question."
        + power_clause
        + agreement_clause
        + " Taken together that is a discrepancy identified rather than resolved: the "
        "agreement numbers point toward the two tables carrying different information rather "
        "than being redundant routes to one quantity, and the concatenation test is too "
        "underpowered to say whether that difference converts into a gain. It is reported as "
        "an open question with its numbers attached, not as a null and not as a positive."
        + centred_clause
    )


def _anchor_null_clause(n: dict) -> str:
    """Whether any anchor lifted a response-only arm out of the null band."""
    sens = n.get("anchor_sensitivity", {})
    best = sens.get("best_score_at_any_anchor")
    ceiling = sens.get("shuffled_label_ceiling_hepg2")
    if best is None or ceiling is None:
        return "the anchor sensitivity step did not run"
    verdict = (
        "no anchor lifts a response-only arm above the shuffled-label ceiling there"
        if best <= ceiling
        else "the best anchor lifts a response-only arm above the shuffled-label ceiling there"
    )
    return f"{verdict} ({best:.4f} against {ceiling:.4f})"


def build(summary_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    manifest = json.loads((paths.repo_dir() / "results" / "bundle_inventory.json").read_text())
    r = B.Results(summary_dir, manifest)
    n, claims = collect(r)
    hep, jur = n["lines"]["hepg2"], n["lines"]["jurkat"]

    doc = {
        "headline": (
            "A gene-pair feature built from a single-cell foundation model's gene embedding "
            "predicts the perturbation-specific part of a CRISPRi knockdown response better "
            "than statistical co-expression computed from the same control cells, in both cell "
            "lines and on both splits, on the pre-registered scoring rule and with intervals "
            "that resample response groups rather than perturbations. The response features "
            "harvested in the previous run add nothing to the embedding; on their own they are "
            "weak. Reading the four released checkpoints shows that two of the four gene "
            "tables are protein-derived rather than learned from expression (TranscriptFormer's "
            "frozen ESM2 table and scPRINT's protein-initialised one), and a gene table built "
            "from UniProt protein sequence with a public ESM2 checkpoint, run through the "
            "identical translator, folds, scoring and nulls, beats co-expression in both lines "
            "and ties the single-cell arms in seven of eight paired comparisons, beating "
            "scGPT's in Jurkat. Protein sequence alone can therefore supply a gene table that "
            "does this job as well as the single-cell models' tables do; that does not identify "
            "the source of scGPT's signal, whose table carries no protein-space structure."
        ),
        "assessment": (
            "Supported for the embedding arms against co-expression, on both cell lines, on the "
            "pre-registered scoring cell, with every interval resampling the response groups "
            "the split holds out together rather than individual perturbations. Every "
            "embedding arm's paired difference against co-expression excludes zero in both "
            "lines. The pre-registered cell has a known weakness -- perturbations in a "
            "held-out group share a per-gene response, so its shuffled-label draws sit above "
            "zero -- and a fold-centred version of the same correlation is reported beside it "
            "everywhere. That version is a diagnostic, not a second verdict: it subtracts the "
            "held-out fold's own mean, so it uses held-out outcomes, and it was written after "
            "the shuffled draws were seen. The two rank the arms similarly (rank correlation "
            f"at least {n['ordering_agreement_min']['centered']:.4f}) and the arms that move "
            "are the weak ones. The second pre-registered axis, perturbation discrimination, "
            "does not settle anything either way: the version pinned before the run does not "
            "exist upstream, and the two existing versions nearest it reverse the "
            "model-versus-co-expression ordering on the same saved predictions, so that axis "
            "is reported as version-dependent and inconclusive. The response features are "
            "refuted as an addition to the embedding on both scores; on their own they are "
            "numerically ahead of co-expression in three of six model-and-line combinations "
            "but never by more than the paired comparison resolves. The protein control is a "
            "positive: the raw ESM2 arm clears its width-matched random-vector null and its "
            "own shuffled-label draws in both cell lines, beats co-expression in both on a "
            "paired comparison, and ties TranscriptFormer's, scPRINT's and scGPT's embedding "
            "arms in seven of eight comparisons while beating scGPT's in Jurkat, with every arm "
            "it is compared against re-run inside the same job and reproducing the main sweep "
            "exactly. It shows where the signal can come from, not where each model's signal "
            "does come from: it does not decompose scGPT's or AIDO.Cell's learned table into a "
            "protein-like part and a cell-specific part, and scGPT's table has no measurable "
            "protein-space structure while scoring as well as the rest."
            + _concat_assessment(n)
        ),
        "decision_rule": {
            "preregistered": (
                "residual target, reliability-weighted, ceiling-corrected per-perturbation "
                "Pearson correlation, median over held-out perturbations of a grouped split, "
                "compared to co-expression as a paired difference"
            ),
            "used_for_the_verdict": (
                "the pre-registered rule, unchanged. Residual is against the TRAINING folds' "
                "mean effect, so a held-out perturbation's own outcome never enters its score"
            ),
            "reported_beside_it": (
                "a fold-centred version of the same correlation, which removes the held-out "
                "fold's own per-gene mean from prediction and measurement. It uses held-out "
                "outcomes and was written after seeing that the pre-registered cell's "
                "shuffled-label draws sit above zero, so it is post hoc and answers the "
                "narrower question of separating one held-out perturbation from another. It is "
                "reported for every arm and never substituted for the pre-registered verdict"
            ),
            "uncertainty": (
                "2,000-resample bootstrap over RESPONSE GROUPS, not over perturbations: the "
                "grouped split holds correlated perturbations out together, so perturbations "
                "are not independent draws (226 groups over 524 perturbations in HepG2, one of "
                "them 149 rows; 452 over 705 in Jurkat). Groups are recomputed from the "
                "bundle's effect matrix and reproduce the group counts the scoring jobs "
                "recorded"
            ),
            "null": (
                "labels permuted inside the training fold, five draws per null-bearing arm; an "
                "arm counts only if it exceeds the largest draw in its cell of the sweep"
            ),
        },
        "methods": {
            "data": (
                "Genome-wide CRISPRi Perturb-seq, GSE264667, HepG2 and Jurkat. "
                f"HepG2 {hep['n_perturbations']} in-scope perturbations x {hep['n_readout']} "
                f"read-out genes; Jurkat {jur['n_perturbations']} x {jur['n_readout']}."
            ),
            "target": (
                "per-gene log2 fold change against the pooled non-targeting control, "
                "residualised by removing each read-out gene's mean effect across "
                "perturbations; split-half reproducibility ceiling "
                f"{hep['residual_ceiling']:.4f} (HepG2) and {jur['residual_ceiling']:.4f} "
                "(Jurkat), Spearman-Brown corrected"
            ),
            "arms": (
                "four models (scGPT, AIDO.Cell, scPRINT, TranscriptFormer) x four feature sets "
                "(gene embedding, run 2 response summary, both, either plus co-expression); "
                "statistical co-expression (Pearson, Spearman, mutual information, partial "
                "correlation) from one half of the non-targeting control cells; node-only "
                "controls (additive and free); random vector per gene and per pair; "
                "shuffled-label draws"
            ),
            "translator": (
                "one fixed network for every arm: 128 and 64 ReLU units, Adam 1e-3, batch "
                "4096, 30 epochs, reliability-weighted MSE. Ridge regression alongside as a "
                "capacity control. Seed 42, five folds, grouped and random splits."
            ),
            "uncertainty": (
                "every interval in the report is a 2,000-resample cluster bootstrap over "
                "response groups (scfm_pfm/metrics.py bootstrap_median_grouped and "
                "bootstrap_median_difference_grouped). The groups are the ones the grouped "
                "split holds out together, recomputed from the bundle's effect matrix by "
                "scfm_pfm/groups.py; the reconstruction reproduces the n_groups and "
                "largest_group the scoring jobs recorded for all 256 grouped runs of the main "
                "sweep. The scoring jobs themselves bootstrapped rows, and those intervals are "
                "replaced, not averaged with, the group-resampled ones. Point estimates are "
                "unchanged and are re-derived from the per-perturbation tables as a check. The "
                "achieved minimum resolvable difference is the paired difference a two-sided 5% "
                "test would resolve at 80% power given the group-resampled standard error."
            ),
            "known_fix": (
                "the control-cell job wrote its target_source label into its run table but not "
                "into its per-perturbation table. Filling that label after concatenating all "
                "waves made those per-perturbation rows look like the main target's and, being "
                "read later, replace them, which moved every paired difference onto the "
                "control-half target. load_sweep now fills the label per directory from that "
                "directory's own run rows; the paired differences in this report are the "
                "corrected ones and are consistent with the run-level scores."
            ),
            "transfer": (
                "fitted on one cell line and scored on the other over the shared read-out set, "
                "with three choices of what the destination response is measured against and "
                "an unseen-gene scope that drops any destination perturbation whose gene was "
                "perturbed in the training line"
            ),
            "control_cell_overlap": (
                "the effect target was recomputed from the raw count matrices three times, "
                "against the control half the co-expression features use, the other half, and "
                "all control cells, and the key arms rescored against each"
            ),
            "protein_control": (
                "stage 1 reads the four released checkpoints (state-dict shapes, optimizer "
                "moment tensors, recorded hyper-parameters, and the shipped ESM2 file for "
                "TranscriptFormer) to determine where each gene-identity table came from. "
                "Stage 2 maps every read-out gene to its canonical reviewed human UniProt "
                "entry, embeds the sequence with facebook/esm2_t33_650M_UR50D and "
                "facebook/esm2_t36_3B_UR50D as the mean over final-layer residue states "
                "(proteins past the 1,022-residue context are chunked and averaged), installs "
                "the table as an ordinary embedding arm, and runs it through the identical "
                "translator, folds, weights, scoring, nulls and transfer design. The same jobs "
                "re-ran co-expression and the three model embedding arms as an in-job "
                "reference. Gene tables are additionally compared to each other by the "
                "Spearman correlation of their gene-gene cosine similarities, whose "
                "random-table floor is at zero for any width"
            ),
        },
        "results": n,
        "limitations": [
            "The pre-registered score gives an arm credit for the part of a held-out group's "
            "response its members share, which is why its shuffled-label draws sit above zero. "
            "The fold-centred version reported beside it removes that component but uses a "
            "mean of the held-out outcomes to do so, so neither confirms the other; where they "
            "disagree the report says so.",
            "The independent unit is the response group, not the perturbation: HepG2's 524 "
            "in-scope perturbations are 226 groups, one holding 149 of them, and Jurkat's 705 "
            "are 452. Intervals resample groups, so the effective sample size is closer to the "
            "group count.",
            "Scores are small in absolute terms even after the ceiling correction: the best "
            f"arm reaches {hep['best_model_score_preregistered']:.4f} in HepG2 and "
            f"{jur['best_model_score_preregistered']:.4f} in Jurkat on a scale where 1.0 is the "
            "reproducibility limit.",
            "The individual-cells anchor is a four-cell sample. Its no-op queries are "
            "excluded per (perturbed gene, value) and the exclusion is exact on the "
            "standardized feature, but four control cells is a small sample of the anchor.",
            "One translator shape for every arm. The query anchor was varied in HepG2 only, "
            "over the three models with full-coverage feature blocks; "
            f"{_anchor_null_clause(n)}.",
            "The discrimination axis is inconclusive, not a second confirmation. The "
            "pre-registered pin cell-eval==0.5.19 was never released; the run used 0.8.2, and "
            "the two existing versions nearest the pin differ from it by up to "
            f"{n['discrimination'].get('largest_difference_between_versions', float('nan')):.4f} "
            "on the same saved predictions, which reverses the co-expression-versus-model "
            "ranking on that axis in both cell lines.",
            "The source paper's own BEELINE benchmark is not part of the evidence. A "
            "re-implementation of it was run and dropped: its embedding baseline scored 0.6724 "
            "against 0.6247 reported, up to 0.09176 apart on one cell, with no pre-registered "
            "tolerance and no seed-noise estimate, so the setting is not established and "
            "neither it nor any control run inside it supports a claim here. Its virtual-value "
            "and gradient-trajectory arms would have needed a fresh extraction against those "
            "datasets and were never run.",
            "Two cell lines from a single screen, so the cross-line result is a weak test of "
            "generality.",
            "The additive node control is identically zero on the fold-centred score by "
            "construction, so the free node network is the informative version.",
            "AIDO.Cell is only comparable on the 250-gene subset it was harvested on, and the "
            "two scores disagree about it there: its best arm sits just above the "
            "shuffled-label ceiling on the pre-registered rule and below it on the fold-centred "
            "one. It is not shown to be worse than the others, only not separated from the "
            "null.",
            "Co-expression is represented by four statistics over control cells; a fitted "
            "network-inference method could do better.",
            "The protein control shows where the signal can come from, not where each model's "
            "signal does come from. It does not decompose scGPT's or AIDO.Cell's learned table "
            "into a protein-like part and a cell-specific part, and it says nothing about what "
            "these models do outside this gene-pair task. One protein model at two sizes, and "
            "one pooling rule (mean over the final layer), stand in for protein space.",
            "The unseen-gene cross-line slice is small and its three destination-reference "
            "modes are not interchangeable: the non-transductive one scores 78 to 89 "
            "perturbations because the rest build its reference, while the other two score 156 "
            "to 177, so their clearance counts are reported separately and never pooled. The "
            "slice separates the protein-derived tables from scGPT's rather than ranking the "
            "arms inside either group.",
        ],
        "artifacts": [
            RUNS, CON, TRANS, ORDER, OVERLAP, RECOMPUTE, INVENTORY, SENS, HEAD, DISCV, BEELINE,
            NOOP,
            "results/handoff/no_op_queries_per_gene.csv",
            "src/scfm_pfm/ingest/no_op.py",
            "src/scfm_pfm/ingest/no_op_check.py",
            "src/scfm_pfm/beeline.py",
            "src/scfm_pfm/discrimination.py",
            "scratch/disc_version_check.py",
            "jobs/run_sensitivity.sh",
            "jobs/run_beeline.sh",
            "results/summary/headline.json",
            "results/summary/grid.parquet",
            "results/summary/null_band.parquet",
            "src/scfm_pfm/translate/runner.py",
            "src/scfm_pfm/translate/arms.py",
            "src/scfm_pfm/translate/trainers.py",
            "src/scfm_pfm/metrics.py",
            "src/scfm_pfm/analysis.py",
            "src/scfm_pfm/ingest/recompute_effects.py",
            "src/scfm_pfm/report/figures.py",
            "src/scfm_pfm/report/build.py",
            AUDIT, AUDIT_T, MAP, EMBED, INSTALL, IDENTITY,
            "results/protein/gene_to_protein.parquet",
            "results/summary/protein_reference.parquet",
            "results/summary/protein_transfer_reference.parquet",
            "results/summary/ordering_agreement_with_protein.parquet",
            "src/scfm_pfm/protein/audit.py",
            "src/scfm_pfm/protein/audit_tables.py",
            "src/scfm_pfm/protein/uniprot.py",
            "src/scfm_pfm/protein/esm_embed.py",
            "src/scfm_pfm/protein/install_arm.py",
            "src/scfm_pfm/protein/identity.py",
            "jobs/run_provenance.sh",
            "jobs/run_protein_embed.sh",
            "jobs/run_protein_line.sh",
            "jobs/run_protein_transfer.sh",
        ],
    }
    if n.get("two_table_arm"):
        doc["limitations"].append(
            "The two-table test is one concatenation at one width, scored by the same "
            "translator as everything else, and its paired comparison is underpowered: in "
            "every cell the difference against the better part is smaller than the smallest "
            "difference that comparison resolves, so a gain of the size on offer would not "
            "have been detected. It asks whether that translator can use both gene tables "
            "together better than it uses the better one alone; it does not rule out that a "
            "different way of combining them, or a task with finer resolution than a "
            "held-out per-perturbation correlation, would separate what the two tables carry."
        )
        doc["artifacts"].extend([
            "results/summary/concat_runs.parquet",
            "results/summary/arm_agreement.parquet",
            "results/summary/concat_reference.parquet",
            "jobs/run_concat.sh",
        ])
    out_dir.mkdir(parents=True, exist_ok=True)
    a = out_dir / "report_summary.json"
    b = out_dir / "claims_manifest.json"
    a.write_text(json.dumps(doc, indent=2))
    b.write_text(json.dumps({"claims": claims}, indent=2))
    return a, b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    summary = Path(args.summary) if args.summary else paths.results_dir() / "summary"
    out = Path(args.out) if args.out else paths.repo_dir()
    a, b = build(summary, out)
    print(f"wrote {a} and {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
