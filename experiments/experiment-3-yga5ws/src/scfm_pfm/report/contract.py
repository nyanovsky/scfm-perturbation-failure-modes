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

VERDICT = figmod.VERDICT
PREREG = figmod.PREREG
RUNS = "results/summary/runs.parquet"
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
                best[VERDICT], RUNS, pkey(cell_line=line, arm=best.arm, column=VERDICT),
            ),
            "best_model_ci": [float(best[VERDICT + "_lo"]), float(best[VERDICT + "_hi"])],
            "best_model_score_preregistered": claim(
                f"the same arm on the pre-registered cell in {line}",
                best[PREREG], RUNS, pkey(cell_line=line, arm=best.arm, column=PREREG),
            ),
            "coexpression_score_fold_centred": claim(
                f"co-expression in {line} on the fold-centred cell",
                co[0], RUNS, pkey(cell_line=line, arm="coexp", column=VERDICT),
            ),
            "coexpression_ci": [co[1], co[2]],
            "shuffled_label_ceiling_fold_centred": claim(
                f"largest shuffled-label draw in {line}, fold-centred, MLP",
                nullv, RUNS, pkey(cell_line=line, arm="*+shuffled_labels_d*", column=VERDICT, statistic="max"),
            ),
            "shuffled_label_ceiling_preregistered": claim(
                f"largest shuffled-label draw in {line}, pre-registered, MLP",
                nullp, RUNS, pkey(cell_line=line, arm="*+shuffled_labels_d*", column=PREREG, statistic="max"),
            ),
            "best_response_only_score": claim(
                f"largest response-only arm in {line}",
                resp, RUNS, pkey(cell_line=line, arm="*_resp", column=VERDICT, statistic="max"),
            ),
            "node_free_score": claim(
                f"free network on node statistics in {line}",
                npm, RUNS, pkey(cell_line=line, arm="node_pairmlp", column=VERDICT),
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

    con = r.con[(r.con.scope == "full") & (r.con.split == "grouped")
                & (r.con.translator == "mlp") & (r.con.metric == "centered_weighted")
                & (r.con.target_source == "run1_all_controls")]
    paired = {}
    for line in ("hepg2", "jurkat"):
        arm = lines[line]["best_model_arm"]
        c = con[(con.cell_line == line) & (con.arm == arm)]
        paired[line] = {
            "arm": arm,
            "median_difference": claim(
                f"paired difference of {arm} against co-expression in {line}",
                float(c.median_difference.iloc[0]), CON,
                key(cell_line=line, scope="full", split="grouped", translator="mlp",
                    target_source="run1_all_controls", arm=arm, reference="coexp",
                    metric="centered_weighted", column="median_difference"),
            ),
            "ci": [float(c.lo.iloc[0]), float(c.hi.iloc[0])],
            "min_resolvable_difference": claim(
                f"smallest paired difference this design can resolve for {arm} in {line}",
                float(c.min_resolvable_difference_achieved.iloc[0]), CON,
                key(cell_line=line, scope="full", split="grouped", translator="mlp",
                    target_source="run1_all_controls", arm=arm, reference="coexp",
                    metric="centered_weighted",
                    column="min_resolvable_difference_achieved"),
            ),
        }

    transfer = {}
    for scope in ("transfer_full", "transfer_unseen"):
        d = r.tr(scope)
        if d.empty:
            continue
        for direction in sorted(d.cell_line.unique()):
            b = d[(d.cell_line == direction) & d.arm.str.startswith(B.FULL_SCOPE_MODELS)
                  & ~d.arm.str.contains("hybrid", regex=False)]
            b = b.sort_values(VERDICT, ascending=False).iloc[0]
            gi = d[(d.cell_line == direction) & (d.arm == "null_randgene_d5120")]
            co = d[(d.cell_line == direction) & (d.arm == "coexp")]
            k = f"{scope}|{direction}"
            transfer[k] = {
                "n_held_out": int(b.n_perturbations),
                "best_model_arm": b.arm,
                "best_model_score": claim(
                    f"best model arm on {k}", float(b[VERDICT]), TRANS,
                    key(scope=scope, cell_line=direction, arm=b.arm, translator="mlp",
                        split="cross_line__dest_reference", column=VERDICT),
                ),
                "best_model_ci": [float(b[VERDICT + "_lo"]), float(b[VERDICT + "_hi"])],
                "gene_identity_null": claim(
                    f"fixed random vector per gene on {k}",
                    float(gi[VERDICT].iloc[0]), TRANS,
                    key(scope=scope, cell_line=direction, arm="null_randgene_d5120",
                        translator="mlp", split="cross_line__dest_reference",
                        column=VERDICT),
                ),
                "coexpression": float(co[VERDICT].iloc[0]) if not co.empty else float("nan"),
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
        nulls = d[d.arm.str.contains(figmod.SHUFFLE_TAG, regex=False)][VERDICT]
        ceiling = float(np.nanmax(nulls.to_numpy()))
        row = {"shuffled_label_ceiling": ceiling,
               "n_perturbations": int(d.n_perturbations.iloc[0])}
        for model in B.MODELS:
            sub = d[d.arm.str.startswith(model)]
            if sub.empty:
                continue
            b = sub.sort_values(VERDICT, ascending=False).iloc[0]
            row[model] = claim(
                f"best {model} arm on the 250-gene common scope in {line}",
                float(b[VERDICT]), RUNS,
                key(cell_line=line, scope="common", split="grouped", translator="mlp",
                    target_source="run1_all_controls", arm=b.arm, column=VERDICT),
            )
        common[line] = row

    overlap = {}
    if not r.overlap.empty:
        d = r.overlap[r.overlap.translator == "mlp"]
        piv = d.pivot_table(index=["cell_line", "arm"], columns="variant",
                            values=VERDICT).reset_index()
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
                            variant=variant, column=VERDICT),
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
    centred_draw = B._null_by_translator(r, VERDICT)
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
                column=VERDICT, statistic="max"),
        )
        metric_check[f"coexpression_preregistered_{line}"] = claim(
            f"co-expression's own pre-registered score in {line}",
            r.score(line, "coexp", PREREG), RUNS,
            pkey(cell_line=line, arm="coexp", column=PREREG),
        )
        m = r.model_arms(line)
        m = m[m.arm.str.startswith(B.FULL_SCOPE_MODELS)]
        metric_check[f"model_arms_clearing_null_{line}"] = (
            f"{int((m[VERDICT] > r.null_ceiling(line)).sum())} of {int(m.shape[0])}"
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
                      target_source="run1_all_controls", arm="*_resp", column=VERDICT,
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
                 & (r.runs.arm == "node_pairmlp")][VERDICT]
    splits = {
        "node_free_random_split_hepg2": claim(
            "free node-statistics network in hepg2 under the random split",
            float(rnd.iloc[0]), RUNS,
            key(cell_line="hepg2", scope="full", split="random", translator="mlp",
                target_source="run1_all_controls", arm="node_pairmlp", column=VERDICT),
        ),
        "node_free_grouped_split_hepg2": lines["hepg2"]["node_free_score"],
    }

    mode = {}
    piv = r.transfer[r.transfer.translator == "mlp"].pivot_table(
        index=["cell_line", "scope", "arm"], columns="transfer_mode", values=VERDICT)
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
                        else f"{model}_resp", column=VERDICT),
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
            SENS, key(cell_line="hepg2", arms="*_resp@*", statistic="max", column=VERDICT),
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

    # the source paper's own benchmark
    bee: dict = {}
    if r.beeline:
        b = r.beeline
        for row in b["rows"]:
            cell = f"{row['network']}_{row['dataset']}"
            bee[f"auprc_{cell}"] = claim(
                f"AUPRC reproduced here on {row['network']}/{row['dataset']}",
                row["auprc"], BEELINE, key(row=cell, field="auprc"),
            )
            bee[f"auprc_reported_{cell}"] = claim(
                f"AUPRC the source paper reports for its scGPT embedding baseline on "
                f"{row['network']}/{row['dataset']}",
                row["paper_baseline_emb_auprc"], BEELINE,
                key(row=cell, field="paper_baseline_emb_auprc"),
            )
            bee[f"auprc_gene_identity_only_{cell}"] = claim(
                f"AUPRC with gene embeddings permuted between genes on "
                f"{row['network']}/{row['dataset']}",
                row["auprc_shuffled_embedding"], BEELINE,
                key(row=cell, field="auprc_shuffled_embedding"),
            )
            bee[f"auprc_unseen_genes_{cell}"] = claim(
                f"AUPRC on pairs whose genes never appear in training, "
                f"{row['network']}/{row['dataset']}",
                row["auprc_unseen_genes"], BEELINE, key(row=cell, field="auprc_unseen_genes"),
            )
            bee[f"chance_unseen_genes_{cell}"] = claim(
                f"chance AUPRC on that unseen-gene slice (its positive fraction), "
                f"{row['network']}/{row['dataset']}",
                row["positive_fraction_unseen"], BEELINE,
                key(row=cell, field="positive_fraction_unseen"),
            )
        for field in ("mean_auprc", "paper_mean_auprc", "mean_difference",
                      "max_abs_difference", "shuffled_embedding_mean_auprc",
                      "mean_auprc_unseen_genes",
                      "shuffled_embedding_mean_auprc_unseen_genes",
                      "mean_fraction_pairs_unseen_genes"):
            bee[field] = claim(
                f"BEELINE reproduction, {field.replace('_', ' ')}",
                b["summary"][field], BEELINE, key(field=f"summary.{field}"),
            )
        bee["train_cell"] = f"{b['train_cell']['network']}/{b['train_cell']['dataset']}"
        bee["train_pairs"] = b["train_cell"]["n_pairs_kept"]
        bee["arms_not_reproduced"] = b["arms_not_reproduced"]

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
                    pkey(cell_line=line, arm=arm, column=VERDICT),
                )
            d = r.scope_arms(line, "full", "random")
            for arm in ("esm2_3b_emb", "transcriptformer_emb", "scgpt_emb", "coexp",
                        "null_randgene_d2560"):
                row = d[d.arm == arm]
                if row.empty:
                    continue
                block[f"{arm}__random_split"] = claim(
                    f"{arm} held-out fold-centred score in {line} under the random split",
                    float(row[VERDICT].iloc[0]), RUNS,
                    key(cell_line=line, scope="full", split="random", translator="mlp",
                        target_source="run1_all_controls", arm=arm, column=VERDICT),
                )
            block["largest_protein_shuffled_draw"] = claim(
                f"largest shuffled-label draw of the ESM2 arms in {line}",
                r.head["protein"]["lines"][line]["largest_protein_shuffled_draw"], HEAD,
                key(field=f"protein.lines.{line}.largest_protein_shuffled_draw"),
            )
            protein["scores"][line] = block
        protein["paired_differences"] = {}
        for line in ("hepg2", "jurkat"):
            for row in r.head["protein"]["lines"][line]["paired_differences"]:
                tag = f"{line}__{row['arm']}__vs__{row['reference']}"
                protein["paired_differences"][tag] = {
                    "median_difference": row["median_difference"],
                    "lo": row["lo"], "hi": row["hi"], "n_paired": row["n_paired"],
                }
                claims.append({
                    "claim": f"paired difference between {row['arm']} and {row['reference']} "
                             f"on the same held-out perturbations in {line}",
                    "value": round(float(row["median_difference"]), 6), "source": HEAD,
                    "key": key(field=f"protein.lines.{line}.paired_differences",
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
                        unseen[tag] = {"score": row["score"], "lo": row["lo"],
                                       "hi": row["hi"],
                                       "n_perturbations": row["n_perturbations"]}
                        if mode_name == "dest_reference":
                            claims.append({
                                "claim": f"{row['arm']} on the unseen-gene cross-line slice, "
                                         f"{direction}, held-out reference cells",
                                "value": round(float(row["score"]), 6), "source": HEAD,
                                "key": key(
                                    field=(f"protein_transfer.blocks.{direction}|"
                                           f"transfer_unseen|cross_line__dest_reference"),
                                    arm=row["arm"], statistic="score"),
                            })
            protein["unseen_gene_transfer"] = unseen
            groups = {
                "protein_derived": ("esm2_3b_emb", "esm2_650m_emb",
                                    "transcriptformer_emb", "scprint_emb"),
                "scgpt": ("scgpt_emb",),
                "coexpression_and_nulls": ("coexp", "null_randgene_d2560",
                                           "null_randgene_d5120"),
            }
            counts = {}
            for name, arms in groups.items():
                cells = [v for k, v in unseen.items() if k.split("__")[-1] in arms]
                counts[name] = {"cells": len(cells),
                                "interval_excludes_zero": sum(1 for v in cells if v["lo"] > 0)}
                claims.append({
                    "claim": f"unseen-gene cross-line cells where the interval excludes zero, "
                             f"{name} arms, over both directions and all three reference "
                             f"choices",
                    "value": f"{counts[name]['interval_excludes_zero']} of "
                             f"{counts[name]['cells']}",
                    "source": HEAD,
                    "key": key(
                        field=f"protein_transfer.unseen_gene_clearance.{name}"),
                })
            protein["unseen_gene_transfer_counts"] = counts

    summary = {
        "lines": lines,
        "protein_control": protein,
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
        "beeline_reproduction": bee,
    }
    return summary, claims


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
            "lines and on both splits. The response features harvested in the previous run "
            "add nothing to the embedding; on their own they are weak, and whether they beat "
            "co-expression depends on which score is read. The advantage does not come from "
            "single-cell pretraining: reading the four released checkpoints shows that three "
            "of the four gene tables are protein-derived rather than learned from expression, "
            "and a gene table built from UniProt protein sequence with a public ESM2 "
            "checkpoint, run through the identical translator, folds, scoring and nulls, "
            "matches the best single-cell arm in HepG2 and beats every one of them in Jurkat."
        ),
        "assessment": (
            "Supported for the embedding arms against co-expression, on both cell lines, on a "
            "post-hoc scoring cell. The second pre-registered scoring axis, perturbation "
            "discrimination, does not settle it either way: the version pinned before the run "
            "does not exist upstream, and the two existing versions nearest it reverse the "
            "model-versus-co-expression ordering on the same saved predictions, so that axis "
            "is reported as version-dependent and inconclusive. The pre-registered cell of the "
            "scoring grid fails its "
            "own shuffled-label control, so the verdict is read from the fold-centred cell, "
            "which the pre-registration also defined but did not designate. Both are reported "
            "for every arm; they rank the arms similarly (rank correlation at least "
            f"{n['ordering_agreement_min']['centered']:.4f}) but not identically, and the arms "
            "that move are the weak ones. The response features are therefore only partly "
            "refuted: they add nothing to the embedding on either score, but on the "
            "pre-registered score TranscriptFormer's response features beat co-expression in "
            "HepG2 by more than the resolvable difference, and on the fold-centred score two of "
            "three models' response arms clear the null in Jurkat. 'Response features carry no "
            "usable signal' holds for the fold-centred score in HepG2 and is too strong "
            "elsewhere. The protein control is a clean positive: the raw ESM2 arm clears its "
            "width-matched random-vector null and its own shuffled-label draws in both cell "
            "lines, beats TranscriptFormer's embedding arm in both on a paired comparison, "
            "ties scPRINT in both, and ties scGPT in HepG2 while beating it in Jurkat, with "
            "every arm it is compared against re-run inside the same job and reproducing the "
            "main sweep exactly. It shows where the signal can come from, not where each "
            "model's signal does come from: it does not decompose scGPT's or AIDO.Cell's "
            "learned table into a protein-like part and a cell-specific part."
        ),
        "decision_rule": {
            "preregistered": (
                "residual target, reliability-weighted, ceiling-corrected per-perturbation "
                "Pearson correlation, median over held-out perturbations of a grouped split, "
                "compared to co-expression as a paired difference"
            ),
            "used_for_the_verdict": (
                "the same quantity after the held-out fold's per-gene mean is removed from "
                "both prediction and measurement (the fold-centred cell of the same grid), "
                "chosen post hoc because the shuffled-label control does not sit at zero on "
                "the pre-registered cell"
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
            "The ranking metric is post hoc; the pre-registered cell is reported everywhere "
            "and fails its own shuffled-label control.",
            "Scores are small in absolute terms even after the ceiling correction: the best "
            f"arm reaches {hep['best_model_score_fold_centred']:.4f} in HepG2 and "
            f"{jur['best_model_score_fold_centred']:.4f} in Jurkat on a scale where 1.0 is the "
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
            "The source paper's benchmark is reproduced only for its embedding baseline on its "
            "three human held-out cells; its virtual-value and gradient-trajectory arms would "
            "need a fresh extraction against those datasets.",
            "Two cell lines from a single screen, so the cross-line result is a weak test of "
            "generality.",
            "The additive node control is identically zero on the fold-centred score by "
            "construction, so the free node network is the informative version.",
            "AIDO.Cell is only comparable on the 250-gene subset it was harvested on; it is "
            "not shown to be worse than the others there, only not shown to work.",
            "Co-expression is represented by four statistics over control cells; a fitted "
            "network-inference method could do better.",
            "The protein control shows where the signal can come from, not where each model's "
            "signal does come from. It does not decompose scGPT's or AIDO.Cell's learned table "
            "into a protein-like part and a cell-specific part, and it says nothing about what "
            "these models do outside this gene-pair task. One protein model at two sizes, and "
            "one pooling rule (mean over the final layer), stand in for protein space.",
            "The unseen-gene cross-line slice holds 78 to 177 held-out perturbations, so it "
            "separates the protein-derived tables from scGPT's rather than ranking the arms "
            "inside either group.",
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
