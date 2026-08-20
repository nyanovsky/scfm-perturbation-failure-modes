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


def key(**kw) -> str:
    return ", ".join(f"{k}={v}" for k, v in kw.items())


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
                best[VERDICT], RUNS, key(cell_line=line, scope="full", split="grouped",
                                         translator="mlp", arm=best.arm, column=VERDICT),
            ),
            "best_model_ci": [float(best[VERDICT + "_lo"]), float(best[VERDICT + "_hi"])],
            "best_model_score_preregistered": claim(
                f"the same arm on the pre-registered cell in {line}",
                best[PREREG], RUNS, key(cell_line=line, arm=best.arm, column=PREREG),
            ),
            "coexpression_score_fold_centred": claim(
                f"co-expression in {line} on the fold-centred cell",
                co[0], RUNS, key(cell_line=line, arm="coexp", column=VERDICT),
            ),
            "coexpression_ci": [co[1], co[2]],
            "shuffled_label_ceiling_fold_centred": claim(
                f"largest shuffled-label draw in {line}, fold-centred, MLP",
                nullv, RUNS, key(cell_line=line, arm="*+shuffled_labels_d*", column=VERDICT),
            ),
            "shuffled_label_ceiling_preregistered": claim(
                f"largest shuffled-label draw in {line}, pre-registered, MLP",
                nullp, RUNS, key(cell_line=line, arm="*+shuffled_labels_d*", column=PREREG),
            ),
            "best_response_only_score": claim(
                f"largest response-only arm in {line}",
                resp, RUNS, key(cell_line=line, arm="*_resp", column=VERDICT),
            ),
            "node_free_score": claim(
                f"free network on node statistics in {line}",
                npm, RUNS, key(cell_line=line, arm="node_pairmlp", column=VERDICT),
            ),
            "residual_ceiling": float(
                r.manifest[line]["reference"]["residual_ceiling"]["median_spearman_brown"]
            ),
            "residual_ceiling_weighted": claim(
                f"reliability-weighted split-half ceiling the pre-registered score in {line} "
                "divides by",
                r.ceiling(line, PREREG), RUNS,
                key(cell_line=line, column="residual__weighted__ceiling"),
            ),
            "fold_centred_ceiling_weighted": claim(
                f"reliability-weighted split-half ceiling the fold-centred score in {line} "
                "divides by",
                r.ceiling(line), RUNS,
                key(cell_line=line, column="centered__weighted__ceiling"),
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
                key(cell_line=line, arm=arm, metric="centered_weighted"),
            ),
            "ci": [float(c.lo.iloc[0]), float(c.hi.iloc[0])],
            "min_resolvable_difference": claim(
                f"smallest paired difference this design can resolve for {arm} in {line}",
                float(c.min_resolvable_difference_achieved.iloc[0]), CON,
                key(cell_line=line, arm=arm, metric="centered_weighted",
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
                    key(scope=scope, cell_line=direction, arm=b.arm,
                        split="cross_line__dest_reference", column=VERDICT),
                ),
                "best_model_ci": [float(b[VERDICT + "_lo"]), float(b[VERDICT + "_hi"])],
                "gene_identity_null": claim(
                    f"fixed random vector per gene on {k}",
                    float(gi[VERDICT].iloc[0]), TRANS,
                    key(scope=scope, cell_line=direction, arm="null_randgene_d5120",
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
                 float(g.spearman_vs_primary.min()), ORDER, key(target=t, statistic="min"))
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
                key(cell_line=line, scope="common", split="grouped", arm=b.arm, column=VERDICT),
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
            overlap = {
                "largest_shift_coexpression_arms": claim(
                    "largest change in a plain co-expression arm's score between a target built "
                    "from the control half it uses and one built from the other half",
                    float(co.delta.abs().max()), OVERLAP,
                    key(arms="coexp, coexp_check", statistic="max|ctrl1-ctrl0|"),
                ),
                "largest_shift_embedding_arms": claim(
                    "the same change for the embedding arms, which touch no control cell",
                    float(mo.delta.abs().max()), OVERLAP,
                    key(arms="scgpt_emb, transcriptformer_emb", statistic="max|ctrl1-ctrl0|"),
                ),
                "per_arm": piv.to_dict("records"),
            }
    for line in ("hepg2", "jurkat"):
        if line in r.recompute:
            t = r.recompute[line]["targets"]["ctrlall"]
            overlap[f"recomputed_target_agreement_{line}"] = claim(
                f"median per-perturbation correlation between the recomputed effect for {line} "
                "and the effect matrix run 1 shipped",
                t["median_agreement_with_run1_effect"], RECOMPUTE,
                key(line=line, target="ctrlall", field="median_agreement_with_run1_effect"),
            )
            overlap[f"recomputed_reliability_weight_{line}"] = claim(
                f"median reliability weight of the recomputed {line} target",
                t["median_reliability_weight"], RECOMPUTE,
                key(line=line, target="ctrlall", field="median_reliability_weight"),
            )

    prereg_draw = B._null_by_translator(r, PREREG)
    centred_draw = B._null_by_translator(r, VERDICT)
    metric_check = {}
    for line in ("hepg2", "jurkat"):
        metric_check[f"shuffled_draw_ridge_preregistered_{line}"] = claim(
            f"largest shuffled-label draw in {line} on the pre-registered cell, ridge",
            prereg_draw[(line, "ridge")], RUNS,
            key(cell_line=line, translator="ridge", arm="*+shuffled_labels_d*", column=PREREG),
        )
        metric_check[f"shuffled_draw_ridge_fold_centred_{line}"] = claim(
            f"the same draws in {line} on the fold-centred cell, ridge",
            centred_draw[(line, "ridge")], RUNS,
            key(cell_line=line, translator="ridge", arm="*+shuffled_labels_d*", column=VERDICT),
        )
        metric_check[f"coexpression_preregistered_{line}"] = claim(
            f"co-expression's own pre-registered score in {line}",
            r.score(line, "coexp", PREREG), RUNS,
            key(cell_line=line, arm="coexp", column=PREREG),
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
            "key": key(cell_line=line, scope="full", split="grouped", translator="mlp",
                       column=VERDICT, rule="score > max shuffled-label draw"),
        })

    feature_deltas = {
        "largest_response_only": claim(
            "largest response-only score in either cell line",
            max(abs(r.score(line, f"{m}_resp"))
                for line in ("hepg2", "jurkat") for m in B.FULL_SCOPE_MODELS),
            RUNS, key(arm="*_resp", statistic="max|score|", column=VERDICT),
        ),
        "largest_change_from_adding_response": claim(
            "largest change from adding the response features to the embedding",
            max(abs(r.score(line, f"{m}_full") - r.score(line, f"{m}_emb"))
                for line in ("hepg2", "jurkat") for m in B.FULL_SCOPE_MODELS),
            RUNS, key(arms="*_full minus *_emb", statistic="max|d|", column=VERDICT),
        ),
        "largest_change_from_adding_coexpression": claim(
            "largest change from adding co-expression to the embedding",
            max(abs(r.score(line, f"{m}_hybrid") - r.score(line, f"{m}_emb"))
                for line in ("hepg2", "jurkat") for m in B.FULL_SCOPE_MODELS),
            RUNS, key(arms="*_hybrid minus *_emb", statistic="max|d|", column=VERDICT),
        ),
    }

    rnd = r.runs[(r.runs.cell_line == "hepg2") & (r.runs.scope == "full")
                 & (r.runs.split == "random") & (r.runs.translator == "mlp")
                 & (r.runs.arm == "node_pairmlp")][VERDICT]
    splits = {
        "node_free_random_split_hepg2": claim(
            "free node-statistics network in hepg2 under the random split",
            float(rnd.iloc[0]), RUNS,
            key(cell_line="hepg2", split="random", arm="node_pairmlp", column=VERDICT),
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
                max(shifts), TRANS, key(statistic="max|other - dest_reference|"),
            ),
            "lowest_rank_correlation_across_modes": claim(
                "lowest rank correlation between the reported mode and either other mode, "
                "within one direction and scope",
                min(ranks), TRANS, key(statistic="min rank correlation"),
            ),
        }

    summary = {
        "lines": lines,
        "paired_vs_coexpression": paired,
        "metric_check": metric_check,
        "feature_deltas": feature_deltas,
        "split_design": splits,
        "transfer": transfer,
        "transfer_reference_mode": mode,
        "ordering_agreement_min": ordering,
        "common_scope": common,
        "control_cell_overlap": overlap,
    }
    return summary, claims


def build(summary_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    manifest = json.loads((paths.repo_dir() / "results" / "bundle_inventory.json").read_text())
    r = B.Results(summary_dir, manifest)
    n, claims = collect(r)
    hep, jur = n["lines"]["hepg2"], n["lines"]["jurkat"]

    doc = {
        "headline": (
            "A gene-pair feature built from a single-cell foundation model's gene embedding "
            "predicts the perturbation-specific part of a CRISPRi knockdown response better "
            "than statistical co-expression computed from the same control cells. The response "
            "features harvested in the previous run add nothing and do not clear the null on "
            "their own."
        ),
        "assessment": (
            "Supported for the embedding arms against co-expression, on both cell lines, on a "
            "post-hoc scoring cell. The pre-registered cell of the scoring grid fails its own "
            "shuffled-label control, so the verdict is read from the fold-centred cell, which "
            "the pre-registration also defined but did not designate. Both are reported for "
            "every arm and they rank the arms the same way (rank correlation at least "
            f"{n['ordering_agreement_min']['centered']:.4f}). The refuted part of the "
            "pre-registered hypothesis is the response features: they were the object of the "
            "previous run and they carry no usable signal here."
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
        },
        "results": n,
        "limitations": [
            "The ranking metric is post hoc; the pre-registered cell is reported everywhere "
            "and fails its own shuffled-label control.",
            "Scores are small in absolute terms even after the ceiling correction: the best "
            f"arm reaches {hep['best_model_score_fold_centred']:.4f} in HepG2 and "
            f"{jur['best_model_score_fold_centred']:.4f} in Jurkat on a scale where 1.0 is the "
            "reproducibility limit.",
            "One anchor (mean control) and one translator shape for every arm.",
            "Two cell lines from a single screen, so the cross-line result is a weak test of "
            "generality.",
            "The additive node control is identically zero on the fold-centred score by "
            "construction, so the free node network is the informative version.",
            "AIDO.Cell is only comparable on the 250-gene subset it was harvested on; it is "
            "not shown to be worse than the others there, only not shown to work.",
            "Co-expression is represented by four statistics over control cells; a fitted "
            "network-inference method could do better.",
        ],
        "artifacts": [
            RUNS, CON, TRANS, ORDER, OVERLAP, RECOMPUTE, INVENTORY,
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
